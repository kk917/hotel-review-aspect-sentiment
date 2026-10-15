"""Core helpers for the hotel-review aspect-sentiment project.

Steps covered here:
    load_reviews     -> read the TripAdvisor CSV and find the text/rating columns
    clean_text       -> light cleaning (keeps case and negations on purpose)
    split_sentences  -> one row per sentence (spaCy sentencizer, regex fallback)
    tag_aspects      -> keyword-lexicon aspect tagging
    add_difficulty_flags / sample_for_labelling -> build the hand-labelling sheet
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Aspect lexicon (first version: edit and extend after looking at your data)
# ---------------------------------------------------------------------------
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "cleanliness": ["clean", "dirty", "spotless", "dust", "dusty", "smell", "smelly", "stain", "stained", "hygiene", "mold", "mould"],
    "staff": ["staff", "service", "reception", "receptionist", "manager", "concierge", "waiter", "waitress", "housekeeping", "front desk", "employee", "rude", "friendly", "helpful"],
    "location": ["location", "located", "walk", "walking distance", "close to", "near", "metro", "station", "beach", "centrally", "central"],
    "room": ["room", "bed", "beds", "bathroom", "shower", "pillow", "mattress", "view", "balcony", "suite", "bedroom", "air conditioning", "ac"],
    "food": ["food", "breakfast", "dinner", "lunch", "restaurant", "buffet", "meal", "coffee", "bar", "drinks"],
    "value": ["price", "priced", "value", "expensive", "cheap", "overpriced", "worth", "cost", "money", "rate"],
    "noise": ["noise", "noisy", "loud", "quiet", "peaceful", "thin walls", "construction"],
    "facilities": ["pool", "gym", "spa", "wifi", "wi-fi", "internet", "parking", "elevator", "lift", "facilities", "lobby"],
}

CONTRAST_WORDS = {"but", "although", "though", "however", "yet", "while", "whereas", "despite", "except"}
NEGATION_WORDS = {"not", "no", "never", "nothing", "neither", "nor", "hardly", "without", "wasn't", "weren't", "isn't", "aren't", "didn't", "don't", "doesn't", "couldn't", "wouldn't", "won't", "can't", "cannot", "n't"}

_ASPECT_PATTERNS = {
    aspect: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b", re.IGNORECASE)
    for aspect, kws in ASPECT_KEYWORDS.items()
}

_TEXT_CANDIDATES = ["review", "text", "review_text", "reviewtext", "content", "comment"]
_RATING_CANDIDATES = ["rating", "stars", "score", "overall"]


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def _find_column(columns, candidates, what):
    lowered = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    raise ValueError(
        f"Could not find the {what} column. Columns in the file: {list(columns)}. "
        f"Rename one of them to one of {candidates}, or edit src/pipeline.py."
    )


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Read the CSV and return a DataFrame with columns: review_id, review, rating."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the TripAdvisor hotel reviews CSV from Kaggle "
            f"(check its licence) and save it there."
        )
    raw = pd.read_csv(path)
    text_col = _find_column(raw.columns, _TEXT_CANDIDATES, "review text")
    rating_col = _find_column(raw.columns, _RATING_CANDIDATES, "rating")
    df = pd.DataFrame({"review": raw[text_col].astype(str), "rating": pd.to_numeric(raw[rating_col], errors="coerce")})
    df = df.dropna(subset=["rating"]).drop_duplicates(subset=["review"]).reset_index(drop=True)
    df["rating"] = df["rating"].astype(int)
    df.insert(0, "review_id", range(len(df)))
    return df


def clean_text(text: str) -> str:
    """Light cleaning. Case, punctuation and negations are kept on purpose:
    they matter for transformers and for the negation/contrast hard cases."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rating_to_sentiment(rating: int) -> str:
    """Assumption: 4-5 stars = positive, 3 = neutral, 1-2 = negative."""
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


# ---------------------------------------------------------------------------
# Sentence splitting and aspect tagging
# ---------------------------------------------------------------------------
def _regex_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def split_sentences(df: pd.DataFrame, text_col: str = "review") -> pd.DataFrame:
    """Return one row per sentence: review_id, sent_id, sentence, rating."""
    try:
        import spacy

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        docs = nlp.pipe(df[text_col].tolist(), batch_size=256)
        split = [[s.text.strip() for s in d.sents if s.text.strip()] for d in docs]
    except ImportError:
        split = [_regex_split(t) for t in df[text_col]]

    rows = []
    for rid, rating, sents in zip(df["review_id"], df["rating"], split):
        for i, s in enumerate(sents):
            rows.append((rid, i, s, rating))
    return pd.DataFrame(rows, columns=["review_id", "sent_id", "sentence", "rating"])


def tag_aspects(sentence: str) -> list[str]:
    """Return the aspects whose keywords appear in the sentence."""
    return [a for a, pat in _ASPECT_PATTERNS.items() if pat.search(sentence)]


def _tokens(sentence: str) -> set[str]:
    return set(re.findall(r"[a-z']+", sentence.lower()))


def add_difficulty_flags(sent_df: pd.DataFrame) -> pd.DataFrame:
    """Add aspects, n_aspects and flags for negation, contrast and multi-aspect."""
    out = sent_df.copy()
    out["aspects"] = out["sentence"].map(tag_aspects)
    out["n_aspects"] = out["aspects"].map(len)
    toks = out["sentence"].map(_tokens)
    out["has_negation"] = toks.map(lambda t: bool(t & NEGATION_WORDS) or any(w.endswith("n't") for w in t))
    out["has_contrast"] = toks.map(lambda t: bool(t & CONTRAST_WORDS))
    out["is_multi_aspect"] = out["n_aspects"] >= 2
    return out


# ---------------------------------------------------------------------------
# Sampling for hand-labelling
# ---------------------------------------------------------------------------
def sample_for_labelling(flagged: pd.DataFrame, n: int = 200, seed: int = 42, min_words: int = 5, max_words: int = 40) -> pd.DataFrame:
    """Draw about n sentences, roughly balanced across four groups:
    negation, contrast, multi-aspect and plain (one aspect, none of the above).
    Only sentences with at least one aspect keyword and a sensible length are used.
    Returns a sheet with empty columns for you to fill in by hand."""
    pool = flagged[flagged["n_aspects"] >= 1].copy()
    n_words = pool["sentence"].str.split().map(len)
    pool = pool[(n_words >= min_words) & (n_words <= max_words)]

    groups = {
        "negation": pool[pool["has_negation"]],
        "contrast": pool[pool["has_contrast"]],
        "multi_aspect": pool[pool["is_multi_aspect"]],
        "plain": pool[~pool["has_negation"] & ~pool["has_contrast"] & ~pool["is_multi_aspect"]],
    }
    per_group = n // len(groups)
    picked, used = [], set()
    for name, g in groups.items():
        g = g[~g.index.isin(used)]
        take = g.sample(n=min(per_group, len(g)), random_state=seed)
        used.update(take.index)
        picked.append(take.assign(sample_group=name))
    sheet = pd.concat(picked).sample(frac=1, random_state=seed).reset_index(drop=True)
    sheet["aspects"] = sheet["aspects"].map(lambda a: ", ".join(a))
    sheet = sheet[["review_id", "sent_id", "sentence", "aspects", "sample_group", "has_negation", "has_contrast", "is_multi_aspect"]]
    # columns to fill in by hand
    sheet["gold_aspect"] = ""
    sheet["gold_sentiment"] = ""  # positive / negative / neutral toward the aspect
    sheet["difficulty"] = ""      # negation / contrast / multi_aspect / plain
    return sheet
