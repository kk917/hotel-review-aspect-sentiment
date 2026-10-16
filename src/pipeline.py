"""Core helpers for the hotel-review aspect-sentiment project.

Dataset: Booking.com "515K Hotel Reviews Data in Europe" (Kaggle), file Hotel_Reviews.csv.
Each row has a Positive_Review box, a Negative_Review box and a Reviewer_Score (2.5-10).

Steps covered here:
    load_reviews          -> read the CSV, sample reviews, clean, build one row per review
    split_sentences       -> one row per sentence; remembers which box it was written in
                             (positive / negative) = a free, noisy "weak" sentiment label
    add_difficulty_flags  -> aspects, negation, contrast, multi-aspect flags per sentence
    sample_for_labelling  -> sheet with one row per (sentence, aspect) pair for hand-labelling
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Lexicons (first version: edit and extend after looking at your data)
# ---------------------------------------------------------------------------
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "cleanliness": ["clean", "cleaned", "cleaning", "cleanliness", "dirty", "filthy", "spotless", "dust", "dusty", "smell", "smelly", "stain", "stained", "hygiene", "mold", "mould"],
    "staff": ["staff", "service", "reception", "receptionist", "manager", "concierge", "waiter", "waitress", "housekeeping", "front desk", "employee", "rude", "friendly", "helpful", "courteous"],
    "location": ["location", "located", "walking distance", "walk", "close to", "near", "metro", "station", "beach", "central", "centrally", "centre", "center"],
    "room": ["room", "bed", "bathroom", "shower", "pillow", "mattress", "view", "balcony", "suite", "bedroom", "air conditioning", "ac", "towel"],
    "food": ["food", "breakfast", "dinner", "lunch", "restaurant", "buffet", "meal", "coffee", "bar", "drinks"],
    "value": ["price", "priced", "value", "expensive", "cheap", "overpriced", "worth", "cost", "money"],
    "noise": ["noise", "noisy", "loud", "quiet", "peaceful", "thin walls", "construction"],
    "facilities": ["pool", "gym", "spa", "wifi", "wi-fi", "internet", "parking", "elevator", "lift", "facilities", "lobby"],
}

# "while" and "yet" are left out on purpose: in reviews they are mostly not contrastive ("while we were there").
CONTRAST_WORDS = {"but", "although", "though", "however", "whereas", "despite", "except"}
NEGATION_WORDS = {"not", "no", "never", "nothing", "neither", "nor", "hardly", "without", "cannot", "cant", "dont", "doesnt", "didnt", "wasnt", "werent", "isnt", "arent", "wouldnt", "couldnt", "wont"}

# optional plural suffix so "rooms" matches "room"
_ASPECT_PATTERNS = {
    aspect: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")(?:s|es)?\b", re.IGNORECASE)
    for aspect, kws in ASPECT_KEYWORDS.items()
}

PLACEHOLDERS = {"no negative", "no positive"}
REQUIRED_COLUMNS = ["Positive_Review", "Negative_Review", "Reviewer_Score"]
OPTIONAL_COLUMNS = ["Hotel_Name"]


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def clean_text(text) -> str:
    """Light cleaning. Case, punctuation and negations are kept on purpose:
    they matter for transformers and for the negation/contrast hard cases."""
    text = str(text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_field(value) -> str:
    """Clean one text box; the dataset's placeholders become empty strings."""
    if pd.isna(value):
        return ""
    text = clean_text(value)
    return "" if text.lower() in PLACEHOLDERS else text


def _ensure_punct(text: str) -> str:
    return text if (not text or text[-1] in ".!?") else text + "."


def load_reviews(path: str | Path, n_sample: int | None = 20000, seed: int = 42) -> pd.DataFrame:
    """Read Hotel_Reviews.csv and return one row per review with columns:
    review_id, hotel, score, positive, negative, review (positive + negative joined)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download Hotel_Reviews.csv from the Kaggle page of the "
            f"515K Hotel Reviews Data in Europe dataset (check its licence) and save it there."
        )
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"Missing columns {missing}. Columns in the file: {list(header)}")
    use = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in header]
    raw = pd.read_csv(path, usecols=use)
    raw = raw.dropna(subset=["Reviewer_Score"])
    if n_sample and len(raw) > n_sample:
        raw = raw.sample(n=n_sample, random_state=seed)

    df = pd.DataFrame(
        {
            "hotel": raw["Hotel_Name"].astype(str) if "Hotel_Name" in raw else "",
            "score": pd.to_numeric(raw["Reviewer_Score"], errors="coerce"),
            "positive": raw["Positive_Review"].map(_clean_field),
            "negative": raw["Negative_Review"].map(_clean_field),
        }
    )
    df["review"] = [
        " ".join(_ensure_punct(t) for t in (p, n) if t) for p, n in zip(df["positive"], df["negative"])
    ]
    df = df[(df["review"] != "") & df["score"].notna()]
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    df.insert(0, "review_id", range(len(df)))
    return df


def score_to_sentiment(score: float) -> str:
    """Assumption: Reviewer_Score >= 8 positive, 6 to <8 neutral, <6 negative.
    (Booking scores skew high, so the cut-offs are higher than a plain 1-5 star scale.)"""
    if score >= 8:
        return "positive"
    if score >= 6:
        return "neutral"
    return "negative"


# ---------------------------------------------------------------------------
# Sentence splitting and aspect tagging
# ---------------------------------------------------------------------------
def _regex_split(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def _get_splitter():
    """spaCy sentencizer if spaCy is installed, otherwise a regex fallback."""
    try:
        import spacy

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

        def split_many(texts):
            return [[s.text.strip() for s in d.sents if s.text.strip()] for d in nlp.pipe(texts, batch_size=256)]

        return split_many
    except ImportError:
        return lambda texts: [_regex_split(t) for t in texts]


def split_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence: review_id, sent_id, sentence, field, score.
    `field` is the box the sentence was written in (positive / negative): a weak label.
    sent_id counts through the whole review (positive sentences first), so
    (review_id, sent_id) is a unique key."""
    split_many = _get_splitter()
    pos_split = split_many(df["positive"].tolist())
    neg_split = split_many(df["negative"].tolist())
    rows = []
    for rid, score, ps, ns in zip(df["review_id"], df["score"], pos_split, neg_split):
        i = 0
        for s in ps:
            rows.append((rid, i, s, "positive", score))
            i += 1
        for s in ns:
            rows.append((rid, i, s, "negative", score))
            i += 1
    return pd.DataFrame(rows, columns=["review_id", "sent_id", "sentence", "field", "score"])


def tag_aspects(sentence: str) -> list[str]:
    """Return the aspects whose keywords appear in the sentence."""
    return [a for a, pat in _ASPECT_PATTERNS.items() if pat.search(sentence)]


def _tokens(sentence: str) -> set[str]:
    # apostrophes are dropped so "wasn't" becomes "wasnt" (matches NEGATION_WORDS)
    return set(re.findall(r"[a-z]+", sentence.lower().replace("'", "")))


def add_difficulty_flags(sent_df: pd.DataFrame) -> pd.DataFrame:
    """Add n_words, aspects, n_aspects and flags for negation, contrast and multi-aspect."""
    out = sent_df.copy()
    out["n_words"] = out["sentence"].str.split().map(len)
    out["aspects"] = out["sentence"].map(tag_aspects)
    out["n_aspects"] = out["aspects"].map(len)
    toks = out["sentence"].map(_tokens)
    out["has_negation"] = toks.map(lambda t: bool(t & NEGATION_WORDS))
    out["has_contrast"] = toks.map(lambda t: bool(t & CONTRAST_WORDS))
    out["is_multi_aspect"] = out["n_aspects"] >= 2
    return out


# ---------------------------------------------------------------------------
# Sampling for hand-labelling (one row per sentence-aspect pair)
# ---------------------------------------------------------------------------
def sample_for_labelling(flagged: pd.DataFrame, n_pairs: int = 200, seed: int = 42, min_words: int = 6, max_words: int = 35, max_aspects: int = 4) -> pd.DataFrame:
    """Build a labelling sheet with about n_pairs rows, one per (sentence, aspect) pair.

    Sentences are drawn per group (contrast, negation, multi_aspect, plain), rarest first,
    so each hard case is represented. All aspects of a chosen sentence are included,
    so contrasting opinions in one sentence can be compared.

    The box a sentence was written in (the weak label) is deliberately NOT in the sheet,
    so it cannot bias your labels. Join it back later using review_id + sent_id.
    """
    n_aspects_ok = flagged["n_aspects"].between(1, max_aspects)
    n_words_ok = flagged["n_words"].between(min_words, max_words)
    pool = flagged[n_aspects_ok & n_words_ok]
    # label each distinct sentence once (common phrases repeat across reviews)
    pool = pool[~pool["sentence"].str.lower().str.strip(" .!").duplicated()]

    groups = [
        ("contrast", pool["has_contrast"]),
        ("negation", pool["has_negation"]),
        ("multi_aspect", pool["is_multi_aspect"]),
        ("plain", ~(pool["has_contrast"] | pool["has_negation"] | pool["is_multi_aspect"])),
    ]
    per_group = n_pairs // len(groups)
    used: set = set()
    rows = []
    for name, mask in groups:
        cand = pool[mask & ~pool.index.isin(used)].sample(frac=1, random_state=seed)
        total = 0
        for idx, r in cand.iterrows():
            if total >= per_group:
                break
            used.add(idx)
            for aspect in r["aspects"]:
                rows.append(
                    {
                        "review_id": r["review_id"],
                        "sent_id": r["sent_id"],
                        "sentence": r["sentence"],
                        "aspect": aspect,
                        "sample_group": name,
                        "has_negation": r["has_negation"],
                        "has_contrast": r["has_contrast"],
                        "is_multi_aspect": r["is_multi_aspect"],
                    }
                )
            total += len(r["aspects"])
    sheet = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    # columns to fill in by hand
    sheet["gold_sentiment"] = ""  # positive / negative / neutral / not_applicable (toward THIS aspect)
    sheet["difficulty"] = ""      # negation / contrast / multi_aspect / plain
    sheet["notes"] = ""
    return sheet
