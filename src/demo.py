"""Manager-facing demo: paste reviews, get aspect-level sentiment with the source sentences.

    from src.demo import analyse_reviews, summarise, build_app
    model = HFAbsaModel()
    table = analyse_reviews(["The room was excellent, but the staff were rude."], model)
    summarise(table)

Design choices (see report, ethics section):
* personal details (emails, phone numbers, links) are masked BEFORE anything is shown or stored;
* results are aggregated per aspect, never per named staff member;
* every count in the summary links back to the exact source sentences (no free-text summary
  is generated, so there is nothing to hallucinate).
"""
from __future__ import annotations

import re

import pandas as pd

from src.pipeline import _regex_split, clean_text, tag_aspects, ASPECT_KEYWORDS

SAMPLE_REVIEWS = """The room was excellent, but the staff were rude at check in.

Location was perfect and the breakfast was good. The wifi didn't work and the street was noisy at night.

Not a clean hotel at all. Bathroom was dirty and we were charged for things we didn't use.

Great value for the price, very comfy beds and a lovely pool."""

_PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d"), "[phone]"),
    (re.compile(r"https?://\S+|www\.\S+"), "[link]"),
]


def mask_personal_details(text: str) -> str:
    """Regex masking of emails, phone numbers and links. It does NOT detect names: a
    named-entity model would be needed for that (listed as future work in the report)."""
    for pat, repl in _PII:
        text = pat.sub(repl, text)
    return text


def split_reviews(raw: str) -> list[str]:
    """Reviews are separated by blank lines."""
    return [r.strip() for r in re.split(r"\n\s*\n", raw or "") if r.strip()]


def analyse_reviews(reviews: list[str], model) -> pd.DataFrame:
    """One row per (sentence, aspect) with the model's sentiment for that aspect."""
    rows = []
    for rid, review in enumerate(reviews):
        for sid, sent in enumerate(_regex_split(clean_text(mask_personal_details(review)))):
            for aspect in tag_aspects(sent):
                rows.append({"review": rid + 1, "sentence": sent, "aspect": aspect})
    table = pd.DataFrame(rows, columns=["review", "sentence", "aspect"])
    if len(table):
        table["sentiment"] = model.predict(table)
    else:
        table["sentiment"] = []
    return table


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Per-aspect counts, a net score and the share of negative mentions, worst first."""
    if table.empty:
        return pd.DataFrame(columns=["aspect", "positive", "neutral", "negative", "mentions", "net_score"])
    counts = (table.pivot_table(index="aspect", columns="sentiment", values="sentence",
                                aggfunc="count", fill_value=0)
              .reindex(columns=["positive", "neutral", "negative"], fill_value=0))
    counts["mentions"] = counts.sum(axis=1)
    counts["net_score"] = ((counts["positive"] - counts["negative"]) / counts["mentions"]).round(2)
    return counts.sort_values(["net_score", "mentions"]).reset_index()


def evidence(table: pd.DataFrame, aspect: str, sentiment: str = "negative", k: int = 5) -> list[str]:
    """Source sentences behind one cell of the summary."""
    m = table[(table["aspect"] == aspect) & (table["sentiment"] == sentiment)]
    return m["sentence"].head(k).tolist()


def build_app(model):
    """Gradio app. Launch with build_app(model).launch(share=True) (Colab) or .launch() (local)."""
    import gradio as gr

    def run(raw):
        reviews = split_reviews(raw)
        if not reviews:
            return pd.DataFrame(), pd.DataFrame(), "Paste at least one review."
        table = analyse_reviews(reviews, model)
        summary = summarise(table)
        worst = summary.iloc[0]["aspect"] if len(summary) else None
        note = (f"Weakest aspect: **{worst}** (net score {summary.iloc[0]['net_score']}). "
                "Each row below links a sentiment to its source sentence."
                if worst else "No aspects were detected in these reviews.")
        return summary, table, note

    with gr.Blocks(title="Hotel review aspect sentiment") as app:
        gr.Markdown("# Hotel review aspect sentiment\nPaste one or more reviews (separate reviews with a blank line). "
                    "Emails, phone numbers and links are masked before analysis. Aspects: "
                    + ", ".join(ASPECT_KEYWORDS) + ".")
        box = gr.Textbox(lines=10, label="Reviews", value=SAMPLE_REVIEWS)
        btn = gr.Button("Analyse", variant="primary")
        note = gr.Markdown()
        summary = gr.Dataframe(label="Manager summary (per aspect, worst first)")
        detail = gr.Dataframe(label="Evidence: every sentence-aspect pair")
        btn.click(run, box, [summary, detail, note])
    return app
