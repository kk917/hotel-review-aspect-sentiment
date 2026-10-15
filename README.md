# Aspect-Level Sentiment Analysis of Hotel Reviews

MAAI 902 (Natural Language Processing & LLMs), Assessment Task 2: NLP Solution Design.

## Problem

Hotel managers receive thousands of free-text reviews, but a single star rating or
review-level sentiment score hides mixed opinions ("the room was excellent, but the
staff were rude"). This project designs an NLP pipeline that turns unstructured reviews
into **aspect-level sentiment** (cleanliness, staff, location, room, food, value, noise,
facilities) that a manager can act on, with particular attention to hard cases:
negation, contrast and multiple aspects in one review.

> TODO: replace with your own one-sentence problem statement.

## Dataset

TripAdvisor hotel reviews (Kaggle). **Not committed to this repo.**

1. Download the CSV from the dataset's Kaggle page and **check its licence** there.
2. Save it as `data/raw/tripadvisor_hotel_reviews.csv`.
3. Expected columns: a free-text review column and a 1-5 star rating column. The loader
   in `src/pipeline.py` auto-detects common column names and raises a clear error if it
   cannot find them.

## Repository layout

```
data/raw/          raw CSV (git-ignored)
data/labelled/     hand-labelled evaluation sentences
src/pipeline.py    loading, cleaning, sentence splitting, aspect tagging, sampling
notebooks/         01_baseline.ipynb (EDA + TF-IDF baseline)
report/            report_template.md (3,000-4,000 words, rubric-aligned)
outputs/           figures, metrics, saved results
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_baseline.ipynb
```

## Pipeline (planned)

1. Clean text (keep case and negations; do not strip stop words for transformers).
2. Split reviews into sentences (spaCy sentencizer).
3. Tag aspects (keyword lexicon first; improve later if time allows).
4. Predict sentiment per aspect sentence with three models:
   TF-IDF + logistic regression (baseline), a pretrained transformer sentiment model,
   and optionally an LLM prompt.
5. Evaluate on a hand-labelled set (about 150-200 sentences), overall and per difficulty
   category (negation, contrast, multi-aspect).
6. Demo: notebook or small Gradio app where a pasted review returns aspect-level sentiment.

## Ethics notes

Anonymise personal details in review text, treat fake or incentivised reviews as a risk,
report performance by review length or sub-group, ground any LLM summary in source
sentences, and aggregate at aspect level instead of naming individual staff.
See the report for details.

## Status

- [ ] Dataset downloaded and licence checked
- [ ] Baseline notebook runs
- [ ] 150-200 sentences hand-labelled
- [ ] Transformer comparison
- [ ] Demo
- [ ] Report (3,000-4,000 words)
