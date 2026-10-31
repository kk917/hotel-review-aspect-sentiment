# Aspect-Level Sentiment Analysis of Hotel Reviews

MAAI 902 (Natural Language Processing & LLMs), Assessment Task 2: NLP Solution Design.

Report: report/report.pdf

## Problem

Hotel managers receive thousands of free-text reviews, but a single score or review-level
sentiment label hides mixed opinions ("the room was excellent, but the staff were rude").
This project designs an NLP pipeline that turns unstructured reviews into **aspect-level
sentiment** (cleanliness, staff, location, room, food, value, noise, facilities) that a
manager can act on, with particular attention to hard cases: negation, contrast and
multiple aspects in one sentence.

**Problem statement:** hotel managers cannot act on a single review score, so this project builds and evaluates an NLP pipeline that extracts aspect-level sentiment from review text, and measures how well different models handle negation, contrast and multi-aspect sentences.

## Dataset

Booking.com "515K Hotel Reviews Data in Europe" (Kaggle), file `Hotel_Reviews.csv`.
**Not committed to this repo.**

1. Download the CSV from the dataset's Kaggle page.
2. Save it as `data/raw/Hotel_Reviews.csv` (on Colab: in your Drive copy of this project).
3. The loader reads `Positive_Review`, `Negative_Review`, `Reviewer_Score` (and `Hotel_Name`),
   samples 20,000 reviews, and treats the placeholders "No Positive" / "No Negative" as empty.

Why this dataset: natural English (capitalisation and function words kept) with separate
positive and negative text boxes. The dataset's creators stripped the punctuation, so each
text box is treated as one short comment (called a "sentence" in the code), and contractions
such as "didn t" are restored to "didn't". The box gives noisy sentiment labels for free (a comment written in the
"Negative" box is usually negative), while a small hand-labelled set covers aspect-level
sentiment and the hard cases.

**Source:** "515K Hotel Reviews Data in Europe", Kaggle dataset uploaded by `jiashenliu` (about 515,000 reviews of 1,493 European hotels, scraped from Booking.com).
**Kaggle URL / licence:** https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe, published under CC0: Public Domain. The licence comes from the uploader and cannot grant rights over the Booking.com content or the reviewers' text, so the data is used for coursework only.
**Changes made:** random sample of 20,000 reviews; text cleaned (whitespace, links, contractions restored); no other modification. The dataset is not redistributed in this repo.

## Repository layout

```
data/raw/          raw CSV (git-ignored)
data/labelled/     labelled_pairs.csv: 201 hand-checked evaluation rows, with the AI draft in frozen ai_* columns
src/pipeline.py    loading, cleaning, sentence splitting, aspect tagging, labelling sampler
src/models.py      majority, TF-IDF+LR, RoBERTa, DeBERTa-ABSA, LLM prompt hook (not evaluated)
src/evaluate.py    label loading, annotation agreement (kappa), scoring by difficulty, plot
src/demo.py        PII masking, review analysis, per-aspect summary, Gradio app
notebooks/         01_baseline.ipynb (EDA, baselines, labelling sheet)
                   02_model_comparison.ipynb (agreement with AI pre-labels, model comparison)
                   03_demo.ipynb (end-to-end demo, Gradio app)
report/            report.pdf, plus main.tex and the demo figures it needs to compile
outputs/           saved results behind the report's tables and figures
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_baseline.ipynb
```

On Colab: put the project folder in Drive, mount Drive, and `%cd` into `notebooks/`.

## Run order

1. `notebooks/01_baseline.ipynb`: EDA, baselines, blank labelling sheet (needs the raw CSV).
2. `data/labelled/labelled_pairs.csv` is the finished labelled set and is already in this repo.
   The labelling sheet from step 1 is only needed to label from scratch.
3. `notebooks/02_model_comparison.ipynb`: agreement with the AI pre-labels, five-model comparison.
4. `notebooks/03_demo.ipynb`: end-to-end demo and Gradio web app.

Seeds are fixed (`SEED=42`, `N_REVIEWS=20000`), so the sample, the labelling sheet and the results are reproducible.

## Pipeline

1. Load a 20,000-review sample; clean text (keep case and negations).
2. Treat each text box as one comment (there is no punctuation to split on), keeping the box as a weak label.
3. Tag aspects with a keyword lexicon.
4. Compare five predictors: majority class, TF-IDF + logistic regression, RoBERTa (aspect-blind),
   DeBERTa ABSA (aspect-aware), and the box label as a reference. A prompted LLM was considered
   but not evaluated, because a language model drafted the labels.
5. Evaluate on 201 hand-checked comment-aspect rows (193 scored), overall and per
   difficulty category (negation, contrast, multi-aspect, plain).
6. Demo: Gradio app where pasted reviews return a per-aspect summary with evidence.

## Ethics notes

Anonymise personal details in review text, treat fake or incentivised reviews as a risk,
note the data's provenance (scraped from a booking site, European hotels only), report
performance by review length or sub-group, ground any LLM summary in source sentences, and
aggregate at aspect level instead of naming individual staff. See the report for details.

## Status

- [x] Dataset downloaded and licence checked
- [x] Baseline notebook runs
- [x] 201 comment-aspect rows labelled (AI pre-labelled, human reviewed)
- [x] Transformer comparison (notebook 02)
- [x] Demo (notebook 03)
- [x] Report (3,000-4,000 words)
