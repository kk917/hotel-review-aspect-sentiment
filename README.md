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

1. Download the CSV from the dataset's Kaggle page and **check its licence there**. Licences are
   set per dataset, so do not reuse licence text from another dataset.
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
**Kaggle URL / licence:** TODO (copy the exact URL and the licence shown on the dataset's Kaggle page; the licence could not be confirmed from secondary sources).
**Changes made:** random sample of 20,000 reviews; text cleaned (whitespace, links, contractions restored); no other modification. The dataset is not redistributed in this repo.

## Repository layout

```
data/raw/          raw CSV (git-ignored)
data/labelled/     hand-labelled evaluation rows
src/pipeline.py    loading, cleaning, sentence splitting, aspect tagging, labelling sampler
src/models.py      majority, TF-IDF+LR, RoBERTa, DeBERTa-ABSA, optional LLM prompt
src/evaluate.py    label loading, annotation agreement (kappa), scoring by difficulty, plot
src/demo.py        PII masking, review analysis, per-aspect summary, Gradio app
notebooks/         01_baseline.ipynb (EDA, baselines, labelling sheet)
                   02_model_comparison.ipynb (agreement with AI pre-labels, model comparison)
                   03_demo.ipynb (end-to-end demo, Gradio app)
report/            report_template.md (3,000-4,000 words, rubric-aligned)
outputs/           figures, metrics, saved results
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_baseline.ipynb
```

On Colab: put the project folder in Drive, mount Drive, and `%cd` into `notebooks/`.

## Run order

1. `notebooks/01_baseline.ipynb`: EDA, baselines, labelling sheet.
2. Label `data/labelled/labelling_sheet.csv`, save as `data/labelled/labelled_pairs.csv`.
3. `notebooks/02_model_comparison.ipynb`: agreement with the AI pre-labels, five-model comparison.
4. `notebooks/03_demo.ipynb`: end-to-end demo and Gradio web app.

Seeds are fixed (`SEED=42`, `N_REVIEWS=20000`), so the sample, the labelling sheet and the results are reproducible.

## Pipeline

1. Load a 20,000-review sample; clean text (keep case and negations).
2. Treat each text box as one comment (there is no punctuation to split on), keeping the box as a weak label.
3. Tag aspects (keyword lexicon first; improve later if time allows).
4. Predict sentiment with three models: TF-IDF + logistic regression (baseline), a pretrained
   transformer sentiment model, and optionally an LLM prompt.
5. Evaluate on a hand-labelled set (about 200 sentence-aspect rows), overall and per
   difficulty category (negation, contrast, multi-aspect).
6. Demo: notebook or small Gradio app where a pasted review returns aspect-level sentiment.

## Ethics notes

Anonymise personal details in review text, treat fake or incentivised reviews as a risk,
note the data's provenance (scraped from a booking site, European hotels only), report
performance by review length or sub-group, ground any LLM summary in source sentences, and
aggregate at aspect level instead of naming individual staff. See the report for details.

## Status

- [ ] Dataset downloaded and licence checked
- [x] Baseline notebook runs
- [x] About 200 sentence-aspect rows labelled (AI pre-labelled, human reviewed)
- [x] Transformer comparison (notebook 02)
- [ ] Demo (notebook 03 written; run it in Colab)
- [ ] Report (3,000-4,000 words)
