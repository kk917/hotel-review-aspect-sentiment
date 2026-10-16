# [Title: e.g. Aspect-Level Sentiment Analysis of Hotel Reviews for Hotel Managers]

Student name / ID: 
Course: MAAI 902, Natural Language Processing & LLMs, Assessment Task 2 (NLP Solution Design)
Repository link: 
Word count target: 3,000-4,000 (plan: about 3,500). Budgets below match `hotel_review_report_plan.xlsx`.

---

## 1. Introduction and problem definition and justification (about 700 words) [rubric: 20%]

- The real-world problem: who has it (hotel managers), what they have today (star ratings, manual reading), why that fails (volume; review-level scores hide mixed opinions).
- Concrete examples of mixed or tricky reviews ("The room was excellent, but the staff were rude.").
- Why NLP is the right tool, and why rules or ratings alone are not enough.
- Scope: what the solution does and does not do.
- Short related-work paragraph (only sources you have opened and read).

## 2. Data considerations and pre-processing (about 550 words) [rubric: 15%]

- Dataset: Booking.com 515K Hotel Reviews Data in Europe (Kaggle): source, size, columns used, licence (from its Kaggle page), date range (Review_Date), geography, and how you sampled 20,000 reviews.
- Data quality: duplicates, length distribution, rating imbalance, noise (typos, slang, mixed languages).
- Limitations: European hotels only (not Gulf-focused), self-selected reviewers, scraped data, mostly English, scores skew high; weak labels from the positive/negative boxes are noisy; reviewers pre-sorted their remarks into positive and negative boxes, so a model trained here may do worse on free-form reviews.
- Pre-processing: cleaning (what you keep and why: case, negations; restoring contractions such as "didn t" to "didn't"), the fact that the dataset has no punctuation so each text box is one comment rather than a true sentence, aspect tagging, tokenisation.
- Embeddings / representations: TF-IDF for the baseline, transformer embeddings for the main model.
- Hand-labelled evaluation set: one row per sentence-aspect pair; how sentences were sampled (by difficulty group), labelled, and by whom; its size and limits.
- Weak labels: sentences inherit the positive/negative box they were written in; explain why this is noisy and how you used it.

## 3. Pipeline and architecture design (about 900 words) [rubric: 25%]

- Diagram of the pipeline: reviews > cleaning > sentence splitting > aspect tagging > aspect-level sentiment > manager output.
- Model choices and why: TF-IDF + logistic regression (baseline), pretrained transformer sentiment model, optional LLM prompt.
- How each choice links to the problem (negation, contrast, multi-aspect).
- Tuning decisions.
- Evaluation design: metrics (macro F1, per-category F1), the hard-case categories, error analysis.
- Results and discussion (once you have them): where each model fails and why.

## 4. Tools, frameworks and platforms (about 350 words) [rubric: 10%]

One short paragraph or table row per tool and its role: Hugging Face Transformers, spaCy, scikit-learn, pandas, Gradio (demo), GitHub (repo). Say why each was chosen over alternatives (e.g. LangChain only if you actually use it).

## 5. Ethical and responsible AI considerations (about 550 words) [rubric: 15%]

For each concern: what the risk is, then your mitigation.
1. Reviewer privacy (names, personal details): anonymise with NER before storage or display.
2. Fake or incentivised reviews: duplicate and burst detection, caveat in outputs.
3. Language and dialect bias: report performance by review length or sub-group.
4. LLM hallucination in summaries: link every claim to source sentences.
5. Unfair consequences for named staff: aggregate at aspect level, do not name individuals.

## 6. Expected output, value proposition and conclusion (about 450 words) [feeds clarity and presentation, 15%]

- What a manager sees (example output table or screenshot from the demo).
- Value: time saved, issues found earlier, prioritisation of fixes.
- Limitations and future work.
- Conclusion.

## References

(Only list sources you have actually read. Use one consistent style, as required by your course.)

## Appendix (does not count toward the word limit if your course allows; check the brief)

- Repo structure, hand-labelling guidelines, extra tables.
