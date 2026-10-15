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

- Dataset: TripAdvisor hotel reviews (Kaggle): source, size, columns, licence, date range, geography.
- Data quality: duplicates, length distribution, rating imbalance, noise (typos, slang, mixed languages).
- Limitations: older data, not Gulf-focused, self-selected reviewers.
- Pre-processing: cleaning (what you keep and why: case, negations), sentence segmentation, aspect tagging, tokenisation.
- Embeddings / representations: TF-IDF for the baseline, transformer embeddings for the main model.
- Hand-labelled evaluation set: how sentences were sampled, labelled, and by whom; its size and limits.

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
