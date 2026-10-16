"""Sentiment models compared on the hand-labelled (sentence, aspect) pairs.

Every model exposes the same call: ``model.predict(df) -> list[str]`` where ``df`` has the
columns ``sentence`` and ``aspect``, and labels are "positive" / "negative" / "neutral".

* MajorityModel       : always predicts the most common training label (floor).
* BoxLabelModel       : the text box the comment came from (needs ``field``); a weak reference.
* TfidfLRModel        : TF-IDF + logistic regression trained on weak (box) labels. Aspect-blind.
* HFSentimentModel    : pretrained transformer sentiment classifier. Aspect-blind.
* HFAbsaModel         : pretrained aspect-based classifier; reads (sentence, aspect). Aspect-aware.
* LLMPromptModel      : optional; you pass in a function that calls an LLM.

Hugging Face models are downloaded on first use (needs internet; works on Colab). The model
names below are public Hub ids; check they still exist if loading fails.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable

import pandas as pd

LABELS = ["negative", "neutral", "positive"]

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ABSA_MODEL = "yangheng/deberta-v3-base-absa-v1.1"


def normalise_label(raw: str) -> str:
    """Map whatever a model returns ("Positive", "LABEL_2", "POS", ...) to our three labels."""
    s = str(raw).strip().lower()
    table = {"label_0": "negative", "label_1": "neutral", "label_2": "positive",
             "neg": "negative", "neu": "neutral", "pos": "positive"}
    s = table.get(s, s)
    if s not in LABELS:
        raise ValueError(f"Unrecognised model label {raw!r}; extend normalise_label().")
    return s


class MajorityModel:
    name = "Majority class"
    aspect_aware = False

    def fit(self, labels: Iterable[str]):
        self.label_ = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, df: pd.DataFrame) -> list[str]:
        return [self.label_] * len(df)


class BoxLabelModel:
    """Predicts 'positive' for comments written in the positive box, 'negative' otherwise."""
    name = "Box label (weak reference)"
    aspect_aware = False

    def __init__(self, field_by_key: dict):
        self.field_by_key = field_by_key  # {(review_id, sent_id): "positive"/"negative"}

    def predict(self, df: pd.DataFrame) -> list[str]:
        return [self.field_by_key[(r, s)] for r, s in zip(df["review_id"], df["sent_id"])]


class TfidfLRModel:
    name = "TF-IDF + LogReg (weak labels)"
    aspect_aware = False

    def __init__(self, seed: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)

    def fit(self, sentences: Iterable[str], labels: Iterable[str]):
        self.clf.fit(self.vec.fit_transform(list(sentences)), list(labels))
        return self

    def predict(self, df: pd.DataFrame) -> list[str]:
        return list(self.clf.predict(self.vec.transform(df["sentence"].tolist())))


class HFSentimentModel:
    name = "RoBERTa sentiment (pretrained)"
    aspect_aware = False

    def __init__(self, model_name: str = SENTIMENT_MODEL, batch_size: int = 32, device=None):
        from transformers import pipeline
        kw = {} if device is None else {"device": device}
        self.pipe = pipeline("text-classification", model=model_name, truncation=True,
                             max_length=256, **kw)
        self.batch_size = batch_size

    def predict(self, df: pd.DataFrame) -> list[str]:
        out = self.pipe(df["sentence"].tolist(), batch_size=self.batch_size)
        return [normalise_label(o["label"]) for o in out]


class HFAbsaModel:
    name = "DeBERTa ABSA (pretrained)"
    aspect_aware = True

    def __init__(self, model_name: str = ABSA_MODEL, batch_size: int = 16, device=None):
        from transformers import pipeline
        kw = {} if device is None else {"device": device}
        self.pipe = pipeline("text-classification", model=model_name, truncation=True,
                             max_length=256, **kw)
        self.batch_size = batch_size

    def predict(self, df: pd.DataFrame) -> list[str]:
        inputs = [{"text": s, "text_pair": a} for s, a in zip(df["sentence"], df["aspect"])]
        out = self.pipe(inputs, batch_size=self.batch_size)
        return [normalise_label(o["label"]) for o in out]


def build_llm_prompt(sentence: str, aspect: str) -> str:
    return (
        "You label hotel-review comments. The text has no punctuation.\n"
        f'Comment: "{sentence}"\nAspect: {aspect}\n'
        "What sentiment does the comment express toward THIS aspect only? "
        "Answer with exactly one word: positive, negative or neutral."
    )


class LLMPromptModel:
    """Optional. ``ask`` is any function str -> str that calls an LLM of your choice."""
    aspect_aware = True

    def __init__(self, ask: Callable[[str], str], name: str = "LLM prompt"):
        self.ask, self.name = ask, name

    def predict(self, df: pd.DataFrame) -> list[str]:
        preds = []
        for s, a in zip(df["sentence"], df["aspect"]):
            reply = self.ask(build_llm_prompt(s, a)).strip().lower()
            preds.append(next((l for l in LABELS if l in reply), "neutral"))
        return preds
