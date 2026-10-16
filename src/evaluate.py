"""Loading the hand-labelled pairs, annotation agreement, and model scoring."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

SENTIMENTS = ["negative", "neutral", "positive"]
ALL_LABELS = SENTIMENTS + ["not_applicable"]
DIFFICULTIES = ["negation", "contrast", "multi_aspect", "plain"]
KEY = ["review_id", "sent_id", "aspect"]


def load_labelled(path) -> pd.DataFrame:
    """Read labelled_pairs.csv and check it is complete and uses the agreed labels."""
    df = pd.read_csv(path)
    for col in KEY + ["sentence", "gold_sentiment", "difficulty"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {path}")
    for col in ["gold_sentiment", "difficulty"]:
        df[col] = df[col].astype(str).str.strip().str.lower().replace({"nan": ""})
    bad = df[~df["gold_sentiment"].isin(ALL_LABELS)]
    if len(bad):
        raise ValueError(f"{len(bad)} rows have a missing/unknown gold_sentiment, e.g. rows {list(bad.index[:5])}: "
                         f"{bad['gold_sentiment'].unique().tolist()}")
    bad = df[~df["difficulty"].isin(DIFFICULTIES)]
    if len(bad):
        raise ValueError(f"{len(bad)} rows have a missing/unknown difficulty, e.g. rows {list(bad.index[:5])}")
    if df.duplicated(KEY).any():
        raise ValueError("Duplicate (review_id, sent_id, aspect) rows found.")
    return df


def annotation_agreement(df: pd.DataFrame) -> dict:
    """Compare the human labels (gold_*) with the frozen AI pre-labels (ai_*)."""
    if "ai_sentiment" not in df.columns:
        raise ValueError("No ai_sentiment column: use the AI pre-labelled sheet as the starting point.")
    g, a = df["gold_sentiment"], df["ai_sentiment"].astype(str).str.strip().str.lower()
    out = {
        "n": len(df),
        "sentiment_agreement": float((g == a).mean()),
        "sentiment_kappa": float(cohen_kappa_score(g, a)),
        "sentiment_changed": int((g != a).sum()),
        "not_applicable_gold": int((g == "not_applicable").sum()),
        "not_applicable_ai": int((a == "not_applicable").sum()),
    }
    if "ai_difficulty" in df.columns:
        ad = df["ai_difficulty"].astype(str).str.strip().str.lower()
        out["difficulty_agreement"] = float((df["difficulty"] == ad).mean())
        out["difficulty_kappa"] = float(cohen_kappa_score(df["difficulty"], ad))
        out["difficulty_changed"] = int((df["difficulty"] != ad).sum())
    labels = [l for l in ALL_LABELS if l in set(g) | set(a)]
    out["confusion"] = pd.DataFrame(confusion_matrix(g, a, labels=labels),
                                    index=[f"human:{l}" for l in labels],
                                    columns=[f"ai:{l}" for l in labels])
    out["changed_rows"] = df.loc[g != a, KEY[:2] + ["aspect", "sentence", "gold_sentiment", "ai_sentiment"]]
    return out


def evaluable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows a sentiment model can be scored on: drop not_applicable (tagger false alarms)."""
    return df[df["gold_sentiment"].isin(SENTIMENTS)].reset_index(drop=True)


def _scores(y, p) -> dict:
    return {"n": len(y), "accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro", labels=SENTIMENTS, zero_division=0)}


def score_models(df: pd.DataFrame, predictions: dict[str, list[str]]) -> pd.DataFrame:
    """Long table: model x subset (all + each difficulty) with n, accuracy and macro F1.

    ``df`` must be the evaluable rows and each prediction list aligned with it."""
    rows = []
    for model, preds in predictions.items():
        if len(preds) != len(df):
            raise ValueError(f"{model}: {len(preds)} predictions for {len(df)} rows")
        p = pd.Series(preds, index=df.index)
        rows.append({"model": model, "subset": "all", **_scores(df["gold_sentiment"], p)})
        for d in DIFFICULTIES:
            m = df["difficulty"] == d
            if m.any():
                rows.append({"model": model, "subset": d, **_scores(df.loc[m, "gold_sentiment"], p[m])})
    return pd.DataFrame(rows)


def bootstrap_ci(df: pd.DataFrame, preds: list[str], metric: str = "accuracy",
                 n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% bootstrap interval over rows. With ~200 rows these are wide: report them."""
    import numpy as np
    rng = np.random.default_rng(seed)
    y, p = df["gold_sentiment"].to_numpy(), pd.Series(preds).to_numpy()
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(_scores(y[i], p[i])[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def error_table(df: pd.DataFrame, preds: list[str], model: str) -> pd.DataFrame:
    out = df.assign(pred=preds, model=model)
    return out[out["pred"] != out["gold_sentiment"]][
        ["model", "difficulty", "aspect", "sentence", "gold_sentiment", "pred"]]


def plot_by_difficulty(scores: pd.DataFrame, metric: str = "accuracy", path=None):
    """Grouped bars: one group per subset, one bar per model (palette slots in fixed order)."""
    import matplotlib.pyplot as plt
    import numpy as np
    # Reference categorical palette, light mode, slots 1-8 in order (adjacent-pair validated).
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    subsets = [s for s in ["all"] + DIFFICULTIES if s in set(scores["subset"])]
    models = list(dict.fromkeys(scores["model"]))
    if len(models) > len(colours):
        raise ValueError("Plot at most 8 models; fold the rest or facet.")
    n = scores[scores["model"] == models[0]].set_index("subset")["n"]
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=surface)
    ax.set_facecolor(surface)
    for i, m in enumerate(models):
        v = scores[scores["model"] == m].set_index("subset")[metric].reindex(subsets)
        ax.bar(np.arange(len(subsets)) + i * w - 0.4 + w / 2, v.values, width=w * 0.92,
               color=colours[i], label=m)
    ax.set_xticks(range(len(subsets)))
    ax.set_xticklabels([f"{s}\n(n={int(n[s])})" for s in subsets], color=ink)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("_", " "), color=muted)
    ax.set_title(f"Model {metric.replace('_', ' ')} by difficulty group (hand-labelled pairs)",
                 loc="left", color=ink, fontsize=11)
    ax.yaxis.grid(True, color=grid, linewidth=0.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, length=0)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              labelcolor=ink)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=surface, bbox_inches="tight")
    return fig
