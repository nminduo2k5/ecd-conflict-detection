"""Pair-level metrics for RQ1–RQ5."""
from __future__ import annotations

import numpy as np

from ecd.core import pair_key


def precision_recall(found: set[tuple[str, str]], truth: set[tuple[str, str]]) -> tuple[float, float]:
    found = {pair_key(*p) for p in found}
    truth = {pair_key(*p) for p in truth}
    tp = len(found & truth)
    prec = tp / len(found) if found else 1.0
    rec = tp / len(truth) if truth else 1.0
    return prec, rec


def recall_on(found: set[tuple[str, str]], subset: set[tuple[str, str]]) -> float:
    found = {pair_key(*p) for p in found}
    subset = {pair_key(*p) for p in subset}
    return len(found & subset) / len(subset) if subset else float("nan")


def recall_at_budget(conflicts_over_time: list[tuple[float, int]], total_truth: int, budget_ms: float) -> float:
    """% of true conflicts found within a wall-clock budget (exact mode, RQ2)."""
    found = 0
    for t, cum in conflicts_over_time:
        if t <= budget_ms:
            found = cum
    return found / total_truth if total_truth else 1.0


def ranking_quality(scores: list[float], labels: list[int]) -> dict[str, float]:
    """AUC + precision@k for the embedding ranker (RQ1)."""
    s = np.asarray(scores)
    y = np.asarray(labels)
    order = np.argsort(-s)
    y_sorted = y[order]
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        auc = float("nan")
    else:
        ranks = np.argsort(np.argsort(s)) + 1
        auc = float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    out = {"auc": auc}
    for k in (10, 50, 100):
        out[f"p@{k}"] = float(y_sorted[:k].mean()) if len(y_sorted) >= k else float("nan")
    return out


def mean_std(xs) -> dict:
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)), "n": int(a.size)}
