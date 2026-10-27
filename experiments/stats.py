"""Statistical tests: Wilcoxon signed-rank + Vargha–Delaney A12 (paper §5)."""
from __future__ import annotations

import numpy as np


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as _w
    return _w(a, b)


def a12(a, b) -> float:
    """Vargha–Delaney A12: P(a > b) + 0.5 P(a = b). 0.5 = no effect,
    >0.71 large, >0.64 medium, >0.56 small."""
    a, b = np.asarray(a), np.asarray(b)
    more = (a[:, None] > b[None, :]).sum()
    same = (a[:, None] == b[None, :]).sum()
    return float((more + 0.5 * same) / (len(a) * len(b)))


def summarize(name: str, treat, control) -> dict:
    w = wilcoxon(treat, control)
    return {"metric": name, "treat_mean": float(np.mean(treat)), "control_mean": float(np.mean(control)),
            "p": float(w.pvalue), "A12": a12(treat, control)}
