"""Baseline B1: exhaustive pairwise SMT (also produces the ground-truth set)."""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from ecd.core import Policy, pair_key
from ecd.verification import SMTChecker


def run(policy: Policy, save_truth: str | Path | None = None) -> tuple[set, float]:
    chk = SMTChecker(policy.space)
    rules = list(policy)
    t0 = time.perf_counter()
    found = set()
    for a, b in itertools.combinations(rules, 2):
        if a.effect == b.effect:
            continue  # by definition not a conflict (not a pruning heuristic)
        if chk.check_pair(a, b).conflict:
            found.add(pair_key(a.id, b.id))
    ms = (time.perf_counter() - t0) * 1000
    if save_truth:
        Path(save_truth).write_text(json.dumps(sorted(found)))
    return found, ms
