"""Baseline B3: regenerate the whole Event-B context after every Δ (no refinement).

Measures #POs generated per Δ versus the incremental scheme (RQ3)."""
from __future__ import annotations

import itertools

from ecd.core import Policy
from ecd.verification.eventb import base_context


def po_count_full(policy: Policy) -> int:
    rules = list(policy)
    pairs = [(a, b) for a, b in itertools.combinations(rules, 2) if a.effect != b.effect]
    return len(base_context(policy, pairs).theorems)
