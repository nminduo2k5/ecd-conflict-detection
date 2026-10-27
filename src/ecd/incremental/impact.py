"""Impact(Δ): the set of rule pairs whose verification status may change.

Theorem 2 (preservation under Δ): for every pair (r_i, r_j) with
r_i, r_j ∉ touched(Δ), the proof obligation PO(r_i, r_j) proved before Δ is
still valid after Δ, because PO(r_i, r_j) depends only on the content of
r_i and r_j, which Δ does not alter.  Hence only pairs with at least one
endpoint in touched(Δ) need (re)verification.

touched(add r)       = {r}
touched(remove r)    = {r}          (pairs are simply dropped)
touched(modify r→r') = {r'}         (old pairs invalidated, new ones generated)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ecd.core import Delta, DeltaKind, Policy, Rule, pair_key


@dataclass
class Impact:
    touched: set[str] = field(default_factory=set)
    invalidated_pairs: set[tuple[str, str]] = field(default_factory=set)
    new_targets: list[Rule] = field(default_factory=list)   # rules needing pair checks


def compute_impact(policy_before: Policy, delta: Delta) -> Impact:
    imp = Impact()
    rid = delta.target_id
    imp.touched.add(rid)
    if delta.kind in (DeltaKind.REMOVE, DeltaKind.MODIFY) and rid in policy_before:
        for other in policy_before.order:
            if other != rid:
                imp.invalidated_pairs.add(pair_key(rid, other))
    if delta.kind in (DeltaKind.ADD, DeltaKind.MODIFY):
        imp.new_targets.append(delta.rule)
    return imp
