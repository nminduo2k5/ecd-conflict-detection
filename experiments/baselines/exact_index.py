"""Baseline B2: exact attribute-value inverted index.

Candidate pairs = rules sharing ≥1 identical (attr, value) literal. This is
the "why not just use an index?" reviewer baseline; it misses heterogeneous
attributes (role vs jobTitle), hierarchies, and raw conditions.

With an `ontology` the index is built over the *canonical* vocabulary
(hand-coded alias map + hierarchy expansion) — the strongest version of the
baseline a practitioner could build.
"""
from __future__ import annotations

from collections import defaultdict

from ecd.core import Ontology, Policy, pair_key


def candidate_pairs(policy: Policy, ontology: Ontology | None = None) -> set[tuple[str, str]]:
    idx: dict[tuple, list[str]] = defaultdict(list)
    for r in policy:
        pred = ontology.canonical_predicate(r.applicability) if ontology else r.applicability
        for a in pred.atoms:
            vals = a.value if isinstance(a.value, tuple) else (a.value,)
            if a.op in ("eq", "in"):
                for v in vals:
                    idx[(a.attr, v)].append(r.id)
            else:
                idx[(a.attr, "*")].append(r.id)
    pairs = set()
    for ids in idx.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = policy.get(ids[i]), policy.get(ids[j])
                if a.effect != b.effect:
                    pairs.add(pair_key(ids[i], ids[j]))
    return pairs
