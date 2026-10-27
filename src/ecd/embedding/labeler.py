"""Automatic label generation for contrastive training (C1, RQ1).

label(r_i, r_j) = 1  if  SAT(⟦r_i⟧ ∧ ⟦r_j⟧)   (canonical regions overlap)
                 = 0  otherwise

Z3 is the oracle (ontology-aware, see core.ontology). To keep labelling
cheap on large policies we:
  1. run Tier-A pruning with ignore_effect=True — pairs it discards are
     provably non-overlapping and get label 0 for free;
  2. sample a bounded number of surviving pairs for the SMT oracle;
  3. balance positives/negatives (hard negatives = survivors that are UNSAT,
     which are exactly the cases a text-similarity model gets wrong).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ecd.core import Policy, Rule
from ecd.filtering.sound_pruning import Pruner
from ecd.verification.smt_checker import SMTChecker


@dataclass
class LabelledPair:
    a: str
    b: str
    label: int          # 1 = overlap, 0 = disjoint
    hard: bool = False  # negative that survived Tier A (hard negative)


def label_pairs(policy: Policy, *, max_smt_calls: int = 20_000,
                neg_ratio: float = 1.0, seed: int = 0,
                checker: SMTChecker | None = None,
                max_pairs_scanned: int | None = 2_000_000) -> list[LabelledPair]:
    rng = random.Random(seed)
    checker = checker or SMTChecker(policy.space)
    pruner = Pruner(policy.space.ontology)
    rules = list(policy)
    n = len(rules)

    survivors: list[tuple[Rule, Rule]] = []
    easy_neg: list[tuple[Rule, Rule]] = []
    all_pairs = n * (n - 1) // 2
    if max_pairs_scanned is not None and all_pairs > max_pairs_scanned:
        # large policy: sample random pairs instead of enumerating all
        seen = set()
        while len(seen) < max_pairs_scanned:
            i, j = rng.randrange(n), rng.randrange(n)
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            d = pruner.decide(rules[key[0]], rules[key[1]], ignore_effect=True)
            (easy_neg if d.pruned else survivors).append((rules[key[0]], rules[key[1]]))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                d = pruner.decide(rules[i], rules[j], ignore_effect=True)
                (easy_neg if d.pruned else survivors).append((rules[i], rules[j]))

    rng.shuffle(survivors)
    out: list[LabelledPair] = []
    pos = 0
    for a, b in survivors[:max_smt_calls]:
        res = checker.check_pair(a, b)
        if res.status == "unknown":
            continue
        out.append(LabelledPair(a.id, b.id, int(res.overlap), hard=not res.overlap))
        pos += int(res.overlap)

    # top up negatives from the cheap pool until we hit the requested ratio
    need = max(0, int(pos * neg_ratio) - sum(1 for p in out if p.label == 0))
    rng.shuffle(easy_neg)
    out += [LabelledPair(a.id, b.id, 0) for a, b in easy_neg[:need]]
    rng.shuffle(out)
    return out


def split_by_rule(pairs: list[LabelledPair], test_frac: float = 0.2, seed: int = 0
                  ) -> tuple[list[LabelledPair], list[LabelledPair]]:
    """Hold out a fraction of *rules*: a test pair touches ≥1 held-out rule, a
    train pair touches none (unseen-rule generalisation, stricter than a pair split)."""
    rng = random.Random(seed)
    ids = sorted({p.a for p in pairs} | {p.b for p in pairs})
    rng.shuffle(ids)
    held = set(ids[: int(len(ids) * test_frac)])
    train = [p for p in pairs if p.a not in held and p.b not in held]
    test = [p for p in pairs if p.a in held or p.b in held]
    return train, test
