"""Step [5]: classify a detected conflict w.r.t. the policy's combining algorithm.

A pair with intersecting applicability and different effects is a *potential*
conflict. Whether it is *observable* depends on the combining algorithm:
  - deny-overrides / permit-overrides : one effect always wins → conflict is
    *masked* (but still a redundancy / intent smell worth reporting).
  - first-applicable : the earlier rule wins → masked, order-dependent.
  - only-one-applicable : both apply → Indeterminate → *real* conflict.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ecd.core import CombiningAlgorithm, Effect, Policy, Rule


class ConflictClass(str, Enum):
    REAL = "real"              # observable decision error / indeterminate
    MASKED = "masked"          # resolved deterministically by combining alg
    ORDER_DEPENDENT = "order-dependent"  # masked but fragile under reordering


@dataclass
class ConflictVerdict:
    cls: ConflictClass
    winner: str | None          # id of rule whose effect prevails
    explanation: str


def classify(policy: Policy, a: Rule, b: Rule) -> ConflictVerdict:
    alg = policy.combining
    if alg is CombiningAlgorithm.DENY_OVERRIDES:
        w = a if a.effect is Effect.DENY else b
        return ConflictVerdict(ConflictClass.MASKED, w.id,
                               f"deny-overrides: {w.id} (Deny) wins on the overlap")
    if alg is CombiningAlgorithm.PERMIT_OVERRIDES:
        w = a if a.effect is Effect.PERMIT else b
        return ConflictVerdict(ConflictClass.MASKED, w.id,
                               f"permit-overrides: {w.id} (Permit) wins on the overlap")
    if alg is CombiningAlgorithm.FIRST_APPLICABLE:
        w = a if policy.index_of(a.id) < policy.index_of(b.id) else b
        return ConflictVerdict(ConflictClass.ORDER_DEPENDENT, w.id,
                               f"first-applicable: {w.id} appears first; swapping order flips the decision")
    return ConflictVerdict(ConflictClass.REAL, None,
                           "only-one-applicable: both rules apply → Indeterminate")
