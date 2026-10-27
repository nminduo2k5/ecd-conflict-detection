"""Core data model: attributes, atoms, predicates, rules.

Formal definitions (paper §4):
    rule r_i = (T_i, C_i, eff_i)
    T_i, C_i : predicates over the attribute space 𝒜
    conflict(r_i, r_j) ⇔ ∃q ∈ 𝒜: T_i(q) ∧ C_i(q) ∧ T_j(q) ∧ C_j(q) ∧ eff_i ≠ eff_j

A Predicate is a *conjunction* of Atoms plus an optional `raw_expr`
(SMT-LIB string) for complex conditions that cannot be expressed as atoms.
Because `raw_expr` is just one more conjunct, any disjointness proven on the
atoms alone remains valid (this is what makes Tier-A pruning sound).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable


class Effect(str, Enum):
    PERMIT = "Permit"
    DENY = "Deny"


class AttrType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class Attribute:
    """Declaration of one attribute in the attribute space 𝒜."""
    name: str
    type: AttrType
    domain: tuple[Any, ...] | None = None  # categorical: allowed values
    lo: float | None = None                # numeric: lower bound (inclusive)
    hi: float | None = None                # numeric: upper bound (inclusive)


# Supported atom operators.
#   categorical : eq, neq, in, notin
#   numeric     : eq, neq, lt, le, gt, ge, range  (value = (lo, hi) inclusive)
#   boolean     : eq
#   any         : attribute is referenced but unconstrained
OPS = {"eq", "neq", "in", "notin", "lt", "le", "gt", "ge", "range", "any"}


@dataclass(frozen=True)
class Atom:
    attr: str
    op: str
    value: Any = None

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"unknown op {self.op!r}; expected one of {sorted(OPS)}")
        # normalise list-valued fields to tuples so the atom stays hashable
        if isinstance(self.value, list):
            object.__setattr__(self, "value", tuple(self.value))

    def to_dict(self) -> dict:
        v = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"attr": self.attr, "op": self.op, "value": v}

    @staticmethod
    def from_dict(d: dict) -> "Atom":
        return Atom(d["attr"], d["op"], d.get("value"))


@dataclass(frozen=True)
class Predicate:
    """Conjunction of atoms, optionally AND-ed with a raw SMT-LIB expression."""
    atoms: tuple[Atom, ...] = ()
    raw_expr: str | None = None  # SMT-LIB2 assertion body, e.g. "(> (+ x y) 10)"

    @property
    def is_true(self) -> bool:
        return not self.atoms and self.raw_expr is None

    def attrs(self) -> set[str]:
        return {a.attr for a in self.atoms}

    def atoms_on(self, attr: str) -> list[Atom]:
        return [a for a in self.atoms if a.attr == attr]

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"atoms": [a.to_dict() for a in self.atoms]}
        if self.raw_expr:
            d["raw_expr"] = self.raw_expr
        return d

    @staticmethod
    def from_dict(d: dict | None) -> "Predicate":
        if not d:
            return Predicate()
        return Predicate(tuple(Atom.from_dict(a) for a in d.get("atoms", [])),
                         d.get("raw_expr"))


@dataclass(frozen=True)
class Rule:
    id: str
    target: Predicate
    condition: Predicate
    effect: Effect
    meta: dict = field(default_factory=dict, hash=False, compare=False)

    # ----- derived -------------------------------------------------------
    @property
    def applicability(self) -> Predicate:
        """T ∧ C as a single predicate (raw_exprs are AND-ed)."""
        raws = [p.raw_expr for p in (self.target, self.condition) if p.raw_expr]
        raw = None if not raws else (raws[0] if len(raws) == 1 else f"(and {' '.join(raws)})")
        return Predicate(self.target.atoms + self.condition.atoms, raw)

    def attrs(self) -> set[str]:
        return self.target.attrs() | self.condition.attrs()

    def has_complex_condition(self) -> bool:
        return bool(self.target.raw_expr or self.condition.raw_expr)

    def content_hash(self) -> str:
        """Hash of semantic content (used for PO reuse across Δ). Memoised: the
        rule is immutable, and this is on the hot path of Tier A / PO cache."""
        h = self.__dict__.get("_hash")
        if h is None:
            payload = json.dumps(
                {"t": self.target.to_dict(), "c": self.condition.to_dict(), "e": self.effect.value},
                sort_keys=True, default=str,
            )
            h = hashlib.sha1(payload.encode()).hexdigest()[:16]
            object.__setattr__(self, "_hash", h)
        return h

    # ----- (de)serialisation ------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target": self.target.to_dict(),
            "condition": self.condition.to_dict(),
            "effect": self.effect.value,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Rule":
        return Rule(
            id=str(d["id"]),
            target=Predicate.from_dict(d.get("target")),
            condition=Predicate.from_dict(d.get("condition")),
            effect=Effect(d["effect"]),
            meta=dict(d.get("meta", {})),
        )


def pair_key(a: str, b: str) -> tuple[str, str]:
    """Canonical unordered key for a rule pair."""
    return (a, b) if a <= b else (b, a)
