"""Value-set abstraction of a single atom / conjunction of atoms on ONE attribute.

Used by Tier-A sound pruning. Two abstract values are either provably disjoint
or "maybe overlapping" (never "provably overlapping" — that is the SMT step's job).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .rule import Atom


@dataclass(frozen=True)
class Interval:
    lo: float = -math.inf
    hi: float = math.inf
    lo_open: bool = False
    hi_open: bool = False

    def is_empty(self) -> bool:
        if self.lo > self.hi:
            return True
        return self.lo == self.hi and (self.lo_open or self.hi_open)

    def intersect(self, o: "Interval") -> "Interval":
        if self.lo > o.lo or (self.lo == o.lo and self.lo_open):
            lo, lo_open = self.lo, self.lo_open
        else:
            lo, lo_open = o.lo, o.lo_open
        if self.hi < o.hi or (self.hi == o.hi and self.hi_open):
            hi, hi_open = self.hi, self.hi_open
        else:
            hi, hi_open = o.hi, o.hi_open
        return Interval(lo, hi, lo_open, hi_open)


@dataclass(frozen=True)
class ValueSet:
    """Categorical abstraction: either an allow-list or a deny-list."""
    allow: frozenset | None = None   # None => unrestricted
    deny: frozenset = frozenset()

    def intersect(self, o: "ValueSet") -> "ValueSet":
        if self.allow is None:
            allow = o.allow
        elif o.allow is None:
            allow = self.allow
        else:
            allow = self.allow & o.allow
        deny = self.deny | o.deny
        if allow is not None:
            allow = allow - deny
            deny = frozenset()
        return ValueSet(allow, deny)

    def is_empty(self) -> bool:
        return self.allow is not None and len(self.allow) == 0


def atom_to_interval(a: Atom) -> Interval | None:
    """Return an Interval for numeric-style atoms, None if not interval-expressible."""
    v = a.value
    if a.op == "eq":
        return Interval(v, v)
    if a.op == "lt":
        return Interval(hi=v, hi_open=True)
    if a.op == "le":
        return Interval(hi=v)
    if a.op == "gt":
        return Interval(lo=v, lo_open=True)
    if a.op == "ge":
        return Interval(lo=v)
    if a.op == "range":
        lo, hi = v
        return Interval(lo, hi)
    if a.op == "any":
        return Interval()
    return None  # neq / in / notin are not single intervals


def atom_to_valueset(a: Atom) -> ValueSet | None:
    v = a.value
    if a.op == "eq":
        return ValueSet(frozenset([v]))
    if a.op == "in":
        return ValueSet(frozenset(v))
    if a.op == "neq":
        return ValueSet(None, frozenset([v]))
    if a.op == "notin":
        return ValueSet(None, frozenset(v))
    if a.op == "any":
        return ValueSet()
    return None


def is_numeric_value(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def abstract_attr(atoms: list[Atom]) -> tuple[Interval | None, ValueSet | None]:
    """Fold all atoms on one attribute into (Interval, ValueSet) abstractions.

    Either component may be None when no atom of that flavour applies.
    Soundness: each abstraction is an OVER-approximation of the set of
    satisfying values, so disjoint abstractions ⇒ disjoint concrete sets.
    """
    itv: Interval | None = None
    vs: ValueSet | None = None
    for a in atoms:
        numeric = a.op in {"lt", "le", "gt", "ge", "range"} or (
            a.op == "eq" and is_numeric_value(a.value))
        if numeric:
            i = atom_to_interval(a)
            if i is not None:
                itv = i if itv is None else itv.intersect(i)
        else:
            s = atom_to_valueset(a)
            if s is not None:
                vs = s if vs is None else vs.intersect(s)
    return itv, vs
