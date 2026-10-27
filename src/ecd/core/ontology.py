"""Ontology over the attribute space: attribute aliases and value hierarchies.

Heterogeneous policies (merged from several sources) describe the same concept
with different vocabularies, e.g. `role = doctor` in one source and
`jobTitle = physician` in another.  Role hierarchies add a second kind of
implicit overlap: a request with `role = senior_doctor` also satisfies a rule
written for `role = doctor`.

The ontology is part of the *semantics* used by every exact step (Tier A,
Z3, Event-B): rules are canonicalised before they are reasoned about.
    canonical attribute : alias attr  → canonical attr
    canonical value     : alias value → canonical value
    hierarchy           : child value → parent value  (per canonical attr)

Canonicalisation of an atom `attr op value`:
    alias         : rename attr, map value(s)
    hierarchy     : `eq v`  → `in  desc(v)`      (desc = v and all its descendants)
                    `neq v` → `notin desc(v)`
                    `in S`  → `in  ∪ desc(s)`,   `notin S` → `notin ∪ desc(s)`
Numeric atoms are never affected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rule import Atom, Predicate


@dataclass
class AliasSpec:
    canonical: str
    values: dict[Any, Any] | None = None     # alias value → canonical value (None: identity)

    def map_value(self, v: Any) -> Any:
        if self.values is None:
            return v
        return self.values.get(v, v)


@dataclass
class Ontology:
    aliases: dict[str, AliasSpec] = field(default_factory=dict)          # alias attr → spec
    hierarchy: dict[str, dict[Any, Any]] = field(default_factory=dict)   # attr → {child: parent}
    _desc_cache: dict[tuple[str, Any], frozenset] = field(default_factory=dict, repr=False)

    # ----- queries -------------------------------------------------------
    def is_empty(self) -> bool:
        return not self.aliases and not self.hierarchy

    def canonical_attr(self, name: str) -> str:
        spec = self.aliases.get(name)
        return spec.canonical if spec else name

    def canonical_value(self, name: str, v: Any) -> Any:
        spec = self.aliases.get(name)
        return spec.map_value(v) if spec else v

    def parent(self, attr: str, v: Any) -> Any | None:
        return self.hierarchy.get(attr, {}).get(v)

    def descendants(self, attr: str, v: Any) -> frozenset:
        """v plus every value that (transitively) specialises v."""
        key = (attr, v)
        if key in self._desc_cache:
            return self._desc_cache[key]
        h = self.hierarchy.get(attr, {})
        out = {v}
        frontier = [v]
        while frontier:
            x = frontier.pop()
            for child, par in h.items():
                if par == x and child not in out:
                    out.add(child)
                    frontier.append(child)
        res = frozenset(out)
        self._desc_cache[key] = res
        return res

    # ----- canonicalisation -----------------------------------------------
    def canonical_atom(self, a: Atom) -> Atom:
        attr = self.canonical_attr(a.attr)
        if a.op in ("lt", "le", "gt", "ge", "range", "any"):
            return a if attr == a.attr else Atom(attr, a.op, a.value)
        if a.op in ("eq", "neq"):
            v = self.canonical_value(a.attr, a.value)
            desc = self.descendants(attr, v) if attr in self.hierarchy else frozenset([v])
            if len(desc) == 1:
                return a if (attr == a.attr and v == a.value) else Atom(attr, a.op, v)
            vals = tuple(sorted(desc, key=str))
            return Atom(attr, "in" if a.op == "eq" else "notin", vals)
        if a.op in ("in", "notin"):
            vals: set = set()
            for x in a.value:
                v = self.canonical_value(a.attr, x)
                vals |= set(self.descendants(attr, v)) if attr in self.hierarchy else {v}
            if attr == a.attr and vals == set(a.value):
                return a
            return Atom(attr, a.op, tuple(sorted(vals, key=str)))
        return a

    def canonical_predicate(self, p: Predicate) -> Predicate:
        if self.is_empty():
            return p
        atoms = tuple(self.canonical_atom(a) for a in p.atoms)
        return p if atoms == p.atoms else Predicate(atoms, p.raw_expr)

    # ----- (de)serialisation ------------------------------------------
    def to_dict(self) -> dict:
        return {
            "aliases": {a: {"canonical": s.canonical,
                            "values": ({str(k): v for k, v in s.values.items()} if s.values else None)}
                        for a, s in self.aliases.items()},
            "hierarchy": {attr: {str(c): p for c, p in m.items()} for attr, m in self.hierarchy.items()},
        }

    @staticmethod
    def from_dict(d: dict | None) -> "Ontology":
        o = Ontology()
        if not d:
            return o
        for a, s in (d.get("aliases") or {}).items():
            o.aliases[a] = AliasSpec(s["canonical"], dict(s["values"]) if s.get("values") else None)
        for attr, m in (d.get("hierarchy") or {}).items():
            o.hierarchy[attr] = dict(m)
        return o
