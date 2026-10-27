"""Policy container + attribute space + evolution operations (Δ)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

from .rule import Attribute, AttrType, Predicate, Rule
from .ontology import Ontology


class CombiningAlgorithm(str, Enum):
    DENY_OVERRIDES = "deny-overrides"
    PERMIT_OVERRIDES = "permit-overrides"
    FIRST_APPLICABLE = "first-applicable"
    ONLY_ONE_APPLICABLE = "only-one-applicable"


@dataclass
class AttributeSpace:
    attributes: dict[str, Attribute] = field(default_factory=dict)
    ontology: Ontology = field(default_factory=Ontology)

    def add(self, attr: Attribute) -> None:
        self.attributes[attr.name] = attr

    def get(self, name: str) -> Attribute | None:
        return self.attributes.get(name)

    def ensure(self, name: str, type_: AttrType = AttrType.CATEGORICAL) -> Attribute:
        if name not in self.attributes:
            self.attributes[name] = Attribute(name, type_)
        return self.attributes[name]

    def canonical(self, p: Predicate) -> Predicate:
        """Predicate rewritten over canonical attributes/values (see core.ontology)."""
        return self.ontology.canonical_predicate(p)

    def applicability(self, r: Rule) -> Predicate:
        """Canonical T ∧ C of a rule — the predicate every exact step reasons about."""
        return self.canonical(r.applicability)

    def to_dict(self) -> dict:
        out = {}
        for a in self.attributes.values():
            d = {"type": a.type.value}
            if a.domain is not None:
                d["domain"] = list(a.domain)
            if a.lo is not None:
                d["lo"] = a.lo
            if a.hi is not None:
                d["hi"] = a.hi
            out[a.name] = d
        return out

    @staticmethod
    def from_dict(d: dict, ontology: dict | None = None) -> "AttributeSpace":
        space = AttributeSpace(ontology=Ontology.from_dict(ontology))
        for name, spec in (d or {}).items():
            dom = spec.get("domain")
            space.add(Attribute(name, AttrType(spec["type"]),
                                tuple(dom) if dom is not None else None,
                                spec.get("lo"), spec.get("hi")))
        return space


@dataclass
class Policy:
    """An ordered set of rules. Order matters for first-applicable."""
    name: str
    rules: dict[str, Rule] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    space: AttributeSpace = field(default_factory=AttributeSpace)
    combining: CombiningAlgorithm = CombiningAlgorithm.DENY_OVERRIDES
    version: int = 0

    # ----- container API --------------------------------------------------
    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> Iterator[Rule]:
        return (self.rules[i] for i in self.order)

    def __contains__(self, rid: str) -> bool:
        return rid in self.rules

    def get(self, rid: str) -> Rule:
        return self.rules[rid]

    def add(self, rule: Rule, position: int | None = None) -> None:
        if rule.id in self.rules:
            raise KeyError(f"rule {rule.id} already exists")
        self.rules[rule.id] = rule
        if position is None:
            self.order.append(rule.id)
        else:
            self.order.insert(position, rule.id)
        self._register_attrs(rule)

    def remove(self, rid: str) -> Rule:
        r = self.rules.pop(rid)
        self.order.remove(rid)
        return r

    def replace(self, rule: Rule) -> Rule:
        old = self.rules[rule.id]
        self.rules[rule.id] = rule
        self._register_attrs(rule)
        return old

    def _register_attrs(self, rule: Rule) -> None:
        for a in rule.target.atoms + rule.condition.atoms:
            self.space.ensure(a.attr)

    def index_of(self, rid: str) -> int:
        return self.order.index(rid)

    def copy(self) -> "Policy":
        p = Policy(self.name, dict(self.rules), list(self.order),
                   AttributeSpace(dict(self.space.attributes), self.space.ontology),
                   self.combining, self.version)
        return p

    # ----- (de)serialisation ------------------------------------------
    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "version": self.version,
            "combining": self.combining.value,
            "attributes": self.space.to_dict(),
            "rules": [self.rules[i].to_dict() for i in self.order],
        }
        if not self.space.ontology.is_empty():
            d["ontology"] = self.space.ontology.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> "Policy":
        p = Policy(name=d.get("name", "policy"),
                   space=AttributeSpace.from_dict(d.get("attributes", {}), d.get("ontology")),
                   combining=CombiningAlgorithm(d.get("combining", "deny-overrides")),
                   version=int(d.get("version", 0)))
        for rd in d["rules"]:
            p.add(Rule.from_dict(rd))
        return p

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: str | Path) -> "Policy":
        return Policy.from_dict(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------
# Evolution Δ
# --------------------------------------------------------------------------
class DeltaKind(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"


@dataclass
class Delta:
    kind: DeltaKind
    rule: Rule | None = None          # ADD / MODIFY: new rule content
    rule_id: str | None = None        # REMOVE / MODIFY: id of affected rule
    position: int | None = None       # ADD: insertion index

    @property
    def target_id(self) -> str:
        return self.rule_id or (self.rule.id if self.rule else "")

    def to_dict(self) -> dict:
        return {"kind": self.kind.value,
                "rule": self.rule.to_dict() if self.rule else None,
                "rule_id": self.rule_id, "position": self.position}

    @staticmethod
    def from_dict(d: dict) -> "Delta":
        return Delta(DeltaKind(d["kind"]),
                     Rule.from_dict(d["rule"]) if d.get("rule") else None,
                     d.get("rule_id"), d.get("position"))


def apply_delta(policy: Policy, delta: Delta) -> Policy:
    """Apply Δ in place and bump version. Returns the same policy."""
    if delta.kind is DeltaKind.ADD:
        policy.add(delta.rule, delta.position)
    elif delta.kind is DeltaKind.REMOVE:
        policy.remove(delta.target_id)
    elif delta.kind is DeltaKind.MODIFY:
        policy.replace(delta.rule)
    policy.version += 1
    return policy
