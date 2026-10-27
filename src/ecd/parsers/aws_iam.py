"""AWS IAM policy JSON → Policy.

Accepts a bare policy document or the `{"PolicyVersion": {"Document": ...}}`
wrapper returned by `aws iam get-policy-version` (as in public managed-policy
mirrors).  Each Statement becomes one rule; `Action`/`Resource` lists become
`in` atoms, `NotAction`/`NotResource` become `notin` atoms, `*` is left
unconstrained, and conditions map to atoms where the operator is simple
(StringEquals, NumericLessThan, Bool, …).

Wildcard actions (`s3:Get*`) are *patterns*.  With an action registry (the
list of all IAM action names, e.g. from MAMIP's action-registry.json) they
are expanded into the concrete set of matching actions, which is exactly what
IAM does at evaluation time; without a registry they stay literal strings
(an exact index then treats `s3:Get*` and `s3:GetObject` as unrelated — the
"heterogeneous vocabulary" case again).
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from ecd.core import Atom, CombiningAlgorithm, Effect, Policy, Predicate, Rule
from .base import PolicyParser

COND_OPS = {
    "StringEquals": "eq", "StringNotEquals": "neq", "StringEqualsIgnoreCase": "eq",
    "StringNotEqualsIgnoreCase": "neq",
    "NumericEquals": "eq", "NumericNotEquals": "neq", "NumericLessThan": "lt",
    "NumericLessThanEquals": "le", "NumericGreaterThan": "gt", "NumericGreaterThanEquals": "ge",
    "Bool": "eq", "ArnEquals": "eq", "ArnNotEquals": "neq",
}


def _as_list(x):
    return x if isinstance(x, list) else [x]


def load_action_registry(path: str | Path) -> list[str]:
    """MAMIP action-registry.json → sorted list of action names (service:Action)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = d.get("entries", d)
    out = set()
    for k, v in entries.items():
        name = v.get("first_action") if isinstance(v, dict) else None
        out.add(name or k.split("#", 1)[-1])
    return sorted(out)


class AWSIAMParser(PolicyParser):
    def __init__(self, action_registry: list[str] | None = None, max_expand: int = 400):
        self.registry = action_registry
        self._reg_lower = [a.lower() for a in action_registry] if action_registry else None
        self.max_expand = max_expand

    def expand_actions(self, patterns: list[str]) -> tuple[list[str], bool]:
        """Return (concrete actions, exact). exact=False when a pattern could not be expanded."""
        out, exact = set(), True
        for p in patterns:
            if "*" not in p and "?" not in p:
                out.add(p)
                continue
            if self._reg_lower is None:
                out.add(p)
                exact = False
                continue
            pl = p.lower()
            hits = [self.registry[i] for i, a in enumerate(self._reg_lower) if fnmatch.fnmatchcase(a, pl)]
            if not hits or len(hits) > self.max_expand:
                out.add(p)
                exact = False
            else:
                out.update(hits)
        return sorted(out), exact

    def parse_document(self, data: dict, name: str, prefix: str = "") -> list[Rule]:
        rules = []
        for si, st in enumerate(_as_list(data.get("Statement", []))):
            effect = Effect.PERMIT if st.get("Effect") == "Allow" else Effect.DENY
            atoms = []
            meta = {"source": "aws-iam", "policy": name}
            for key, attr, neg in (("Action", "action", False), ("NotAction", "action", True),
                                   ("Resource", "resource", False), ("NotResource", "resource", True)):
                if key not in st:
                    continue
                vals = [str(v) for v in _as_list(st[key])]
                if "*" in vals and not neg:
                    continue                         # unconstrained
                if attr == "action":
                    vals, exact = self.expand_actions(vals)
                    if not exact:
                        meta["pattern_actions"] = True
                if "*" in vals and neg:
                    atoms.append(Atom(attr, "in", ()))   # NotAction: * → applies to nothing
                    continue
                atoms.append(Atom(attr, "notin" if neg else "in", tuple(vals)))
            cond_atoms = []
            for op, kv in (st.get("Condition") or {}).items():
                base = op.replace("IfExists", "").replace("ForAnyValue:", "").replace("ForAllValues:", "")
                if base not in COND_OPS:
                    meta["approx"] = True            # unsupported condition operator → over-approximated
                    continue
                for key, val in kv.items():
                    vals = _as_list(val)
                    if COND_OPS[base] == "eq" and len(vals) > 1:
                        cond_atoms.append(Atom(key, "in", tuple(vals)))
                    elif COND_OPS[base] == "neq" and len(vals) > 1:
                        cond_atoms.append(Atom(key, "notin", tuple(vals)))
                    else:
                        cond_atoms.append(Atom(key, COND_OPS[base], vals[0] if len(vals) == 1 else tuple(vals)))
            sid = st.get("Sid") or f"stmt{si}"
            rules.append(Rule(f"{prefix}{sid}", Predicate(tuple(atoms)), Predicate(tuple(cond_atoms)), effect, meta))
        return rules

    def parse(self, path: str | Path) -> Policy:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "PolicyVersion" in data:
            data = data["PolicyVersion"].get("Document", data["PolicyVersion"])
        pol = Policy(name=Path(path).stem, combining=CombiningAlgorithm.DENY_OVERRIDES)
        for r in self.parse_document(data, Path(path).stem):
            pol.add(r)
        return pol

    def parse_many(self, paths: list[str | Path], name: str = "aws-merged") -> Policy:
        """Merge several managed policies into one Policy (ids prefixed by policy name)."""
        pol = Policy(name=name, combining=CombiningAlgorithm.DENY_OVERRIDES)
        for p in paths:
            p = Path(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            if "PolicyVersion" in data:
                data = data["PolicyVersion"].get("Document", data["PolicyVersion"])
            for r in self.parse_document(data, p.stem, prefix=f"{p.stem}/"):
                if r.id in pol:
                    continue
                pol.add(r)
        return pol
