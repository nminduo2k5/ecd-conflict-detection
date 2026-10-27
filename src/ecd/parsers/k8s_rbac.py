"""Kubernetes RBAC (Role/ClusterRole YAML, multi-document or a `kind: List`) → Policy.

RBAC is allow-only; conflicts in the Permit/Deny sense do not occur. We
still normalise it because (a) rules overlap heavily (redundancy), and
(b) when merged with NetworkPolicy / OPA deny rules it yields a realistic
heterogeneous attribute space (role vs. jobTitle, see paper §3).
"""
from __future__ import annotations

from pathlib import Path

from ecd.core import Atom, CombiningAlgorithm, Effect, Policy, Predicate, Rule
from .base import PolicyParser


class K8sRBACParser(PolicyParser):
    def parse(self, path: str | Path) -> Policy:
        import yaml
        docs = [d for d in yaml.safe_load_all(Path(path).read_text(encoding="utf-8")) if d]
        pol = Policy(name=Path(path).stem, combining=CombiningAlgorithm.PERMIT_OVERRIDES)
        n = 0
        flat = []
        for d in docs:
            if d.get("kind") == "List":
                flat += [x for x in d.get("items", []) if x]
            else:
                flat.append(d)
        for d in flat:
            if d.get("kind") not in ("Role", "ClusterRole"):
                continue
            ns = d.get("metadata", {}).get("namespace")
            rname = d.get("metadata", {}).get("name", f"role{n}")
            for ri, rule in enumerate(d.get("rules") or []):
                atoms = []
                for key, attr in (("apiGroups", "apiGroup"), ("resources", "resource"), ("verbs", "verb")):
                    vals = rule.get(key) or []
                    vals = [("core" if v == "" else v) for v in vals]
                    if vals and "*" not in vals:
                        atoms.append(Atom(attr, "in", tuple(vals)))
                if rule.get("resourceNames"):
                    atoms.append(Atom("resourceName", "in", tuple(rule["resourceNames"])))
                if rule.get("nonResourceURLs"):
                    atoms.append(Atom("nonResourceURL", "in", tuple(rule["nonResourceURLs"])))
                if ns:
                    atoms.append(Atom("namespace", "eq", ns))
                pol.add(Rule(f"{rname}#{ri}", Predicate(tuple(atoms)), Predicate(), Effect.PERMIT,
                             {"source": "k8s-rbac", "kind": d["kind"]}))
                n += 1
        return pol
