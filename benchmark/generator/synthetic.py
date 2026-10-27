"""Synthetic evolving-policy generator (C4).

Parameters
----------
n_rules          : 1k … 100k
n_attrs          : size of the attribute space (mix of categorical / numeric)
domain_size      : values per categorical attribute
atoms_per_rule   : how constrained each rule is (fewer atoms ⇒ more overlap)
core_attrs       : attributes constrained in EVERY rule (resourceType, action),
                   as in real policies — keeps random overlap realistic
conflict_rate    : fraction of rules that get a *planted* conflicting partner
hetero_rate      : fraction of rules written in the *second vocabulary*
                   (role→jobTitle, department→orgUnit, resourceType→assetClass),
                   i.e. the policy is a merge of two sources — the case where
                   exact literal indexes fail (§3).  The ontology (alias map)
                   is written into the dataset; every exact step uses it.
hierarchy_rate   : fraction of `role` atoms that use a *specialised* role
                   (`role_3_sr` ⊑ `role_3`): overlap only visible through the
                   value hierarchy.
complex_rate     : fraction of rules carrying a raw SMT condition
n_deltas         : length of the Δ sequence (add / modify / remove mix)

Ground truth: planted pairs are guaranteed conflicts; the full-SMT baseline
(experiments/baselines/full_smt.py) computes the complete truth set once and
caches it next to the dataset so recall can be measured against *all* conflicts,
not only planted ones.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from ecd.core import (AliasSpec, Atom, Attribute, AttrType, Delta, DeltaKind, Effect, Policy,
                      Predicate, Rule)

ALIASES = {"role": "jobTitle", "department": "orgUnit", "resourceType": "assetClass", "action": "operation"}
HIER_ATTR = "role"


@dataclass
class GenConfig:
    n_rules: int = 1000
    n_attrs: int = 12
    n_numeric: int = 3
    domain_size: int = 12
    atoms_per_rule: int = 4
    core_attrs: int = 2          # attributes present in EVERY rule (resourceType, action)
    conflict_rate: float = 0.05
    hetero_rate: float = 0.2
    hierarchy_rate: float = 0.0
    complex_rate: float = 0.05
    n_deltas: int = 100
    delta_mix: tuple[float, float, float] = (0.5, 0.3, 0.2)  # add / modify / remove
    seed: int = 0


class SyntheticGenerator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.space, self.alias_specs, self.hierarchy = self._make_space()
        self.by_name = {a.name: a for a in self.space}
        self.cat_attrs = [a for a in self.space if a.type is AttrType.CATEGORICAL and a.name not in ALIASES.values()]
        self.num_attrs = [a for a in self.space if a.type is AttrType.NUMERIC]
        self.planted: list[tuple[str, str]] = []
        self.planted_kind: dict[str, str] = {}       # partner id → "plain" | "alias" | "hierarchy"

    # ----- attribute space -------------------------------------------------
    def _make_space(self):
        attrs: list[Attribute] = []
        names = ["role", "department", "resourceType", "action", "location", "clearance",
                 "shift", "project", "device", "env", "team", "region", "tenant", "zone"]
        n_cat = self.cfg.n_attrs - self.cfg.n_numeric
        hierarchy: dict[str, dict] = {}
        for i in range(n_cat):
            name = names[i] if i < len(names) else f"cat{i}"
            dom = [f"{name}_{k}" for k in range(self.cfg.domain_size)]
            if name == HIER_ATTR and self.cfg.hierarchy_rate > 0:
                # every second base value gets a specialised child value (role_k_sr ⊑ role_k)
                h = {f"{name}_{k}_sr": f"{name}_{k}" for k in range(0, self.cfg.domain_size, 2)}
                hierarchy[name] = h
                dom += list(h.keys())
            attrs.append(Attribute(name, AttrType.CATEGORICAL, tuple(dom)))
        for j in range(self.cfg.n_numeric):
            attrs.append(Attribute(["hour", "age", "riskScore", "amount"][j] if j < 4 else f"num{j}",
                                   AttrType.NUMERIC, lo=0, hi=100))
        # alias attributes mirror the domain of their primary (heterogeneous sources)
        specs: dict[str, AliasSpec] = {}
        if self.cfg.hetero_rate > 0:
            for prim, alias in ALIASES.items():
                base = next((a for a in attrs if a.name == prim), None)
                if base:
                    vmap = {v.replace(prim, alias, 1): v for v in base.domain}
                    attrs.append(Attribute(alias, AttrType.CATEGORICAL, tuple(vmap.keys())))
                    specs[alias] = AliasSpec(prim, vmap)
        return attrs, specs, hierarchy

    def _install_ontology(self, pol: Policy) -> None:
        for a in self.space:
            pol.space.add(a)
        pol.space.ontology.aliases = dict(self.alias_specs)
        pol.space.ontology.hierarchy = {k: dict(v) for k, v in self.hierarchy.items()}

    # ----- vocabularies ----------------------------------------------------
    def _to_alias(self, a: Atom) -> Atom:
        """Rewrite a canonical categorical atom into the second vocabulary (if it has one)."""
        alias = ALIASES.get(a.attr)
        if alias is None or alias not in self.alias_specs:
            return a
        inv = {v: k for k, v in self.alias_specs[alias].values.items()}
        if a.op in ("eq", "neq"):
            return Atom(alias, a.op, inv.get(a.value, a.value))
        if a.op in ("in", "notin"):
            return Atom(alias, a.op, tuple(inv.get(v, v) for v in a.value))
        return a

    def _vocab(self, atoms: list[Atom], use_alias: bool) -> list[Atom]:
        return [self._to_alias(a) for a in atoms] if use_alias else atoms

    # ----- rules -----------------------------------------------------------
    def _random_atom(self, attr: Attribute) -> Atom:
        r = self.rng
        if attr.type is AttrType.CATEGORICAL:
            base_vals = [v for v in attr.domain if not v.endswith("_sr")]
            k = r.choice([1, 1, 2, 3])
            vals = r.sample(base_vals, min(k, len(base_vals)))
            if attr.name in self.hierarchy and r.random() < self.cfg.hierarchy_rate:
                # use a specialised value where one exists
                children = {p: c for c, p in self.hierarchy[attr.name].items()}
                vals = [children.get(v, v) for v in vals]
            return Atom(attr.name, "eq", vals[0]) if k == 1 else Atom(attr.name, "in", tuple(vals))
        lo = r.randint(0, 80)
        op = r.choice(["range", "range", "ge", "lt"])
        if op == "range":
            return Atom(attr.name, "range", (lo, lo + r.randint(5, 20)))
        return Atom(attr.name, op, lo)

    def _random_rule(self, rid: str) -> Rule:
        r = self.rng
        pool = [a for a in self.space if a.name not in ALIASES.values()]
        core = [a for a in pool if a.name in ("resourceType", "action")][: self.cfg.core_attrs]
        rest = [a for a in pool if a not in core]
        chosen = core + r.sample(rest, max(0, self.cfg.atoms_per_rule - len(core)))
        atoms = [self._random_atom(a) for a in chosen]
        use_alias = r.random() < self.cfg.hetero_rate
        atoms = self._vocab(atoms, use_alias)
        split = max(1, len(atoms) // 2)
        raw = None
        if self.num_attrs and r.random() < self.cfg.complex_rate:
            a, b = r.sample(self.num_attrs, min(2, len(self.num_attrs))) if len(self.num_attrs) > 1 else (self.num_attrs[0],) * 2
            raw = f"(> (+ {a.name} {b.name}) {r.randint(50, 150)})"
        return Rule(rid, Predicate(tuple(atoms[:split])), Predicate(tuple(atoms[split:]), raw),
                    r.choice([Effect.PERMIT, Effect.DENY]),
                    {"vocab": "B" if use_alias else "A"})

    def _canonical_atoms(self, rule: Rule) -> list[Atom]:
        ont_free = []
        for a in rule.applicability.atoms:
            spec = self.alias_specs.get(a.attr)
            if spec is None:
                ont_free.append(a)
            elif a.op in ("eq", "neq"):
                ont_free.append(Atom(spec.canonical, a.op, spec.map_value(a.value)))
            elif a.op in ("in", "notin"):
                ont_free.append(Atom(spec.canonical, a.op, tuple(spec.map_value(v) for v in a.value)))
            else:
                ont_free.append(Atom(spec.canonical, a.op, a.value))
        return ont_free

    def _conflicting_partner(self, base: Rule, rid: str) -> Rule:
        """Build r' that provably intersects `base` with opposite effect."""
        r = self.rng
        atoms: list[Atom] = []
        kind = "plain"
        for a in self._canonical_atoms(base):
            if a.op in ("eq", "in"):
                v = a.value if a.op == "eq" else r.choice(a.value)
                if (a.attr in self.hierarchy and r.random() < self.cfg.hierarchy_rate):
                    h = self.hierarchy[a.attr]
                    if v in h:                       # base uses a child → partner uses the parent (still overlaps)
                        v, kind = h[v], "hierarchy"
                    else:
                        children = [c for c, p in h.items() if p == v]
                        if children:                 # base uses the parent → partner uses a child
                            v, kind = r.choice(children), "hierarchy"
                atoms.append(Atom(a.attr, "eq", v))
            elif a.op == "range":
                lo, hi = a.value
                atoms.append(Atom(a.attr, "range", (lo, hi + r.randint(0, 10))))
            elif a.op in ("ge", "lt"):
                atoms.append(Atom(a.attr, a.op, a.value))
        # drop one atom so the partner is a superset region (still overlaps)
        if len(atoms) > 1 and r.random() < 0.5:
            atoms.pop(r.randrange(len(atoms)))
        use_alias = r.random() < self.cfg.hetero_rate
        if use_alias:
            alias_atoms = self._vocab(atoms, True)
            if any(x.attr != y.attr for x, y in zip(alias_atoms, atoms)):
                kind = "alias" if kind == "plain" else kind
            atoms = alias_atoms
        eff = Effect.DENY if base.effect is Effect.PERMIT else Effect.PERMIT
        self.planted_kind[rid] = kind
        return Rule(rid, Predicate(tuple(atoms[:1])), Predicate(tuple(atoms[1:])), eff,
                    {"planted_with": base.id, "vocab": "B" if use_alias else "A", "planted_kind": kind})

    def _is_satisfiable(self, rule: Rule) -> bool:
        from ecd.verification.smt_checker import SMTChecker
        if not hasattr(self, "_sat_checker"):
            tmp = Policy("tmp")
            self._install_ontology(tmp)
            self._sat_checker = SMTChecker(tmp.space)
        return self._sat_checker.check_pair(rule, rule).overlap

    def generate(self) -> Policy:
        pol = Policy("synthetic")
        self._install_ontology(pol)
        n_plant = int(self.cfg.n_rules * self.cfg.conflict_rate)
        n_base = self.cfg.n_rules - n_plant
        rules = [self._random_rule(f"r{i}") for i in range(n_base)]
        for j in range(n_plant):
            base = self.rng.choice(rules[:n_base])
            while not self._is_satisfiable(base):   # a vacuous base cannot host a conflict
                base = self.rng.choice(rules[:n_base])
            partner = self._conflicting_partner(base, f"r{n_base + j}")
            rules.append(partner)
            self.planted.append((base.id, partner.id))
        self.rng.shuffle(rules)
        for r in rules:
            pol.add(r)
        return pol

    # ----- delta sequence ----------------------------------------------------
    def deltas(self, pol: Policy) -> list[Delta]:
        out = []
        ids = list(pol.order)
        live: dict[str, Rule] = dict(pol.rules)   # track the evolving rule set without mutating pol
        next_id = len(pol)
        pa, pm, pr = self.cfg.delta_mix
        for _ in range(self.cfg.n_deltas):
            x = self.rng.random()
            if x < pa or len(ids) < 2:
                rid = f"r{next_id}"
                next_id += 1
                if self.rng.random() < self.cfg.conflict_rate * 4:  # Δs are conflict-prone
                    base = live[self.rng.choice(ids)]
                    rule = self._conflicting_partner(base, rid)
                else:
                    rule = self._random_rule(rid)
                out.append(Delta(DeltaKind.ADD, rule))
                ids.append(rid)
                live[rid] = rule
            elif x < pa + pm:
                rid = self.rng.choice(ids)
                rule = self._random_rule(rid)
                out.append(Delta(DeltaKind.MODIFY, rule, rid))
                live[rid] = rule
            else:
                rid = self.rng.choice(ids)
                ids.remove(rid)
                del live[rid]
                out.append(Delta(DeltaKind.REMOVE, rule_id=rid))
        return out


def write_dataset(cfg: GenConfig, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen = SyntheticGenerator(cfg)
    pol = gen.generate()
    pol.save(out / "policy_v0.json")
    deltas = gen.deltas(pol)
    (out / "deltas.json").write_text(json.dumps([d.to_dict() for d in deltas], indent=1))
    (out / "planted_conflicts.json").write_text(json.dumps(gen.planted))
    (out / "planted_kinds.json").write_text(json.dumps(gen.planted_kind))
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    for stale in ("truth_conflicts.json", "labels.json"):
        if (out / stale).exists():
            (out / stale).unlink()                 # derived files must be recomputed
    return out


if __name__ == "__main__":  # quick manual run
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="data/processed/synthetic_1k")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(write_dataset(GenConfig(n_rules=a.n, seed=a.seed), a.out))
