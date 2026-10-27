"""Convert the real / near-real policy sources under data/raw into native datasets.

    python benchmark/convert_real.py            # writes data/processed/real/<name>/policy_v0.json

Sources (see scripts/download_datasets.sh)
  * XACML benchmark policies from the XPA repository (iTrust, HL7, gpms, pluto,
    conference, fedora, kmarket, obligation, Simpson): one dataset per policy.
  * AWS managed policies (SummitRoute mirror) merged into one policy; wildcard
    actions expanded with the MAMIP action registry.
  * Kubernetes bootstrap ClusterRoles (kubernetes/kubernetes testdata).

Each dataset also gets a synthetic-but-realistic Δ sequence derived from its own
rules (`deltas.json`): ADD clones an existing rule with the opposite effect
(conflict-prone), MODIFY drops/changes an atom, REMOVE deletes a rule.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecd.core import Atom, Delta, DeltaKind, Effect, Policy, Predicate, Rule  # noqa: E402
from ecd.parsers.aws_iam import AWSIAMParser, load_action_registry  # noqa: E402
from ecd.parsers.k8s_rbac import K8sRBACParser  # noqa: E402
from ecd.parsers.xacml import XACMLParser  # noqa: E402

RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed/real"

XACML_FILES = {
    "itrust": "xacml/xpa/Experiments/itrust3/itrust3.xml",
    "hl7": "xacml/xpa/Experiments/HL7/HL7.xml",
    "gpms": "xacml/xpa/Experiments/gpms/gpms.xml",
    "pluto": "xacml/xpa/Experiments/pluto3/pluto3.xml",
    "conference": "xacml/xpa/Experiments/conference3/conference3.xml",
    "fedora": "xacml/xpa/Experiments/fedora-rule3/fedora-rule3.xml",
    "kmarket": "xacml/xpa/Experiments/kmarket-gold-policy/kmarket-gold-policy.xml",
    "obligation": "xacml/xpa/Experiments/obligation3/obligation3.xml",
    "simpson": "xacml/xpa/Experiments/Simpson1/Simpson1.xml",
}


def real_deltas(pol: Policy, n: int, seed: int = 0) -> list[Delta]:
    rng = random.Random(seed)
    ids = list(pol.order)
    live = dict(pol.rules)
    out, k = [], 0
    for _ in range(n):
        x = rng.random()
        if x < 0.5 or len(ids) < 3:
            base = live[rng.choice(ids)]
            atoms = list(base.applicability.atoms)
            if len(atoms) > 1 and rng.random() < 0.5:
                atoms.pop(rng.randrange(len(atoms)))         # superset region
            eff = Effect.DENY if base.effect is Effect.PERMIT else Effect.PERMIT
            rid = f"new{k}"
            k += 1
            r = Rule(rid, Predicate(tuple(atoms[:1])), Predicate(tuple(atoms[1:]), base.applicability.raw_expr),
                     eff, {"derived_from": base.id})
            out.append(Delta(DeltaKind.ADD, r))
            ids.append(rid)
            live[rid] = r
        elif x < 0.8:
            rid = rng.choice(ids)
            base = live[rid]
            atoms = list(base.applicability.atoms)
            if atoms and rng.random() < 0.5:
                atoms.pop(rng.randrange(len(atoms)))
                r = Rule(rid, Predicate(tuple(atoms[:1])), Predicate(tuple(atoms[1:]), base.applicability.raw_expr),
                         base.effect, dict(base.meta))
            else:
                eff = Effect.DENY if base.effect is Effect.PERMIT else Effect.PERMIT
                r = Rule(rid, base.target, base.condition, eff, dict(base.meta))
            out.append(Delta(DeltaKind.MODIFY, r, rid))
            live[rid] = r
        else:
            rid = rng.choice(ids)
            ids.remove(rid)
            del live[rid]
            out.append(Delta(DeltaKind.REMOVE, rule_id=rid))
    return out


def write(pol: Policy, name: str, extra: dict | None = None, n_deltas: int = 50) -> dict:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    pol.save(d / "policy_v0.json")
    deltas = real_deltas(pol, n_deltas)
    (d / "deltas.json").write_text(json.dumps([x.to_dict() for x in deltas], indent=1))
    (d / "planted_conflicts.json").write_text("[]")
    info = {"name": name, "rules": len(pol), "attributes": len(pol.space.attributes),
            "combining": pol.combining.value, **(extra or {})}
    (d / "info.json").write_text(json.dumps(info, indent=2))
    for stale in ("truth_conflicts.json",):
        if (d / stale).exists():
            (d / stale).unlink()
    print(json.dumps(info))
    return info


def main():
    infos = []
    for name, rel in XACML_FILES.items():
        p = RAW / rel
        if not p.exists():
            print(f"skip {name}: {p} missing")
            continue
        parser = XACMLParser()
        pol = parser.parse(p)
        st = dict(parser.stats)
        st["untranslated"] = sorted(set(st["untranslated"]))
        infos.append(write(pol, f"xacml_{name}", {"xacml_rules": st["rules"], "sub_rules": st["sub_rules"],
                                                  "approx_rules": st["approx_rules"],
                                                  "untranslated": st["untranslated"]}))

    reg_path = RAW / "aws_iam/mamip/data/action-registry.json"
    registry = load_action_registry(reg_path) if reg_path.exists() else None
    pol_dir = RAW / "summitroute/policies"
    if not pol_dir.exists():
        pol_dir = RAW / "aws_iam/summitroute/policies"
    if pol_dir.exists():
        files = sorted(pol_dir.glob("*"))
        rng = random.Random(0)
        rng.shuffle(files)
        for n_pol in (100, 300):
            parser = AWSIAMParser(registry)
            pol = parser.parse_many(files[:n_pol], name=f"aws_managed_{n_pol}")
            n_pat = sum(1 for r in pol if r.meta.get("pattern_actions"))
            infos.append(write(pol, f"aws_managed_{n_pol}", {"policies": n_pol, "registry_actions": len(registry or []),
                                                             "rules_with_unexpanded_patterns": n_pat}))
    k8s = RAW / "k8s_rbac/cluster-roles.yaml"
    if k8s.exists():
        pol = K8sRBACParser().parse(k8s)
        infos.append(write(pol, "k8s_clusterroles"))
    (OUT / "index.json").write_text(json.dumps(infos, indent=2))


if __name__ == "__main__":
    main()
