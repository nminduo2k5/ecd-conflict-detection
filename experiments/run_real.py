"""Real / near-real policies: full check + Δ replay on every dataset under data/processed/real.

Reports per dataset: rules (after DNF splitting), pairs, Tier A pruning, pairs verified,
conflicts by combining-algorithm class, approximate-mode recall (top-k, hash embedding),
and the mean incremental cost of 50 Δs derived from the policy's own rules.
"""
from __future__ import annotations

import json
from collections import Counter

from common import ROOT, load_cfg, load_dataset, ensure_truth, save_result
from ecd.embedding import HashedAtomEmbedder
from ecd.filtering import Mode
from ecd.pipeline import Detector
from metrics import precision_recall, mean_std


def main():
    cfg = load_cfg("real")
    results = {}
    for ds in cfg["datasets"]:
        if not (ROOT / ds / "policy_v0.json").exists():
            print(f"skip {ds}")
            continue
        pol, deltas, _, truth = load_dataset(ds)
        info = json.loads((ROOT / ds / "info.json").read_text()) if (ROOT / ds / "info.json").exists() else {}
        truth = ensure_truth(pol, ds, truth)
        det = Detector(pol.copy(), HashedAtomEmbedder(), Mode.EXACT, use_faiss=False)
        rep = det.full_check()
        found = {(c.a, c.b) for c in rep.conflicts}
        p, r = precision_recall(found, truth)
        classes = Counter(c.verdict.cls.value for c in rep.conflicts)
        approx = Detector(pol.copy(), HashedAtomEmbedder(), Mode.APPROXIMATE, top_k=cfg["top_k"], use_faiss=False).full_check()
        pa, ra = precision_recall({(c.a, c.b) for c in approx.conflicts}, truth)
        n_approx_rules = sum(1 for x in pol if x.meta.get("approx"))
        res = {"rules": len(pol), "source_rules": info.get("xacml_rules") or info.get("policies"),
               "attributes": len(pol.space.attributes), "combining": pol.combining.value,
               "rules_with_overapprox_condition": n_approx_rules,
               "pairs_total": rep.pairs_total, "pruning_ratio": rep.pruning_ratio,
               "verified": rep.pairs_verified, "conflicts": len(found), "recall": r, "precision": p,
               "conflict_classes": dict(classes),
               "full_check_ms": rep.time_ms.get("select+verify"),
               f"approx_k{cfg['top_k']}": {"verified": approx.pairs_verified, "recall": ra},
               "examples": [c.to_dict() for c in rep.conflicts[:3]]}
        # Δ replay (incremental path) on the real rules
        rows = []
        for d in deltas[: cfg["n_deltas"]]:
            rr = det.apply(d)
            rows.append({"kind": d.kind.value, "ms": rr.time_ms.get("total"), "verified": rr.pairs_verified,
                         "conflicts": len(rr.conflicts)})
        res["delta_ms"] = mean_std([x["ms"] for x in rows])
        res["delta_verified"] = mean_std([x["verified"] for x in rows])
        res["delta_conflicts_introduced"] = sum(x["conflicts"] for x in rows)
        results[ds] = res
        print(ds, json.dumps({k: v for k, v in res.items() if k != "examples"}, default=float))
        save_result("real_policies", results)


if __name__ == "__main__":
    main()
