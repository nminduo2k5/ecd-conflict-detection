"""RQ3 — Incremental Event-B: proof obligations and time per Δ vs full re-verification.

Per Δ we record, from the Event-B ledger (one THM PO per pair theorem):
    po_new             POs generated in PolicyCtx_v{N}   (pairs in Impact(Δ) that survived Tier A)
    po_inherited       POs of all ancestor contexts that stay valid (Theorem 2)
    po_reuse_rate      po_inherited / (po_inherited + po_new)
    po_full_regen      POs a non-incremental scheme regenerates: every opposite-effect pair
                       of the whole policy (no Tier A, no refinement)
    po_full_tierA      same but with Tier A (what "re-run ecd from scratch" would prove)
    incremental_ms     wall-clock of the Δ path (impact + Tier A/B + Z3 + Event-B emission)
and every `full_every` Δs a genuine from-scratch re-verification is timed for comparison.
"""
from __future__ import annotations

import itertools
import json
import time

from common import load_cfg, load_dataset, model_path, results_path, save_result
from ecd.embedding import HashedAtomEmbedder
from ecd.filtering import Mode, Pruner
from ecd.pipeline import Detector
from metrics import mean_std


def po_full_counts(policy) -> tuple[int, int]:
    rules = list(policy)
    pr = Pruner(policy.space.ontology)
    regen = surv = 0
    for a, b in itertools.combinations(rules, 2):
        if a.effect == b.effect:
            continue
        regen += 1
        if not pr.decide(a, b).pruned:
            surv += 1
    return regen, surv


def main():
    cfg = load_cfg("rq3_incremental")
    ds = cfg["dataset"]
    pol, deltas, _, _ = load_dataset(ds)
    mp = model_path(ds)
    if mp.exists():
        from ecd.embedding.gnn import GNNEmbedder
        emb = GNNEmbedder.load(mp, pol.space)
    else:
        emb = HashedAtomEmbedder()
    out_dir = results_path(cfg["eventb_out"])        # results/eventb/rq3 → <ECD_RESULTS_DIR>/eventb/rq3
    for f in out_dir.glob("PolicyCtx_v*.*") if out_dir.exists() else []:
        f.unlink()
    n_rules_v0 = len(pol)                       # before the Δ replay mutates the policy
    det = Detector(pol, emb, Mode.EXACT, eventb_out=out_dir, use_faiss=False)
    t0 = time.perf_counter()
    rep0 = det.full_check()
    full_ms_v0 = (time.perf_counter() - t0) * 1000
    print(f"v0: {len(rep0.conflicts)} conflicts, {rep0.pairs_verified} POs, {full_ms_v0 / 1000:.1f}s")

    rows = []
    for i, d in enumerate(deltas[: cfg["n_deltas"]], 1):
        rep = det.apply(d)
        st = det.eventb.po_stats()[-1]
        regen, surv = po_full_counts(det.policy)
        row = {"delta": i, "kind": d.kind.value, "version": det.policy.version,
               "po_new": st["po_new"], "po_inherited": st["po_inherited"], "po_reuse_rate": st["po_reuse_rate"],
               "po_full_regen": regen, "po_full_tierA": surv,
               "incremental_ms": rep.time_ms.get("total"), "eventb_gen_ms": rep.time_ms.get("eventb_gen"),
               "conflicts_introduced": len(rep.conflicts), "live_conflicts": len(det.all_conflicts())}
        if cfg.get("full_every") and i % cfg["full_every"] == 0:
            t = time.perf_counter()
            Detector(det.policy.copy(), HashedAtomEmbedder(), Mode.EXACT, use_faiss=False).full_check()
            row["full_reverify_ms"] = (time.perf_counter() - t) * 1000
        rows.append(row)
        print(json.dumps(row))
    summary = {"dataset": ds, "n_rules_v0": n_rules_v0, "full_check_ms_v0": full_ms_v0,
               "po_v0": rep0.pairs_verified,
               "incremental_ms": mean_std([r["incremental_ms"] for r in rows]),
               "po_new": mean_std([r["po_new"] for r in rows]),
               "po_reuse_rate": mean_std([r["po_reuse_rate"] for r in rows]),
               "po_full_regen": mean_std([r["po_full_regen"] for r in rows]),
               "full_reverify_ms": mean_std([r.get("full_reverify_ms") for r in rows if "full_reverify_ms" in r]),
               "rodin_project": str(out_dir)}
    print(json.dumps(summary, indent=1))
    save_result("rq3_incremental", {"summary": summary, "rows": rows})


if __name__ == "__main__":
    main()
