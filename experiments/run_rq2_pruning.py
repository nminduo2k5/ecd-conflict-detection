"""RQ2 — Pair reduction while keeping recall = 100 % (exact mode), empirical recall in
approximate mode, and *earlier* detection (SMT calls / time to first conflict).

Part A (full check of v0, ablation "with / without Tier A", "with / without embedding"):
    no_tierA        : every opposite-effect pair goes to Z3 (= full-SMT baseline)
    tierA_only      : Tier A, survivors in arbitrary order
    exact_<emb>     : Tier A + embedding order, all survivors verified  (recall 100 %)
    approx_<emb>_k  : Tier A + embedding order, top-k survivors only
    exact_index     : attribute-value inverted index (literal), candidates + recall
    exact_index+ont : same index over the canonical vocabulary (hand-coded ontology)

Part B (Δ path, where ordering matters): replay the Δ sequence; for every Δ that
introduces ≥ 1 conflict record how many SMT calls the exact mode needed before the
first conflict appeared, with and without the embedding.  Paired Wilcoxon + A12.
"""
from __future__ import annotations

import json

import numpy as np

from common import ROOT, load_cfg, load_dataset, ensure_truth, model_path, save_result
from baselines.exact_index import candidate_pairs
from ecd.embedding import HashedAtomEmbedder
from ecd.filtering import Mode
from ecd.pipeline import Detector
from metrics import precision_recall, recall_at_budget, mean_std
from stats import a12, wilcoxon


def embedders(pol, dataset):
    out = {"hash": lambda: HashedAtomEmbedder()}
    if not pol.space.ontology.is_empty():
        out["hash+ont"] = lambda: HashedAtomEmbedder(use_ontology=True)
    mp = model_path(dataset)
    if mp.exists():
        from ecd.embedding.gnn import GNNEmbedder
        out["gnn"] = lambda: GNNEmbedder.load(mp, pol.space)
    return out


def full_variant(name, pol, emb, mode, k, truth, use_tier_a=True):
    det = Detector(pol.copy(), emb, mode, top_k=k, use_tier_a=use_tier_a, use_faiss=False)
    rep = det.full_check()
    found = {(c.a, c.b) for c in rep.conflicts}
    p, r = precision_recall(found, truth)
    row = {"pairs_total": rep.pairs_total, "pruning_ratio": rep.pruning_ratio,
           "verified": rep.pairs_verified, "precision": p, "recall": r,
           "n_conflicts": len(found),
           "ttfc_ms": rep.time_to_first_conflict_ms,
           "smt_calls_to_first_conflict": rep.smt_calls_to_first_conflict,
           "recall@1s": recall_at_budget(rep.conflicts_over_time, len(truth), 1000),
           "recall@5s": recall_at_budget(rep.conflicts_over_time, len(truth), 5000),
           "time_ms": rep.time_ms}
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "time_ms"}),
          f"{rep.time_ms.get('select+verify', 0) / 1000:.1f}s")
    return row


def delta_path(pol, deltas, emb_factory, n):
    """Replay Δs in exact mode; return per-Δ (smt calls to first conflict, verified, conflicts)."""
    det = Detector(pol.copy(), emb_factory() if emb_factory else None, Mode.EXACT, use_faiss=False)
    rows = []
    for d in deltas[:n]:
        rep = det.apply(d)
        rows.append({"kind": d.kind.value, "calls_to_first": rep.smt_calls_to_first_conflict,
                     "verified": rep.pairs_verified, "conflicts": len(rep.conflicts),
                     "ms": rep.time_ms.get("total")})
    return rows


def main():
    import sys
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "rq2_pruning"     # e.g. rq2_complex
    cfg = load_cfg(cfg_name)
    ds = cfg["dataset"]
    pol, deltas, planted, truth = load_dataset(ds)
    truth = ensure_truth(pol, ds, truth)
    embs = embedders(pol, ds)
    out = {"dataset": ds, "n_rules": len(pol), "n_truth": len(truth), "embedders": list(embs)}

    # ---- Part A: full check variants ---------------------------------------
    out["no_tierA"] = full_variant("no_tierA", pol, None, Mode.EXACT, 0, truth, use_tier_a=False)
    out["tierA_only"] = full_variant("tierA_only", pol, None, Mode.EXACT, 0, truth)
    for name, mk in embs.items():
        out[f"exact_{name}"] = full_variant(f"exact_{name}", pol, mk(), Mode.EXACT, 0, truth)
        for k in cfg["top_ks"]:
            out[f"approx_{name}_k{k}"] = full_variant(f"approx_{name}_k{k}", pol, mk(), Mode.APPROXIMATE, k, truth)
    idx = candidate_pairs(pol)
    out["exact_index"] = {"candidates": len(idx), "recall_on_candidates": len(idx & truth) / len(truth) if truth else 1.0}
    if not pol.space.ontology.is_empty():
        idx2 = candidate_pairs(pol, pol.space.ontology)
        out["exact_index+ont"] = {"candidates": len(idx2), "recall_on_candidates": len(idx2 & truth) / len(truth) if truth else 1.0}
    print("exact_index", out["exact_index"], out.get("exact_index+ont"))
    save_result(cfg_name, out)

    # ---- Part B: Δ path, calls to first conflict ---------------------------
    n = cfg["n_deltas"]
    paths = {"none": delta_path(pol, deltas, None, n)}
    for name, mk in embs.items():
        paths[name] = delta_path(pol, deltas, mk, n)
    out["delta_path"] = {}
    idx_conf = [i for i, r in enumerate(paths["none"]) if r["conflicts"] > 0]
    base = np.array([paths["none"][i]["calls_to_first"] for i in idx_conf], dtype=float)
    for name_, rows in paths.items():
        calls = np.array([rows[i]["calls_to_first"] for i in idx_conf], dtype=float)
        rec = {"n_deltas_with_conflict": len(idx_conf),
               "calls_to_first_conflict": mean_std(calls),
               "verified_per_delta": mean_std([r["verified"] for r in rows]),
               "ms_per_delta": mean_std([r["ms"] for r in rows])}
        if name_ != "none" and len(idx_conf) >= 5:
            try:
                rec["wilcoxon_p_vs_none"] = float(wilcoxon(calls, base).pvalue)
            except ValueError:
                rec["wilcoxon_p_vs_none"] = None
            rec["A12_vs_none"] = a12(calls, base)      # < 0.5 ⇒ fewer calls than without embedding
        out["delta_path"][name_] = rec
        print("delta_path", name_, json.dumps(rec, default=float))
    save_result(cfg_name, out)


if __name__ == "__main__":
    main()
