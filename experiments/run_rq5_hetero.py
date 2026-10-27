"""RQ5 — Where the embedding beats an exact index: heterogeneous vocabularies,
role hierarchies and complex conditions (paper §3, "dedicated experiment").

Datasets: hetero_1k_h00 / h25 / h50 (0 %, 25 %, 50 % of rules written in the
second vocabulary; all with a role hierarchy and raw conditions).  Ground truth
is the ontology-aware exhaustive Z3 check.  Candidate selectors compared at a
fixed verification budget (top-k per rule):

    exact_index        literal (attr, value) index                       — no ontology
    exact_index+ont    index over the canonical vocabulary               — hand-coded ontology
    approx_hash_k      Tier A + hash embedding, top-k                    — no ontology
    approx_hash+ont_k  Tier A + hash over canonical tokens, top-k        — hand-coded ontology
    approx_gnn_k       Tier A + GNN (ontology-aware XFG, contrastive), top-k   (ours)
    exact_gnn          Tier A + GNN, all survivors                       (ours, recall 100 %)

Recall is reported overall and on the planted conflicts that are only visible
through an alias ("alias") or through the hierarchy ("hierarchy").
"""
from __future__ import annotations

import json

from common import ROOT, load_cfg, load_dataset, ensure_truth, load_labels, model_path, save_result, train_gnn
from baselines.exact_index import candidate_pairs
from ecd.core import pair_key
from ecd.embedding import HashedAtomEmbedder, split_by_rule
from ecd.filtering import Mode
from ecd.pipeline import Detector
from metrics import precision_recall, recall_on


def run_variant(pol, emb, mode, k, truth, subsets):
    det = Detector(pol.copy(), emb, mode, top_k=k, use_faiss=False)
    rep = det.full_check()
    found = {(c.a, c.b) for c in rep.conflicts}
    p, r = precision_recall(found, truth)
    row = {"verified": rep.pairs_verified, "recall": r, "precision": p,
           "time_ms": rep.time_ms.get("select+verify")}
    for name, sub in subsets.items():
        row[f"recall_{name}"] = recall_on(found, sub)
    return row


def main():
    cfg = load_cfg("rq5_hetero")
    results = {}
    for ds in cfg["datasets"]:
        if not (ROOT / ds / "policy_v0.json").exists():
            print(f"skip {ds}")
            continue
        pol, _, planted, truth = load_dataset(ds)
        truth = ensure_truth(pol, ds, truth)
        kinds = json.loads((ROOT / ds / "planted_kinds.json").read_text())
        subsets = {"planted": {pair_key(*p) for p in planted}}
        for kind in ("alias", "hierarchy"):
            subsets[kind] = {pair_key(a, b) for a, b in planted if kinds.get(b) == kind}
        # cross-vocabulary conflicts: true conflicts whose two rules share no (attr, value) literal at all —
        # invisible to any literal index by construction
        def literals(r):
            out = set()
            for a in r.applicability.atoms:
                vals = a.value if isinstance(a.value, tuple) else (a.value,)
                out |= {(a.attr, v) for v in vals}
            return out
        lits = {r.id: literals(r) for r in pol}
        subsets["no_shared_literal"] = {p for p in truth if not (lits[p[0]] & lits[p[1]])}
        info = json.loads((ROOT / ds / "config.json").read_text())
        res = {"hetero_rate": info["hetero_rate"], "hierarchy_rate": info["hierarchy_rate"],
               "n_rules": len(pol), "n_truth": len(truth),
               "n_planted": len(subsets["planted"]), "n_alias": len(subsets["alias"]),
               "n_hierarchy": len(subsets["hierarchy"]), "n_no_shared_literal": len(subsets["no_shared_literal"])}
        top_ks = cfg.get("top_ks") or [cfg["top_k"]]
        idx = candidate_pairs(pol)
        res["exact_index"] = {"candidates": len(idx), "recall": recall_on(idx, truth),
                              **{f"recall_{n}": recall_on(idx, s) for n, s in subsets.items()}}
        idx2 = candidate_pairs(pol, pol.space.ontology)
        res["exact_index+ont"] = {"candidates": len(idx2), "recall": recall_on(idx2, truth),
                                  **{f"recall_{n}": recall_on(idx2, s) for n, s in subsets.items()}}
        # GNN trained on this dataset's own Z3 labels (rule-level split, test rules unseen);
        # a model saved by an earlier run of this script is reused.
        pairs = load_labels(pol, ds, cfg["max_smt_calls"], cfg["seed"])
        tr, te = split_by_rule(pairs, 0.2, cfg["seed"])
        mp = model_path(ds)
        if mp.exists():
            from ecd.embedding.gnn import GNNEmbedder, evaluate_auc
            emb = GNNEmbedder.load(mp, pol.space)
            res["gnn_val_auc"] = evaluate_auc(emb, pol, te)
        else:
            emb, hist = train_gnn(pol, tr, cfg, eval_pairs=te, log=None)
            emb.save(mp)
            res["gnn_val_auc"] = hist[-1].get("val_auc")
        for k in top_ks:
            res[f"approx_hash_k{k}"] = run_variant(pol, HashedAtomEmbedder(), Mode.APPROXIMATE, k, truth, subsets)
            res[f"approx_hash+ont_k{k}"] = run_variant(pol, HashedAtomEmbedder(use_ontology=True), Mode.APPROXIMATE, k, truth, subsets)
            res[f"approx_gnn_k{k}"] = run_variant(pol, emb, Mode.APPROXIMATE, k, truth, subsets)
        res["exact_gnn"] = run_variant(pol, emb, Mode.EXACT, 0, truth, subsets)
        res["tierA_only"] = run_variant(pol, None, Mode.EXACT, 0, truth, subsets)
        results[ds] = res
        print(ds, json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in res.items()
                              if not isinstance(v, dict)}))
        for kk, v in res.items():
            if isinstance(v, dict):
                print("   ", kk, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}))
        save_result("rq5_hetero", results)


if __name__ == "__main__":
    main()
