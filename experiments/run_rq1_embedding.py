"""RQ1 — Does the embedding distance reflect *overlap* (not text similarity)?

Protocol: label pairs with Z3 (ontology-aware) → hold out 20 % of the RULES
(test pairs touch ≥ 1 unseen rule) → for each embedder compute AUC / P@k of
-distance against the overlap label.

Ablation (paper §5): text-only (SBERT) / bag-of-atoms (hash) / hash + hand-coded
ontology / GNN untrained (structure only) / GNN + contrastive (ours).
Runs on every dataset listed in configs/experiments/rq1_embedding.yaml
(synthetic + real policies).
"""
from __future__ import annotations

import json
import time

import numpy as np

from common import ROOT, load_cfg, load_dataset, load_labels, model_path, save_result, train_gnn
from ecd.embedding import HashedAtomEmbedder, split_by_rule
from ecd.embedding.labeler import LabelledPair
from metrics import ranking_quality


def evaluate(embedder, policy, pairs: list[LabelledPair]) -> dict:
    ids = sorted({p.a for p in pairs} | {p.b for p in pairs})
    embedder.bind(policy)
    t0 = time.perf_counter()
    vec = dict(zip(ids, embedder.embed([policy.get(i) for i in ids])))
    ms = (time.perf_counter() - t0) * 1000
    scores = [-float(np.linalg.norm(vec[p.a] - vec[p.b])) for p in pairs]
    out = ranking_quality(scores, [p.label for p in pairs])
    out["embed_ms_per_rule"] = ms / max(1, len(ids))
    return out


def run_dataset(cfg: dict, dataset: str) -> dict:
    pol, _, _, _ = load_dataset(dataset)
    pairs = load_labels(pol, dataset, cfg["max_smt_calls"], cfg["seed"])
    train, test = split_by_rule(pairs, 0.2, cfg["seed"])
    res = {"n_rules": len(pol), "n_pairs": len(pairs), "n_train": len(train), "n_test": len(test),
           "pos_rate": sum(p.label for p in pairs) / max(1, len(pairs)),
           "hash": evaluate(HashedAtomEmbedder(), pol, test)}
    if not pol.space.ontology.is_empty():
        res["hash+ontology"] = evaluate(HashedAtomEmbedder(use_ontology=True), pol, test)
    if cfg.get("use_sbert"):
        try:
            from baselines.sbert import make
            res["sbert"] = evaluate(make(), pol, test)
        except Exception as ex:  # sentence-transformers missing / offline
            res["sbert"] = {"error": str(ex)}
    if cfg.get("use_gnn") and any(p.label == 1 for p in train):
        import torch
        from ecd.embedding.gnn import XFGEncoder, GNNEmbedder
        g = cfg.get("gnn", {})
        torch.manual_seed(cfg["seed"])            # untrained baseline: seeded random initialisation
        res["gnn_untrained"] = evaluate(GNNEmbedder(XFGEncoder(g.get("dim", 128), g.get("layers", 3)), pol.space),
                                        pol, test)
        emb, hist = train_gnn(pol, train, cfg, eval_pairs=test)
        res["gnn_contrastive"] = evaluate(emb, pol, test)
        res["gnn_contrastive"]["history"] = hist
        emb.save(model_path(dataset))
    return res


def main():
    cfg = load_cfg("rq1_embedding")
    results = {}
    for ds in cfg["datasets"]:
        if not (ROOT / ds / "policy_v0.json").exists():
            print(f"skip {ds} (missing)")
            continue
        print(f"=== {ds}")
        results[ds] = run_dataset(cfg, ds)
        summary = {k: (round(v["auc"], 3) if isinstance(v, dict) and "auc" in v else v)
                   for k, v in results[ds].items() if k not in ("n_rules", "n_pairs", "n_train", "n_test", "pos_rate")}
        print(json.dumps(summary))
        save_result("rq1_embedding", results)


if __name__ == "__main__":
    main()
