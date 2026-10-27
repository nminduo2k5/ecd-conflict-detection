"""Label (Z3) and train the XFG-GNN for one dataset → results/models/<dataset>.pt

    python train_model.py data/processed/complex_1k [max_smt_calls]
"""
from __future__ import annotations

import sys

from common import load_cfg, load_dataset, load_labels, model_path, train_gnn
from ecd.embedding import split_by_rule


def main():
    ds = sys.argv[1]
    max_smt = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    cfg = load_cfg("rq1_embedding")
    pol, _, _, _ = load_dataset(ds)
    pairs = load_labels(pol, ds, max_smt, cfg["seed"])
    tr, te = split_by_rule(pairs, 0.2, cfg["seed"])
    emb, hist = train_gnn(pol, tr, cfg, eval_pairs=te)
    emb.save(model_path(ds))
    print(f"saved {model_path(ds)}  val_auc={hist[-1].get('val_auc')}")


if __name__ == "__main__":
    main()
