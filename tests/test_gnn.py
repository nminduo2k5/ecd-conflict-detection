"""GNN embedder: train/inference consistency and learning signal (skipped without torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmark.generator.synthetic import GenConfig, SyntheticGenerator  # noqa: E402
from ecd.embedding import label_pairs, split_by_rule  # noqa: E402
from ecd.embedding.gnn import GNNEmbedder, TrainConfig, XFGEncoder, evaluate_auc, train_contrastive  # noqa: E402
from ecd.xfg import build_batch_graph, build_rule_graph  # noqa: E402


@pytest.fixture(scope="module")
def small():
    gen = SyntheticGenerator(GenConfig(n_rules=150, conflict_rate=0.1, hetero_rate=0.3, seed=3, n_deltas=0))
    pol = gen.generate()
    pairs = label_pairs(pol, max_smt_calls=1500, seed=3)
    return pol, pairs


def test_embed_matches_single_rule_graph(small):
    pol, _ = small
    emb = GNNEmbedder(XFGEncoder(32, 2), pol.space)
    rules = list(pol)[:5]
    batch = emb.embed(rules)
    with torch.no_grad():
        for r, v in zip(rules, batch):
            single = emb.model.rule_vectors(build_rule_graph(r, pol.space))[r.id].numpy()
            assert np.allclose(single, v, atol=1e-5)


def test_batch_graph_is_union_of_rule_graphs(small):
    pol, _ = small
    rules = list(pol)[:3]
    g = build_batch_graph(rules, pol.space)
    assert g.num_nodes == sum(build_rule_graph(r, pol.space).num_nodes for r in rules)


def test_contrastive_training_learns_overlap(small):
    """Learning signal on unseen rules (the full RQ1 comparison against the hash
    baseline is run on the 1k dataset by experiments/run_rq1_embedding.py)."""
    pol, pairs = small
    tr, te = split_by_rule(pairs, 0.25, seed=1)
    m = XFGEncoder(64, 2)
    auc_before = evaluate_auc(GNNEmbedder(m, pol.space), pol, te)
    hist = train_contrastive(m, pol, tr, TrainConfig(epochs=25, lr=3e-3, batch_anchors=64, seed=1), log=None)
    auc_after = evaluate_auc(GNNEmbedder(m, pol.space), pol, te)
    assert hist[-1]["loss"] < hist[0]["loss"]
    assert auc_after > auc_before + 0.05
    assert auc_after > 0.6


def test_save_load_roundtrip(small, tmp_path):
    pol, _ = small
    emb = GNNEmbedder(XFGEncoder(16, 1), pol.space)
    emb.save(tmp_path / "m.pt")
    emb2 = GNNEmbedder.load(tmp_path / "m.pt", pol.space)
    r = list(pol)[:2]
    assert np.allclose(emb.embed(r), emb2.embed(r))
