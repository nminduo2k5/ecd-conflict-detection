"""Step [2]: overlap-aware rule embedding — GNN over the XFG + contrastive loss (C1).

Requires torch only (a plain-torch GIN layer is used, no torch-geometric).

Training objective (supervised contrastive / InfoNCE with hard negatives):
    for an anchor rule r, positives P(r) = rules whose region overlaps r (label 1,
    from the Z3 oracle), negatives N(r) = label 0 (hard negatives — pairs that
    survived Tier A but are UNSAT — weighted higher).
    L(r) = -log  Σ_{p∈P(r)} exp(s(r,p)/τ)  /  Σ_{x∈P(r)∪N(r)} w_x exp(s(r,x)/τ)
with s = cosine similarity.

Train / inference consistency: both see the *same* per-rule XFG (private
leaves, ontology edges, numeric interval features), built by
`xfg.build_batch_graph` for training and `embed()` for inference.  Earlier
versions trained on a policy-level graph with shared leaves and embedded
single-rule graphs at inference, which made the learned model useless in the
pipeline; this module fixes that.

Usage:
    model = XFGEncoder(dim=128)
    train_contrastive(model, policy, labelled_pairs, TrainConfig(epochs=60))
    emb = GNNEmbedder(model, policy.space)      # implements Embedder
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict

import numpy as np

from ecd.core import AttributeSpace, Policy, Rule
from ecd.xfg.builder import Graph, NODE_TYPES, NUM_FEATS, VOCAB_BUCKETS, build_batch_graph
from .base import Embedder
from .labeler import LabelledPair

try:  # optional heavy dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("Install torch to use the GNN embedder: pip install torch")


if HAS_TORCH:

    class GraphTensors:
        """Graph → tensors (built once per graph)."""

        def __init__(self, g: Graph, device: str = "cpu"):
            self.t = torch.tensor(g.node_type, dtype=torch.long, device=device)
            self.k = torch.tensor(g.node_tok, dtype=torch.long, device=device)
            self.num = torch.tensor(g.node_num, dtype=torch.float32, device=device)
            self.gi = torch.tensor(g.node_graph, dtype=torch.long, device=device)
            self.n_graphs = g.num_graphs
            if g.edges:
                self.ei = torch.tensor(g.edges, dtype=torch.long, device=device).t().contiguous()
            else:
                self.ei = torch.zeros((2, 0), dtype=torch.long, device=device)
            self.rule_nodes = g.rule_nodes
            self.rule_graphs = g.rule_graphs
            self.n = len(g.node_type)

    class GINLayer(nn.Module):
        """Minimal GIN layer: h' = MLP((1+ε)h + Σ_neighbours h)."""

        def __init__(self, dim: int):
            super().__init__()
            self.eps = nn.Parameter(torch.zeros(1))
            self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.norm = nn.LayerNorm(dim)

        def forward(self, h, edge_index):
            src, dst = edge_index
            agg = torch.zeros_like(h).index_add_(0, dst, h[src])
            return self.norm(self.mlp((1 + self.eps) * h + agg))

    class XFGEncoder(nn.Module):
        """GIN message passing on the XFG + graph-level readout.

        The rule vector is the *sum-pooled* jumping-knowledge representation of
        every node of the rule's sub-graph (as in graph classification with GIN),
        not the RULE node alone: the RULE node is 3 hops away from the VALUE
        nodes, and relying on it alone made the signal too weak to train."""

        def __init__(self, dim: int = 128, layers: int = 3, vocab: int = VOCAB_BUCKETS):
            super().__init__()
            self.dim, self.n_layers, self.vocab = dim, layers, vocab
            self.type_emb = nn.Embedding(len(NODE_TYPES), dim)
            self.tok_emb = nn.Embedding(vocab, dim)
            self.num_proj = nn.Sequential(nn.Linear(NUM_FEATS, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.layers = nn.ModuleList([GINLayer(dim) for _ in range(layers)])
            self.out = nn.Sequential(nn.LayerNorm(dim * (layers + 1)), nn.Linear(dim * (layers + 1), dim),
                                     nn.ReLU(), nn.Linear(dim, dim))

        def node_states(self, g: "GraphTensors"):
            h = self.type_emb(g.t) + self.tok_emb(g.k) + self.num_proj(g.num) * g.num[:, :1]
            hs = [h]
            for layer in self.layers:
                h = F.relu(layer(h, g.ei))
                hs.append(h)
            return torch.cat(hs, dim=-1)                                   # [N, dim·(L+1)]

        def forward(self, g: "Graph | GraphTensors"):
            """Graph-level vectors, one per rule sub-graph: [n_graphs, dim], L2-normalised."""
            if isinstance(g, Graph):
                g = GraphTensors(g)
            cat = self.node_states(g)
            pooled = torch.zeros((g.n_graphs, cat.shape[1]), dtype=cat.dtype, device=cat.device)
            pooled = pooled.index_add_(0, g.gi, cat)
            return F.normalize(self.out(pooled), dim=-1)

        def rule_vectors(self, g: "Graph | GraphTensors") -> dict[str, "torch.Tensor"]:
            if isinstance(g, Graph):
                g = GraphTensors(g)
            v = self.forward(g)
            return {rid: v[gi] for rid, gi in g.rule_graphs.items()}

        def config(self) -> dict:
            return {"dim": self.dim, "layers": self.n_layers, "vocab": self.vocab}


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 3e-3
    tau: float = 0.1
    hard_neg_weight: float = 2.0
    batch_anchors: int = 256
    in_batch_weight: float = 0.5     # other anchors of the batch as extra negatives (0 = off);
    #                                  unlabelled random pairs are ~97 % disjoint (Tier A statistics)
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"
    eval_every: int = 10


def _pad(lists: list[list], fill, dtype):
    L = max((len(x) for x in lists), default=1)
    out = torch.full((len(lists), L), fill, dtype=dtype)
    for i, x in enumerate(lists):
        if x:
            out[i, :len(x)] = torch.tensor(x, dtype=dtype)
    return out


def train_contrastive(model, policy: Policy, pairs: list[LabelledPair],
                      cfg: TrainConfig = TrainConfig(), log=print,
                      eval_pairs: list[LabelledPair] | None = None) -> list[dict]:
    """Supervised contrastive training on per-rule XFGs (same graphs as inference).

    Returns a history of {"epoch", "loss", ["val_auc"]} dicts."""
    _require_torch()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model.to(cfg.device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    rules = list(policy)
    g = GraphTensors(build_batch_graph(rules, policy.space), cfg.device)
    node_of = g.rule_graphs                     # rule id → row of the graph-level output

    # anchor → (others, labels, weights), both directions
    by_anchor: dict[str, list[tuple[str, int, float]]] = {}
    for p in pairs:
        if p.a not in node_of or p.b not in node_of:
            continue
        w = cfg.hard_neg_weight if (p.hard and p.label == 0) else 1.0
        by_anchor.setdefault(p.a, []).append((p.b, p.label, w))
        by_anchor.setdefault(p.b, []).append((p.a, p.label, w))
    anchors = [a for a, lst in by_anchor.items() if any(l == 1 for _, l, _ in lst)]
    if not anchors:
        raise ValueError("no anchor has a positive pair; cannot train")
    a_nodes = torch.tensor([node_of[a] for a in anchors], dtype=torch.long)
    o_nodes = _pad([[node_of[o] for o, _, _ in by_anchor[a]] for a in anchors], 0, torch.long)
    labels = _pad([[l for _, l, _ in by_anchor[a]] for a in anchors], -1, torch.long)
    weights = _pad([[w for _, _, w in by_anchor[a]] for a in anchors], 1.0, torch.float32)
    mask = labels >= 0
    pos_mask = labels == 1
    log_w = torch.log(weights)
    known_pos = {(p.a, p.b) for p in pairs if p.label == 1} | {(p.b, p.a) for p in pairs if p.label == 1}

    history: list[dict] = []
    n = len(anchors)
    steps_per_epoch = max(1, math.ceil(n / cfg.batch_anchors))
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(steps_per_epoch):
            rows = perm[s * cfg.batch_anchors:(s + 1) * cfg.batch_anchors]
            opt.zero_grad()
            h = model(g)
            ha = h[a_nodes[rows]]                                    # [B, d]
            ho = h[o_nodes[rows]]                                    # [B, L, d]
            sims = (ha[:, None, :] * ho).sum(-1) / cfg.tau            # [B, L]
            s_all = (sims + log_w[rows]).masked_fill(~mask[rows], -1e9)
            s_pos = sims.masked_fill(~pos_mask[rows], -1e9)
            if cfg.in_batch_weight > 0 and len(rows) > 1:
                ids = [anchors[int(i)] for i in rows]
                ib = (ha @ ha.t()) / cfg.tau + math.log(cfg.in_batch_weight)   # [B, B]
                ib_mask = torch.eye(len(rows), dtype=torch.bool)
                for i, x in enumerate(ids):                            # never push a known positive apart
                    for j, y in enumerate(ids):
                        if i != j and (x, y) in known_pos:
                            ib_mask[i, j] = True
                s_all = torch.cat([s_all, ib.masked_fill(ib_mask, -1e9)], dim=1)
            loss = (torch.logsumexp(s_all, 1) - torch.logsumexp(s_pos, 1)).mean()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        rec = {"epoch": ep, "loss": tot / steps_per_epoch}
        if eval_pairs and ((ep + 1) % cfg.eval_every == 0 or ep == cfg.epochs - 1):
            rec["val_auc"] = evaluate_auc(GNNEmbedder(model, policy.space), policy, eval_pairs)
            model.train()
        history.append(rec)
        if log and ("val_auc" in rec or ep == 0 or ep == cfg.epochs - 1):
            log(f"epoch {ep:3d}  loss {rec['loss']:.4f}" + (f"  val_auc {rec['val_auc']:.3f}" if "val_auc" in rec else ""))
    model.eval()
    return history


def evaluate_auc(embedder: Embedder, policy: Policy, pairs: list[LabelledPair]) -> float:
    """AUC of -distance against the overlap label (RQ1 metric)."""
    ids = sorted({p.a for p in pairs} | {p.b for p in pairs})
    vec = dict(zip(ids, embedder.embed([policy.get(i) for i in ids])))
    scores = np.array([-np.linalg.norm(vec[p.a] - vec[p.b]) for p in pairs])
    y = np.array([p.label for p in pairs])
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(scores)) + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


class GNNEmbedder(Embedder):
    """Embeds rules on their own per-rule XFG (ontology edges + numeric features
    from `space`), in chunks, exactly as during training."""

    def __init__(self, model, space: AttributeSpace | None = None, chunk: int = 4096):
        _require_torch()
        self.model = model.eval()
        self.dim = model.dim
        self.space = space
        self.chunk = chunk

    def bind(self, policy: Policy) -> None:
        self.space = policy.space

    def embed(self, rules: list[Rule]) -> np.ndarray:
        if not rules:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(rules), self.chunk):
                chunk = rules[i:i + self.chunk]
                g = GraphTensors(build_batch_graph(chunk, self.space))
                v = self.model(g)
                idx = torch.tensor([g.rule_graphs[r.id] for r in chunk], dtype=torch.long)
                out.append(v[idx].cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def embed_policy(self, policy: Policy) -> dict[str, np.ndarray]:
        rules = list(policy)
        return dict(zip((r.id for r in rules), self.embed(rules)))

    def save(self, path: str) -> None:
        torch.save({"state": self.model.state_dict(), "config": self.model.config()}, path)

    @staticmethod
    def load(path: str, space: AttributeSpace | None = None) -> "GNNEmbedder":
        _require_torch()
        blob = torch.load(path, map_location="cpu")
        cfg = blob["config"]
        m = XFGEncoder(cfg["dim"], cfg["layers"], cfg["vocab"])
        m.load_state_dict(blob["state"])
        return GNNEmbedder(m, space)


def train_config_from_dict(d: dict) -> TrainConfig:
    keys = set(asdict(TrainConfig()).keys())
    return TrainConfig(**{k: v for k, v in d.items() if k in keys})
