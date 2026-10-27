"""Shared helpers for the RQ scripts."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from ecd.core import Delta, Policy  # noqa: E402


def _env_dir(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    if not v:
        return default
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


# Where this run writes its outputs (rq*.json, models/, eventb/).  Set ECD_RESULTS_DIR
# (absolute, or relative to the ecd/ folder) to keep a re-run completely apart from
# ecd/results, the folder the paper was built from — see experiments/run_all.ps1.
RESULTS = _env_dir("ECD_RESULTS_DIR", ROOT / "results")
# ECD_FRESH_CACHE=1: recompute the Z3 ground truth and the overlap labels under
# <RESULTS>/cache/ instead of reusing the copies cached next to the datasets.
FRESH_CACHE = os.environ.get("ECD_FRESH_CACHE", "").strip().lower() in ("1", "true", "yes")
if os.environ.get("ECD_RESULTS_DIR"):
    print(f"[ecd] results dir: {RESULTS}" + ("  (fresh cache: truth + labels recomputed)" if FRESH_CACHE else ""))


def cache_file(dataset: str | Path, name: str) -> Path:
    """Derived data of a dataset (truth_conflicts.json, labels.json): next to the dataset by
    default (shared by every run, deterministic), under <RESULTS>/cache/ with ECD_FRESH_CACHE=1."""
    d = ROOT / dataset
    if not FRESH_CACHE:
        return d / name
    try:
        rel = d.resolve().relative_to((ROOT / "data" / "processed").resolve())
    except ValueError:
        rel = Path(d.name)
    c = RESULTS / "cache" / rel
    c.mkdir(parents=True, exist_ok=True)
    return c / name


def results_path(rel: str | Path) -> Path:
    """Map a config path such as 'results/eventb/rq3' into the active results dir."""
    parts = Path(rel).parts
    if parts and parts[0] == "results":
        parts = parts[1:]
    return RESULTS.joinpath(*parts)


def load_cfg(name: str) -> dict:
    base = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    exp = yaml.safe_load((ROOT / f"configs/experiments/{name}.yaml").read_text())
    base.update(exp)
    return base


def load_dataset(path: str | Path):
    d = ROOT / path
    pol = Policy.load(d / "policy_v0.json")
    deltas = [Delta.from_dict(x) for x in json.loads((d / "deltas.json").read_text())]
    planted = {tuple(p) for p in json.loads((d / "planted_conflicts.json").read_text())}
    truth_file = cache_file(path, "truth_conflicts.json")
    truth = {tuple(p) for p in json.loads(truth_file.read_text())} if truth_file.exists() else None
    return pol, deltas, planted, truth


def ensure_truth(pol: Policy, path: str | Path, truth):
    """Ground truth = exhaustive pairwise Z3 (cached, see cache_file)."""
    if truth is not None:
        return truth
    from baselines.full_smt import run as full_smt
    t, ms = full_smt(pol, cache_file(path, "truth_conflicts.json"))
    print(f"ground truth: {len(t)} conflicts in {ms / 1000:.1f} s")
    return t


def load_labels(pol: Policy, path: str | Path, max_smt_calls: int, seed: int = 0):
    """Z3 overlap labels (cached as labels.json, see cache_file)."""
    from ecd.embedding import label_pairs
    from ecd.embedding.labeler import LabelledPair
    f = cache_file(path, "labels.json")
    if f.exists():
        return [LabelledPair(**p) for p in json.loads(f.read_text())]
    t0 = time.perf_counter()
    pairs = label_pairs(pol, max_smt_calls=max_smt_calls, seed=seed)
    f.write_text(json.dumps([p.__dict__ for p in pairs]))
    print(f"labelled {len(pairs)} pairs in {time.perf_counter() - t0:.0f} s")
    return pairs


def train_gnn(pol: Policy, train_pairs, cfg: dict, eval_pairs=None, log=print):
    """Train a fresh XFG-GNN with the config's gnn section; returns (embedder, history)."""
    import torch
    from ecd.embedding.gnn import XFGEncoder, GNNEmbedder, TrainConfig, train_contrastive
    g = cfg.get("gnn", {})
    torch.manual_seed(cfg.get("seed", 0))        # seed the weight initialisation too (train_contrastive seeds the rest)
    m = XFGEncoder(g.get("dim", 128), g.get("layers", 3))
    tc = TrainConfig(epochs=g.get("epochs", 60), lr=g.get("lr", 3e-3), tau=g.get("tau", 0.1),
                     batch_anchors=g.get("batch_anchors", 256), seed=cfg.get("seed", 0),
                     hard_neg_weight=g.get("hard_neg_weight", 2.0),
                     in_batch_weight=g.get("in_batch_weight", 0.5))
    hist = train_contrastive(m, pol, train_pairs, tc, log=log, eval_pairs=eval_pairs)
    return GNNEmbedder(m, pol.space), hist


def save_result(name: str, obj) -> Path:
    out = RESULTS / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=1, default=str))
    return out


def model_path(dataset: str) -> Path:
    p = RESULTS / "models" / (Path(dataset).name + ".pt")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
