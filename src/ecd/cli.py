"""Command-line interface.

    ecd gen        --n 1000 --out data/processed/synthetic_1k [--hetero 0.2 --hierarchy 0.1]
    ecd check      data/processed/synthetic_1k/policy_v0.json --mode exact
    ecd evolve     data/processed/synthetic_1k --mode approximate --top-k 50
    ecd label      data/processed/synthetic_1k/policy_v0.json --out results/labels.json
    ecd train      data/processed/synthetic_1k/policy_v0.json results/labels.json --out results/gnn.pt
    ecd rodin      results/eventb/demo --project ECD [--build]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Embedding-guided incremental conflict detection")


def _embedder(name: str, model_path: str | None):
    if name == "none":
        return None
    if name == "hash":
        from ecd.embedding import HashedAtomEmbedder
        return HashedAtomEmbedder()
    if name == "hash+ont":
        from ecd.embedding import HashedAtomEmbedder
        return HashedAtomEmbedder(use_ontology=True)
    if name == "sbert":
        from ecd.embedding import SBertEmbedder
        return SBertEmbedder()
    if name == "gnn":
        from ecd.embedding.gnn import GNNEmbedder
        if not model_path:
            raise typer.BadParameter("--model is required for the gnn embedder")
        return GNNEmbedder.load(model_path)
    raise typer.BadParameter(f"unknown embedder {name}")


@app.command()
def gen(n: int = 1000, out: str = "data/processed/synthetic_1k", seed: int = 0,
        conflict_rate: float = 0.05, hetero: float = 0.2, hierarchy: float = 0.0,
        complex_rate: float = 0.05, n_deltas: int = 100):
    """Generate a synthetic evolving policy benchmark (with alias ontology when --hetero > 0)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmark.generator.synthetic import GenConfig, write_dataset
    p = write_dataset(GenConfig(n_rules=n, seed=seed, conflict_rate=conflict_rate,
                                hetero_rate=hetero, hierarchy_rate=hierarchy,
                                complex_rate=complex_rate, n_deltas=n_deltas), out)
    typer.echo(f"wrote {p}")


@app.command()
def check(policy: str, mode: str = "exact", embedder: str = "hash", model: str | None = None,
          top_k: int = 50, eventb_out: str | None = None, report: str | None = None):
    """Full conflict check of one policy file (native JSON / XACML / IAM / RBAC)."""
    from ecd.parsers import load_any
    from ecd.pipeline import Detector
    from ecd.filtering import Mode
    pol = load_any(policy)
    det = Detector(pol, _embedder(embedder, model), Mode(mode), top_k, eventb_out=eventb_out)
    rep = det.full_check()
    _print_report(rep)
    if report:
        Path(report).write_text(json.dumps(rep.to_dict(), indent=1))


@app.command()
def evolve(dataset: str, mode: str = "exact", embedder: str = "hash", model: str | None = None,
           top_k: int = 50, eventb_out: str | None = None, report: str | None = None,
           limit: int | None = None, skip_full: bool = False):
    """Replay deltas.json on policy_v0.json incrementally.

    --skip-full: don't run the quadratic initial check (use when you only
    want to time the Δ path on large policies)."""
    from ecd.core import Delta, Policy
    from ecd.pipeline import Detector
    from ecd.filtering import Mode
    d = Path(dataset)
    pol = Policy.load(d / "policy_v0.json")
    deltas = [Delta.from_dict(x) for x in json.loads((d / "deltas.json").read_text())]
    if limit:
        deltas = deltas[:limit]
    det = Detector(pol, _embedder(embedder, model), Mode(mode), top_k, eventb_out=eventb_out)
    if not skip_full:
        rep0 = det.full_check()
        typer.echo(f"v0: {len(rep0.conflicts)} conflicts, pruned {rep0.pruning_ratio:.1%}, "
                   f"{rep0.time_ms.get('select+verify', 0) / 1000:.1f}s")
    for i, dl in enumerate(deltas, 1):
        r = det.apply(dl)
        typer.echo(f"Δ{i:3d} {dl.kind.value:6s} {dl.target_id:>8s}: +{len(r.conflicts)} conflicts, "
                   f"verified {r.pairs_verified}, PO reuse {r.po_reuse_rate:.0%}, {r.time_ms.get('total', 0):.0f} ms")
    typer.echo(f"live conflicts: {len(det.all_conflicts())}")
    if det.eventb:
        st = det.eventb.po_stats()
        typer.echo(f"Event-B: {len(st)} contexts, THM POs new/inherited at last Δ: "
                   f"{st[-1]['po_new']}/{st[-1]['po_inherited']}")
    if report:
        Path(report).write_text(json.dumps([r.to_dict() for r in det.reports], indent=1))


@app.command()
def label(policy: str, out: str = "results/labels.json", max_smt: int = 20000, seed: int = 0):
    """Generate overlap labels with Z3 for contrastive training."""
    from ecd.core import Policy
    from ecd.embedding import label_pairs
    pol = Policy.load(policy)
    pairs = label_pairs(pol, max_smt_calls=max_smt, seed=seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps([p.__dict__ for p in pairs]))
    pos = sum(p.label for p in pairs)
    typer.echo(f"{len(pairs)} pairs ({pos} positive, {len(pairs) - pos} negative) → {out}")


@app.command()
def train(policy: str, labels: str, out: str = "results/gnn.pt", dim: int = 128, layers: int = 3,
          epochs: int = 60, lr: float = 3e-3, batch: int = 256, tau: float = 0.1):
    """Train the XFG-GNN with supervised contrastive loss."""
    from ecd.core import Policy
    from ecd.embedding.labeler import LabelledPair, split_by_rule
    from ecd.embedding.gnn import XFGEncoder, TrainConfig, train_contrastive, GNNEmbedder
    pol = Policy.load(policy)
    pairs = [LabelledPair(**p) for p in json.loads(Path(labels).read_text())]
    tr, va = split_by_rule(pairs, 0.2)
    model = XFGEncoder(dim, layers)
    train_contrastive(model, pol, tr, TrainConfig(epochs=epochs, lr=lr, batch_anchors=batch, tau=tau),
                      log=typer.echo, eval_pairs=va)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    GNNEmbedder(model, pol.space).save(out)
    typer.echo(f"saved {out}")


@app.command()
def rodin(ctx_dir: str, project: str = "ECD", workspace: str = "results/rodin_ws",
          rodin_path: str | None = None, build: bool = False):
    """Lay out a Rodin project from generated .buc contexts; --build runs Rodin headlessly."""
    from ecd.verification.eventb import RodinRunner, po_stats_from_dir
    rr = RodinRunner(rodin_path, workspace)
    proj = rr.prepare(ctx_dir, project)
    typer.echo(f"Rodin project written to {proj}")
    for row in po_stats_from_dir(ctx_dir):
        typer.echo(f"  v{row['version']}: new THM POs {row['po_new']}, inherited {row['po_inherited']}")
    if build:
        if not rr.available():
            typer.echo("Rodin executable not found (set --rodin-path); project files are ready for a manual build.")
            raise typer.Exit(1)
        res = rr.build(project)
        typer.echo(f"discharged {res.discharged}, undischarged {len(res.undischarged)}")
        for u in res.undischarged[:20]:
            typer.echo(f"  {u}")


def _print_report(rep):
    typer.echo(f"pairs: {rep.pairs_total}  pruned: {rep.pairs_pruned} ({rep.pruning_ratio:.1%})  "
               f"verified: {rep.pairs_verified}  conflicts: {len(rep.conflicts)}")
    for c in rep.conflicts[:10]:
        typer.echo(f"  {c.a} × {c.b} [{c.verdict.cls.value}] cex={c.counterexample}")
    if len(rep.conflicts) > 10:
        typer.echo(f"  … {len(rep.conflicts) - 10} more")


if __name__ == "__main__":
    app()
