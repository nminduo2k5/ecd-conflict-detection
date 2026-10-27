"""RQ4 — Scalability 1k → 100k rules, 30 repetitions, Wilcoxon + A12.

For every size and seed a fresh policy is generated; the Δ path (10 Δs) is timed
in exact mode and in approximate mode (top-k) on identical copies.  The full
quadratic check is not part of the Δ path (it is a one-off cost reported for
the small sizes only).  The embedder is the hash baseline so that the numbers
measure the pipeline, not the GNN; the GNN's own embedding cost per size is
measured separately (`gnn_embed_ms`) with the model trained on synthetic_1k.
"""
from __future__ import annotations

import json
import time

from common import RESULTS, load_cfg, save_result, model_path
from benchmark.generator.synthetic import GenConfig, SyntheticGenerator
from ecd.embedding import HashedAtomEmbedder
from ecd.filtering import Mode
from ecd.pipeline import Detector
from metrics import mean_std
from stats import summarize


def delta_run(pol, deltas, approx: bool, k: int) -> dict:
    det = Detector(pol.copy(), HashedAtomEmbedder(),
                   Mode.APPROXIMATE if approx else Mode.EXACT, top_k=k, use_faiss=False)
    t0 = time.perf_counter()
    per = []
    for d in deltas:
        rep = det.apply(d)
        per.append((rep.time_ms.get("total", 0.0), rep.pairs_verified, rep.time_ms.get("select", 0.0)))
    total = (time.perf_counter() - t0) * 1000
    return {"total_ms": total, "per_delta_ms": [p[0] for p in per],
            "verified": [p[1] for p in per], "select_ms": [p[2] for p in per]}


def main():
    cfg = load_cfg("rq4_scalability")
    # Checkpoints: a finished size is in rq4_scalability.json, the seeds of the size in progress in
    # rq4_partial.json (written after every seed).  Re-running continues where it stopped; every
    # (size, seed) policy is regenerated deterministically, so a resumed run measures the same inputs.
    out_path, part_path = RESULTS / "rq4_scalability.json", RESULTS / "rq4_partial.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    partial = json.loads(part_path.read_text()) if part_path.exists() else {}
    gnn = None
    mp = model_path("data/processed/synthetic_1k")
    for n in cfg["sizes"]:
        if str(n) in out and out[str(n)].get("repeats") == cfg["repeats"]:
            print(f"n={n}: already complete in {out_path.name}, skipping")
            continue
        rows = partial.get(str(n), [])
        exact_t = [r["exact"]["total_ms"] for r in rows]
        approx_t = [r["approx"]["total_ms"] for r in rows]
        if rows:
            print(f"n={n}: resuming at seed {len(rows)} ({len(rows)} seeds restored from {part_path.name})")
        for s in range(len(rows), cfg["repeats"]):
            gen = SyntheticGenerator(GenConfig(n_rules=n, seed=s, n_deltas=cfg["n_deltas"]))
            t = time.perf_counter()
            pol = gen.generate()
            deltas = gen.deltas(pol)
            gen_ms = (time.perf_counter() - t) * 1000
            e = delta_run(pol, deltas, False, cfg["top_k"])
            a = delta_run(pol, deltas, True, cfg["top_k"])
            exact_t.append(e["total_ms"])
            approx_t.append(a["total_ms"])
            rows.append({"seed": s, "gen_ms": gen_ms, "exact": e, "approx": a})
            if s == 0 and mp.exists():
                try:
                    from ecd.embedding.gnn import GNNEmbedder
                    gnn = gnn or GNNEmbedder.load(mp, pol.space)
                    gnn.bind(pol)
                    t = time.perf_counter()
                    gnn.embed(list(pol))
                    rows[-1]["gnn_embed_ms"] = (time.perf_counter() - t) * 1000
                except Exception as ex:  # pragma: no cover
                    rows[-1]["gnn_embed_error"] = str(ex)
            print(f"n={n} seed={s}: exact {e['total_ms'] / len(deltas):.0f} ms/delta, "
                  f"approx {a['total_ms'] / len(deltas):.0f} ms/delta, verified/delta exact {sum(e['verified']) / len(deltas):.0f}")
            partial[str(n)] = rows                                   # checkpoint after every seed
            part_path.write_text(json.dumps(partial))
        stats = summarize(f"delta_time_ms@{n}", approx_t, exact_t)
        stats.update({
            "exact_ms_per_delta": mean_std([x / cfg["n_deltas"] for x in exact_t]),
            "approx_ms_per_delta": mean_std([x / cfg["n_deltas"] for x in approx_t]),
            "exact_verified_per_delta": mean_std([sum(r["exact"]["verified"]) / cfg["n_deltas"] for r in rows]),
            "approx_verified_per_delta": mean_std([sum(r["approx"]["verified"]) / cfg["n_deltas"] for r in rows]),
            "tierA_select_ms_per_delta": mean_std([sum(r["exact"]["select_ms"]) / cfg["n_deltas"] for r in rows]),
            "gen_ms": mean_std([r["gen_ms"] for r in rows]),
            "gnn_embed_ms_all_rules": rows[0].get("gnn_embed_ms"),
            "repeats": cfg["repeats"], "n_deltas": cfg["n_deltas"], "runs": rows})
        out[str(n)] = stats
        print(n, json.dumps({k: v for k, v in stats.items() if k != "runs"}, default=float))
        save_result("rq4_scalability", out)
        partial.pop(str(n), None)                                    # size finished → drop its partial rows
        part_path.write_text(json.dumps(partial))
    if part_path.exists() and not partial:
        part_path.unlink()


if __name__ == "__main__":
    main()
