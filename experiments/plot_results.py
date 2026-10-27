"""Publication figures for RQ1–RQ5 from results/*.json  →  results/figures/*.png"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9})
ROOT = Path(__file__).resolve().parents[1]
_env = os.environ.get("ECD_RESULTS_DIR")                     # same override as experiments/common.py
RES = (Path(_env) if Path(_env).is_absolute() else ROOT / _env) if _env else ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def plot_rq1(d):
    methods = ["sbert", "hash", "hash+ontology", "gnn_untrained", "gnn_contrastive"]
    labels = ["SBERT (text)", "hash (atoms)", "hash + ontology", "GNN untrained", "GNN + contrastive"]
    dsets = [k for k in d if isinstance(d[k], dict)]
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(dsets) + 2), 3.6), dpi=200)
    w = 0.8 / len(methods)
    x = np.arange(len(dsets))
    for i, (m, lab) in enumerate(zip(methods, labels)):
        vals = [d[ds].get(m, {}).get("auc", np.nan) if isinstance(d[ds].get(m), dict) else np.nan for ds in dsets]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=lab)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([Path(ds).name.replace("synthetic_", "syn ").replace("xacml_", "") for ds in dsets], rotation=15)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC (distance vs. Z3 overlap label)")
    ax.set_title("RQ1: overlap-awareness of the embedding (unseen rules)")
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "rq1_embedding.png")
    plt.close(fig)


def plot_rq2(d):
    variants = [k for k in d if isinstance(d[k], dict) and "verified" in d[k]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)
    v = [d[k]["verified"] for k in variants]
    r = [100 * d[k]["recall"] for k in variants]
    a1.bar(range(len(variants)), v, color="#2b5c8f")
    a1.set_yscale("log")
    a1.set_xticks(range(len(variants)))
    a1.set_xticklabels(variants, rotation=35, ha="right")
    a1.set_ylabel("pairs sent to Z3 (log)")
    a1.set_title(f"(a) verification work, {d['n_rules']} rules")
    a2.bar(range(len(variants)), r, color="#1b9e77")
    a2.set_xticks(range(len(variants)))
    a2.set_xticklabels(variants, rotation=35, ha="right")
    a2.set_ylim(0, 105)
    a2.set_ylabel("conflict recall (%)")
    a2.set_title("(b) recall vs. exhaustive Z3 ground truth")
    for i, val in enumerate(r):
        a2.text(i, val + 1, f"{val:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "rq2_pruning.png")
    plt.close(fig)
    dp = d.get("delta_path")
    if dp:
        fig, ax = plt.subplots(figsize=(5, 3.4), dpi=200)
        names = list(dp)
        means = [dp[n]["calls_to_first_conflict"]["mean"] for n in names]
        stds = [dp[n]["calls_to_first_conflict"]["std"] for n in names]
        ax.bar(names, means, yerr=stds, capsize=3, color=["#7f7f7f"] + ["#1b9e77"] * (len(names) - 1))
        ax.set_ylabel("Z3 calls before first conflict (per Δ)")
        ax.set_title("RQ2: earlier detection with embedding order")
        fig.tight_layout()
        fig.savefig(FIG / "rq2_delta_path.png")
        plt.close(fig)


def plot_rq3(d):
    rows = d["rows"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), dpi=200, sharex=True)
    x = [r["delta"] for r in rows]
    a1.plot(x, [r["po_full_regen"] for r in rows], "--", color="#d95f02", label="full regeneration (all opposite-effect pairs)")
    a1.plot(x, [r["po_full_tierA"] for r in rows], ":", color="#e6ab02", label="from-scratch with Tier A")
    a1.plot(x, [max(r["po_new"], 0.5) for r in rows], "o-", ms=3, color="#1b9e77", label="incremental: new POs at this Δ")
    a1.set_yscale("log")
    a1.set_ylabel("THM proof obligations")
    a1.set_title("(a) proof obligations per Δ")
    a1.legend(fontsize=8)
    cm = {"add": "#2b5c8f", "modify": "#e6ab02", "remove": "#e7298a"}
    for kind in cm:
        xs = [r["delta"] for r in rows if r["kind"] == kind]
        ys = [r["incremental_ms"] for r in rows if r["kind"] == kind]
        a2.scatter(xs, ys, label=f"Δ {kind}", color=cm[kind], s=18)
    fr = [(r["delta"], r["full_reverify_ms"]) for r in rows if "full_reverify_ms" in r]
    if fr:
        a2.scatter([f[0] for f in fr], [f[1] for f in fr], marker="x", color="#d95f02", label="full re-verification")
    a2.set_yscale("log")
    a2.set_xlabel("Δ step")
    a2.set_ylabel("ms (log)")
    a2.set_title("(b) time per Δ vs. full re-verification")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "rq3_incremental.png")
    plt.close(fig)


def plot_rq4(d):
    sizes = sorted(int(k) for k in d)
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=200)
    for key, lab, c in (("exact_ms_per_delta", "exact (recall 100 %)", "#d95f02"),
                        ("approx_ms_per_delta", "approximate (top-50)", "#1b9e77")):
        m = [d[str(s)][key]["mean"] for s in sizes]
        sd = [d[str(s)][key]["std"] for s in sizes]
        ax.errorbar(sizes, m, yerr=sd, marker="o", capsize=3, label=lab, color=c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rules")
    ax.set_ylabel("ms per Δ (mean ± sd, 30 seeds)")
    ax.set_title("RQ4: Δ path cost vs. policy size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "rq4_scalability.png")
    plt.close(fig)


def plot_rq5(d):
    dsets = list(d)
    methods = [m for m in ("exact_index", "exact_index+ont", "approx_hash_k50", "approx_hash+ont_k50", "approx_gnn_k50")
               if all(m in d[ds] for ds in dsets)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=200, sharey=True)
    for ax, key, title in zip(axes, ("recall", "recall_no_shared_literal", "recall_hierarchy"),
                              ("all conflicts", "cross-vocabulary conflicts (no shared literal)", "hierarchy-only conflicts")):
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            vals = [100 * (d[ds][m].get(key) if d[ds][m].get(key) == d[ds][m].get(key) else 0) for ds in dsets]
            ax.bar(np.arange(len(dsets)) + i * w - 0.4 + w / 2, vals, w, label=m)
        ax.set_xticks(range(len(dsets)))
        ax.set_xticklabels([f"hetero {int(100 * d[ds]['hetero_rate'])}%" for ds in dsets])
        ax.set_title(title)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("recall (%)")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("RQ5: candidate selection under heterogeneous vocabularies / hierarchies")
    fig.tight_layout()
    fig.savefig(FIG / "rq5_hetero.png")
    plt.close(fig)


def main():
    for name, fn in (("rq1_embedding", plot_rq1), ("rq2_pruning", plot_rq2), ("rq3_incremental", plot_rq3),
                     ("rq4_scalability", plot_rq4), ("rq5_hetero", plot_rq5)):
        d = _load(name)
        if d:
            try:
                fn(d)
                print("wrote", FIG / f"{name}.png")
            except Exception as ex:  # keep going if one result file is partial
                print(f"{name}: {ex}")


if __name__ == "__main__":
    main()
