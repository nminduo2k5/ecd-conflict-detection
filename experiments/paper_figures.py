"""Publication figures (vector PDF), one per research question, straight from results/*.json.

    python experiments/paper_figures.py            # → ../ecd_latex/figures/rq{1..5}.pdf
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_env = os.environ.get("ECD_RESULTS_DIR")                     # same override as experiments/common.py
RES = (Path(_env) if Path(_env).is_absolute() else ROOT / _env) if _env else ROOT / "results"
# output: argv[1], else the paper's figure folder (only when reading the paper's own results dir)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else (RES / "paper_figures" if _env else ROOT.parent / "ecd_latex" / "figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
# colour-blind-safe palette + hatches so the bars survive greyscale printing
PAL = ["#bdbdbd", "#7f7f7f", "#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
HATCH = ["", "//", "", "\\\\", "xx", "..", "++"]


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def grouped_bars(ax, groups, series, values, width=0.8, ylim=None, ylabel="", ncol=1, legend_loc="upper left",
                 log=False):
    """groups: x labels; series: legend labels; values[s][g] (nan = missing)."""
    x = np.arange(len(groups))
    n = len(series)
    w = width / n
    for i, s in enumerate(series):
        v = np.array([values[i][g] for g in range(len(groups))], dtype=float)
        ax.bar(x + (i - (n - 1) / 2) * w, v, w, label=s, color=PAL[i % len(PAL)], hatch=HATCH[i % len(HATCH)],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    if log:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc=legend_loc, ncol=ncol, frameon=False, handlelength=1.3, columnspacing=0.8,
              handletextpad=0.4, borderaxespad=0.3, labelspacing=0.35)


def fig_rq1():
    rq1 = load("rq1_embedding")
    ds = [("data/processed/synthetic_1k", "synthetic_1k"), ("data/processed/hetero_1k_h50", "hetero_1k_h50"),
          ("data/processed/real/aws_managed_300", "AWS managed 300"), ("data/processed/real/xacml_gpms", "gpms"),
          ("data/processed/real/xacml_pluto", "pluto")]
    meths = [("sbert", "SBERT (text)"), ("hash", "hash (bag of atoms)"), ("hash+ontology", "hash + ontology"),
             ("gnn_untrained", "GNN untrained"), ("gnn_contrastive", "GNN + contrastive (ours)")]
    vals = [[(rq1.get(d, {}).get(m) or {}).get("auc", np.nan) if isinstance(rq1.get(d, {}).get(m), dict) else np.nan
             for d, _ in ds] for m, _ in meths]
    fig, ax = plt.subplots(figsize=(4.8, 1.8), dpi=300)
    grouped_bars(ax, [l for _, l in ds], [l for _, l in meths], vals, ylim=(0, 1.3), ylabel="AUC on unseen rules",
                 ncol=3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(0.5, color="k", ls=":", lw=0.7)
    ax.text(0.995, 0.51, "chance level", transform=ax.get_yaxis_transform(), fontsize=7.5, va="bottom", ha="right")
    for i, (m, _) in enumerate(meths):                       # value labels for the GNN+contrastive bars
        if m == "gnn_contrastive":
            for g in range(len(ds)):
                v = vals[i][g]
                if v == v:
                    ax.text(g + (i - 2) * 0.16, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "rq1.pdf")
    plt.close(fig)


def fig_rq2():
    a2, c2 = load("rq2_pruning"), load("rq2_complex")
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.8, 1.75), dpi=300, gridspec_kw={"width_ratios": [1.3, 1]})
    # (a) pairs sent to Z3 per variant, both datasets, log scale
    variants = [("no_tierA", "full\nSMT"), ("tierA_only", "Tier A\nexact"), ("approx_gnn_k10", "Tier A\ntop-10"),
                ("approx_gnn_k50", "Tier A\ntop-50")]
    idx = [("exact_index", "literal\nindex"), ("exact_index+ont", "index +\nontology")]
    groups = [l for _, l in variants] + [l for _, l in idx]
    vals = []
    for d in (a2, c2):
        row = [(d.get(k) or {}).get("verified", np.nan) for k, _ in variants]
        row += [(d.get(k) or {}).get("candidates", np.nan) for k, _ in idx]
        vals.append(row)
    grouped_bars(a, groups, ["synthetic_1k", "complex_1k"], vals, ylabel="pairs sent to the solver", log=True,
                 ylim=(1e3, 3e6), ncol=2)
    a.tick_params(axis="x", labelsize=7)
    a.set_title("(a) verification work per variant", loc="left")
    # (b) recall vs budget k for hash and GNN orders on both datasets
    ks = [10, 25, 50]
    for d, name, ls in ((a2, "synthetic_1k", "-"), (c2, "complex_1k", "--")):
        for emb, c, mk in (("hash", PAL[1], "s"), ("gnn", PAL[4], "o")):
            r = [100 * (d.get(f"approx_{emb}_k{k}") or {}).get("recall", np.nan) for k in ks]
            b.plot(ks, r, ls=ls, marker=mk, ms=3.5, lw=1.0, color=c, label=f"{emb.upper() if emb == 'gnn' else emb}, {name}")
    b.axhline(100, color="k", lw=0.6, ls=":")
    b.set_xticks(ks)
    b.set_xlabel("budget $k$ (survivors verified per rule)")
    b.set_ylabel("recall (%)")
    b.set_ylim(55, 103)
    b.grid(linewidth=0.3, alpha=0.6)
    b.legend(loc="lower right", frameon=False, handlelength=1.8, fontsize=6.8, labelspacing=0.3)
    b.set_title("(b) approximate mode", loc="left")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "rq2.pdf")
    plt.close(fig)


def fig_rq3():
    rq3 = load("rq3_incremental")
    rows = rq3.get("rows", [])
    fig, ax = plt.subplots(figsize=(4.8, 1.6), dpi=300)
    if rows:
        x = np.array([r["delta"] for r in rows])
        po = np.array([max(r["po_new"], 0.6) for r in rows], dtype=float)   # 0 → tiny bar on log axis
        kinds = {"add": PAL[2], "modify": PAL[3], "remove": PAL[6]}
        for kind, c in kinds.items():
            m = np.array([r["kind"] == kind for r in rows])
            ax.bar(x[m], po[m], 0.8, color=c, edgecolor="black", linewidth=0.3, label=f"new POs, Δ {kind}")
        regen = np.mean([r["po_full_regen"] for r in rows])
        tier = np.mean([r["po_full_tierA"] for r in rows])
        ax.axhline(regen, color="k", ls="--", lw=0.8)
        ax.axhline(tier, color="k", ls=":", lw=0.8)
        ax.text(x[-1], regen * 1.35, f"regenerate all opposite-effect pairs ({regen:,.0f} POs)", ha="right", fontsize=7)
        ax.text(x[-1], tier * 1.35, f"from scratch with Tier A ({tier:,.0f} POs)", ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_ylim(0.5, regen * 8)
        ax.set_xlabel("change Δ (policy version)")
        ax.set_ylabel("THM proof obligations")
        # legend sits in the free band between the tallest bar and the Tier-A reference line
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.58), ncol=3, frameon=False, handlelength=1.1,
                  columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(OUT / "rq3.pdf")
    plt.close(fig)


def fig_rq4():
    rq4 = load("rq4_scalability")
    fig, ax = plt.subplots(figsize=(3.4, 1.9), dpi=300)
    if rq4:
        sizes = sorted(int(k) for k in rq4)
        for key, lab, c, mk in (("exact_ms_per_delta", "exact mode (recall 100 %)", PAL[3], "s"),
                                ("approx_ms_per_delta", "approximate mode, k = 50", PAL[2], "o"),
                                ("tierA_select_ms_per_delta", "Tier A alone", PAL[1], "^")):
            m = [rq4[str(s)][key]["mean"] for s in sizes]
            sd = [rq4[str(s)][key]["std"] for s in sizes]
            ax.errorbar(sizes, m, yerr=sd, marker=mk, ms=3.5, lw=1.0, capsize=2, color=c, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rules in the policy")
        ax.set_ylabel("ms per change (mean ± sd, 30 seeds)")
        ax.grid(which="major", linewidth=0.3, alpha=0.6)
        ax.legend(loc="upper left", frameon=False, handlelength=1.6, handletextpad=0.4, borderaxespad=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "rq4.pdf")
    plt.close(fig)


def fig_rq5():
    rq5 = load("rq5_hetero")
    dsets = [("data/processed/hetero_1k_h25", "25 % of rules in the second vocabulary"),
             ("data/processed/hetero_1k_h50", "50 % of rules in the second vocabulary")]
    meths = [("exact_index", "literal index"), ("exact_index+ont", "index + ontology"),
             ("approx_hash_k5", "Tier A + hash, k=5"), ("approx_hash+ont_k5", "Tier A + hash+ont., k=5"),
             ("approx_gnn_k5", "Tier A + GNN, k=5 (ours)"), ("approx_gnn_k50", "Tier A + GNN, k=50"),
             ("tierA_only", "Tier A, all survivors (exact)")]
    vals = [[100 * (rq5.get(d, {}).get(m) or {}).get("recall_no_shared_literal", np.nan) for d, _ in dsets]
            for m, _ in meths]
    fig, ax = plt.subplots(figsize=(4.8, 1.75), dpi=300)
    grouped_bars(ax, [l for _, l in dsets], [l for _, l in meths], vals, ylim=(0, 172),
                 ylabel="recall on cross-vocabulary conflicts (%)", ncol=3, width=0.75)
    ax.set_yticks([0, 25, 50, 75, 100])
    for i in range(len(meths)):                              # value labels
        for g in range(len(dsets)):
            v = vals[i][g]
            if v == v:
                ax.text(g + (i - 3) * (0.75 / 7), v + 1.5, f"{v:.0f}", ha="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "rq5.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_rq1, fig_rq2, fig_rq3, fig_rq4, fig_rq5):
        f()
    for stale in ("rq1_rq5.pdf", "rq3_rq4.pdf"):
        (OUT / stale).unlink(missing_ok=True)
    print("wrote", ", ".join(f"{OUT / f'rq{i}.pdf'}" for i in range(1, 6)))
