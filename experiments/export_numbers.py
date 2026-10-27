"""Export results/*.json as LaTeX macros → <paper dir>/numbers.tex

    python experiments/export_numbers.py ../ecd_latex/numbers.tex

Every macro the paper uses is always defined; values missing from the result
files render as \\todo so the paper compiles before the runs finish.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_env = os.environ.get("ECD_RESULTS_DIR")                     # same override as experiments/common.py
RES = (Path(_env) if Path(_env).is_absolute() else ROOT / _env) if _env else ROOT / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "numbers.tex"


def load(name, kind=dict):
    p = RES / f"{name}.json"
    if not p.exists():
        return kind()
    d = json.loads(p.read_text())
    return d if isinstance(d, kind) else kind()


macros: dict[str, str] = {}


def m(name: str, value, fmt="{:.2f}"):
    name = re.sub(r"[^A-Za-z]", "", name)
    if value is None or (isinstance(value, float) and value != value):
        macros[name] = r"\todo"
    elif fmt == "{:.4f}" and isinstance(value, float) and value < 1e-4:   # p-values
        macros[name] = r"$<$0.0001"
    elif isinstance(value, bool):
        macros[name] = str(value)
    elif isinstance(value, int) or fmt == "{:d}":
        macros[name] = f"{int(round(value)):,}".replace(",", r"\,")
    elif isinstance(value, float):
        macros[name] = fmt.format(value)
    else:
        macros[name] = str(value)


def pct(x):
    return None if x is None else 100 * x


def g(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


_WORDS = {"k10": "KTen", "k25": "KTwentyFive", "k50": "KFifty", "100": "Hundred", "300": "ThreeHundred",
          "+ont": "Ont", "+ontology": "Ont"}


def _key(k: str) -> str:
    """Result key → CamelCase macro fragment with digits spelled out (LaTeX macro names are letters only)."""
    for a, b in _WORDS.items():
        k = k.replace(a, f"_{b}_")
    return re.sub(r"[^A-Za-z]", "", k.replace("_", " ").title())


# ---- RQ1 -----------------------------------------------------------------
rq1 = load("rq1_embedding")
RQ1_DS = {"data/processed/synthetic_1k": "Syn", "data/processed/hetero_1k_h50": "Het",
          "data/processed/real/aws_managed_300": "Aws", "data/processed/real/xacml_gpms": "Gpms",
          "data/processed/real/xacml_pluto": "Pluto", "data/processed/real/xacml_itrust": "Itrust"}
for ds, key in RQ1_DS.items():
    r = rq1.get(ds) or {}
    for meth, short in (("hash", "Hash"), ("hash+ontology", "HashOnt"), ("sbert", "Sbert"),
                        ("gnn_untrained", "GnnU"), ("gnn_contrastive", "Gnn")):
        m(f"rqOne{key}{short}", g(r, meth, "auc"), "{:.3f}")
        m(f"rqOne{key}{short}PTen", g(r, meth, "p@10"), "{:.2f}")
    m(f"rqOne{key}Rules", r.get("n_rules"), "{:d}")
    m(f"rqOne{key}Test", r.get("n_test"), "{:d}")
    m(f"rqOne{key}Pairs", r.get("n_pairs"), "{:d}")

# ---- RQ2 (synthetic_1k → rqTwo…, complex_1k → rqTwoC…) --------------------
RQ2_VARIANTS = ["no_tierA", "tierA_only", "exact_hash", "exact_hash+ont", "exact_gnn",
                "approx_hash_k10", "approx_hash_k25", "approx_hash_k50",
                "approx_hash+ont_k10", "approx_hash+ont_k25", "approx_hash+ont_k50",
                "approx_gnn_k10", "approx_gnn_k25", "approx_gnn_k50"]


def export_rq2(rq2: dict, P: str):
    m(f"{P}Rules", rq2.get("n_rules"), "{:d}")
    m(f"{P}Truth", rq2.get("n_truth"), "{:d}")
    surv = g(rq2, "tierA_only", "verified")
    m(f"{P}SurvivorConflictRate", pct(rq2["n_truth"] / surv) if (surv and rq2.get("n_truth")) else None, "{:.1f}")
    for k in RQ2_VARIANTS:
        v = rq2.get(k) or {}
        key = _key(k)
        m(f"{P}{key}Verified", v.get("verified"), "{:d}")
        m(f"{P}{key}Pruning", pct(v.get("pruning_ratio")), "{:.1f}")
        m(f"{P}{key}Recall", pct(v.get("recall")), "{:.1f}")
        m(f"{P}{key}Precision", pct(v.get("precision")), "{:.1f}")
        t = g(v, "time_ms", "select+verify")
        m(f"{P}{key}Time", t / 1000 if t is not None else None, "{:.1f}")
        m(f"{P}{key}Pairs", v.get("pairs_total"), "{:d}")
        m(f"{P}{key}Calls", v.get("smt_calls_to_first_conflict"), "{:d}")
    for k, key in (("exact_index", "ExactIndex"), ("exact_index+ont", "ExactIndexOnt")):
        v = rq2.get(k) or {}
        m(f"{P}{key}Candidates", v.get("candidates"), "{:d}")
        m(f"{P}{key}Recall", pct(v.get("recall_on_candidates")), "{:.1f}")
        rc = v.get("recall_on_candidates")
        m(f"{P}{key}Miss", pct(1 - rc) if rc is not None else None, "{:.1f}")   # conflicts the index cannot propose
        surv = g(rq2, "tierA_only", "verified")
        m(f"{P}{key}VsTierA", v["candidates"] / surv if (surv and v.get("candidates")) else None, "{:.0f}")
    dp = rq2.get("delta_path") or {}
    m(f"{P}DeltaN", g(dp, "none", "n_deltas_with_conflict"), "{:d}")
    for k in ("none", "hash", "hash+ont", "gnn"):
        v = dp.get(k) or {}
        key = _key(k)
        m(f"{P}Delta{key}Calls", g(v, "calls_to_first_conflict", "mean"), "{:.1f}")
        m(f"{P}Delta{key}CallsSd", g(v, "calls_to_first_conflict", "std"), "{:.1f}")
        m(f"{P}Delta{key}CallsMedian", g(v, "calls_to_first_conflict", "median"), "{:.0f}")
        m(f"{P}Delta{key}Verified", g(v, "verified_per_delta", "mean"), "{:.1f}")
        m(f"{P}Delta{key}Ms", g(v, "ms_per_delta", "mean"), "{:.0f}")
        m(f"{P}Delta{key}P", v.get("wilcoxon_p_vs_none"), "{:.4f}")
        m(f"{P}Delta{key}A", v.get("A12_vs_none"), "{:.2f}")


export_rq2(load("rq2_pruning"), "rqTwo")
export_rq2(load("rq2_complex"), "rqTwoC")

# ---- RQ3 -----------------------------------------------------------------
rq3 = load("rq3_incremental")
s = rq3.get("summary") or {}
rows = rq3.get("rows") or []
m("rqThreeRules", s.get("n_rules_v0"), "{:d}")
m("rqThreeDeltas", len(rows) if rows else None, "{:d}")
m("rqThreePoZero", s.get("po_v0"), "{:d}")
m("rqThreeFullMsZero", s["full_check_ms_v0"] / 1000 if s.get("full_check_ms_v0") else None, "{:.1f}")
m("rqThreePoNew", g(s, "po_new", "mean"), "{:.1f}")
m("rqThreePoNewMax", max((r["po_new"] for r in rows), default=None), "{:d}")
m("rqThreeReuse", pct(g(s, "po_reuse_rate", "mean")), "{:.2f}")
m("rqThreeIncMs", g(s, "incremental_ms", "mean"), "{:.0f}")
m("rqThreeIncMsSd", g(s, "incremental_ms", "std"), "{:.0f}")
m("rqThreeIncMsMedian", g(s, "incremental_ms", "median"), "{:.0f}")
m("rqThreePoRegen", g(s, "po_full_regen", "mean"), "{:.0f}")
fr = g(s, "full_reverify_ms", "mean")
m("rqThreeFullReverifyS", fr / 1000 if fr else None, "{:.1f}")
inc = g(s, "incremental_ms", "mean")
m("rqThreeSpeedup", fr / inc if (fr and inc) else None, "{:.0f}")
m("rqThreePoTierA", (sum(r["po_full_tierA"] for r in rows) / len(rows)) if rows else None, "{:.0f}")

# ---- RQ4 -----------------------------------------------------------------
rq4 = load("rq4_scalability")
for n, key in (("1000", "OneK"), ("5000", "FiveK"), ("10000", "TenK"), ("20000", "TwentyK"),
               ("50000", "FiftyK"), ("100000", "HundredK")):
    v = rq4.get(n) or {}
    m(f"rqFour{key}Exact", g(v, "exact_ms_per_delta", "mean"), "{:.0f}")
    m(f"rqFour{key}ExactSd", g(v, "exact_ms_per_delta", "std"), "{:.0f}")
    m(f"rqFour{key}Approx", g(v, "approx_ms_per_delta", "mean"), "{:.0f}")
    m(f"rqFour{key}ApproxSd", g(v, "approx_ms_per_delta", "std"), "{:.0f}")
    m(f"rqFour{key}Verified", g(v, "exact_verified_per_delta", "mean"), "{:.0f}")
    m(f"rqFour{key}TierA", g(v, "tierA_select_ms_per_delta", "mean"), "{:.0f}")
    m(f"rqFour{key}P", v.get("p"), "{:.4f}")
    m(f"rqFour{key}A", v.get("A12"), "{:.2f}")
    ge = v.get("gnn_embed_ms_all_rules")
    m(f"rqFour{key}GnnEmbed", ge / 1000 if ge else None, "{:.1f}")
    if v:
        m("rqFourRepeats", v.get("repeats"), "{:d}")
        m("rqFourDeltas", v.get("n_deltas"), "{:d}")
if "rqFourRepeats" not in macros:
    m("rqFourRepeats", None)
    m("rqFourDeltas", None)

# ---- RQ5 -----------------------------------------------------------------
rq5 = load("rq5_hetero")
for ds, key in (("data/processed/hetero_1k_h00", "HZero"), ("data/processed/hetero_1k_h25", "HTwentyFive"),
                ("data/processed/hetero_1k_h50", "HFifty")):
    r = rq5.get(ds) or {}
    m(f"rqFive{key}Truth", r.get("n_truth"), "{:d}")
    m(f"rqFive{key}Alias", r.get("n_alias"), "{:d}")
    m(f"rqFive{key}Hier", r.get("n_hierarchy"), "{:d}")
    m(f"rqFive{key}NoLit", r.get("n_no_shared_literal"), "{:d}")
    m(f"rqFive{key}GnnAuc", r.get("gnn_val_auc"), "{:.3f}")
    for meth, short in (("exact_index", "Idx"), ("exact_index+ont", "IdxOnt"), ("approx_hash_k50", "Hash"),
                        ("approx_hash+ont_k50", "HashOnt"), ("approx_gnn_k50", "Gnn"), ("exact_gnn", "GnnExact"),
                        ("approx_hash_k10", "HashKTen"), ("approx_hash+ont_k10", "HashOntKTen"), ("approx_gnn_k10", "GnnKTen"),
                        ("approx_hash_k5", "HashKFive"), ("approx_hash+ont_k5", "HashOntKFive"), ("approx_gnn_k5", "GnnKFive"),
                        ("tierA_only", "TierA")):
        v = r.get(meth) or {}
        m(f"rqFive{key}{short}Recall", pct(v.get("recall")), "{:.1f}")
        m(f"rqFive{key}{short}Alias", pct(v.get("recall_alias")), "{:.1f}")
        m(f"rqFive{key}{short}Hier", pct(v.get("recall_hierarchy")), "{:.1f}")
        m(f"rqFive{key}{short}NoLit", pct(v.get("recall_no_shared_literal")), "{:.1f}")
        m(f"rqFive{key}{short}Verified", v.get("verified", v.get("candidates")), "{:d}")

# ---- real ----------------------------------------------------------------
real = load("real_policies")
REAL_DS = ["xacml_itrust", "xacml_hl7", "xacml_gpms", "xacml_pluto", "xacml_conference", "xacml_fedora",
           "xacml_kmarket", "xacml_obligation", "xacml_simpson", "aws_managed_100", "aws_managed_300",
           "k8s_clusterroles"]
for name in REAL_DS:
    r = real.get(f"data/processed/real/{name}") or {}
    key = _key(name.replace("xacml_", ""))
    m(f"real{key}Rules", r.get("rules"), "{:d}")
    m(f"real{key}Src", r.get("source_rules"), "{:d}")
    m(f"real{key}Pairs", r.get("pairs_total"), "{:d}")
    m(f"real{key}Pruning", pct(r.get("pruning_ratio")), "{:.1f}")
    m(f"real{key}Verified", r.get("verified"), "{:d}")
    m(f"real{key}Conflicts", r.get("conflicts"), "{:d}")
    m(f"real{key}FullMs", r.get("full_check_ms"), "{:.0f}")
    m(f"real{key}DeltaMs", g(r, "delta_ms", "mean"), "{:.0f}")
    ak = next((k for k in r if k.startswith("approx_k")), None)
    m(f"real{key}ApproxRecall", pct(g(r, ak, "recall")) if ak else None, "{:.1f}")
    m(f"real{key}Approx", r.get("rules_with_overapprox_condition"), "{:d}")

lines = [r"% auto-generated by experiments/export_numbers.py — do not edit",
         r"\providecommand{\todo}{\textbf{[??]}}"]
for k, v in sorted(macros.items()):
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(macros)} macros, {sum(1 for v in macros.values() if v == chr(92) + 'todo')} pending)")
