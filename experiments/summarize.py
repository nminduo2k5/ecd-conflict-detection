"""Collect results/*.json into results/SUMMARY.md (tables ready for the paper)."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_env = os.environ.get("ECD_RESULTS_DIR")                     # same override as experiments/common.py
RES = (Path(_env) if Path(_env).is_absolute() else ROOT / _env) if _env else ROOT / "results"


def _load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x):
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def ms(d):
    return "–" if not d or d.get("n", 0) == 0 else f"{d['mean']:.0f} ± {d['std']:.0f}"


out = ["# Results summary", ""]

rq1 = _load("rq1_embedding")
if rq1:
    out += ["## RQ1 — overlap-awareness of the embedding (AUC on unseen rules, rule-level hold-out)", "",
            "| dataset | rules | pairs (test) | SBERT | hash | hash+ontology | GNN untrained | **GNN+contrastive** |",
            "|---|---|---|---|---|---|---|---|"]
    for ds, r in rq1.items():
        g = lambda k: (r.get(k) or {}).get("auc") if isinstance(r.get(k), dict) else None  # noqa: E731
        out.append(f"| {Path(ds).name} | {r['n_rules']} | {r['n_test']} | {f(g('sbert'))} | {f(g('hash'))} | "
                   f"{f(g('hash+ontology'))} | {f(g('gnn_untrained'))} | **{f(g('gnn_contrastive'))}** |")
    out.append("")

for res_name in ("rq2_pruning", "rq2_complex"):
    rq2 = _load(res_name)
    if not rq2:
        continue
    out += [f"## RQ2 ({res_name}) — pair reduction / recall on {Path(rq2['dataset']).name} ({rq2['n_rules']} rules, "
            f"{rq2['n_truth']} true conflicts)", "",
            "| variant | pairs to Z3 | pruning | recall | precision | Z3 calls to 1st conflict | time (s) |",
            "|---|---|---|---|---|---|---|"]
    for k, v in rq2.items():
        if isinstance(v, dict) and "verified" in v:
            out.append(f"| {k} | {v['verified']} | {pct(v['pruning_ratio'])} % | {pct(v['recall'])} % | "
                       f"{pct(v['precision'])} % | {f(v.get('smt_calls_to_first_conflict'))} | "
                       f"{f((v['time_ms'].get('select+verify') or 0) / 1000, 1)} |")
    for k in ("exact_index", "exact_index+ont"):
        if k in rq2:
            out.append(f"| {k} (candidates only) | {rq2[k]['candidates']} | – | {pct(rq2[k]['recall_on_candidates'])} % | – | – | – |")
    dp = rq2.get("delta_path")
    if dp:
        out += ["", f"Δ path ({list(dp.values())[0]['n_deltas_with_conflict']} Δs that introduce a conflict), exact mode:", "",
                "| ordering | Z3 calls before 1st conflict (mean ± sd) | verified pairs / Δ | ms / Δ | Wilcoxon p vs none | A12 vs none |",
                "|---|---|---|---|---|---|"]
        for k, v in dp.items():
            out.append(f"| {k} | {ms(v['calls_to_first_conflict'])} | {ms(v['verified_per_delta'])} | "
                       f"{ms(v['ms_per_delta'])} | {f(v.get('wilcoxon_p_vs_none'), 4)} | {f(v.get('A12_vs_none'))} |")
    out.append("")

rq3 = _load("rq3_incremental")
if rq3:
    s = rq3["summary"]
    out += [f"## RQ3 — incremental Event-B on {Path(s['dataset']).name} ({s['n_rules_v0']} rules, {len(rq3['rows'])} Δ)", "",
            f"- full check v0: {s['po_v0']} THM POs, {s['full_check_ms_v0'] / 1000:.1f} s",
            f"- per Δ: new POs {ms(s['po_new'])}, PO reuse rate {s['po_reuse_rate']['mean']:.4f}, "
            f"incremental time {ms(s['incremental_ms'])} ms",
            f"- full regeneration would produce {ms(s['po_full_regen'])} POs per Δ; from-scratch re-verification "
            f"takes {ms(s['full_reverify_ms'])} ms",
            f"- Event-B contexts (.ctx/.buc) in `{s['rodin_project']}`", ""]

rq4 = _load("rq4_scalability")
if rq4:
    out += ["## RQ4 — scalability of the Δ path (10 Δ per run, 30 seeds, hash embedder)", "",
            "| rules | exact ms/Δ | approx (top-50) ms/Δ | pairs verified/Δ exact | Tier A ms/Δ | Wilcoxon p | A12 (approx<exact) | GNN embed all rules (ms) |",
            "|---|---|---|---|---|---|---|---|"]
    for n in sorted(rq4, key=int):
        v = rq4[n]
        out.append(f"| {int(n):,} | {ms(v['exact_ms_per_delta'])} | {ms(v['approx_ms_per_delta'])} | "
                   f"{ms(v['exact_verified_per_delta'])} | {ms(v['tierA_select_ms_per_delta'])} | {f(v['p'], 4)} | "
                   f"{f(v['A12'])} | {f(v.get('gnn_embed_ms_all_rules'), 0)} |")
    out.append("")

rq5 = _load("rq5_hetero")
if rq5:
    out += ["## RQ5 — heterogeneous vocabularies / hierarchies (recall %, top-50 budget where applicable)", "",
            "| dataset | hetero | method | pairs to Z3 | recall all | recall cross-vocab (no shared literal) | recall alias-planted | recall hierarchy-planted |",
            "|---|---|---|---|---|---|---|---|"]
    for ds, r in rq5.items():
        out.append(f"| {Path(ds).name} | {int(100 * r['hetero_rate'])} % | *true conflicts* | {r['n_truth']} | – | "
                   f"n={r.get('n_no_shared_literal')} | n={r['n_alias']} | n={r['n_hierarchy']} |")
        for k, v in r.items():
            if isinstance(v, dict):
                out.append(f"| {Path(ds).name} | {int(100 * r['hetero_rate'])} % | {k} | "
                           f"{v.get('verified', v.get('candidates'))} | {pct(v.get('recall'))} | "
                           f"{pct(v.get('recall_no_shared_literal'))} | {pct(v.get('recall_alias'))} | {pct(v.get('recall_hierarchy'))} |")
    out.append("")

real = _load("real_policies")
if real:
    out += ["## Real policies (exact mode, hash embedder)", "",
            "| dataset | rules (norm.) | source rules | pairs | Tier A pruned | pairs to Z3 | conflicts | classes | recall | full check (ms) | Δ ms (mean ± sd) | approx top-50 recall |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, r in real.items():
        ak = next(k for k in r if k.startswith("approx_k"))
        out.append(f"| {Path(ds).name} | {r['rules']} | {r.get('source_rules') or '–'} | {r['pairs_total']} | "
                   f"{pct(r['pruning_ratio'])} % | {r['verified']} | {r['conflicts']} | {r['conflict_classes']} | "
                   f"{pct(r['recall'])} % | {r['full_check_ms']:.0f} | {ms(r['delta_ms'])} | {pct(r[ak]['recall'])} % |")
    out.append("")

(RES / "SUMMARY.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
