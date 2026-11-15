# ECD — Sound and Incremental Conflict Detection in Evolving Access Control Policies with Verifier-Supervised Graph Embeddings

Code, datasets and results for the paper

> **Sound and Incremental Conflict Detection in Evolving Access Control Policies with Verifier-Supervised Graph
> Embeddings**
>
> Duong-Nguyen Minh¹, Cuong-Nguyen Van¹ ✉, Binh-Trinh Thanh¹, Tuan-Bui Anh¹, and Ninh-Thuan Truong²
>
> ¹ Faculty of Information Systems, Phenikaa University, Hanoi, Vietnam\
> ² VNU University of Engineering and Technology, Hanoi, Vietnam\
> ✉ Corresponding author: cuong.nguyenvan@phenikaa-uni.edu.vn

| author | affiliation | ORCID |
|---|---|---|
| Duong-Nguyen Minh | Phenikaa University | [0009-0003-2075-4758](https://orcid.org/0009-0003-2075-4758) |
| Cuong-Nguyen Van (corresponding) | Phenikaa University | [0009-0008-7778-5121](https://orcid.org/0009-0008-7778-5121) |
| Binh-Trinh Thanh | Phenikaa University | [0009-0008-0434-1255](https://orcid.org/0009-0008-0434-1255) |
| Tuan-Bui Anh | Phenikaa University | [0009-0004-5664-198X](https://orcid.org/0009-0004-5664-198X) |
| Ninh-Thuan Truong | VNU University of Engineering and Technology | — |

**Keywords:** Attribute-based access control · XACML · Policy conflict detection · SMT solving · Graph neural
networks · Incremental verification.

Code, data and results: <https://github.com/nminduo2k5/ecd-conflict-detection>.

## Abstract

Access control policies with tens of thousands of rules evolve continuously, and every change may introduce a
conflict: two rules that apply to the same request but prescribe opposite decisions. Exact detection is a pairwise
satisfiability problem, quadratic in the number of rules, and similarity-based detectors are fast but offer no
guarantee. We propose the ECD method, which keeps exact verification separate from candidate ordering. A sound
symbolic pre-filter discards the pairs that provably cannot conflict; the remaining pairs are verified exactly by Z3,
with counterexample requests, and their verdicts are kept in a versioned ledger, so that a change re-verifies only
the pairs it affects while all other verdicts provably remain valid. An overlap-aware rule embedding, learned by a
graph neural network with a contrastive objective whose labels come from the solver, only orders the candidates and
selects them under a verification budget. On synthetic policies of up to 10⁵ rules and eleven real XACML, AWS IAM
and Kubernetes policies, the exact mode keeps 100 % recall while discarding 98.0 % of the rule pairs before the
solver, and a change is verified in tens to hundreds of milliseconds on policies of 10³–10⁴ rules and in about 5 s
at 10⁵. The embedding ranks overlapping rules better than bag-of-atoms baselines; after the filter, its benefit is
recall under a small verification budget, and a fixed budget loses recall as policies grow.

## The ECD method

ECD (*embedding-guided conflict detection*) lets a learned embedding choose only the **order** and the **number** of
pairs that an exact verifier checks; every verdict comes from Z3. Two rules *conflict* when some request satisfies
both and their effects differ; the witness request is reported as a counterexample. The paper's contributions are:

1. **Two-tier candidate selection** with a provably sound first tier, and an exact and an approximate mode (G1 —
   skip most pairs without losing a conflict).
2. **Incremental verification**: a change re-verifies only the pairs it affects, with a preservation theorem for all
   other verdicts (G2 — reuse earlier verdicts with a formal guarantee).
3. **A rule embedding trained on SMT overlap labels** that only orders the candidates (G3 — use learned similarity
   without letting it decide).
4. **An open-source implementation** evaluated on synthetic policies of up to 10⁵ rules and eleven real policies.

```
policy P + change Δ + ontology O
   ──[1] normalise  DNF sub-rules r#k, canonical applicability ⟦r⟧ under O, rule feature graph (RFG) per rule
   ──[2] embed      GIN over the RFG, e(r) ∈ ℝ^d, trained contrastively on Z3 overlap labels (hard negatives)
   ──[3] select     Tier A: sound pruning (Theorem 1) → Tier B: order by ‖e(t) − e(r)‖
                    exact mode = all survivors, approximate mode = top-k per rule
   ──[4] verify     Z3 on ⟦t⟧ ∧ ⟦r⟧: unsat (disjoint) or sat + counterexample request q
   ──[5] report     conflict pair, request q, class (masked / order-dependent / real);
                    verdicts outside Impact(Δ) are reused from the verdict ledger (Theorem 2)
```

* **Normalisation.** Enclosing `PolicySet`/`Policy` targets are conjoined with the rule's target; targets and
  conditions are put into DNF and each disjunct becomes a sub-rule with the same effect. Literals outside the
  supported fragment (regular expressions, date functions, comparisons between two attributes) are replaced by
  *true* and the sub-rule is flagged `approx`: this over-approximation can add a candidate but never hides a
  conflict. Each IAM statement becomes one rule (wildcards expanded against the IAM action registry); each RBAC rule
  becomes an allow-only rule.
* **Ontology.** Alias attributes and values (`jobTitle=physician ≡ role=doctor`) and value hierarchies
  (`senior_doctor ⊑ doctor`) are canonicalised before every exact step. The embedding sees the raw vocabulary plus
  the ontology as graph edges and has to *learn* the same relations from the solver labels.
* **Tier A — sound symbolic pruning.** For each attribute, a per-rule abstraction (interval intersection for
  numeric atoms, allow/deny value set for categorical atoms) is memoised. A pair is discarded iff the effects are
  equal (P0) or some shared attribute has disjoint abstractions (P1); mixed abstractions and raw formulas never
  cause a discard. Under the single-valued request model no discarded pair is a conflict (Theorem 1, soundness of
  Tier A). Abstractions are memoised per rule, so a change against a policy of *n* rules costs O(*n*) look-ups.
  Tier A is also complete when every atom constrains a single attribute; the solver is still needed for the
  counterexample and for cross-attribute formulas φ (e.g. `hour + riskScore > 30`).
* **Tier B — overlap-aware embedding.** Survivors are ordered by increasing embedding distance. In the *exact* mode
  every survivor is verified, so recall is 100 % and the embedding only decides how soon conflicts appear; in the
  *approximate* mode only the top-*k* survivors per rule are verified (for CI/CD gates), with the empirical recall
  reported. No verdict depends on the ranking.
* **Exact verification and the verdict ledger.** Z3 decides every candidate pair. Categorical attributes are
  encoded as integers with a per-attribute value coding and a domain axiom, numeric attributes as bounded integers,
  booleans as booleans, and raw formulas are parsed as SMT-LIB; the encoding of ⟦r⟧ is cached per rule and the
  solver is reused with push/pop (about 0.5 ms per disjoint pair, 1–2 ms per conflicting pair, whose model is
  decoded into a concrete request). For every verified pair the ledger stores the content hashes of both canonical
  rules, the verdict (*unsat*, or *sat* with the witness request) and the class. A change Δ (add, remove or modify a
  rule; a modification is a removal followed by an addition) removes the entries of removed or modified rules and
  verifies only the Tier-A survivors of Impact(Δ); every other verdict remains correct in the next version for a
  fixed ontology (Theorem 2, preservation under Δ), and the stored hashes let the ledger check this at run time.

Supported policy languages: XACML 3.0, AWS IAM and Kubernetes RBAC, plus a synthetic generator (1k–100k rules,
planted conflicts, arithmetic conditions, heterogeneous vocabularies with an alias/hierarchy ontology, change
sequences).

## Main results

All numbers are from the paper (`synthetic_1k` unless stated otherwise).

| RQ | question | result |
|---|---|---|
| RQ1 | Does the embedding distance reflect the *overlap* of two rules rather than their textual similarity? | AUC on pairs touching a held-out rule: GNN 0.846 ± 0.014 (`synthetic_1k`), 0.853 ± 0.011 (`hetero_1k_h50`) and 0.814 ± 0.053 (gpms) over five training seeds, above every baseline (SBERT, hash, hash+ontology, untrained GNN); SBERT is at chance on the synthetic policy (0.482). On the merged AWS policy SBERT is slightly better (0.778 vs 0.753) because IAM action names carry meaning; on pluto the AUC is degenerate. |
| RQ2 | How far can the verified pairs be reduced at 100 % recall, and does the embedding find conflicts earlier? | Tier A discards 98.0 % of 499 500 pairs with 100 % recall (precision 100 %) and cuts solver time from 114.5 s to 17.7 s; the literal index proposes 11× more candidates and still misses 1.2 % of the conflicts. 98.6 % of the 10 125 survivors are conflicts, so the order barely matters; on `complex_1k` (60 % cross-attribute arithmetic) the share drops to 91.4 % and the order still changes little. The second half of RQ2 is answered negatively: after the sound filter, the embedding does not find conflicts earlier. |
| RQ3 | How much work does incremental verification save per change? | The initial version needs 10 125 pair verifications (142 *unsat*, 9 983 *sat*), 23.3 s. Each later change verifies 18.0 pairs on average (at most 56) and reuses 99.83 % of the verdicts, against 253 457 pairs without Tier A and 10 209 with it for a full re-verification; 54 ± 34 ms per change (median 52 ms) against 18.3 s from scratch (about 340×). |
| RQ4 | Does the approach scale to 10⁵ rules? | Exact mode: 60 ms per change at 10³ rules, 502 ms at 10⁴, 5 150 ms at 10⁵. The approximate mode (*k* = 50) is significantly cheaper from 5 000 rules up (982 ms at 10⁵), but the conflicts introduced by a change grow with the policy while *k* does not: its recall is 98 % at 10³, about half at 5 000 and 2.5 % at 10⁵, so the exact mode stays the default. Embedding all 10⁵ rules with the GNN once takes 34.2 s. |
| RQ5 | Does a learned embedding beat a literal index across vocabularies? | With 50 % of rules in a second vocabulary, a literal index finds 8.0 % of the cross-vocabulary conflicts (81 810 candidates) and a hand-written ontology 98.4 % (124 351 candidates); Tier A needs no ontology, keeps all of them among 11 934 survivors, and the exact mode finds all. With *k* = 50 every order reaches 97.9–98.7 %; with *k* = 5 the hash order reaches 12.0 %, the hash over canonical tokens 30.4 % and the GNN order 33.3 %. |
| real | eleven real XACML, AWS IAM and Kubernetes policies | The exact mode reaches 100 % recall on all eleven; Tier A prunes 99.0 % (gpms), 98.9 % (pluto) and 99.7 % (300-policy AWS merge) of the pairs, and the full check of that merge takes 3 800 ms and a change 238 ms. In approximate mode (*k* = 50) only that merge loses conflicts (36.9 % recall): the single `AWSDenyAll` statement accounts for 752 of its 761 conflicts. iTrust and the Kubernetes roles are allow-only and have no conflicts. |

**Limitations (paper §4.3).** Theorem 1 assumes single-valued attributes. XACML attributes are bags; with bags
encoded exactly (one Boolean variable per value) and (P1) disabled on them, `synthetic_1k` with `role` as a bag has
10 344 instead of 9 983 conflicts and gpms with its four subject attributes as bags 417 instead of 214, so the
single-valued assumption hides 3.5 % and 49 % of the bag conflicts and bag attributes must be declared (Tier A still
prunes 97.9 % and 98.1 % of the pairs). Over-approximated XACML sub-expressions cannot hide a conflict but may report
a potential one; such rules are flagged. Verdicts rely on the correctness of Z3 and of the SMT encoding. The GNN is
trained per policy and retraining under vocabulary drift was not evaluated; the real policies are small, so the
scalability claims rest on the generator. Future work: retraining under vocabulary drift and the combination of
several policies.

## Installation

ECD is implemented in Python (about 3 000 lines of library code plus 1 700 for the generator and the experiments).
Python ≥ 3.10 (the reported runs used Python 3.13 on Windows 10, on a dual-core Intel i3-7020U laptop CPU with four
threads and no GPU).

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"                                              :: z3-solver, numpy, scipy, pyyaml, typer, pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu   :: GNN encoder (the CPU build is enough)
pip install sentence-transformers faiss-cpu                          :: optional: SBERT baseline, ANN index
pytest                                                               :: 38 tests (GNN tests are skipped without torch)
```

## Reproducing the paper

`results_rerun/` is the run reported in the paper: one JSON per research question, per-step logs, `RUN_INFO.md`
(machine, package versions, step durations), `SUMMARY.md`, `numbers.tex` (the paper's numbers as LaTeX macros), the
generated result tables and the vector figures.

```bat
experiments\run_all.cmd -ResultsDir results_mine                     :: every step into a fresh folder (~1 h, RQ4 is the long step)
experiments\run_all.cmd -ResultsDir results_mine -Steps rq1,rq2       :: a subset of the steps
python experiments\compare_results.py results_rerun results_mine     :: macro-by-macro comparison with the reported run
```

Steps, in order: `rq1`, `train_complex`, `rq2`, `rq2_complex`, `rq3`, `rq5`, `real`, `rq4`, `summarize`, `export`,
`tables`, `figures`, `plots`. Deterministic numbers (pair counts, pruning ratios, recall of the exact mode and of the
hash/index baselines, pairs verified and verdicts reused per change, real-policy counts) reproduce exactly. Numbers
that depend on the trained GNN vary by about ±0.01 between runs; timings depend on the machine. The trained GNN
models (8 × 25 MB) are not tracked; `run_all` retrains them. RQ4 checkpoints after every seed and resumes if
interrupted.

GNN settings used in the paper: *d* = 128, *L* = 3 GIN layers, τ = 0.1, hard-negative weight *w* = 2, in-batch
weight λ = 0.5, Adam (3·10⁻³), 256 anchors per batch, 30 epochs, 20 000 Z3 labels per policy, 20 % of the rules held
out (`configs/experiments/rq1_embedding.yaml`). Recall and precision are measured against the exhaustive pairwise
Z3 ground truth; RQ4 repeats every size with 30 seeds, other timings are means over a change sequence, and
comparisons use the paired Wilcoxon test and the Vargha–Delaney Â12 effect size.

Every experiment script also runs on its own from `experiments/`:

| script | research question | output |
|---|---|---|
| `run_rq1_embedding.py` | RQ1 — is the embedding overlap-aware? AUC of the embedding distance against Z3 overlap labels on held-out rules; SBERT / hash / hash+ontology / untrained-GNN baselines | `rq1_embedding.json` |
| `run_rq2_pruning.py` [`rq2_complex`] | RQ2 — pairs pruned, recall, time, solver calls before the first conflict; full-SMT, literal-index and index+ontology baselines; `complex_1k` variant with 60 % cross-attribute arithmetic conditions | `rq2_pruning.json`, `rq2_complex.json` |
| `run_rq3_incremental.py` | RQ3 — pairs verified and verdicts reused per change versus full re-verification; change time versus from-scratch re-verification | `rq3_incremental.json` |
| `run_rq4_scalability.py` | RQ4 — cost of one change for 1k–100k rules, 30 seeds, exact versus approximate mode, Wilcoxon test and Â12 | `rq4_scalability.json` |
| `run_rq5_hetero.py` | RQ5 — heterogeneous vocabularies and role hierarchy: literal index versus Tier A versus embedding orders | `rq5_hetero.json` |
| `run_real.py` | eleven real policies (XPA XACML set, AWS managed IAM merges, Kubernetes ClusterRoles) | `real_policies.json` |
| `summarize.py`, `export_numbers.py`, `export_tables.py`, `paper_figures.py`, `plot_results.py` | summary, LaTeX macros, result tables, paper figures, diagnostic plots | `SUMMARY.md`, `numbers.tex`, `paper_tables/`, `paper_figures/`, `figures/` |

> **Note on RQ3 timing and Event-B.** The code still contains an optional Event-B/Rodin export of the ledger
> (`src/ecd/verification/eventb/`, `eventb/`, `ecd rodin`) from an earlier version of this work; the paper does not
> use it. The RQ3 script runs with that export switched on, so `SUMMARY.md` and `numbers.tex` report 66 ± 35 ms per
> change, and the JSON fields keep the old names (`po_new` = pairs verified per change, `po_reuse_rate` = verdicts
> reused). The paper's 54 ± 34 ms (median 52 ms, about 340×) is the same run without the export:
> `incremental_ms − eventb_gen_ms` in `results_rerun/rq3_incremental.json`.

Configuration lives in `configs/default.yaml` and `configs/experiments/*.yaml`. Set `ECD_RESULTS_DIR=<folder>` to
write everything (results, models, cache) somewhere other than `results/`, and `ECD_FRESH_CACHE=1` to recompute the
Z3 ground truth and overlap labels instead of using the cached copies next to each dataset.

## Command-line tool

```bash
ecd gen --n 1000 --out data/processed/synthetic_1k --hetero 0.2        # synthetic benchmark with alias ontology
ecd label data/processed/synthetic_1k/policy_v0.json --out data/processed/synthetic_1k/labels.json
ecd train data/processed/synthetic_1k/policy_v0.json data/processed/synthetic_1k/labels.json --out results/models/synthetic_1k.pt
ecd evolve data/processed/synthetic_1k --embedder gnn --model results/models/synthetic_1k.pt --mode exact --limit 20
ecd check data/raw/xacml/xpa/Experiments/itrust3/itrust3.xml            # a real XACML policy
```

## Datasets

```bash
bash scripts/download_datasets.sh && python benchmark/convert_real.py
```

| dataset | rules / conflicts | characteristics |
|---|---|---|
| `synthetic_1k` | 1 000 / 9 983 | 12 attributes (9 categorical × 12 values, 3 numeric), two core attributes and four atoms per rule; 5 % planted conflicts, 5 % arithmetic conditions, 20 % in a second vocabulary with an alias ontology |
| `complex_1k` | 1 000 / 9 785 | as above, 60 % cross-attribute arithmetic conditions |
| `hetero_1k_h00/h25/h50` | 1 000 / 11 775 | 10 % planted and arithmetic, a role hierarchy, 0 / 25 / 50 % in a second vocabulary |
| XPA XACML 3.0 | 3–210 / 0–214 | 8 policies: iTrust 64, HL7 22, gpms 210, pluto 176, conference 15, fedora 15, kmarket 3, Simpson 4 rules |
| AWS IAM merges | 239 / 240, 756 / 761 | merges of 100 and of 300 public managed policies, one rule per statement, wildcards expanded with the IAM action registry |
| Kubernetes roles | 138 / 0 | the 29 bootstrap ClusterRoles, allow-only |

Rules are counted after normalisation; conflicts are the exhaustive pairwise Z3 ground truth. Synthetic policies
come with 100 changes each, real policies with 50.

`data/processed/` is tracked and contains every converted dataset together with its cached Z3 ground truth
(`truth_conflicts.json`) and overlap labels (`labels.json`): `synthetic_1k`, `complex_1k`, `hetero_1k_h00/h25/h50`
and `real/{xacml_itrust, xacml_hl7, xacml_gpms, xacml_pluto, xacml_conference, xacml_fedora, xacml_kmarket,
xacml_simpson, aws_managed_100, aws_managed_300, k8s_clusterroles}` (the eleven real policies of the paper), plus
`real/xacml_obligation`, which is converted but not used in the paper. The raw third-party sources (`data/raw/`,
about 60 MB) are not tracked; the download script fetches them. See `benchmark/datasets/README.md`.

## Repository layout

```
ecd/
├── src/ecd/               # library (pip install -e .)
│   ├── core/              # Rule, Predicate, Policy, Delta, Ontology, interval abstraction
│   ├── parsers/           # XACML 3.0 / AWS IAM / Kubernetes RBAC → Policy
│   ├── xfg/               # rule feature graph (ontology-aware)
│   ├── embedding/         # GIN encoder, contrastive trainer, Z3 labeller, hash and SBERT baselines
│   ├── filtering/         # Tier A (sound pruning), Tier B (ranking), candidate selector
│   ├── verification/      # Z3 checker with counterexamples, combining-algorithm analysis (+ optional Event-B export)
│   ├── incremental/       # Impact(Δ), verdict ledger keyed by rule content hashes
│   ├── pipeline.py        # Detector: full_check() / apply(Δ)
│   └── cli.py             # the `ecd` command
├── benchmark/             # synthetic generator, convert_real.py, dataset notes
├── experiments/           # run_rq1..5, run_real, run_all, export/compare scripts, baselines/, metrics, stats
├── configs/               # default.yaml + experiments/*.yaml
├── data/processed/        # converted datasets with cached ground truth and labels
├── results_rerun/         # the run reported in the paper
├── docs/                  # architecture, formal definitions, data format
└── tests/                 # pytest suite with XACML / IAM fixtures
```

## Citation

```bibtex
@misc{minh2026ecd,
  title  = {Sound and Incremental Conflict Detection in Evolving Access Control Policies
            with Verifier-Supervised Graph Embeddings},
  author = {Minh, Duong-Nguyen and Van, Cuong-Nguyen and Thanh, Binh-Trinh and Anh, Tuan-Bui
            and Truong, Ninh-Thuan},
  year   = {2026},
  note   = {Code and data: \url{https://github.com/nminduo2k5/ecd-conflict-detection}}
}
```

## Licence

MIT for the code (see `LICENSE`). The converted policy datasets under `data/processed/` are derived from public
sources — the XPA XACML benchmark policies, AWS managed IAM policies (SummitRoute mirror) with the MAMIP action
registry, and the Kubernetes bootstrap ClusterRoles — and remain subject to the licences of those sources
(`benchmark/datasets/README.md`, `scripts/download_datasets.sh`).
