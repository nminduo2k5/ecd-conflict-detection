# ECD — Verifier-Supervised Graph Embeddings for Sound and Incremental Conflict Detection

Code, datasets and results for the paper **"Verifier-Supervised Graph Embeddings for Sound and Incremental
Conflict Detection in Evolving Access Control Policies"** (Duong Nguyen Minh, Cuong Nguyen Van, Binh Trinh Thanh —
Phenikaa University; Nguyen Viet Ha — VNU University of Engineering and Technology).

## Idea in one paragraph

A *conflict* is a pair of rules that apply to the same request and render opposite decisions. ECD lets a learned
embedding **guide, but never replace**, exact verification:

* **Tier A — sound symbolic pruning.** Per-attribute abstractions discard every pair that provably cannot conflict
  (Theorem 1). Nothing is lost: the exact mode keeps 100 % recall.
* **Tier B — overlap-aware embedding.** A graph neural network (GIN over a rule feature graph) is trained
  contrastively on *overlap labels produced by Z3*, so its distance reflects whether two rules can apply to the
  same request rather than whether they read alike. It only orders the surviving pairs (exact mode) or truncates
  them to the top-*k* per rule (approximate mode, with the empirical recall reported).
* **Exact verification.** Every candidate pair is checked by Z3; a satisfiable query yields a counterexample
  request and the pair is classified under the combining algorithm (masked, order-dependent or real).
* **Incremental verification.** A policy change only re-verifies the pairs that touch the changed rule; a
  proof-obligation ledger keeps the verdicts of all other pairs valid across versions (Theorem 2), so a change
  costs milliseconds instead of a full re-check.
* **Ontology.** Alias attributes and value hierarchies (`jobTitle ≡ role`, `senior_doctor ⊑ doctor`) are
  canonicalised before every exact step; the embedding has to *learn* the same relations from the solver labels.

Supported policy languages: XACML 3.0, AWS IAM (wildcards expanded with the IAM action registry) and Kubernetes
RBAC, plus a synthetic generator (1k–100k rules, planted conflicts, arithmetic conditions, heterogeneous
vocabularies with an alias/hierarchy ontology, change sequences).

```
policy P + change Δ + ontology O
   ──[1] normalise (DNF sub-rules, canonical form, rule feature graph)
   ──[2] embed (GIN, contrastive training on Z3 overlap labels)
   ──[3] select (Tier A sound pruning → Tier B ordering / top-k)
   ──[4] verify (Z3: satisfiability + counterexample; PO ledger per version)
   ──[5] report (conflict pairs, counterexample requests, combining-algorithm class)
```

## Installation

Python ≥ 3.10 (the reported runs used Python 3.13 on Windows 10).

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
(machine, package versions, step durations), `SUMMARY.md`, `numbers.tex` (every number quoted in the paper as a
LaTeX macro), the generated result tables and the vector figures.

```bat
experiments\run_all.cmd -ResultsDir results_mine                     :: every step into a fresh folder (~1 h, RQ4 is the long step)
experiments\run_all.cmd -ResultsDir results_mine -Steps rq1,rq2       :: a subset of the steps
python experiments\compare_results.py results_rerun results_mine     :: macro-by-macro comparison with the reported run
```

Steps, in order: `rq1`, `train_complex`, `rq2`, `rq2_complex`, `rq3`, `rq5`, `real`, `rq4`, `summarize`, `export`,
`tables`, `figures`, `plots`. Deterministic numbers (pair counts, pruning ratios, recall of the exact mode and of the
hash/index baselines, proof-obligation counts, real-policy counts) reproduce exactly. Numbers that depend on the
trained GNN vary by about ±0.01 between runs; timings depend on the machine. The trained GNN models (8 × 25 MB) are
not tracked; `run_all` retrains them. RQ4 checkpoints after every seed and resumes if interrupted.

Every experiment script also runs on its own from `experiments/`:

| script | research question | output |
|---|---|---|
| `run_rq1_embedding.py` | RQ1 — is the embedding overlap-aware? AUC of the embedding distance against Z3 overlap labels on held-out rules; SBERT / hash / hash+ontology / untrained-GNN ablations | `rq1_embedding.json` |
| `run_rq2_pruning.py` [`rq2_complex`] | RQ2 — pairs pruned, recall, time, solver calls before the first conflict; full-SMT and literal-index baselines; `complex_1k` variant with 60 % cross-attribute arithmetic conditions | `rq2_pruning.json`, `rq2_complex.json` |
| `run_rq3_incremental.py` | RQ3 — proof obligations per change versus full regeneration; change time versus from-scratch re-verification | `rq3_incremental.json` |
| `run_rq4_scalability.py` | RQ4 — cost of one change for 1k–100k rules, 30 seeds, exact versus approximate mode, Wilcoxon test and Â12 | `rq4_scalability.json` |
| `run_rq5_hetero.py` | RQ5 — heterogeneous vocabularies and role hierarchy: literal index versus Tier A versus embedding orders | `rq5_hetero.json` |
| `run_real.py` | eleven real policies (XPA XACML set, AWS managed IAM merges, Kubernetes ClusterRoles) | `real_policies.json` |
| `summarize.py`, `export_numbers.py`, `export_tables.py`, `paper_figures.py`, `plot_results.py` | summary, LaTeX macros, result tables, paper figures, diagnostic plots | `SUMMARY.md`, `numbers.tex`, `paper_tables/`, `paper_figures/`, `figures/` |

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

`data/processed/` is tracked and contains every converted dataset together with its cached Z3 ground truth
(`truth_conflicts.json`) and overlap labels (`labels.json`): `synthetic_1k`, `complex_1k`, `hetero_1k_h00/h25/h50`
and `real/{xacml_itrust, xacml_hl7, xacml_gpms, xacml_pluto, xacml_conference, xacml_fedora, xacml_kmarket,
xacml_obligation, xacml_simpson, aws_managed_100, aws_managed_300, k8s_clusterroles}`. The raw third-party sources
(`data/raw/`, about 60 MB) are not tracked; the download script fetches them. See `benchmark/datasets/README.md`.

## Repository layout

```
ecd/
├── src/ecd/               # library (pip install -e .)
│   ├── core/              # Rule, Predicate, Policy, Delta, Ontology, interval abstraction
│   ├── parsers/           # XACML 3.0 / AWS IAM / Kubernetes RBAC → Policy
│   ├── xfg/               # rule feature graph (ontology-aware)
│   ├── embedding/         # GIN encoder, contrastive trainer, Z3 labeller, hash and SBERT baselines
│   ├── filtering/         # Tier A (sound pruning), Tier B (ranking), candidate selector
│   ├── verification/      # Z3 checker with counterexamples, combining-algorithm analysis, versioned contexts + PO ledger
│   ├── incremental/       # impact of a change, PO cache
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

## Licence

MIT for the code (see `LICENSE`). The converted policy datasets under `data/processed/` are derived from public
sources — the XPA XACML benchmark policies, AWS managed IAM policies (SummitRoute mirror) with the MAMIP action
registry, and the Kubernetes bootstrap ClusterRoles — and remain subject to the licences of those sources
(`benchmark/datasets/README.md`, `scripts/download_datasets.sh`).
