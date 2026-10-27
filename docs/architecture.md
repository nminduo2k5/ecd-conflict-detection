# Architecture (paper §3)

```
Policy P (XACML / IAM / RBAC / native JSON)  +  Δ ∈ {add, remove, modify}  +  ontology O
        │
 [1] ecd.parsers          → ecd.core.Policy   (rule = (T, C, eff), atoms over 𝒜, raw_expr for complex conditions;
     ecd.core.ontology       XACML targets/conditions put in DNF → one sub-rule per disjunct)
     ecd.xfg.builder      → XFG per rule (private leaves, ontology edges, numeric interval features)
        │
 [2] ecd.embedding        → e(r) ∈ R^d
     gnn.XFGEncoder (GIN + sum-pooled jumping-knowledge readout) + train_contrastive
                          (labels from embedding.labeler via ontology-aware Z3)
     baselines.HashedAtomEmbedder [use_ontology] / SBertEmbedder  (ablation)
     index.VectorIndex    (FAISS HNSW or exact numpy)
        │
 [3] ecd.filtering.CandidateSelector(target r, others)
     Tier A  sound_pruning.Pruner   — same effect | provably disjoint attribute on ⟦r⟧   (Theorem 1)
     Tier B  ranking.EmbeddingRanker — order survivors by -‖e(r)-e(r')‖
     Mode.EXACT: verify all survivors in that order      Mode.APPROXIMATE: verify top-k only
     use_tier_a=False → ablation "without Tier A"
        │
 [4] ecd.verification
     smt_checker.SMTChecker.check_pair → sat/unsat + counterexample request q   (on ⟦r⟧)
     eventb.EventBLedger                → PolicyCtx_v{N} extends PolicyCtx_v{N-1} (.ctx + .buc),
                                          versioned constants for modified rules, theorem per pair
     eventb.RodinRunner                 → Rodin project + headless Ant build + .bps/.bpo parsing
     incremental.cache.POCache          → discharged POs keyed by (ids, content hashes)
     incremental.impact.compute_impact  → touched rules, invalidated pairs                  (Theorem 2)
        │
 [5] ecd.pipeline.Detector → RunReport(conflicts[+counterexample +combining verdict], pruning, PO reuse,
                                        timings, SMT calls / time to first conflict)
```

## Where each contribution lives

| | Module |
|---|---|
| C1 overlap-aware embedding | `embedding/gnn.py`, `embedding/labeler.py`, `xfg/builder.py` |
| C2 two-tier filtering + soundness | `filtering/sound_pruning.py` (proof sketch in docstring), `core/intervals.py`, `core/ontology.py` |
| C3 incremental Event-B | `verification/eventb/refinement.py` (ledger), `generator.py` (.ctx/.buc), `rodin_runner.py`, `incremental/` |
| C4 benchmark + tool | `benchmark/generator/synthetic.py`, `benchmark/convert_real.py`, `cli.py` |

## Invariants worth keeping

* `Predicate` is always a **conjunction** (atoms ∧ raw_expr). Disjunctions are split into
  multiple rules by the parser. This is what keeps Tier A sound.
* Every exact step reasons about the **canonical** predicate `space.applicability(r)`;
  the embedding sees the raw vocabulary plus ontology *edges* (it must learn the rest).
* `Rule.content_hash()` is the identity used by the PO cache; any semantic change to a rule
  must change the hash.
* The embedding is never consulted for the *verdict*; grep for `verdict`/`conflict` in
  `filtering/` — there must be none.
* XFG RULE/TARGET/COND nodes carry no per-rule identity token (otherwise the GNN memorises
  a lookup table and does not generalise to unseen rules — the bug behind the first RQ1 run).
