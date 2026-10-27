# Roadmap / TODO

## Done (Oct 2026)
- [x] Alias/ontology map (`core.Ontology`): canonicalisation in Tier A, Z3, Event-B; ontology edges in the XFG;
      dedicated experiment RQ5 (`experiments/run_rq5_hetero.py`).
- [x] Role hierarchy (value hierarchy expansion before Tier A / Z3 / Event-B).
- [x] GNN: train/inference on the same per-rule XFG, graph-level readout, numeric interval features,
      mini-batched contrastive training, rule-level hold-out, save/load with config.
- [x] Event-B: versioned constants for modified rules, `.buc` emitter, Rodin project layout + headless Ant
      build + `.bps/.bpo` parsing (`ecd rodin`). Not executed here (no Rodin install on the machine).
- [x] XACML parser: PolicySet nesting, DNF sub-rules, Condition translation (and/or/not, comparisons,
      arithmetic, membership), untranslatable → over-approximation flagged `approx`.
- [x] Real datasets converted (XPA XACML set, AWS managed policies, k8s ClusterRoles) — `benchmark/convert_real.py`.
- [x] RQ4 up to 100k rules × 30 repetitions.

## Must-have before camera-ready
- [ ] Run the Rodin headless build on a machine with Rodin 3.x + SMT plugin and replace the Z3-only
      PO verdicts in RQ3 with Rodin's `.bps` status (the project folder is already generated).
- [ ] RQ1 hyper-parameter sweep (dim, layers, τ, epochs) and more seeds (currently 1 seed per dataset).
- [ ] Cross-attribute comparisons in XACML conditions (`any-of-any(equal, attrA, attrB)`) need a shared
      value universe in the Z3 encoding; currently over-approximated (4 HL7 rules).

## Nice-to-have
- [ ] Parallel SMT (multiprocessing per target rule) for the full check.
- [ ] HNSW deletion: periodic rebuild instead of tombstones; use `VectorIndex.search` for top-k instead of
      brute-force ranking of Tier A survivors.
- [ ] Redundancy detection (same-effect overlaps) as a secondary report.
- [ ] Amazon Employee Access (Kaggle) rule mining.
