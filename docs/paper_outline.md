# Paper outline (20–25 pages) ↔ artefacts

1. Introduction — motivation, illustrative example, C1–C4
2. Background — XACML/ABAC, Event-B, XFG, embeddings/GNN
3. Motivating example — healthcare policy evolving over 3 versions
   → `tests/conftest.py::hospital` + `tests/fixtures/sample_xacml.xml`; extend to v1, v2 with deltas
4. Approach — §3 pipeline + §4 formal definitions → `docs/architecture.md`, `docs/formal_definitions.md`
5. Implementation — tool (`ecd` CLI), Rodin plugin, FAISS
6. Evaluation — RQ1 `run_rq1_embedding.py`, RQ2 `run_rq2_pruning.py`, RQ3 `run_rq3_incremental.py`, RQ4 `run_rq4_scalability.py`
   Baselines: full SMT, exact index, full Event-B, SBERT, published XACML conflict detectors
   Stats: 30 runs, Wilcoxon, A12 (`experiments/stats.py`)
7. Discussion & threats to validity
8. Related work — policy conflict detection, ML for access control, incremental verification
9. Conclusion
