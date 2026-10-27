# ECD — Verifier-Supervised Graph Embeddings for Sound and Incremental Conflict Detection

Code, datasets and results for the paper **"Verifier-Supervised Graph Embeddings for Sound and Incremental
Conflict Detection in Evolving Access Control Policies"** (Cuong Nguyen Van, Duong Nguyen Minh — School of
Computing, Phenikaa University).

**In one sentence:** a GNN rule embedding, trained contrastively on overlap labels produced by an SMT solver,
*orders and narrows* the rule pairs that an exact verifier (Z3, Event-B context extensions) checks — it never
decides a conflict itself. A sound symbolic pre-filter (Tier A, Theorem 1) guarantees 100 % recall in exact mode;
each policy change becomes an Event-B context extension whose proof obligations are confined to the affected
pairs (Theorem 2). Policies: XACML 3.0, AWS IAM, Kubernetes RBAC, plus a synthetic generator (1k–100k rules,
heterogeneous vocabularies with an alias/hierarchy ontology).

**Reproduce the paper's numbers** (Windows, Python ≥ 3.10; ~1 h on a 4-core laptop, RQ4 is the long step):

```bat
python -m venv .venv && .venv\Scripts\activate && pip install -e ".[ml,dev]"
experiments\run_all.cmd -ResultsDir results_mine        :: all RQs into a fresh folder (per-step logs + RUN_INFO.md)
python experiments\compare_results.py results_rerun results_mine   :: macro-by-macro comparison with the paper's run
```

`results_rerun/` holds the run the paper reports (JSON per RQ, `RUN_INFO.md`, logs, `SUMMARY.md`, `numbers.tex`
with every LaTeX macro, vector figures). Deterministic numbers (pair counts, pruning, recall of the exact mode and
of the hash/index baselines, proof obligations, real-policy counts) reproduce exactly; GNN-dependent numbers vary by
about ±0.01; timings depend on the machine. Trained GNN models (8 × 25 MB) are not tracked — `run_all` retrains them.

**Licence:** MIT for the code (see `LICENSE`). The converted policy datasets under `data/processed/` are derived
from public sources — the XPA XACML benchmark policies, AWS managed IAM policies (SummitRoute mirror) with the
MAMIP action registry, and the Kubernetes bootstrap ClusterRoles — and remain subject to the licences of those
sources (`benchmark/datasets/README.md`, `scripts/download_datasets.sh`).
The remainder of this README is in Vietnamese.

---

Mã nguồn cho bài báo *Verifier-Supervised Graph Embeddings for Sound and Incremental Conflict Detection in
Evolving Access Control Policies*.

**Ý tưởng một câu:** embedding **không quyết định** xung đột, nó **dẫn đường** cho bước
kiểm chứng. Tầng A (lọc ký hiệu an toàn, Định lý 1) bảo đảm không mất recall; tầng B (embedding)
sắp thứ tự/thu hẹp cặp cần kiểm; Z3 + Event-B kiểm chứng chính xác; mỗi Δ chỉ sinh PO cho
vùng ảnh hưởng (Định lý 2). Mọi bước chính xác làm việc trên **dạng chuẩn hóa theo ontology**
(alias `jobTitle ≡ role`, phân cấp `senior_doctor ⊑ doctor`), còn embedding phải *học* quan hệ đó.

```
Policy P + Δ + ontology ──[1] normalise (DNF, canonical form) ──[2] XFG → GNN embedding
            ──[3] Tier A prune → Tier B rank ──[4] Z3 / Event-B (refinement per Δ, .ctx + .buc)
            ──[5] conflicts + counterexample + combining-algorithm analysis
```

## Trạng thái

| Thành phần | File | Trạng thái |
|---|---|---|
| Mô hình luật / chính sách / Δ / **ontology** (§4) | `src/ecd/core/` | ✅ test |
| Parser XACML 3.0 (PolicySet lồng nhau, DNF, Condition), AWS IAM (wildcard theo action registry), K8s RBAC | `src/ecd/parsers/` | ✅ chạy trên 20 policy thật, không crash; điều kiện không dịch được → over-approx có cờ `approx` |
| XFG builder (ontology edges, numeric interval features) | `src/ecd/xfg/` | ✅ |
| **C1** GNN (GIN + sum-pooled readout) + contrastive + nhãn Z3 | `src/ecd/embedding/gnn.py`, `labeler.py` | ✅ train/inference cùng đồ thị; hold-out theo luật |
| **C2** Tầng A (Pruner, cache abstraction) + Tầng B + 2 chế độ + ablation không Tầng A | `src/ecd/filtering/` | ✅ test soundness so với Z3 (có ontology) |
| SMT checker (Z3, ontology-aware) + phản ví dụ | `src/ecd/verification/smt_checker.py` | ✅ |
| Phân tích combining algorithm | `src/ecd/verification/combining.py` | ✅ |
| **C3** Event-B: ledger refinement theo Δ, hằng số versioned cho MODIFY, `.ctx` + `.buc`, Rodin project + headless build + parse `.bps/.bpo` | `src/ecd/verification/eventb/` | ✅ sinh file; **Rodin chưa chạy trên máy này** (không cài Rodin) |
| **C4** Benchmark tổng hợp 1k→100k (+ alias vocab, hierarchy) + dữ liệu thật | `benchmark/` | ✅ |
| Thí nghiệm RQ1–RQ5 + real, Wilcoxon/A12 | `experiments/` | ✅ |

## Cài đặt

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -e ".[dev]"                                # z3-solver, numpy, pyyaml, typer, scipy, pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu   # GNN (C1)
pip install faiss-cpu sentence-transformers            # tùy chọn: ANN index, baseline SBERT
pytest                                                 # 38 tests (GNN tests bỏ qua nếu thiếu torch)
```

## Chạy nhanh

```bash
ecd gen --n 1000 --out data/processed/synthetic_1k --hetero 0.2          # benchmark + alias ontology
ecd label data/processed/synthetic_1k/policy_v0.json --out data/processed/synthetic_1k/labels.json
ecd train data/processed/synthetic_1k/policy_v0.json data/processed/synthetic_1k/labels.json --out results/models/synthetic_1k.pt
ecd evolve data/processed/synthetic_1k --embedder gnn --model results/models/synthetic_1k.pt --mode exact --limit 20 --eventb-out results/eventb/demo
ecd check data/raw/xacml/xpa/Experiments/itrust3/itrust3.xml              # policy thật
ecd rodin results/eventb/demo --project ECD [--build --rodin-path /opt/rodin/rodin]
```

## Dữ liệu thật

```bash
bash scripts/download_datasets.sh && python benchmark/convert_real.py
```
→ `data/processed/real/{xacml_itrust, xacml_hl7, xacml_gpms, xacml_pluto, xacml_conference, xacml_fedora,
xacml_kmarket, xacml_obligation, xacml_simpson, aws_managed_100, aws_managed_300, k8s_clusterroles}`
(xem `benchmark/datasets/README.md`).

## Thí nghiệm

```bash
cd experiments
python run_rq1_embedding.py    # AUC/P@k của khoảng cách embedding vs nhãn overlap Z3, luật chưa thấy; ablation SBERT/hash/hash+ont/GNN
python run_rq2_pruning.py      # tỉ lệ cắt giảm, recall (exact = 100%), ablation không Tầng A, top-k, số lần gọi Z3 tới xung đột đầu tiên
python run_rq3_incremental.py  # PO mới / PO kế thừa mỗi Δ (từ context Event-B), thời gian Δ vs kiểm lại toàn bộ
python run_rq4_scalability.py  # 1k → 100k luật, 30 seed, exact vs approximate, Wilcoxon + A12
python run_rq5_hetero.py       # thuộc tính không đồng nhất + phân cấp: index chính xác vs embedding
python run_real.py             # 11 bộ policy thật
python plot_results.py         # hình vào results/figures/
```
Thêm: `python run_rq2_pruning.py rq2_complex` (RQ2 trên `complex_1k`, 60 % luật có điều kiện số liên thuộc tính),
`python train_model.py <dataset>` (gán nhãn + train GNN cho một dataset), `python export_numbers.py ../ecd_latex/numbers.tex`
(sinh macro số liệu cho bài LaTeX trong `../ecd_latex`), `python paper_figures.py` (hình vector rq1..rq5.pdf cho bài,
ghi vào `../ecd_latex/figures/`).
Cấu hình trong `configs/experiments/*.yaml`; kết quả JSON trong `results/` (mỗi RQ một file: `rq1_embedding.json`,
`rq2_pruning.json`, `rq2_complex.json`, `rq3_incremental.json`, `rq4_scalability.json`, `rq5_hetero.json`,
`real_policies.json`); tóm tắt số liệu trong `results/SUMMARY.md` (`python summarize.py`), hình trong `results/figures/`.

**Chạy lại tách biệt:** đặt `ECD_RESULTS_DIR=<thư mục khác>` (tuyệt đối hoặc tương đối so với `ecd/`) thì mọi script
ghi vào đó thay vì `results/` (kể cả `models/`, `eventb/`); thêm `ECD_FRESH_CACHE=1` để tính lại ground truth và nhãn Z3
vào `<thư mục>/cache/`. `experiments\run_all.ps1` chạy tuần tự mọi bước vào `results_rerun/` (log + `RUN_INFO.md`),
`python experiments/compare_results.py results results_rerun` đối chiếu từng macro số liệu giữa hai thư mục
(xem `results_rerun/README.md`).

## Cấu trúc thư mục

```
ecd/
├── src/ecd/               # thư viện chính (pip install -e .)
│   ├── core/              # Rule, Predicate, Policy, Delta, Ontology, interval abstraction
│   ├── parsers/           # XACML / AWS IAM / K8s RBAC → Policy
│   ├── xfg/               # đồ thị XFG (ontology-aware)
│   ├── embedding/         # Embedder, GNN+contrastive, labeler Z3, FAISS index, baselines
│   ├── filtering/         # Tầng A (Pruner) + Tầng B (ranking) + CandidateSelector
│   ├── verification/      # Z3 checker, combining analysis, eventb/ (generator, ledger, rodin)
│   ├── incremental/       # Impact(Δ), PO cache
│   ├── pipeline.py        # Detector: full_check() / apply(Δ)
│   └── cli.py             # lệnh `ecd`
├── benchmark/             # generator tổng hợp, convert_real.py, hướng dẫn dataset
├── experiments/           # run_rq1..5, run_real, baselines/, metrics.py, stats.py, plot_results.py
├── eventb/                # hướng dẫn Rodin, template .buc
├── configs/               # default.yaml + experiments/*.yaml
├── docs/                  # architecture, formal_definitions, data_format, paper_outline, roadmap
├── tests/                 # pytest (+ fixtures XACML/IAM)
├── data/raw | processed   # dữ liệu
└── results/               # kết quả JSON, models/, eventb/, figures/
```

Việc còn lại: `docs/roadmap.md`.
