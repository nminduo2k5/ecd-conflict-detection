# results_rerun — chạy lại toàn bộ thực nghiệm, tách biệt với `results/`

> **Trạng thái 27/10/2026:** lần chạy lại này đã hoàn tất (xem `RUN_INFO.md`, `logs\`) và **là nguồn số liệu
> hiện tại của bài báo**: `numbers.tex` và `paper_figures\rq*.pdf` ở đây đã được chép sang `..\ecd_latex\`.
> `results/` (lần chạy 26–27/10) chỉ còn là lưu trữ; khác biệt giữa hai lần được ghi trong `COMPARE_vs_results.md`
> (533/754 macro trùng, phần còn lại là thời gian và các số phụ thuộc model GNN ±0.01; riêng RQ1 trên
> `hetero_1k_h50` lần cũ dùng phiên bản dataset trước khi sinh lại nên số cũ không hợp lệ).
> Sau lần chạy này, khởi tạo trọng số GNN đã được cố định seed (`common.train_gnn`, `run_rq1_embedding`),
> nên các lần chạy sau tái lập chính xác hơn; số GNN có thể lệch ±0.01 so với lần này.

Thư mục này dành cho việc **tự chạy lại** các RQ để kiểm chứng số liệu của bài báo.
Mọi thứ sinh ra khi chạy sẽ nằm ở đây; thư mục `results/` (nguồn số liệu hiện tại của
bài báo, đã được khoá dùng cho `ecd_latex/numbers.tex`) **không bị đụng tới**.

## Chạy

Mở **Command Prompt (cmd)** tại `H:\ecd-project\ecd` rồi:

```bat
experiments\run_all.cmd                       :: toàn bộ, kết quả ghi vào results_rerun\
experiments\run_all.cmd -FreshCache           :: tính lại cả ground truth Z3 và nhãn overlap (độc lập hoàn toàn, lâu hơn)
experiments\run_all.cmd -Steps rq2,rq2_complex   :: chạy lại một phần (model đã có trong results_rerun\models)
```

`run_all.cmd` gọi `run_all.ps1` với `-ExecutionPolicy Bypass` cho riêng tiến trình đó, nên không cần đổi
execution policy của máy (PowerShell mặc định chặn `.ps1`: "running scripts is disabled"). Trong PowerShell
cũng dùng được: `.\experiments\run_all.cmd ...`, hoặc `powershell -ExecutionPolicy Bypass -File experiments\run_all.ps1 ...`.

Script đặt biến môi trường `ECD_RESULTS_DIR=<thư mục này>` rồi chạy lần lượt các bước; mỗi bước
có log riêng trong `logs\` và được ghi vào `RUN_INFO.md` (máy, phiên bản thư viện, giờ bắt đầu,
thời lượng, mã thoát). Bước nào lỗi thì dừng; sửa xong chạy lại bằng `-Steps` từ bước đó.

Muốn chạy tay từng script (bất kỳ shell nào), chỉ cần đặt biến môi trường trước:

```powershell
$env:ECD_RESULTS_DIR = "H:\ecd-project\ecd\results_rerun"   # bắt buộc, nếu không sẽ ghi vào results\
$env:ECD_FRESH_CACHE = "1"                                   # tuỳ chọn
.\.venv\Scripts\python.exe experiments\run_rq1_embedding.py
```

## Thứ tự và thời gian ước lượng (laptop 4 nhân, không GPU)

| bước | script | ghi ra | ước lượng |
|---|---|---|---|
| rq1 | `run_rq1_embedding.py` | `rq1_embedding.json`, `models\*.pt` (synthetic_1k, hetero_1k_h50, aws_managed_300, xacml_gpms, xacml_pluto) | ~20–30 phút |
| train_complex | `train_model.py data/processed/complex_1k` | `models\complex_1k.pt` | ~5 phút |
| rq2 | `run_rq2_pruning.py` | `rq2_pruning.json` | ~15 phút |
| rq2_complex | `run_rq2_pruning.py rq2_complex` | `rq2_complex.json` | ~15 phút |
| rq3 | `run_rq3_incremental.py` | `rq3_incremental.json`, `eventb\rq3\PolicyCtx_v*.{ctx,buc}` | ~5 phút |
| rq5 | `run_rq5_hetero.py` | `rq5_hetero.json`, `models\hetero_1k_h00.pt`, `hetero_1k_h25.pt` | ~30 phút |
| real | `run_real.py` | `real_policies.json` | ~2 phút |
| rq4 | `run_rq4_scalability.py` | `rq4_scalability.json` | **~1,5–2 giờ** (100k luật × 30 seed) |
| summarize | `summarize.py` | `SUMMARY.md` | giây |
| export | `export_numbers.py <đây>\numbers.tex` | `numbers.tex` (macro LaTeX) | giây |
| figures | `paper_figures.py <đây>\paper_figures` | `paper_figures\rq1..rq5.pdf` | giây |
| plots | `plot_results.py` | `figures\*.png` | giây |

Với `-FreshCache` cộng thêm ground truth Z3 (≈2 phút mỗi policy 1k, vài phút cho AWS-300)
và 20 000 nhãn Z3 mỗi dataset huấn luyện; cache mới nằm trong `cache\<dataset>\`.
Không có `-FreshCache` thì dùng lại `truth_conflicts.json` / `labels.json` cạnh dataset
(kết quả tất định: Z3 vét cạn + lấy mẫu theo seed), chỉ model GNN và mọi phép đo là chạy lại.

## So sánh với số liệu bài báo

```powershell
.\.venv\Scripts\python.exe experiments\compare_results.py results results_rerun
```

In ra và ghi `results_rerun\COMPARE_vs_results.md`: macro nào khác nhau, tách riêng nhóm
**thời gian** (ms, s, speed-up — phụ thuộc máy, khác là bình thường) và nhóm **phi thời gian**
(số cặp, recall, AUC, PO, p-value, Â12 — phải khớp trong ±2 %, có thể chỉnh `--tol`).
Nhóm có thể lệch nhẹ dù cùng seed: AUC/recall của GNN (PyTorch trên CPU không hoàn toàn tất định,
±0,01), p-value/Â12 của RQ4 (phụ thuộc thời gian đo).

## Đưa kết quả mới vào bài

Khi đã hài lòng: chép `numbers.tex` ở đây đè lên `..\ecd_latex\numbers.tex`, chép
`paper_figures\rq*.pdf` vào `..\ecd_latex\figures\`, biên dịch lại bài. Hoặc giữ nguyên
bài và chỉ dùng `COMPARE_vs_results.md` làm bằng chứng tái lập.
