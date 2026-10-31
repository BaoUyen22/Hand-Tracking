# Nhóm Hihi · Nhận dạng Ngôn ngữ Ký hiệu ASL (realtime, word-level)

Mô hình cuối: **GRU-B (Bi-GRU + Multi-Head Attention)**, 168 điểm hand-only, **131 lớp** (gộp biến thể cùng nghĩa).
Kết quả test (signer-grouped held-out, **n = 1823**): **top-1 = 0.730 · top-5 = 0.916 · macro-F1 = 0.725**.
Inference CPU (ONNX) ~**2.4 ms**. Seed cố định = **42**.

---

## 0. Môi trường

- **Python 3.11**. Khuyến nghị tạo virtualenv riêng.

```bash
# tạo & kích hoạt venv (Windows)
python -m venv .venv
.venv\Scripts\activate

# cài thư viện
pip install -r requirements.txt
```

Thư viện chính: `mediapipe==0.10.14`, `tensorflow>=2.17`, `onnxruntime`, `scikit-learn`, `numpy 2.x`, `pandas`, `matplotlib`, `seaborn`.

> Mọi notebook/script **tự định vị PROJECT_ROOT** (thư mục chứa `scripts/`, `models/`, `data/`) nên không hardcode đường dẫn máy cá nhân. Đặt thư mục `Nhom19_Tuan5/` ngay trong gốc project (cạnh `scripts/`, `models/`, `data/`).

---

## 1. Cấu trúc thư mục nộp

```
Nhom19_Tuan5/
├── README.md                 # file này
├── requirements.txt          # thư viện + version
├── build_notebooks.py        # sinh lại 4 notebook (tái lập)
├── verify_consistency.py     # kiểm tra số liệu nhất quán (PASS/FAIL)
├── notebooks/
│   ├── 01_final_data_pipeline.ipynb    # pipeline dữ liệu (chạy đầu→cuối)
│   ├── 02_final_model.ipynb            # kết quả + so sánh + verify CM
│   └── 03_error_analysis_demo.ipynb    # phân tích lỗi
├── demo/
│   └── demo.ipynb            # demo offline (không cần webcam)
├── src/
│   └── predict.py            # demo CLI: 1 file .npy → top-5
├── data_sample/
│   └── sample_input.npy      # 1 mẫu landmark (30,168) để demo
├── results/
│   ├── final_metrics.csv             # bảng 3 mức (số chính thức)
│   ├── final_results.csv             # so sánh + ưu/hạn chế
│   ├── classification_report.csv     # per-class P/R/F1 (khớp CM)
│   ├── confusion_matrix.csv          # CM thô 131×131
│   ├── error_analysis.csv            # 15 mẫu sai thật
│   └── figures/
│       ├── confusion_matrix.png
│       └── confusion_top_zoom.png
├── report/
│   └── BaoCao_Tuan5_Nhom19.md        # báo cáo (xuất .docx khi nộp)
└── slides/
    └── outline_week5_Nhom19.md       # dàn ý thuyết trình
```

---

## 2. Chạy nhanh

```bash
# (1) Kiểm tra số liệu nhất quán — phải toàn [PASS]
python verify_consistency.py

# (2) Demo offline (không cần webcam)
python src/predict.py
# hoặc mở demo/demo.ipynb → Run All

# (3) App realtime (cần webcam) — chạy từ gốc project
python scripts/app_unified_webcam.py --model gru
#   phải thấy log: "Extractor: HandPoseExtractor (hands-only 168, train/serve-matched)"

# (4) Sinh lại notebooks (tuỳ chọn)
python build_notebooks.py
```

Mở 3 notebook trong `notebooks/` rồi **Run All** — mỗi notebook chạy được từ đầu đến cuối và đọc số liệu trực tiếp từ artifact thật.

---

## 3. Bản đồ tái lập (từ dữ liệu gốc → kết quả)

Các script nằm ở gốc project trong `scripts/`:

| Bước | Lệnh | Output |
|---|---|---|
| 1. Dựng manifest split | `python scripts/prepare_138.py` | `data/splits_138/manifest_138.csv` |
| 2. Trích landmark 168 | `python scripts/extract_138.py` | `data/landmarks_138/hand168/vid*.npy` |
| 3. Train GRU-B (gộp 131) | `python scripts/train_gru_138.py --merge` | `models/gru_138_b_hand168_merged_final.h5` (+ config, labels) |
| 4. Baseline RF | `python scripts/train_rf_138.py` | `models/rf_138_merged_metrics.json` |
| 5. Export ONNX | `python scripts/convert_h5_to_onnx.py --input models/gru_138_b_hand168_merged_final.h5 --output models/gru_138_b_hand168_merged.onnx` | `.onnx` |
| 6. Báo cáo/eval | `python scripts/eval_report_138.py` | `results_week5/*` |

> Bước 2 (~100 phút) và 3 (~21 phút) nặng nhưng resume-safe / seed cố định = 42.

---

## 4. Số liệu chính thức (đã chạy thật — dùng đúng ở mọi nơi)

| Mô hình | Vai trò | Lớp | top-1 | top-5 | macro-F1 | infer (CPU) |
|---|---|---|---|---|---|---|
| RandomForest (flatten 5040) | Baseline | 131 | 0.4493 | 0.7202 | 0.4445 | 0.24 ms |
| GRU-B (138 raw) | Phương pháp chính | 138 | 0.7208 | 0.9002 | 0.7114 | ~2.4 ms |
| **GRU-B (131 merged)** | **Tốt nhất** | **131** | **0.7301** | **0.9161** | **0.7250** | **2.4 ms** |

Dataset: ASL Citizen, **split theo người ký** (signer-grouped, không overlap participant) — train 34 / val 5 / test 11 người ký.

---

## 5. Ghi chú quan trọng (bug train/serve đã sửa)

Train dùng `HandPoseExtractor` (MediaPipe **Hands**). Nếu serve bằng MediaPipe **Holistic** thì accuracy sụp về ~1.3% (so với 78.7% khi đúng extractor). App đã được khoá `serve_extractor="hands168"` để dùng đúng extractor — xem `models/deploy_log_138_merged.json`.
