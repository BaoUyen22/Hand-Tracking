# 🖐️ Hand-Tracking — Nhận diện 34 ký hiệu bàn tay tĩnh

**Đồ án môn học:** Máy học (Machine Learning)  
**Mục tiêu:** Xây dựng hệ thống nhận diện **34 ký hiệu tĩnh** (10 chữ số `0–9` và 24 chữ cái `A–Y`, không gồm `J`, `Z`) từ webcam, ưu tiên chạy trên CPU.  
**Công nghệ:** Python 3.11 · OpenCV · MediaPipe Hands · NumPy · TensorFlow/Keras (MLP) · ONNX Runtime.

## 1. Tổng quan

Hệ thống sử dụng **MediaPipe Hands** để xác định 21 điểm mốc (landmarks) trên bàn tay, chuyển chúng thành vector đặc trưng để mô hình **MLP** phân loại ký hiệu. Mô hình sau huấn luyện dự kiến được xuất sang **ONNX** để chạy trong ứng dụng webcam.

**Luồng xử lý dữ liệu:**

```text
Ảnh ASL-HG Raw
  → Kiểm kê, lọc ảnh và xử lý trùng lặp
  → Rà soát thủ công, chốt manifest
  → Chia Train / Validation / Test theo người
  → MediaPipe Hands → chuẩn hóa đặc trưng 63 chiều
  → Kiểm tra dữ liệu → bàn giao cho huấn luyện MLP
  → Xuất ONNX → ứng dụng webcam (giai đoạn tích hợp)
```

Dữ liệu hiện tại được chia theo **Participant**, không chia ngẫu nhiên từng ảnh:

| Tập | Participant | Số mẫu trích xuất thành công |
|---|---|---:|
| Train | P1–P6 | 19.953 |
| Validation | P7–P8 | 6.735 |
| Test | P9–P10 | 6.651 |
| **Tổng** | | **33.339** |

Bộ đặc trưng hiện tại có **63 chiều** (`21 landmarks × 3 tọa độ x, y, z`) và chỉ nhận ảnh phát hiện **đúng một bàn tay**. Việc hỗ trợ đầy đủ ký hiệu `0` theo quy ước hai tay cần được kiểm chứng thêm. Kết quả kiểm tra cấu trúc dữ liệu và rò rỉ giữa các tập hiện tại: **PASS**; đây **không phải** độ chính xác của mô hình phân loại.

## 2. Cấu trúc dự án
```text
Hand-Tracking/
├── data/
│   ├── raw/                         # Ảnh nguồn (không đưa lên Git)
│   └── processed/                   # X/y/sample_ids theo split, classes.json, metadata.json
├── manifests/                       # Manifest, kết quả review, split và thống kê
├── scripts/
│   ├── prepare_data.py              # Kiểm kê, lọc chất lượng, phát hiện trùng
│   ├── apply_reviews.py             # Áp dụng quyết định kiểm tra thủ công
│   ├── split_data.py                # Chia tập theo Participant
│   ├── extract_features.py          # Trích xuất đặc trưng MediaPipe 63D
│   ├── validate_handoff.py          # Kiểm tra dữ liệu trước bàn giao
│   └── create_review_contact_sheet.py # Tạo ảnh tổng hợp để rà soát
├── src/
│   ├── features.py                  # Chuẩn hóa đặc trưng dùng chung
│   ├── model.py                     # Kiến trúc MLP (Thành viên B)
│   └── utils.py                     # Tiện ích của dự án
├── models/                          # Mô hình phát hành sau huấn luyện
├── results/                         # Kết quả và biểu đồ đánh giá
├── docs/
│   └── member_A.md                  # Tài liệu phần dữ liệu và đặc trưng
├── 01_collect_data.py              # Mã thu thập webcam có sẵn; cần hoàn thiện theo nhu cầu
├── 02_train.py                     # Điểm vào huấn luyện (Thành viên B)
├── 03_app.py                       # Điểm vào webcam (Thành viên C)
├── requirements.txt
├── .gitignore
└── README.md
```

> Cấu trúc trên mô tả các thành phần chính của repo và vai trò dự kiến. Một số file của phần huấn luyện và webcam là mã khung/chưa được xác nhận hoàn thiện; `data/raw/` và các tệp sinh ra có thể không xuất hiện sau khi clone vì quy tắc Git.

## 3. Cài đặt và kích hoạt môi trường
Khuyến nghị **Python 3.11** (kế hoạch lấy Python 3.11.9 làm mốc).

### Bước 1 — Clone dự án
```bash
git clone https://github.com/BaoUyen22/Hand-Tracking.git
cd Hand-Tracking
```
### Bước 2 — Tạo môi trường ảo `.venv311`
**Windows (PowerShell hoặc CMD):**
```powershell
py -3.11 --version
py -3.11 -m venv .venv311
```
**macOS / Linux** (máy cần cài sẵn Python 3.11):
```bash
python3.11 --version
python3.11 -m venv .venv311
```

### Bước 3 — Kích hoạt môi trường
**Windows PowerShell:**
```powershell
.\.venv311\Scripts\Activate.ps1
```
**Windows Command Prompt (CMD):**
```bat
.venv311\Scripts\activate.bat
```
**macOS / Linux:**
```bash
source .venv311/bin/activate
```
Sau khi kích hoạt thành công, terminal thường hiển thị `(.venv311)` ở đầu dòng. Để thoát môi trường ảo, dùng `deactivate`.

> Nếu PowerShell chặn chạy `Activate.ps1`, có thể chạy trực tiếp bằng `\.venv311\Scripts\python.exe` như các lệnh ở phần tiếp theo.

### Bước 4 — Cài đặt thư viện
Sau khi kích hoạt môi trường:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

## 4. Chạy phần chuẩn bị dữ liệu (Thành viên A)
Trước tiên cần chuẩn bị bộ dữ liệu ASL-HG Raw trong `data/raw/` đúng cấu trúc mà script sử dụng. **Không chạy lại quá trình Cleaning/Extract nếu chỉ cần dùng bộ dữ liệu đã bàn giao để train.**

Chi tiết phần chuẩn bị dữ liệu xem tại [`docs/member_A.md`](docs/member_A.md).

## 5. Huấn luyện và ứng dụng webcam
Theo phân công:

- **Thành viên A:** chuẩn bị dữ liệu, manifest, split, đặc trưng chung và kiểm tra leakage.
- **Thành viên B:** baseline, huấn luyện MLP, đánh giá và xuất ONNX.
- **Thành viên C:** tích hợp ứng dụng webcam, kiểm thử luồng suy luận và trải nghiệm sử dụng.

Các điểm vào `02_train.py` và `03_app.py` được giữ theo cấu trúc ban đầu. **Chỉ chạy khi thành viên phụ trách đã hoàn thiện và xác nhận hướng dẫn**; không giả định hiện đã có mô hình ONNX chạy được hoặc ngưỡng confidence cố định.
