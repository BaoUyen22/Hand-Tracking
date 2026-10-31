# 🖐️ Hệ Thống Nhận Dạng Ký Hiệu Bàn Tay Tĩnh Theo Thời Gian Thực
*(Real-Time Static Hand Gesture / Sign Language Recognition using MediaPipe & MLP)*

> **Đồ án môn học:** Máy học (Machine Learning)  
> **Nền tảng:** Python 3.10 / 3.11 · OpenCV · MediaPipe · TensorFlow · ONNX Runtime  
> **Mục tiêu:** Nhận diện bảng chữ cái / chữ số ký hiệu tay thời gian thực qua webcam máy tính phổ thông (chạy hoàn toàn trên CPU, không yêu cầu GPU rời, độ trễ suy luận ~0.01 ms).

---

## 1. Tổng Quan Kiến Trúc & Phương Pháp

```text
Hand-Tracking/
│
├── data/                            # Thư mục dữ liệu
│   └── processed/                   # Chứa dữ liệu tọa độ đã bóc tách (.npy)
│       ├── X.npy                    # Ma trận đặc trưng 84 chiều (N_samples, 84)
│       ├── y.npy                    # Nhãn số tương ứng (N_samples,)
│       └── classes.npy              # Danh sách nhãn chữ/số ['A', 'B', 'C', ...]
│
├── models/                          # Lưu trữ mô hình sau huấn luyện
│   └── static_gesture_mlp.onnx      # File mô hình ONNX siêu nhẹ dùng cho realtime
│
├── results/                         # Kết quả đánh giá
│   └── figures/                     # Lưu ma trận nhầm lẫn (Confusion Matrix)
│
├── src/                             # Mã nguồn module tái sử dụng
│   ├── __init__.py                  # Đánh dấu package Python
│   ├── utils.py                     # Hàm chuẩn hóa tọa độ 84-d (Wrist-anchor + Palm-scale)
│   └── model.py                     # Định nghĩa kiến trúc mạng nơ-ron MLP
│
├── 01_collect_data.py               # Script thu thập tọa độ mẫu từ webcam
├── 02_train.py                      # Script huấn luyện MLP và xuất ra file ONNX
├── 03_app.py                        # Script mở webcam chạy nhận diện thời gian thực
│
├── requirements.txt                 # Danh sách thư viện cần thiết có ghim phiên bản
├── .gitignore                       # Cấu hình loại trừ file nặng khỏi Git
└── README.md                        # Tài liệu hướng dẫn sử dụng

## 2. Yêu Cầu Hệ Thống & Cài Đặt

### 2.1. Yêu cầu phần cứng & phần mềm
- Hệ điều hành: Windows 10/11, macOS, hoặc Ubuntu Linux.
- Camera: Webcam tích hợp hoặc USB Webcam hoạt động bình thường.
- Python: Phiên bản Python 3.10 hoặc 3.11 (khuyến nghị tránh dùng Python 3.13 do một số thư viện ML chưa hỗ trợ wheel build sẵn).

### 3.2. Các bước cài đặt chi tiết
**Bước 1: Clone kho mã nguồn về máy**
```text
git clone https://github.com/BaoUyen22/Hand-Tracking.git
cd Hand-Tracking
```
**Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)**
Trên Windows (PowerShell / Command Prompt):
```text
python -m venv .venv
.venv\Scripts\activate
```
Trên macOS / Linux:
```text
python3 -m venv .venv
source .venv/bin/activate
```
**Bước 3: Nâng cấp pip và cài đặt thư viện**
```text
python -m pip install --upgrade pip
pip install -r requirements.txt
```
**Lưu ý quan trọng**:
Để tránh xung đột thư viện với các dự án khác trên máy, bạn nên luôn chạy các lệnh trên trong môi trường ảo vừa kích hoạt (có dấu `.venv` ở đầu dòng lệnh).

## Kịch Bản 1: Chạy Thử Ngay Mô Hình Có Sẵn (Quickstart Demo)

Chạy ứng dụng webcam:
```text
python 03_app.py
```
### Thao tác khi chạy:
- Đưa bàn tay của bạn vào giữa khung hình webcam.
- Khi hệ thống nhận diện được cử chỉ với độ tin cậy ≥70%, nhãn ký hiệu và tỷ lệ phần trăm tự tin sẽ hiển thị màu xanh lá cây ngay trên màn hình.
- Nếu độ tin cậy <70%, hệ thống hiển thị trạng thái Uncertain (màu cam) để tránh phán đoán sai.
- Nhấn phím q để đóng cửa sổ ứng dụng và thoát camera an toàn.

## Kịch Bản 2: Thu Thập Dữ Liệu & Huấn Luyện Lại Từ Đầu
Khi bạn muốn bổ sung thêm các cử chỉ mới (ví dụ thêm các chữ cái khác hoặc số đếm):
### Bước 1: Khai báo cử chỉ & Thu thập mẫu
1. Mở file 01_collect_data.py.
2. Chỉnh sửa biến LABELS theo danh sách các ký hiệu bạn muốn dạy cho máy:
LABELS = ['A', 'B', 'C', 'D', 'E', 'HELLO', 'THANKS']
SAMPLES_PER_CLASS = 150  # Số frame thu thập cho mỗi ký hiệu
3. Chạy script thu thập:
```text
python 01_collect_data.py
```
4. Quy trình thu thập:
- Màn hình sẽ hiện chữ nhắc: Chuan bi: [Tên nhãn] (Nhan SPACE).
- Tạo dáng bàn tay trước camera, sau đó nhấn phím SPACE.
- Giữ dáng tay và di chuyển nhẹ (nghiêng nhẹ các góc, đưa lại gần/xa một chút) trong khi thanh đếm chạy từ 0/150 đến 150/150.
- Lặp lại cho đến khi hoàn thành tất cả các nhãn. Dữ liệu sẽ tự động được đóng gói thành các file .npy lưu tại data/processed/.
### Bước 2: Huấn luyện mô hình
Chạy kịch bản huấn luyện:
```text
python 02_train.py
```
- Script tự động chia tập dữ liệu thành 80% Train và 20% Test (có phân tầng nhãn).
- Quá trình huấn luyện mạng MLP 50 epochs chỉ mất khoảng 10–20 giây trên CPU.
- Xem chỉ số độ chính xác (Accuracy / Loss) in ra trên Terminal.
- Mô hình tự động được chuyển đổi sang dạng models/static_gesture_mlp.onnx.
### Bước 3: Kiểm thử sản phẩm
Chạy script demo để tận hưởng mô hình mới vừa huấn luyện:
```text
python 03_app.py
```

---

## 🧠 1. Tổng Quan Kiến Trúc & Phương Pháp

Thay vì đưa trực tiếp hình ảnh thô (RGB) vào mạng CNN dễ gây hiện tượng quá khớp (overfitting) với màu da hoặc phông nền phòng, hệ thống áp dụng phương pháp **Trích xuất đặc trưng hình học (Landmark-based Feature Extraction)**: