# 📌 Hợp Đồng Phạm Vi & Quy Ước Dự Án (Scope & Conventions)

## 1. Phạm vi nhãn (34 nhãn tĩnh)
- **10 Chữ số:** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **24 Chữ cái tĩnh:** A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
*(Loại trừ 2 chữ cái động: J và Z)*

## 2. Nguồn dữ liệu
- Dataset: ASL-HG Raw.
- Quy mô nguồn: 36.000 ảnh, 36 nhãn, từ P1 đến P10.
- Chỉ sử dụng 34 nhãn thuộc phạm vi để huấn luyện.
- Giữ nguyên dữ liệu gốc, không xóa hoặc chỉnh sửa ảnh Raw.
- Sử dụng Manifest để theo dõi ảnh hợp lệ, ảnh trùng, ảnh lỗi và các quyết định xử lý.
Không gộp Raw và Processed thành các mẫu độc lập.

## 3. Phân chia dữ liệu
Phương án chia dự kiến:

| Tập | Participant |
|---|---|
| Train | P1–P6 |
| Validation | P7–P8 |
| Test | P9–P10 |

### Quy tắc phân chia dữ liệu
- Dữ liệu được chia theo người tham gia (P1–P10), không chia ngẫu nhiên từng ảnh.
- Nếu ảnh của hai người khác nhau bị trùng hoàn toàn, cần xếp hai người đó vào cùng một nhóm khi chia dữ liệu.
- P5 và P6 được xếp chung vào tập Train vì đã phát hiện ảnh trùng giữa hai người này.
- Không để ảnh trùng hoặc các nhóm người có liên quan xuất hiện ở cả Train, Validation và Test.
- Nếu phát hiện ảnh trùng giữa các tập, cần kiểm tra và điều chỉnh cách chia trước khi huấn luyện mô hình.


## 4. Quy ước đặc trưng
- Sử dụng MediaPipe để trích xuất 21 điểm mốc bàn tay.
- Đưa gốc tọa độ về cổ tay (Landmark 0).
- Chuẩn hóa kích thước theo khoảng cách giữa Landmark 0 và Landmark 9.
- Lưu vector đặc trưng với kiểu dữ liệu `float32`.
- Bảo đảm quy trình chuẩn hóa nhất quán giữa huấn luyện và ứng dụng webcam.
- Kiểm tra giá trị không hợp lệ, kích thước vector và trường hợp scale gần bằng 0.

Phương án số chiều cần đánh giá:
- 63D: 21 điểm × 3 tọa độ x, y, z.
- 84D: 63 tọa độ và 21 giá trị visibility, nếu được xác minh có ích.
- 131D: phương án đề xuất biểu diễn hai tay.

**Trạng thái: Chưa chốt số chiều cuối cùng.**

## 5. Ký hiệu 0 và chế độ nhận diện
**Ký hiệu 0:**
- Cần xác nhận quy ước biểu diễn hai tay.
- Thực hiện thử nghiệm MediaPipe để đánh giá khả năng phát hiện đầy đủ hai tay.
- Không tự ý thay ký hiệu 0 bằng ký hiệu khác.

**Chế độ chữ/số:**
- Đề xuất cho phép người dùng chọn chế độ chữ cái hoặc chữ số.
- Chế độ này là tính năng hỗ trợ, không thay thế chức năng nhận diện 34 nhãn. Kết quả đánh giá có sử dụng chế độ và không sử dụng chế độ phải được trình bày riêng biệt.

**Trạng thái: Chờ xác nhận.**

## 6. Công nghệ sử dụng
- Python 3.11.
- OpenCV: đọc và xử lý hình ảnh.
- MediaPipe Hands: phát hiện điểm mốc bàn tay.
- NumPy: xử lý vector đặc trưng.
- TensorFlow/Keras: huấn luyện MLP.
- ONNX Runtime: suy luận trên CPU.

## 7. Sản phẩm và điều kiện nghiệm thu
Sản phẩm dự kiến:
- Mã nguồn hệ thống nhận diện.
- Mô hình phân loại 34 nhãn.
- Ứng dụng demo webcam.
- Báo cáo thực nghiệm và hướng dẫn chạy.

