# Hợp đồng dữ liệu và quy ước phối hợp (Data Contract)

## 1. Mục đích
Thống nhất cấu trúc dữ liệu, định dạng đầu vào/đầu ra và trách nhiệm giữa ba thành viên khi xây dựng hệ thống nhận diện 34 ký hiệu bàn tay.

## 2. Quy ước nhãn
Hệ thống sử dụng 34 nhãn theo thứ tự cố định:
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y.

- Mỗi nhãn được ánh xạ thành một số nguyên từ 0 đến 33.
- Thứ tự nhãn phải thống nhất trong dữ liệu huấn luyện, mô hình và ứng dụng.
- Lưu danh sách nhãn vào `classes.json`.
- Manifest vẫn kiểm kê đủ 36 nhãn của dataset nguồn; J và Z không tham gia huấn luyện.

## 3. Cấu trúc Manifest
Mỗi ảnh có một bản ghi với các thông tin chính:
- `sample_id`: mã định danh duy nhất.
- `relative_path`: đường dẫn ảnh tương đối.
- `label`: nhãn gốc.
- `participant_id`: mã người tham gia.
- `file_sha256`: mã hash của file.
- `pixel_sha256`: mã hash nội dung pixel, có quy ước kích thước và thứ tự kênh.
- `duplicate_group`: nhóm ảnh trùng nếu có.
- `subject_group`: nhóm Participant liên quan.
- `split`: train, val, test hoặc chưa phân chia.
- `review_status`: approved, review_needed hoặc rejected.
- `keep`: ảnh được phép dùng trong pipeline hay không.
- `reason_codes`: lý do đánh dấu hoặc loại ảnh.
- `preprocessing_version`: phiên bản xử lý dữ liệu.
- `feature_status`: trạng thái trích xuất đặc trưng.

Các trường còn lại như kích thước ảnh, session, reviewer và hand_count được lưu để phục vụ kiểm tra và truy vết.

## 4. Quy ước chia dữ liệu
Phương án dự kiến:
- Train: P1–P6.
- Validation: P7–P8.
- Test: P9–P10.

Chỉ khóa cách chia sau khi kiểm tra không có nhóm ảnh trùng hoặc nhóm Participant liên thông xuất hiện ở nhiều tập.
Ảnh chưa được duyệt không được tự động sử dụng để huấn luyện.

## 5. Quy ước đặc trưng
- MediaPipe cung cấp 21 điểm mốc bàn tay.
- Chuẩn hóa tọa độ lấy cổ tay làm gốc.
- Chuẩn hóa tỷ lệ dựa trên khoảng cách Landmark 0–9 và kiểm tra trường hợp scale không hợp lệ.
- Dữ liệu đặc trưng sử dụng `float32`.
- Không cho phép vector chứa NaN hoặc Infinity.
- Hàm chuẩn hóa phải dùng chung giữa trích xuất dữ liệu và ứng dụng webcam.

Các phương án đang được xem xét:
- 63 chiều: 21 × (x, y, z).
- 84 chiều: thêm 21 giá trị visibility sau khi kiểm chứng.
- 131 chiều: ứng viên biểu diễn hai tay.

**Số chiều cuối cùng chưa được chốt.** Khi chọn phương án, nhóm phải cập nhật đồng bộ dữ liệu, MLP, ONNX và ứng dụng.

## 6. Dữ liệu bàn giao cho Thành viên B
- `X_train.npy`, `y_train.npy`.
- `X_val.npy`, `y_val.npy`.
- `X_test.npy`, `y_test.npy`.
- Danh sách `sample_ids` tương ứng từng tập.
- `classes.json`.
- Metadata chứa số chiều, phiên bản tiền xử lý và thông tin split.
Các hàng của X, y và sample_ids phải tương ứng chính xác.
## 7. Bàn giao cho Thành viên C
Thành viên A cung cấp hàm tạo vector đặc trưng dùng chung trong `src/features.py`.
Thành viên C sử dụng hàm này để xử lý landmark từ webcam, bảo đảm cách chuẩn hóa giống lúc huấn luyện.
Ứng dụng phải kiểm tra số chiều, kiểu dữ liệu và phiên bản tiền xử lý trước khi đưa vector vào mô hình.

## 8. Trách nhiệm module
| Module | Thành viên |
|---|---|
| Manifest, làm sạch dữ liệu, split | A |
| Feature Extraction, kiểm tra dữ liệu | A |
| Baseline, MLP, ONNX | B |
| Webcam, suy luận và giao diện | C |
| Kiểm thử tích hợp, báo cáo | Cả nhóm |

## 9. Quy tắc Git
- Không đưa môi trường ảo `.venv311` vào Git.
- Không commit ZIP, ảnh Raw, video hoặc cache lớn.
- Không ghi đường dẫn cá nhân cố định trong mã nguồn.
- Mỗi task thực hiện trên nhánh riêng và tạo Pull Request.
- Mọi thay đổi liên quan đến schema, số chiều hoặc thứ tự nhãn phải thông báo cho các thành viên khác.

## 10. Trạng thái
Hợp đồng ở trạng thái dự thảo, chờ A/B/C xác nhận các quy ước đầu vào, đầu ra và phương án đặc trưng cuối cùng.