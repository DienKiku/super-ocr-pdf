# Thư Mục Trọng Số Mô Hình AI (Offline Weights)

Thư mục này chứa các trọng số mô hình trí tuệ nhân tạo phục vụ nhận diện chữ và văn bản tiếng Việt hoàn toàn Offline (không cần kết nối Internet).

## 1. Danh Sách Mô Hình

| Mô hình | Kích thước | Chức năng | Nguồn gốc |
| :--- | :--- | :--- | :--- |
| **ch_PP-OCRv4_det_infer.onnx** | 4.74 MB | Định vị khung bao văn bản (Text Detection) với độ chính xác cao theo cấu trúc DBNet v4 | PaddleOCR chính thức (v4) |
| **rec_custom_v4_infer/** | 7.68 MB | Nhận diện ký tự tiếng Việt đã được tinh chỉnh (Fine-tuned) trên bộ dữ liệu thực tế (tuning_data) | Xuất từ quy trình PaddleOCR fine-tuning |
| **vgg_transformer.pth** | 151.8 MB | Mô hình Transformer nhận diện chữ viết tay và tiếng Việt tự nhiên | VietOCR (VGG + Transformer) |

---

## 2. Lưu Ý Về Dung Lượng & GitHub

- Theo chính sách của GitHub, các tệp nhị phân có dung lượng > 100 MB (như vgg_transformer.pth 151.8 MB) sẽ bị GitHub từ chối khi commit và push thông thường.
- Do đó, tệp vgg_transformer.pth được quản lý tự động qua .gitignore.
- Khi ứng dụng khởi chạy lần đầu tiên trên một máy mới, nếu tệp vgg_transformer.pth chưa có sẵn trong thư mục weights/, hệ thống sẽ tự động tải trọng số chính thức từ kho lưu trữ của VietOCR về bộ nhớ đệm cache (~/.cache/vietocr/) hoặc bạn có thể tải thủ công và đặt vào thư mục weights/.
