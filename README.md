# 🚀 Super OCR & High-Res PDF Studio

Ứng dụng Desktop chuyên nghiệp dành cho Windows giúp **đọc ảnh, phục chế làm nét chữ siêu phân giải (Super-Resolution), quét OCR offline và xuất tệp PDF chất lượng cao (Searchable PDF & High-Res Image PDF)**.

---

## 🌟 Các tính năng nổi bật

### 1. 🔍 Phục chế & Làm nét chữ siêu phân giải
- **Tăng độ nét viền chữ (Text Sharpening):** Thuật toán kết hợp Unsharp Masking và Laplacian Filter giúp khử nhòe do rung tay, làm rõ từng nét chữ thanh đậm.
- **Siêu phân giải (Lanczos-4 Super-Resolution):** Phóng to tài liệu 1.5x, 2x, 3x, 4x lên chuẩn **300 - 600 DPI** cho chất lượng in ấn sắc nét mà không bị vỡ hạt.
- **Tẩy trắng nền & Khử bóng (Background Whitening & Shadow Removal):** Xóa bỏ các vết bóng đổ của điện thoại/tay cầm, khử ố vàng và làm trắng phông nền trang giấy sạch sẽ.
- **Tăng tương phản thông minh (CLAHE):** Tự động cân bằng ánh sáng, làm đậm mực chữ mà không làm cháy sáng tài liệu.
- **Bộ Preset thông minh 1-click:**
  - ⭐ **Văn bản siêu nét (Ultra Sharp):** Tối ưu đọc văn bản, tăng độ nét viền chữ tối đa.
  - 📄 **Quét sạch nền & Khử bóng (Clean Scan):** Xóa bóng, làm trắng giấy như máy scan.
  - 🖨️ **Đen trắng chuẩn tài liệu (Crisp B&W):** Nhị phân hóa thích ứng tạo bản scan đen trắng siêu sạch.
  - 🔍 **Phóng to siêu phân giải (Super Res 3x):** Phóng đại độ phân giải ảnh lên mức cực cao.
  - 🖼️ **Màu gốc tự nhiên (Natural Color):** Giữ nguyên màu sắc trung thực của ảnh.

### 2. ⚡ Thanh trượt so sánh Trước / Sau (Split Slider Preview)
- Kéo thanh trượt trực tiếp trên ảnh để so sánh độ nét giữa **Ảnh gốc (Trước)** và **Ảnh đã làm nét (Sau)** theo thời gian thực.
- Hỗ trợ phóng to (Zoom in/out), thu nhỏ, xem tỷ lệ 100% (1:1 thực tế), vừa khung hình và kéo di chuyển (Pan).

### 3. 📝 Quét OCR Offline tốc độ cao (RapidOCR)
- Chạy **offline 100%**, không cần kết nối mạng hay cài đặt thêm phần mềm rườm rà.
- Nhận diện chữ tiếng Việt, tiếng Anh, số, bảng biểu và ký tự đặc biệt với độ chính xác cao.
- Trích xuất tọa độ chính xác từng từ / dòng (Bounding Box) để tạo lớp chữ ẩn cho PDF.
- Bảng hiển thị kết quả OCR: cho phép xem trước, tìm kiếm, chỉnh sửa văn bản và copy hoặc xuất ra file `.txt`.

### 4. 📄 Xuất PDF Siêu Phân Giải & Tìm kiếm được (Searchable PDF)
- **PDF Tìm kiếm được (Searchable PDF):** Tự động nhúng lớp chữ OCR vô hình chuẩn xác theo vị trí của ảnh nét. Khi mở file trên bất kỳ trình đọc nào (Acrobat, Chrome, Edge), bạn có thể **bôi đen, copy và tìm kiếm (Ctrl+F)**.
- **PDF Ảnh siêu phân giải (High-Res Image PDF):** Giữ trọn độ nét cao nhất cho in ấn và lưu trữ.
- Tùy chọn khổ trang: Khớp kích thước ảnh nét, Chuẩn A4 Dọc, Chuẩn A4 Ngang.
- Tùy chọn gom tất cả ảnh vào 1 file PDF duy nhất hoặc xuất từng file riêng lẻ.

---

## 🛠️ Hướng dẫn cài đặt & Khởi động

### Cách 1: Khởi động nhanh bằng file `run.bat` (Khuyên dùng)
- Bạn chỉ cần **nhấp đúp chuột** vào tệp `run.bat` trong thư mục ứng dụng để mở phần mềm ngay lập tức.

### Cách 2: Chạy bằng dòng lệnh Terminal
```bash
cd C:\Users\A1\.gemini\antigravity\scratch\super_ocr_pdf
python main.py
```

---

## ⌨️ Phím tắt tiện lợi

| Phím tắt | Chức năng |
|---|---|
| `Ctrl + O` | Mở hộp thoại thêm hình ảnh |
| `F5` | Quét OCR trang hiện tại |
| `F6` | Quét OCR cho tất cả các trang |
| `Ctrl + E` | Mở hộp thoại xuất PDF siêu phân giải |
| `Cuộn chuột` | Phóng to / Thu nhỏ ảnh |
| `Kéo chuột phải / Chuột giữa` | Di chuyển khung hình (Pan) |

---

## 📂 Cấu trúc thư mục

```
super_ocr_pdf/
├── run.bat                     # File kích hoạt nhanh 1-click cho Windows
├── requirements.txt            # Danh sách thư viện Python
├── main.py                     # File chạy chính của ứng dụng
├── test_app.py                 # Bộ kiểm thử tự động toàn diện
├── core/
│   ├── enhancer.py             # Bộ thuật toán làm nét chữ, siêu phân giải, khử bóng
│   ├── ocr_engine.py           # Bộ nhận diện OCR offline và trích xuất tọa độ
│   └── pdf_builder.py          # Bộ sinh file PDF (Searchable PDF & High-Res PDF)
└── ui/
    ├── main_window.py          # Cửa sổ chính, thanh công cụ, menu
    ├── image_list_widget.py    # Danh sách thumbnail, sắp xếp, xoay, xóa
    ├── preview_widget.py       # Khung xem ảnh Trước/Sau (Split slider & Zoom/Pan)
    ├── settings_panel.py       # Thanh điều khiển độ nét, tương phản, phóng to, preset
    ├── ocr_panel.py            # Bảng hiển thị, tìm kiếm, chỉnh sửa và copy text OCR
    ├── export_dialog.py        # Hộp thoại xuất PDF với tiến trình nền
    └── styles.py               # Thiết kế theme Dark Mode sang trọng
```
