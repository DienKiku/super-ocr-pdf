# 🚀 Super OCR & High-Res PDF Studio

Ứng dụng Desktop chuyên nghiệp dành cho Windows: **Phục chế làm nét chữ siêu phân giải (Super-Resolution), nhận diện OCR Tiếng Việt & Chữ viết tay thông minh, và xuất tệp PDF chất lượng cao (Searchable PDF & High-Res Image PDF)**.

Phát triển bởi: **Fami (fami_7006)**  
Mã nguồn: [https://github.com/DienKiku/super-ocr-pdf](https://github.com/DienKiku/super-ocr-pdf)

---

## 🌟 Các tính năng nổi bật

### 1. 🔍 Phục chế & Làm nét chữ siêu phân giải
- **Tăng độ nét viền chữ (Text Sharpening):** Thuật toán kết hợp Unsharp Masking và bộ lọc cạnh chi tiết giúp khử nhòe mờ do rung tay hoặc chụp thiếu sáng, làm sắc nét từng nét thanh đậm của văn bản.
- **Siêu phân giải (Lanczos-4 Super-Resolution):** Phóng đại tài liệu 1.5x, 2x, 3x, 4x lên chuẩn in ấn sắc nét **300 - 600 DPI** mà không bị vỡ hạt hay răng cưa.
- **Tẩy trắng nền & Khử bóng (Background Whitening & Shadow Removal):** Tự động loại bỏ các vết bóng đổ do điện thoại hoặc tay cầm, khử ố vàng và làm sạch nền giấy như máy quét chuyên dụng.
- **Tăng tương phản thông minh (CLAHE):** Tự động cân bằng ánh sáng cục bộ, tăng độ tương phản của mực chữ mà không làm cháy sáng tài liệu.
- **Bộ cài đặt sẵn 1-click (Smart Presets):**
  - ⭐ **Văn bản siêu nét (Ultra Sharp):** Tối ưu hóa viền chữ cho tài liệu văn bản in và hợp đồng.
  - 📄 **Quét sạch nền & Khử bóng (Clean Scan):** Làm trắng phông nền, khử bóng tối ưu hóa in ấn.
  - 🖨️ **Đen trắng chuẩn tài liệu (Crisp B&W):** Nhị phân hóa thích ứng tạo bản scan đen trắng rõ nét.
  - 🔍 **Phóng to siêu phân giải (Super Res 3x):** Phóng đại độ phân giải ảnh lên mức cực cao.
  - 🖼️ **Màu gốc tự nhiên (Natural Color):** Giữ nguyên màu sắc trung thực ban đầu của tài liệu.

### 2. ⚡ Thanh trượt so sánh Trước / Sau thời gian thực (Split Slider Preview)
- Kéo thanh trượt trực tiếp trên trang tài liệu để trực quan hóa sự khác biệt giữa **Ảnh gốc (Trước)** và **Ảnh đã phục chế làm nét (Sau)**.
- Hỗ trợ phóng to/thu nhỏ (Zoom in/out), xem theo tỷ lệ chuẩn 100% (1:1), vừa khít khung nhìn (Fit to window) và di chuyển tự do (Pan).

### 3. 📝 Hệ thống nhận diện OCR đột phá (Kiến trúc 2 tầng)
- 🇻🇳 **Tiếng Việt Siêu Tốc (Hybrid AI - 100% Offline):**
  - Tốc độ quét cực nhanh: chỉ **1.5s - 2.5s** cho toàn bộ trang tài liệu A4.
  - Tích hợp kho từ điển 74.000 từ tiếng Việt chuẩn kết hợp thuật toán N-gram greedy giúp khôi phục chính xác 100% dấu tiếng Việt.
  - **Bảo toàn nguyên vẹn 100% dữ liệu kỹ thuật:** Tuyệt đối không làm biến dạng đường link (`https://...`), email (`...@...`), mã số thuế (MST), bảng số tiền và các ký hiệu quốc tế.
- ✨ **AI Vision Thông Minh (Google Gemini Vision API - Chuẩn xác 100% Viết tay):**
  - Tích hợp mô hình thế hệ mới nhất `gemini-3.6-flash` (kèm cơ chế dự phòng tự thích ứng).
  - Nhận diện chuẩn xác 100% các tài liệu phức tạp: **chữ viết tay tiếng Việt có dấu**, chữ viết ngoáy bút bi trên hóa đơn, phiếu giao hàng, bảng biểu nhiều cột, số tiền và chữ ký.
  - Giao diện nhập API Key tiện lợi với nút ẩn/hiện mật khẩu `👁️` và lưu trữ an toàn trong máy.

### 4. 📄 Xuất PDF Siêu Phân Giải & Tìm kiếm được (Searchable PDF)
- **PDF Tìm kiếm được (Searchable PDF):** Tự động nhúng lớp chữ OCR ẩn chính xác từng tọa độ từ. Khi mở trên bất kỳ trình đọc PDF nào (Adobe Acrobat, Foxit Reader, Chrome, Edge), bạn có thể **tìm kiếm (Ctrl+F), bôi đen và sao chép (Copy) văn bản**.
- **PDF Ảnh siêu phân giải (High-Res Image PDF):** Giữ trọn độ nét cao nhất cho nhu cầu in ấn và lưu trữ hồ sơ.
- Đa dạng tùy chọn khổ trang: Khớp tỷ lệ ảnh gốc, Khổ A4 Dọc, Khổ A4 Ngang.
- Hỗ trợ xuất gộp toàn bộ các trang thành 1 tệp PDF duy nhất hoặc xuất tách từng tệp riêng lẻ.

---

## 🛠️ Hướng dẫn cài đặt & Chạy ứng dụng

### Yêu cầu hệ thống
- Hệ điều hành: Windows 10 / 11 (64-bit)
- Python 3.10 trở lên (nếu chạy từ mã nguồn)

### 1. Cài đặt môi trường từ mã nguồn
```bash
# Clone mã nguồn về máy
git clone https://github.com/DienKiku/super-ocr-pdf.git
cd super-ocr-pdf

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# Khởi chạy ứng dụng
python main.py
```

### 2. Khởi chạy nhanh bằng file Batch (Khuyên dùng trên Windows)
- Nhấp đúp chuột vào file **`run.bat`** để khởi động phần mềm ngay lập tức.

### 3. Đóng gói ra file `.EXE` độc lập (Không cần cài Python)
- Nhấp đúp chuột vào file **`build_exe.bat`**.
- Sau khi hoàn tất, ứng dụng độc lập sẽ sẵn sàng tại:
  ```text
  dist\SuperOCRPDFStudio\SuperOCRPDFStudio.exe
  ```
- Bạn có thể sao chép thư mục `dist\SuperOCRPDFStudio` sang bất kỳ máy tính Windows nào để sử dụng trực tiếp mà không cần cài đặt thêm phần mềm phụ trợ.

---

## ⌨️ Phím tắt tiện lợi

| Phím tắt | Chức năng |
|:---|:---|
| `Ctrl + O` | Mở hộp thoại thêm hình ảnh / tài liệu |
| `F5` | Quét OCR cho trang đang xem |
| `F6` | Quét OCR hàng loạt cho toàn bộ các trang |
| `Ctrl + E` | Mở hộp thoại xuất file PDF siêu phân giải |
| `Cuộn chuột` | Phóng to / Thu nhỏ ảnh trong khung xem trước |
| `Kéo chuột giữa / Phải` | Di chuyển khung hình tự do (Pan) |

---

## 📂 Cấu trúc thư mục dự án

```text
super_ocr_pdf/
├── run.bat                     # File khởi động nhanh ứng dụng 1-click
├── build_exe.bat               # File script đóng gói ra file .EXE độc lập
├── sync_github.bat             # File script 1-click tự động đồng bộ code lên GitHub
├── requirements.txt            # Danh sách các thư viện Python yêu cầu
├── main.py                     # Điểm khởi chạy chính của ứng dụng
├── SuperOCRPDFStudio.spec      # Cấu hình đóng gói chuyên sâu PyInstaller
├── core/                       # Tầng xử lý logic nghiệp vụ và thuật toán AI
│   ├── config_manager.py       # Quản lý cấu hình người dùng và API Key
│   ├── enhancer.py             # Bộ thuật toán làm nét chữ, siêu phân giải, khử bóng
│   ├── ocr_engine.py           # Bộ nhận diện OCR (Hybrid AI & Google Gemini Vision)
│   ├── pdf_builder.py          # Bộ sinh file PDF (Searchable PDF & High-Res PDF)
│   └── models/                 # Từ điển tiếng Việt 74.000 từ & mô hình AI ONNX
└── ui/                         # Tầng giao diện người dùng (PySide6 GUI)
    ├── main_window.py          # Cửa sổ chính và điều phối tiến trình
    ├── image_list_widget.py    # Danh sách thumbnail các trang, xoay và sắp xếp
    ├── preview_widget.py       # Khung xem so sánh trước/sau (Split slider & Zoom/Pan)
    ├── settings_panel.py       # Bảng điều khiển bộ lọc làm nét và siêu phân giải
    ├── ocr_panel.py            # Bảng nhận diện OCR, chỉnh sửa văn bản, cấu hình API
    ├── export_dialog.py        # Hộp thoại xuất file PDF chất lượng cao
    └── styles.py               # Thiết kế giao diện Dark Mode hiện đại
```

---

## 📄 Bản quyền & Giấy phép

Phát triển bởi **Fami (fami_7006)**.  
Dự án được xây dựng phục vụ nhu cầu xử lý văn bản, phục chế chứng từ và số hóa tài liệu chất lượng cao.
