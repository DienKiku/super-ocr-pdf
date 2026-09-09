"""
Export Dialog and Background Worker for High-Res Searchable PDF generation.
"""

from typing import List, Optional
import os
import subprocess
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QCheckBox, QComboBox, QSlider,
    QProgressBar, QFileDialog, QMessageBox, QGroupBox, QLineEdit
)

from core.enhancer import DocumentEnhancer, EnhanceParams
from core.ocr_engine import OCREngine, OCRResult
from core.pdf_builder import PDFBuilder, PageData, PDFConfig
from ui.image_list_widget import ImageItem


class ExportWorker(QThread):
    """Background thread to process enhancement, OCR, and build PDFs without freezing the UI."""

    progress = Signal(int, int, str)
    finished = Signal(bool, str, str)  # success, error_msg, final_path

    def __init__(
        self,
        items: List[ImageItem],
        output_path: str,
        is_single_file: bool,
        pdf_config: PDFConfig,
        parent=None
    ):
        super().__init__(parent)
        self.items = items
        self.output_path = output_path
        self.is_single_file = is_single_file
        self.pdf_config = pdf_config
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            total_items = len(self.items)
            ocr_engine = OCREngine.get_instance() if self.pdf_config.searchable else None

            if self.is_single_file:
                pages_data: List[PageData] = []

                for idx, item in enumerate(self.items):
                    if self._is_cancelled:
                        self.finished.emit(False, "Đã hủy tiến trình xuất.", "")
                        return

                    self.progress.emit(
                        idx + 1,
                        total_items * 2,
                        f"Đang làm nét trang {idx + 1}/{total_items}..."
                    )

                    # Enhance if not cached
                    if item.enhanced_image is None:
                        enhanced = DocumentEnhancer.process(item.original_image, item.params)
                        item.enhanced_image = enhanced
                    else:
                        enhanced = item.enhanced_image

                    # OCR if searchable is requested
                    ocr_res = item.ocr_result
                    if self.pdf_config.searchable:
                        if ocr_res is None:
                            self.progress.emit(
                                idx + 1,
                                total_items * 2,
                                f"Đang quét OCR trang {idx + 1}/{total_items}..."
                            )
                            ocr_res = ocr_engine.recognize(enhanced)
                            item.ocr_result = ocr_res

                    pages_data.append(PageData(
                        image=enhanced,
                        ocr_result=ocr_res,
                        title=f"Trang {idx + 1}"
                    ))

                if self._is_cancelled:
                    self.finished.emit(False, "Đã hủy tiến trình xuất.", "")
                    return

                def on_pdf_progress(cur, tot, msg):
                    self.progress.emit(total_items + cur, total_items * 2, msg)

                PDFBuilder.create_pdf(
                    pages_data,
                    self.output_path,
                    self.pdf_config,
                    progress_callback=on_pdf_progress
                )

                self.finished.emit(True, "", self.output_path)

            else:
                # Separate files
                out_dir = self.output_path
                for idx, item in enumerate(self.items):
                    if self._is_cancelled:
                        self.finished.emit(False, "Đã hủy tiến trình xuất.", "")
                        return

                    self.progress.emit(
                        idx + 1,
                        total_items,
                        f"Đang xử lý & xuất file trang {idx + 1}/{total_items}..."
                    )

                    if item.enhanced_image is None:
                        enhanced = DocumentEnhancer.process(item.original_image, item.params)
                        item.enhanced_image = enhanced
                    else:
                        enhanced = item.enhanced_image

                    ocr_res = item.ocr_result
                    if self.pdf_config.searchable and ocr_res is None:
                        ocr_res = ocr_engine.recognize(enhanced)
                        item.ocr_result = ocr_res

                    page = PageData(image=enhanced, ocr_result=ocr_res, title=f"Trang {idx + 1}")
                    base_name = os.path.splitext(item.filename)[0]
                    single_out = os.path.join(out_dir, f"{base_name}_highres.pdf")
                    PDFBuilder.create_pdf([page], single_out, self.pdf_config)

                self.finished.emit(True, "", out_dir)

        except Exception as e:
            self.finished.emit(False, str(e), "")


class ExportDialog(QDialog):
    """Configuration dialog for PDF export."""

    def __init__(self, items: List[ImageItem], parent=None):
        super().__init__(parent)
        self.items = items
        self.worker: Optional[ExportWorker] = None
        self.final_output_path = ""
        self.setWindowTitle("Xuất Tài Liệu PDF Siêu Phân Giải")
        self.resize(520, 520)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Output Mode Group
        group_mode = QGroupBox("CHẾ ĐỘ TỆP ĐẦU RA")
        gm_layout = QVBoxLayout(group_mode)
        self.radio_single = QRadioButton("Gom tất cả trang vào 1 file PDF duy nhất (Khuyên dùng)")
        self.radio_single.setChecked(True)
        self.radio_separate = QRadioButton("Xuất mỗi trang thành một file PDF riêng biệt")
        gm_layout.addWidget(self.radio_single)
        gm_layout.addWidget(self.radio_separate)
        layout.addWidget(group_mode)

        # Destination Path
        dest_layout = QHBoxLayout()
        self.edit_dest = QLineEdit()
        self.edit_dest.setText(os.path.abspath("document_highres.pdf"))
        self.btn_browse = QPushButton("Chọn nơi lưu...")
        self.btn_browse.clicked.connect(self._browse_destination)
        dest_layout.addWidget(self.edit_dest)
        dest_layout.addWidget(self.btn_browse)
        layout.addLayout(dest_layout)

        # Searchable OCR Feature
        group_ocr = QGroupBox("TÍNH NĂNG OCR & TÌM KIẾM")
        go_layout = QVBoxLayout(group_ocr)
        self.chk_searchable = QCheckBox("Tạo PDF Tìm kiếm được (Searchable PDF)")
        self.chk_searchable.setChecked(True)
        lbl_ocr_hint = QLabel(
            "✓ Nhúng lớp văn bản OCR ẩn dưới ảnh siêu nét\n"
            "✓ Cho phép bôi đen, copy và tìm kiếm (Ctrl + F) trực tiếp trong file PDF"
        )
        lbl_ocr_hint.setStyleSheet("color: #10b981; font-size: 11px; margin-left: 20px;")
        go_layout.addWidget(self.chk_searchable)
        go_layout.addWidget(lbl_ocr_hint)
        layout.addWidget(group_ocr)

        # PDF Page Dimensions
        group_page = QGroupBox("KÍCH THƯỚC KHỔ GIẤY")
        gp_layout = QHBoxLayout(group_page)
        self.combo_page_size = QComboBox()
        self.combo_page_size.addItem("Khớp theo kích thước ảnh nét (Tối đa độ phân giải)", "fit_image")
        self.combo_page_size.addItem("Chuẩn khổ A4 Dọc (A4 Portrait)", "a4_portrait")
        self.combo_page_size.addItem("Chuẩn khổ A4 Ngang (A4 Landscape)", "a4_landscape")
        gp_layout.addWidget(QLabel("Khổ trang:"))
        gp_layout.addWidget(self.combo_page_size)
        layout.addWidget(group_page)

        # Quality & DPI
        group_quality = QGroupBox("CHẤT LƯỢNG NÉN & DPI")
        gq_layout = QVBoxLayout(group_quality)

        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("Chất lượng ảnh JPEG:"))
        self.lbl_q_val = QLabel("95% (Cực nét)")
        self.lbl_q_val.setStyleSheet("color: #60a5fa; font-weight: bold;")
        q_row.addStretch()
        q_row.addWidget(self.lbl_q_val)
        gq_layout.addLayout(q_row)

        self.slider_quality = QSlider(Qt.Horizontal)
        self.slider_quality.setRange(80, 100)
        self.slider_quality.setValue(95)
        self.slider_quality.valueChanged.connect(
            lambda v: self.lbl_q_val.setText(f"{v}% (Cực nét)" if v >= 95 else f"{v}%")
        )
        gq_layout.addWidget(self.slider_quality)
        layout.addWidget(group_quality)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #93c5fd; font-weight: 500;")
        self.lbl_status.setVisible(False)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.progress_bar)

        # Action Buttons
        btn_bar = QHBoxLayout()
        self.btn_export = QPushButton("🚀 Tiến hành Xuất PDF")
        self.btn_export.setObjectName("primary_btn")
        self.btn_export.clicked.connect(self._start_export)

        self.btn_cancel = QPushButton("Hủy bỏ")
        self.btn_cancel.clicked.connect(self._cancel_or_close)

        self.btn_open = QPushButton("📂 Mở tệp PDF")
        self.btn_open.setObjectName("success_btn")
        self.btn_open.setVisible(False)
        self.btn_open.clicked.connect(self._open_exported_file)

        btn_bar.addStretch()
        btn_bar.addWidget(self.btn_open)
        btn_bar.addWidget(self.btn_export)
        btn_bar.addWidget(self.btn_cancel)
        layout.addLayout(btn_bar)

    def _browse_destination(self):
        if self.radio_single.isChecked():
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Chọn nơi lưu file PDF",
                self.edit_dest.text(),
                "Tệp PDF (*.pdf);;Tất cả (*.*)"
            )
            if path:
                self.edit_dest.setText(path)
        else:
            folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu các file PDF")
            if folder:
                self.edit_dest.setText(folder)

    def _start_export(self):
        out_path = self.edit_dest.text().strip()
        if not out_path:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn đường dẫn lưu tệp!")
            return

        is_single = self.radio_single.isChecked()
        if is_single and not out_path.lower().endswith(".pdf"):
            out_path += ".pdf"
            self.edit_dest.setText(out_path)

        # Build config
        cfg = PDFConfig(
            page_size=self.combo_page_size.currentData(),
            searchable=self.chk_searchable.isChecked(),
            jpeg_quality=self.slider_quality.value(),
            dpi=300
        )

        # UI state
        self.btn_export.setEnabled(False)
        self.btn_browse.setEnabled(False)
        self.radio_single.setEnabled(False)
        self.radio_separate.setEnabled(False)
        self.chk_searchable.setEnabled(False)
        self.combo_page_size.setEnabled(False)
        self.slider_quality.setEnabled(False)

        self.progress_bar.setVisible(True)
        self.lbl_status.setVisible(True)
        self.lbl_status.setText("Đang khởi tạo tiến trình...")

        # Start worker
        self.worker = ExportWorker(self.items, out_path, is_single, cfg)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_progress(self, current: int, total: int, message: str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.lbl_status.setText(message)

    def _on_worker_finished(self, success: bool, error_msg: str, final_path: str):
        self.btn_export.setEnabled(True)
        self.btn_browse.setEnabled(True)
        self.radio_single.setEnabled(True)
        self.radio_separate.setEnabled(True)
        self.chk_searchable.setEnabled(True)
        self.combo_page_size.setEnabled(True)
        self.slider_quality.setEnabled(True)

        if success:
            self.final_output_path = final_path
            self.lbl_status.setText("✅ Đã xuất PDF thành công!")
            self.btn_open.setVisible(True)
            QMessageBox.information(
                self,
                "Xuất PDF thành công",
                f"Đã xuất tài liệu PDF chất lượng cao thành công tại:\n{final_path}"
            )
        else:
            self.lbl_status.setText("❌ Xuất PDF thất bại.")
            QMessageBox.critical(self, "Lỗi xuất PDF", f"Không thể hoàn tất xuất PDF:\n{error_msg}")

    def _cancel_or_close(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1500)
        self.reject()

    def _open_exported_file(self):
        if self.final_output_path and os.path.exists(self.final_output_path):
            try:
                # Open with default Windows viewer
                os.startfile(self.final_output_path)
            except Exception as e:
                QMessageBox.warning(self, "Thông báo", f"Không thể mở file: {e}")
