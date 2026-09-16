"""
OCR Recognition and Text Extraction Panel.
Displays extracted text, lets user choose OCR engine (VietOCR AI vs RapidOCR),
search, edit, copy, and export TXT files.
"""

from typing import Optional
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QTextCursor, QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QComboBox, QGroupBox, QCheckBox
)

from core.ocr_engine import OCRResult, OCREngine
from core.config_manager import ConfigManager


class OCRPanel(QWidget):
    """Panel displaying recognized OCR text and action controls."""

    scan_current_requested = Signal()
    scan_all_requested = Signal()
    engine_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Group: Engine Selection (100% Offline)
        grp_engine = QGroupBox("CHẾ ĐỘ NHẬN DIỆN (100% CỤC BỘ / OFFLINE)")
        eng_layout = QVBoxLayout(grp_engine)

        lbl_engine_info = QLabel("🔥 <b>Mô hình Deep Learning Tiếng Việt & Viết tay</b> (VietOCR + Cinnamon AI)")
        lbl_engine_info.setStyleSheet("color: #38bdf8; font-size: 12px;")
        eng_layout.addWidget(lbl_engine_info)

        lbl_pipeline = QLabel("<b>Quy trình 4 Bước:</b> Tiền xử lý ➜ PaddleOCR DBNet (Định vị 1:1) ➜ Deep Learning ➜ Mô hình Ngôn ngữ (LM)")
        lbl_pipeline.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        eng_layout.addWidget(lbl_pipeline)

        self.chk_use_lm = QCheckBox("🧠 Kích hoạt Hậu xử lý Mô hình Ngôn ngữ (Language Model - LM)")
        self.chk_use_lm.setChecked(True)
        self.chk_use_lm.setToolTip("Tự động sửa lỗi chính tả từ vựng theo kho 74.000 từ, khôi phục thanh dấu ngữ cảnh, chuẩn hóa dấu câu và bảng biểu.")
        self.chk_use_lm.toggled.connect(self._on_lm_toggled)
        eng_layout.addWidget(self.chk_use_lm)

        layout.addWidget(grp_engine)

        # Trigger Buttons
        btn_layout = QHBoxLayout()
        self.btn_scan_current = QPushButton("🔍 Quét trang này")
        self.btn_scan_current.setObjectName("primary_btn")
        self.btn_scan_current.clicked.connect(self._on_scan_current_clicked)

        self.btn_scan_all = QPushButton("⚡ Quét tất cả trang")
        self.btn_scan_all.setObjectName("success_btn")
        self.btn_scan_all.clicked.connect(self._on_scan_all_clicked)

        btn_layout.addWidget(self.btn_scan_current)
        btn_layout.addWidget(self.btn_scan_all)
        layout.addLayout(btn_layout)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Search Bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔎 Tìm kiếm từ trong văn bản...")
        self.search_input.textChanged.connect(self._on_search_text)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)

        # Text Editor
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText(
            "Kết quả nhận diện văn bản OCR sẽ hiển thị tại đây...\n\n"
            "• Chế độ 100% Cục bộ / Offline (Không cần Internet hay API Key).\n"
            "• PaddleOCR DBNet quét tọa độ khung chữ 1:1 bảo toàn vị trí văn bản.\n"
            "• Mô hình Deep Learning nhận diện tiếng Việt có dấu và chữ viết tay chuyên sâu.\n"
            "• Hậu xử lý Mô hình Ngôn ngữ (LM) tự động sửa lỗi chính tả theo kho 74.000 từ và chuẩn hóa bảng biểu.\n\n"
            "Bạn có thể trực tiếp chỉnh sửa văn bản trước khi xuất PDF."
        )
        layout.addWidget(self.text_edit)

        # Stats Bar
        self.lbl_stats = QLabel("Ký tự: 0 | Từ: 0 | Thời gian quét: 0.0s")
        self.lbl_stats.setStyleSheet("color: #71717a; font-size: 11px;")
        layout.addWidget(self.lbl_stats)

        # Export Buttons
        action_layout = QHBoxLayout()
        self.btn_copy = QPushButton("📋 Sao chép (Copy)")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)

        self.btn_save_txt = QPushButton("💾 Lưu file TXT")
        self.btn_save_txt.clicked.connect(self._save_to_txt)

        action_layout.addWidget(self.btn_copy)
        action_layout.addWidget(self.btn_save_txt)
        layout.addLayout(action_layout)

    def _on_engine_changed(self, mode: str = "offline"):
        OCREngine.get_instance().engine_mode = mode
        ConfigManager.get_instance().set("preferred_engine", mode)
        self.engine_changed.emit(mode)

    def _on_lm_toggled(self, checked: bool):
        OCREngine.get_instance().use_language_model = checked
        ConfigManager.get_instance().set("use_language_model", checked)

    def is_use_lm(self) -> bool:
        return self.chk_use_lm.isChecked()

    def _on_scan_current_clicked(self):
        self.scan_current_requested.emit()

    def _on_scan_all_clicked(self):
        self.scan_all_requested.emit()

    def set_scanning_state(self, is_scanning: bool):
        """Khóa/mở nút bấm và cập nhật trạng thái khi đang quét OCR."""
        self.btn_scan_current.setEnabled(not is_scanning)
        self.btn_scan_all.setEnabled(not is_scanning)
        if is_scanning:
            self.btn_scan_current.setText("⏳ Đang quét OCR...")
        else:
            self.btn_scan_current.setText("🔍 Quét trang này")

    def get_selected_engine(self) -> str:
        return "offline"

    def set_ocr_result(self, result: Optional[OCRResult]):
        """Hiển thị kết quả OCR văn bản thuần và thông số thống kê."""
        if result is None:
            self.text_edit.clear()
            self.lbl_stats.setText("Chưa quét OCR cho trang này.")
            return

        if result.error:
            self.text_edit.setPlainText(f"Lỗi: {result.error}")
            self.lbl_stats.setText("Quét thất bại.")
            return

        self.text_edit.setPlainText(result.full_text)
        self.lbl_stats.setText(
            f"Ký tự: {result.char_count} | Từ: {result.word_count} | Thời gian quét: {result.elapse_time}s"
        )

    def get_text(self) -> str:
        return self.text_edit.toPlainText()

    def set_progress(self, current: int, total: int, message: str = ""):
        if total > 0 and current < total:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            self.progress_bar.setFormat(f"{message} ({current}/{total})")
        else:
            self.progress_bar.setVisible(False)

    def _on_search_text(self, query: str):
        """Highlight matching query in text edit."""
        if not query:
            return

        cursor = self.text_edit.textCursor()
        document = self.text_edit.document()
        found = document.find(query)
        if not found.isNull():
            self.text_edit.setTextCursor(found)

    def _copy_to_clipboard(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Thông báo", "Chưa có nội dung văn bản để sao chép.")
            return
        from PySide6.QtGui import QGuiApplication
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "Thành công", "Đã sao chép văn bản vào bộ nhớ tạm!")

    def _save_to_txt(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Thông báo", "Chưa có nội dung văn bản để lưu.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu văn bản OCR thành file TXT",
            "ocr_result.txt",
            "Tệp văn bản (*.txt);;Tất cả (*.*)"
        )
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(text)
                QMessageBox.information(self, "Thành công", f"Đã lưu tệp văn bản tại:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu file: {e}")
