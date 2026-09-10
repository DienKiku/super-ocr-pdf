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
    QProgressBar, QComboBox, QGroupBox
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

        # Group: Engine Selection
        grp_engine = QGroupBox("CHẾ ĐỘ NHẬN DIỆN OCR")
        eng_layout = QVBoxLayout(grp_engine)
        self.combo_engine = QComboBox()
        self.combo_engine.addItem("🌐 Online (AI Vision Cloud) — Chuẩn 100% Viết tay & In ấn (Google Gemini)", "online")
        self.combo_engine.addItem("💻 Offline (Mã nguồn mở Local) — Siêu tốc 2s, Miễn phí (PP-OCR + Smart AI)", "offline")
        self.combo_engine.currentIndexChanged.connect(self._on_engine_changed)
        eng_layout.addWidget(self.combo_engine)

        # Gemini API Key Setup Widget
        self.api_key_widget = QWidget()
        key_layout = QVBoxLayout(self.api_key_widget)
        key_layout.setContentsMargins(0, 4, 0, 4)
        key_layout.setSpacing(4)

        key_input_layout = QHBoxLayout()
        self.txt_api_key = QLineEdit()
        self.txt_api_key.setPlaceholderText("Dán Google Gemini API Key vào đây...")
        self.txt_api_key.setEchoMode(QLineEdit.Password)
        self.txt_api_key.setText(ConfigManager.get_instance().get_gemini_api_key())
        self.txt_api_key.textChanged.connect(self._on_api_key_changed)

        self.btn_toggle_echo = QPushButton("👁️")
        self.btn_toggle_echo.setFixedWidth(32)
        self.btn_toggle_echo.setToolTip("Hiện / Ẩn API Key")
        self.btn_toggle_echo.clicked.connect(self._toggle_api_key_visibility)

        self.btn_save_key = QPushButton("💾 Lưu")
        self.btn_save_key.setFixedWidth(60)
        self.btn_save_key.clicked.connect(self._save_api_key)

        key_input_layout.addWidget(self.txt_api_key)
        key_input_layout.addWidget(self.btn_toggle_echo)
        key_input_layout.addWidget(self.btn_save_key)
        key_layout.addLayout(key_input_layout)

        self.lbl_get_key = QLabel('<a href="https://aistudio.google.com/app/apikey" style="color: #60a5fa; text-decoration: underline;">👉 Bấm vào đây để lấy Gemini API Key miễn phí (Google AI Studio)</a>')
        self.lbl_get_key.setOpenExternalLinks(True)
        self.lbl_get_key.setStyleSheet("font-size: 11px;")
        key_layout.addWidget(self.lbl_get_key)

        eng_layout.addWidget(self.api_key_widget)
        layout.addWidget(grp_engine)

        saved_key = ConfigManager.get_instance().get_gemini_api_key()
        pref = ConfigManager.get_instance().get("preferred_engine", "offline")
        if pref in ("online", "gemini") and saved_key:
            idx = self.combo_engine.findData("online")
        else:
            idx = self.combo_engine.findData("offline")

        if idx >= 0:
            self.combo_engine.setCurrentIndex(idx)
        self._on_engine_changed(self.combo_engine.currentIndex())

        # Trigger Buttons
        btn_layout = QHBoxLayout()
        self.btn_scan_current = QPushButton("🔍 Quét trang này")
        self.btn_scan_current.setObjectName("primary_btn")
        self.btn_scan_current.clicked.connect(self.scan_current_requested.emit)

        self.btn_scan_all = QPushButton("⚡ Quét tất cả trang")
        self.btn_scan_all.setObjectName("success_btn")
        self.btn_scan_all.clicked.connect(self.scan_all_requested.emit)

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
            "• Chế độ 🌐 Online (Google Gemini AI): Chuẩn 100% chữ in, chữ viết tay, bảng biểu, hoá đơn.\n"
            "• Chế độ 💻 Offline (PP-OCR + Smart AI): Quét siêu tốc 2s hoàn toàn Offline, khôi phục dấu tiếng Việt chuẩn xác.\n\n"
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

    def _on_engine_changed(self, index: int):
        mode = self.combo_engine.currentData()
        self.api_key_widget.setVisible(mode in ("online", "gemini"))
        OCREngine.get_instance().engine_mode = mode
        ConfigManager.get_instance().set("preferred_engine", mode)
        self.engine_changed.emit(mode)

    def _toggle_api_key_visibility(self):
        if self.txt_api_key.echoMode() == QLineEdit.Password:
            self.txt_api_key.setEchoMode(QLineEdit.Normal)
            self.btn_toggle_echo.setText("🔒")
        else:
            self.txt_api_key.setEchoMode(QLineEdit.Password)
            self.btn_toggle_echo.setText("👁️")

    def _save_api_key(self):
        key = self.txt_api_key.text().strip()
        ConfigManager.get_instance().set_gemini_api_key(key)
        QMessageBox.information(self, "Đã lưu API Key", "Đã lưu Google Gemini API Key thành công!")

    def _on_api_key_changed(self, text: str):
        ConfigManager.get_instance().set_gemini_api_key(text.strip())

    def get_selected_engine(self) -> str:
        return self.combo_engine.currentData()

    def set_ocr_result(self, result: Optional[OCRResult]):
        """Populate OCR text and stats."""
        if result is None:
            self.text_edit.clear()
            self.lbl_stats.setText("Chưa quét OCR cho trang này.")
            return

        if result.error:
            self.text_edit.setPlainText(f"Lỗi: {result.error}")
            self.lbl_stats.setText("Quét thất bại.")
            return

        self.text_edit.setPlainText(result.full_text)

        stat_text = f"Ký tự: {result.char_count} | Từ: {result.word_count} | Thời gian: {result.elapse_time}s"
        if getattr(result, "extracted_fields", None):
            tags = []
            if "cccd" in result.extracted_fields:
                tags.append(f"CCCD: {', '.join(result.extracted_fields['cccd'])}")
            if "mst" in result.extracted_fields:
                tags.append(f"MST: {', '.join(result.extracted_fields['mst'])}")
            if "amounts" in result.extracted_fields:
                tags.append(f"Tiền: {result.extracted_fields['amounts'][0]}")
            if tags:
                stat_text += f" | 📌 {' • '.join(tags)}"

        self.lbl_stats.setText(stat_text)

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
