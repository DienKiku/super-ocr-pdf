"""
Main Application Window for Super OCR & High-Res PDF Studio.
Integrates image list, split preview canvas, enhancement settings, and OCR panel.
"""

from typing import Optional, List
import os
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QThreadPool, QRunnable
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QTabWidget, QToolBar, QStatusBar, QMessageBox, QLabel
)

from core.enhancer import DocumentEnhancer, EnhanceParams
from core.ocr_engine import OCREngine, OCRResult
from ui.image_list_widget import ImageListWidget, ImageItem
from ui.preview_widget import PreviewWidget
from ui.settings_panel import SettingsPanel
from ui.ocr_panel import OCRPanel
from ui.export_dialog import ExportDialog


class AsyncEnhanceWorker(QThread):
    """Worker to run enhancement in background without lagging the GUI."""
    enhanced_ready = Signal(int, int, object)  # req_id, index, enhanced np.ndarray

    def __init__(self, req_id: int, index: int, img: np.ndarray, params: EnhanceParams, parent=None):
        super().__init__(parent)
        self.req_id = req_id
        self.index = index
        self.img = img
        self.params = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        try:
            if self._cancelled:
                return
            enhanced = DocumentEnhancer.process(self.img, self.params, cancel_check=self.is_cancelled)
            if not self._cancelled:
                self.enhanced_ready.emit(self.req_id, self.index, enhanced)
        except Exception as e:
            print(f"Error in enhancement: {e}")


class AsyncSingleOCRWorker(QThread):
    """Worker to run single page OCR in background without freezing the GUI."""
    progress = Signal(int, int, str)
    finished = Signal(int, object)  # req_id, OCRResult

    def __init__(self, req_id: int, img: np.ndarray, mode: str, parent=None):
        super().__init__(parent)
        self.req_id = req_id
        self.img = img
        self.mode = mode
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled:
                return
            ocr = OCREngine.get_instance()
            res = ocr.recognize(self.img, mode=self.mode, progress_callback=self._on_progress)
            if not self._cancelled:
                self.finished.emit(self.req_id, res)
        except Exception as e:
            if not self._cancelled:
                self.finished.emit(self.req_id, OCRResult(error=str(e)))

    def _on_progress(self, current: int, total: int, message: str):
        if not self._cancelled:
            self.progress.emit(current, total, message)


class AsyncBatchOCRWorker(QThread):
    """Worker to run batch OCR across pages."""
    progress = Signal(int, int, str)
    finished = Signal()

    def __init__(self, items: List[ImageItem], mode: str = "vietnamese", parent=None):
        super().__init__(parent)
        self.items = items
        self.mode = mode
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        ocr = OCREngine.get_instance()
        total = len(self.items)
        for idx, item in enumerate(self.items):
            if self._cancelled:
                break
            self.progress.emit(idx + 1, total, f"Đang quét trang {idx + 1}/{total}...")

            # For OCR: use clean original image (rotated if user adjusted orientation)
            ocr_image = item.original_image
            if item.rotation != 0:
                ocr_image = DocumentEnhancer.rotate_image(ocr_image, item.rotation)

            if item.ocr_result is None:
                item.ocr_result = ocr.recognize(ocr_image, mode=self.mode)

        self.finished.emit()


class MainWindow(QMainWindow):
    """Main Application Window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Super OCR & High-Res PDF Studio - Làm Nét Chữ Siêu Phân Giải & Xuất PDF")
        self.resize(1280, 800)

        self._enhance_req_id = 0
        self._active_enhance_workers: List[AsyncEnhanceWorker] = []
        self._ocr_req_id = 0
        self._active_ocr_workers: List[AsyncSingleOCRWorker] = []
        self.batch_ocr_worker: Optional[AsyncBatchOCRWorker] = None

        self._init_ui()
        self._init_toolbar()
        self._init_statusbar()
        self._connect_signals()

    def _cleanup_enhance_worker(self, worker: AsyncEnhanceWorker):
        if worker in self._active_enhance_workers:
            self._active_enhance_workers.remove(worker)

    def _cleanup_ocr_worker(self, worker: AsyncSingleOCRWorker):
        if worker in self._active_ocr_workers:
            self._active_ocr_workers.remove(worker)

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Splitter dividing Left List - Center Preview - Right Tabs
        self.splitter = QSplitter(Qt.Horizontal)

        # 1. Left: Image List Sidebar
        self.image_list = ImageListWidget(self)
        self.splitter.addWidget(self.image_list)

        # 2. Center: Interactive Preview (Split Slider)
        self.preview = PreviewWidget(self)
        self.splitter.addWidget(self.preview)

        # 3. Right: Tabbed Panel (Settings & OCR)
        self.tabs = QTabWidget(self)

        self.settings_panel = SettingsPanel(self)
        self.tabs.addTab(self.settings_panel, "⚙️ Tinh Chỉnh Nét")

        self.ocr_panel = OCRPanel(self)
        self.tabs.addTab(self.ocr_panel, "📝 Bảng OCR")

        self.splitter.addWidget(self.tabs)

        # Initial splitter proportion
        self.splitter.setSizes([260, 680, 340])
        main_layout.addWidget(self.splitter)

    def _init_toolbar(self):
        toolbar = QToolBar("Thanh công cụ chính", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Add Files
        action_add_files = QAction("➕ Thêm ảnh", self)
        action_add_files.setShortcut(QKeySequence.Open)
        action_add_files.triggered.connect(self.image_list.prompt_add_files)
        toolbar.addAction(action_add_files)

        # Add Folder
        action_add_folder = QAction("📁 Thêm thư mục", self)
        action_add_folder.triggered.connect(self.image_list.prompt_add_folder)
        toolbar.addAction(action_add_folder)

        toolbar.addSeparator()

        # Auto-Crop & Flatten
        action_crop_flatten = QAction("✂️ Cắt viền & Nắn phẳng", self)
        action_crop_flatten.setToolTip("Tự động nhận diện 4 góc giấy, cắt bỏ viền nền thừa (bàn, sàn gỗ) và nắn phẳng trang giấy")
        action_crop_flatten.setShortcut(QKeySequence("F4"))
        action_crop_flatten.triggered.connect(self._toggle_crop_flatten_current)
        toolbar.addAction(action_crop_flatten)

        toolbar.addSeparator()

        # Scan Current OCR
        action_scan_current = QAction("🔍 Quét OCR trang này", self)
        action_scan_current.setShortcut(QKeySequence("F5"))
        action_scan_current.triggered.connect(self._run_ocr_current)
        toolbar.addAction(action_scan_current)

        # Scan All OCR
        action_scan_all = QAction("⚡ Quét OCR tất cả", self)
        action_scan_all.setShortcut(QKeySequence("F6"))
        action_scan_all.triggered.connect(self._run_ocr_all)
        toolbar.addAction(action_scan_all)

        toolbar.addSeparator()

        # Export PDF
        action_export_pdf = QAction("🚀 XUẤT PDF SIÊU PHÂN GIẢI", self)
        action_export_pdf.setShortcut(QKeySequence("Ctrl+E"))
        action_export_pdf.triggered.connect(self._open_export_dialog)
        toolbar.addAction(action_export_pdf)

        toolbar.addSeparator()

        # Help / About
        action_about = QAction("ℹ️ Giới thiệu", self)
        action_about.triggered.connect(self._show_about)
        toolbar.addAction(action_about)

    def _init_statusbar(self):
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.lbl_page_info = QLabel("Chưa nạp trang nào")
        self.lbl_page_info.setStyleSheet("color: #a1a1aa; padding-right: 15px;")

        self.lbl_dimension_info = QLabel("")
        self.lbl_dimension_info.setStyleSheet("color: #60a5fa; padding-right: 15px;")

        self.status_bar.addWidget(self.lbl_page_info)
        self.status_bar.addWidget(self.lbl_dimension_info)
        self.status_bar.showMessage("Sẵn sàng. Hãy kéo thả ảnh hoặc bấm 'Thêm ảnh' để bắt đầu.")

    def _connect_signals(self):
        # Image list events
        self.image_list.page_selected.connect(self._on_page_selected)
        self.image_list.pages_changed.connect(self._on_pages_changed)

        # Settings panel events
        self.settings_panel.params_changed.connect(self._on_params_changed)
        self.settings_panel.apply_all_requested.connect(self._on_apply_all_requested)

        # OCR panel events
        self.ocr_panel.scan_current_requested.connect(self._run_ocr_current)
        self.ocr_panel.scan_all_requested.connect(self._run_ocr_all)

    def _on_pages_changed(self):
        total = len(self.image_list.items)
        if total == 0:
            self.preview.set_images(None, None)
            self.ocr_panel.set_ocr_result(None)
            self.lbl_page_info.setText("Chưa nạp trang nào")
            self.lbl_dimension_info.setText("")
        else:
            cur = self.image_list.get_current_index() + 1
            self.lbl_page_info.setText(f"Trang {cur} / {total}")

    def _on_page_selected(self, index: int):
        if index < 0 or index >= len(self.image_list.items):
            self.preview.set_images(None, None)
            self.ocr_panel.set_ocr_result(None)
            self.lbl_page_info.setText("Chưa chọn trang")
            self.lbl_dimension_info.setText("")
            return

        item = self.image_list.items[index]
        self.lbl_page_info.setText(f"Trang {index + 1} / {len(self.image_list.items)}")

        # Sync settings panel sliders with current item's params
        self.settings_panel.set_params(item.params)

        # Load cached OCR
        self.ocr_panel.set_ocr_result(item.ocr_result)

        # Re-render enhancement for this page
        self._trigger_enhancement(index)

    def _on_params_changed(self, params: EnhanceParams):
        item = self.image_list.get_current_item()
        if item:
            item.params = params
            item.params.rotation = item.rotation
            # Invalidate cached enhanced image & OCR
            item.enhanced_image = None
            item.ocr_result = None
            self._trigger_enhancement(self.image_list.get_current_index())

    def _toggle_crop_flatten_current(self):
        idx = self.image_list.get_current_index()
        if idx < 0 or idx >= len(self.image_list.items):
            QMessageBox.information(self, "Thông báo", "Vui lòng thêm hoặc chọn một trang tài liệu để nắn phẳng.")
            return

        params = self.settings_panel.get_params()
        params.auto_flatten = not params.auto_flatten
        self.settings_panel.set_params(params, emit_signal=True)
        msg = "✂️ ĐÃ BẬT tự động cắt viền thừa & nắn phẳng." if params.auto_flatten else "ĐÃ TẮT tự động cắt viền."
        self.status_bar.showMessage(msg, 3000)

    def _on_apply_all_requested(self, params: EnhanceParams):
        total = len(self.image_list.items)
        if total == 0:
            return

        for item in self.image_list.items:
            # Copy params but preserve individual rotation
            rot = item.rotation
            item.params = EnhanceParams(
                sharpness=params.sharpness,
                contrast=params.contrast,
                whitening=params.whitening,
                denoise=params.denoise,
                upscale_factor=params.upscale_factor,
                color_mode=params.color_mode,
                auto_flatten=params.auto_flatten,
                rotation=rot
            )
            item.enhanced_image = None
            item.ocr_result = None

        self.status_bar.showMessage(f"Đã áp dụng thông số làm nét cho toàn bộ {total} trang.", 3000)
        self._trigger_enhancement(self.image_list.get_current_index())

    def _trigger_enhancement(self, index: int):
        if index < 0 or index >= len(self.image_list.items):
            return

        item = self.image_list.items[index]

        # Check if already cached - instant 0ms display!
        if item.enhanced_image is not None:
            self._apply_enhanced_result(index, item.enhanced_image)
            return

        # Cancel any previous enhance workers gracefully (never terminate!)
        for w in self._active_enhance_workers:
            w.cancel()

        self._enhance_req_id += 1
        req_id = self._enhance_req_id

        self.status_bar.showMessage("Đang làm nét chữ và tăng siêu phân giải...")
        worker = AsyncEnhanceWorker(req_id, index, item.original_image, item.params, parent=self)
        self._active_enhance_workers.append(worker)
        worker.enhanced_ready.connect(self._on_enhance_ready)
        worker.finished.connect(lambda w=worker: self._cleanup_enhance_worker(w))
        worker.start()

    def _on_enhance_ready(self, req_id: int, index: int, enhanced: np.ndarray):
        # Discard stale results from previous requests
        if req_id != self._enhance_req_id:
            return

        if 0 <= index < len(self.image_list.items):
            self.image_list.items[index].enhanced_image = enhanced

            # Only display if still on this page
            if index == self.image_list.get_current_index():
                self._apply_enhanced_result(index, enhanced)

    def _apply_enhanced_result(self, index: int, enhanced: np.ndarray):
        item = self.image_list.items[index]

        # Rotate original if needed for preview comparison
        orig_disp = item.original_image
        if item.rotation != 0:
            orig_disp = DocumentEnhancer.rotate_image(orig_disp, item.rotation)

        self.preview.set_images(orig_disp, enhanced)

        orig_h, orig_w = orig_disp.shape[:2]
        enh_h, enh_w = enhanced.shape[:2]
        ratio = enh_w / orig_w
        self.lbl_dimension_info.setText(
            f"Kích thước gốc: {orig_w}x{orig_h} ➜ Siêu nét: {enh_w}x{enh_h} ({ratio:.1f}x)"
        )
        self.status_bar.showMessage("Đã xử lý làm nét xong.", 2000)

    def _run_ocr_current(self):
        item = self.image_list.get_current_item()
        if not item:
            QMessageBox.information(self, "Thông báo", "Vui lòng thêm hoặc chọn một trang để quét OCR.")
            return

        self.tabs.setCurrentWidget(self.ocr_panel)

        # Ensure we have enhanced image
        if item.enhanced_image is None:
            item.enhanced_image = DocumentEnhancer.process(item.original_image, item.params)

        # Cancel existing single OCR workers if running
        for w in self._active_ocr_workers:
            w.cancel()

        engine_mode = self.ocr_panel.get_selected_engine()
        if engine_mode == "gemini":
            from core.config_manager import ConfigManager
            api_key = ConfigManager.get_instance().get_gemini_api_key()
            if not api_key:
                QMessageBox.warning(
                    self,
                    "Cần Gemini API Key",
                    "Chế độ 'AI Vision Thông Minh (Google Gemini)' cần API Key để hoạt động.\n\n"
                    "Vui lòng nhập API Key tại ô bên dưới mục chọn chế độ OCR và bấm 'Lưu' trước khi quét.\n"
                    "(Bạn có thể lấy API Key miễn phí tại https://aistudio.google.com)"
                )
                return
            mode_label = "Google Gemini Vision AI"
        elif engine_mode == "vietnamese":
            mode_label = "Tiếng Việt Hybrid AI"
        elif engine_mode == "vietocr":
            mode_label = "Viết tay VietOCR"
        else:
            mode_label = "Đa ngôn ngữ RapidOCR"
        # For OCR: use clean original image (rotated if user adjusted orientation)
        ocr_image = item.original_image
        if item.rotation != 0:
            ocr_image = DocumentEnhancer.rotate_image(ocr_image, item.rotation)

        self._ocr_req_id += 1
        req_id = self._ocr_req_id

        worker = AsyncSingleOCRWorker(req_id, ocr_image, engine_mode, parent=self)
        self._active_ocr_workers.append(worker)
        worker.progress.connect(self.ocr_panel.set_progress)
        worker.finished.connect(self._on_single_ocr_finished)
        worker.finished.connect(lambda w=worker: self._cleanup_ocr_worker(w))
        worker.start()

    def _on_single_ocr_finished(self, req_id: int, res: OCRResult):
        if req_id != self._ocr_req_id:
            return
        self.ocr_panel.set_progress(0, 0)
        item = self.image_list.get_current_item()
        if item:
            item.ocr_result = res
            self.ocr_panel.set_ocr_result(res)

        if res.error:
            self.status_bar.showMessage(f"Lỗi nhận diện: {res.error}", 4000)
            QMessageBox.warning(self, "Lỗi OCR", f"Không thể nhận diện văn bản: {res.error}")
        else:
            self.status_bar.showMessage(
                f"Đã nhận diện xong: {res.word_count} từ, {res.char_count} ký tự ({res.elapse_time}s)",
                4000
            )

    def _run_ocr_all(self):
        if not self.image_list.items:
            QMessageBox.information(self, "Thông báo", "Vui lòng thêm ít nhất một trang để quét OCR.")
            return

        self.tabs.setCurrentWidget(self.ocr_panel)
        engine_mode = self.ocr_panel.get_selected_engine()
        if engine_mode == "gemini":
            from core.config_manager import ConfigManager
            api_key = ConfigManager.get_instance().get_gemini_api_key()
            if not api_key:
                QMessageBox.warning(
                    self,
                    "Cần Gemini API Key",
                    "Chế độ 'AI Vision Thông Minh (Google Gemini)' cần API Key để hoạt động.\n\n"
                    "Vui lòng nhập API Key tại ô bên dưới mục chọn chế độ OCR và bấm 'Lưu' trước khi quét.\n"
                    "(Bạn có thể lấy API Key miễn phí tại https://aistudio.google.com)"
                )
                return

        self.batch_ocr_worker = AsyncBatchOCRWorker(self.image_list.items, mode=engine_mode, parent=self)
        self.batch_ocr_worker.progress.connect(self.ocr_panel.set_progress)
        self.batch_ocr_worker.finished.connect(self._on_batch_ocr_finished)
        self.batch_ocr_worker.start()

    def _on_batch_ocr_finished(self):
        self.ocr_panel.set_progress(0, 0)
        item = self.image_list.get_current_item()
        if item and item.ocr_result:
            self.ocr_panel.set_ocr_result(item.ocr_result)
        self.status_bar.showMessage("Đã hoàn tất quét OCR cho toàn bộ tài liệu!", 4000)
        QMessageBox.information(self, "Thành công", "Đã hoàn thành quét OCR cho tất cả các trang!")

    def _open_export_dialog(self):
        if not self.image_list.items:
            QMessageBox.warning(self, "Thông báo", "Vui lòng thêm ít nhất 1 hình ảnh trước khi xuất PDF!")
            return

        dialog = ExportDialog(self.image_list.items, self)
        dialog.exec()

    def _show_about(self):
        msg = (
            "<h3>Super OCR & High-Res PDF Studio</h3>"
            "<p><b>Phần mềm phục chế làm nét văn bản, quét OCR và xuất PDF siêu phân giải</b></p>"
            "<ul>"
            "<li><b>Làm nét chữ:</b> Unsharp Masking, CLAHE, lọc viền chi tiết, khử nhòe mờ</li>"
            "<li><b>Siêu phân giải:</b> Phóng to 2x, 3x, 4x với thuật toán Lanczos-4 không vỡ hạt</li>"
            "<li><b>Tẩy trắng nền:</b> Khử bóng đổ, làm trắng trang giấy sạch sẽ</li>"
            "<li><b>Quét OCR Đa Năng:</b> Nhận diện tiếng Việt siêu tốc (Hybrid AI) & AI Vision Thông minh (Google Gemini - chuẩn xác 100% viết tay)</li>"
            "<li><b>Xuất PDF:</b> Hỗ trợ Searchable PDF (có lớp chữ ẩn tìm kiếm/copy được) và High-Res Image PDF</li>"
            "</ul>"
            "<p><i>Phát triển bởi Fami (fami_7006)</i></p>"
        )
        QMessageBox.about(self, "Giới thiệu ứng dụng", msg)

    def closeEvent(self, event):
        for w in self._active_enhance_workers:
            w.cancel()
            w.wait(400)
        self._active_enhance_workers.clear()

        if self.batch_ocr_worker and self.batch_ocr_worker.isRunning():
            self.batch_ocr_worker.cancel()
            self.batch_ocr_worker.wait(500)

        for w in self._active_ocr_workers:
            w.cancel()
            w.wait(400)
        self._active_ocr_workers.clear()

        event.accept()
