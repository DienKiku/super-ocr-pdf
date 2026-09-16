"""
Image List Sidebar Widget with thumbnails, drag-and-drop reordering,
rotation, and page management.
"""

from typing import List, Optional, Tuple
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageSequence
import pymupdf

try:
    import pillow_jxl  # noqa: F401
except ImportError:
    pass

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap, QImage, QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFileDialog, QAbstractItemView, QMessageBox, QApplication
)

from core.enhancer import EnhanceParams
from core.ocr_engine import OCRResult

SUPPORTED_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".jfif", ".jpe", ".pjpeg", ".pjp", ".gif",
    ".jxl"
}
SUPPORTED_MULTI_EXTS = {".tiff", ".tif", ".pdf"}
SUPPORTED_ALL_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_MULTI_EXTS


def natural_sort_key(s: str):
    """Sort strings containing numbers in natural human order (e.g. img_1, img_2, img_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def load_single_image_data(file_path: str, page_index: int = 0) -> np.ndarray:
    """Lazy load full resolution BGR image array for a single page on demand."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            doc = pymupdf.open(file_path)
            if 0 <= page_index < len(doc):
                page = doc[page_index]
                pix = page.get_pixmap(dpi=200)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                doc.close()
                if pix.n == 4:
                    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif pix.n == 1:
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            doc.close()
        except Exception as e:
            print(f"[ImageList] Error lazy-loading PDF page {file_path}: {e}")

    # Standard / Pillow formats (including JXL, TIFF, JFIF, PNG, WebP)
    try:
        with Image.open(file_path) as pil_img:
            if getattr(pil_img, "n_frames", 1) > 1 and page_index > 0:
                pil_img.seek(page_index)
            rgb = pil_img.convert("RGB")
            arr = np.array(rgb)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    try:
        img_data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    return np.full((600, 800, 3), 240, dtype=np.uint8)


def generate_fast_thumbnail(file_path: str, page_index: int = 0) -> Optional[QPixmap]:
    """Generates miniature thumbnail directly without keeping full-resolution array in RAM."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            doc = pymupdf.open(file_path)
            if 0 <= page_index < len(doc):
                page = doc[page_index]
                pix = page.get_pixmap(dpi=36)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                doc.close()
                if pix.n == 4:
                    rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif pix.n == 3:
                    rgb = img
                else:
                    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                return QPixmap.fromImage(qimg).scaled(90, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            doc.close()
        except Exception:
            pass

    try:
        with Image.open(file_path) as pil_img:
            if getattr(pil_img, "n_frames", 1) > 1 and page_index > 0:
                pil_img.seek(page_index)
            pil_img.thumbnail((90, 120), Image.Resampling.BILINEAR)
            rgb = pil_img.convert("RGB")
            arr = np.array(rgb)
            h, w, ch = arr.shape
            qimg = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)
    except Exception:
        pass

    return None


def load_document_pages(file_path: str) -> List[Tuple[str, Optional[np.ndarray], int, int]]:
    """
    Load all pages metadata/frames from a file (standard image, multi-page TIFF, PDF, JXL).
    Returns a list of tuples: (file_path, bgr_image_or_none, page_index, total_pages)
    """
    ext = os.path.splitext(file_path)[1].lower()
    pages: List[Tuple[str, Optional[np.ndarray], int, int]] = []

    # 1. PDF Documents via PyMuPDF
    if ext == ".pdf":
        try:
            doc = pymupdf.open(file_path)
            total = len(doc)
            for idx in range(total):
                page = doc[idx]
                pix = page.get_pixmap(dpi=200)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                if pix.n == 4:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif pix.n == 1:
                    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                pages.append((file_path, bgr, idx, total))
            doc.close()
            return pages
        except Exception as e:
            print(f"[ImageList] Error reading PDF {file_path}: {e}")
            return []

    # 2. Multi-page TIFF via Pillow
    if ext in [".tiff", ".tif"]:
        try:
            with Image.open(file_path) as pil_img:
                frames = [frame.copy() for frame in ImageSequence.Iterator(pil_img)]
                total = len(frames)
                for idx, frame in enumerate(frames):
                    rgb = frame.convert("RGB")
                    arr = np.array(rgb)
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    pages.append((file_path, bgr, idx, total))
            if pages:
                return pages
        except Exception as e:
            print(f"[ImageList] Warning loading TIFF with PIL: {e}")

    # 3. Standard Images (JPG, PNG, WEBP, JFIF, JXL, BMP, GIF, etc.)
    # Primary attempt: OpenCV with Unicode path support
    try:
        img_data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is not None:
            return [(file_path, img, 0, 1)]
    except Exception:
        pass

    # Secondary fallback: Pillow (handles JXL, JFIF, exotic PNG, WebP)
    try:
        with Image.open(file_path) as pil_img:
            n_frames = getattr(pil_img, "n_frames", 1)
            if n_frames > 1:
                frames = [frame.copy() for frame in ImageSequence.Iterator(pil_img)]
                total = len(frames)
                for idx, frame in enumerate(frames):
                    rgb = frame.convert("RGB")
                    arr = np.array(rgb)
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    pages.append((file_path, bgr, idx, total))
                return pages
            else:
                rgb = pil_img.convert("RGB")
                arr = np.array(rgb)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return [(file_path, bgr, 0, 1)]
    except Exception as e:
        print(f"[ImageList] Error loading image {file_path}: {e}")

    return []


class ImageItem:
    """Holds data for a single document page with lazy-loading and thumbnail caching."""
    def __init__(
        self,
        file_path: str,
        original_image: Optional[np.ndarray] = None,
        page_index: int = 0,
        total_pages: int = 1,
        thumbnail: Optional[QPixmap] = None
    ):
        self.file_path = file_path
        self._original_image = original_image
        self.page_index = page_index
        self.total_pages = total_pages
        self.rotation = 0  # 0, 90, 180, 270
        self.params = EnhanceParams()
        self.enhanced_image: Optional[np.ndarray] = None
        self.ocr_result: Optional[OCRResult] = None
        self._thumbnail_pixmap = thumbnail

    @property
    def original_image(self) -> np.ndarray:
        """Lazy loads full resolution image into RAM only when requested."""
        if self._original_image is None:
            self._original_image = load_single_image_data(self.file_path, self.page_index)
        return self._original_image

    @original_image.setter
    def original_image(self, val: np.ndarray):
        self._original_image = val

    @property
    def filename(self) -> str:
        base = os.path.basename(self.file_path) if self.file_path else "Untitled"
        if self.total_pages > 1:
            return f"{base} (Trang {self.page_index + 1}/{self.total_pages})"
        return base


class ImageListWidget(QWidget):
    """Sidebar widget managing the loaded document images."""

    # Signals
    page_selected = Signal(int)       # page index
    pages_changed = Signal()          # count or list changed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: List[ImageItem] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header Title
        header_layout = QHBoxLayout()
        title = QLabel("📑 DANH SÁCH TRANG")
        title.setStyleSheet("font-weight: bold; color: #93c5fd; font-size: 13px;")
        self.count_label = QLabel("(0 trang)")
        self.count_label.setStyleSheet("color: #71717a; font-size: 12px;")
        header_layout.addWidget(title)
        header_layout.addWidget(self.count_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Action Buttons Row 1: Add Files & Add Folder
        btn_row1 = QHBoxLayout()
        self.btn_add_files = QPushButton("➕ Thêm tệp")
        self.btn_add_files.setObjectName("primary_btn")
        self.btn_add_files.setToolTip("Thêm tệp hình ảnh hoặc tài liệu PDF")
        self.btn_add_files.clicked.connect(self.prompt_add_files)

        self.btn_add_folder = QPushButton("📁 Thư mục")
        self.btn_add_folder.setToolTip("Thêm tất cả hình ảnh & PDF trong thư mục")
        self.btn_add_folder.clicked.connect(self.prompt_add_folder)

        btn_row1.addWidget(self.btn_add_files)
        btn_row1.addWidget(self.btn_add_folder)
        layout.addLayout(btn_row1)

        # List Widget for Thumbnails
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(90, 120))
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.list_widget.model().rowsMoved.connect(self._on_rows_reordered)
        self.setAcceptDrops(True)
        layout.addWidget(self.list_widget)

        # Action Buttons Row 2: Rotate & Move
        btn_row2 = QHBoxLayout()
        self.btn_rotate_left = QPushButton("↺ 90°")
        self.btn_rotate_left.setToolTip("Xoay trái 90 độ")
        self.btn_rotate_left.clicked.connect(lambda: self.rotate_current(-90))

        self.btn_rotate_right = QPushButton("↻ 90°")
        self.btn_rotate_right.setToolTip("Xoay phải 90 độ")
        self.btn_rotate_right.clicked.connect(lambda: self.rotate_current(90))

        self.btn_move_up = QPushButton("▲ Lên")
        self.btn_move_up.setToolTip("Chuyển trang lên trên")
        self.btn_move_up.clicked.connect(self.move_current_up)

        self.btn_move_down = QPushButton("▼ Xuống")
        self.btn_move_down.setToolTip("Chuyển trang xuống dưới")
        self.btn_move_down.clicked.connect(self.move_current_down)

        btn_row2.addWidget(self.btn_rotate_left)
        btn_row2.addWidget(self.btn_rotate_right)
        btn_row2.addWidget(self.btn_move_up)
        btn_row2.addWidget(self.btn_move_down)
        layout.addLayout(btn_row2)

        # Action Buttons Row 3: Delete & Clear
        btn_row3 = QHBoxLayout()
        self.btn_delete = QPushButton("🗑️ Xóa trang")
        self.btn_delete.setObjectName("danger_btn")
        self.btn_delete.clicked.connect(self.delete_current)

        self.btn_clear = QPushButton("Dọn sạch")
        self.btn_clear.clicked.connect(self.clear_all)

        btn_row3.addWidget(self.btn_delete)
        btn_row3.addWidget(self.btn_clear)
        layout.addLayout(btn_row3)

    # Drag & Drop files into the list
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_paths = []
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if os.path.isfile(local_path):
                    ext = os.path.splitext(local_path)[1].lower()
                    if ext in SUPPORTED_ALL_EXTS:
                        file_paths.append(local_path)
                elif os.path.isdir(local_path):
                    for root, dirs, files in os.walk(local_path):
                        dirs.sort(key=natural_sort_key)
                        for f in sorted(files, key=natural_sort_key):
                            ext = os.path.splitext(f)[1].lower()
                            if ext in SUPPORTED_ALL_EXTS:
                                file_paths.append(os.path.join(root, f))
            if file_paths:
                file_paths.sort(key=lambda p: (os.path.dirname(p), natural_sort_key(os.path.basename(p))))
                self.add_images(file_paths)
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def prompt_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn hình ảnh văn bản / tài liệu / PDF / JPEG XL",
            "",
            "Tất cả tài liệu hỗ trợ (*.jpg *.jpeg *.png *.bmp *.webp *.jfif *.jxl *.tiff *.tif *.pdf);;"
            "Hình ảnh (*.jpg *.jpeg *.png *.bmp *.webp *.jfif *.jxl *.tiff *.tif);;"
            "Tài liệu PDF (*.pdf);;"
            "Tất cả (*.*)"
        )
        if files:
            files.sort(key=lambda p: (os.path.dirname(p), natural_sort_key(os.path.basename(p))))
            self.add_images(files)

    def prompt_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa tài liệu / ảnh / JPEG XL")
        if folder:
            file_paths = []
            for root, dirs, files in os.walk(folder):
                dirs.sort(key=natural_sort_key)
                for f in sorted(files, key=natural_sort_key):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORTED_ALL_EXTS:
                        file_paths.append(os.path.join(root, f))
            if not file_paths:
                QMessageBox.information(
                    self,
                    "Không tìm thấy tài liệu",
                    "Thư mục đã chọn không chứa tệp hình ảnh, JPEG XL hoặc PDF hợp lệ nào."
                )
                return

            # Nếu thư mục có số lượng tệp rất lớn (như tuning_data > 150 tệp)
            if len(file_paths) > 150:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Thư mục số lượng lớn")
                msg_box.setText(f"Thư mục chứa <b>{len(file_paths):,}</b> tệp tài liệu / ảnh.")
                msg_box.setInformativeText("Bạn có muốn nạp toàn bộ danh sách (với thanh tiến trình) hay nạp nhanh 100 tệp đầu để xem thử?")
                btn_all = msg_box.addButton(f"Nạp toàn bộ ({len(file_paths):,} tệp)", QMessageBox.AcceptRole)
                btn_100 = msg_box.addButton("Nạp 100 tệp đầu", QMessageBox.ActionRole)
                btn_cancel = msg_box.addButton("Hủy", QMessageBox.RejectRole)
                msg_box.setDefaultButton(btn_all)
                msg_box.exec()

                clicked = msg_box.clickedButton()
                if clicked == btn_cancel:
                    return
                elif clicked == btn_100:
                    file_paths = file_paths[:100]

            self.add_images(file_paths)

    def add_images(self, paths: List[str]):
        """Load and add images/documents into list with memory-efficient pagination."""
        added_any = False
        initial_index = self.list_widget.currentRow()
        total_p = len(paths)

        # Thanh tiến trình nếu số lượng tệp lớn
        progress = None
        if total_p > 25:
            from PySide6.QtWidgets import QProgressDialog
            progress = QProgressDialog("Đang nạp danh sách tài liệu...", "Hủy", 0, total_p, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(300)

        for i, p in enumerate(paths):
            if progress:
                if progress.wasCanceled():
                    break
                progress.setValue(i)
                progress.setLabelText(f"Đang nạp tài liệu ({i + 1}/{total_p}):\n{os.path.basename(p)}")

            pages = load_document_pages(p)
            for file_path, img, page_idx, total_pages in pages:
                thumb = generate_fast_thumbnail(file_path, page_idx)
                # Nếu số lượng lớn (> 25 tệp), giải phóng bộ nhớ RAM bằng cách nạp lười (lazy load)
                stored_img = img if total_p <= 25 else None
                item = ImageItem(file_path, stored_img, page_idx, total_pages, thumbnail=thumb)
                self.items.append(item)
                added_any = True

            if i % 8 == 0:
                QApplication.processEvents()

        if progress:
            progress.close()

        if added_any:
            self._rebuild_list_ui()
            if initial_index < 0 and self.items:
                self.list_widget.setCurrentRow(0)
            self.pages_changed.emit()

    def _rebuild_list_ui(self, selected_row: Optional[int] = None):
        """Refreshes list widget thumbnails and titles with cached miniature icons."""
        # Update page count label
        self.count_label.setText(f"({len(self.items)} trang)")

        self.list_widget.blockSignals(True)
        curr_row = selected_row if selected_row is not None else self.list_widget.currentRow()
        self.list_widget.clear()

        for idx, it in enumerate(self.items):
            if it._thumbnail_pixmap is not None and it.rotation == 0:
                thumb_pixmap = it._thumbnail_pixmap
            else:
                thumb_pixmap = self._generate_thumbnail(it.original_image, it.rotation, idx + 1)
                it._thumbnail_pixmap = thumb_pixmap

            list_item = QListWidgetItem(QIcon(thumb_pixmap), f"Trang {idx + 1}\n{it.filename}")
            list_item.setSizeHint(QSize(180, 100))
            list_item.setData(Qt.UserRole, idx)  # Store original index for drag-drop sync
            self.list_widget.addItem(list_item)

        target_row = -1
        if 0 <= curr_row < len(self.items):
            target_row = curr_row
        elif self.items:
            target_row = 0

        self.list_widget.blockSignals(False)
        if target_row >= 0:
            if self.list_widget.currentRow() != target_row:
                self.list_widget.setCurrentRow(target_row)
            else:
                self.page_selected.emit(target_row)

    def _generate_thumbnail(self, img: np.ndarray, rotation: int, page_num: int) -> QPixmap:
        """Create a crisp miniature thumbnail with a page badge."""
        # Rotate if needed
        disp_img = img
        if rotation != 0:
            from core.enhancer import DocumentEnhancer
            disp_img = DocumentEnhancer.rotate_image(img, rotation)

        h, w = disp_img.shape[:2]
        thumb_h = 100
        thumb_w = int(w * (thumb_h / h))
        if thumb_w > 90:
            thumb_w = 90
            thumb_h = int(h * (thumb_w / w))

        small = cv2.resize(disp_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, thumb_w, thumb_h, 3 * thumb_w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        return pix

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self.items):
            self.page_selected.emit(row)

    def _on_rows_reordered(self, parent, start, end, destination, row):
        """Synchronize self.items order with list_widget order after drag-drop."""
        # After drag-drop, QListWidget has rearranged visually but self.items is stale.
        # Read the new order from the stored original indices in each list item's data.
        new_items = []
        for i in range(self.list_widget.count()):
            list_item = self.list_widget.item(i)
            orig_idx = list_item.data(Qt.UserRole)
            if orig_idx is not None and 0 <= orig_idx < len(self.items):
                new_items.append(self.items[orig_idx])

        if len(new_items) == len(self.items):
            self.items = new_items

        # Rebuild UI to fix numbering (Trang 1, Trang 2, ...) after reorder
        self._rebuild_list_ui()
        self.pages_changed.emit()

    def get_current_item(self) -> Optional[ImageItem]:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.items):
            return self.items[row]
        return None

    def get_current_index(self) -> int:
        return self.list_widget.currentRow()

    def rotate_current(self, delta_angle: int):
        item = self.get_current_item()
        if item:
            item.rotation = (item.rotation + delta_angle) % 360
            item.params.rotation = item.rotation
            item._thumbnail_pixmap = None
            # Invalidate cached enhanced image & OCR
            item.enhanced_image = None
            item.ocr_result = None
            self._rebuild_list_ui()
            self.page_selected.emit(self.list_widget.currentRow())
            self.pages_changed.emit()

    def move_current_up(self):
        row = self.list_widget.currentRow()
        if row > 0:
            self.items[row - 1], self.items[row] = self.items[row], self.items[row - 1]
            self._rebuild_list_ui(selected_row=row - 1)
            self.pages_changed.emit()

    def move_current_down(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.items) - 1:
            self.items[row + 1], self.items[row] = self.items[row], self.items[row + 1]
            self._rebuild_list_ui(selected_row=row + 1)
            self.pages_changed.emit()

    def delete_current(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.items):
            del self.items[row]
            new_row = min(row, len(self.items) - 1) if self.items else -1
            self._rebuild_list_ui(selected_row=new_row)
            if not self.items:
                self.page_selected.emit(-1)
            self.pages_changed.emit()

    def clear_all(self):
        if not self.items:
            return
        reply = QMessageBox.question(
            self,
            "Xác nhận dọn sạch",
            "Bạn có chắc muốn xóa toàn bộ danh sách trang hiện tại?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.items.clear()
            self._rebuild_list_ui()
            self.page_selected.emit(-1)
            self.pages_changed.emit()
