"""
Image List Sidebar Widget with thumbnails, drag-and-drop reordering,
rotation, and page management.
"""

from typing import List, Optional, Tuple
import os
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap, QImage, QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFileDialog, QAbstractItemView, QMessageBox
)

from core.enhancer import EnhanceParams
from core.ocr_engine import OCRResult


class ImageItem:
    """Holds data for a single document page."""
    def __init__(self, file_path: str, original_image: np.ndarray):
        self.file_path = file_path
        self.original_image = original_image
        self.rotation = 0  # 0, 90, 180, 270
        self.params = EnhanceParams()
        self.enhanced_image: Optional[np.ndarray] = None
        self.ocr_result: Optional[OCRResult] = None

    @property
    def filename(self) -> str:
        return os.path.basename(self.file_path) if self.file_path else "Untitled"


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
        self.btn_add_files = QPushButton("➕ Thêm ảnh")
        self.btn_add_files.setObjectName("primary_btn")
        self.btn_add_files.clicked.connect(self.prompt_add_files)

        self.btn_add_folder = QPushButton("📁 Thư mục")
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
                    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"]:
                        file_paths.append(local_path)
                elif os.path.isdir(local_path):
                    for root, _, files in os.walk(local_path):
                        for f in files:
                            ext = os.path.splitext(f)[1].lower()
                            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"]:
                                file_paths.append(os.path.join(root, f))
            if file_paths:
                self.add_images(file_paths)
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def prompt_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn hình ảnh văn bản / tài liệu",
            "",
            "Hình ảnh (*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif);;Tất cả (*.*)"
        )
        if files:
            self.add_images(files)

    def prompt_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa ảnh")
        if folder:
            file_paths = []
            for root, _, files in os.walk(folder):
                for f in sorted(files):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"]:
                        file_paths.append(os.path.join(root, f))
            if file_paths:
                self.add_images(file_paths)

    def add_images(self, paths: List[str]):
        """Load and add images into list."""
        added_any = False
        initial_index = self.list_widget.currentRow()

        for p in paths:
            # Read image using OpenCV with Unicode path support
            try:
                # np.fromfile handles Unicode Windows paths properly
                img_data = np.fromfile(p, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                item = ImageItem(p, img)
                self.items.append(item)
                added_any = True
            except Exception as e:
                print(f"Error loading {p}: {e}")

        if added_any:
            self._rebuild_list_ui()
            if initial_index < 0 and self.items:
                self.list_widget.setCurrentRow(0)
            self.pages_changed.emit()

    def _rebuild_list_ui(self):
        """Refreshes list widget thumbnails and titles."""
        self.list_widget.blockSignals(True)
        curr_row = self.list_widget.currentRow()
        self.list_widget.clear()

        for idx, it in enumerate(self.items):
            thumb_pixmap = self._generate_thumbnail(it.original_image, it.rotation, idx + 1)
            list_item = QListWidgetItem(QIcon(thumb_pixmap), f"Trang {idx + 1}\n{it.filename}")
            list_item.setSizeHint(QSize(180, 95))
            self.list_widget.addItem(list_item)

        if 0 <= curr_row < len(self.items):
            self.list_widget.setCurrentRow(curr_row)
        elif self.items:
            self.list_widget.setCurrentRow(0)

        self.list_widget.blockSignals(False)
        self.count_label.setText(f"({len(self.items)} trang)")

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
        qimg = QImage(rgb.data, thumb_w, thumb_h, 3 * thumb_w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        return pix

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self.items):
            self.page_selected.emit(row)

    def _on_rows_reordered(self, parent, start, end, destination, row):
        # Synchronize self.items order with list_widget order
        # When dragged, list_widget already rearranged items visually
        pass

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
            self._rebuild_list_ui()
            self.list_widget.setCurrentRow(row - 1)
            self.pages_changed.emit()

    def move_current_down(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.items) - 1:
            self.items[row + 1], self.items[row] = self.items[row], self.items[row + 1]
            self._rebuild_list_ui()
            self.list_widget.setCurrentRow(row + 1)
            self.pages_changed.emit()

    def delete_current(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.items):
            del self.items[row]
            self._rebuild_list_ui()
            if self.items:
                new_row = min(row, len(self.items) - 1)
                self.list_widget.setCurrentRow(new_row)
                self.page_selected.emit(new_row)
            else:
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
