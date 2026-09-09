"""
Interactive High-DPI Document Preview Widget with Before/After Split Slider,
smooth Zoom, Pan, and side-by-side comparison.
"""

from typing import Optional
import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QRect
from PySide6.QtGui import (
    QPainter, QPixmap, QImage, QColor, QPen, QBrush, QFont,
    QMouseEvent, QWheelEvent, QPaintEvent, QCursor
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QButtonGroup, QFrame
)


class CanvasWidget(QWidget):
    """Internal drawing canvas for Split-view rendering with zoom and pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Image references
        self.original_qimg: Optional[QImage] = None
        self.enhanced_qimg: Optional[QImage] = None

        # View settings
        self.split_position: float = 0.5   # 0.0 (all enhanced) to 1.0 (all original)
        self.view_mode: str = "split"      # "split", "enhanced", "original"
        self.scale: float = 1.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0

        # Mouse interaction
        self.is_dragging_slider: bool = False
        self.is_panning: bool = False
        self.last_mouse_pos = QPointF()

    def set_images(self, orig_bgr: Optional[np.ndarray], enh_bgr: Optional[np.ndarray]):
        """Update source images and repaint."""
        if orig_bgr is not None:
            rgb_orig = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_orig.shape
            self.original_qimg = QImage(rgb_orig.data, w, h, ch * w, QImage.Format_RGB888).copy()
        else:
            self.original_qimg = None

        if enh_bgr is not None:
            rgb_enh = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_enh.shape
            self.enhanced_qimg = QImage(rgb_enh.data, w, h, ch * w, QImage.Format_RGB888).copy()
        else:
            self.enhanced_qimg = None

        self.reset_view_fit()
        self.update()

    def reset_view_fit(self):
        """Scale and center image to fit within canvas."""
        ref_img = self.enhanced_qimg or self.original_qimg
        if ref_img is None or ref_img.isNull():
            self.scale = 1.0
            self.offset_x = 0.0
            self.offset_y = 0.0
            self.update()
            return

        cw = self.width()
        ch = self.height()
        iw = ref_img.width()
        ih = ref_img.height()

        if cw > 20 and ch > 20:
            scale_x = (cw - 40) / iw
            scale_y = (ch - 40) / ih
            self.scale = max(0.05, min(scale_x, scale_y, 1.5))
            self.offset_x = (cw - iw * self.scale) / 2.0
            self.offset_y = (ch - ih * self.scale) / 2.0
        self.update()

    def reset_view_1to1(self):
        """Set zoom to 100% actual pixels."""
        ref_img = self.enhanced_qimg or self.original_qimg
        if ref_img is None:
            return
        cw = self.width()
        ch = self.height()
        self.scale = 1.0
        self.offset_x = (cw - ref_img.width()) / 2.0
        self.offset_y = (ch - ref_img.height()) / 2.0
        self.update()

    def set_view_mode(self, mode: str):
        self.view_mode = mode
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Background canvas fill
        painter.fillRect(self.rect(), QColor("#141417"))

        ref_img = self.enhanced_qimg or self.original_qimg
        if ref_img is None or ref_img.isNull():
            # Placeholder text
            painter.setPen(QColor("#71717a"))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "Chưa có hình ảnh nào được chọn\nHãy thêm hoặc kéo thả ảnh vào ứng dụng")
            return

        # Calculate bounding box on canvas
        disp_w = ref_img.width() * self.scale
        disp_h = ref_img.height() * self.scale
        target_rect = QRectF(self.offset_x, self.offset_y, disp_w, disp_h)

        # Draw subtle shadow / border around page
        painter.setPen(QColor("#2e2e38"))
        painter.setBrush(QColor("#1e1e24"))
        painter.drawRect(target_rect.adjusted(-1, -1, 1, 1))

        if self.view_mode == "original" or self.enhanced_qimg is None:
            # Draw original only
            if self.original_qimg:
                painter.drawImage(target_rect, self.original_qimg)
            self._draw_badge(painter, "ẢNH GỐC", target_rect.left() + 15, target_rect.top() + 15, "#eab308")

        elif self.view_mode == "enhanced" or self.original_qimg is None:
            # Draw enhanced only
            if self.enhanced_qimg:
                painter.drawImage(target_rect, self.enhanced_qimg)
            self._draw_badge(painter, "ĐÃ LÀM NÉT", target_rect.left() + 15, target_rect.top() + 15, "#10b981")

        else:
            # SPLIT VIEW: Draw Original on left, Enhanced on right
            split_x = target_rect.left() + target_rect.width() * self.split_position

            # 1. Draw Enhanced Image across entire rect
            painter.drawImage(target_rect, self.enhanced_qimg)

            # 2. Clip left region and draw Original Image
            painter.save()
            left_clip = QRectF(
                target_rect.left(),
                target_rect.top(),
                target_rect.width() * self.split_position,
                target_rect.height()
            )
            painter.setClipRect(left_clip)
            painter.drawImage(target_rect, self.original_qimg)
            painter.restore()

            # 3. Draw Divider Line
            divider_pen = QPen(QColor("#3b82f6"), 2, Qt.SolidLine)
            painter.setPen(divider_pen)
            painter.drawLine(QPointF(split_x, target_rect.top()), QPointF(split_x, target_rect.bottom()))

            # 4. Draw Center Circular Handle
            center_y = target_rect.top() + target_rect.height() / 2.0
            painter.setPen(QPen(QColor("#1e3a8a"), 2))
            painter.setBrush(QBrush(QColor("#3b82f6")))
            painter.drawEllipse(QPointF(split_x, center_y), 14, 14)

            # Arrow icon inside handle
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(QRectF(split_x - 14, center_y - 14, 28, 28), Qt.AlignCenter, "◀ ▶")

            # 5. Draw Badges
            self._draw_badge(painter, "ẢNH GỐC (TRƯỚC)", target_rect.left() + 15, target_rect.top() + 15, "#eab308")
            self._draw_badge(painter, "ĐÃ LÀM NÉT (SAU)", target_rect.right() - 150, target_rect.top() + 15, "#10b981")

    def _draw_badge(self, painter: QPainter, text: str, x: float, y: float, color_hex: str):
        """Draw semi-transparent badge indicating image version."""
        painter.save()
        font = QFont("Segoe UI", 10, QFont.Bold)
        painter.setFont(font)
        badge_w = 135
        badge_h = 26
        badge_rect = QRectF(x, y, badge_w, badge_h)

        # Background pill
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(24, 24, 27, 210))
        painter.drawRoundedRect(badge_rect, 6, 6)

        # Text with color
        painter.setPen(QColor(color_hex))
        painter.drawText(badge_rect, Qt.AlignCenter, text)
        painter.restore()

    def mousePressEvent(self, event: QMouseEvent):
        ref_img = self.enhanced_qimg or self.original_qimg
        if ref_img is None:
            return

        disp_w = ref_img.width() * self.scale
        target_rect = QRectF(self.offset_x, self.offset_y, disp_w, ref_img.height() * self.scale)
        split_x = target_rect.left() + target_rect.width() * self.split_position

        if event.button() == Qt.LeftButton:
            # Check if clicked near split slider handle
            if self.view_mode == "split" and abs(event.position().x() - split_x) < 25:
                self.is_dragging_slider = True
                self.setCursor(Qt.SplitHCursor)
            else:
                self.is_panning = True
                self.last_mouse_pos = event.position()
                self.setCursor(Qt.ClosedHandCursor)
        elif event.button() in [Qt.MiddleButton, Qt.RightButton]:
            self.is_panning = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        ref_img = self.enhanced_qimg or self.original_qimg
        if ref_img is None:
            return

        disp_w = ref_img.width() * self.scale
        target_rect = QRectF(self.offset_x, self.offset_y, disp_w, ref_img.height() * self.scale)
        split_x = target_rect.left() + target_rect.width() * self.split_position

        if self.is_dragging_slider:
            new_pos = (event.position().x() - target_rect.left()) / max(1.0, target_rect.width())
            self.split_position = max(0.01, min(0.99, new_pos))
            self.update()

        elif self.is_panning:
            delta = event.position() - self.last_mouse_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_mouse_pos = event.position()
            self.update()

        else:
            # Update hover cursor
            if self.view_mode == "split" and abs(event.position().x() - split_x) < 20:
                self.setCursor(Qt.SplitHCursor)
            else:
                self.setCursor(Qt.OpenHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.is_dragging_slider = False
        self.is_panning = False
        self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom smoothly towards mouse position."""
        ref_img = self.enhanced_qimg or self.original_qimg
        if ref_img is None:
            return

        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else 1.0 / 1.15

        old_scale = self.scale
        new_scale = max(0.05, min(8.0, self.scale * factor))
        self.scale = new_scale

        # Zoom relative to mouse cursor position
        mouse_pos = event.position()
        self.offset_x = mouse_pos.x() - (mouse_pos.x() - self.offset_x) * (new_scale / old_scale)
        self.offset_y = mouse_pos.y() - (mouse_pos.y() - self.offset_y) * (new_scale / old_scale)

        self.update()


class PreviewWidget(QWidget):
    """Container widget providing the canvas and top toolbar controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Control Bar
        toolbar = QFrame()
        toolbar.setStyleSheet("background-color: #1e1e24; border-bottom: 1px solid #2e2e38; padding: 4px;")
        t_layout = QHBoxLayout(toolbar)
        t_layout.setContentsMargins(8, 4, 8, 4)
        t_layout.setSpacing(6)

        # View Mode Buttons
        self.btn_split = QPushButton("⚡ So sánh Trước/Sau")
        self.btn_split.setCheckable(True)
        self.btn_split.setChecked(True)
        self.btn_split.clicked.connect(lambda: self._set_mode("split"))

        self.btn_enhanced = QPushButton("✨ Chỉ xem nét")
        self.btn_enhanced.setCheckable(True)
        self.btn_enhanced.clicked.connect(lambda: self._set_mode("enhanced"))

        self.btn_orig = QPushButton("📷 Ảnh gốc")
        self.btn_orig.setCheckable(True)
        self.btn_orig.clicked.connect(lambda: self._set_mode("original"))

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.btn_split)
        self.mode_group.addButton(self.btn_enhanced)
        self.mode_group.addButton(self.btn_orig)

        t_layout.addWidget(self.btn_split)
        t_layout.addWidget(self.btn_enhanced)
        t_layout.addWidget(self.btn_orig)

        t_layout.addStretch()

        # Zoom buttons
        self.btn_fit = QPushButton("⛶ Vừa màn hình")
        self.btn_fit.clicked.connect(self._fit_screen)

        self.btn_100 = QPushButton("1:1 Thực tế")
        self.btn_100.clicked.connect(self._actual_size)

        self.btn_zoom_in = QPushButton("🔍+")
        self.btn_zoom_in.clicked.connect(self._zoom_in)

        self.btn_zoom_out = QPushButton("🔍-")
        self.btn_zoom_out.clicked.connect(self._zoom_out)

        t_layout.addWidget(self.btn_fit)
        t_layout.addWidget(self.btn_100)
        t_layout.addWidget(self.btn_zoom_in)
        t_layout.addWidget(self.btn_zoom_out)

        layout.addWidget(toolbar)

        # Canvas
        self.canvas = CanvasWidget(self)
        layout.addWidget(self.canvas)

    def set_images(self, orig_bgr: Optional[np.ndarray], enh_bgr: Optional[np.ndarray]):
        self.canvas.set_images(orig_bgr, enh_bgr)

    def _set_mode(self, mode: str):
        self.canvas.set_view_mode(mode)

    def _fit_screen(self):
        self.canvas.reset_view_fit()

    def _actual_size(self):
        self.canvas.reset_view_1to1()

    def _zoom_in(self):
        self.canvas.scale = min(8.0, self.canvas.scale * 1.25)
        self.canvas.update()

    def _zoom_out(self):
        self.canvas.scale = max(0.05, self.canvas.scale / 1.25)
        self.canvas.update()
