"""
Settings and Adjustment Panel for Image Enhancement and Super-Resolution.
Controls sharpness, contrast, whitening, denoise, and upscale factors.
"""

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QPushButton, QGroupBox, QScrollArea,
    QFrame
)

from core.enhancer import EnhanceParams, PRESETS


class SettingsPanel(QWidget):
    """Panel with sliders and presets for tuning document clarity."""

    params_changed = Signal(EnhanceParams)
    apply_all_requested = Signal(EnhanceParams)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_params = EnhanceParams()
        self._is_updating_ui = False

        # Debounce timer to prevent lagging during fast slider dragging
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(80)  # 80ms debounce
        self._debounce_timer.timeout.connect(self._emit_params_changed)

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Group 1: Presets
        group_presets = QGroupBox("BỘ THIẾT LẬP NHANH (PRESETS)")
        gp_layout = QVBoxLayout(group_presets)
        self.combo_presets = QComboBox()
        self.combo_presets.addItem("⭐ Văn bản siêu nét (Ultra Sharp)", "ultra_sharp")
        self.combo_presets.addItem("📄 Quét sạch nền & Khử bóng (Clean Scan)", "clean_scan")
        self.combo_presets.addItem("🖨️ Đen trắng chuẩn scan (Crisp B&W)", "crisp_bw")
        self.combo_presets.addItem("🔍 Phóng to siêu phân giải 3x (Super Res)", "super_res")
        self.combo_presets.addItem("🖼️ Tự nhiên / Màu gốc (Natural Color)", "natural")
        self.combo_presets.addItem("⚙️ Tự điều chỉnh thủ công (Custom)", "custom")
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        gp_layout.addWidget(self.combo_presets)
        layout.addWidget(group_presets)

        # Group 2: Clarity & Sharpening
        group_clarity = QGroupBox("TĂNG CƯỜNG ĐỘ NÉT & KHỬ MỜ")
        gc_layout = QVBoxLayout(group_clarity)

        # Sharpness Slider
        self.lbl_sharpness_val = QLabel("80%")
        self.lbl_sharpness_val.setStyleSheet("color: #60a5fa; font-weight: bold;")
        s_header = QHBoxLayout()
        s_header.addWidget(QLabel("Độ sắc nét viền chữ:"))
        s_header.addStretch()
        s_header.addWidget(self.lbl_sharpness_val)
        gc_layout.addLayout(s_header)

        self.slider_sharpness = QSlider(Qt.Horizontal)
        self.slider_sharpness.setRange(0, 200)
        self.slider_sharpness.setValue(80)
        self.slider_sharpness.valueChanged.connect(self._on_slider_moved)
        gc_layout.addWidget(self.slider_sharpness)

        # Contrast Slider
        self.lbl_contrast_val = QLabel("30%")
        self.lbl_contrast_val.setStyleSheet("color: #60a5fa; font-weight: bold;")
        c_header = QHBoxLayout()
        c_header.addWidget(QLabel("Tương phản chữ/nền (CLAHE):"))
        c_header.addStretch()
        c_header.addWidget(self.lbl_contrast_val)
        gc_layout.addLayout(c_header)

        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(0, 150)
        self.slider_contrast.setValue(30)
        self.slider_contrast.valueChanged.connect(self._on_slider_moved)
        gc_layout.addWidget(self.slider_contrast)

        layout.addWidget(group_clarity)

        # Group 3: Cleaning & Super-Resolution
        group_clean = QGroupBox("TẨY NỀN & SIÊU PHÂN GIẢI")
        gl_layout = QVBoxLayout(group_clean)

        # Whitening / Shadow Removal
        self.lbl_whitening_val = QLabel("40%")
        self.lbl_whitening_val.setStyleSheet("color: #60a5fa; font-weight: bold;")
        w_header = QHBoxLayout()
        w_header.addWidget(QLabel("Tẩy trắng nền & Khử bóng:"))
        w_header.addStretch()
        w_header.addWidget(self.lbl_whitening_val)
        gl_layout.addLayout(w_header)

        self.slider_whitening = QSlider(Qt.Horizontal)
        self.slider_whitening.setRange(0, 100)
        self.slider_whitening.setValue(40)
        self.slider_whitening.valueChanged.connect(self._on_slider_moved)
        gl_layout.addWidget(self.slider_whitening)

        # Denoise
        self.lbl_denoise_val = QLabel("15%")
        self.lbl_denoise_val.setStyleSheet("color: #60a5fa; font-weight: bold;")
        d_header = QHBoxLayout()
        d_header.addWidget(QLabel("Khử nhiễu / Hạt ảnh:"))
        d_header.addStretch()
        d_header.addWidget(self.lbl_denoise_val)
        gl_layout.addLayout(d_header)

        self.slider_denoise = QSlider(Qt.Horizontal)
        self.slider_denoise.setRange(0, 100)
        self.slider_denoise.setValue(15)
        self.slider_denoise.valueChanged.connect(self._on_slider_moved)
        gl_layout.addWidget(self.slider_denoise)

        # Super-Resolution Upscale
        u_header = QHBoxLayout()
        u_header.addWidget(QLabel("Tỷ lệ Siêu Phân Giải (Upscale):"))
        self.combo_upscale = QComboBox()
        self.combo_upscale.addItem("1.0x (Giữ nguyên)", 1.0)
        self.combo_upscale.addItem("1.5x (Nét cao)", 1.5)
        self.combo_upscale.addItem("2.0x (Siêu phân giải - Khuyên dùng)", 2.0)
        self.combo_upscale.addItem("3.0x (Cực cao 4K)", 3.0)
        self.combo_upscale.addItem("4.0x (Tối đa)", 4.0)
        self.combo_upscale.setCurrentIndex(2)  # Default 2.0x
        self.combo_upscale.currentIndexChanged.connect(self._on_slider_moved)
        u_header.addWidget(self.combo_upscale)
        gl_layout.addLayout(u_header)

        # Color Mode
        m_header = QHBoxLayout()
        m_header.addWidget(QLabel("Chế độ màu:"))
        self.combo_color = QComboBox()
        self.combo_color.addItem("Màu sắc tăng nét", "color")
        self.combo_color.addItem("Thang xám sắc nét", "grayscale")
        self.combo_color.addItem("Đen trắng nhị phân (Scan B&W)", "clean_bw")
        self.combo_color.currentIndexChanged.connect(self._on_slider_moved)
        m_header.addWidget(self.combo_color)
        gl_layout.addLayout(m_header)

        # Auto Crop & Flatten Checkbox
        self.chk_auto_flatten = QCheckBox("✂️ Tự động cắt viền & nắn phẳng giấy (Auto-Crop & Flatten)")
        self.chk_auto_flatten.setToolTip(
            "Tự động nhận diện 4 góc của trang giấy trên bàn, sàn gỗ, ga giường...\n"
            "Cắt bỏ phần thừa xung quanh và nắn phẳng phối cảnh chuẩn như máy scan chuyên nghiệp."
        )
        self.chk_auto_flatten.stateChanged.connect(self._on_slider_moved)
        gl_layout.addWidget(self.chk_auto_flatten)

        # Auto Deskew Checkbox
        self.chk_deskew = QCheckBox("📐 Tự động nắn thẳng & chống nghiêng (Auto-Deskew)")
        self.chk_deskew.setToolTip(
            "Tự động phát hiện hướng nghiêng của văn bản & bảng biểu để xoay thẳng chuẩn xác.\n"
            "Tích hợp bộ lọc thông minh (deadband) giúp giữ nguyên độ nét gốc nếu tài liệu đã thẳng."
        )
        self.chk_deskew.stateChanged.connect(self._on_slider_moved)
        gl_layout.addWidget(self.chk_deskew)

        layout.addWidget(group_clean)

        # Action Buttons: Apply
        self.btn_apply_all = QPushButton("⚡ Áp dụng thông số cho TẤT CẢ các trang")
        self.btn_apply_all.setObjectName("primary_btn")
        self.btn_apply_all.clicked.connect(self._on_apply_all)
        layout.addWidget(self.btn_apply_all)

        self.btn_reset = QPushButton("↺ Khôi phục mặc định")
        self.btn_reset.clicked.connect(self._on_reset)
        layout.addWidget(self.btn_reset)

        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    def _on_preset_changed(self, index: int):
        if self._is_updating_ui:
            return
        preset_key = self.combo_presets.currentData()
        if preset_key in PRESETS:
            p = PRESETS[preset_key]
            self.set_params(p, emit_signal=True)

    def _on_slider_moved(self):
        if self._is_updating_ui:
            return

        # Update labels
        self.lbl_sharpness_val.setText(f"{self.slider_sharpness.value()}%")
        self.lbl_contrast_val.setText(f"{self.slider_contrast.value()}%")
        self.lbl_whitening_val.setText(f"{self.slider_whitening.value()}%")
        self.lbl_denoise_val.setText(f"{self.slider_denoise.value()}%")

        self._debounce_timer.start()

    def _emit_params_changed(self):
        # Read values into params object
        self._current_params.sharpness = self.slider_sharpness.value() / 100.0
        self._current_params.contrast = self.slider_contrast.value() / 100.0
        self._current_params.whitening = self.slider_whitening.value() / 100.0
        self._current_params.denoise = self.slider_denoise.value() / 100.0
        self._current_params.upscale_factor = float(self.combo_upscale.currentData())
        self._current_params.color_mode = str(self.combo_color.currentData())
        self._current_params.auto_flatten = self.chk_auto_flatten.isChecked()
        self._current_params.auto_deskew = self.chk_deskew.isChecked()

        self.params_changed.emit(self._current_params)

    def set_params(self, params: EnhanceParams, emit_signal: bool = False):
        """Update UI sliders from params object without triggering unnecessary re-enhancement."""
        self._is_updating_ui = True
        self._debounce_timer.stop()
        self._current_params = params

        self.slider_sharpness.setValue(int(params.sharpness * 100))
        self.lbl_sharpness_val.setText(f"{int(params.sharpness * 100)}%")

        self.slider_contrast.setValue(int(params.contrast * 100))
        self.lbl_contrast_val.setText(f"{int(params.contrast * 100)}%")

        self.slider_whitening.setValue(int(params.whitening * 100))
        self.lbl_whitening_val.setText(f"{int(params.whitening * 100)}%")

        self.slider_denoise.setValue(int(params.denoise * 100))
        self.lbl_denoise_val.setText(f"{int(params.denoise * 100)}%")

        # Upscale combo
        for i in range(self.combo_upscale.count()):
            if abs(float(self.combo_upscale.itemData(i)) - params.upscale_factor) < 0.05:
                self.combo_upscale.setCurrentIndex(i)
                break

        # Color mode combo
        for i in range(self.combo_color.count()):
            if self.combo_color.itemData(i) == params.color_mode:
                self.combo_color.setCurrentIndex(i)
                break

        self.chk_auto_flatten.setChecked(params.auto_flatten)
        self.chk_deskew.setChecked(params.auto_deskew)

        self._is_updating_ui = False
        if emit_signal:
            self._emit_params_changed()

    def get_params(self) -> EnhanceParams:
        return self._current_params

    def _on_apply_all(self):
        self._emit_params_changed()
        self.apply_all_requested.emit(self._current_params)

    def _on_reset(self):
        default_preset = PRESETS["ultra_sharp"]
        self.combo_presets.setCurrentIndex(0)
        self.set_params(default_preset, emit_signal=True)
