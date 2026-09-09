"""
Dark theme modern styling for Super OCR & High-Res PDF Studio.
"""

DARK_THEME = """
QWidget {
    background-color: #18181b;
    color: #f4f4f5;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
    selection-background-color: #3b82f6;
    selection-color: #ffffff;
}

QMainWindow {
    background-color: #121215;
}

/* ToolBar */
QToolBar {
    background-color: #1e1e24;
    border-bottom: 1px solid #2e2e38;
    padding: 6px;
    spacing: 8px;
}

QToolButton {
    background-color: transparent;
    color: #e4e4e7;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 500;
}

QToolButton:hover {
    background-color: #2e2e38;
    border: 1px solid #3f3f4e;
    color: #ffffff;
}

QToolButton:pressed {
    background-color: #3b82f6;
    color: #ffffff;
}

/* PushButtons */
QPushButton {
    background-color: #27272a;
    color: #f4f4f5;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #3f3f46;
    border-color: #52525b;
}

QPushButton:pressed {
    background-color: #18181b;
}

QPushButton:disabled {
    background-color: #202024;
    color: #71717a;
    border-color: #2e2e38;
}

QPushButton#primary_btn {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #3b82f6;
    font-weight: 600;
}

QPushButton#primary_btn:hover {
    background-color: #1d4ed8;
    border-color: #60a5fa;
}

QPushButton#primary_btn:pressed {
    background-color: #1e40af;
}

QPushButton#success_btn {
    background-color: #059669;
    color: #ffffff;
    border: 1px solid #10b981;
    font-weight: 600;
}

QPushButton#success_btn:hover {
    background-color: #047857;
    border-color: #34d399;
}

QPushButton#danger_btn {
    background-color: #dc2626;
    color: #ffffff;
    border: 1px solid #ef4444;
}

QPushButton#danger_btn:hover {
    background-color: #b91c1c;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #2e2e38;
    background-color: #1e1e24;
    border-radius: 8px;
    top: -1px;
}

QTabBar::tab {
    background-color: #18181b;
    color: #a1a1aa;
    padding: 8px 16px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    border: 1px solid #27272a;
    border-bottom: none;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #1e1e24;
    color: #60a5fa;
    border-bottom: 2px solid #3b82f6;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background-color: #24242a;
    color: #e4e4e7;
}

/* Groups and Frames */
QGroupBox {
    border: 1px solid #2e2e38;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: 600;
    color: #93c5fd;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    background-color: #1e1e24;
}

/* List Widget (Thumbnails) */
QListWidget {
    background-color: #18181b;
    border: 1px solid #2e2e38;
    border-radius: 8px;
    padding: 4px;
}

QListWidget::item {
    background-color: #222228;
    border: 1px solid #2e2e38;
    border-radius: 6px;
    padding: 4px;
    margin: 4px;
    color: #f4f4f5;
}

QListWidget::item:hover {
    background-color: #2a2a34;
    border-color: #3b82f6;
}

QListWidget::item:selected {
    background-color: #1e3a8a;
    border: 2px solid #3b82f6;
    color: #ffffff;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #2e2e38;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #3b82f6;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #3b82f6;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #93c5fd;
    transform: scale(1.1);
}

/* ComboBox */
QComboBox {
    background-color: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    padding: 6px 12px;
    color: #f4f4f5;
}

QComboBox:hover {
    border-color: #3b82f6;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: none;
}

QComboBox QAbstractItemView {
    background-color: #27272a;
    border: 1px solid #3f3f46;
    selection-background-color: #3b82f6;
    selection-color: #ffffff;
    padding: 4px;
}

/* TextEdit */
QTextEdit, QPlainTextEdit, QLineEdit {
    background-color: #141416;
    color: #f4f4f5;
    border: 1px solid #2e2e38;
    border-radius: 6px;
    padding: 6px;
}

QTextEdit:focus, QPlainTextEdit:focus, QLineEdit:focus {
    border: 1px solid #3b82f6;
}

/* ScrollBar */
QScrollBar:vertical {
    background: #18181b;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #3f3f46;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #52525b;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* ProgressBar */
QProgressBar {
    border: 1px solid #2e2e38;
    border-radius: 6px;
    text-align: center;
    background-color: #141416;
    color: #ffffff;
    font-weight: 600;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #06b6d4);
    border-radius: 5px;
}

/* Splitter */
QSplitter::handle {
    background-color: #2e2e38;
}

QSplitter::handle:hover {
    background-color: #3b82f6;
}

/* StatusBar */
QStatusBar {
    background-color: #141416;
    border-top: 1px solid #2e2e38;
    color: #a1a1aa;
}
"""
