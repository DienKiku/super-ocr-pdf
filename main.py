"""
Entry point for Super OCR & High-Res PDF Studio.
Version: 3.2.0 (Accurate Natural Vietnamese OCR & New App Icon)
Updated: 2026-09-14 by Fami (fami_7006)
"""

import sys
import os

# Add project root or PyInstaller temp directory to sys.path
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    sys.path.insert(0, sys._MEIPASS)
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon

from ui.main_window import MainWindow
from ui.styles import DARK_THEME


def main():
    # Enable high-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Super OCR & High-Res PDF Studio")
    app.setOrganizationName("Fami")

    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Apply modern dark theme
    app.setStyleSheet(DARK_THEME)

    # Set application icon
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    icon_path = os.path.join(base_dir, "assets", "logo.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
