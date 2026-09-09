"""
Entry point for Super OCR & High-Res PDF Studio.
Version: 2.1.0 (Dual-Engine AI Vision & High-Speed OCR)
Updated: 2026-09-09 by Fami (fami_7006)
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
from PySide6.QtGui import QFont

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

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
