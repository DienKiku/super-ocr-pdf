"""
Entry point for Super OCR & High-Res PDF Studio.
Version: 3.4.0 (100% Offline Deep Vietnamese OCR & Language Model)
Updated: 2026-09-16 by Fami (fami_7006)
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


import traceback


def safe_main():
    try:
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
    except Exception as e:
        err_msg = traceback.format_exc()
        try:
            with open("crash.log", "w", encoding="utf-8") as f:
                f.write(err_msg)
        except Exception:
            pass

        # Show native Windows Error MessageBox if GUI failed
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                f"Đã xảy ra lỗi khi khởi động ứng dụng:\n\n{str(e)}\n\n"
                f"Chi tiết lỗi đã được ghi vào tệp 'crash.log'.\n\n"
                f"Nếu lỗi do thiếu DLL, vui lòng cài đặt: Microsoft Visual C++ 2015-2022 Redistributable (x64).",
                "Super OCR & High-Res PDF Studio - Lỗi Khởi Động",
                0x10  # MB_ICONERROR
            )
        except Exception:
            print("ERROR STARTING APP:", err_msg)
        sys.exit(1)


if __name__ == "__main__":
    safe_main()
