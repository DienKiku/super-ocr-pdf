"""
Entry point for Super OCR & High-Res PDF Studio.
Version: 3.4.1 (100% Offline Deep Vietnamese OCR & Language Model)
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
from core.windows_integration import (
    setup_windows_integration,
    apply_native_window_icon,
    get_asset_path,
)

import traceback


def safe_main():
    try:
        # Register explicit AppUserModelID, HKCU registry, and Start Menu shortcut
        setup_windows_integration()

        # Register JPEG XL support in Pillow
        try:
            import pillow_jxl  # noqa: F401
        except ImportError:
            pass

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

        # Set application icon (prioritize transparent multi-resolution .ico on Windows)
        ico_path = get_asset_path("logo.ico")
        png_path = get_asset_path("logo.png")
        if os.path.exists(ico_path):
            app.setWindowIcon(QIcon(ico_path))
        elif os.path.exists(png_path):
            app.setWindowIcon(QIcon(png_path))

        window = MainWindow()
        window.show()

        # Ensure native Windows 11 Taskbar & Alt-Tab icons are applied to HWND
        if sys.platform == 'win32':
            apply_native_window_icon(int(window.winId()), ico_path)

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
