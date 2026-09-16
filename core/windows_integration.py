"""
Windows Taskbar & Shell Integration for Super OCR & High-Res PDF Studio.
Handles AppUserModelID registration, Windows Registry entries, Start Menu shortcuts,
and native Win32 window icon handling (WM_SETICON) across Windows 10 & 11.
"""

import sys
import os
import winreg

APP_ID = "fami.superocrpdfstudio.app.3.4.1"
APP_NAME = "Super OCR & High-Res PDF Studio"


def get_asset_path(filename: str) -> str:
    """Safely resolves an asset path in both development and PyInstaller environments."""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            p = os.path.join(meipass, "assets", filename)
            if os.path.exists(p):
                return p
        exe_dir = os.path.dirname(sys.executable)
        p1 = os.path.join(exe_dir, "_internal", "assets", filename)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(exe_dir, "assets", filename)
        if os.path.exists(p2):
            return p2

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "assets", filename)


def setup_windows_integration(ico_path: str = None):
    """
    Registers process identity, registry entries, and Start Menu shortcut.
    Must be called before QApplication or window creation.
    """
    if sys.platform != "win32":
        return

    # 1. Set process explicit AppUserModelID
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass

    if not ico_path:
        ico_path = get_asset_path("logo.ico")

    if not os.path.exists(ico_path):
        return

    abs_ico_path = os.path.abspath(ico_path)

    # 2. Register AppUserModelID in HKCU registry
    try:
        key_path = f"Software\\Classes\\AppUserModelId\\{APP_ID}"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, APP_NAME)
            winreg.SetValueEx(key, "IconUri", 0, winreg.REG_SZ, abs_ico_path)
            winreg.SetValueEx(key, "ShowInSettings", 0, winreg.REG_DWORD, 1)
    except Exception:
        pass

    # 3. Register Applications\SuperOCRPDFStudio.exe in HKCU
    try:
        app_key_path = r"Software\Classes\Applications\SuperOCRPDFStudio.exe"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, app_key_path) as key:
            winreg.SetValueEx(key, "FriendlyAppName", 0, winreg.REG_SZ, APP_NAME)
            with winreg.CreateKey(key, "DefaultIcon") as icon_key:
                winreg.SetValueEx(icon_key, "", 0, winreg.REG_SZ, f"{abs_ico_path},0")
    except Exception:
        pass

    # 4. Create or update Start Menu shortcut
    try:
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            programs_dir = os.path.join(appdata, r"Microsoft\Windows\Start Menu\Programs")
            if os.path.exists(programs_dir):
                shortcut_path = os.path.join(programs_dir, f"{APP_NAME}.lnk")

                if getattr(sys, "frozen", False):
                    target_path = sys.executable
                    work_dir = os.path.dirname(sys.executable)
                else:
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    run_bat = os.path.join(base_dir, "run.bat")
                    if os.path.exists(run_bat):
                        target_path = run_bat
                        work_dir = base_dir
                    else:
                        target_path = sys.executable
                        work_dir = base_dir

                try:
                    import win32com.client
                    ws = win32com.client.Dispatch("WScript.Shell")
                    s = ws.CreateShortcut(shortcut_path)
                    s.TargetPath = target_path
                    s.WorkingDirectory = work_dir
                    s.IconLocation = f"{abs_ico_path},0"
                    s.Description = APP_NAME
                    s.Save()
                except Exception:
                    pass
    except Exception:
        pass


def apply_native_window_icon(hwnd: int, ico_path: str = None):
    """
    Applies native Win32 icons (WM_SETICON & SetClassLongPtr) to the window handle.
    Must be called after window.show() or when HWND is valid.
    """
    if sys.platform != "win32":
        return

    if not ico_path:
        ico_path = get_asset_path("logo.ico")

    if not os.path.exists(ico_path):
        return

    try:
        import ctypes
        user32 = ctypes.windll.user32
        LR_LOADFROMFILE = 0x00000010
        IMAGE_ICON = 1
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        GCLP_HICON = -14
        GCLP_HICONSM = -34

        abs_ico = os.path.abspath(ico_path)
        hicon_big = user32.LoadImageW(0, abs_ico, IMAGE_ICON, 32, 32, LR_LOADFROMFILE)
        hicon_small = user32.LoadImageW(0, abs_ico, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)

        SetClassLongPtr = getattr(user32, "SetClassLongPtrW", getattr(user32, "SetClassLongW"))

        if hicon_big:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)
            SetClassLongPtr(hwnd, GCLP_HICON, hicon_big)

        if hicon_small:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)
            SetClassLongPtr(hwnd, GCLP_HICONSM, hicon_small)
    except Exception:
        pass
