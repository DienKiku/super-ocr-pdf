@echo off
chcp 65001 > nul
title Đóng gói Super OCR sang file EXE độc lập

echo ================================================================
echo    DONG GOI SUPER OCR & HIGH-RES PDF STUDIO SANG TEP .EXE
echo ================================================================
echo.
echo Dang kiem tra moi truong va tien hanh dong goi bang PyInstaller...
echo Vui long doi trong it phut de PyInstaller thu thap tat ca cac thu vien,
echo mo hinh AI ONNX va tap tin thuc thi...
echo.

cd /d "%~dp0"

pyinstaller --noconfirm --onedir --windowed ^
    --name "SuperOCRPDFStudio" ^
    --add-data "core;core" ^
    --add-data "ui;ui" ^
    --collect-all onnxruntime ^
    --collect-all rapidocr_onnxruntime ^
    --collect-all pymupdf ^
    --collect-all PySide6 ^
    --collect-all google.genai ^
    --collect-all pydantic ^
    --hidden-import "cv2" ^
    --hidden-import "numpy" ^
    --hidden-import "PIL" ^
    --hidden-import "yaml" ^
    --hidden-import "onnxruntime" ^
    --hidden-import "google" ^
    --hidden-import "google.genai" ^
    --hidden-import "google.genai.types" ^
    --hidden-import "pydantic" ^
    main.py

if errorlevel 1 (
    echo.
    echo ================================================================
    echo [LOI] Dong goi that bai! Vui long kiem tra lai thong bao loi o tren.
    echo ================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================
echo DONG GOI HOAN TAT THANH CONG!
echo File chay nam tai: dist\SuperOCRPDFStudio\SuperOCRPDFStudio.exe
echo Ban co the sao chep toan bo thu muc:
echo      dist\SuperOCRPDFStudio
echo sang bat ky may tinh Windows nao khac de chay ma khong can cai dat Python.
echo ================================================================
pause
