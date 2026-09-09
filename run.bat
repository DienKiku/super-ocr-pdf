@echo off
chcp 65001 > nul
title Super OCR & High-Res PDF Studio

echo ================================================================
echo      SUPER OCR & HIGH-RES PDF STUDIO (LAM NET CHU & XUAT PDF)
echo ================================================================
echo.
echo Dang khoi dong ung dung, vui long doi giay lat...
echo.

cd /d "%~dp0"

:: Kiem tra neu thieu thu vien thi tu dong cai dat
python -c "import PySide6, cv2, rapidocr_onnxruntime, pymupdf" 2>nul
if errorlevel 1 (
    echo Phat hien thieu thu vien, dang tu dong cai dat cac goi can thiet...
    pip install -r requirements.txt
)

python main.py

if errorlevel 1 (
    echo.
    echo Da xay ra loi khi khoi dong ung dung.
    pause
)
