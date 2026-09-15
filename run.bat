@echo off
chcp 65001 > nul
title Super OCR & High-Res PDF Studio

echo ================================================================
echo      SUPER OCR & HIGH-RES PDF STUDIO (LAM NET CHU & XUAT PDF)
echo ================================================================
echo.

cd /d "%~dp0"

:: 1. Neu da co ban build EXE san, uu tien khoi chay truc tiep khong can Python
if exist "dist\SuperOCRPDFStudio\SuperOCRPDFStudio.exe" (
    echo Phat hien ban ung dung doc lap (EXE), dang khoi dong truc tiep...
    start "" "dist\SuperOCRPDFStudio\SuperOCRPDFStudio.exe"
    exit /b 0
)

:: 2. Kiem tra xem may tinh da cai Python chua
where python >nul 2>nul
if errorlevel 1 (
    echo ================================================================
    echo [THONG BAO QUAN TRONG]
    echo May tinh cua ban CHUA CAI DAT PYTHON hoac CHUA THEM VAO PATH!
    echo.
    echo De chay tu ma nguon, ban co 2 lua chon:
    echo.
    echo   Cach 1 (Don gian nhat - Khong can Python):
    echo     Chay file "build_exe.bat" tren may da cai dat, sau do copy
    echo     toan bo thu muc "dist\SuperOCRPDFStudio" sang may nay.
    echo.
    echo   Cach 2:
    echo     1. Tai Python 3.11 hoac 3.12 (64-bit) tai: https://www.python.org/downloads/
    echo     2. Khi cai dat, NHO TICH VAO O: [x] "Add python.exe to PATH"
    echo     3. Sau do mo lai file run.bat nay.
    echo ================================================================
    echo.
    pause
    exit /b 1
)

:: 3. Kiem tra phien ban Python (Yeu cau Python >= 3.10)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo Tim thay Python phien ban: %PY_VER%

:: 4. Kiem tra va tu dong cai dat thu vien
python -c "import PySide6, cv2, rapidocr_onnxruntime, pymupdf" 2>nul
if errorlevel 1 (
    echo.
    echo Dang thieu thu vien, tien hanh cai dat tu dong...
    echo (Vui long giu ket noi mang on dinh)...
    echo.
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo.
        echo [CANH BAO] Khong the tu dong cai dat thu vien.
        echo Vui long kiem tra lai ket noi Internet hoac phien ban Python.
        pause
        exit /b 1
    )
)

echo.
echo Dang khoi dong Super OCR PDF Studio...
python main.py

if errorlevel 1 (
    echo.
    echo ================================================================
    echo [LOI] Ung dung gap su co khi khoi dong.
    echo Neu co tep "crash.log", vui long mo tep do de xem chi tiet loi.
    echo ================================================================
    pause
)

