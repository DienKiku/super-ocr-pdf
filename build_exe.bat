@echo off
chcp 65001 > nul
title Đóng gói Super OCR sang file EXE độc lập

echo ================================================================
echo    DONG GOI SUPER OCR & HIGH-RES PDF STUDIO SANG TEP .EXE
echo ================================================================
echo.
echo Dang kiem tra moi truong va tien hanh dong goi bang PyInstaller...
echo Vui long doi trong it phut de PyInstaller thu thap tat ca cac thu vien,
echo mo hinh AI ONNX, trong so VietOCR va tap tin thuc thi...
echo.

cd /d "%~dp0"

pyinstaller --noconfirm SuperOCRPDFStudio.spec

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
echo.
echo ⚠️ LUU Y QUAN TRONG KHI MANG SANG MAY KHAC:
echo   Khong duoc chi copy moi file SuperOCRPDFStudio.exe!
echo   Ban phai copy TOAN BO thu muc: "dist\SuperOCRPDFStudio"
echo   (Bao gom ca file .exe va thu muc _internal ben canh).
echo.
echo File chay nam tai: dist\SuperOCRPDFStudio\SuperOCRPDFStudio.exe
echo.
echo Dang tu dong nen thanh goi Portable ZIP de tien chia se...
powershell -Command "Compress-Archive -Path 'dist\SuperOCRPDFStudio\*' -DestinationPath 'dist\SuperOCRPDFStudio_v3.5.0_Portable_Win64.zip' -Force" 2>nul
if exist "dist\SuperOCRPDFStudio_v3.5.0_Portable_Win64.zip" (
    echo Da tao goi nen thanh cong: dist\SuperOCRPDFStudio_v3.5.0_Portable_Win64.zip
    echo Ban chi can copy file ZIP nay sang bat ky may Windows nao, giai nen la chay duoc ngay!
)
echo ================================================================
pause
