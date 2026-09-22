@echo off
chcp 65001 > nul
title Dong goi Super OCR sang file EXE doc lap

echo ================================================================
echo    DONG GOI SUPER OCR ^& HIGH-RES PDF STUDIO SANG TEP .EXE
echo ================================================================
echo.

cd /d "%~dp0"

:: Kiem tra xem Python co san tren may khong
where python >nul 2>nul
if errorlevel 1 (
    echo ================================================================
    echo [CHU Y] MAY TINH NAY CHUA CAI DAT PYTHON!
    echo.
    echo File "build_exe.bat" chi danh cho MAY LAP TRINH VIEN de dong goi
    echo tu ma nguon sang file .EXE.
    echo.
    echo Neu ban muon CHAY UNG DUNG tren may nay:
    echo   1. Hay tai goi PORTABLE da dong goi san tai:
    echo      https://github.com/DienKiku/super-ocr-pdf/releases
    echo      (File: SuperOCRPDFStudio_v3.6.0_Portable_Win64.zip)
    echo   2. Giai nen va chay truc tiep SuperOCRPDFStudio.exe
    echo      khong can cai dat Python hay bat ky thu vien nao.
    echo ================================================================
    echo.
    pause
    exit /b 1
)

:: Kiem tra PyInstaller
where pyinstaller >nul 2>nul
if errorlevel 1 (
    echo [THONG BAO] Chua tim thay PyInstaller. Dang tien hanh cai dat...
    pip install pyinstaller
    if errorlevel 1 (
        echo [LOI] Khong the cai dat PyInstaller. Vui long kiem tra ket noi mang.
        pause
        exit /b 1
    )
)

echo Dang kiem tra moi truong va tien hanh dong goi bang PyInstaller...
echo Vui long doi trong it phut de PyInstaller thu thap tat ca cac thu vien,
echo mo hinh AI ONNX, trong so VietOCR va tap tin thuc thi...
echo.

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
echo [!] LUU Y QUAN TRONG KHI MANG SANG MAY KHAC:
echo   Khong duoc chi copy moi file SuperOCRPDFStudio.exe!
echo   Ban phai copy TOAN BO thu muc: "dist\SuperOCRPDFStudio"
echo   (Bao gom ca file .exe va thu muc _internal ben canh).
echo.
echo File chay nam tai: dist\SuperOCRPDFStudio\SuperOCRPDFStudio.exe
echo.
echo Dang tu dong don dep cac goi ZIP phien ban cu va nen goi Portable moi...
powershell -Command "Get-ChildItem -Path 'dist' -Filter '*.zip' | Remove-Item -Force" 2>nul
tar -a -c -f dist\SuperOCRPDFStudio_v3.6.0_Portable_Win64.zip -C dist SuperOCRPDFStudio 2>nul || powershell -Command "Compress-Archive -Path 'dist\SuperOCRPDFStudio\*' -DestinationPath 'dist\SuperOCRPDFStudio_v3.6.0_Portable_Win64.zip' -Force" 2>nul
if exist "dist\SuperOCRPDFStudio_v3.6.0_Portable_Win64.zip" (
    echo Da tao goi nen thanh cong: dist\SuperOCRPDFStudio_v3.6.0_Portable_Win64.zip
    echo Ban chi can copy file ZIP nay sang bat ky may Windows nao, giai nen la chay duoc ngay!
)
echo ================================================================
pause
