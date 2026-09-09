@echo off
chcp 65001 > nul
title Dong bo du an len GitHub

echo ================================================================
echo      DONG BO MA NGUON SUPER OCR PDF STUDIO LEN GITHUB
echo ================================================================
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo [LOI] May tinh cua ban chua cai dat Git!
    echo Vui long tai va cai dat Git tai: https://git-scm.com/download/win
    echo Sau do mo lai file nay.
    pause
    exit /b 1
)

cd /d "%~dp0"

echo [1/4] Kiem tra trang thai tep thay doi...
git status -s
echo.

set /p msg="Nhap noi dung ghi chu commit (Enter de dung mac dinh): "
if "%msg%"=="" (
    for /f "tokens=1-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%b-%%a)
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do (set mytime=%%a:%%b)
    set msg=Cap nhat ma nguon tu dong vao luc %mytime% %mydate%
)

echo.
echo [2/4] Dang gom cac tep thay doi (git add .)...
git add .

echo [3/4] Dang dong goi commit...
git commit -m "%msg%"

echo [4/4] Dang day len GitHub (git push)...
git push

if errorlevel 1 (
    echo.
    echo ================================================================
    echo [CHUYEN Y] Neu day len that bai, hay dam bao ban da lien ket Remote:
    echo      git remote add origin https://github.com/TEN_USER/TEN_REPO.git
    echo      git branch -M main
    echo      git push -u origin main
    echo ================================================================
) else (
    echo.
    echo ================================================================
    echo DONG BO LEN GITHUB THANH CONG!
    echo ================================================================
)

pause
