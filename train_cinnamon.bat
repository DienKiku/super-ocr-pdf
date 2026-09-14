@echo off
chcp 65001 > nul
title Huan Luyen Mo Hinh Fine-tuning Chu Viet Tay Tieng Viet Cinnamon AI
echo ================================================================
echo    HUAN LUYEN PADDLEOCR FINE-TUNING CHU VIET TAY (CINNAMON AI)
echo ================================================================
echo.
cd /d "%~dp0"

if not exist ".venv_paddle\Scripts\python.exe" (
    echo [LOI] Khong tim thay moi truong .venv_paddle!
    pause
    exit /b 1
)

cd PaddleOCR
..\.venv_paddle\Scripts\python.exe tools/train.py -c configs/rec/PP-OCRv4/cinnamon_handwriting_PP-OCRv4_rec.yml

if errorlevel 1 (
    echo.
    echo [LOI] Qua trinh huan luyen bi gian doan!
    pause
    exit /b 1
)

echo.
echo ================================================================
echo HUAN LUYEN HOAN TAT! DANG XUAT MO HINH INFERENCE...
echo ================================================================
..\.venv_paddle\Scripts\python.exe tools/export_model.py -c configs/rec/PP-OCRv4/cinnamon_handwriting_PP-OCRv4_rec.yml -o Global.pretrained_model=./output/rec_cinnamon_v4/best_accuracy Global.save_inference_dir=../weights/rec_cinnamon_v4_infer

echo.
echo Da xuat mo hinh thanh cong vao: weights\rec_cinnamon_v4_infer\
pause
