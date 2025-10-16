@echo off
chcp 65001 > nul
echo ========================================
echo 실시간 수화 인식 시작
echo ========================================
echo.

python main.py inference --model models/saved_models/sign_language_model.h5

pause
