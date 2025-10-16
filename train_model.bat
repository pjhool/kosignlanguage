@echo off
chcp 65001 > nul
echo ========================================
echo 모델 학습 시작
echo ========================================
echo.

python main.py train --data data/raw

echo.
echo 학습이 완료되었습니다!
echo 모델 파일: models/saved_models/sign_language_model.h5
echo.
pause
