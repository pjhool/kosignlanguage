@echo off
chcp 65001 > nul
echo ========================================
echo 데이터 수집 시작
echo ========================================
echo.

set /p label="수화 단어를 입력하세요 (예: 안녕하세요): "
set /p samples="수집할 샘플 개수 (기본값 100): "
if "%samples%"=="" set samples=100

echo.
echo 라벨: %label%
echo 샘플 수: %samples%
echo.

python main.py collect --label "%label%" --samples %samples% --frames 30

pause
