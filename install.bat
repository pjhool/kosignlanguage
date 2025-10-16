@echo off
chcp 65001 > nul
echo ========================================
echo MediaPipe 수화 인식 시스템 - 설치 스크립트
echo ========================================
echo.

:: Python 버전 확인
python --version
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo Python 3.8 이상을 설치해주세요.
    pause
    exit /b 1
)

echo.
echo [1/4] pip 업그레이드 중...
python -m pip install --upgrade pip

echo.
echo [2/4] 필수 패키지 설치 중...
pip install -r requirements.txt

echo.
echo [3/4] 디렉토리 구조 생성 중...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\labels" mkdir "data\labels"
if not exist "models\checkpoints" mkdir "models\checkpoints"
if not exist "models\saved_models" mkdir "models\saved_models"

echo.
echo [4/4] 시스템 테스트 실행 중...
python test_system.py

echo.
echo ========================================
echo 설치 완료!
echo ========================================
echo.
echo 다음 단계:
echo 1. python quick_start.py - 빠른 시작 가이드
echo 2. python main.py collect --label "안녕하세요" --samples 100 - 데이터 수집
echo 3. python main.py train --data data/raw - 모델 학습
echo 4. python main.py inference - 실시간 추론
echo.
pause
