@echo off
chcp 65001 > nul

:: MediaPipe 수화 인식 시스템 - Docker 실행 스크립트 (Windows)

echo ======================================
echo MediaPipe 수화 인식 - Docker 실행
echo ======================================
echo.

:menu
echo.
echo 무엇을 하시겠습니까?
echo 1) Docker 이미지 빌드
echo 2) 시스템 테스트
echo 3) 데이터 수집
echo 4) 모델 학습
echo 5) 실시간 추론
echo 6) 인터랙티브 셸
echo 7) 모든 컨테이너 중지
echo 0) 종료
echo.

set /p choice="선택: "

if "%choice%"=="1" goto build
if "%choice%"=="2" goto test
if "%choice%"=="3" goto collect
if "%choice%"=="4" goto train
if "%choice%"=="5" goto inference
if "%choice%"=="6" goto shell
if "%choice%"=="7" goto stop
if "%choice%"=="0" goto end
echo 잘못된 선택입니다.
goto menu

:build
echo.
echo [빌드] Docker 이미지 빌드 중...
docker-compose build
goto menu

:test
echo.
echo [테스트] 시스템 테스트 실행 중...
docker-compose run --rm mediapipe-app python test_system.py
goto menu

:collect
echo.
echo [수집] 데이터 수집 시작
set /p label="수화 단어를 입력하세요: "
set /p samples="샘플 개수 (기본값 100): "
if "%samples%"=="" set samples=100

docker-compose run --rm mediapipe-app python main.py collect --label "%label%" --samples %samples%
goto menu

:train
echo.
echo [학습] 모델 학습 시작...
docker-compose --profile training up trainer
goto menu

:inference
echo.
echo [추론] 실시간 추론 시작...
docker-compose --profile inference up inference
goto menu

:shell
echo.
echo [셸] 인터랙티브 셸 시작...
docker-compose run --rm mediapipe-app /bin/bash
goto menu

:stop
echo.
echo [중지] 모든 컨테이너 중지 중...
docker-compose down
goto menu

:end
echo.
echo 종료합니다.
exit /b 0
