#!/bin/bash

# MediaPipe 수화 인식 시스템 - Docker 실행 스크립트 (Linux/Mac)

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "MediaPipe 수화 인식 - Docker 실행"
echo "======================================"

# X11 디스플레이 권한 설정
echo -e "${YELLOW}X11 디스플레이 권한 설정 중...${NC}"
xhost +local:docker

# 함수 정의
build_image() {
    echo -e "${GREEN}Docker 이미지 빌드 중...${NC}"
    docker-compose build
}

run_test() {
    echo -e "${GREEN}시스템 테스트 실행 중...${NC}"
    docker-compose run --rm mediapipe-app python test_system.py
}

collect_data() {
    echo -e "${GREEN}데이터 수집 시작...${NC}"
    echo -n "수화 단어를 입력하세요: "
    read label
    echo -n "샘플 개수 (기본값 100): "
    read samples
    samples=${samples:-100}
    
    docker-compose run --rm mediapipe-app \
        python main.py collect --label "$label" --samples $samples
}

train_model() {
    echo -e "${GREEN}모델 학습 시작...${NC}"
    docker-compose --profile training up trainer
}

run_inference() {
    echo -e "${GREEN}실시간 추론 시작...${NC}"
    docker-compose --profile inference up inference
}

interactive_shell() {
    echo -e "${GREEN}인터랙티브 셸 시작...${NC}"
    docker-compose run --rm mediapipe-app /bin/bash
}

stop_all() {
    echo -e "${YELLOW}모든 컨테이너 중지 중...${NC}"
    docker-compose down
}

# 메뉴
show_menu() {
    echo ""
    echo "무엇을 하시겠습니까?"
    echo "1) Docker 이미지 빌드"
    echo "2) 시스템 테스트"
    echo "3) 데이터 수집"
    echo "4) 모델 학습"
    echo "5) 실시간 추론"
    echo "6) 인터랙티브 셸"
    echo "7) 모든 컨테이너 중지"
    echo "0) 종료"
    echo ""
    echo -n "선택: "
}

# 메인 루프
while true; do
    show_menu
    read choice
    
    case $choice in
        1) build_image ;;
        2) run_test ;;
        3) collect_data ;;
        4) train_model ;;
        5) run_inference ;;
        6) interactive_shell ;;
        7) stop_all ;;
        0) 
            echo -e "${GREEN}종료합니다.${NC}"
            xhost -local:docker
            exit 0
            ;;
        *) 
            echo -e "${RED}잘못된 선택입니다.${NC}"
            ;;
    esac
done
