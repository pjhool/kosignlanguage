"""
빠른 시작 가이드 스크립트
프로젝트의 주요 기능을 순서대로 실행
"""

import os
import sys

def print_step(step_num, title):
    """단계 제목 출력"""
    print("\n" + "=" * 60)
    print(f"STEP {step_num}: {title}")
    print("=" * 60)


def main():
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║   MediaPipe 수화 인식 시스템 - 빠른 시작 가이드   ║
    ╚═══════════════════════════════════════════════════╝
    
    이 스크립트는 프로젝트의 전체 워크플로우를 안내합니다.
    """)
    
    # Step 1: 환경 설정 확인
    print_step(1, "환경 설정 확인")
    print("""
    필요한 패키지 설치:
    
    pip install -r requirements.txt
    
    설치 후 Enter를 눌러 계속하세요...
    """)
    input()
    
    # Step 2: 데이터 수집
    print_step(2, "데이터 수집")
    print("""
    수화 동작 데이터를 수집합니다.
    
    예시:
    python main.py collect --label "안녕하세요" --samples 100 --frames 30
    python main.py collect --label "감사합니다" --samples 100 --frames 30
    python main.py collect --label "사랑해요" --samples 100 --frames 30
    
    최소 3개 이상의 수화 단어를 수집하는 것을 권장합니다.
    각 단어당 100개 이상의 샘플을 수집하세요.
    
    조작법:
    - SPACE: 녹화 시작/정지
    - Q: 종료
    
    데이터 수집을 완료한 후 Enter를 눌러 계속하세요...
    """)
    input()
    
    # Step 3: 데이터 확인
    print_step(3, "수집된 데이터 확인")
    print("""
    data/raw/ 폴더에 수집된 데이터를 확인하세요.
    
    폴더 구조:
    data/raw/
    ├── 안녕하세요/
    │   ├── 안녕하세요_20250101_120000_0000.npy
    │   ├── 안녕하세요_20250101_120001_0001.npy
    │   └── ...
    ├── 감사합니다/
    └── 사랑해요/
    
    확인 후 Enter를 눌러 계속하세요...
    """)
    input()
    
    # Step 4: 모델 학습
    print_step(4, "모델 학습")
    print("""
    수집된 데이터로 모델을 학습합니다.
    
    명령어:
    python main.py train --data data/raw
    
    학습 시간은 데이터 양과 하드웨어에 따라 다릅니다.
    (예상: 수백 개 샘플 기준 5-20분)
    
    학습이 완료되면:
    - models/saved_models/sign_language_model.h5 (모델 파일)
    - models/saved_models/label_encoder.pkl (라벨 인코더)
    - models/saved_models/classes.txt (클래스 정보)
    - models/saved_models/training_history.png (학습 그래프)
    
    학습 완료 후 Enter를 눌러 계속하세요...
    """)
    input()
    
    # Step 5: 실시간 추론
    print_step(5, "실시간 수화 인식")
    print("""
    학습된 모델로 실시간 수화를 인식합니다.
    
    명령어:
    python main.py inference --model models/saved_models/sign_language_model.h5
    
    조작법:
    - Q: 종료
    - S: 음성 출력 켜기/끄기
    - R: 시퀀스 버퍼 리셋
    
    카메라 앞에서 학습한 수화 동작을 해보세요!
    시스템이 자동으로 인식하고 텍스트로 표시합니다.
    """)
    
    # Step 6: 추가 개선
    print_step(6, "추가 개선 방안")
    print("""
    모델 성능을 향상시키기 위한 방법:
    
    1. 더 많은 데이터 수집
       - 샘플 수 증가 (클래스당 200-500개)
       - 다양한 환경에서 촬영 (조명, 배경)
       - 다양한 사람의 데이터
    
    2. 모델 하이퍼파라미터 조정
       - config.yaml 파일 수정
       - hidden_dim, num_layers, dropout 등
    
    3. 데이터 증강
       - preprocessing.py의 augment_landmarks 활용
    
    4. 다른 모델 아키텍처 시도
       - LSTM → GRU 또는 Transformer
       - config.yaml에서 model.type 변경
    
    5. 시퀀스 길이 조정
       - config.yaml의 sequence_length 수정
       - 짧은 동작: 20-30 프레임
       - 긴 동작: 40-60 프레임
    """)
    
    print("\n" + "=" * 60)
    print("빠른 시작 가이드를 완료했습니다!")
    print("행운을 빕니다! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
