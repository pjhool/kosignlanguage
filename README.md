# MediaPipe 수화 인식 시스템

실시간 수화 인식 및 텍스트/음성 변환 시스템

## 프로젝트 구조

```
MediaPipe/
├── data/                      # 데이터셋 저장
│   ├── raw/                   # 원본 영상/이미지
│   ├── processed/             # 전처리된 랜드마크 데이터
│   └── labels/                # 라벨 정보
├── models/                    # 학습된 모델 저장
│   ├── checkpoints/           # 체크포인트
│   └── saved_models/          # 최종 모델
├── src/                       # 소스 코드
│   ├── data_collection.py     # 데이터 수집
│   ├── preprocessing.py       # 전처리 & 정규화
│   ├── model.py              # ML 모델 정의
│   ├── train.py              # 모델 학습
│   └── inference.py          # 실시간 추론
├── utils/                     # 유틸리티 함수
│   ├── mediapipe_utils.py    # MediaPipe 헬퍼
│   ├── visualization.py      # 시각화 함수
│   └── tts_utils.py          # TTS 함수
├── config.yaml               # 설정 파일
├── requirements.txt          # 의존성
└── main.py                   # 메인 실행 파일
```

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 수집
```bash
python src/data_collection.py --label "안녕하세요" --samples 100
```

### 2. 모델 학습
```bash
python src/train.py --config config.yaml
```

### 3. 실시간 추론
```bash
python main.py
```

## 파이프라인

1. **카메라 입력** → MediaPipe로 영상 캡처
2. **랜드마크 추출** → Hands/Holistic 모델로 좌표 추출
3. **전처리** → 정규화 및 시퀀스 변환
4. **예측** → LSTM/Transformer 모델로 수화 단어 예측
5. **출력** → 텍스트 표시 및 음성 변환

## 기능

- ✅ MediaPipe Hands/Holistic 랜드마크 추출
- ✅ 실시간 수화 인식
- ✅ 시계열 데이터 처리 (LSTM)
- ✅ 텍스트 출력
- ✅ 음성 변환 (TTS)

## 설정

`config.yaml` 파일에서 다음을 설정할 수 있습니다:
- 모델 하이퍼파라미터
- MediaPipe 설정
- 데이터 경로
- TTS 옵션
