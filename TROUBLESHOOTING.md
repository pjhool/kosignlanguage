# 문제 해결 가이드 (Troubleshooting)

이 문서는 프로젝트 사용 중 발생할 수 있는 일반적인 문제와 해결 방법을 제공합니다.

## 목차
1. [설치 문제](#설치-문제)
2. [카메라 문제](#카메라-문제)
3. [MediaPipe 문제](#mediapipe-문제)
4. [모델 학습 문제](#모델-학습-문제)
5. [추론 문제](#추론-문제)
6. [TTS 문제](#tts-문제)

---

## 설치 문제

### 문제: `pip install -r requirements.txt` 실패

**증상:**
```
ERROR: Could not find a version that satisfies the requirement...
```

**해결 방법:**
1. Python 버전 확인 (3.8-3.10 권장)
   ```bash
   python --version
   ```

2. pip 업그레이드
   ```bash
   python -m pip install --upgrade pip
   ```

3. 개별 패키지 설치
   ```bash
   pip install opencv-python mediapipe tensorflow numpy
   ```

### 문제: TensorFlow 설치 오류 (Windows)

**해결 방법:**
- Microsoft Visual C++ 재배포 패키지 설치
- https://aka.ms/vs/16/release/vc_redist.x64.exe

---

## 카메라 문제

### 문제: 카메라를 열 수 없음

**증상:**
```
카메라를 열 수 없습니다.
```

**해결 방법:**
1. 다른 프로그램에서 카메라 사용 중인지 확인
2. `config.yaml`에서 카메라 ID 변경
   ```yaml
   inference:
     camera_id: 0  # 0, 1, 2 등으로 변경
   ```

3. 카메라 권한 확인 (Windows 설정 → 개인정보 → 카메라)

### 문제: 카메라 프레임이 느림

**해결 방법:**
1. `config.yaml`에서 해상도 낮추기
   ```yaml
   inference:
     display_width: 640
     display_height: 480
   ```

2. MediaPipe 복잡도 낮추기
   ```yaml
   mediapipe:
     model_complexity: 0  # 0, 1, 2 중 선택
   ```

---

## MediaPipe 문제

### 문제: 손이 감지되지 않음

**해결 방법:**
1. 조명 개선 (밝은 환경에서 촬영)
2. 손을 카메라 중앙에 배치
3. 신뢰도 임계값 낮추기
   ```yaml
   mediapipe:
     min_detection_confidence: 0.3
     min_tracking_confidence: 0.3
   ```

### 문제: 랜드마크가 떨림

**해결 방법:**
1. `preprocessing.py`의 필터링 활용
   ```python
   from src.preprocessing import filter_landmarks
   filtered = filter_landmarks(landmarks, filter_type='median', window_size=5)
   ```

2. 추적 신뢰도 올리기
   ```yaml
   mediapipe:
     min_tracking_confidence: 0.7
   ```

---

## 모델 학습 문제

### 문제: 메모리 부족 오류

**증상:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**해결 방법:**
1. 배치 크기 줄이기
   ```yaml
   training:
     batch_size: 16  # 32 → 16
   ```

2. 시퀀스 길이 줄이기
   ```yaml
   data:
     sequence_length: 20  # 30 → 20
   ```

3. 모델 크기 줄이기
   ```yaml
   model:
     hidden_dim: 64  # 128 → 64
     num_layers: 1   # 2 → 1
   ```

### 문제: 학습이 진행되지 않음 (Loss가 감소하지 않음)

**해결 방법:**
1. 학습률 조정
   ```yaml
   training:
     learning_rate: 0.0001  # 더 작게
   ```

2. 데이터 품질 확인
   - 각 클래스당 최소 50개 이상 샘플
   - 다양한 각도와 속도로 촬영

3. 데이터 증강 활용
   ```python
   from src.preprocessing import augment_landmarks
   augmented = augment_landmarks(landmarks)
   ```

### 문제: 과적합 (Overfitting)

**증상:**
- Train accuracy는 높지만 Validation accuracy가 낮음

**해결 방법:**
1. Dropout 증가
   ```yaml
   model:
     dropout: 0.5  # 0.3 → 0.5
   ```

2. 데이터 증강
3. Early Stopping patience 조정
   ```yaml
   training:
     early_stopping_patience: 5
   ```

---

## 추론 문제

### 문제: 모델을 로드할 수 없음

**증상:**
```
오류: 모델 파일을 찾을 수 없습니다
```

**해결 방법:**
1. 모델 파일 경로 확인
   ```bash
   ls models/saved_models/
   ```

2. 학습 먼저 실행
   ```bash
   python main.py train --data data/raw
   ```

### 문제: 예측이 부정확함

**해결 방법:**
1. 더 많은 데이터로 재학습
2. 시퀀스 버퍼가 충분히 찰 때까지 대기
3. 신뢰도 임계값 조정
   ```yaml
   inference:
     confidence_threshold: 0.8  # 0.7 → 0.8
   ```

### 문제: FPS가 낮음

**해결 방법:**
1. MediaPipe 복잡도 낮추기
2. 모델 경량화
3. GPU 사용 (TensorFlow GPU 버전 설치)

---

## TTS 문제

### 문제: 음성이 출력되지 않음

**해결 방법:**
1. 다른 TTS 엔진 사용
   ```yaml
   tts:
     engine: 'gtts'  # pyttsx3 → gtts
   ```

2. pyttsx3 재설치
   ```bash
   pip uninstall pyttsx3
   pip install pyttsx3
   ```

3. 시스템 볼륨 확인

### 문제: 한국어 음성이 이상함

**해결 방법:**
1. gTTS 사용 (인터넷 필요)
   ```yaml
   tts:
     engine: 'gtts'
     language: 'ko'
   ```

2. pyttsx3의 경우 한국어 음성 팩 설치 필요

---

## 일반적인 팁

### 데이터 수집 팁
- 다양한 배경에서 촬영
- 다양한 속도로 동작 수행
- 손의 크기와 각도 변화
- 양손 모두 사용하는 경우 포함

### 모델 성능 향상 팁
1. 클래스당 최소 100개 샘플
2. 데이터 증강 활용
3. 하이퍼파라미터 튜닝
4. Transformer 모델 시도

### 디버깅 팁
1. 테스트 스크립트 실행
   ```bash
   python test_system.py
   ```

2. 로그 확인
3. 단계별로 테스트 (데이터 → 학습 → 추론)

---

## 추가 도움이 필요한 경우

1. GitHub Issues에 문제 등록
2. 오류 메시지 전체 복사
3. 환경 정보 포함 (OS, Python 버전, GPU 여부)
4. 재현 가능한 최소 예제 제공
