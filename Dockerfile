# MediaPipe 수화 인식 시스템 Dockerfile
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    ffmpeg \
    espeak \
    pulseaudio \
    alsa-utils \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# 데이터 및 모델 디렉토리 생성
RUN mkdir -p data/raw data/processed data/labels \
    models/checkpoints models/saved_models

# 권한 설정
RUN chmod +x *.py

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# 포트 노출 (API 서버용 - 향후 확장)
EXPOSE 8000

# 기본 명령어
CMD ["python", "main.py", "--help"]
