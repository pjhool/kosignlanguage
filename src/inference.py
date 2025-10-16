"""
실시간 수화 인식 및 추론
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pickle
import cv2
import time
from collections import deque
import tensorflow as tf

# 프로젝트 루트를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mediapipe_utils import MediaPipeHandler
from utils.visualization import draw_info_panel
from utils.tts_utils import TTSHandler
from src.preprocessing import LandmarkPreprocessor


class SignLanguageInference:
    """실시간 수화 인식 클래스"""
    
    def __init__(self, config, model_path):
        """
        Args:
            config: YAML 설정
            model_path: 학습된 모델 경로
        """
        self.config = config
        
        # MediaPipe 초기화
        self.mp_handler = MediaPipeHandler(config)
        
        # 모델 로드
        print("모델 로드 중...")
        self.model = tf.keras.models.load_model(model_path)
        print("  모델 로드 완료")
        
        # 라벨 인코더 로드
        label_encoder_path = os.path.join(
            os.path.dirname(model_path), 'label_encoder.pkl'
        )
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"  클래스: {self.label_encoder.classes_}")
        
        # 전처리기 로드
        preprocessor_path = os.path.join(
            os.path.dirname(model_path), 'preprocessor.pkl'
        )
        self.preprocessor = LandmarkPreprocessor()
        if os.path.exists(preprocessor_path):
            self.preprocessor.load(preprocessor_path)
        
        # TTS 초기화
        self.tts = TTSHandler(config)
        
        # 시퀀스 버퍼
        self.sequence_length = config['data']['sequence_length']
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        
        # 카메라 설정
        self.cap = cv2.VideoCapture(config['inference']['camera_id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['inference']['display_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['inference']['display_height'])
        
        # 추론 설정
        self.confidence_threshold = config['inference']['confidence_threshold']
        self.show_fps = config['inference']['fps_display']
        
        # 상태 변수
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_cooldown = 2.0  # 초 (같은 예측 반복 방지)
    
    def run(self):
        """실시간 추론 실행"""
        print("\n=== 실시간 수화 인식 시작 ===")
        print("조작법:")
        print("  Q: 종료")
        print("  S: 음성 출력 켜기/끄기")
        print("  R: 시퀀스 버퍼 리셋")
        print("=" * 40)
        
        fps_time = time.time()
        fps = 0
        enable_tts = True
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("카메라 읽기 실패")
                break
            
            # FPS 계산
            if self.show_fps:
                current_time = time.time()
                fps = 1.0 / (current_time - fps_time)
                fps_time = current_time
            
            # MediaPipe 처리
            results, landmarks = self.mp_handler.process_frame(frame)
            
            # 랜드마크가 감지되면 버퍼에 추가
            if landmarks.size > 0:
                # 정규화
                normalized_landmarks = self.preprocessor.normalize_landmarks(landmarks)
                self.sequence_buffer.append(normalized_landmarks)
            
            # 시퀀스가 충분히 쌓이면 예측
            predictions = []
            if len(self.sequence_buffer) == self.sequence_length:
                predictions = self._predict()
                
                # 음성 출력
                if enable_tts and predictions and len(predictions) > 0:
                    top_class, top_prob = predictions[0]
                    if top_prob >= self.confidence_threshold:
                        self._handle_prediction(top_class)
            
            # 시각화
            annotated_frame = self.mp_handler.draw_landmarks(frame, results)
            annotated_frame = draw_info_panel(
                annotated_frame,
                predictions,
                predictions[0][1] if predictions else 0.0,
                fps=fps if self.show_fps else None
            )
            
            # 버퍼 상태 표시
            buffer_text = f"Buffer: {len(self.sequence_buffer)}/{self.sequence_length}"
            cv2.putText(annotated_frame, buffer_text, (10, 
                       annotated_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # TTS 상태 표시
            tts_text = f"TTS: {'ON' if enable_tts else 'OFF'}"
            cv2.putText(annotated_frame, tts_text, (10, 
                       annotated_frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0) if enable_tts else (0, 0, 255), 2)
            
            # 화면 표시
            cv2.imshow('Sign Language Recognition', annotated_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                enable_tts = not enable_tts
                print(f"TTS: {'ON' if enable_tts else 'OFF'}")
            elif key == ord('r'):
                self.sequence_buffer.clear()
                print("시퀀스 버퍼 리셋")
        
        # 정리
        self.cleanup()
    
    def _predict(self):
        """
        현재 시퀀스로 예측
        
        Returns:
            predictions: [(클래스명, 확률), ...] 정렬된 리스트
        """
        # 시퀀스를 배열로 변환
        sequence = np.array(list(self.sequence_buffer))
        sequence = np.expand_dims(sequence, axis=0)  # (1, seq_len, features)
        
        # 예측
        pred_probs = self.model.predict(sequence, verbose=0)[0]
        
        # 상위 3개 예측 결과
        top_indices = np.argsort(pred_probs)[::-1][:3]
        predictions = []
        
        for idx in top_indices:
            class_name = self.label_encoder.classes_[idx]
            probability = pred_probs[idx]
            predictions.append((class_name, probability))
        
        return predictions
    
    def _handle_prediction(self, predicted_class):
        """
        예측 결과 처리 (음성 출력 등)
        
        Args:
            predicted_class: 예측된 클래스명
        """
        current_time = time.time()
        
        # 같은 예측이 연속으로 나오는 것 방지
        if (predicted_class != self.last_prediction or 
            current_time - self.last_prediction_time > self.prediction_cooldown):
            
            print(f"\n예측: {predicted_class}")
            self.tts.speak(predicted_class)
            
            self.last_prediction = predicted_class
            self.last_prediction_time = current_time
    
    def cleanup(self):
        """리소스 해제"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.mp_handler.close()
        self.tts.close()
        print("\n종료됨")


def main():
    parser = argparse.ArgumentParser(description='실시간 수화 인식')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--model', type=str, 
                       default='models/saved_models/sign_language_model.h5',
                       help='모델 파일 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 모델 파일 확인
    if not os.path.exists(args.model):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {args.model}")
        print("먼저 src/train.py로 모델을 학습시켜주세요.")
        return
    
    # 추론 실행
    inference = SignLanguageInference(config, args.model)
    inference.run()


if __name__ == "__main__":
    main()
