"""
데이터 수집 스크립트
실시간으로 카메라에서 수화 동작을 녹화하고 저장
"""

import cv2
import numpy as np
import os
import sys
import argparse
import yaml
from datetime import datetime

# 프로젝트 루트를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mediapipe_utils import MediaPipeHandler
from utils.visualization import draw_text


class DataCollector:
    """데이터 수집 클래스"""
    
    def __init__(self, config, label, output_dir):
        """
        Args:
            config: YAML 설정
            label: 수집할 수화 단어 라벨
            output_dir: 저장 디렉토리
        """
        self.config = config
        self.label = label
        self.output_dir = output_dir
        
        # MediaPipe 초기화
        self.mp_handler = MediaPipeHandler(config)
        
        # 저장 디렉토리 생성
        self.label_dir = os.path.join(output_dir, label)
        os.makedirs(self.label_dir, exist_ok=True)
        
        # 카메라 설정
        self.cap = cv2.VideoCapture(config['inference']['camera_id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['inference']['display_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['inference']['display_height'])
        
        # 수집 상태
        self.is_recording = False
        self.current_sequence = []
        self.sample_count = 0
    
    def collect(self, num_samples, frames_per_sample=30):
        """
        데이터 수집 시작
        
        Args:
            num_samples: 수집할 샘플 개수
            frames_per_sample: 샘플당 프레임 수
        """
        print(f"\n=== 데이터 수집 시작 ===")
        print(f"라벨: {self.label}")
        print(f"목표 샘플 수: {num_samples}")
        print(f"샘플당 프레임: {frames_per_sample}")
        print("\n조작법:")
        print("  SPACE: 녹화 시작/정지")
        print("  Q: 종료")
        print("=" * 40)
        
        while self.sample_count < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                print("카메라 읽기 실패")
                break
            
            # MediaPipe 처리
            results, landmarks = self.mp_handler.process_frame(frame)
            
            # 랜드마크 그리기
            annotated_frame = self.mp_handler.draw_landmarks(frame, results)
            
            # 녹화 중일 때 랜드마크 저장
            if self.is_recording and landmarks.size > 0:
                self.current_sequence.append(landmarks)
                
                # 프레임 수 도달 시 저장
                if len(self.current_sequence) >= frames_per_sample:
                    self._save_sequence()
                    self.is_recording = False
                    self.current_sequence = []
            
            # UI 표시
            annotated_frame = self._draw_ui(annotated_frame)
            
            # 화면 표시
            cv2.imshow('Data Collection', annotated_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not self.is_recording:
                    self.is_recording = True
                    self.current_sequence = []
                    print(f"\n[녹화 시작] 샘플 {self.sample_count + 1}/{num_samples}")
                else:
                    if len(self.current_sequence) > 0:
                        self._save_sequence()
                    self.is_recording = False
                    self.current_sequence = []
        
        # 정리
        self.cleanup()
        print(f"\n수집 완료! 총 {self.sample_count}개 샘플 저장됨")
    
    def _save_sequence(self):
        """현재 시퀀스를 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.label}_{timestamp}_{self.sample_count:04d}.npy"
        filepath = os.path.join(self.label_dir, filename)
        
        # numpy 배열로 저장
        sequence_array = np.array(self.current_sequence)
        np.save(filepath, sequence_array)
        
        self.sample_count += 1
        print(f"  저장됨: {filename} (shape: {sequence_array.shape})")
    
    def _draw_ui(self, frame):
        """UI 정보 그리기"""
        # 녹화 상태
        if self.is_recording:
            status_text = f"REC [{len(self.current_sequence)} frames]"
            color = (0, 0, 255)  # 빨강
        else:
            status_text = "READY"
            color = (0, 255, 0)  # 초록
        
        frame = draw_text(frame, status_text, position=(10, 30), 
                         font_scale=1.0, color=color, thickness=2)
        
        # 샘플 카운트
        frame = draw_text(frame, f"Samples: {self.sample_count}", 
                         position=(10, 70), font_scale=0.8, color=(255, 255, 255))
        
        # 라벨
        frame = draw_text(frame, f"Label: {self.label}", 
                         position=(10, 110), font_scale=0.8, color=(255, 255, 255))
        
        return frame
    
    def cleanup(self):
        """리소스 해제"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.mp_handler.close()


def main():
    parser = argparse.ArgumentParser(description='수화 데이터 수집')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--label', type=str, required=True,
                       help='수집할 수화 단어 라벨')
    parser.add_argument('--samples', type=int, default=100,
                       help='수집할 샘플 개수')
    parser.add_argument('--frames', type=int, default=30,
                       help='샘플당 프레임 수')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 데이터 수집
    collector = DataCollector(config, args.label, args.output)
    collector.collect(args.samples, args.frames)


if __name__ == "__main__":
    main()
