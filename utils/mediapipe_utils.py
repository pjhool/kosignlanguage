"""
MediaPipe 유틸리티 함수
손/포즈 랜드마크 추출 및 처리
"""

import cv2
import mediapipe as mp
import numpy as np


class MediaPipeHandler:
    """MediaPipe Hands/Holistic 처리 클래스"""
    
    def __init__(self, config):
        """
        Args:
            config: YAML 설정 딕셔너리
        """
        self.config = config['mediapipe']
        self.use_holistic = self.config['use_holistic']
        
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 모델 초기화
        if self.use_holistic:
            self.model = self.mp_holistic.Holistic(
                model_complexity=self.config['model_complexity'],
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_tracking_confidence']
            )
        else:
            self.model = self.mp_hands.Hands(
                model_complexity=self.config['model_complexity'],
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_tracking_confidence'],
                max_num_hands=self.config['max_num_hands']
            )
    
    def process_frame(self, frame):
        """
        프레임에서 랜드마크 추출
        
        Args:
            frame: BGR 이미지 (OpenCV)
        
        Returns:
            results: MediaPipe 결과 객체
            landmarks: 추출된 랜드마크 배열 (정규화된 좌표)
        """
        # BGR을 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # MediaPipe 처리
        results = self.model.process(image_rgb)
        
        # 랜드마크 추출
        landmarks = self.extract_landmarks(results)
        
        return results, landmarks
    
    def extract_landmarks(self, results):
        """
        MediaPipe 결과에서 랜드마크 좌표 추출
        
        Args:
            results: MediaPipe 결과 객체
        
        Returns:
            landmarks: numpy 배열 (N, 3) - N개 랜드마크의 (x, y, z) 좌표
        """
        landmarks = []
        
        if self.use_holistic:
            # Holistic 모드: 포즈 + 얼굴 + 손
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0] * 33 * 3)  # 33개 포즈 랜드마크
            
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0] * 21 * 3)  # 21개 손 랜드마크
            
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0] * 21 * 3)
        else:
            # Hands 모드: 양손
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                
                # 한 손만 감지된 경우 나머지를 0으로 채움
                num_detected_hands = len(results.multi_hand_landmarks)
                if num_detected_hands < self.config['max_num_hands']:
                    landmarks.extend([0.0] * 21 * 3 * (self.config['max_num_hands'] - num_detected_hands))
            else:
                # 손이 감지되지 않은 경우 모두 0
                landmarks.extend([0.0] * 21 * 3 * self.config['max_num_hands'])
        
        return np.array(landmarks, dtype=np.float32)
    
    def draw_landmarks(self, frame, results):
        """
        프레임에 랜드마크 그리기
        
        Args:
            frame: BGR 이미지
            results: MediaPipe 결과 객체
        
        Returns:
            annotated_frame: 랜드마크가 그려진 프레임
        """
        annotated_frame = frame.copy()
        annotated_frame.flags.writeable = True
        
        if self.use_holistic:
            # 포즈 그리기
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # 손 그리기
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        else:
            # Hands 모드
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        
        return annotated_frame
    
    def close(self):
        """MediaPipe 리소스 해제"""
        self.model.close()
