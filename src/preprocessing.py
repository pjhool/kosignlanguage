"""
데이터 전처리 및 정규화
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


class LandmarkPreprocessor:
    """랜드마크 데이터 전처리 클래스"""
    
    def __init__(self, scaler_type='standard'):
        """
        Args:
            scaler_type: 'standard' 또는 'minmax'
        """
        self.scaler_type = scaler_type
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"지원하지 않는 스케일러: {scaler_type}")
        
        self.is_fitted = False
    
    def normalize_landmarks(self, landmarks):
        """
        랜드마크 정규화 (손목 기준 상대 좌표)
        
        Args:
            landmarks: (N, 3) 형태의 랜드마크 배열 또는 (N*3,) 형태
        
        Returns:
            normalized: 정규화된 랜드마크
        """
        # 1D 배열을 2D로 변환
        original_shape = landmarks.shape
        if len(landmarks.shape) == 1:
            landmarks = landmarks.reshape(-1, 3)
        
        # 손목(0번 랜드마크)을 기준으로 상대 좌표 계산
        if landmarks.shape[0] > 0:
            wrist = landmarks[0:1, :]  # 첫 번째 랜드마크 (손목)
            normalized = landmarks - wrist
            
            # 스케일 정규화 (손의 크기 차이 보정)
            max_distance = np.max(np.linalg.norm(normalized, axis=1))
            if max_distance > 0:
                normalized = normalized / max_distance
        else:
            normalized = landmarks
        
        # 원래 형태로 복원
        if len(original_shape) == 1:
            normalized = normalized.flatten()
        
        return normalized
    
    def fit(self, data):
        """
        스케일러 학습
        
        Args:
            data: (num_samples, num_features) 형태의 데이터
        """
        self.scaler.fit(data)
        self.is_fitted = True
    
    def transform(self, data):
        """
        데이터 변환
        
        Args:
            data: (num_samples, num_features) 형태의 데이터
        
        Returns:
            transformed: 변환된 데이터
        """
        if not self.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다. fit()을 먼저 호출하세요.")
        return self.scaler.transform(data)
    
    def fit_transform(self, data):
        """
        학습 후 변환
        
        Args:
            data: (num_samples, num_features) 형태의 데이터
        
        Returns:
            transformed: 변환된 데이터
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        """
        역변환
        
        Args:
            data: 변환된 데이터
        
        Returns:
            original: 원본 스케일의 데이터
        """
        if not self.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다.")
        return self.scaler.inverse_transform(data)
    
    def save(self, filepath):
        """
        스케일러 저장
        
        Args:
            filepath: 저장 경로
        """
        with open(filepath, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_fitted': self.is_fitted}, f)
        print(f"스케일러 저장: {filepath}")
    
    def load(self, filepath):
        """
        스케일러 로드
        
        Args:
            filepath: 로드 경로
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.is_fitted = data['is_fitted']
        print(f"스케일러 로드: {filepath}")


def create_sequences(data, labels, sequence_length):
    """
    시계열 시퀀스 생성
    
    Args:
        data: 랜드마크 데이터 리스트 [(frame1, frame2, ...), ...]
        labels: 라벨 리스트
        sequence_length: 시퀀스 길이
    
    Returns:
        sequences: (num_sequences, sequence_length, num_features)
        sequence_labels: (num_sequences, num_classes)
    """
    sequences = []
    sequence_labels = []
    
    for sample_data, label in zip(data, labels):
        # 데이터가 시퀀스보다 짧으면 패딩
        if len(sample_data) < sequence_length:
            padding = np.zeros((sequence_length - len(sample_data), sample_data.shape[1]))
            sample_data = np.vstack([sample_data, padding])
        
        # 슬라이딩 윈도우로 시퀀스 생성
        for i in range(len(sample_data) - sequence_length + 1):
            seq = sample_data[i:i + sequence_length]
            sequences.append(seq)
            sequence_labels.append(label)
    
    return np.array(sequences), np.array(sequence_labels)


def augment_landmarks(landmarks, augmentation_params=None):
    """
    랜드마크 데이터 증강
    
    Args:
        landmarks: (N, 3) 형태의 랜드마크
        augmentation_params: 증강 파라미터 딕셔너리
    
    Returns:
        augmented: 증강된 랜드마크
    """
    if augmentation_params is None:
        augmentation_params = {
            'rotation': 10,  # 회전 각도 (도)
            'scale': 0.1,    # 스케일 변화
            'translation': 0.05,  # 이동
            'noise': 0.01    # 노이즈
        }
    
    augmented = landmarks.copy()
    
    # 회전 (Z축 기준)
    if 'rotation' in augmentation_params:
        angle = np.random.uniform(-augmentation_params['rotation'], 
                                 augmentation_params['rotation'])
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        augmented = np.dot(augmented, rotation_matrix.T)
    
    # 스케일
    if 'scale' in augmentation_params:
        scale_factor = 1 + np.random.uniform(-augmentation_params['scale'],
                                            augmentation_params['scale'])
        augmented *= scale_factor
    
    # 이동
    if 'translation' in augmentation_params:
        translation = np.random.uniform(-augmentation_params['translation'],
                                       augmentation_params['translation'],
                                       size=3)
        augmented += translation
    
    # 노이즈 추가
    if 'noise' in augmentation_params:
        noise = np.random.normal(0, augmentation_params['noise'], 
                                augmented.shape)
        augmented += noise
    
    return augmented


def filter_landmarks(landmarks, filter_type='median', window_size=3):
    """
    랜드마크 필터링 (노이즈 제거)
    
    Args:
        landmarks: (num_frames, num_features) 형태의 시퀀스
        filter_type: 'median' 또는 'mean'
        window_size: 필터 윈도우 크기
    
    Returns:
        filtered: 필터링된 랜드마크
    """
    from scipy.ndimage import median_filter, uniform_filter1d
    
    if filter_type == 'median':
        filtered = median_filter(landmarks, size=(window_size, 1))
    elif filter_type == 'mean':
        filtered = uniform_filter1d(landmarks, size=window_size, axis=0)
    else:
        raise ValueError(f"지원하지 않는 필터: {filter_type}")
    
    return filtered


# 테스트 코드
if __name__ == "__main__":
    # 샘플 데이터 생성
    sample_landmarks = np.random.rand(21, 3)  # 21개 랜드마크
    
    # 전처리
    preprocessor = LandmarkPreprocessor()
    normalized = preprocessor.normalize_landmarks(sample_landmarks)
    
    print("원본 랜드마크 shape:", sample_landmarks.shape)
    print("정규화된 랜드마크 shape:", normalized.shape)
    print("\n정규화된 랜드마크 샘플:")
    print(normalized[:5])
