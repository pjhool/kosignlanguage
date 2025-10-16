"""
모델 학습 스크립트
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# 프로젝트 루트를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model, compile_model
from src.preprocessing import LandmarkPreprocessor, create_sequences
from utils.visualization import visualize_training_history


class SignLanguageTrainer:
    """수화 인식 모델 학습 클래스"""
    
    def __init__(self, config):
        """
        Args:
            config: YAML 설정 딕셔너리
        """
        self.config = config
        self.preprocessor = LandmarkPreprocessor()
        self.label_encoder = LabelEncoder()
        
        # 디렉토리 생성
        os.makedirs(config['training']['checkpoint_path'], exist_ok=True)
        os.makedirs(config['training']['model_save_path'], exist_ok=True)
    
    def load_data(self, data_path):
        """
        데이터 로드
        
        Args:
            data_path: 데이터 디렉토리 경로
        
        Returns:
            X, y: 데이터와 라벨
        """
        print(f"\n데이터 로드 중: {data_path}")
        
        X_data = []
        y_labels = []
        
        # 각 라벨 폴더에서 데이터 로드
        for label in os.listdir(data_path):
            label_path = os.path.join(data_path, label)
            if not os.path.isdir(label_path):
                continue
            
            print(f"  라벨: {label}")
            sample_count = 0
            
            for filename in os.listdir(label_path):
                if not filename.endswith('.npy'):
                    continue
                
                filepath = os.path.join(label_path, filename)
                data = np.load(filepath)
                
                X_data.append(data)
                y_labels.append(label)
                sample_count += 1
            
            print(f"    샘플 수: {sample_count}")
        
        print(f"\n총 샘플 수: {len(X_data)}")
        print(f"총 클래스 수: {len(set(y_labels))}")
        
        return X_data, y_labels
    
    def preprocess_data(self, X_data, y_labels):
        """
        데이터 전처리
        
        Args:
            X_data: 원본 데이터 리스트
            y_labels: 라벨 리스트
        
        Returns:
            X_train, X_val, y_train, y_val: 전처리된 데이터
        """
        print("\n데이터 전처리 중...")
        
        # 시퀀스 길이 맞추기
        sequence_length = self.config['data']['sequence_length']
        processed_X = []
        
        for sample in X_data:
            if len(sample) < sequence_length:
                # 패딩
                padding = np.zeros((sequence_length - len(sample), sample.shape[1]))
                sample = np.vstack([sample, padding])
            elif len(sample) > sequence_length:
                # 자르기
                sample = sample[:sequence_length]
            
            # 정규화
            normalized_sample = []
            for frame in sample:
                normalized_frame = self.preprocessor.normalize_landmarks(frame)
                normalized_sample.append(normalized_frame)
            
            processed_X.append(np.array(normalized_sample))
        
        X = np.array(processed_X)
        
        # 라벨 인코딩
        y_encoded = self.label_encoder.fit_transform(y_labels)
        y = keras.utils.to_categorical(y_encoded)
        
        # Train/Validation 분할
        train_split = self.config['data']['train_split']
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=train_split, random_state=42, stratify=y_encoded
        )
        
        print(f"  학습 데이터: {X_train.shape}")
        print(f"  검증 데이터: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def train(self, X_train, X_val, y_train, y_val):
        """
        모델 학습
        
        Args:
            X_train, X_val: 학습/검증 데이터
            y_train, y_val: 학습/검증 라벨
        
        Returns:
            model: 학습된 모델
            history: 학습 히스토리
        """
        print("\n모델 생성 중...")
        
        # 모델 설정 업데이트
        model_config = self.config['model'].copy()
        model_config['num_classes'] = y_train.shape[1]
        
        # 모델 생성 및 컴파일
        model = create_model(model_config)
        model = compile_model(model, self.config['training'])
        
        # 모델 빌드
        model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
        model.summary()
        
        # 콜백 설정
        callbacks = self._create_callbacks()
        
        # 학습
        print("\n학습 시작...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def _create_callbacks(self):
        """학습 콜백 생성"""
        callbacks = []
        
        # Model Checkpoint
        checkpoint_path = os.path.join(
            self.config['training']['checkpoint_path'],
            'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5'
        )
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # ReduceLROnPlateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        log_dir = os.path.join(self.config['training']['checkpoint_path'], 'logs')
        tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
        callbacks.append(tensorboard)
        
        return callbacks
    
    def save_model(self, model):
        """
        모델 및 관련 파일 저장
        
        Args:
            model: 학습된 모델
        """
        save_path = self.config['training']['model_save_path']
        
        # 모델 저장
        model_path = os.path.join(save_path, 'sign_language_model.h5')
        model.save(model_path)
        print(f"\n모델 저장: {model_path}")
        
        # 라벨 인코더 저장
        label_encoder_path = os.path.join(save_path, 'label_encoder.pkl')
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"라벨 인코더 저장: {label_encoder_path}")
        
        # 전처리기 저장
        preprocessor_path = os.path.join(save_path, 'preprocessor.pkl')
        self.preprocessor.save(preprocessor_path)
        
        # 클래스 정보 저장
        class_info_path = os.path.join(save_path, 'classes.txt')
        with open(class_info_path, 'w', encoding='utf-8') as f:
            for i, label in enumerate(self.label_encoder.classes_):
                f.write(f"{i}: {label}\n")
        print(f"클래스 정보 저장: {class_info_path}")
    
    def evaluate(self, model, X_val, y_val):
        """
        모델 평가
        
        Args:
            model: 학습된 모델
            X_val: 검증 데이터
            y_val: 검증 라벨
        """
        print("\n모델 평가 중...")
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        print(f"  검증 손실: {loss:.4f}")
        print(f"  검증 정확도: {accuracy:.4f}")
        
        # 혼동 행렬 계산
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n분류 리포트:")
        print(classification_report(
            y_true_classes, y_pred_classes,
            target_names=self.label_encoder.classes_
        ))
        
        print("\n혼동 행렬:")
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        print(cm)


def main():
    parser = argparse.ArgumentParser(description='수화 인식 모델 학습')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--data', type=str, default='data/raw',
                       help='데이터 디렉토리 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 학습 파이프라인
    trainer = SignLanguageTrainer(config)
    
    # 1. 데이터 로드
    X_data, y_labels = trainer.load_data(args.data)
    
    # 2. 전처리
    X_train, X_val, y_train, y_val = trainer.preprocess_data(X_data, y_labels)
    
    # 3. 학습
    model, history = trainer.train(X_train, X_val, y_train, y_val)
    
    # 4. 평가
    trainer.evaluate(model, X_val, y_val)
    
    # 5. 저장
    trainer.save_model(model)
    
    # 6. 학습 히스토리 시각화
    history_plot_path = os.path.join(
        config['training']['model_save_path'],
        'training_history.png'
    )
    visualize_training_history(history, save_path=history_plot_path)
    
    print("\n학습 완료!")


if __name__ == "__main__":
    main()
