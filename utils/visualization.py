"""
시각화 유틸리티 함수
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_text(frame, text, position=(10, 30), font_scale=1.0, color=(0, 255, 0), thickness=2):
    """
    프레임에 텍스트 그리기
    
    Args:
        frame: 이미지 프레임
        text: 표시할 텍스트
        position: 텍스트 위치 (x, y)
        font_scale: 폰트 크기
        color: 색상 (B, G, R)
        thickness: 두께
    
    Returns:
        annotated_frame: 텍스트가 추가된 프레임
    """
    frame_copy = frame.copy()
    
    # 한글 지원을 위한 PIL 사용 (선택사항)
    try:
        from PIL import ImageFont, ImageDraw, Image
        
        # 한글 폰트 (없으면 기본 폰트)
        try:
            font = ImageFont.truetype("malgun.ttf", int(font_scale * 30))
        except:
            font = ImageFont.load_default()
        
        # PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        
        # OpenCV 이미지로 변환
        frame_copy = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        # PIL이 없거나 실패하면 OpenCV 사용 (영어만 지원)
        cv2.putText(frame_copy, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness, cv2.LINE_AA)
    
    return frame_copy


def draw_info_panel(frame, predictions, confidence, fps=None):
    """
    정보 패널 그리기 (예측 결과, 신뢰도, FPS 등)
    
    Args:
        frame: 이미지 프레임
        predictions: 예측 결과 리스트 [(클래스명, 확률), ...]
        confidence: 최고 신뢰도
        fps: FPS 값 (선택사항)
    
    Returns:
        annotated_frame: 정보 패널이 추가된 프레임
    """
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    # 반투명 패널 생성
    overlay = frame_copy.copy()
    panel_height = 150
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    alpha = 0.6
    frame_copy = cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0)
    
    # FPS 표시
    if fps is not None:
        frame_copy = draw_text(frame_copy, f"FPS: {fps:.1f}", 
                              position=(10, 30), color=(0, 255, 255))
    
    # 최고 예측 결과 표시
    if predictions and len(predictions) > 0:
        top_class, top_prob = predictions[0]
        frame_copy = draw_text(frame_copy, f"Prediction: {top_class}", 
                              position=(10, 70), font_scale=1.2, color=(0, 255, 0))
        frame_copy = draw_text(frame_copy, f"Confidence: {top_prob:.2%}", 
                              position=(10, 110), color=(0, 255, 0))
    else:
        frame_copy = draw_text(frame_copy, "No prediction", 
                              position=(10, 70), color=(0, 0, 255))
    
    return frame_copy


def plot_landmarks(landmarks, title="Landmarks", save_path=None):
    """
    랜드마크를 3D 플롯으로 시각화
    
    Args:
        landmarks: (N, 3) 형태의 랜드마크 배열
        title: 플롯 제목
        save_path: 저장 경로 (선택사항)
    """
    if landmarks.size == 0:
        print("랜드마크가 비어있습니다.")
        return
    
    # 랜드마크를 21개씩 나누기 (손)
    landmarks = landmarks.reshape(-1, 3)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 랜드마크 포인트 그리기
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], 
              c='blue', marker='o', s=50)
    
    # 연결선 그리기 (손가락 구조)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 엄지
        (0, 5), (5, 6), (6, 7), (7, 8),  # 검지
        (0, 9), (9, 10), (10, 11), (11, 12),  # 중지
        (0, 13), (13, 14), (14, 15), (15, 16),  # 약지
        (0, 17), (17, 18), (18, 19), (19, 20),  # 새끼
        (5, 9), (9, 13), (13, 17)  # 손바닥
    ]
    
    for i, j in connections:
        if i < len(landmarks) and j < len(landmarks):
            ax.plot([landmarks[i, 0], landmarks[j, 0]],
                   [landmarks[i, 1], landmarks[j, 1]],
                   [landmarks[i, 2], landmarks[j, 2]], 'r-', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def visualize_training_history(history, save_path='training_history.png'):
    """
    학습 히스토리 시각화
    
    Args:
        history: Keras History 객체 또는 히스토리 딕셔너리
        save_path: 저장 경로
    """
    if hasattr(history, 'history'):
        history = history.history
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 플롯
    axes[0].plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy 플롯
    acc_key = 'accuracy' if 'accuracy' in history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
    
    if acc_key in history:
        axes[1].plot(history[acc_key], label='Train Accuracy')
        if val_acc_key in history:
            axes[1].plot(history[val_acc_key], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"학습 히스토리 저장: {save_path}")
    plt.close()
