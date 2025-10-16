"""
테스트 스크립트
MediaPipe와 기본 기능 테스트
"""

import cv2
import sys
import os
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mediapipe_utils import MediaPipeHandler


def test_camera():
    """카메라 테스트"""
    print("\n=== 카메라 테스트 ===")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return False
    
    ret, frame = cap.read()
    if ret:
        print(f"✅ 카메라 작동 중 (해상도: {frame.shape[1]}x{frame.shape[0]})")
    else:
        print("❌ 카메라에서 프레임을 읽을 수 없습니다.")
        cap.release()
        return False
    
    cap.release()
    return True


def test_mediapipe():
    """MediaPipe 테스트"""
    print("\n=== MediaPipe 테스트 ===")
    
    try:
        # 설정 로드
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # MediaPipe 초기화
        mp_handler = MediaPipeHandler(config)
        print("✅ MediaPipe 초기화 성공")
        
        # 카메라로 테스트
        cap = cv2.VideoCapture(0)
        
        print("\n손을 카메라 앞에 보여주세요.")
        print("Q를 눌러 종료하세요.")
        
        frame_count = 0
        detected_count = 0
        
        while frame_count < 100:  # 100 프레임만 테스트
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe 처리
            results, landmarks = mp_handler.process_frame(frame)
            
            # 랜드마크 감지 확인
            if landmarks.size > 0:
                detected_count += 1
            
            # 시각화
            annotated_frame = mp_handler.draw_landmarks(frame, results)
            
            # 정보 표시
            cv2.putText(annotated_frame, 
                       f"Frames: {frame_count}/100", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, 
                       f"Detected: {detected_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            if landmarks.size > 0:
                cv2.putText(annotated_frame, 
                           "HAND DETECTED!", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
            
            cv2.imshow('MediaPipe Test', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        mp_handler.close()
        
        print(f"\n✅ 테스트 완료: {detected_count}/{frame_count} 프레임에서 손 감지됨")
        
        if detected_count > 0:
            print("✅ MediaPipe가 정상적으로 작동합니다!")
            return True
        else:
            print("⚠️  손이 감지되지 않았습니다. 카메라 앞에 손을 보여주세요.")
            return False
            
    except Exception as e:
        print(f"❌ MediaPipe 테스트 실패: {e}")
        return False


def test_tts():
    """TTS 테스트"""
    print("\n=== TTS 테스트 ===")
    
    try:
        from utils.tts_utils import TTSHandler
        
        # 설정 로드
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        tts = TTSHandler(config)
        print("✅ TTS 초기화 성공")
        
        print("음성 출력 테스트 중...")
        tts.speak("안녕하세요. 수화 인식 시스템 테스트입니다.")
        
        tts.close()
        print("✅ TTS 테스트 성공")
        return True
        
    except Exception as e:
        print(f"❌ TTS 테스트 실패: {e}")
        print("TTS가 작동하지 않아도 시스템은 사용 가능합니다.")
        return False


def test_config():
    """설정 파일 테스트"""
    print("\n=== 설정 파일 테스트 ===")
    
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 필수 키 확인
        required_keys = ['data', 'mediapipe', 'model', 'training', 'inference', 'tts']
        for key in required_keys:
            if key not in config:
                print(f"❌ 설정 파일에 '{key}' 항목이 없습니다.")
                return False
        
        print("✅ 설정 파일 로드 성공")
        print(f"   - 모델 타입: {config['model']['type']}")
        print(f"   - 시퀀스 길이: {config['data']['sequence_length']}")
        print(f"   - 클래스 수: {config['model']['num_classes']}")
        return True
        
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        return False


def test_directories():
    """디렉토리 구조 테스트"""
    print("\n=== 디렉토리 구조 테스트 ===")
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'models/checkpoints',
        'models/saved_models',
        'src',
        'utils'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (생성 필요)")
            os.makedirs(dir_path, exist_ok=True)
            print(f"   → 생성됨")
    
    return all_exist


def main():
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║   MediaPipe 수화 인식 시스템 - 테스트 스크립트    ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    # 테스트 실행
    results = {}
    
    results['directories'] = test_directories()
    results['config'] = test_config()
    results['camera'] = test_camera()
    results['mediapipe'] = test_mediapipe()
    results['tts'] = test_tts()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{test_name.capitalize()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ 모든 테스트를 통과했습니다!")
        print("이제 데이터 수집을 시작할 수 있습니다.")
    else:
        print("\n⚠️  일부 테스트에 실패했습니다.")
        print("문제를 해결한 후 다시 테스트하세요.")


if __name__ == "__main__":
    main()
