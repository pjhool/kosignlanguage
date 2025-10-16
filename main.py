"""
MediaPipe 수화 인식 시스템 메인 실행 파일
"""

import os
import sys
import argparse
import yaml

# 프로젝트 루트를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection import DataCollector
from src.train import SignLanguageTrainer
from src.inference import SignLanguageInference


def print_banner():
    """배너 출력"""
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║   MediaPipe 수화 인식 시스템                      ║
    ║   Sign Language Recognition System                 ║
    ╚═══════════════════════════════════════════════════╝
    """
    print(banner)


def collect_data_mode(config, args):
    """데이터 수집 모드"""
    print("\n[모드] 데이터 수집")
    
    if not args.label:
        print("오류: 라벨을 지정해주세요. (--label)")
        return
    
    collector = DataCollector(config, args.label, args.output_dir)
    collector.collect(args.num_samples, args.frames_per_sample)


def train_mode(config, args):
    """학습 모드"""
    print("\n[모드] 모델 학습")
    
    trainer = SignLanguageTrainer(config)
    
    # 데이터 로드
    X_data, y_labels = trainer.load_data(args.data_dir)
    
    if len(X_data) == 0:
        print("오류: 데이터가 없습니다. 먼저 데이터를 수집해주세요.")
        return
    
    # 전처리
    X_train, X_val, y_train, y_val = trainer.preprocess_data(X_data, y_labels)
    
    # 학습
    model, history = trainer.train(X_train, X_val, y_train, y_val)
    
    # 평가
    trainer.evaluate(model, X_val, y_val)
    
    # 저장
    trainer.save_model(model)
    
    # 학습 히스토리 시각화
    from utils.visualization import visualize_training_history
    history_plot_path = os.path.join(
        config['training']['model_save_path'],
        'training_history.png'
    )
    visualize_training_history(history, save_path=history_plot_path)
    
    print("\n학습 완료!")


def inference_mode(config, args):
    """추론 모드"""
    print("\n[모드] 실시간 추론")
    
    model_path = args.model_path
    
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 모델을 학습시켜주세요.")
        return
    
    inference = SignLanguageInference(config, model_path)
    inference.run()


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='MediaPipe 수화 인식 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  데이터 수집:
    python main.py collect --label "안녕하세요" --samples 100
  
  모델 학습:
    python main.py train --data data/raw
  
  실시간 추론:
    python main.py inference --model models/saved_models/sign_language_model.h5
        """
    )
    
    # 공통 인자
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='설정 파일 경로')
    
    # 서브커맨드
    subparsers = parser.add_subparsers(dest='mode', help='실행 모드')
    
    # 데이터 수집 모드
    collect_parser = subparsers.add_parser('collect', help='데이터 수집')
    collect_parser.add_argument('--label', type=str, required=True,
                               help='수집할 수화 단어 라벨')
    collect_parser.add_argument('--samples', type=int, default=100,
                               dest='num_samples',
                               help='수집할 샘플 개수')
    collect_parser.add_argument('--frames', type=int, default=30,
                               dest='frames_per_sample',
                               help='샘플당 프레임 수')
    collect_parser.add_argument('--output', type=str, default='data/raw',
                               dest='output_dir',
                               help='출력 디렉토리')
    
    # 학습 모드
    train_parser = subparsers.add_parser('train', help='모델 학습')
    train_parser.add_argument('--data', type=str, default='data/raw',
                             dest='data_dir',
                             help='데이터 디렉토리 경로')
    
    # 추론 모드
    inference_parser = subparsers.add_parser('inference', help='실시간 추론')
    inference_parser.add_argument('--model', type=str,
                                 default='models/saved_models/sign_language_model.h5',
                                 dest='model_path',
                                 help='모델 파일 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    if not os.path.exists(args.config):
        print(f"오류: 설정 파일을 찾을 수 없습니다: {args.config}")
        return
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 모드에 따라 실행
    if args.mode == 'collect':
        collect_data_mode(config, args)
    elif args.mode == 'train':
        train_mode(config, args)
    elif args.mode == 'inference':
        inference_mode(config, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
