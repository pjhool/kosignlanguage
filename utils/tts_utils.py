"""
TTS (Text-to-Speech) 유틸리티
"""

import pyttsx3
from gtts import gTTS
import os
import tempfile
import pygame


class TTSHandler:
    """텍스트를 음성으로 변환하는 클래스"""
    
    def __init__(self, config):
        """
        Args:
            config: YAML 설정 딕셔너리
        """
        self.config = config['tts']
        self.engine_type = self.config['engine']
        
        if self.engine_type == 'pyttsx3':
            self._init_pyttsx3()
        elif self.engine_type == 'gtts':
            self._init_gtts()
        else:
            raise ValueError(f"지원하지 않는 TTS 엔진: {self.engine_type}")
    
    def _init_pyttsx3(self):
        """pyttsx3 엔진 초기화 (오프라인)"""
        try:
            self.engine = pyttsx3.init()
            
            # 음성 속도 설정
            self.engine.setProperty('rate', self.config['rate'])
            
            # 음량 설정
            self.engine.setProperty('volume', self.config['volume'])
            
            # 한국어 음성 찾기
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'korean' in voice.name.lower() or 'ko' in voice.languages:
                    self.engine.setProperty('voice', voice.id)
                    break
            
            print(f"[TTS] pyttsx3 엔진 초기화 완료")
        except Exception as e:
            print(f"[TTS] pyttsx3 초기화 실패: {e}")
            print("[TTS] gTTS로 전환합니다.")
            self.engine_type = 'gtts'
            self._init_gtts()
    
    def _init_gtts(self):
        """gTTS 초기화 (온라인 필요)"""
        pygame.mixer.init()
        print(f"[TTS] gTTS 엔진 초기화 완료")
    
    def speak(self, text):
        """
        텍스트를 음성으로 변환하여 재생
        
        Args:
            text: 변환할 텍스트
        """
        if not text or text.strip() == "":
            return
        
        try:
            if self.engine_type == 'pyttsx3':
                self._speak_pyttsx3(text)
            elif self.engine_type == 'gtts':
                self._speak_gtts(text)
        except Exception as e:
            print(f"[TTS] 음성 변환 오류: {e}")
    
    def _speak_pyttsx3(self, text):
        """pyttsx3로 음성 재생"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def _speak_gtts(self, text):
        """gTTS로 음성 재생"""
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
        
        try:
            # gTTS로 음성 생성
            tts = gTTS(text=text, lang=self.config['language'], slow=False)
            tts.save(temp_file)
            
            # pygame으로 재생
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # 재생 완료 대기
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        finally:
            # 임시 파일 삭제
            try:
                os.remove(temp_file)
            except:
                pass
    
    def close(self):
        """리소스 해제"""
        if self.engine_type == 'pyttsx3' and hasattr(self, 'engine'):
            try:
                self.engine.stop()
            except:
                pass
        elif self.engine_type == 'gtts':
            try:
                pygame.mixer.quit()
            except:
                pass


# 간단한 테스트 함수
def test_tts():
    """TTS 테스트"""
    import yaml
    
    # 테스트 설정
    config = {
        'tts': {
            'engine': 'pyttsx3',
            'language': 'ko',
            'rate': 150,
            'volume': 1.0
        }
    }
    
    tts = TTSHandler(config)
    tts.speak("안녕하세요. 수화 인식 시스템입니다.")
    tts.close()


if __name__ == "__main__":
    test_tts()
