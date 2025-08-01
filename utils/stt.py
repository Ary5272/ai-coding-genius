# utils/stt.py
from transformers import pipeline

class SpeechToText:
    def __init__(self, model_name="openai/whisper-small"):
        self.pipe = pipeline("automatic-speech-recognition", model=model_name)

    def transcribe(self, audio_path):
        return self.pipe(audio_path)["text"]
