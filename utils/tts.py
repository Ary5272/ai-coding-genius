# utils/tts.py
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset
import torch
import scipy.io.wavfile as wavfile

class TextToSpeech:
    def __init__(self):
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        # Load speaker embedding
        embeds = load_dataset("Matthijs/cmu-arctic-xvectors", name="cmu_us_awb", split="validation")
        self.speaker_embeddings = torch.tensor(embeds[7306]["xvector"]).unsqueeze(0)

    def speak(self, text, output_path="assets/response.wav"):
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings)
        wavfile.write(output_path, rate=16000, data=speech.numpy())
        return output_path
