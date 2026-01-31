import whisper
import torch
import librosa
import numpy as np

from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    pipeline
)

# -----------------------------
# Load models ONCE
# -----------------------------

whisper_model = whisper.load_model("base")

hubert_model_name = "superb/hubert-large-superb-er"
hubert_model = HubertForSequenceClassification.from_pretrained(hubert_model_name)
hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)

text_emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)


hubert_model.eval()

# Label map for HuBERT
HUBERT_LABELS = {
    0: "Neutral",
    1: "Happy",
    2: "Angry",
    3: "Sad"
}

# -----------------------------
# Whisper
# -----------------------------

def transcribe_audio(audio_path: str) -> str:
    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()

# -----------------------------
# HuBERT Emotion (Voice)
# -----------------------------

def detect_voice_emotion(audio_path: str):
    y, sr = librosa.load(audio_path, sr=16000)

    inputs = hubert_processor(
        y,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = hubert_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    emotions = {
        HUBERT_LABELS[i]: float(round(probs[i], 3))
        for i in range(len(probs))
    }

    dominant = max(emotions, key=emotions.get)
    return dominant, emotions

# -----------------------------
# Text Emotion
# -----------------------------

def detect_text_emotion(text: str):
    result = text_emotion_pipeline(text)

    # result = [{'label': 'joy', 'score': 0.92}]
    top = result[0]

    emotions = {
        top["label"].capitalize(): float(round(top["score"], 3))
    }

    dominant = top["label"].capitalize()
    return dominant, emotions


# -----------------------------
# Fusion
# -----------------------------

def fuse_emotions(text_emotion, voice_emotion):
    if text_emotion == voice_emotion:
        return text_emotion, "Text and voice agree"
    return voice_emotion, "Voice dominates (prosody-first)"
