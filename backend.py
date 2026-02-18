import whisper
import torch
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor


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
# Session Emotion Log
# -----------------------------

emotion_log = []


# -----------------------------
# Audio Preprocessing
# -----------------------------

def preprocess_audio(audio_path: str, target_sr: int = 16000):
    """
    Load and standardize audio:
    - mono
    - resampled
    - normalized
    """

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # resample if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # normalize amplitude
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, target_sr


# -----------------------------
# HuBERT Emotion (Voice)
# -----------------------------

def detect_voice_emotion(audio_path: str):
    y, sr = preprocess_audio(audio_path)

    inputs = hubert_processor(
        y,
        sampling_rate=sr,
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

    top = result[0]

    emotions = {
        top["label"].capitalize(): float(round(top["score"], 3))
    }

    dominant = top["label"].capitalize()
    return dominant, emotions


# -----------------------------
# Intelligent Fusion + Confidence
# -----------------------------

def fuse_emotions(text_emotion, voice_emotion, features,
                  text_scores=None, voice_scores=None):

    jitter = features["jitter_percent"]
    pitch = features["pitch_mean"]

    confidence = 0.6  # base confidence
    reasoning = ""

    # ---- Anxiety inference ----
    if voice_emotion == "Neutral" and jitter > 1.5:
        confidence = min(1.0, 0.6 + jitter / 10)
        reasoning = f"Neutral voice + high jitter ({jitter}%) → anxiety inferred"
        return "Anxious", round(confidence, 2), reasoning

    # ---- Distress inference ----
    if voice_emotion == "Sad" and pitch > 250:
        confidence = 0.8
        reasoning = f"Sad voice + high pitch ({pitch} Hz) → distress inferred"
        return "Distressed", round(confidence, 2), reasoning

    # ---- Agreement boost ----
    if text_emotion == voice_emotion:
        confidence = 0.9
        reasoning = "Text and voice agree"
        return voice_emotion, round(confidence, 2), reasoning

    # ---- Default ----
    confidence = 0.7
    reasoning = "Voice dominates (prosody-first)"
    return voice_emotion, round(confidence, 2), reasoning



# -----------------------------
# Acoustic Biomarker Extraction
# -----------------------------

def extract_acoustic_features(audio_path: str):
    """
    Extract vocal stress biomarkers:
    - jitter (micro tremor)
    - pitch mean
    """

    y, sr = preprocess_audio(audio_path)

    # ---------- Pitch tracking ----------
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_values = pitch_values[pitch_values > 0]

    if len(pitch_values) > 0:
        pitch_mean = float(np.mean(pitch_values))
    else:
        pitch_mean = 0.0

    # ---------- Jitter estimation ----------
    # Simple jitter proxy: frame-to-frame pitch variation
    if len(pitch_values) > 1:
        diffs = np.abs(np.diff(pitch_values))
        jitter = float(np.mean(diffs) / (pitch_mean + 1e-6))
    else:
        jitter = 0.0

    # Convert to percentage
    jitter_percent = jitter * 100

    return {
        "pitch_mean": round(pitch_mean, 2),
        "jitter_percent": round(jitter_percent, 2)
    }


def log_emotion_event(entry: dict):
    """
    Store emotion event in session memory
    """
    emotion_log.append(entry)


# -----------------------------
# Parallel Multimodal Analysis
# -----------------------------

def run_multimodal_analysis(audio_path: str):
    """
    Runs transcription + voice emotion + acoustic features in parallel, improves speed
    """

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_transcript = executor.submit(transcribe_audio, audio_path)
        future_voice = executor.submit(detect_voice_emotion, audio_path)
        future_features = executor.submit(extract_acoustic_features, audio_path)

        transcript = future_transcript.result()
        voice_label, voice_scores = future_voice.result()
        features = future_features.result()

    text_label, text_scores = detect_text_emotion(transcript)

    final_emotion, confidence, reasoning = fuse_emotions(
        text_label,
        voice_label,
        features,
        text_scores,
        voice_scores
    )

    import time

    event = {
        "timestamp": round(time.time(), 2),
        "transcript": transcript,
        "voice": voice_label,
        "text": text_label,
        "final": final_emotion,
        "confidence": confidence,
        "jitter": features["jitter_percent"],
        "pitch": features["pitch_mean"]
    }

    log_emotion_event(event)


    return {
        "transcript": transcript,
        "voice_label": voice_label,
        "voice_scores": voice_scores,
        "text_label": text_label,
        "text_scores": text_scores,
        "features": features,
        "final_emotion": final_emotion,
        "confidence": confidence,
        "reasoning": reasoning,
        "log_length": len(emotion_log)
}
