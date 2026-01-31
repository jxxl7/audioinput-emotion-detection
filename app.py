import streamlit as st
import tempfile
import os

from backend import (
    transcribe_audio,
    detect_voice_emotion,
    detect_text_emotion,
    fuse_emotions
)

st.set_page_config(
    page_title="Multimodal Emotion Detection",
    layout="centered"
)

st.title("🎙️ Multimodal Emotion Detection")
st.caption("Whisper (what was said) + HuBERT (how it was said)")

# -----------------------------
# INPUT MODE SELECTION
# -----------------------------

mode = st.radio(
    "Choose input method:",
    ["🎤 Speak (Microphone)", "📁 Upload Audio File"]
)

audio_path = None

# -----------------------------
# MICROPHONE INPUT
# -----------------------------

if mode == "🎤 Speak (Microphone)":
    audio_bytes = st.audio_input("Speak now")

    if audio_bytes is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes.getvalue())
            audio_path = tmp.name

# -----------------------------
# FILE UPLOAD INPUT
# -----------------------------

else:
    uploaded_file = st.file_uploader(
        "Upload speech audio (WAV / MP3 / FLAC)",
        type=["wav", "mp3", "flac"]
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

# -----------------------------
# PROCESS AUDIO
# -----------------------------

if audio_path:
    st.audio(audio_path)

    with st.spinner("Analyzing speech..."):
        transcript = transcribe_audio(audio_path)

        voice_label, voice_scores = detect_voice_emotion(audio_path)
        text_label, text_scores = detect_text_emotion(transcript)

        final_emotion, reasoning = fuse_emotions(
            text_label,
            voice_label
        )

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------

    st.subheader("📝 Transcription")
    st.write(transcript)

    st.subheader("🗣 Voice Emotion (HuBERT)")
    st.json(voice_scores)

    st.subheader("📖 Text Emotion")
    st.json(text_scores)

    st.subheader("🧠 Final Emotion")
    st.success(final_emotion)
    st.caption(reasoning)

    os.unlink(audio_path)
