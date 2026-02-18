import streamlit as st
import tempfile
import os

from backend import run_multimodal_analysis

st.set_page_config(
    page_title="Multimodal Emotion Detection",
    layout="centered"
)

st.title(" Multimodal Emotion Detection")
st.caption("Whisper (what was said) + HuBERT (how it was said)")

# -----------------------------
# INPUT MODE SELECTION
# -----------------------------

mode = st.radio(
    "Choose input method:",
    [" Speak (Microphone)", " Upload Audio File"]
)

audio_path = None

# -----------------------------
# MICROPHONE INPUT
# -----------------------------

if mode == " Speak (Microphone)":
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
        results = run_multimodal_analysis(audio_path)

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------

    st.subheader(" Transcription")
    st.write(results["transcript"])

    st.subheader(" Voice Emotion (HuBERT)")
    st.json(results["voice_scores"])

    st.subheader(" Text Emotion")
    st.json(results["text_scores"])

    st.subheader(" Acoustic Features")
    st.json(results["features"])

    st.subheader(" Session Emotion Log Length")
    st.write(results["log_length"])

    st.subheader(" Final Emotion")
    st.success(f'{results["final_emotion"]} (confidence: {results["confidence"]})')
    st.caption(results["reasoning"])




    os.unlink(audio_path)
