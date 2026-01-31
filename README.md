# ===============================
# HOW TO RUN THIS PROJECT
# ===============================

# 1. Clone the repository
#    git clone https://github.com/<your-username>/multimodal-emotion-detection.git
#    cd multimodal-emotion-detection

# 2. Create a virtual environment
#    python -m venv ser_env

# 3. Activate the virtual environment
#    Windows:
#      ser_env\Scripts\activate
#    Linux / macOS:
#      source ser_env/bin/activate

# 4. Install dependencies
#    pip install -r requirements.txt

# 5. Install FFmpeg (REQUIRED for Whisper)
#    Windows:
#      Download "ffmpeg-release-full-shared" from:
#      https://www.gyan.dev/ffmpeg/builds/
#      Add ffmpeg/bin to PATH
#      Verify: ffmpeg -version
#
#    Linux:
#      sudo apt install ffmpeg
#
#    macOS:
#      brew install ffmpeg

# 6. Run the application
#    streamlit run app.py

# 7. Open browser at:
#    http://localhost:8501

# ===============================
# INPUT MODES
# ===============================
# - Speak directly using microphone
# - Upload an audio file (WAV / MP3 / FLAC)

# ===============================
# OUTPUT
# ===============================
# - Whisper transcription (what was said)
# - Voice emotion via HuBERT (how it was said)
# - Text emotion via RoBERTa
# - Final fused emotion with explanation
