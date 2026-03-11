"""
audio_transcriber.py
────────────────────
Two simple functions:
  1. record_audio(save_dir)  – press Enter to stop recording
  2. transcribe(audio_path)  – converts .wav file to text

Install:
    pip install faster-whisper pyaudio
    # On Linux: sudo apt-get install portaudio19-dev && pip install pyaudio
"""

import os
import wave
import threading
from datetime import datetime

import pyaudio
from faster_whisper import WhisperModel


# ── Audio settings ────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS    = 1
CHUNK_SIZE  = 1024
FORMAT      = pyaudio.paInt16

# ── Whisper model (loaded once at import) ─────────────────
print("⏳ Loading Whisper model...")
_whisper = WhisperModel("medium", device="cpu", compute_type="int8")
print("✅ Whisper ready.")


# ─────────────────────────────────────────────────────────
# 1. RECORD AUDIO
# ─────────────────────────────────────────────────────────
def record_audio(save_dir: str) -> str:
    """
    Record audio from microphone. Press Enter to stop.

    Parameter
    ---------
    save_dir : folder where the .wav file will be saved
               (created automatically if it doesn't exist)

    Returns
    -------
    str : full path to the saved .wav file
          e.g. "recordings/session_20250311_143022.wav"

    Usage
    -----
    path = record_audio("recordings")
    """
    os.makedirs(save_dir, exist_ok=True)

    # File named: session_YYYYMMDD_HHMMSS.wav
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"session_{timestamp}.wav"
    filepath  = os.path.join(save_dir, filename)

    # Background thread listens for Enter key
    stop_event = threading.Event()

    def _wait_for_enter():
        input()             # blocks until user presses Enter
        stop_event.set()

    threading.Thread(target=_wait_for_enter, daemon=True).start()

    # Open mic stream
    audio  = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    print("🎙️  Recording... press Enter to stop.")
    frames = []
    while not stop_event.is_set():
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Write WAV file
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    duration = len(frames) * CHUNK_SIZE / SAMPLE_RATE
    print(f"✅ Saved {duration:.1f}s  →  {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────
# 2. TRANSCRIBE AUDIO → TEXT
# ─────────────────────────────────────────────────────────
def transcribe(audio_path: str) -> str:
    """
    Convert a .wav file to English text using Whisper.

    Parameter
    ---------
    audio_path : path to the .wav file to transcribe

    Returns
    -------
    str : full transcript as plain text

    Usage
    -----
    text = transcribe("recordings/session_20250311_143022.wav")
    print(text)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    print(f"🔍 Transcribing: {audio_path} ...")

    segments, _ = _whisper.transcribe(
        audio_path,
        language="en",          # English only
        beam_size=5,
        vad_filter=True,        # skip silent gaps automatically
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    text = " ".join(seg.text.strip() for seg in segments)

    print(f"✅ Done ({len(text)} chars)")
    return text


# ─────────────────────────────────────────────────────────
# QUICK DEMO  (python audio_transcriber.py)
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    audio_path = record_audio("recordings")
    transcript = transcribe(audio_path)

    print("\n── Transcript ──────────────────────────")
    print(transcript)