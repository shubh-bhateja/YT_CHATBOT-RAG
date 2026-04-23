import os
import static_ffmpeg
from datetime import datetime
from faster_whisper import WhisperModel
from pytubefix import YouTube

static_ffmpeg.add_paths()

# ── FIX 4: Load Whisper model once at module level instead of per call ──
# This avoids reloading the model from disk every time transcribe_audio() runs.
_whisper_model = None

def _get_whisper_model(model_size="base"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _whisper_model
# ───────────────────────────────────────────────────────────────────────

def download_audio(url, output_dir="temp_audio"):
    """
    Download audio using pytubefix.
    output_dir can be overridden with a tempfile.TemporaryDirectory path
    so the caller controls cleanup (see app.py FIX 5).
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').last()
    if not audio_stream:
        raise Exception("No audio stream found for this video.")
    out_file = audio_stream.download(output_path=output_dir, filename=f"audio_{timestamp}")
    return out_file

def transcribe_audio(audio_path, model_size="base"):
    """
    Transcribe audio using faster-whisper.
    Reuses the already-loaded model instead of reloading from disk each call.
    """
    # ── FIX 4 (cont): use cached model ──
    model = _get_whisper_model(model_size)
    segments, info = model.transcribe(audio_path, beam_size=5)

    formatted_transcript = []
    for segment in segments:
        formatted_transcript.append({
            'start': segment.start,
            'text': segment.text.strip(),
            'duration': segment.end - segment.start
        })
    return formatted_transcript

def cleanup_audio(audio_path):
    """Manual cleanup — kept for backward compatibility but prefer TemporaryDirectory."""
    if os.path.exists(audio_path):
        os.remove(audio_path)

# End of file
