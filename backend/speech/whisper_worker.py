"""Isolated Whisper tiny worker: video to English transcript."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "tools"


def configure_ffmpeg():
    executable = TOOLS_DIR / "ffmpeg.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"Project FFmpeg not found: {executable}")
    os.environ["PATH"] = str(TOOLS_DIR) + os.pathsep + os.environ.get("PATH", "")
    check = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check.returncode:
        raise RuntimeError("FFmpeg verification failed:\n" + check.stderr[-3000:])
    return executable, check.stdout.splitlines()[0]


def transcribe_worker(video_path, result_json_path):
    ffmpeg_path, ffmpeg_version = configure_ffmpeg()
    import torch
    import whisper

    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", prefix="sia_whisper_", delete=False) as file:
            wav_path = Path(file.name)
        extraction = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1",
             "-ar", "22050", "-acodec", "pcm_s16le", str(wav_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace",
        )
        if extraction.returncode:
            raise RuntimeError("FFmpeg audio extraction failed:\n" + extraction.stderr[-3000:])
        if not wav_path.is_file() or wav_path.stat().st_size == 0:
            raise FileNotFoundError("Extracted audio WAV was not created or is empty.")

        model = whisper.load_model("tiny", device=device)
        result = model.transcribe(str(wav_path), fp16=(device == "cuda"), language="en")
        text = result.get("text", "").strip()
        if not text:
            raise ValueError("Whisper produced an empty transcription")
        payload = {
            "text": text, "device": device, "model": "tiny",
            "cuda_available": torch.cuda.is_available(), "fp16": device == "cuda",
            "torch_version": torch.__version__, "whisper_import_path": whisper.__file__,
            "ffmpeg_path": str(ffmpeg_path), "ffmpeg_version": ffmpeg_version,
        }
        with Path(result_json_path).open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: whisper_worker.py VIDEO_PATH RESULT_JSON_PATH")
    transcribe_worker(sys.argv[1], sys.argv[2])
