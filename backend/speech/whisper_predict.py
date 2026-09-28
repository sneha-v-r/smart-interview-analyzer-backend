"""Run speech recognition in a separate process to release model memory."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

WORKER_PATH = Path(__file__).resolve().with_name("whisper_worker.py")


def transcribe_video(video_path):
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    with tempfile.NamedTemporaryFile(suffix=".json", prefix="sia_whisper_", delete=False) as file:
        result_path = Path(file.name)
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [sys.executable, "-B", str(WORKER_PATH), str(video_path), str(result_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
        )
        if completed.returncode:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"Whisper tiny worker failed (exit {completed.returncode}):\n{details[-5000:]}"
            )
        try:
            with result_path.open(encoding="utf-8") as file:
                result = json.load(file)
        except (OSError, ValueError) as error:
            raise RuntimeError("Whisper worker returned no valid result JSON.") from error
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise RuntimeError("Whisper worker result is missing a transcript string.")
        result["text"] = result["text"].strip()
        if not result["text"]:
            raise RuntimeError("Whisper worker returned an empty transcript.")
        # Preserve successful-worker diagnostics, including library warnings.
        result["worker_stderr"] = completed.stderr.strip()
        return result
    finally:
        result_path.unlink(missing_ok=True)
