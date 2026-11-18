"""Local Wav2Vec2 emotion inference on raw 16 kHz mono audio."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "audio" / "wav2vec2_best_model"
SAMPLE_RATE = 16000
TARGET_LENGTH = SAMPLE_RATE * 8
EMOTION_CLASSES = {"angry", "disgust", "fear", "happy", "neutral", "sad"}


@lru_cache(maxsize=1)
def _load_model():
    """Load only the local safetensors checkpoint; never download or unpickle."""
    for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
        if not (MODEL_PATH / filename).is_file():
            raise FileNotFoundError(f"Required Wav2Vec2 model file not found: {MODEL_PATH / filename}")
    with (MODEL_PATH / "config.json").open(encoding="utf-8") as source:
        config = json.load(source)
    mapping = config.get("id2label", {})
    if set(mapping) != {str(i) for i in range(6)}:
        raise ValueError("Wav2Vec2 config.id2label must contain IDs 0 through 5.")
    labels = [mapping[str(i)] for i in range(6)]
    if set(labels) != EMOTION_CLASSES:
        raise ValueError("Wav2Vec2 config must define the six supported emotion labels.")

    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL_PATH), local_files_only=True)
    if extractor.sampling_rate != SAMPLE_RATE:
        raise ValueError("Wav2Vec2 feature extractor must use 16000 Hz audio.")
    model, loading_info = Wav2Vec2ForSequenceClassification.from_pretrained(
        str(MODEL_PATH), local_files_only=True, use_safetensors=True,
        output_loading_info=True,
    )
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
        raise ValueError(f"Incomplete Wav2Vec2 checkpoint: {loading_info}")
    model.to(torch.device("cpu"))
    model.eval()
    return model, extractor, labels


def extract_audio_from_video(video_path):
    """Decode any supported media using the existing bundled FFmpeg setup."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {path}")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
        wav_path = Path(temporary.name)
    try:
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error",
             "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1",
             "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le", str(wav_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            errors="replace", timeout=300,
        )
        if result.returncode:
            raise ValueError("Audio extraction failed (invalid media or missing audio): " + result.stderr[-1000:])
        return str(wav_path)
    except Exception:
        wav_path.unlink(missing_ok=True)
        raise


def load_audio(audio_path):
    if audio_path is None:
        raise ValueError("Audio path must not be None.")
    temporary = extract_audio_from_video(audio_path)
    try:
        with wave.open(temporary, "rb") as source:
            if (source.getnchannels(), source.getframerate(), source.getsampwidth()) != (1, SAMPLE_RATE, 2):
                raise ValueError("Decoded audio must be 16 kHz mono PCM16.")
            signal = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if not signal.size:
            raise ValueError("Audio is empty.")
        if not np.isfinite(signal).all():
            raise ValueError("Audio contains non-finite samples.")
        return signal
    finally:
        Path(temporary).unlink(missing_ok=True)


def predict_audio_emotion(audio_path):
    """Return percentage probabilities and the number of eight-second segments."""
    signal = load_audio(audio_path)
    try:
        import torch

        model, extractor, labels = _load_model()
        # Pad only exceptionally short tails to the convolutional receptive field.
        minimum_samples = 1
        for kernel, stride in reversed(list(zip(model.config.conv_kernel, model.config.conv_stride))):
            minimum_samples = (minimum_samples - 1) * stride + kernel
        probabilities = []
        durations = []
        with torch.inference_mode():
            for start in range(0, len(signal), TARGET_LENGTH):
                segment = signal[start:start + TARGET_LENGTH]
                durations.append(len(segment))
                if len(segment) < minimum_samples:
                    segment = np.pad(segment, (0, minimum_samples - len(segment)))
                inputs = extractor(segment, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False)
                logits = model(**inputs).logits
                if tuple(logits.shape) != (1, 6) or not torch.isfinite(logits).all():
                    raise ValueError("Wav2Vec2 returned invalid emotion logits.")
                probabilities.append(torch.softmax(logits, dim=-1)[0].cpu().numpy())
        averaged = np.average(probabilities, axis=0, weights=durations)
        averaged /= averaged.sum()
        index = int(np.argmax(averaged))
        return {
            "emotion": labels[index],
            "confidence": round(float(averaged[index]) * 100, 2),
            "probabilities": {label: round(float(value) * 100, 2) for label, value in zip(labels, averaged)},
            "segments": len(probabilities),
        }
    except (FileNotFoundError, ValueError):
        raise
    except Exception as error:
        raise RuntimeError(f"Wav2Vec2 audio prediction failed: {error}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", nargs="?", help="Audio or video file; defaults to scanning backend/uploads")
    args = parser.parse_args()
    paths = [Path(args.audio_path)] if args.audio_path else sorted((ROOT / "uploads").glob("*"))
    paths = [path for path in paths if path.is_file()]
    if args.audio_path and not paths:
        paths = [Path(args.audio_path)]
    if not paths:
        print("No audio/video files found in backend/uploads.", file=sys.stderr)
        return 1
    try:
        for path in paths:
            print(json.dumps(predict_audio_emotion(str(path))))
        return 0
    except Exception as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
