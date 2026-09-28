"""Video emotion inference using the confirmed six-class video checkpoint."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "video" / "best_video_cnn.keras"
EXPECTED_INPUT_SHAPE = (None, 128, 128, 3)
EXPECTED_OUTPUT_SHAPE = (None, 6)
EMOTION_CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad")
SAMPLE_INTERVAL_SECONDS = 0.25
MAX_SAMPLED_FRAMES = 120
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


@lru_cache(maxsize=1)
def load_video_model():
    """Load once on demand and enforce the checkpoint's exact shapes."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Video checkpoint not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    if tuple(model.input_shape) != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"Expected video input {EXPECTED_INPUT_SHAPE}, got {model.input_shape}"
        )
    if tuple(model.output_shape) != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(
            f"Expected video output {EXPECTED_OUTPUT_SHAPE}, got {model.output_shape}"
        )
    return model


def load_face_detector():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar face cascade: {CASCADE_PATH}")
    return detector


def prepare_face(frame, detector):
    """Return one normalized RGB face batch, or None for an unusable frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    margin_x = int(width * 0.15)
    margin_y = int(height * 0.15)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame.shape[1], x + width + margin_x)
    y2 = min(frame.shape[0], y + height + margin_y)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (128, 128))
    face = face.astype(np.float32) / 255.0
    return np.expand_dims(face, axis=0)


def predict_video_emotion(video_path):
    """Average predictions over usable faces in at most 120 sampled frames."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = load_video_model()
    detector = load_face_detector()
    capture = cv2.VideoCapture(str(video_path))
    probabilities = []
    sampled_frames = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps is None or not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        step = max(1, int(round(fps * SAMPLE_INTERVAL_SECONDS)))
        frame_index = 0

        while sampled_frames < MAX_SAMPLED_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0:
                sampled_frames += 1
                batch = prepare_face(frame, detector)
                if batch is not None:
                    if batch.shape != (1, 128, 128, 3):
                        raise ValueError(f"Unexpected video input batch: {batch.shape}")
                    probs = np.asarray(model.predict(batch, verbose=0)[0], dtype=np.float32)
                    if probs.shape != (6,) or not np.isfinite(probs).all():
                        raise ValueError("Video model must return six finite probabilities.")
                    probabilities.append(probs)
            frame_index += 1
    finally:
        capture.release()

    if not probabilities:
        raise ValueError("No usable face was detected in the sampled video frames.")

    final_probs = np.mean(np.asarray(probabilities), axis=0)
    final_probs = final_probs / (final_probs.sum() + 1e-8)
    predicted_index = int(np.argmax(final_probs))
    return {
        "emotion": EMOTION_CLASSES[predicted_index],
        "confidence": round(float(final_probs[predicted_index]) * 100, 2),
        "probabilities": {
            label: round(float(final_probs[index]) * 100, 2)
            for index, label in enumerate(EMOTION_CLASSES)
        },
        "sampled_frames": sampled_frames,
        "usable_face_frames": len(probabilities),
    }
