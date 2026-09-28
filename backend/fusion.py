"""Weighted late fusion using the frozen local prediction backends."""

import math
from pathlib import Path

from text.fluency_analyzer import analyze_fluency

TEXT_WEIGHT = 0.40
AUDIO_WEIGHT = 0.30
VIDEO_WEIGHT = 0.30

TEXT_SCORE_VALUES = {"Poor": 0.0, "Average": 50.0, "Good": 100.0}
EMOTION_SCORE_VALUES = {
    "angry": 20.0,
    "disgust": 20.0,
    "fear": 30.0,
    "happy": 90.0,
    "neutral": 75.0,
    "sad": 35.0,
}


def _expected_score(probabilities, score_values, branch):
    """Convert percentage probabilities to fractions without rounding."""
    if set(probabilities) != set(score_values):
        raise ValueError(f"{branch} probability labels must be {list(score_values)}.")
    percentages = {label: float(probabilities[label]) for label in score_values}
    if any(not math.isfinite(value) or not 0 <= value <= 100
           for value in percentages.values()):
        raise ValueError(f"{branch} probabilities must be finite percentages from 0 to 100.")
    # Frozen predictors round each percentage to two decimal places.
    if not math.isclose(sum(percentages.values()), 100.0, rel_tol=0, abs_tol=0.05):
        raise ValueError(f"{branch} probabilities must sum approximately to 100%.")
    return sum(percentages[label] / 100.0 * value for label, value in score_values.items())


def score_to_grade(score):
    if score < 40:
        return "Poor"
    if score < 70:
        return "Average"
    return "Good"


def analyze_interview(video_path, interview_question):
    """Run every required branch; propagate failures without retries or fallbacks."""
    if video_path is None:
        raise ValueError("Video path must not be None.")
    try:
        path = Path(video_path)
    except TypeError as error:
        raise ValueError("Video path must be a string or path-like value.") from error
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not isinstance(interview_question, str) or not interview_question.strip():
        raise ValueError("Interview question must be a non-empty string.")

    # Some frozen modules load models at import time. Import only after validation
    # and immediately before each branch to preserve the requested execution order.
    from video.video_predict import predict_video_emotion
    video_result = predict_video_emotion(str(path))

    from audio.audio_predict import predict_audio_emotion
    audio_result = predict_audio_emotion(str(path))

    from speech.whisper_predict import transcribe_video
    speech_result = transcribe_video(str(path))
    transcript = speech_result["text"]
    fluency_result = analyze_fluency(transcript)

    from text.text_predict import predict_answer_quality
    text_result = predict_answer_quality(question=interview_question, answer=transcript)

    text_score = _expected_score(text_result["probabilities"], TEXT_SCORE_VALUES, "Text")
    audio_score = _expected_score(audio_result["probabilities"], EMOTION_SCORE_VALUES, "Audio")
    video_score = _expected_score(video_result["probabilities"], EMOTION_SCORE_VALUES, "Video")
    final_score = (
        TEXT_WEIGHT * text_score
        + AUDIO_WEIGHT * audio_score
        + VIDEO_WEIGHT * video_score
    )

    return {
        "transcript": transcript,
        "fluency": fluency_result,
        "text": {
            "label": text_result["label"],
            "probabilities": text_result["probabilities"],
            "score": text_score,
        },
        "audio": {
            "emotion": audio_result["emotion"],
            "confidence": audio_result["confidence"],
            "probabilities": audio_result["probabilities"],
            "score": audio_score,
            "segments": audio_result.get("segments"),
        },
        "video": {
            "emotion": video_result["emotion"],
            "probabilities": video_result["probabilities"],
            "score": video_score,
            "frames": video_result.get("usable_face_frames"),
            "sampled_frames": video_result.get("sampled_frames"),
            "face_frames": video_result.get("usable_face_frames"),
        },
        "final": {"score": final_score, "grade": score_to_grade(final_score)},
    }
