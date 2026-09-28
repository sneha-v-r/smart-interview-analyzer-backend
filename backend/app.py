"""Development HTTP API for the Smart Interview Analyzer."""

from pathlib import Path
import tempfile

from flask import Flask, jsonify, request
from flask_cors import CORS

from fusion import analyze_interview

app = Flask(__name__)
CORS(app)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Smart Interview Analyzer Backend",
    }), 200


@app.post("/analyze")
def analyze():
    question = request.form.get("question")
    if question is None or not question.strip():
        return jsonify({"error": "A non-empty question field is required."}), 400
    video = request.files.get("video")
    if video is None:
        return jsonify({"error": "The video file field is required."}), 400
    if not video.filename or not video.filename.strip():
        return jsonify({"error": "The uploaded video filename must not be empty."}), 400
    extension = Path(video.filename).suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            "error": "Unsupported video extension.",
            "details": "Allowed: .mp4, .mov, .avi, .mkv, .webm",
        }), 400

    temp_video_path = None
    try:
        try:
            # Close the handle before FileStorage.save reopens it on Windows.
            with tempfile.NamedTemporaryFile(
                prefix="sia_upload_", suffix=extension, delete=False,
            ) as temporary_video:
                temp_video_path = temporary_video.name
            video.save(temp_video_path)
            result = analyze_interview(
                video_path=temp_video_path,
                interview_question=question,
            )
            return jsonify(result)
        finally:
            if temp_video_path is not None:
                Path(temp_video_path).unlink(missing_ok=True)
    except Exception as error:
        app.logger.exception("Analysis failed")
        return jsonify({"error": "Analysis failed", "details": str(error)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
