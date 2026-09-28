import os
import json
import subprocess
import tempfile

import imageio_ffmpeg
import librosa
import numpy as np
import tensorflow as tf


print("\n===================================")
print(" AUDIO PREDICT V7 - GLOBAL NORMALIZATION ONLY")
print("===================================")
print("Running script:", os.path.abspath(__file__))


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    ROOT,
    "models",
    "audio",
    "best_curve_model.keras"
)

LABEL_PATH = os.path.join(
    ROOT,
    "models",
    "audio",
    "label_mapping_v7.json"
)

NORMALIZATION_PATH = os.path.join(
    ROOT,
    "models",
    "audio",
    "audio_normalization_stats.npz"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 22050
DURATION = 3
TARGET_LENGTH = SAMPLE_RATE * DURATION

N_MFCC = 40
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 126
TARGET_FEATURES = 184
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

for path, name in [
    (MODEL_PATH, "Audio model"),
    (LABEL_PATH, "Label mapping"),
    (NORMALIZATION_PATH, "Normalization statistics"),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading audio model...")

audio_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Audio model loaded successfully!")
print("Model input shape:", audio_model.input_shape)
print("Model output shape:", audio_model.output_shape)

if tuple(audio_model.input_shape) != (None, TARGET_FRAMES, TARGET_FEATURES):
    raise ValueError(f"V7 requires input (None, 126, 184), got {audio_model.input_shape}")
if tuple(audio_model.output_shape) != (None, 6):
    raise ValueError(f"V7 requires output (None, 6), got {audio_model.output_shape}")


# ============================================================
# LOAD LABELS
# ============================================================

with open(
    LABEL_PATH,
    "r",
    encoding="utf-8"
) as file:
    label_mapping = json.load(file)

print("Label mapping:", label_mapping)

if label_mapping != {label: index for index, label in enumerate(EMOTION_CLASSES)}:
    raise ValueError("V7 label mapping must be angry, disgust, fear, happy, neutral, sad (IDs 0-5).")


# ============================================================
# LOAD TRAINING NORMALIZATION STATS
# ============================================================

normalization_data = np.load(
    NORMALIZATION_PATH
)

global_feature_mean = normalization_data[
    "mean"
].astype(np.float32)

global_feature_std = normalization_data[
    "std"
].astype(np.float32)
normalization_data.close()

if global_feature_mean.size != TARGET_FEATURES or global_feature_std.size != TARGET_FEATURES:
    raise ValueError("V7 normalization must contain 184 mean and 184 std values.")
if not np.isfinite(global_feature_mean).all() or not np.isfinite(global_feature_std).all():
    raise ValueError("V7 normalization statistics must be finite.")
global_feature_mean = global_feature_mean.reshape(1, 1, TARGET_FEATURES)
global_feature_std = global_feature_std.reshape(1, 1, TARGET_FEATURES)

global_feature_std = np.where(
    global_feature_std < 1e-8,
    1.0,
    global_feature_std
).astype(np.float32)

print("Normalization statistics loaded successfully!")
print("Normalization mean shape:", global_feature_mean.shape)
print("Normalization std shape :", global_feature_std.shape)


# ============================================================
# LABEL FUNCTION
# ============================================================

def get_label(index):
    index_string = str(index)

    if index_string in label_mapping:
        return str(
            label_mapping[index_string]
        ).lower()

    for label, value in label_mapping.items():
        if str(value) == index_string:
            return str(label).lower()

    return f"class_{index}"


# ============================================================
# EXTRACT AUDIO FROM VIDEO
# ============================================================

def extract_audio_from_video(video_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    wav_path = temp_file.name
    temp_file.close()

    command = [
        ffmpeg_path,
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        wav_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        if os.path.exists(wav_path):
            os.remove(wav_path)

        raise RuntimeError(
            "Audio extraction failed.\n"
            + result.stderr[-1000:]
        )

    return wav_path


# ============================================================
# LOAD FULL RECORDING (STANDARDIZE EACH SEGMENT LATER)
# ============================================================

def load_audio(audio_path):
    temp_wav = None

    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Input file not found:\n{audio_path}"
            )

        extension = os.path.splitext(
            audio_path
        )[1].lower()

        if extension in (
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm"
        ):
            print("Extracting audio from video...")
            temp_wav = extract_audio_from_video(
                audio_path
            )
            path_to_load = temp_wav
        else:
            path_to_load = audio_path

        signal, _ = librosa.load(
            path_to_load,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(signal) == 0:
            raise ValueError("Audio is empty")

        return signal.astype(np.float32)

    finally:
        if (
            temp_wav is not None
            and os.path.exists(temp_wav)
        ):
            try:
                os.remove(temp_wav)
            except OSError:
                pass


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def standardize_waveform(signal):
    """Apply V7 waveform preprocessing independently to one segment."""
    signal = np.asarray(signal, dtype=np.float32)
    signal, _ = librosa.effects.trim(signal, top_db=30)
    if len(signal) == 0:
        signal = np.zeros(TARGET_LENGTH, dtype=np.float32)
    else:
        signal = librosa.util.normalize(signal)
    if len(signal) < TARGET_LENGTH:
        signal = np.pad(signal, (0, TARGET_LENGTH - len(signal)), mode="constant")
    else:
        signal = signal[:TARGET_LENGTH]
    return signal.astype(np.float32)


def extract_features(signal):
    """
    IMPORTANT:
    Do NOT perform per-file z-score normalization here.

    The V7 model is normalized later using the saved
    training-set mean/std from audio_normalization_stats.npz.
    """

    signal = standardize_waveform(signal)

    # 40 MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=False
    )

    # 40 Delta
    delta = librosa.feature.delta(
        mfcc
    )

    # 40 Delta-Delta
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # 64 Log-Mel
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0,
        center=False
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # 184 raw features
    features = np.vstack([
        mfcc,
        delta,
        delta2,
        log_mel
    ])

    # NO LOCAL NORMALIZATION HERE

    return features.T.astype(
        np.float32
    )


# ============================================================
# PREPARE FOR MODEL
# ============================================================

def prepare_for_model(features):
    expected_time = int(
        audio_model.input_shape[1]
    )

    expected_features = int(
        audio_model.input_shape[2]
    )

    print(
        "Extracted RAW feature shape:",
        features.shape
    )

    if features.shape[1] != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features, "
            f"but extracted {features.shape[1]}."
        )

    if features.shape[0] > expected_time:
        features = features[
            :expected_time,
            :
        ]

    elif features.shape[0] < expected_time:
        padding = (
            expected_time
            -
            features.shape[0]
        )

        features = np.pad(
            features,
            (
                (0, padding),
                (0, 0)
            ),
            mode="constant"
        )

    features = np.expand_dims(
        features,
        axis=0
    ).astype(np.float32)

    if features.shape != (1, TARGET_FRAMES, TARGET_FEATURES):
        raise ValueError(f"V7 requires prepared shape (1, 126, 184), got {features.shape}")

    print(
        "Raw model input prepared:",
        features.shape
    )

    return features


# ============================================================
# GLOBAL TRAINING NORMALIZATION
# ============================================================

def apply_training_normalization(features):
    normalized = (
        (
            features
            -
            global_feature_mean
        )
        /
        global_feature_std
    ).astype(np.float32)

    print(
        "Normalized model input:",
        normalized.shape
    )

    print(
        "Normalized input mean:",
        round(
            float(np.mean(normalized)),
            4
        )
    )

    print(
        "Normalized input std :",
        round(
            float(np.std(normalized)),
            4
        )
    )

    return normalized


# ============================================================
# PREDICT
# ============================================================

def predict_audio_emotion(audio_path):
    signal = load_audio(
        audio_path
    )

    segment_probabilities = []
    for start in range(0, len(signal), TARGET_LENGTH):
        segment = signal[start:start + TARGET_LENGTH]
        # Match the notebook: skip tails shorter than half a second.
        if len(segment) < SAMPLE_RATE * 0.5:
            continue
        features = prepare_for_model(extract_features(segment))
        features = apply_training_normalization(features)
        segment_probabilities.append(
            np.asarray(audio_model.predict(features, verbose=0)[0], dtype=np.float32)
        )

    # Still analyze a recording whose entire duration is below half a second.
    if not segment_probabilities:
        features = prepare_for_model(extract_features(signal))
        features = apply_training_normalization(features)
        segment_probabilities.append(
            np.asarray(audio_model.predict(features, verbose=0)[0], dtype=np.float32)
        )

    predictions = np.mean(np.asarray(segment_probabilities), axis=0)
    predictions = predictions / (predictions.sum() + 1e-8)

    predicted_index = int(
        np.argmax(predictions)
    )

    emotion = get_label(
        predicted_index
    )

    confidence = float(
        predictions[predicted_index]
    )

    probabilities = {}

    for index, probability in enumerate(
        predictions
    ):
        probabilities[
            get_label(index)
        ] = round(
            float(probability) * 100,
            2
        )

    return {
        "emotion": emotion,
        "confidence": round(
            confidence * 100,
            2
        ),
        "probabilities": probabilities,
        "segments": len(segment_probabilities)
    }


# ============================================================
# TEST CREMA-D FILES
# ============================================================

if __name__ == "__main__":
    uploads_folder = os.path.join(
        ROOT,
        "uploads"
    )

    emotion_codes = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad"
    }

    if not os.path.exists(
        uploads_folder
    ):
        print(
            "Uploads folder not found:",
            uploads_folder
        )
        raise SystemExit

    files = [
        filename
        for filename
        in os.listdir(uploads_folder)
        if filename.lower().endswith(
            (
                ".wav",
                ".mp3",
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".webm"
            )
        )
    ]

    files.sort()

    print()
    print("===================================")
    print("       AUDIO MODEL TEST")
    print("===================================")

    if not files:
        print("No audio/video files found.")
        raise SystemExit

    total = 0
    correct = 0

    for filename in files:
        file_path = os.path.join(
            uploads_folder,
            filename
        )

        print()
        print("-----------------------------------")
        print("File:", filename)

        parts = filename.split("_")
        expected = "Unknown"

        if len(parts) >= 3:
            emotion_code = parts[2].upper()

            expected = emotion_codes.get(
                emotion_code,
                "Unknown"
            )

        try:
            result = predict_audio_emotion(
                file_path
            )

            print()
            print("Expected emotion :", expected)
            print("Predicted emotion:", result["emotion"])
            print("Confidence       :", result["confidence"], "%")

            print()
            print("Probabilities:")

            for emotion, probability in (
                result["probabilities"].items()
            ):
                print(
                    f"  {emotion}: {probability}%"
                )

            if expected != "Unknown":
                total += 1

                if (
                    result["emotion"].lower()
                    ==
                    expected.lower()
                ):
                    correct += 1
                    print()
                    print("✅ Correct prediction")
                else:
                    print()
                    print("❌ Wrong prediction")

        except Exception as error:
            print()
            print(
                "❌ Error:",
                repr(error)
            )

    if total > 0:
        accuracy = (
            correct / total
        ) * 100

        print()
        print("===================================")
        print("          TEST SUMMARY")
        print("===================================")
        print("Files tested :", total)
        print("Correct      :", correct)
        print("Wrong        :", total - correct)
        print(
            "Test accuracy:",
            round(accuracy, 2),
            "%"
        )
