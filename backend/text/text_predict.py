from pathlib import Path

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    ROOT
    / "models"
    / "text"
    / "distilbert_model_v2"
)

MAX_LENGTH = 256
CLASSES = ("Average", "Good", "Poor")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading text model...")

if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f"V2 text model directory not found: {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
)

model.eval()

id2label = {
    index: label
    for index, label
    in enumerate(CLASSES)
}

if model.config.num_labels != 3:
    raise ValueError(f"V2 requires 3 output classes, got {model.config.num_labels}.")
if model.config.id2label != id2label:
    raise ValueError(f"Unexpected V2 label order: {model.config.id2label}")


print("✅ Text model loaded successfully!")
print("Classes:", list(CLASSES))


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_answer_quality(question, answer):

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer must be a non-empty string.")

    # Same format used during training
    text = (
        f"Question: {question.strip()} "
        f"Answer: {answer.strip()}"
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    with torch.no_grad():

        output = model(
            **encoded
        )

    probabilities = torch.softmax(
        output.logits,
        dim=-1
    )[0].cpu().numpy()

    predicted_id = int(
        np.argmax(probabilities)
    )

    predicted_label = id2label[
        predicted_id
    ]

    confidence = (
        float(
            probabilities[predicted_id]
        )
        * 100
    )

    class_probabilities = {}

    for index, probability in enumerate(
        probabilities
    ):

        class_probabilities[
            id2label[index]
        ] = round(
            float(probability) * 100,
            2
        )

    return {
        "label": predicted_label,
        "confidence": round(
            confidence,
            2
        ),
        "probabilities":
            class_probabilities
    }


# ============================================================
# TEST ANSWERS
# ============================================================

if __name__ == "__main__":

    tests = [

        {
            "name": "GOOD ANSWER",
            "question": "What is SQL?",
            "answer":
                "SQL is a language used to create, "
                "retrieve, update and manage data "
                "stored in relational databases."
        },

        {
            "name": "AVERAGE ANSWER",
            "question": "What is SQL?",
            "answer":
                "SQL is used to work with data "
                "in databases."
        },

        {
            "name": "POOR ANSWER",
            "question": "What is SQL?",
            "answer":
                "SQL is used to design web page colors."
        }

    ]

    print()
    print("===================================")
    print("       TEXT MODEL TEST")
    print("===================================")

    for test in tests:

        print()
        print("-----------------------------------")
        print(test["name"])

        print(
            "Question:",
            test["question"]
        )

        print(
            "Answer:",
            test["answer"]
        )

        result = predict_answer_quality(
            test["question"],
            test["answer"]
        )

        print()
        print(
            "Predicted quality:",
            result["label"]
        )

        print(
            "Confidence:",
            result["confidence"],
            "%"
        )

        print()
        print("Probabilities:")

        for label, probability in (
            result["probabilities"].items()
        ):

            print(
                f"  {label}: "
                f"{probability}%"
            )
