"""Auxiliary filler-word analysis, independent of the fusion score."""

import re


FILLER_PATTERNS = (
    r"\bum+\b",
    r"\buh+\b",
    r"\bah+\b",
    r"\ber+\b",
    r"\bhmm+\b",
    r"\byou know\b",
    r"\bi mean\b",
    r"\bbasically\b",
    r"\bactually\b",
)


def analyze_fluency(transcript):
    """Count filler occurrences and calculate an auxiliary fluency score."""
    transcript = transcript.lower()
    filler_count = sum(len(re.findall(pattern, transcript)) for pattern in FILLER_PATTERNS)
    total_words = len(re.findall(r"\b\w+\b", transcript))
    filler_rate = filler_count / total_words * 100 if total_words else 0.0

    if filler_rate <= 2:
        fluency_label, fluency_score = "Excellent", 100
    elif filler_rate <= 5:
        fluency_label, fluency_score = "Good", 80
    elif filler_rate <= 10:
        fluency_label, fluency_score = "Average", 60
    else:
        fluency_label, fluency_score = "Needs Improvement", 40

    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "filler_rate": round(filler_rate, 2),
        "fluency_label": fluency_label,
        "fluency_score": fluency_score,
    }
