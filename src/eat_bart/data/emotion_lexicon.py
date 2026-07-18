"""NRC Emotion Intensity Lexicon loading utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

EMOTION_LABELS: tuple[str, ...] = (
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
)


@dataclass(frozen=True)
class LexiconEntry:
    """Emotion intensities for one lexical item."""

    token: str
    scores: tuple[float, float, float, float, float, float, float, float]


def load_nrc_lexicon(path: str | Path) -> dict[str, LexiconEntry]:
    """Load NRC scores into 8-dimensional emotion vectors.

    Output shape per token: [8], ordered by EMOTION_LABELS.
    """
    lexicon_path = Path(path)
    entries: dict[str, LexiconEntry] = {}

    with lexicon_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        expected_columns = ("word", *EMOTION_LABELS)
        missing_columns = [column for column in expected_columns if column not in reader.fieldnames]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"NRC lexicon is missing required columns: {missing}")

        for row in reader:
            token = row["word"].strip().lower()
            if not token:
                continue

            scores = tuple(float(row[label]) for label in EMOTION_LABELS)
            entries[token] = LexiconEntry(token=token, scores=scores)

    return entries


def get_emotion_vector(
    token: str,
    lexicon: dict[str, LexiconEntry],
) -> tuple[float, float, float, float, float, float, float, float]:
    """Return one token vector from the lexicon.

    Output shape: [8], ordered by EMOTION_LABELS.
    """
    entry = lexicon.get(token.strip().lower())
    if entry is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return entry.scores
