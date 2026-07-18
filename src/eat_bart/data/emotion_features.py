"""Build token-level emotion features for BART inputs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

import torch

from eat_bart.data.emotion_lexicon import LexiconEntry, get_emotion_vector

EmotionVector = tuple[float, float, float, float, float, float, float, float]
SubwordStrategy = Literal["single", "replicate"]
RepresentativeToken = Literal["first", "last"]

ZERO_EMOTION_VECTOR: EmotionVector = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]")


def split_text_for_emotion(text: str) -> list[str]:
    """Split raw text into words used for lexicon lookup."""
    return WORD_PATTERN.findall(text.lower())


def tokenize_texts_with_emotion_features(
    texts: Sequence[str],
    tokenizer: Any,
    lexicon: dict[str, LexiconEntry],
    max_length: int | None = None,
    padding: bool | str = True,
    truncation: bool = True,
    strategy: SubwordStrategy = "single",
    representative_token: RepresentativeToken = "last",
    dtype: torch.dtype = torch.float32,
) -> tuple[Any, torch.Tensor]:
    """Tokenize text and build BART-token-aligned emotion features.

    texts shape: [batch_size]
    encoded input_ids shape: [batch_size, seq_len]
    return emotion_features shape: [batch_size, seq_len, 8]
    """
    batch_words = [split_text_for_emotion(text) for text in texts]
    encoded = tokenizer(
        batch_words,
        is_split_into_words=True,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
    )

    try:
        batch_word_ids = [encoded.word_ids(batch_index) for batch_index in range(len(batch_words))]
    except ValueError as error:
        raise ValueError("Emotion feature alignment requires a Hugging Face fast tokenizer.") from error

    emotion_features = build_emotion_features(
        batch_words=batch_words,
        batch_word_ids=batch_word_ids,
        lexicon=lexicon,
        strategy=strategy,
        representative_token=representative_token,
        dtype=dtype,
    )

    return encoded, emotion_features


def build_emotion_features(
    batch_words: Sequence[Sequence[str]],
    batch_word_ids: Sequence[Sequence[int | None]],
    lexicon: dict[str, LexiconEntry],
    strategy: SubwordStrategy = "single",
    representative_token: RepresentativeToken = "last",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return BART-token-aligned emotion features.

    batch_words shape: [batch_size, num_words]
    batch_word_ids shape: [batch_size, seq_len]
    return shape: [batch_size, seq_len, 8]
    """
    if len(batch_words) != len(batch_word_ids):
        raise ValueError("batch_words and batch_word_ids must have the same batch size.")

    feature_rows = [
        build_single_example_emotion_features(
            words=words,
            word_ids=word_ids,
            lexicon=lexicon,
            strategy=strategy,
            representative_token=representative_token,
            dtype=dtype,
        )
        for words, word_ids in zip(batch_words, batch_word_ids, strict=True)
    ]

    return torch.stack(feature_rows, dim=0)


def build_single_example_emotion_features(
    words: Sequence[str],
    word_ids: Sequence[int | None],
    lexicon: dict[str, LexiconEntry],
    strategy: SubwordStrategy = "single",
    representative_token: RepresentativeToken = "last",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return emotion features for one tokenized example.

    words shape: [num_words]
    word_ids shape: [seq_len]
    return shape: [seq_len, 8]
    """
    if strategy not in ("single", "replicate"):
        raise ValueError(f"Unsupported subword emotion strategy: {strategy}")
    if representative_token not in ("first", "last"):
        raise ValueError(f"Unsupported representative token: {representative_token}")

    token_vectors = [ZERO_EMOTION_VECTOR for _ in word_ids]

    for word_index, word in enumerate(words):
        vector = get_emotion_vector(word, lexicon)
        if vector == ZERO_EMOTION_VECTOR:
            continue

        token_positions = [
            token_index for token_index, token_word_id in enumerate(word_ids) if token_word_id == word_index
        ]
        if not token_positions:
            continue

        if strategy == "replicate":
            for token_index in token_positions:
                token_vectors[token_index] = vector
        else:
            token_index = token_positions[0] if representative_token == "first" else token_positions[-1]
            token_vectors[token_index] = vector

    return torch.tensor(token_vectors, dtype=dtype)
