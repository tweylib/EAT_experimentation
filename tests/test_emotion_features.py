from pathlib import Path

import torch

from eat_bart.data.emotion_features import build_emotion_features, split_text_for_emotion
from eat_bart.data.emotion_lexicon import EMOTION_LABELS, get_emotion_vector, load_nrc_lexicon


def test_emotion_labels_match_project_brief() -> None:
    assert EMOTION_LABELS == (
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    )


def test_load_nrc_lexicon_csv(tmp_path: Path) -> None:
    lexicon_file = tmp_path / "nrc_lexicon.csv"
    lexicon_file.write_text(
        "word,anger,anticipation,disgust,fear,joy,sadness,surprise,trust\n"
        "abandon,0.0,0.0,0.0,0.531,0.0,0.703,0.0,0.0\n",
        encoding="utf-8",
    )

    lexicon = load_nrc_lexicon(lexicon_file)

    assert lexicon["abandon"].scores == (0.0, 0.0, 0.0, 0.531, 0.0, 0.703, 0.0, 0.0)


def test_unknown_token_returns_zero_vector() -> None:
    assert get_emotion_vector("missing", {}) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_split_text_for_emotion_lowercases_and_keeps_punctuation() -> None:
    assert split_text_for_emotion("I can't cope.") == ["i", "can't", "cope", "."]


def test_single_strategy_assigns_vector_to_last_subword(tmp_path: Path) -> None:
    lexicon_file = tmp_path / "nrc_lexicon.csv"
    lexicon_file.write_text(
        "word,anger,anticipation,disgust,fear,joy,sadness,surprise,trust\n"
        "unbelievable,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8\n",
        encoding="utf-8",
    )
    lexicon = load_nrc_lexicon(lexicon_file)

    features = build_emotion_features(
        batch_words=[["unbelievable"]],
        batch_word_ids=[[None, 0, 0, None]],
        lexicon=lexicon,
        strategy="single",
        representative_token="last",
    )

    assert tuple(features.shape) == (1, 4, 8)
    assert torch.equal(features[0, 0], torch.zeros(8))
    assert torch.equal(features[0, 1], torch.zeros(8))
    assert torch.equal(features[0, 2], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    assert torch.equal(features[0, 3], torch.zeros(8))


def test_replicate_strategy_assigns_vector_to_all_subwords(tmp_path: Path) -> None:
    lexicon_file = tmp_path / "nrc_lexicon.csv"
    lexicon_file.write_text(
        "word,anger,anticipation,disgust,fear,joy,sadness,surprise,trust\n"
        "unbelievable,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8\n",
        encoding="utf-8",
    )
    lexicon = load_nrc_lexicon(lexicon_file)

    features = build_emotion_features(
        batch_words=[["unbelievable"]],
        batch_word_ids=[[None, 0, 0, None]],
        lexicon=lexicon,
        strategy="replicate",
    )

    expected = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert torch.equal(features[0, 1], expected)
    assert torch.equal(features[0, 2], expected)
