"""Prepare NRC Emotion Intensity Lexicon artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eat_bart.data.emotion_lexicon import load_nrc_lexicon


def main() -> None:
    """Prepare lexicon-derived files."""
    raise NotImplementedError("Lexicon preparation will be implemented once raw file format is confirmed.")


if __name__ == "__main__":
    main()
