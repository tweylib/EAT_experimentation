"""Command-line evaluation entry point."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eat_bart.training.evaluate import evaluate


if __name__ == "__main__":
    evaluate()
