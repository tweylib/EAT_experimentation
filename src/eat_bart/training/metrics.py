"""Metrics for response generation experiments."""

from __future__ import annotations


def compute_generation_metrics() -> dict[str, float]:
    """Compute generation metrics for predictions."""
    raise NotImplementedError("Metrics will be selected when the dataset format is fixed.")
