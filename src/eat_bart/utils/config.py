"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, including optional parent config inheritance."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    parent_path = config.pop("inherits", None)
    if parent_path is None:
        return config

    resolved_parent_path = Path(parent_path)
    if not resolved_parent_path.is_absolute():
        resolved_parent_path = config_path.parent.parent / resolved_parent_path

    parent_config = load_yaml_config(resolved_parent_path)
    return _deep_merge(parent_config, config)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested config dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged
