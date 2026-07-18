"""Model factory for EAT-BART."""

from __future__ import annotations

from transformers import BartConfig, BartForConditionalGeneration

from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.patch_bart import patch_bart_self_attention

DEFAULT_MODEL_NAME = "facebook/bart-base"


def build_eat_bart_model_from_config(
    config: BartConfig,
    eat_config: EATAttentionConfig | None = None,
) -> BartForConditionalGeneration:
    """Build an EAT-BART model from an in-memory BART config."""
    model = BartForConditionalGeneration(config)
    return patch_bart_self_attention(model, eat_config=eat_config)


def load_eat_bart_model(
    model_name: str = DEFAULT_MODEL_NAME,
    eat_config: EATAttentionConfig | None = None,
    local_files_only: bool = False,
) -> BartForConditionalGeneration:
    """Load pretrained BART and apply EAT self-attention patches."""
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError("Changing the base model from facebook/bart-base requires explicit approval.")

    model = BartForConditionalGeneration.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    return patch_bart_self_attention(model, eat_config=eat_config)
