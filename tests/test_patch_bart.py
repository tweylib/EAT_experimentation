from pathlib import Path

import torch
from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartAttention

from eat_bart.modeling.eat_bart_attention import EATBartAttention
from eat_bart.modeling.eat_bart_model import build_eat_bart_model_from_config
from eat_bart.modeling.patch_bart import patch_bart_self_attention


def test_project_contract_preserves_cross_attention() -> None:
    brief = Path("PROJECT_BRIEF.md").read_text(encoding="utf-8")
    rules = Path("CODING_RULES.md").read_text(encoding="utf-8")

    assert "Do NOT modify cross-attention" in brief
    assert "Do not modify cross-attention" in rules


def test_patch_bart_self_attention_preserves_cross_attention() -> None:
    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=99,
    )
    model = BartForConditionalGeneration(config)
    original_cross_attention = model.model.decoder.layers[0].encoder_attn

    patch_bart_self_attention(model)

    assert isinstance(model.model.encoder.layers[0].self_attn, EATBartAttention)
    assert isinstance(model.model.decoder.layers[0].self_attn, EATBartAttention)
    assert model.model.decoder.layers[0].encoder_attn is original_cross_attention
    assert isinstance(model.model.decoder.layers[0].encoder_attn, BartAttention)


def test_patched_bart_forward_accepts_encoder_and_decoder_emotion_features() -> None:
    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=99,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
    )
    model = BartForConditionalGeneration(config)
    patch_bart_self_attention(model)

    input_ids = torch.tensor([[0, 5, 6, 2]])
    decoder_input_ids = torch.tensor([[2, 7, 8]])
    encoder_emotion_features = torch.zeros(1, 4, 8)
    decoder_emotion_features = torch.zeros(1, 3, 8)

    output = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        encoder_emotion_features=encoder_emotion_features,
        decoder_emotion_features=decoder_emotion_features,
    )

    assert tuple(output.logits.shape) == (1, 3, 99)


def test_build_eat_bart_model_from_config_patches_self_attention() -> None:
    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=99,
    )

    model = build_eat_bart_model_from_config(config)

    assert isinstance(model.model.encoder.layers[0].self_attn, EATBartAttention)
    assert isinstance(model.model.decoder.layers[0].self_attn, EATBartAttention)
