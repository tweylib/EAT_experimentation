import torch

from eat_bart.modeling.eat_attention import EATAttentionConfig, EmotionInteraction


def test_emotion_interaction_parameter_shapes() -> None:
    module = EmotionInteraction(EATAttentionConfig(num_heads=12))

    assert tuple(module.w1_s.shape) == (12, 8, 32)
    assert tuple(module.w2_s.shape) == (12, 8, 32)
    assert tuple(module.alpha.shape) == (12,)


def test_emotion_interaction_output_shape() -> None:
    module = EmotionInteraction(EATAttentionConfig(num_heads=2, emotion_hidden_dim=4))
    emotion_features = torch.randn(3, 5, 8)

    scores = module(emotion_features)

    assert tuple(scores.shape) == (3, 2, 5, 5)


def test_zero_emotion_features_produce_zero_scores() -> None:
    module = EmotionInteraction(EATAttentionConfig(num_heads=2, emotion_hidden_dim=4))
    emotion_features = torch.zeros(3, 5, 8)

    scores = module(emotion_features)

    assert torch.equal(scores, torch.zeros(3, 2, 5, 5))


def test_additive_formula_combines_before_masking() -> None:
    module = EmotionInteraction(EATAttentionConfig(num_heads=2, alpha_init=0.5))
    attention_scores = torch.ones(1, 2, 3, 3)
    emotion_scores = torch.full((1, 2, 3, 3), 2.0)

    combined = module.combine_with_attention_scores(attention_scores, emotion_scores)

    assert torch.equal(combined, torch.full((1, 2, 3, 3), 2.0))


def test_multiplicative_ablation_formula() -> None:
    module = EmotionInteraction(
        EATAttentionConfig(num_heads=1, alpha_init=0.5, formula="multiplicative")
    )
    attention_scores = torch.ones(1, 1, 3, 3)
    emotion_scores = torch.full((1, 1, 3, 3), 2.0)

    combined = module.combine_with_attention_scores(attention_scores, emotion_scores)

    expected = torch.tensor([[[[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]]])
    assert torch.equal(combined, expected)
