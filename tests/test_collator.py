import torch

from eat_bart.data.collator import shift_emotion_features_right


def test_shift_emotion_features_right_aligns_with_decoder_inputs() -> None:
    target_emotion_features = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    decoder_input_ids = torch.tensor([[2, 10, 1]])

    shifted = shift_emotion_features_right(
        target_emotion_features=target_emotion_features,
        decoder_input_ids=decoder_input_ids,
        pad_token_id=1,
    )

    assert torch.equal(shifted[0, 0], torch.zeros(8))
    assert torch.equal(shifted[0, 1], target_emotion_features[0, 0])
    assert torch.equal(shifted[0, 2], torch.zeros(8))
