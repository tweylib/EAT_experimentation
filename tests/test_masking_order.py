from pathlib import Path


def test_attention_contract_mentions_masking_last() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Padding and causal masks are applied last" in readme
    assert "masked_fill" in readme
