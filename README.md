# EAT-BART

Emotion-Aware Transformer using `facebook/bart-base` for mental-health response generation.

The implementation will inject NRC Emotion Intensity Lexicon features into BART encoder self-attention and decoder self-attention only. Cross-attention must remain unchanged.

## Project Shape

- `configs/`: local and Kaggle experiment settings.
- `src/eat_bart/data/`: datasets, token emotion features, and NRC lexicon loading.
- `src/eat_bart/modeling/`: EAT attention modules and BART patching.
- `src/eat_bart/training/`: training, evaluation, and metrics helpers.
- `src/eat_bart/utils/`: config, seed, and device utilities.
- `scripts/`: command-line entry points.
- `tests/`: focused tests for shapes, masking, lexicon features, and BART patching.

## Attention Contract

Base attention scores:

```text
A = QK^T / sqrt(d_k)
```

Main EAT formula:

```text
A_eat = A + alpha_h * S_h
```

Ablation formula:

```text
A_eat = A * (I + alpha_h * S_h)
```

Padding and causal masks are applied last with `masked_fill`.
