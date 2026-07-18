Project: Emotion-Aware Transformer using BART.

Goal:
Implement and evaluate EAT-BART for mental-health response generation.

Main model:
facebook/bart-base

Architecture:
Modify BART encoder self-attention and decoder self-attention.
Do NOT modify cross-attention.

Emotion source:
NRC Emotion Intensity Lexicon.
Each token receives an 8-dimensional emotion vector:
anger, anticipation, disgust, fear, joy, sadness, surprise, trust.

EAT attention:
Compute raw attention scores A = QK^T / sqrt(d_k).
Compute emotion interaction matrix S from emotion features.
Combine A and S before masking.

Main formula:
A_eat = A + alpha_h * S_h

Ablation formula:
A_eat = A * (I + alpha_h * S_h)

Masking:
Apply padding/causal masks last using masked_fill.
Do not do arithmetic with already-masked attention scores.

Trainable parameters:
W1_s: [num_heads, 8, d_s]
W2_s: [num_heads, 8, d_s]
alpha: [num_heads], initialized to 0.05
d_s = 32

Constraints:
Code must run on Kaggle GPU.
Use small batch sizes, fp16, gradient accumulation.