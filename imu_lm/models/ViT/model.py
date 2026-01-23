"""
ViT encoder placeholder (expects spectrogram [B, C, F, TT]).

Pseudocode:
- class ViTEncoder(nn.Module):
    - __init__(cfg): define patch embed over freq/time, transformer blocks; produce embedding_dim
    - forward(x): returns embeddings
"""
