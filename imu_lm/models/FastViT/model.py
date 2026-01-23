"""
FastViT encoder placeholder (expects spectrogram [B, C, F, TT]).

Pseudocode:
- class FastViTEncoder(nn.Module):
    - __init__(cfg): define fastvit blocks/patch embedding; produce embedding_dim
    - forward(x): returns embeddings
"""
