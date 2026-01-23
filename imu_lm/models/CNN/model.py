"""
CNN encoder placeholder (expects raw [B, C, T]).

Pseudocode:
- class CNNEncoder(nn.Module):
    - __init__(cfg): define conv stack + pooling; produce embedding_dim
    - forward(x): returns embeddings
"""
