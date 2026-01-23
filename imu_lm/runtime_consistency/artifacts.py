"""
artifacts.py
------------
Save/load encoder artifacts and metadata.

Pseudocode:
- def save_encoder(encoder, meta, path):
    - torch.save(state_dict) or torch.jit
    - write encoder_meta.json with fields: embedding_dim, encoding, objective, backbone, input_spec
- def load_encoder(path, device):
    - load state_dict or torchscript
    - return encoder module on device
- helpers to resolve artifact paths under runs/<run>/artifacts/
"""
