"""
io.py
-----
Load/save probe checkpoints and encoder artifacts.

Pseudocode:
- def load_encoder_artifact(path, device):
    - load encoder state_dict/torchscript + encoder_meta.json
    - return encoder, meta
- def save_probe_checkpoint(path, head_state, metrics):
    - torch.save state_dict + metrics
- def load_probe_checkpoint(path, device):
    - load head state_dict + stored metrics
"""
