"""
mae.py
------
Masked autoencoder-style objective.

Contract:
def forward_loss(batch, encoder, cfg):
    Returns: loss (torch.Tensor), logs (Dict[str, float])

Pseudocode:
- unpack batch: inputs, maybe masks, metadata
- apply encoder to masked inputs
- compute reconstruction loss on masked regions
- compute metrics/log scalars for logging
- return loss, logs
"""
