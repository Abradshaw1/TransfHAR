"""MAE objective - forward loss only. Model loading handled in model.py."""

from __future__ import annotations


def forward_loss(batch, model, cfg):
    """Compute MAE reconstruction loss.
    
    Args:
        batch: (x, y) where x is [B,3,F,TT] image-like spectrogram.
        model: ViTEncoder with mae_model already attached.
    Returns:
        loss, logs dict
    """
    x, _ = batch
    mask_ratio = model.mae_model.config.mask_ratio

    x_img = model._prepare(x)
    outputs = model.mae_model(pixel_values=x_img, mask_ratio=mask_ratio)
    loss = outputs.loss

    return loss, {"loss": loss.detach().item(), "recon_mse": loss.detach().item(), "mask_ratio": mask_ratio}
