"""Loss factory and implementations for VAE training.

Provides a SimpleVAELoss (L1 or MSE reconstruction + KLD).
Use `get_vae_loss(config)` to instantiate the appropriate loss module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVAELoss(nn.Module):
    """Simple VAE loss using L1 or MSE reconstruction + KLD.

    Args:
        mode: 'l1' or 'mse'
        beta: weight for KLD term
    """

    def __init__(self, mode='l1', beta=1.0):
        super().__init__()
        self.mode = mode
        self.beta = beta

    def forward(self, recon_x, x, mu, logvar):
        if self.mode == 'mse':
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        elif self.mode == 'l1':
            recon_loss = F.l1_loss(recon_x, x, reduction='sum')
        else:
            raise ValueError(f"Unknown recon loss mode: {self.mode}")

        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + (self.beta * kld_loss)
        return total, recon_loss, kld_loss


def get_vae_loss(loss_config: dict):
    """Factory: instantiate a loss module from config.

    Expected keys in loss_config:
      - name: 'l1' | 'mse'
      - beta_kld: float
    """
    name = loss_config.get('name', 'l1').lower()
    beta = loss_config.get('beta_kld', loss_config.get('beta', 0.001))
    if name in ('l1', 'mse'):
        return SimpleVAELoss(mode=name, beta=beta)
    raise ValueError(f"Unknown loss name: {name}. Use 'l1' or 'mse'.")