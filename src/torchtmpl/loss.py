"""Loss factory and implementations for VAE training.

Provides a SimpleVAELoss (L1 or MSE) and a VGGPerceptualLoss. Use
`get_vae_loss(config)` to instantiate the appropriate loss module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using pretrained VGG16 features + KLD.

    Notes:
    - Expects inputs in [0,1]. Converts to 3 channels if needed and applies
      ImageNet normalization before passing through VGG.
    - KLD is normalized by batch size to balance magnitudes.
    """

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        # Load VGG16 features (support newest and older torchvision APIs)
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        except Exception:
            vgg = models.vgg16(pretrained=True).features

        # Split into blocks to extract intermediate features
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
            vgg[16:23] # relu4_3
        ])

        for param in self.blocks.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, recon_x, x, mu, logvar):
        # Ensure 3 channels for VGG
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            recon_x = recon_x.repeat(1, 3, 1, 1)

        x_norm = (x - self.mean) / self.std
        recon_norm = (recon_x - self.mean) / self.std

        loss_feat = 0.0
        x_feat = x_norm
        recon_feat = recon_norm
        for block in self.blocks:
            x_feat = block(x_feat)
            recon_feat = block(recon_feat)
            loss_feat = loss_feat + F.l1_loss(recon_feat, x_feat, reduction='mean')

        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / x.shape[0]

        total = loss_feat + (self.beta * kld_loss)
        return total, loss_feat, kld_loss


def get_vae_loss(loss_config: dict):
    """Factory: instantiate a loss module from config.

    Expected keys in loss_config:
      - name: 'l1' | 'mse' | 'perceptual'
      - beta_kld: float
    """
    name = loss_config.get('name', 'l1').lower()
    beta = loss_config.get('beta_kld', loss_config.get('beta', 0.001))
    if name in ('l1', 'mse'):
        return SimpleVAELoss(mode=name, beta=beta)
    if name == 'perceptual':
        return VGGPerceptualLoss(beta=beta)
    raise ValueError(f"Unknown loss name: {name}. Use 'l1', 'mse' or 'perceptual'.")