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
    """Hybrid loss: Pixel L1 + VGG perceptual + KLD.

    For wireframes (thin black lines on white), a pure perceptual loss often
    produces "dreamy" textures: it can match VGG features without placing
    pixels exactly. Adding a strong pixel-space term anchors structure.

    Args:
        beta: weight for KLD term
        perceptual_weight: relative weight for the VGG term (after rescaling)

    Notes:
    - Expects inputs in [0,1]. Converts to 3 channels if needed and applies
      ImageNet normalization before passing through VGG.
    - Pixel term uses reduction='sum' to keep the same scale as SimpleVAELoss.
    - VGG feature term uses reduction='mean' then is rescaled to match 'sum'.
    """

    def __init__(self, beta=1.0, perceptual_weight: float = 0.1):
        super().__init__()
        self.beta = float(beta)
        self.perceptual_weight = float(perceptual_weight)
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
        # 1) Pixel loss: crucial for structure
        pixel_loss = F.l1_loss(recon_x, x, reduction='sum')

        # 2) VGG perceptual term
        if x.shape[1] == 1:
            x_vgg = x.repeat(1, 3, 1, 1)
            recon_vgg = recon_x.repeat(1, 3, 1, 1)
        else:
            x_vgg, recon_vgg = x, recon_x

        x_norm = (x_vgg - self.mean) / self.std
        recon_norm = (recon_vgg - self.mean) / self.std

        loss_feat = 0.0
        x_feat = x_norm
        recon_feat = recon_norm
        for block in self.blocks:
            x_feat = block(x_feat)
            recon_feat = block(recon_feat)
            loss_feat = loss_feat + F.l1_loss(recon_feat, x_feat, reduction='mean')

        # Rescale feature loss (mean) to be comparable to pixel_loss (sum)
        scale_factor = x.numel()
        vgg_term = loss_feat * scale_factor * self.perceptual_weight

        # 3) KLD
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total = pixel_loss + vgg_term + (self.beta * kld_loss)
        return total, pixel_loss, kld_loss


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
        perceptual_weight = float(loss_config.get('perceptual_weight', 0.05))
        return VGGPerceptualLoss(beta=beta, perceptual_weight=perceptual_weight)
    raise ValueError(f"Unknown loss name: {name}. Use 'l1', 'mse' or 'perceptual'.")