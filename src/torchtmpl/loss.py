# Define VAE reconstruction and KL losses with masking and warmup-aware weighting.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVAELoss(nn.Module):
    """VAE ELBO loss: pixel reconstruction + KL divergence.

    Adapted for variable-density wireframes (overlapping rectangles).
    Uses per-pixel and per-latent-spatial-volume normalization to keep
    training stable across varying image sizes (fully convolutional setup).

    Args:
        mode: 'l1' (recommended with gradient accumulation), 'mse', or 'bce'.
        beta: Weight for the KL divergence term.
    """

    def __init__(self, mode: str = 'l1', beta: float = 1.0):
        super().__init__()
        self.mode = mode
        self.beta = beta

    def forward(self, recon_x, x, mu, logvar, mask=None):
        """
        Args:
            recon_x: Reconstructed image tensor (B, C, H, W), values in [0, 1].
            x: Target image tensor (B, C, H, W), values in [0, 1].
            mu, logvar: Latent distribution parameters (B, C_latent, H_latent, W_latent).
            mask: Binary mask of valid pixels for variable-shape samples.
        """

        # Keep computations in float32 for numerical stability.
        recon_x = recon_x.float()
        x = x.float()
        mu = mu.float()
        logvar = logvar.float()
        B = mu.size(0)




        # Count valid pixels only when masking is used.
        if mask is not None:
            num_pixels = mask.sum().clamp(min=1.0)
            mask = mask.float()
        else:
            num_pixels = x.numel()




        # Reconstruction term: compare output and target in image space.
        if mask is not None:
            if self.mode == 'l1':
                error_map = torch.abs(recon_x - x)
                term_recon = (error_map * mask).sum()
            elif self.mode == 'mse':
                error_map = (recon_x - x) ** 2
                term_recon = (error_map * mask).sum()
            elif self.mode == 'bce':
                bce = F.binary_cross_entropy(recon_x, x, reduction='none')
                term_recon = (bce * mask).sum()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        else:
            if self.mode == 'l1':
                term_recon = F.l1_loss(recon_x, x, reduction='sum')
            elif self.mode == 'mse':
                term_recon = F.mse_loss(recon_x, x, reduction='sum')
            elif self.mode == 'bce':
                term_recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
            else:
                raise ValueError(f"Unknown mode: {self.mode}")


        # Normalize by pixel count so loss stays comparable across image sizes.
        recon_loss = term_recon / num_pixels




        # KL term regularizes latent distribution toward N(0, I).
        kld_tensor = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if mu.dim() == 4:

            # Match mask resolution to latent feature map for fully conv models.
            mask_latent = F.interpolate(mask, size=(mu.size(2), mu.size(3)), mode='nearest')

            kld_tensor = kld_tensor * mask_latent


            # Normalize each sample by its valid latent area.
            spatial_volume = mask_latent.sum(dim=[2, 3]).mean(dim=1).clamp(min=1.0)
        else:
            spatial_volume = torch.ones(B, device=mu.device)


        kld_per_sample = kld_tensor.view(B, -1).sum(dim=1)
        kld_loss = (kld_per_sample / spatial_volume).mean()




        # Final ELBO-style objective.
        total = recon_loss + self.beta * kld_loss

        return total, recon_loss, kld_loss


def get_vae_loss(loss_config: dict) -> nn.Module:
    """Instantiate a VAE loss module from a config dict.

    Recognised keys:
        name: ``'l1'`` | ``'mse'``
        beta_kld: float -- weight for the KL divergence term.
    """
    name = loss_config.get('name', 'l1').lower()
    beta = loss_config.get('beta_kld', loss_config.get('beta', 1.0))

    if name in ('l1', 'mse'):
        return SimpleVAELoss(mode=name, beta=beta)

    raise ValueError(f"Unknown loss name: {name}. Use 'l1' or 'mse'.")