"""Loss functions for VAE training.

Provides SimpleVAELoss (L1/MSE + KLD) and PerceptualVAELoss
(L1 + SSIM + gradient + multi-scale + KLD), plus a factory function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g)


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss (1 - SSIM)."""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5, 
                 data_range: float = 1.0, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.size_average = size_average
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2
        
        self.register_buffer('window', None)

    def _get_window(self, channels: int, device: torch.device) -> torch.Tensor:
        if self.window is None or self.window.device != device:
            kernel = gaussian_kernel(self.window_size, self.sigma, device)
            window = kernel.unsqueeze(0).unsqueeze(0)
            window = window.expand(channels, 1, -1, -1).contiguous()
            self.window = window
        return self.window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        channels = pred.shape[1]
        window = self._get_window(channels, pred.device)

        if mask is not None:
            pred = pred * mask
            target = target * mask
        
        pad = self.window_size // 2
        mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
        mu2 = F.conv2d(target, window, padding=pad, groups=channels)

        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        if mask is not None:
            mask_small = F.interpolate(mask, size=ssim_map.shape[2:], mode='nearest')
            ssim_map = ssim_map * mask_small
            return 1 - ssim_map.sum() / (mask_small.sum() + 1e-8)
        if self.size_average:
            return 1 - ssim_map.mean()
        return 1 - ssim_map.sum()


class SimpleVAELoss(nn.Module):
    """VAE ELBO loss: pixel reconstruction + KL divergence.

    Both terms are normalised to per-dimension averages so that their
    magnitudes are comparable regardless of image size or latent dim.

    Args:
        mode: ``'l1'`` or ``'mse'``.
        beta: Weight for the KLD term.
    """

    def __init__(self, mode: str = 'l1', beta: float = 1.0):
        super().__init__()
        self.mode = mode
        self.beta = beta

    def forward(self, recon_x, x, mu, logvar, mask=None):
        recon_x = recon_x.float()
        x = x.float()
        mu = mu.float()
        logvar = logvar.float()

        K = mu.shape[1]

        # Reconstruction: mean over pixels, mean over batch
        if mask is not None:
            mask = mask.float()
            if self.mode == 'mse':
                diff = ((recon_x - x) ** 2) * mask
            else:
                diff = torch.abs(recon_x - x) * mask
            num_valid = mask.sum().clamp(min=1.0)
            recon_loss = diff.sum() / num_valid
        else:
            if self.mode == 'mse':
                recon_loss = F.mse_loss(recon_x, x, reduction='mean')
            elif self.mode == 'l1':
                recon_loss = F.l1_loss(recon_x, x, reduction='mean')
            else:
                raise ValueError(f"Unknown mode: {self.mode}. Use 'l1' or 'mse'.")

        # KLD: mean over latent dims, mean over batch
        kld_per_sample = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        )
        kld_loss = kld_per_sample.mean() / K

        total = recon_loss + self.beta * kld_loss

        return total, recon_loss, kld_loss


class GradientLoss(nn.Module):
    """Edge-preserving loss using Sobel gradients."""
    
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-8)
        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        target_grad = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-8)
        diff = torch.abs(pred_grad - target_grad)
        if mask is not None:
            return (diff * mask).sum()
        return diff.sum()


class MultiScaleLoss(nn.Module):
    """Multi-scale reconstruction loss."""
    
    def __init__(self, scales: list = [1.0, 0.5, 0.25], mode: str = 'l1'):
        super().__init__()
        self.scales = scales
        self.mode = mode
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        total_loss = 0.0
        for scale in self.scales:
            if scale < 1.0:
                size = (int(pred.shape[2] * scale), int(pred.shape[3] * scale))
                pred_scaled = F.interpolate(pred, size=size, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
                mask_scaled = F.interpolate(mask, size=size, mode='nearest') if mask is not None else None
            else:
                pred_scaled = pred
                target_scaled = target
                mask_scaled = mask
            if self.mode == 'l1':
                diff = torch.abs(pred_scaled - target_scaled)
            else:
                diff = (pred_scaled - target_scaled) ** 2
            if mask_scaled is not None:
                loss = (diff * mask_scaled).sum()
            else:
                loss = diff.sum()
            total_loss += loss * scale
        return total_loss / sum(self.scales)


class PerceptualVAELoss(nn.Module):
    """VAE loss combining L1 + SSIM + gradient + multi-scale + KLD."""
    
    def __init__(self, beta: float = 1.0,
                 lambda_l1: float = 1.0,
                 lambda_ssim: float = 0.5,
                 lambda_gradient: float = 0.1,
                 lambda_multiscale: float = 0.2,
                 use_multiscale: bool = True,
                 scale: float = 1.0):
        super().__init__()
        self.beta = beta
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_gradient = lambda_gradient
        self.lambda_multiscale = lambda_multiscale
        self.use_multiscale = use_multiscale
        self.scale = scale
        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()
        if use_multiscale:
            self.multiscale_loss = MultiScaleLoss(scales=[1.0, 0.5, 0.25])
    
    def forward(self, recon_x, x, mu, logvar, mask=None):
        if mask is not None:
            l1_loss = (torch.abs(recon_x - x) * mask).sum()
        else:
            l1_loss = F.l1_loss(recon_x, x, reduction='sum')
        ssim_val = self.ssim_loss(recon_x, x, mask)
        grad_val = self.gradient_loss(recon_x, x, mask)
        ms_val = self.multiscale_loss(recon_x, x, mask) if self.use_multiscale else 0.0

        recon_loss = (self.lambda_l1 * l1_loss
                      + self.lambda_ssim * ssim_val * l1_loss.detach()
                      + self.lambda_gradient * grad_val
                      + self.lambda_multiscale * ms_val)
        kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kld_loss = kld_per_sample.mean()
        total = (recon_loss + self.beta * kld_loss) * self.scale
        return total, recon_loss, kld_loss


def get_vae_loss(loss_config: dict) -> nn.Module:
    """Instantiate a VAE loss module from a config dict.

    Recognised keys:
        name: ``'l1'`` | ``'mse'`` | ``'perceptual'``
        beta_kld: float -- weight for the KL divergence term.
    """
    name = loss_config.get('name', 'l1').lower()
    beta = loss_config.get('beta_kld', loss_config.get('beta', 1.0))
    
    if name in ('l1', 'mse'):
        return SimpleVAELoss(mode=name, beta=beta)
    
    if name == 'perceptual':
        scale = float(loss_config.get('loss_scale', 1.0))
        return PerceptualVAELoss(
            beta=beta,
            lambda_l1=loss_config.get('lambda_l1', 1.0),
            lambda_ssim=loss_config.get('lambda_ssim', 0.5),
            lambda_gradient=loss_config.get('lambda_gradient', 0.1),
            lambda_multiscale=loss_config.get('lambda_multiscale', 0.2),
            use_multiscale=loss_config.get('use_multiscale', True),
            scale=scale
        )
    
    raise ValueError(f"Unknown loss name: {name}. Use 'l1', 'mse', or 'perceptual'.")