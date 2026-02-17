import torch
import torch.nn as nn
import torch.nn.functional as F


# SSIM Loss Implementation
def gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g)


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss for preserving structure.
    
    SSIM is ideal for wireframes because it focuses on:
    - Luminance (mean pixel intensity)
    - Contrast (standard deviation)
    - Structure (correlation between patterns)
    """
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5, 
                 data_range: float = 1.0, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.size_average = size_average
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2
        
        # Pre-compute Gaussian window
        self.register_buffer('window', None)
    
    def _get_window(self, channels: int, device: torch.device) -> torch.Tensor:
        """Get or create Gaussian window for given channels."""
        if self.window is None or self.window.device != device:
            kernel = gaussian_kernel(self.window_size, self.sigma, device)
            window = kernel.unsqueeze(0).unsqueeze(0)
            window = window.expand(channels, 1, -1, -1).contiguous()
            self.window = window
        return self.window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)."""
        channels = pred.shape[1]
        window = self._get_window(channels, pred.device)
        
        # Apply mask if provided
        if mask is not None:
            pred = pred * mask
            target = target * mask
        
        # Compute local means
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances and covariance
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channels) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        # Apply mask to SSIM map
        if mask is not None:
            mask_small = F.interpolate(mask, size=ssim_map.shape[2:], mode='nearest')
            ssim_map = ssim_map * mask_small
            # Mean over valid pixels only
            loss = 1 - ssim_map.sum() / (mask_small.sum() + 1e-8)
        elif self.size_average:
            loss = 1 - ssim_map.mean()
        else:
            loss = 1 - ssim_map.sum()
        
        return loss


# ===================== SIMPLE VAE LOSS =====================
class SimpleVAELoss(nn.Module):
    """Simple VAE loss using L1, MSE, or SSIM reconstruction + KLD.

    Standard VAE ELBO loss: L = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
    
    Normalisation strategies:
    - recon_loss (L1/MSE): 
        * 'sum': sum over ALL pixels → L1 ~50,000 (1M pixels, B=4)
                 Use beta ~0.0001-0.001
        
        * 'batch_mean': sum per image, average over batch → L1 ~12,500 per image
                        Use beta ~10-100 (RECOMMANDÉ pour interprétabilité)
        
        * 'mean': average over ALL pixels → L1 ~0.01-0.1 per pixel
                  Use beta ~0.01-1
    
    - kld_loss: sum over latent dims, mean over batch (standard VAE)
                Typ. ~50-200 pour latent_dim=128 au début, ~10-50 après convergence
    
    Recommendation: 'batch_mean' avec beta=10-100 donne le meilleur équilibre
    et permet d'interpréter recon_loss comme "erreur L1 moyenne par image"
    
    Args:
        mode: 'l1', 'mse', or 'ssim'
        beta: weight for KLD (depends on recon_reduction)
        recon_reduction: 'sum', 'batch_mean', or 'mean'
        scale: global scaling factor for display only
    """

    def __init__(self, mode='l1', beta=1.0, recon_reduction='batch_mean', scale=1.0):
        super().__init__()
        self.mode = mode
        self.beta = beta
        self.recon_reduction = recon_reduction
        self.scale = scale
        
        # Initialize SSIM loss module if needed
        if mode == 'ssim':
            self.ssim_loss = SSIMLoss()

    def forward(self, recon_x, x, mu, logvar, mask=None):
        # Force FP32 for precision
        recon_x = recon_x.float()
        x = x.float()
        
        B, C, H, W = x.shape
        num_pixels = B * C * H * W
        
        # Reconstruction loss with configurable reduction
        if self.mode == 'ssim':
            # SSIM loss returns (1 - SSIM) in [0, 1] range
            ssim_loss_raw = self.ssim_loss(recon_x, x, mask)
            if self.recon_reduction == 'mean':
                recon_loss = ssim_loss_raw  # Already in [0,1]
            elif self.recon_reduction == 'batch_mean':
                recon_loss = ssim_loss_raw * (C * H * W)  # Scale to per-image magnitude
            else:  # sum
                recon_loss = ssim_loss_raw * num_pixels  # Scale to total magnitude
        elif mask is not None:
            mask = mask.float()
            if self.mode == 'mse':
                diff = ((recon_x - x) ** 2) * mask
            else:  # l1
                diff = torch.abs(recon_x - x) * mask
            
            diff_sum = diff.sum()
            if self.recon_reduction == 'mean':
                # Mean over all pixels (all images)
                num_valid = mask.sum() + 1e-8
                recon_loss = diff_sum / num_valid
            elif self.recon_reduction == 'batch_mean':
                # Mean per image (average over batch)
                recon_loss = diff_sum / B
            else:  # sum
                recon_loss = diff_sum
        else:
            # Standard L1 or MSE
            if self.mode == 'mse':
                if self.recon_reduction == 'batch_mean':
                    # Sum over pixels, mean over batch
                    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / B
                else:
                    recon_loss = F.mse_loss(recon_x, x, reduction=self.recon_reduction)
            elif self.mode == 'l1':
                if self.recon_reduction == 'batch_mean':
                    # Sum over pixels, mean over batch
                    recon_loss = F.l1_loss(recon_x, x, reduction='sum') / B
                else:
                    recon_loss = F.l1_loss(recon_x, x, reduction=self.recon_reduction)
            else:
                raise ValueError(f"Unknown recon loss mode: {self.mode}. Use 'l1', 'mse', or 'ssim'.")

        # KL Divergence loss (standard VAE: sum over latent dims, mean over batch)
        # Formula: -0.5 * sum_j(1 + logvar_j - mu_j^2 - exp(logvar_j))
        mu = mu.float()
        logvar = logvar.float()
        
        # KLD: sum over latent dimensions, mean over batch (consistent with recon batch averaging)
        kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
        kld_loss = kld_per_sample.mean()  # Average over batch
        
        # Total loss
        # With recon_reduction='batch_mean': L1 ~10,000, KLD ~10-50 → use beta~10-100
        # With recon_reduction='mean': L1 ~0.01-0.1, KLD ~10-50 → use beta~0.01-1
        # With recon_reduction='sum': L1 ~40,000, KLD ~10-50 → use beta~0.0001-0.001
        total = recon_loss + (self.beta * kld_loss)
        
        # Scale for display (does not change relative gradients)
        total_scaled = total * self.scale
        
        return total_scaled, recon_loss, kld_loss