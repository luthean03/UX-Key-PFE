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


# Gradient Loss Implementation
class GradientLoss(nn.Module):
    """Edge-preserving loss using Sobel gradients.
    
    Wireframes are defined by their edges (lines, boxes).
    This loss ensures edges are sharp and correctly positioned.
    """
    
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute gradient magnitude difference."""
        # Compute gradients for prediction
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-8)
        
        # Compute gradients for target
        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        target_grad = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-8)
        
        # L1 difference of gradient magnitudes
        diff = torch.abs(pred_grad - target_grad)
        
        if mask is not None:
            return (diff * mask).sum()
        return diff.sum()


# Multi-Scale Loss Implementation
class MultiScaleLoss(nn.Module):
    """Multi-scale reconstruction loss for capturing both global layout and fine details.
    
    Wireframes have hierarchical structure:
    - Global: page sections, major blocks
    - Medium: UI components, buttons, forms
    - Fine: text placeholders, icons, small elements
    """
    
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
            else:  # mse
                diff = (pred_scaled - target_scaled) ** 2
            
            if mask_scaled is not None:
                loss = (diff * mask_scaled).sum()
            else:
                loss = diff.sum()
            
            # Weight by scale (larger scales = more weight)
            total_loss += loss * scale
        
        return total_loss / sum(self.scales)


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


# Perceptual VAE Loss Implementation
class PerceptualVAELoss(nn.Module):
    """Advanced VAE loss with perceptual components for structure preservation.
    
    Combines:
    - L1 reconstruction (pixel accuracy, mean per pixel)
    - SSIM (structural similarity, ~[0,1])
    - Gradient loss (edge preservation, normalized per pixel)
    - Multi-scale loss (hierarchical features, normalized)
    - KL divergence (sum over dims, mean over batch - standard VAE)
    
    Note: KLD is typically ~50-200 (for latent_dim=128), so beta should
    be small (~0.0001-0.001) to balance with recon_loss (~0.05).
    """
    
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
        B = x.size(0)
        
        # 1. L1 Reconstruction Loss (SUM over pixels, not mean)
        # Changed to sum for natural balance with beta_kld=1
        if mask is not None:
            l1_loss = (torch.abs(recon_x - x) * mask).sum()
        else:
            l1_loss = F.l1_loss(recon_x, x, reduction='sum')
        
        # 2. SSIM Loss (structural) - already normalized [0, 1]
        ssim_loss = self.ssim_loss(recon_x, x, mask)
        
        # 3. Gradient Loss (edges) - returns sum, need to scale appropriately
        grad_loss_raw = self.gradient_loss(recon_x, x, mask)
        # Scale gradient loss relative to L1 loss for balance
        grad_loss = grad_loss_raw
        
        # 4. Multi-scale Loss (hierarchical) - returns mean-reduced value
        if self.use_multiscale:
            ms_loss_raw = self.multiscale_loss(recon_x, x, mask)
            ms_loss = ms_loss_raw
        else:
            ms_loss = 0.0
        
        # Combine reconstruction losses
        # All components are now on similar scale (sum-based except SSIM which is [0,1])
        recon_loss = (self.lambda_l1 * l1_loss + 
                      self.lambda_ssim * ssim_loss * l1_loss.detach() +
                      self.lambda_gradient * grad_loss +
                      self.lambda_multiscale * ms_loss)
        
        # 5. KL Divergence (standard: sum over dims, mean over batch)
        kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
        kld_loss = kld_per_sample.mean()  # Average over batch
        
        total = recon_loss + (self.beta * kld_loss)
        
        # Global scaling for display
        total_scaled = total * self.scale
        
        return total_scaled, recon_loss, kld_loss


# Factory function
def get_vae_loss(loss_config: dict):
    """Factory: instantiate a loss module from config.

    Expected keys in loss_config:
      - name: 'l1' | 'mse' | 'ssim' | 'perceptual'
      - beta_kld: float (weight for KL divergence)
      - recon_reduction: 'sum' | 'batch_mean' | 'mean' (how to reduce reconstruction loss)
      - loss_scale: float (global scaling factor, default 1.0)
      - lambda_ssim: float (weight for SSIM, only for perceptual)
      - lambda_gradient: float (weight for gradient loss, only for perceptual)
    """
    name = loss_config.get('name', 'l1').lower()
    beta = loss_config.get('beta_kld', loss_config.get('beta', 0.001))
    scale = float(loss_config.get('loss_scale', 1.0))
    recon_reduction = loss_config.get('recon_reduction', 'batch_mean')  # Default to 'batch_mean'
    
    if name in ('l1', 'mse', 'ssim'):
        return SimpleVAELoss(mode=name, beta=beta, recon_reduction=recon_reduction, scale=scale)
    
    if name == 'perceptual':
        return PerceptualVAELoss(
            beta=beta,
            lambda_l1=loss_config.get('lambda_l1', 1.0),
            lambda_ssim=loss_config.get('lambda_ssim', 0.5),
            lambda_gradient=loss_config.get('lambda_gradient', 0.1),
            lambda_multiscale=loss_config.get('lambda_multiscale', 0.2),
            use_multiscale=loss_config.get('use_multiscale', True),
            scale=scale
        )
    
    raise ValueError(f"Unknown loss name: {name}. Use 'l1', 'mse', 'ssim', or 'perceptual'.")