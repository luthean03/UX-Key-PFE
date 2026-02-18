import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVAELoss(nn.Module):
    """VAE ELBO loss: pixel reconstruction + KL divergence.
    
    Adapté pour les wireframes à densité variable (rectangles superposés).
    Utilise une normalisation 'Per-Pixel' pour stabiliser l'entraînement
    indépendamment de la taille des images (SmartBatching) ou de la complexité.

    Args:
        mode: 'l1' (recommandé pour l'accumulation), 'mse', ou 'bce'.
        beta: Poids pour le terme KLD.
    """

    def __init__(self, mode: str = 'l1', beta: float = 1.0):
        super().__init__()
        self.mode = mode
        self.beta = beta

    def forward(self, recon_x, x, mu, logvar, mask=None):
        """
        Args:
            recon_x: Image reconstruite (B, C, H, W) -> Valeurs dans [0, 1]
            x: Image cible (B, C, H, W) -> Valeurs dans [0, 1] (nuances de gris)
            mu, logvar: Paramètres de l'espace latent
            mask: Masque binaire des pixels valides (pour géométrie variable)
        """
        # Cast pour éviter les erreurs de type (float16/float32)
        recon_x = recon_x.float()
        x = x.float()
        mu = mu.float()
        logvar = logvar.float()

        # ---------------------------------------------------------------------
        # 1. NORMALISATION UNIVERSELLE (Le "Per-Pixel")
        # ---------------------------------------------------------------------
        # On compte le nombre exact de pixels valides sur lesquels on apprend.
        # Cela rend la loss invariante à la résolution de l'image.
        if mask is not None:
            num_pixels = mask.sum().clamp(min=1.0)
        else:
            num_pixels = x.numel()

        # ---------------------------------------------------------------------
        # 2. TERME DE RECONSTRUCTION (Énergie de l'erreur)
        # ---------------------------------------------------------------------
        # On calcule la SOMME des erreurs, pas la moyenne immédiate.
        
        if mask is not None:
            mask = mask.float()
            if self.mode == 'l1':
                # Idéal pour vos rectangles : erreur proportionnelle au nombre de couches manquantes
                error_map = torch.abs(recon_x - x)
                term_recon = (error_map * mask).sum()
            elif self.mode == 'mse':
                # Punit plus fort les grosses erreurs, mais peut flouter les bords des rectangles
                error_map = (recon_x - x) ** 2
                term_recon = (error_map * mask).sum()
            elif self.mode == 'bce':
                # Moins adapté si vos gris sont des compteurs (0, 1, 2...) normalisés
                bce = F.binary_cross_entropy(recon_x, x, reduction='none')
                term_recon = (bce * mask).sum()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        else:
            # Version sans masque (standard)
            if self.mode == 'l1':
                term_recon = F.l1_loss(recon_x, x, reduction='sum')
            elif self.mode == 'mse':
                term_recon = F.mse_loss(recon_x, x, reduction='sum')
            elif self.mode == 'bce':
                term_recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        # ---------------------------------------------------------------------
        # 3. TERME KL DIVERGENCE (Coût de la structure latente)
        # ---------------------------------------------------------------------
        # On somme sur toutes les dimensions latentes (dim=1) et tout le batch.
        # Formule : -0.5 * sum(1 + log(var) - mu^2 - var)
        kld_sum = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # ---------------------------------------------------------------------
        # 4. ASSEMBLAGE ET NORMALISATION
        # ---------------------------------------------------------------------
        # On divise les deux termes par le MÊME nombre de pixels.
        # Ainsi, le ratio entre reconstruction et KL est physiquement cohérent.
        
        recon_loss = term_recon / num_pixels
        kld_loss = kld_sum / num_pixels

        total = recon_loss + self.beta * kld_loss

        return total, recon_loss, kld_loss
    
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