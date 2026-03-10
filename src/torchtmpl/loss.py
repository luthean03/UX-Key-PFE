import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVAELoss(nn.Module):
    """VAE ELBO loss: pixel reconstruction + KL divergence.
    
    Adapté pour les wireframes à densité variable (rectangles superposés).
    Utilise une normalisation 'Per-Pixel' et 'Per-Latent-Spatial-Volume'
    pour stabiliser l'entraînement indépendamment de la taille des images (Fully Conv).

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
            mu, logvar: Paramètres de l'espace latent (B, C_latent, H_latent, W_latent)
            mask: Masque binaire des pixels valides (pour géométrie variable)
        """
        # Cast pour éviter les erreurs de type (float16/float32)
        recon_x = recon_x.float()
        x = x.float()
        mu = mu.float()
        logvar = logvar.float()
        B = mu.size(0)

        # ---------------------------------------------------------------------
        # 1. NORMALISATION DE LA RECONSTRUCTION (Le "Per-Pixel")
        # ---------------------------------------------------------------------
        if mask is not None:
            num_pixels = mask.sum().clamp(min=1.0)
            mask = mask.float()
        else:
            num_pixels = x.numel()

        # ---------------------------------------------------------------------
        # 2. TERME DE RECONSTRUCTION (Énergie de l'erreur)
        # ---------------------------------------------------------------------
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

        # Normalisation de la reconstruction par pixel
        recon_loss = term_recon / num_pixels

        # ---------------------------------------------------------------------
        # 3. TERME KL DIVERGENCE (Universel : Latent 1D ou Spatial 3D)
        # ---------------------------------------------------------------------
        kld_tensor = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # On aplatit les dimensions non-batch pour faire une somme sécurisée
        kld_per_sample = kld_tensor.view(B, -1).sum(dim=1)
        
        # NORMALISATION SPATIALE DYNAMIQUE : 
        if mu.dim() == 4:
            # Pour le nouveau FullyConvVAE (B, C, H, W)
            spatial_volume = mu.size(2) * mu.size(3)
        else:
            # Pour l'ancien VAE avec Linear (B, D)
            spatial_volume = 1.0 
            
        kld_loss = (kld_per_sample / spatial_volume).mean()

        # ---------------------------------------------------------------------
        # 4. ASSEMBLAGE
        # ---------------------------------------------------------------------
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