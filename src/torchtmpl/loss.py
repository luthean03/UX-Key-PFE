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

        # Normalisation de la reconstruction par pixel
        recon_loss = term_recon / num_pixels

        # ---------------------------------------------------------------------
        # 3. TERME KL DIVERGENCE
        # ---------------------------------------------------------------------
        # Somme sur l'espace latent (dim=1) pour avoir la KLD totale par image,
        # puis moyenne sur la dimension du batch (dim=0).
        # Indépendant de la taille des images — échelle stable.
        kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kld_loss = kld_per_sample.mean()

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