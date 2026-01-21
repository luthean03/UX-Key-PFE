"""SPP-based VAE for variable-size images with structured latent space.

This model is designed for UI/UX wireframe learning with focus on:
1. Clustering: Clear separation of archetypes in latent space
2. Interpolation: Smooth transitions between designs (SLERP support)
3. Generation: Valid wireframe sampling from N(0,I)

Uses Spatial Pyramid Pooling for fixed-size latent regardless of input height,
and GroupNorm to work with batch_size=1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MaskedGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x, mask):
        # Si pas de masque, comportement standard
        if mask is None:
            # F.group_norm attend un poids de taille (C,), pas (1, C, 1, 1)
            return F.group_norm(x, self.num_groups, self.weight.squeeze(), self.bias.squeeze(), self.eps)

        B, C, H, W = x.shape
        G = self.num_groups
        
        # Reshape pour séparer les groupes
        x_g = x.view(B, G, C // G, H, W)
        mask_g = mask.view(B, 1, 1, H, W) # Broadcast sur les groupes

        # Moyenne pondérée (somme / compte)
        x_sum = (x_g * mask_g).sum(dim=[2, 3, 4], keepdim=True)
        mask_sum = mask_g.expand_as(x_g).sum(dim=[2, 3, 4], keepdim=True)
        mean = x_sum / (mask_sum + self.eps)

        # Variance pondérée
        var_sum = ((x_g - mean).pow(2) * mask_g).sum(dim=[2, 3, 4], keepdim=True)
        var = var_sum / (mask_sum + self.eps)

        # Normalisation + Affine
        x_norm = (x_g - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(B, C, H, W) * mask
        
        return x_norm * self.weight + self.bias


class MaskedSPPLayer(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        super().__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        features = []
        
        # Appliquer masque pour sécurité (zéros parfaits sur padding)
        if mask is not None:
            x = x * mask
        
        for size in self.pool_sizes:
            if mask is None:
                pool = F.adaptive_avg_pool2d(x, (size, size))
            else:
                # Moyenne masquée adaptative
                x_sum = F.adaptive_avg_pool2d(x, (size, size)) 
                mask_sum = F.adaptive_avg_pool2d(mask, (size, size))
                # x_sum est en fait Moyenne(x), mask_sum est Moyenne(mask)
                # Le ratio corrige la dilution par les zéros
                pool = x_sum / (mask_sum + 1e-6)
                
            features.append(pool.view(batch_size, -1))
            
        return torch.cat(features, dim=1)


class CBAM(nn.Module):
    """Module d'Attention (Channel + Spatial) pour affiner les features"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 1. Channel Attention (Qu'est-ce qui est important ?)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = x * self.sigmoid_channel(avg_out + max_out)
        
        # 2. Spatial Attention (Où est-ce important ?)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial_out = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        
        return out * spatial_out

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        num_groups = min(32, out_c)
        if out_c % num_groups != 0:
            num_groups = out_c # Fallback to instance norm style if not divisible
            
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.gn1 = MaskedGroupNorm(num_groups, out_c) 
        
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.gn2 = MaskedGroupNorm(num_groups, out_c)
        
        self.cbam = CBAM(out_c) 
        
        self.shortcut = nn.Sequential()
        self.use_projection = (stride != 1 or in_c != out_c)
        if self.use_projection:
            self.shortcut_conv = nn.Conv2d(in_c, out_c, 1, stride, bias=False)
            self.shortcut_gn = MaskedGroupNorm(num_groups, out_c)

    def forward(self, x, mask=None):
        # Assurer que le masque d'entrée correspond à l'entrée x
        if mask is not None and mask.shape[2:] != x.shape[2:]:
            mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')

        out = self.conv1(x)
        
        # Préparer le masque pour la sortie (peut avoir rétréci si stride > 1)
        # Note: ceci devient le masque actif pour out, gn1, conv2, gn2
        mask_out = mask
        if mask is not None and mask.shape[2:] != out.shape[2:]:
            mask_out = F.interpolate(mask, size=out.shape[2:], mode='nearest')

        out = F.relu(self.gn1(out, mask_out)) 
        
        out = self.conv2(out)
        out = self.gn2(out, mask_out) 
        
        out = self.cbam(out)
        
        identity = x
        if self.use_projection:
            identity = self.shortcut_conv(identity)
            # shortcut_conv a le même stride que conv1, donc la sortie a la taille de mask_out
            identity = self.shortcut_gn(identity, mask_out) 
            
        out += identity
        return F.relu(out)

class VAE(nn.Module):
    def __init__(self, config, input_size, num_classes=0):
        super(VAE, self).__init__()
        self.latent_dim = config.get("latent_dim", 128)
        self.spp_levels = [1, 2, 4]
        self.dropout_p = config.get("dropout_p", 0.0)
        self.use_skip_connections = config.get("use_skip_connections", False)

        # Encoder avec stockage des features intermédiaires
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.enc_block1 = ResidualBlock(64, 128, stride=2)   # → features 128 canaux
        self.enc_block2 = ResidualBlock(128, 256, stride=2)  # → features 256 canaux
        self.enc_block3 = ResidualBlock(256, 512, stride=2)  # → features 512 canaux

        self.spp = MaskedSPPLayer(self.spp_levels)
        self.spp_out_dim = 512 * sum([s * s for s in self.spp_levels])

        self.fc_mu = nn.Linear(self.spp_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.spp_out_dim, self.latent_dim)

        # Make decoder base with 2:1 aspect ratio (tall for mobile screens)
        # Height:Width = 32:16 = 2:1 for vertical mobile layouts
        self.dec_h, self.dec_w = 32, 16
        self.dec_channels = 256
        self.fc_decode = nn.Linear(self.latent_dim, self.dec_channels * self.dec_h * self.dec_w)

        # Decoder architecture with optional skip connections and dropout
        self.dec_unflatten = nn.Unflatten(1, (self.dec_channels, self.dec_h, self.dec_w))
        
        if self.dropout_p > 0:
            self.dec_dropout1 = nn.Dropout2d(self.dropout_p)
        
        self.dec_up1 = nn.Upsample(scale_factor=2)
        
        # Ajustement des canaux pour skip connections
        if self.use_skip_connections:
            self.dec_block1 = ResidualBlock(256 + 256, 128)  # Concat skip de enc_block2
        else:
            self.dec_block1 = ResidualBlock(256, 128)
        
        if self.dropout_p > 0:
            self.dec_dropout2 = nn.Dropout2d(self.dropout_p)
        
        self.dec_up2 = nn.Upsample(scale_factor=2)
        
        if self.use_skip_connections:
            self.dec_block2 = ResidualBlock(128 + 128, 64)  # Concat skip de enc_block1
        else:
            self.dec_block2 = ResidualBlock(128, 64)
        
        self.dec_up3 = nn.Upsample(scale_factor=2)
        self.dec_block3 = ResidualBlock(64, 32)
        
        self.dec_up4 = nn.Upsample(scale_factor=2)
        self.dec_final = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode from latent code, compatible with skip connections.
        
        When skip connections are enabled, creates zero tensors as dummy skips.
        This ensures interpolation works correctly.
        
        Args:
            z: (batch, latent_dim) latent codes
            
        Returns:
            recon: (batch, 1, H, W) reconstructed images
        """
        d = self.fc_decode(z)
        d = self.dec_unflatten(d)
        
        if self.dropout_p > 0:
            d = self.dec_dropout1(d)
        
        d = self.dec_up1(d)  # (batch, 256, 128, 8)
        
        # Si skip connections activées, créer dummy skip de zéros
        if self.use_skip_connections:
            # Dummy skip de 256 canaux (de enc_block2)
            dummy_skip1 = torch.zeros(d.shape[0], 256, d.shape[2], d.shape[3], 
                                     device=d.device, dtype=d.dtype)
            d = torch.cat([d, dummy_skip1], dim=1)  # (batch, 512, 128, 8)
        
        d = self.dec_block1(d)
        
        if self.dropout_p > 0:
            d = self.dec_dropout2(d)
        
        d = self.dec_up2(d)  # (batch, 128, 256, 16)
        
        # Si skip connections activées, créer dummy skip de zéros
        if self.use_skip_connections:
            # Dummy skip de 128 canaux (de enc_block1)
            dummy_skip2 = torch.zeros(d.shape[0], 128, d.shape[2], d.shape[3],
                                     device=d.device, dtype=d.dtype)
            d = torch.cat([d, dummy_skip2], dim=1)  # (batch, 256, 256, 16)
        
        d = self.dec_block2(d)
        
        d = self.dec_up3(d)
        d = self.dec_block3(d)
        d = self.dec_up4(d)
        recon = self.dec_final(d)
        
        return recon

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input validation
        assert x.dim() == 4, f"Expected 4D input (B,C,H,W), got {x.dim()}D shape {x.shape}"
        assert x.shape[1] == 1, f"Expected 1 channel, got {x.shape[1]}"
        assert x.shape[0] > 0, "Batch size must be positive"
        
        if mask is not None:
            assert mask.shape == x.shape, f"Mask shape {mask.shape} != input shape {x.shape}"
            assert mask.dtype == torch.float32, f"Mask dtype should be float32, got {mask.dtype}"
        
        orig_h, orig_w = x.shape[2], x.shape[3]
        
        # --- Encoder avec propagation du masque ---
        
        # enc_conv1 (manuel car c'est un Sequential dans ton code actuel)
        e1 = self.enc_conv1[0](x) # Conv
        e1 = self.enc_conv1[1](e1) # GN (Standard)
        e1 = self.enc_conv1[2](e1) # ReLU
        e1 = self.enc_conv1[3](e1) # MaxPool
        
        # Maintenant on a un masque 1/2 taille (MaxPool stride 2)
        if mask is not None:
            m1 = F.interpolate(mask, size=e1.shape[2:], mode='nearest')
        else:
            m1 = None

        # Blocs résiduels (qui acceptent le masque maintenant)
        e2 = self.enc_block1(e1, mask=m1)
        if mask is not None:
             m2 = F.interpolate(mask, size=e2.shape[2:], mode='nearest')
        else: m2 = None

        e3 = self.enc_block2(e2, mask=m2)
        if mask is not None:
             m3 = F.interpolate(mask, size=e3.shape[2:], mode='nearest')
        else: m3 = None

        if self.use_skip_connections:
            features = self.enc_block3(e3, mask=m3)
        else:
            features = self.enc_block3(e3, mask=m3)
        
        if mask is not None:
            m_feat = F.interpolate(mask, size=features.shape[2:], mode='nearest')
        else:
            m_feat = None
        
        # Masked SPP
        pooled = self.spp(features, mask=m_feat)
        mu, logvar = self.fc_mu(pooled), self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_small = self.decode(z)
        
        recon = F.interpolate(recon_small, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        
        # Appliquer le masque final sur la reconstruction pour nettoyer la sortie
        if mask is not None:
            recon = recon * mask
            
        return recon, mu, logvar
    
    def sample(self, num_samples: int = 1, device: torch.device = None, 
               output_size: tuple = (1024, 128)) -> torch.Tensor:
        """Generate new wireframes by sampling from N(0, I).
        
        Args:
            num_samples: Number of wireframes to generate
            device: Device to generate on (defaults to model's device)
            output_size: (height, width) of generated images
            
        Returns:
            samples: (num_samples, 1, H, W) generated wireframe images
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from standard normal
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode
        samples = self.decode(z)
        
        # Resize to desired output size
        if samples.shape[2:] != output_size:
            samples = F.interpolate(samples, size=output_size, 
                                   mode='bilinear', align_corners=False)
        
        return samples
    
    def encode(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """Encode input images to latent space.
        
        Args:
            x: (B, 1, H, W) input wireframe images
            mask: (B, 1, H, W) optional mask for variable-size handling
            
        Returns:
            mu: (B, latent_dim) mean of latent distribution
            logvar: (B, latent_dim) log-variance of latent distribution
            z: (B, latent_dim) sampled latent code
        """
        # Encoder avec propagation du masque
        e1 = self.enc_conv1[0](x)  # Conv
        e1 = self.enc_conv1[1](e1)  # GN
        e1 = self.enc_conv1[2](e1)  # ReLU
        e1 = self.enc_conv1[3](e1)  # MaxPool
        
        if mask is not None:
            m1 = F.interpolate(mask, size=e1.shape[2:], mode='nearest')
        else:
            m1 = None

        e2 = self.enc_block1(e1, mask=m1)
        if mask is not None:
            m2 = F.interpolate(mask, size=e2.shape[2:], mode='nearest')
        else:
            m2 = None

        e3 = self.enc_block2(e2, mask=m2)
        if mask is not None:
            m3 = F.interpolate(mask, size=e3.shape[2:], mode='nearest')
        else:
            m3 = None

        features = self.enc_block3(e3, mask=m3)
        
        if mask is not None:
            m_feat = F.interpolate(mask, size=features.shape[2:], mode='nearest')
        else:
            m_feat = None
        
        # Masked SPP
        pooled = self.spp(features, mask=m_feat)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)
        
        return mu, logvar, z