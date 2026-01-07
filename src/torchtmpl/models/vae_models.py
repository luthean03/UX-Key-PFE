"""SPP-based VAE for variable-size images.

This model uses Spatial Pyramid Pooling to obtain a fixed-size
representation regardless of input height, and GroupNorm so it
works with batch_size=1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPPLayer(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        super(SPPLayer, self).__init__()
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in pool_sizes])

    def forward(self, x):
        batch_size = x.size(0)
        features = []
        for pool in self.pools:
            features.append(pool(x).view(batch_size, -1))
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
    """Bloc ResNet Amélioré avec Attention CBAM"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.gn1 = nn.GroupNorm(32, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(32, out_c)
        
        # === AJOUT DE L'ATTENTION ===
        self.cbam = CBAM(out_c)
        # ============================
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.GroupNorm(32, out_c)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        # Application de l'attention avant l'addition
        out = self.cbam(out)
        
        out += self.shortcut(x)
        return F.relu(out)

class VAE(nn.Module):
    def __init__(self, config, input_size, num_classes=0):
        super(VAE, self).__init__()
        self.latent_dim = config.get("latent_dim", 128)
        self.spp_levels = [1, 2, 4]
        
        # AMÉLIORATION: Dropout configurable pour régularisation
        self.dropout_p = config.get("dropout_p", 0.0)
        
        # AMÉLIORATION: Skip Connections U-Net style
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
        
        # Ancien encoder (pour compatibilité)
        self.encoder = nn.Sequential(
            self.enc_conv1,
            self.enc_block1,
            self.enc_block2,
            self.enc_block3,
        )

        self.spp = SPPLayer(self.spp_levels)
        self.spp_out_dim = 512 * sum([s * s for s in self.spp_levels])

        self.fc_mu = nn.Linear(self.spp_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.spp_out_dim, self.latent_dim)

        # Make decoder base more elongated to better match tall web pages
        # (increase vertical resolution to reduce upsample distortion)
        self.dec_h, self.dec_w = 64, 4
        self.dec_channels = 256
        self.fc_decode = nn.Linear(self.latent_dim, self.dec_channels * self.dec_h * self.dec_w)

        # AMÉLIORATION: Decoder avec Skip Connections U-Net et Dropout2d
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

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        
        # Encoder avec sauvegarde des features pour skip connections
        if self.use_skip_connections:
            e1 = self.enc_conv1(x)           # features à résolution élevée
            e2 = self.enc_block1(e1)         # 128 canaux
            e3 = self.enc_block2(e2)         # 256 canaux
            features = self.enc_block3(e3)   # 512 canaux (bottleneck)
        else:
            features = self.encoder(x)
        
        pooled = self.spp(features)
        mu, logvar = self.fc_mu(pooled), self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)

        # Decoder avec skip connections U-Net style
        d = self.fc_decode(z)
        d = self.dec_unflatten(d)
        
        if self.dropout_p > 0:
            d = self.dec_dropout1(d)
        
        d = self.dec_up1(d)  # Upsample
        
        # Skip connection depuis enc_block2 (256 canaux)
        if self.use_skip_connections:
            # Resize skip feature map to match decoder size
            e3_resized = F.interpolate(e3, size=(d.shape[2], d.shape[3]), mode='bilinear', align_corners=False)
            d = torch.cat([d, e3_resized], dim=1)  # Concatenation
        
        d = self.dec_block1(d)
        
        if self.dropout_p > 0:
            d = self.dec_dropout2(d)
        
        d = self.dec_up2(d)  # Upsample
        
        # Skip connection depuis enc_block1 (128 canaux)
        if self.use_skip_connections:
            e2_resized = F.interpolate(e2, size=(d.shape[2], d.shape[3]), mode='bilinear', align_corners=False)
            d = torch.cat([d, e2_resized], dim=1)
        
        d = self.dec_block2(d)
        d = self.dec_up3(d)
        d = self.dec_block3(d)
        d = self.dec_up4(d)
        recon_small = self.dec_final(d)
        
        recon = F.interpolate(recon_small, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        return recon, mu, logvar