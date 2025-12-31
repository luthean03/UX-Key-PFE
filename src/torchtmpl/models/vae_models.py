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

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
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

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (self.dec_channels, self.dec_h, self.dec_w)),
            nn.Upsample(scale_factor=2),
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2),
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2),
            ResidualBlock(64, 32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        features = self.encoder(x)
        pooled = self.spp(features)
        mu, logvar = self.fc_mu(pooled), self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)

        recon_small = self.decoder(self.fc_decode(z))
        recon = F.interpolate(recon_small, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        return recon, mu, logvar

        # Map latent vector back to decoder input size
        h_dec = self.fc_decode(z)
        recon_x = self.decoder(h_dec)
        return recon_x, mu, logvar