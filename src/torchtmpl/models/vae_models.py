"""SPP-based VAE for variable-size grayscale wireframe images.

Uses Spatial Pyramid Pooling for a fixed-size latent regardless of input height,
MaskedGroupNorm for variable-size support, and CBAM attention blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__all__ = ["VAE"]


class MaskedGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x, mask):
        if mask is None:
            # F.group_norm expects weight of shape (C,)
            return F.group_norm(x, self.num_groups, self.weight.squeeze(), self.bias.squeeze(), self.eps)

        B, C, H, W = x.shape
        G = self.num_groups

        # Resize mask to match feature dimensions
        if mask.shape[2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode='nearest')
        
        # Reshape into groups
        x_g = x.view(B, G, C // G, H, W)
        mask_g = mask.view(B, 1, 1, H, W)  # Broadcast over groups

        # Weighted mean
        x_sum = (x_g * mask_g).sum(dim=[2, 3, 4], keepdim=True)
        mask_sum = mask_g.expand_as(x_g).sum(dim=[2, 3, 4], keepdim=True)
        mean = x_sum / (mask_sum + self.eps)

        # Weighted variance
        var_sum = ((x_g - mean).pow(2) * mask_g).sum(dim=[2, 3, 4], keepdim=True)
        var = var_sum / (mask_sum + self.eps)

        # Normalise + affine
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

        if mask is not None:
            x = x * mask
        
        for size in self.pool_sizes:
            if mask is None:
                pool = F.adaptive_avg_pool2d(x, (size, size))
            else:
                # Corrected masked average: ratio compensates for zero-padding
                x_sum = F.adaptive_avg_pool2d(x, (size, size))
                mask_sum = F.adaptive_avg_pool2d(mask, (size, size))
                pool = x_sum / (mask_sum + 1e-6)
                
            features.append(pool.view(batch_size, -1))
            
        return torch.cat(features, dim=1)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        out = x * channel_att

        # Spatial attention
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_concat))

        del avg_out, max_out, spatial_concat, channel_att
        return out * spatial_att

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        num_groups = min(32, out_c)
        if out_c % num_groups != 0:
            num_groups = out_c

        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.gn1 = MaskedGroupNorm(num_groups, out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.gn2 = MaskedGroupNorm(num_groups, out_c)

        self.cbam = CBAM(out_c)
        
        self.use_projection = (stride != 1 or in_c != out_c)
        if self.use_projection:
            self.shortcut_conv = nn.Conv2d(in_c, out_c, 1, stride, bias=False)
            self.shortcut_gn = MaskedGroupNorm(num_groups, out_c)

    def forward(self, x, mask=None):
        # Ensure mask matches input spatial dims
        if mask is not None and mask.shape[2:] != x.shape[2:]:
            mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')

        out = self.conv1(x)

        # Downsample mask if stride reduced spatial dims
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
            # shortcut_conv has same stride as conv1, so output has size of mask_out
            identity = self.shortcut_gn(identity, mask_out)

        out += identity
        return F.relu(out)

class VAE(nn.Module):
    def __init__(self, config, input_size, num_classes=0):
        super(VAE, self).__init__()
        self.latent_dim = config.get("latent_dim", 128)
        self.spp_levels = [1, 2, 4]
        self.dropout_p = config.get("dropout_p", 0.0)

        # --- Encoder ---
        self.enc_conv1_conv = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.enc_conv1_gn = MaskedGroupNorm(8, 64)
        self.enc_conv1_relu = nn.ReLU(True)
        self.enc_conv1_pool = nn.MaxPool2d(3, 2, 1)
        
        self.enc_block1 = ResidualBlock(64, 128, stride=2)
        self.enc_block2 = ResidualBlock(128, 256, stride=2)
        self.enc_block3 = ResidualBlock(256, 512, stride=2)

        self.spp = MaskedSPPLayer(self.spp_levels)
        self.spp_out_dim = 512 * sum([s * s for s in self.spp_levels])

        self.fc_mu = nn.Linear(self.spp_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.spp_out_dim, self.latent_dim)

        # --- Decoder ---
        self.dec_channels = 512
        self.seed_size = 4  # Initial spatial seed 4x4

        # Lightweight dense projection (512 * 4 * 4 = 8192 outputs)
        self.fc_decode = nn.Linear(self.latent_dim, self.dec_channels * self.seed_size * self.seed_size)
        self.dec_unflatten = nn.Unflatten(1, (self.dec_channels, self.seed_size, self.seed_size))

        if self.dropout_p > 0:
            self.dec_dropout = nn.Dropout2d(self.dropout_p)

        # Progressive channel reduction: 512 -> 256 -> 128 -> 64 -> 32 -> 1
        self.dec_up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_block1 = ResidualBlock(512, 256)

        self.dec_up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_block2 = ResidualBlock(256, 128)

        self.dec_up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_block3 = ResidualBlock(128, 64)

        self.dec_up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_block4 = ResidualBlock(64, 32)

        self.dec_final = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, target_size: Optional[tuple] = None) -> torch.Tensor:
        """Decode latent vector, dynamically adapting to the target image ratio.

        Args:
            z: Latent tensor of shape (B, latent_dim).
            target_size: (Height, Width) of the original input image.
                         If None, defaults to (256, 128).
        """
        if target_size is None:
            target_size = (256, 128)

        # 1. Project to 4x4 seed
        d = self.fc_decode(z)
        d = self.dec_unflatten(d)

        if hasattr(self, 'dec_dropout'):
            d = self.dec_dropout(d)

        # 2. Adapt seed to target aspect ratio (4 upsamples → factor 16)
        base_h = max(1, target_size[0] // 16)
        base_w = max(1, target_size[1] // 16)
        d = F.interpolate(d, size=(base_h, base_w), mode='bilinear', align_corners=False)

        # 3. Standard upsampling path
        d = self.dec_up1(d)
        d = self.dec_block1(d)

        d = self.dec_up2(d)
        d = self.dec_block2(d)

        d = self.dec_up3(d)
        d = self.dec_block3(d)

        d = self.dec_up4(d)
        d = self.dec_block4(d)

        recon = self.dec_final(d)

        # 4. Final adjustment if divisions by 16 were not exact
        if recon.shape[2:] != target_size:
            recon = F.interpolate(recon, size=target_size, mode='bilinear', align_corners=False)

        return recon

    def _encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple:
        """Shared encoder pipeline used by forward() and encode()."""
        e1 = self.enc_conv1_conv(x)
        e1 = self.enc_conv1_gn(e1, mask)
        e1 = self.enc_conv1_relu(e1)
        e1 = self.enc_conv1_pool(e1)

        m1 = F.interpolate(mask, size=e1.shape[2:], mode='nearest') if mask is not None else None

        e2 = self.enc_block1(e1, mask=m1)
        del e1, m1

        m2 = F.interpolate(mask, size=e2.shape[2:], mode='nearest') if mask is not None else None

        e3 = self.enc_block2(e2, mask=m2)
        del e2, m2

        m3 = F.interpolate(mask, size=e3.shape[2:], mode='nearest') if mask is not None else None

        features = self.enc_block3(e3, mask=m3)
        del e3, m3

        m_feat = F.interpolate(mask, size=features.shape[2:], mode='nearest') if mask is not None else None

        pooled = self.spp(features, mask=m_feat)
        del features, m_feat

        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)
        del pooled

        return mu, logvar, z

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.dim() == 4, f"Expected 4D input (B,C,H,W), got {x.dim()}D shape {x.shape}"
        assert x.shape[1] == 1, f"Expected 1 channel, got {x.shape[1]}"
        assert x.shape[0] > 0, "Batch size must be positive"
        
        if mask is not None:
            assert mask.shape == x.shape, f"Mask shape {mask.shape} != input shape {x.shape}"
            assert mask.dtype == torch.float32, f"Mask dtype should be float32, got {mask.dtype}"
        
        orig_size = (x.shape[2], x.shape[3])
        
        mu, logvar, z = self._encode(x, mask)
        recon = self.decode(z, target_size=orig_size)
        
        if mask is not None:
            recon = recon * mask
            
        return recon, mu, logvar
    
    def sample(self, num_samples: int = 1, device: torch.device = None,
               target_size: tuple = (1024, 128)) -> torch.Tensor:
        """Generate wireframes by sampling from N(0, I)."""
        if device is None:
            device = next(self.parameters()).device
        
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z, target_size=target_size)
        return samples

    def encode(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """Encode images to latent space (mu, logvar, z)."""
        return self._encode(x, mask)