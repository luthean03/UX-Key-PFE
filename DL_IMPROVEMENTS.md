# Deep Learning Engineering Recommendations

**Target:** Improve model performance, training stability, and scientific rigor  
**Timeline:** 3-4 weeks of focused optimization

---

## 1️⃣ LATENT SPACE OPTIMIZATION

### Problem: Posterior Collapse Risk

Your model may suffer from "posterior collapse" where the KLD → 0 and latent space is ignored.

**Evidence to check:**
```python
# Add to training loop:
if epoch % 10 == 0:
    with torch.no_grad():
        z_samples = []
        for batch in valid_loader:
            x, _ = batch
            mu, logvar, _, _ = model(x)
            z_samples.append(mu.detach().cpu().numpy())
        
        z_combined = np.vstack(z_samples)
        # Check: should NOT all be zeros/identical
        print(f"Latent std per dim: {z_combined.std(axis=0).mean():.4f}")
        # If < 0.01, posterior collapse likely
```

### Solution: Improved KLD Annealing

**Current (config-vae.yaml):**
```yaml
warmup_epochs: 10
beta_kld: 1.0
```

**Recommended:**
```yaml
# Cycle 1: Ramp up
warmup_epochs: 20
beta_kld: 0.0  # Start at 0 (no KLD penalty)

# Then in code: gradually increase beta over epochs
def get_beta(epoch, total_epochs=200):
    warmup_epochs = 20
    if epoch < warmup_epochs:
        # Linear ramp: 0 → 0.5 over first 20 epochs
        return 0.5 * (epoch / warmup_epochs)
    else:
        # Then exponential approach to 1.0
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 1.0 - 0.5 * np.exp(-3 * progress)

# Usage in loss:
beta = get_beta(epoch)
loss = recon_loss + beta * kld_loss
wandb.log({'training/beta_kld': beta})
```

### Alternative: Conditional VAE (β-VAE)

If posterior collapse persists, try different β values empirically:

```python
# config_beta_sweep.yaml - Run multiple experiments:
experiments:
  - beta_kld: 0.1
  - beta_kld: 0.3
  - beta_kld: 0.5
  - beta_kld: 1.0
  - beta_kld: 3.0  # More regularization

# Then compare latent structure & reconstruction quality
```

---

## 2️⃣ LOSS FUNCTION ENGINEERING

### Current Setup Analysis

Your loss combines:
- **Reconstruction**: SSIM (good for wireframes ✓)
- **Regularization**: KLD (standard ✓)

### Enhancement: Add Perceptual Loss

```python
# src/torchtmpl/loss.py - NEW

class EnhancedVAELoss(nn.Module):
    """VAE loss with perceptual components."""
    
    def __init__(self, beta_kld=1.0, lambda_perceptual=0.1):
        super().__init__()
        self.beta_kld = beta_kld
        self.lambda_perceptual = lambda_perceptual
        
        # Pre-trained feature extractor (lightweight)
        # Use pretrained ResNet or custom CNN for perceptual matching
        self.feature_extractor = self._build_feature_extractor()
    
    def forward(self, recon, target, mu, logvar):
        # 1. Reconstruction (SSIM already good)
        ssim_loss = 1 - ssim(recon, target)
        
        # 2. Perceptual loss (match high-level features)
        feat_recon = self.feature_extractor(recon)
        feat_target = self.feature_extractor(target)
        perceptual_loss = F.mse_loss(feat_recon, feat_target)
        
        # 3. KLD regularization
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Weighted combination
        total_loss = (
            ssim_loss +
            self.lambda_perceptual * perceptual_loss +
            self.beta_kld * kld_loss
        )
        
        return total_loss, {
            'ssim': ssim_loss.item(),
            'perceptual': perceptual_loss.item(),
            'kld': kld_loss.item(),
        }
```

### Recommendation: Curriculum Learning

Train in phases:

**Phase 1 (Epochs 1-50):** Focus on reconstruction
```python
beta_kld = 0.0  # Ignore regularization
lambda_perceptual = 0.0
```

**Phase 2 (Epochs 51-150):** Balance reconstruction & regularization
```python
beta_kld = 0.5  # Gradually increase
lambda_perceptual = 0.05
```

**Phase 3 (Epochs 151-200):** Full loss
```python
beta_kld = 1.0
lambda_perceptual = 0.1
```

---

## 3️⃣ ARCHITECTURE IMPROVEMENTS

### Current Model: ✅ Good, ⚠️ Can Improve

**Current strengths:**
- SPP handles variable sizes
- CBAM attention is modern
- ResBlocks are proven

### Recommendation 1: Adversarial Learning (GAN-VAE Hybrid)

Add discriminator to improve sample quality:

```python
class DiscriminatorVAE(nn.Module):
    """Lightweight discriminator for VAE-GAN."""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 256 → 128
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),  # 128 → 64
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Training loss becomes:
# L_total = L_VAE + λ * L_adversarial
```

**Pros:** Higher-quality samples  
**Cons:** Harder to train, more hyperparams  
**Recommendation:** Try AFTER base VAE is working well

### Recommendation 2: Hierarchical VAE (Skip Connections)

You've disabled skip connections. Should investigate:

```python
# Experiment: Enable skip connections with warm-up
def train_with_skip_warmup(model, epochs):
    for epoch in range(epochs):
        if epoch < 50:
            # Warm-up phase: freeze skip path
            model.use_skip_connections = False
        else:
            # Main phase: enable skips
            model.use_skip_connections = True
        
        train_epoch(model)
```

### Recommendation 3: Dense Connections (DenseNet-VAE)

Instead of ResNet blocks, try DenseNet:

```python
class DenseBlock(nn.Module):
    """Dense block from DenseNet (all layers connected)."""
    
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_c = in_channels + i * growth_rate
            out_c = growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.GroupNorm(16, in_c),
                    nn.ReLU(),
                    nn.Conv2d(in_c, out_c, 3, 1, 1)
                )
            )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)
```

**Why:** DenseNet may preserve more information for reconstruction

---

## 4️⃣ TRAINING OPTIMIZATION

### Current Setup Analysis

✅ Good: Gradient accumulation, AMP, SmartBatching  
⚠️ Needs: Proper tuning, profiling

### Recommendation 1: Learning Rate Scheduling

**Current:**
```yaml
scheduler:
  name: "CosineAnnealingWarmRestarts"
  params:
    T_0: 10
    T_mult: 2
```

**Enhanced:**
```yaml
scheduler:
  name: "CosineAnnealingWarmRestarts"
  params:
    T_0: 20          # Longer initial period
    T_mult: 1.5      # More moderate restart
    eta_min: 1e-7    # Smaller floor
    
# PLUS: Learning rate warmup (first 5 epochs)
optimizer:
  algo: "AdamW"
  params:
    lr: 0.0003
  # Add warmup:
  warmup:
    enabled: true
    epochs: 5
    strategy: "linear"  # or "exponential"
```

### Recommendation 2: Batch Size Tuning

Test different batch sizes (GPU memory vs convergence):

```python
# config_batch_sweep.yaml
experiments:
  - batch_size: 8    # Good for gradient variance
  - batch_size: 16   # Current (balanced)
  - batch_size: 32   # Faster but more memory
  - batch_size: 64   # Very fast, risky stability

# Monitor: loss smoothness, final SSIM score
```

### Recommendation 3: Gradient Clipping

Prevent exploding gradients:

```python
# In training loop:
optimizer.zero_grad()
loss.backward()

# Clip gradient norms
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
scheduler.step()

# Log gradient stats:
grad_norm = 0
for p in model.parameters():
    if p.grad is not None:
        grad_norm += p.grad.data.norm(2).item() ** 2
grad_norm = grad_norm ** 0.5
wandb.log({'training/grad_norm': grad_norm})
```

---

## 5️⃣ DATA AUGMENTATION STRATEGIES

### Current Augmentations: Good, Can Improve

**Current (config-vae.yaml):**
```yaml
augment: true
sp_prob: 0.04              # Salt & pepper
random_erasing_prob: 0.35  # Masking
rotation_degrees: 5        # Light rotation
perspective_p: 0.3
brightness_jitter: 0.1
contrast_jitter: 0.1
```

### Recommendation 1: AutoAugment

Use learned augmentation policies:

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

augmentation = AutoAugment(
    policy=AutoAugmentPolicy.IMAGENET
)
```

### Recommendation 2: Mixup for VAE

Mix latent codes in addition to images:

```python
def mixup_latent(z1, z2, alpha=0.2):
    """Mixup in latent space (after encoder)."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    z_mixed = lam * z1 + (1 - lam) * z2
    return z_mixed, lam

# Use in training:
z1 = model.encode(x1)
z2 = model.encode(x2)
z_mixed, lam = mixup_latent(z1, z2)
recon = model.decode(z_mixed)

# Loss should weight both x1 and x2
```

### Recommendation 3: Domain Randomization

Add synthetic noise realistic to wireframes:

```python
class WireframeAugmentation(nn.Module):
    """Augmentations realistic for wireframes."""
    
    def forward(self, img):
        # 1. Add lines/boxes (wireframe elements)
        if random.random() < 0.3:
            img = self.add_random_line(img)
        
        # 2. Add text blocks (wireframe text)
        if random.random() < 0.2:
            img = self.add_random_box(img)
        
        # 3. Color jitter (to grayscale)
        if random.random() < 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
        
        return img
```

---

## 6️⃣ EVALUATION METRICS

### Current: Limited Metrics

You track SSIM, but need more:

### Recommendation: Comprehensive Metrics

```python
class VAEMetrics:
    """Comprehensive VAE evaluation metrics."""
    
    @staticmethod
    def reconstruction_quality(recon, target):
        """MSE, SSIM, LPIPS, FID."""
        mse = F.mse_loss(recon, target).item()
        ssim_val = ssim(recon, target).item()
        # LPIPS = Learned Perceptual Image Patch Similarity
        lpips_val = compute_lpips(recon, target)
        return {'mse': mse, 'ssim': ssim_val, 'lpips': lpips_val}
    
    @staticmethod
    def latent_quality(z_samples):
        """Measure latent space quality."""
        # 1. Variance per dimension
        var = z_samples.var(axis=0).mean()
        
        # 2. Kullback-Leibler divergence from N(0,I)
        mean = z_samples.mean(axis=0)
        std = z_samples.std(axis=0)
        kl_approx = -0.5 * np.mean(1 + np.log(std ** 2) - mean ** 2 - std ** 2)
        
        # 3. Correlation between dims (should be ~0)
        corr = np.abs(np.corrcoef(z_samples.T)).mean()
        
        return {'variance': var, 'kl_approx': kl_approx, 'correlation': corr}
    
    @staticmethod
    def interpolation_quality(model, z1, z2):
        """Quality of interpolation path."""
        recon_path = []
        for alpha in np.linspace(0, 1, 10):
            z_interp = slerp(z1, z2, alpha)
            recon = model.decode(z_interp)
            recon_path.append(recon)
        
        # Smooth path should have low variance in SSIM changes
        ssims = [ssim(recon_path[i], recon_path[i+1]) for i in range(len(recon_path)-1)]
        smoothness = np.std(ssims)  # Lower is better
        
        return smoothness

# Tracking in training:
def validate_epoch(model, val_loader):
    metrics_list = []
    
    for x, _ in val_loader:
        recon = model(x)[3]
        
        recon_metrics = VAEMetrics.reconstruction_quality(recon, x)
        
        metrics_list.append(recon_metrics)
    
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) 
                   for k in metrics_list[0].keys()}
    
    wandb.log({f'validation/{k}': v for k, v in avg_metrics.items()})
    
    return avg_metrics
```

---

## 7️⃣ EXPERIMENTAL VALIDATION

### Recommended Experiments (Priority Order)

**Experiment 1: Posterior Collapse Check (Week 1)**
- Run current model
- Measure KLD, latent variance
- If problematic, implement improved annealing

**Experiment 2: Loss Component Ablation (Week 1-2)**
```
- Baseline: SSIM + KLD
- +Gradient: SSIM + Gradient + KLD
- +Perceptual: SSIM + Gradient + Perceptual + KLD
- +Combined: All components with learned weights
```

**Experiment 3: Architecture Comparison (Week 2-3)**
```
- ResNet-VAE (current)
- DenseNet-VAE
- DenseNet-VAE + Skip Connections
- DenseNet-VAE + Attention
```

**Experiment 4: Hyperparameter Grid (Week 3-4)**
```
- beta_kld: [0.1, 0.5, 1.0, 3.0]
- latent_dim: [32, 64, 128, 256]
- batch_size: [8, 16, 32]
- lr: [1e-4, 3e-4, 1e-3]
```

### Tracking with MLflow

```python
import mlflow

mlflow.set_experiment("VAE-Optimization")

with mlflow.start_run(run_name="run-v1-baseline"):
    mlflow.log_params({
        'beta_kld': 1.0,
        'latent_dim': 128,
        'batch_size': 16,
    })
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(...)
        val_metrics = validate_epoch(...)
        
        mlflow.log_metrics(val_metrics, step=epoch)
    
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifact("config/config-vae.yaml")
```

---

## 8️⃣ DEPLOYMENT CONSIDERATIONS

### Model Quantization

For inference on-device:

```python
# Quantize for mobile
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Size reduction: typically 4x smaller
print(f"Original: {model_size_mb:.1f} MB")
print(f"Quantized: {quantized_model_size_mb:.1f} MB")
```

### ONNX Export

```python
import torch.onnx

dummy_input = torch.randn(1, 1, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "vae_model.onnx",
    export_params=True,
    opset_version=14,
    input_names=['image'],
    output_names=['mu', 'logvar', 'z', 'reconstruction']
)
```

---

## SUMMARY TABLE

| Improvement | Priority | Effort | Impact | Timeline |
|------------|----------|--------|--------|----------|
| Fix posterior collapse | 🔴 HIGH | 2h | 🔴 HIGH | Week 1 |
| Add gradient clipping | 🔴 HIGH | 1h | 🟡 MEDIUM | Week 1 |
| Comprehensive metrics | 🔴 HIGH | 3h | 🔴 HIGH | Week 1-2 |
| Experiment tracking (MLflow) | 🟠 MEDIUM | 2h | 🔴 HIGH | Week 1 |
| Improved LR scheduling | 🟠 MEDIUM | 2h | 🟡 MEDIUM | Week 2 |
| Augmentation expansion | 🟠 MEDIUM | 3h | 🟡 MEDIUM | Week 2 |
| Architecture experiments | 🟠 MEDIUM | 8h | 🟡 MEDIUM | Week 2-3 |
| GAN-VAE hybrid | 🟡 LOW | 6h | 🟢 LOW | Week 4+ |
| Quantization/ONNX | 🟡 LOW | 4h | 🟢 LOW | Week 4+ |

---

**Next Step:** Start with Posterior Collapse diagnosis (Week 1)
