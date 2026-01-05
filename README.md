# Rapport Technique - VAE pour Wireframes UI/UX

**Date:** 5 janvier 2026  
**Projet:** UX-Key-PFE - Variational AutoEncoder pour la génération et reconstruction de wireframes web

---

## Table des Matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Pipeline de génération des données](#2-pipeline-de-génération-des-données)
3. [Architecture du dataset et chargement](#3-architecture-du-dataset-et-chargement)
4. [Augmentations de données](#4-augmentations-de-données)
5. [Architecture du modèle VAE](#5-architecture-du-modèle-vae)
6. [Fonctions de perte (Loss)](#6-fonctions-de-perte-loss)
7. [Optimisation et entraînement](#7-optimisation-et-entraînement)
8. [Monitoring et métriques](#8-monitoring-et-métriques)
9. [Déploiement sur cluster SLURM](#9-déploiement-sur-cluster-slurm)
10. [Configuration complète](#10-configuration-complète)

---

## 1. Vue d'ensemble du projet

### Objectif
Développer un **Variational AutoEncoder (VAE)** capable de reconstruire et générer des wireframes de pages web à partir de données JSON représentant des layouts UI/UX. Le modèle doit gérer des images de tailles variables (hauteurs jusqu'à 2048px) et apprendre une représentation latente compressée des structures de wireframes.

### Défis principaux
- **Tailles d'images variables** : Les wireframes peuvent avoir des hauteurs très différentes (pages courtes vs longues pages web)
- **Détails fins** : Lignes minces et structures précises à préserver
- **Mémoire GPU limitée** : Nécessité d'optimiser pour les vieux GPU (CUDA 11.8, PyTorch 2.1.2)
- **Batch size = 1** : Utilisation d'accumulation de gradients pour simuler des batchs plus grands

### Stack technologique
- **Framework**: PyTorch 2.1.2 + CUDA 11.8
- **Vision**: torchvision 0.16.2
- **Métriques**: torchmetrics (SSIM)
- **Visualisation**: TensorBoard, Wandb (optionnel)
- **Cluster**: SLURM (partition gpu_prod_long, 48h max)

---

## 2. Pipeline de génération des données

### Script: `img_vae_new.py`

Le script convertit des fichiers JSON (représentant des wireframes) en images PNG niveaux de gris.

#### Formats JSON supportés

**Format ancien** (compact):
```json
{
  "r": { "b": [x, y, w, h], "c": [...] },
  "w": 1200,
  "h": 3000
}
```

**Format nouveau** (verbeux):
```json
{
  "bounds": { "x": 0, "y": 0, "width": 1200, "height": 3000 },
  "children": [...]
}
```

#### Algorithme de peinture additive

```python
def paint_additive_recursive(node, canvas, parent_x, parent_y, width, height):
    """
    Chaque nœud UI ajoute +1 au canvas (superposition des couches)
    - Plus une zone a d'éléments empilés, plus elle sera claire
    - Les feuilles de l'arbre sont moins claires que les conteneurs
    """
    canvas[y_start:y_end, x_start:x_end] += 1.0
    for child in children:
        paint_additive_recursive(child, ...)
```

#### Post-traitement

1. **Normalisation linéaire** : `img_linear = canvas / max_depth`
   - Valeurs entre [0, 1]
   - Sauvegarde: `{name}_linear.png` (utilisé pour l'entraînement)

2. **Gamma encoding** : `img_contrast = img_linear^2.0`
   - Améliore le contraste visuel pour inspection humaine
   - Sauvegarde: `{name}_visu.png` (visualisation uniquement)

#### Sortie
- Dossier: `vae_dataset/` (ou configurable)
- Format: PNG 8-bit grayscale
- Dimensions: Variables (W fixe ~1200px, H jusqu'à 2048px)

---

## 3. Architecture du dataset et chargement

### Classe: `VariableSizeDataset`

**Fichier**: `src/torchtmpl/data.py`

#### Caractéristiques principales

```python
class VariableSizeDataset(Dataset):
    def __init__(self, 
                 root_dir,                    # Dossier des PNG
                 noise_level=0.0,             # Bruit gaussien (denoising)
                 max_height=2048,             # Limite de hauteur (crop)
                 augment=False,               # Active/désactive augmentations
                 files_list=None,             # Liste explicite de fichiers
                 sp_prob=0.02,                # Prob salt-and-pepper
                 perspective_p=0.3,           # Prob perspective transform
                 perspective_distortion_scale=0.08,
                 random_erasing_prob=0.5):    # Prob random erasing
```

#### Processus de chargement (`__getitem__`)

1. **Chargement et conversion**
   ```python
   clean_image = Image.open(img_path).convert('L')  # Grayscale
   ```

2. **Crop intelligent (gestion OOM)**
   ```python
   if h > self.max_height:
       if self.augment:  # Train: crop aléatoire
           top = random.randint(0, h - self.max_height)
       else:             # Valid: crop centré (déterministe)
           top = (h - self.max_height) // 2
       clean_image = clean_image.crop((0, top, w, top + self.max_height))
   ```

3. **Augmentations** (si `augment=True`, voir section 4)

4. **Ajout de bruit (denoising)**
   ```python
   if self.noise_level > 0.0:
       noise = torch.randn_like(clean_tensor) * self.noise_level
       noisy_tensor = torch.clamp(clean_tensor + noise, 0.0, 1.0)
   ```

5. **Retour**
   ```python
   return noisy_tensor, clean_tensor  # (input, target)
   ```

### Fonction: `get_dataloaders`

**Responsabilités**:
- Split train/validation (ratio 80/20 par défaut, configurable)
- Création de deux datasets avec paramètres différents
- Configuration des DataLoaders avec seeding reproductible

```python
train_dataset = VariableSizeDataset(
    augment=True,      # Augmentations activées
    noise_level=0.15,  # Bruit pour denoising
    files_list=train_files
)

valid_dataset = VariableSizeDataset(
    augment=False,     # Pas d'augmentations
    noise_level=0.0,   # Pas de bruit (eval sur clean data)
    files_list=valid_files
)
```

**Paramètres DataLoader**:
- `batch_size=1` (requis pour tailles variables + SPP)
- `num_workers=8` (parallélisation I/O)
- `pin_memory=True` (si GPU disponible)
- `generator` avec seed fixe (reproductibilité)

---

## 4. Augmentations de données

### Philosophie: "Web-Safe Augmentations"

Les augmentations sont conçues pour simuler des variations réalistes de wireframes sans détruire la structure sémantique.

### Augmentations implémentées

#### 4.1 RandomPerspective
**Probabilité**: 30% (`perspective_p=0.3`)  
**Distortion**: 0.08 (`perspective_distortion_scale`)

```python
T.RandomPerspective(distortion_scale=0.08, p=0.3)
```

**But**: Simule des variations de rendu navigateur, perspectives légères

#### 4.2 Bruit Gaussien (Denoising)
**Niveau**: 0.15 (`noise_level`)

```python
noise = torch.randn_like(clean_tensor) * 0.15
noisy_tensor = torch.clamp(clean_tensor + noise, 0.0, 1.0)
```

**But**: Force le modèle à apprendre des features robustes au bruit

#### 4.3 RandomErasing (Inpainting)
**Probabilité**: 20% (`random_erasing_prob=0.2`)  
**Zone effacée**: 2-10% de l'image  
**Aspect ratio**: 0.3 à 3.3

```python
if random.random() < 0.2:
    eraser = T.RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
    noisy_tensor = eraser(noisy_tensor)
```

**But**: Robustesse au masquage, capacité d'inpainting

#### 4.4 Salt-and-Pepper Noise
**Probabilité**: 50% d'activation, puis 2% des pixels (`sp_prob=0.02`)

```python
mask = torch.rand_like(noisy_tensor) < 0.02
rnd = torch.rand_like(noisy_tensor)
noisy_tensor[mask & (rnd < 0.5)] = 0.0  # Salt (blanc)
noisy_tensor[mask & (rnd >= 0.5)] = 1.0 # Pepper (noir)
```

**But**: Simule des artefacts de compression, pixels morts

### Pipeline complet (ordre d'application)

1. **Crop** (si `h > max_height`)
2. **RandomPerspective** (sur PIL Image)
3. **Conversion en Tensor**
4. **Bruit Gaussien**
5. **RandomErasing** (sur Tensor)
6. **Salt-and-Pepper** (sur Tensor)

### Configuration validation

**Important**: Le dataset de validation a **toutes les augmentations désactivées**:
```python
valid_dataset = VariableSizeDataset(
    noise_level=0.0,          # Pas de bruit
    augment=False,            # Pas d'augmentations géométriques
    sp_prob=0.0,              # Pas de salt-and-pepper
    random_erasing_prob=0.0   # Pas d'erasing
)
```

---

## 5. Architecture du modèle VAE

### Fichier: `src/torchtmpl/models/vae_models.py`

### Vue d'ensemble

Le modèle utilise une architecture **VAE avec Spatial Pyramid Pooling (SPP)** et **blocs résiduels avec attention CBAM**.

```
Input (1, H, W) → Encoder → SPP → Latent (128D) → Decoder → Output (1, H, W)
                              ↓
                          μ, log(σ²)
```

### Composants clés

#### 5.1 SPPLayer (Spatial Pyramid Pooling)

**Problème résolu**: Tailles d'images variables  
**Solution**: Pooling adaptatif à plusieurs échelles

```python
class SPPLayer(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        self.pools = [AdaptiveAvgPool2d(s) for s in pool_sizes]
    
    def forward(self, x):
        # x: (B, 512, H_feat, W_feat) avec H_feat variable
        features = []
        for pool in self.pools:
            features.append(pool(x).view(batch_size, -1))
        # Concaténation: (B, 512*(1*1 + 2*2 + 4*4)) = (B, 512*21) = (B, 10752)
        return torch.cat(features, dim=1)
```

**Niveaux**: [1x1, 2x2, 4x4] → 21 grilles → 512×21 = 10752 features

#### 5.2 CBAM (Convolutional Block Attention Module)

**Double attention**: Channel + Spatial

```python
class CBAM(nn.Module):
    def forward(self, x):
        # 1. Channel Attention: "Quels features sont importants ?"
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = x * sigmoid(avg_out + max_out)
        
        # 2. Spatial Attention: "Où sont les zones importantes ?"
        avg_spatial = mean(out, dim=1)
        max_spatial = max(out, dim=1)
        spatial_weights = sigmoid(conv7x7(cat([avg_spatial, max_spatial])))
        
        return out * spatial_weights
```

**Réduction**: 16 (canaux divisés par 16 dans le bottleneck)

#### 5.3 ResidualBlock

Bloc ResNet amélioré avec **GroupNorm** (compatible batch_size=1) et **CBAM**

```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        out = ReLU(GroupNorm(Conv3x3(x)))
        out = GroupNorm(Conv3x3(out))
        out = CBAM(out)              # Attention avant skip connection
        out += Shortcut(x)
        return ReLU(out)
```

**GroupNorm**: 32 groupes (vs BatchNorm qui nécessite batch_size > 1)

#### 5.4 Encoder

```python
self.encoder = nn.Sequential(
    Conv2d(1, 64, kernel=7, stride=2, padding=3),  # Downscale /2
    GroupNorm(32, 64),
    ReLU(),
    MaxPool2d(3, stride=2),                        # Downscale /2
    ResidualBlock(64, 128, stride=2),              # Downscale /2
    ResidualBlock(128, 256, stride=2),             # Downscale /2
    ResidualBlock(256, 512, stride=2),             # Downscale /2
)
# Total downscale: 2^5 = 32x
# Input: (1, 2048, W) → Features: (512, 64, W/32)
```

**SPP + Latent projection**:
```python
features = self.encoder(x)              # (B, 512, H_feat, W_feat)
pooled = self.spp(features)             # (B, 10752)
mu = self.fc_mu(pooled)                 # (B, 128)
logvar = self.fc_logvar(pooled)         # (B, 128)
```

#### 5.5 Reparameterization Trick

```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)       # σ = exp(0.5 * log(σ²))
    eps = torch.randn_like(std)         # ε ~ N(0, 1)
    return mu + eps * std                # z = μ + σ * ε
```

**Gradient flow**: Les gradients passent par `mu` et `logvar`, pas par le sampling aléatoire

#### 5.6 Decoder

```python
self.fc_decode = Linear(128, 256*64*4)  # Latent → Features initiales

self.decoder = nn.Sequential(
    Unflatten(1, (256, 64, 4)),         # (B, 128) → (B, 256, 64, 4)
    Upsample(scale=2),                  # (B, 256, 128, 8)
    ResidualBlock(256, 128),
    Upsample(scale=2),                  # (B, 128, 256, 16)
    ResidualBlock(128, 64),
    Upsample(scale=2),                  # (B, 64, 512, 32)
    ResidualBlock(64, 32),
    Upsample(scale=2),                  # (B, 32, 1024, 64)
    Conv2d(32, 1, kernel=3, padding=1),
    Sigmoid()                           # Output ∈ [0, 1]
)
```

**Interpolation finale**:
```python
recon_small = self.decoder(z)           # (B, 1, 1024, W_small)
recon = F.interpolate(recon_small, 
                      size=(orig_h, orig_w), 
                      mode='bilinear', 
                      align_corners=False)
```

**Aspect ratio préservé**: Le decoder génère une forme allongée (64×4) pour mieux correspondre aux pages web verticales

### Paramètres totaux
- **Latent dimension**: 128
- **Encoder channels**: 64 → 128 → 256 → 512
- **Decoder channels**: 256 → 128 → 64 → 32 → 1
- **SPP output**: 10,752 features

---

## 6. Fonctions de perte (Loss)

### Fichier: `src/torchtmpl/loss.py`

### 6.1 SimpleVAELoss (Baseline)

**Modes**: `l1` ou `mse`

```python
class SimpleVAELoss(nn.Module):
    def forward(self, recon_x, x, mu, logvar):
        # Reconstruction term
        recon_loss = F.l1_loss(recon_x, x, reduction='sum')
        
        # KL Divergence: KL(q(z|x) || p(z))
        # p(z) = N(0, I), q(z|x) = N(μ, σ²)
        kld_loss = -0.5 * sum(1 + logvar - mu² - exp(logvar))
        
        # ELBO = -Recon + β*KLD (on minimise)
        total = recon_loss + β * kld_loss
        return total, recon_loss, kld_loss
```

**Formule KLD**:
$$
\text{KL} = -\frac{1}{2} \sum_{i=1}^{D} (1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2)
$$

### 6.2 VGGPerceptualLoss (Avancé)

**Motivation**: Pour les wireframes (lignes fines), une loss pixel uniquement peut donner des résultats flous. La loss perceptuelle VGG encourage des features visuelles de plus haut niveau.

```python
class VGGPerceptualLoss(nn.Module):
    def __init__(self, beta=1.0, perceptual_weight=0.1):
        # VGG16 pré-entraîné (ImageNet)
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        
        # Extraction de features à différentes profondeurs
        self.blocks = [
            vgg[:4],    # relu1_2 (shallow features)
            vgg[4:9],   # relu2_2
            vgg[9:16],  # relu3_3
            vgg[16:23]  # relu4_3 (deep features)
        ]
        
    def forward(self, recon_x, x, mu, logvar):
        # 1. Pixel-level loss (ancrage structurel)
        pixel_loss = F.l1_loss(recon_x, x, reduction='sum')
        
        # 2. Perceptual loss (features VGG)
        # Conversion 1-channel → 3-channels + normalisation ImageNet
        x_norm = normalize_imagenet(repeat_channels(x))
        recon_norm = normalize_imagenet(repeat_channels(recon_x))
        
        loss_feat = 0.0
        for block in self.blocks:
            x_feat = block(x_feat)
            recon_feat = block(recon_feat)
            loss_feat += F.l1_loss(recon_feat, x_feat, reduction='mean')
        
        # Rescale (mean → sum)
        vgg_term = loss_feat * scale_factor * perceptual_weight
        
        # 3. KL Divergence
        kld_loss = -0.5 * sum(1 + logvar - mu² - exp(logvar))
        
        # Total
        total = pixel_loss + vgg_term + β * kld_loss
        return total, pixel_loss, kld_loss
```

**Normalisation ImageNet**:
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
x_norm = (x_rgb - mean) / std
```

### Factory

```python
def get_vae_loss(loss_config):
    name = loss_config['name']  # 'l1', 'mse', 'perceptual'
    beta = loss_config.get('beta_kld', 0.001)
    
    if name in ('l1', 'mse'):
        return SimpleVAELoss(mode=name, beta=beta)
    if name == 'perceptual':
        perceptual_weight = loss_config.get('perceptual_weight', 0.05)
        return VGGPerceptualLoss(beta=beta, perceptual_weight=perceptual_weight)
```

### Comparaison

| Loss | Avantages | Inconvénients | Cas d'usage |
|------|-----------|---------------|-------------|
| **L1** | Simple, rapide, bon ancrage pixel | Peut donner des résultats flous sur détails fins | Baseline, convergence rapide |
| **MSE** | Pénalise fortement grandes erreurs | Favorise la moyenne (flou) | Rarement utilisé pour images |
| **Perceptual** | Préserve structures visuelles, meilleur sur détails | Plus lent (forward VGG), tuning délicat | Production, qualité maximale |

---

## 7. Optimisation et entraînement

### Fichier: `src/torchtmpl/main.py`

### 7.1 Optimiseur

**Algorithme**: Adam  
**Configuration**:
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,           # Learning rate bas (VAE sensibles)
    weight_decay=0.00001 # L2 regularization légère
)
```

### 7.2 Learning Rate Scheduler

**Type**: ReduceLROnPlateau (adaptatif)

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # Minimise la loss
    factor=0.5,          # LR = LR * 0.5 lors d'un plateau
    patience=5,          # Attend 5 epochs sans amélioration
    verbose=True,        # Logs les changements
    min_lr=0.00001       # Plancher minimum
)
```

**Utilisation**:
```python
scheduler.step(test_loss)  # Met à jour le LR basé sur valid loss
```

### 7.3 Gradient Accumulation

**Problème**: `batch_size=1` → bruit dans les gradients  
**Solution**: Accumuler les gradients sur N images avant `optimizer.step()`

```python
accumulation_steps = 128  # Batch virtuel de 128 images

optimizer.zero_grad()
for i, (inputs, targets) in enumerate(train_loader):
    recon, mu, logvar = model(inputs)
    loss, _, _ = criterion(recon, targets, mu, logvar)
    
    # Normalise la loss par le nombre d'étapes
    (loss / accumulation_steps).backward()
    
    # Met à jour les poids tous les N pas
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

**Gradient Clipping**: `max_norm=1.0` (évite les explosions de gradients)

### 7.4 KL Annealing

**Problème**: Si β est trop élevé dès le début, le modèle ignore la reconstruction et collapse vers prior  
**Solution**: Augmenter progressivement β de 0 → β_target

```python
warmup_epochs = 20
target_beta = 0.001

for epoch in range(nepochs):
    if epoch < warmup_epochs:
        current_beta = target_beta * (epoch / warmup_epochs)
    else:
        current_beta = target_beta
    
    criterion.beta = current_beta  # Met à jour le coefficient KL
```

**Exemple**:
- Epoch 0: β = 0.000
- Epoch 10: β = 0.0005
- Epoch 20+: β = 0.001 (stable)

### 7.5 Boucle d'entraînement complète

```python
for epoch in range(100):
    # === TRAIN ===
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        recon, mu, logvar = model(inputs)
        total_loss, recon_loss, kld_loss = criterion(recon, targets, mu, logvar)
        
        (total_loss / grad_accumulation_steps).backward()
        
        if (i + 1) % grad_accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
    
    # === VALIDATION ===
    model.eval()
    with torch.no_grad():
        for inputs, targets in valid_loader:
            recon, mu, logvar = model(inputs)
            loss, _, _ = criterion(recon, targets, mu, logvar)
            # Accumule loss + SSIM
    
    # === CHECKPOINT ===
    if test_loss < best_loss:
        torch.save(model.state_dict(), 'best_model.pt')
    
    # === SCHEDULER ===
    scheduler.step(test_loss)
```

---

## 8. Monitoring et métriques

### 8.1 Métriques suivies

#### ELBO (Evidence Lower Bound)
```python
ELBO = -Reconstruction_Loss + β * KL_Divergence
```
- **train_ELBO**: Sur données augmentées + bruitées
- **test_ELBO**: Sur données clean (validation)

#### SSIM (Structural Similarity Index)
```python
from torchmetrics.image import StructuralSimilarityIndexMeasure

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
ssim_score = ssim_metric(recon, target)  # ∈ [-1, 1], 1 = identique
```

**Avantages**: Meilleure corrélation avec perception humaine que MSE/PSNR

#### Learning Rate
```python
current_lr = optimizer.param_groups[0]['lr']
```

Logged à chaque epoch pour détecter les plateaux

### 8.2 TensorBoard

**Activation**: `logging.tensorboard: True` dans config

```python
writer = SummaryWriter(log_dir=logdir)

# Scalars
writer.add_scalar('train_ELBO', train_loss, epoch)
writer.add_scalar('test_ELBO', test_loss, epoch)
writer.add_scalar('test_SSIM', avg_ssim, epoch)
writer.add_scalar('kl_beta', current_beta, epoch)
writer.add_scalar('Learning Rate', current_lr, epoch)

# Images (reconstructions)
comparison = torch.cat([target, recon])
writer.add_image('reconstructions', make_grid(comparison, nrow=1), epoch)

# Text
writer.add_text('summary', summary_text)
```

**Visualisation**:
```bash
tensorboard --logdir=./logs
```

### 8.3 Wandb (optionnel)

**Configuration**:
```yaml
logging:
  wandb:
    project: "ux-vae"
    entity: "your-team"
```

**Logs identiques** à TensorBoard, avec interface cloud

### 8.4 Checkpointing

```python
class ModelCheckpoint:
    def update(self, score):
        if score < self.best_score:  # min_is_better
            torch.save(model.state_dict(), savepath)
            self.best_score = score
            return True
        return False
```

**Fichiers sauvegardés**:
- `best_model.pt`: Meilleur modèle (test_loss minimale)
- `last_model.pt`: Dernier epoch (pour reprise)
- `config.yaml`: Configuration exacte utilisée
- `summary.txt`: Architecture + hyperparamètres

### 8.5 Visualisations générées

#### Reconstructions
```
logs/VAE_X/reconstruction_epoch_{0,5,10,...,99}.png
```
Format: [Target | Reconstruction] côte-à-côte

#### Latent Space (t-SNE)
```python
# Extraction de μ pour 1000 images de validation
latents = []
for inputs, _ in valid_loader:
    _, mu, _ = model(inputs)
    latents.append(mu.cpu().numpy())

# Projection 2D
from sklearn.manifold import TSNE
z_embedded = TSNE(n_components=2).fit_transform(latents)

plt.scatter(z_embedded[:, 0], z_embedded[:, 1])
plt.savefig('latent_space_tsne.png')
```

**Interprétation**: Clusters = types de layouts similaires

---

## 9. Déploiement sur cluster SLURM

### Fichier: `job.sbatch`

### 9.1 Configuration SLURM

```bash
#SBATCH --job-name=vae-ux-key
#SBATCH --nodes=1
#SBATCH --exclude=dani[01-17],tx[00-16],sh[10-19],sh00  # Vieux GPU exclus
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1  # Job unique (pas d'array)
```

### 9.2 Pipeline d'exécution

#### Étape 1: Copie du code vers $TMPDIR

```bash
# Exclusions importantes (gain de temps)
rsync -r --exclude logslurms \
         --exclude configs \
         --exclude archetypes \
         --exclude archetypes_png \
         --exclude vae_dataset \  # Dataset reste sur stockage partagé
         . $TMPDIR/code
```

**Raison**: Le dataset (vae_dataset/) reste sur le stockage partagé pour éviter la copie (~GB de PNG)

#### Étape 2: Checkout du commit exact

```bash
cd $TMPDIR/code
git checkout c90547c808fecfbbf38fdcf5cece2957f89bff99
```

**Reproductibilité**: Garantit le même code pour tous les runs

#### Étape 3: Environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate

# PyTorch compatible vieux GPU (CUDA 11.8)
pip install --index-url https://download.pytorch.org/whl/cu118 \
            torch==2.1.2+cu118 torchvision==0.16.2+cu118

# Dépendances du projet
pip install .
```

**Contraintes**:
- `numpy<2` (compatibilité PyTorch 2.1)
- CUDA 11.8 (support vieux GPU)

#### Étape 4: Optimisation mémoire

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Effet**: Allocation dynamique de la mémoire GPU (réduit fragmentation)

#### Étape 5: Génération config dynamique

```bash
CONFIG_IN="$TMPDIR/code/input_config.yaml"
cat > "$CONFIG_IN" << 'EOFCONFIG'
data:
  data_dir: "/usr/users/sdim/sdim_31/UX-Key-PFE/vae_dataset"
  batch_size: 1
  # ... (reste du config)
EOFCONFIG
```

**Flexibilité**: Permet de modifier config via variables d'environnement

#### Étape 6: Lancement training

```bash
python -m torchtmpl.main "$CONFIG_IN" train
```

### 9.3 Script de soumission: `submit-slurm.py`

**Utilisation**:
```bash
python submit-slurm.py config/config-vae.yaml [train|test] [--dry-run] [--allow-dirty]
```

**Features**:
- Vérifie l'état Git (empêche commits non-poussés par défaut)
- Encode le config en base64 pour passage au job
- Génère dynamiquement le script SBATCH
- Soumet via `sbatch`

**Exemple**:
```bash
python submit-slurm.py config/config-vae.yaml train
# → Soumet job.sbatch avec config embedded
# → SLURM job ID: 127440
# → Logs: logslurms/slurm-127440_1.{out,err}
```

---

## 10. Configuration complète

### Fichier: `config/config-vae.yaml`

```yaml
# ==================== DATA ====================
data:
  data_dir: "/usr/users/sdim/sdim_31/UX-Key-PFE/vae_dataset"
  batch_size: 1                  # Requis pour tailles variables
  num_workers: 8                 # Parallélisation I/O
  valid_ratio: 0.2               # 80/20 train/valid split
  seed: 42                       # Reproductibilité
  
  # Gestion mémoire
  max_height: 2048               # Limite hauteur (crop si dépasse)
  
  # Denoising
  noise_level: 0.15              # Bruit gaussien σ=0.15
  
  # Augmentations (train uniquement)
  augment: true
  sp_prob: 0.02                  # Salt-and-pepper (2% pixels)
  random_erasing_prob: 0.2       # Random erasing (20% images)
  perspective_p: 0.3             # Perspective transform (30%)
  perspective_distortion_scale: 0.08

# ==================== MODEL ====================
model:
  name: "vae"
  class: "VAE"                   # Doit matcher classe dans vae_models.py
  latent_dim: 128                # Dimension espace latent
  input_channels: 1              # Grayscale

# ==================== LOSS ====================
loss:
  name: "perceptual"             # 'l1', 'mse', 'perceptual'
  perceptual_weight: 0.01        # Poids VGG (si perceptual)
  beta_kld: 0.001                # Coefficient KL Divergence
  warmup_epochs: 20              # KL annealing (0→β sur 20 epochs)

# ==================== OPTIMIZER ====================
optim:
  algo: "Adam"
  params:
    lr: 0.0005                   # Learning rate (bas pour VAE)
    weight_decay: 0.00001        # L2 regularization
  
  epochs: 100
  beta_kld: 0.001                # Répété (legacy, utilise loss.beta_kld)
  
  # Scheduler
  scheduler:
    name: "ReduceLROnPlateau"
    params:
      mode: "min"                # Minimise loss
      factor: 0.5                # LR *= 0.5 lors plateau
      patience: 5                # Attend 5 epochs
      verbose: true
      min_lr: 0.00001            # Plancher

# ==================== OPTIMIZATION ====================
optimization:
  lr: 0.0005                     # Répété (legacy)
  weight_decay: 0.00001
  accumulation_steps: 128        # Gradient accumulation (batch virtuel)

# ==================== LOGGING ====================
logging:
  logdir: "./logs"
  tensorboard: true              # Active TensorBoard
  # wandb:                       # (Optionnel)
  #   project: "ux-vae"
  #   entity: "your-team"

# ==================== TRAINING ====================
nepochs: 100                     # Nombre total d'epochs
```

### Explication des duplications

Certains paramètres apparaissent dans plusieurs sections pour compatibilité legacy:
- `lr`: dans `optim.params` et `optimization`
- `beta_kld`: dans `optim` et `loss`

Le code utilise prioritairement:
- `loss.beta_kld` pour le coefficient KL
- `optimization.accumulation_steps` pour gradient accumulation
- `optim.params.lr` pour learning rate initial

---

## Résumé des innovations

### Techniques avancées implémentées

1. **Spatial Pyramid Pooling (SPP)**
   - Gère tailles variables sans padding/resize destructeur
   - Multi-échelle (1x1, 2x2, 4x4)

2. **CBAM Attention**
   - Channel + Spatial attention
   - Améliore focus sur structures importantes

3. **GroupNorm**
   - Compatible batch_size=1
   - Stable vs BatchNorm

4. **Gradient Accumulation**
   - Simule batch_size=128 avec batch_size=1
   - Réduit variance gradients

5. **KL Annealing**
   - Évite posterior collapse
   - Warmup progressif β: 0→0.001 sur 20 epochs

6. **Perceptual Loss (VGG)**
   - Préserve structures visuelles
   - Multi-layer features (relu1_2 → relu4_3)

7. **Augmentations Web-Safe**
   - Perspective, Erasing, Salt-and-Pepper
   - Denoising robustness

8. **ReduceLROnPlateau**
   - Adaptation automatique LR
   - Détecte plateaux (patience=5)

9. **SSIM Monitoring**
   - Métrique perceptuelle
   - Meilleure que MSE pour images

10. **Reproductibilité complète**
    - Seeds fixes (data split, dataloaders)
    - Commit ID tracké
    - Config sauvegardée avec checkpoints

### Résultats attendus

- **Reconstruction SSIM**: >0.85 sur validation
- **Latent space**: Embeddings structurés par type de layout
- **Génération**: Interpolations smooth entre wireframes
- **Inpainting**: Remplissage zones effacées (via RandomErasing training)

---

## Références techniques

### Papers
- **VAE**: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
- **SPP**: He et al., "Spatial Pyramid Pooling in Deep Convolutional Networks", ECCV 2014
- **CBAM**: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
- **Perceptual Loss**: Johnson et al., "Perceptual Losses for Real-Time Style Transfer", ECCV 2016

### Librairies
- PyTorch: https://pytorch.org/docs/2.1/
- torchmetrics: https://torchmetrics.readthedocs.io/
- TensorBoard: https://www.tensorflow.org/tensorboard

---

**Auteur**: Généré automatiquement à partir du code source  
**Version**: 1.0  
**Date**: 5 janvier 2026
