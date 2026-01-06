# β-VAE pour Espace Latent Structuré de Wireframes UI/UX

**Présentation technique rigoureuse**  
**Auteur:** Samir Dimachkie | **Date:** 6 janvier 2026  
**Framework:** PyTorch 2.1.2 + CUDA 11.8

---

## 🎯 Problématique scientifique

### Objectif de recherche

Apprendre un espace latent **dense et structuré** où :
1. Les **15 archétypes sémantiques** (Accueil, Contact, Tarifs...) forment des **clusters distincts**
2. Les **interpolations linéaires** $z_t = (1-t)z_1 + t z_2$ produisent des wireframes **sémantiquement cohérents**
3. L'espace latent suit $p(z) = \mathcal{N}(0, I)$ pour faciliter la génération

### Pourquoi β-VAE et pas AutoEncoder classique ?

| Critère | AutoEncoder | β-VAE (notre choix) |
|---------|-------------|---------------------|
| **Espace latent** | Chaotique, discontinu | Dense, gaussien structuré |
| **Interpolations** | Artefacts, sauts brusques | Fluides, cohérentes |
| **Génération** | ❌ Impossible | ✅ $z \sim \mathcal{N}(0,I)$ |
| **Clustering** | ❌ Pas de garantie | ✅ Séparation naturelle |

---

## 📊 Contraintes et solutions architecturales

### Contraintes matérielles

| Contrainte | Solution technique |
|------------|-------------------|
| Vieux GPU (~8GB) | Mixed Precision (AMP), PyTorch 2.1.2 |
| **Tailles variables** (500-2048px) | **Spatial Pyramid Pooling (SPP)** |
| **batch_size = 1** | **Gradient Accumulation (×128)** |
| Lignes fines (wireframes) | **VGG Perceptual Loss** |

### Architecture globale

```
Input (1, H_var, W)
    ↓ Encoder ResNet + CBAM
Features (512, H/32, W/32)
    ↓ SPP (multi-scale pooling)
Latent (128D) : μ, log σ²
    ↓ z = μ + σ·ε, ε ~ N(0,1)
    ↓ Decoder ResNet
Output (1, H_var, W)
```

---

## 🧮 Formulation mathématique

### 1. Evidence Lower Bound (ELBO)

**Objectif :** Maximiser $\log p_\theta(x)$ (intractable)

**Borne variationnelle :**
$$
\log p_\theta(x) \geq \underbrace{\mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)]}_{\text{Reconstruction}} - \underbrace{\text{KL}(q_\phi(z|x) \| p(z))}_{\text{Régularisation}}
$$

### 2. β-VAE : Disentanglement et clusters

**Loss standard** (β=1) :
$$
\mathcal{L}_{\text{VAE}} = -\mathbb{E}[\log p_\theta(x|z)] + \text{KL}(q_\phi(z|x) \| \mathcal{N}(0, I))
$$

**β-VAE** (Higgins et al., 2017) :
$$
\boxed{\mathcal{L}_{\beta\text{-VAE}} = -\mathbb{E}[\log p_\theta(x|z)] + \beta \cdot \text{KL}(q_\phi(z|x) \| \mathcal{N}(0, I))}
$$

**Notre choix : β = 1.0**
- Force la distribution latente vers $\mathcal{N}(0, I)$
- Sépare naturellement les archétypes en clusters
- Interpolations fluides (pas de trous dans l'espace)

### 3. KL Divergence analytique

**Distributions :**
- Prior : $p(z) = \mathcal{N}(0, I_{128})$
- Posterior : $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$

**Formule fermée :**
$$
\boxed{\text{KL} = \frac{1}{2} \sum_{i=1}^{128} \left( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \right)}
$$

**Implémentation PyTorch :**
```python
kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

**Interprétation :**
- $\mu_i^2$ : Pénalise écart à l'origine
- $\sigma_i^2$ : Pénalise variance excessive
- $-\log \sigma_i^2$ : Évite collapse ($\sigma \to 0$)

---

## 🏗️ Architecture VAE-SPP-CBAM

### Encoder : ResNet avec downsampling ×32

```python
Encoder = Sequential(
    Conv2d(1→64, k=7, s=2), GroupNorm, ReLU, MaxPool(s=2),  # ÷4
    ResBlock(64→128, s=2),   # ÷8
    ResBlock(128→256, s=2),  # ÷16
    ResBlock(256→512, s=2)   # ÷32
)
# Input: (1, 2048, 1024) → Output: (512, 64, 32)
```

**Pourquoi GroupNorm au lieu de BatchNorm ?**

BatchNorm instable avec batch_size=1 :
$$
\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma_{\text{batch}}^2 + \epsilon}} \quad \leftarrow \text{undefined si batch=1}
$$

GroupNorm calcule stats **par image** :
$$
\hat{x}_{i,g} = \frac{x_{i,g} - \mu_g(x_i)}{\sqrt{\sigma_g^2(x_i) + \epsilon}} \quad \leftarrow \text{stable}
$$

### Spatial Pyramid Pooling (SPP)

**Problème :** Features shape $(B, 512, H_{feat}, W_{feat})$ avec $H_{feat}$ variable  
**Solution :** Multi-scale adaptive pooling

$$
\text{SPP}(x) = \text{Concat}\left[ \text{AvgPool}_{1 \times 1}(x), \text{AvgPool}_{2 \times 2}(x), \text{AvgPool}_{4 \times 4}(x) \right]
$$

**Dimension fixe :**
$$
d_{\text{SPP}} = 512 \times (1^2 + 2^2 + 4^2) = 512 \times 21 = 10{,}752
$$

**Invariance spatiale :** Quelle que soit $H$, sortie toujours $10{,}752$D

### CBAM : Channel + Spatial Attention

**Channel Attention :** "Quels features sont importants ?"
$$
\text{CA}(F) = \sigma\left( \text{MLP}(\text{AvgPool}(F)) + \text{MLP}(\text{MaxPool}(F)) \right)
$$

**Spatial Attention :** "Où sont les zones importantes ?"
$$
\text{SA}(F) = \sigma\left( \text{Conv}_{7×7}\left( [\text{AvgPool}_c(F); \text{MaxPool}_c(F)] \right) \right)
$$

**Application :**
$$
F' = F \odot \text{CA}(F) \quad \text{puis} \quad F'' = F' \odot \text{SA}(F')
$$

**Impact :** Focus sur lignes/bordures (zones critiques wireframes)

### Reparameterization Trick

**Projection latente :**
$$
\mu = W_\mu \cdot \text{SPP}(x) \in \mathbb{R}^{128}, \quad \log \sigma^2 = W_\sigma \cdot \text{SPP}(x) \in \mathbb{R}^{128}
$$

**Sampling avec gradient flow :**
$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_{128})
$$

**Pourquoi $\log \sigma^2$ et pas $\sigma$ ?**
- Stabilité : $\sigma = \exp(0.5 \log \sigma^2)$ toujours positif
- Pas de contrainte : $\log \sigma^2 \in \mathbb{R}$

---

## 🎨 Loss hybride : Pixel + Perceptual + KLD

### Motivation : Pourquoi loss pixel seule échoue

**Problème wireframes :**
- Lignes fines (1-2px)
- Fond blanc dominant (95% pixels)
- MSE favorise le **blur** (minimisation par moyenne)

**Exemple :**
```
Ground Truth:  |  (ligne nette)
MSE optimum:   ≡  (ligne floue, intensité réduite)
```

### VGG Perceptual Loss

**Features multi-échelles VGG16 pré-entraîné :**
$$
\mathcal{L}_{\text{VGG}} = \sum_{\ell \in \{\text{relu1\_2, relu2\_2, relu3\_3, relu4\_3}\}} \| \phi_\ell(x) - \phi_\ell(\hat{x}) \|_1
$$

où $\phi_\ell$ = activations VGG à la couche $\ell$

**Preprocessing :** Grayscale → RGB + normalisation ImageNet
$$
x_{\text{norm}} = \frac{\text{repeat}(x, 3) - [0.485, 0.456, 0.406]}{[0.229, 0.224, 0.225]}
$$

### Loss totale

$$
\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pixel}} + \alpha \cdot \mathcal{L}_{\text{VGG}} + \beta \cdot \text{KL}}
$$

**Composantes :**
1. **Pixel L1** : $\sum_{i,j} |x_{i,j} - \hat{x}_{i,j}|$ (ancrage structurel)
2. **VGG** : Features perceptuelles multi-échelle (netteté)
3. **KL** : Régularisation latente (clusters)

**Hyperparamètres critiques :**
- $\alpha$ (perceptual_weight) : 0.01 → 0.08 (warmup)
- $\beta$ (beta_kld) : 0 → 1.0 (warmup)

### Problème d'échelles

**Magnitudes observées (epoch 30) :**
```
Pixel loss : ~560,000  (somme sur 2M pixels)
VGG loss   : ~126,000  (features maps)
KL loss    : ~76       (128 dimensions)
```

**Contributions à la loss totale :**
$$
\mathcal{L} = 560k + (0.08 \times 126k) + (1.0 \times 76) \approx 570k
$$

**Poids effectifs :**
- Pixel : 98.2% (domine - reconstruction fidèle)
- VGG : 1.8% (affine qualité)
- KL : 0.01% (régularise structure latente)

**Justification :** Balance reconstruction/structure optimale pour wireframes

---

## 📈 Schedules d'annealing

### KL Annealing : Éviter posterior collapse

**Problème :** Si $\beta$ élevé dès epoch 0 :
$$
\min \mathcal{L} \Rightarrow q_\phi(z|x) \approx \mathcal{N}(0, I) \quad \forall x
$$
→ Encoder ignore $x$ (collapse)

**Solution (Bowman et al., 2016) :**
$$
\beta(t) = \begin{cases}
\beta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t < T_{\text{warmup}} \\
\beta_{\max} & t \geq T_{\text{warmup}}
\end{cases}
$$

**Configuration :**
```yaml
loss:
  beta_kld: 1.0        # β_max
  warmup_epochs: 20    # T_warmup
```

**Évolution :**
- Epoch 0-10 : Apprentissage reconstruction libre
- Epoch 10-20 : Régularisation progressive
- Epoch 20+ : Structure gaussienne stricte

### Perceptual Weight Warmup

**Observation empirique :** $\alpha=0.08$ constant → pixel loss monte

**Curriculum learning :**
$$
\alpha(t) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \frac{t}{T_{\text{warmup}}}
$$

**Stratégie :**
1. Epoch 0-10 : $\alpha \approx 0.01$ → Pixel prioritaire
2. Epoch 10-30 : Transition progressive
3. Epoch 30+ : $\alpha = 0.08$ → Qualité perceptuelle max

**Avantage :** Pixel loss descend d'abord, VGG affine ensuite

---

## 🔧 Stratégie d'optimisation

### Gradient Accumulation

**Problème :** batch_size=1 → variance gradients élevée

**Solution :** Accumuler sur $N$ images
```python
accumulation_steps = 128

for i, (x, y) in enumerate(train_loader):
    loss = criterion(model(x), y, mu, logvar)
    (loss / accumulation_steps).backward()
    
    if (i+1) % accumulation_steps == 0:
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

**Effet :** Batch virtuel de 128 → gradients stables

### Optimizer : Adam

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003,           # LR bas (VAE sensibles)
    weight_decay=0.0001  # L2 regularization
)
```

### Scheduler : CosineAnnealingWarmRestarts

**Motivation :** Échapper minima locaux

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i}\pi\right)\right)
$$

**Configuration :**
```yaml
scheduler:
  name: CosineAnnealingWarmRestarts
  params:
    T_0: 10        # Premier cycle
    T_mult: 2      # Cycles : 10, 30, 70 epochs
    eta_min: 0.00003
```

**Warm restarts :** Epochs 10, 30, 70 → exploration nouvelles solutions

---

## 📊 Métriques d'évaluation

### 1. Reconstruction : SSIM

$$
\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x\mu_{\hat{x}} + c_1)(2\sigma_{x\hat{x}} + c_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + c_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + c_2)}
$$

**Avantage :** Meilleure corrélation perception humaine que MSE

### 2. Cluster Quality : Silhouette Score

$$
s_i = \frac{b_i - a_i}{\max(a_i, b_i)}
$$

- $a_i$ : Distance moyenne intra-cluster
- $b_i$ : Distance moyenne au cluster le plus proche

**Interprétation :**
- $s > 0.5$ : Clusters bien séparés ✅
- $s < 0.3$ : Clusters mélangés ❌

### 3. Interpolation Quality

**Smoothness :** Variance des différences frame-to-frame

$$
\text{Smoothness} = \text{Var}\left( \{\|\hat{x}_t - \hat{x}_{t+1}\|_1\}_{t=0}^{N-1} \right)
$$

**Low variance** = interpolation fluide

### 4. Latent Space Density

**k-NN Distance :** Détecte les trous
$$
d_{\text{kNN}}(z) = \frac{1}{k} \sum_{j=1}^{k} \|z - z_j^{\text{neighbor}}\|_2
$$

**Effective Dimensionality :** PCA explained variance (95%)

---

## 🔬 Résultats expérimentaux

### Configuration finale

```yaml
loss:
  beta_kld: 1.0                  # β-VAE fort
  perceptual_weight: 0.08        # VGG modéré
  perceptual_warmup_epochs: 30
  warmup_epochs: 20

model:
  latent_dim: 128
  use_skip_connections: true     # U-Net style

optimization:
  lr: 0.0003
  accumulation_steps: 64
  mixed_precision: true          # AMP
```

### Métriques attendues

| Métrique | Target (β=1.0) | Baseline (β=0.005) |
|----------|----------------|-------------------|
| **SSIM** | ~0.75-0.78 | 0.815 |
| **Silhouette** | **>0.5** | <0.2 |
| **Min Cluster Dist** | **>15.0** | <5.0 |
| **Interpolation Var** | **<0.1** | >0.5 |

**Trade-off accepté :** Légère baisse SSIM (-5%) pour structure latente optimale

### Visualisations TensorBoard

**Scalars (tous les 10 epochs) :**
- `latent/silhouette_score` : Qualité clusters
- `latent/interpolation_smoothness` : Fluidité interpolations
- `latent/mean_knn_distance` : Densité espace latent

**Images :**
- `latent/tsne_visualization` : t-SNE 2D coloré par archetype
- `reconstructions` : Grille [Target | Reconstruction]

---

## 🚀 Infrastructure SLURM

### Pipeline exécution

```bash
# 1. Copie code + archetypes vers nœud local
rsync -r . $TMPDIR/code
rsync -r archetypes_png/ $TMPDIR/code/archetypes_png/

# 2. Setup environnement
python3 -m venv venv
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118

# 3. Optimisation mémoire
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Lancement
python -m torchtmpl.main config.yaml train
```

### Configuration SLURM

```bash
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --nodes=1
```

---

## 📚 Références clés

1. **VAE** : Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
2. **β-VAE** : Higgins et al., "β-VAE: Learning Basic Visual Concepts", ICLR 2017
3. **SPP** : He et al., "Spatial Pyramid Pooling", ECCV 2014
4. **CBAM** : Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
5. **Perceptual Loss** : Johnson et al., "Perceptual Losses for Real-Time Style Transfer", ECCV 2016
6. **KL Annealing** : Bowman et al., "Generating Sentences from a Continuous Space", CoNLL 2016

---

## 🎯 Innovations techniques

| Innovation | Justification mathématique | Impact |
|------------|---------------------------|--------|
| **β=1.0** | Force $q(z\|x) \to \mathcal{N}(0,I)$ | Clusters séparés |
| **SPP** | Pooling adaptatif multi-échelle | Tailles variables |
| **Perceptual Warmup** | Curriculum learning | Pixel↓ puis VGG↑ |
| **GroupNorm** | Stats par image | Stable batch=1 |
| **Gradient Accum** | Batch virtuel ×128 | Variance↓ |
| **CBAM** | Double attention C+S | Focus lignes fines |

---

**Auteur :** Samir Dimachkie  
**Contact :** sdim_31@esiee.fr  
**Repository :** `/usr/users/sdim/sdim_31/UX-Key-PFE`
