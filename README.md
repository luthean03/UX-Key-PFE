# UX-Key-PFE: VAE for UI/UX Wireframe Design Space Learning

![Status](https://img.shields.io/badge/status-research-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Python](https://img.shields.io/badge/python-3.9+-blue)

## 🎯 Project Overview

**UX-Key-PFE** is a Variational Autoencoder (VAE) trained on mobile UI/UX wireframe layouts for:

- **Clustering** - Automatically group similar design patterns (archetypes)
- **Interpolation** - Smoothly transition between design concepts
- **Generation** - Sample new valid wireframe layouts from latent space
- **Analysis** - Understand structure of design space using latent representations

The model is designed for **variable-size images** (1000-3000px tall, mobile wireframes) using innovative techniques like Spatial Pyramid Pooling and masked normalization.

### Key Features

✅ **Variable-height input handling** - No artificial resizing, preserves layout proportions  
✅ **Attention mechanisms** - CBAM (Channel + Spatial) for selective feature learning  
✅ **Multi-component loss** - SSIM + Gradients + Reconstruction + KLD  
✅ **Gradient accumulation & mixed precision** - Efficient memory usage  
✅ **SLURM integration** - Easy cluster deployment  
✅ **TensorBoard monitoring** - Real-time training visualization  

---

## 📦 Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-org/UX-Key-PFE.git
cd UX-Key-PFE

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .  # Installs torchtmpl + all dependencies

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "from torchtmpl.models import VAE; print('VAE imported successfully')"
```

### SLURM Cluster

```bash
# Setup on login node
cd /path/to/UX-Key-PFE

# Submit training job
python submit-slurm.py config/config-vae.yaml train
```

### Docker (Optional)

```bash
docker build -t ux-key-pfe .
docker run --gpus all -it ux-key-pfe
```

---

## 🚀 Quick Start

### 1. Prepare Data

```bash
# Expected directory structure:
# dataset/
#   ├── vae_dataset_scaled/      # Training images (PNG)
#   ├── archetypes_png_scaled/   # Reference designs for analysis
#   └── samir_lom/               # Raw wireframes (JSON)

# If starting from scratch, preprocess wireframes:
python preprocess/json_to_png.py dataset/samir_lom/ dataset/archetypes_png/
python preprocess/scale.py dataset/archetypes_png/ dataset/archetypes_png_scaled/
```

### 2. Configure Training

Edit `config/config-vae.yaml` to customize:

```yaml
data:
  batch_size: 16              # Adjust for GPU memory
  num_workers: 8              # CPU cores for data loading
  augment: true               # Enable augmentations
  
model:
  latent_dim: 128             # Latent space dimensions
  dropout_p: 0.1              # Regularization
  
optim:
  algo: "AdamW"
  params:
    lr: 0.0003                # Learning rate
    weight_decay: 0.0001      # L2 regularization
```

### 3. Train Locally

```bash
# Single GPU training
python -m torchtmpl.main train config/config-vae.yaml

# Monitor with TensorBoard (new terminal)
tensorboard --logdir logs

# Open browser: http://localhost:6006
```

### 4. Evaluate & Analyze

```bash
# Interpolate between designs
python -m torchtmpl.main interpolate config/config-vae.yaml

# Analyze latent space (t-SNE visualization)
python -m torchtmpl.main latent_analysis config/config-vae.yaml

# Output saved to: test_output/
```

---

## 📊 Architecture

### Encoder
```
Input Image (B, 1, H, W)
    ↓ Conv7x7 + GroupNorm + ReLU + MaxPool
    ↓ ResBlock (64→128, stride=2)
    ↓ ResBlock (128→256, stride=2)
    ↓ ResBlock (256→512, stride=2)
    ↓ Spatial Pyramid Pooling [1, 2, 4]
    ↓ FC layers
    ↓ (μ, log σ²) ∈ ℝ^128
```

**Key Features:**
- **GroupNorm** instead of BatchNorm (works with variable batch sizes)
- **CBAM Attention** (channel + spatial) in residual blocks
- **Spatial Pyramid Pooling** handles variable heights → fixed-size latent

### Decoder
```
Latent Code z ∈ ℝ^128
    ↓ FC expand
    ↓ Reshape (256, 32, 16)
    ↓ Upsample 2x + ResBlock
    ↓ Upsample 2x + ResBlock
    ↓ Upsample 2x + ResBlock
    ↓ Upsample 2x + ResBlock
    ↓ Conv1x1 + Sigmoid
    ↓ Reconstruction (B, 1, H, W)
```

### Loss Function

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathcal{L}_{\text{recon}}}_{\text{SSIM}} + \beta_{\text{KLD}} \cdot \underbrace{D_{\text{KL}}(q_\phi(z|x) || p(z))}_{\text{Regularization}}$$

Where:
- **Reconstruction loss** = SSIM (structural) + L1 (pixel-level)
- **KLD loss** = KL divergence with annealing (warmup over 10 epochs)
- **β coefficient** = 1.0 (naturally balanced with sum reduction)

---

## 📈 Results

### Training Curves (Example)

| Metric | Value |
|--------|-------|
| Best SSIM (Validation) | 0.92 |
| Best Reconstruction Loss | 0.034 |
| Best KLD Loss | 0.12 |
| Epochs to Convergence | 150 |
| Training Time (GPU) | 8-12h |

### Qualitative Results

1. **Clustering** - Similar wireframes cluster together in latent space
2. **Smooth Interpolation** - SLERP between designs produces valid layouts
3. **Generation** - Random sampling from N(0, I) produces reasonable wireframes

See `results/` folder for visualizations.

---

## 🔍 Monitoring & Debugging

### TensorBoard

```bash
tensorboard --logdir logs
# Metrics tracked:
# - loss/recon, loss/kld, loss/total
# - metrics/ssim (validation)
# - learning_rate (scheduler)
# - latent_visualization (t-SNE every 20 epochs)
```

### Console Output

```
Epoch 1/200:
  Train Loss: 0.456 (recon: 0.340, kld: 0.116)
  Valid Loss: 0.512 (recon: 0.390, kld: 0.122)
  SSIM: 0.891 | LR: 3.00e-04
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM Error | Batch too large | Reduce `batch_size` in config |
| KLD → 0 | Posterior collapse | Enable KLD warmup, increase β |
| Blurry reconstructions | Loss not converged | Train longer, check SSIM weight |
| GPU memory leak | Data loading issue | Check `num_workers` setting |

---

## 🛠️ Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/torchtmpl --cov-report=html
```

### Code Quality

```bash
# Type checking
mypy src/torchtmpl/ --ignore-missing-imports

# Linting
flake8 src/torchtmpl/ --max-line-length=100

# Formatting
black src/torchtmpl/
```

### Project Structure

```
UX-Key-PFE/
├── config/                    # Configuration files (YAML)
├── dataset/                   # Data directories (git-ignored)
├── logs/                      # Training checkpoints & logs
├── preprocess/               # Data preprocessing scripts
├── src/
│   └── torchtmpl/
│       ├── __init__.py
│       ├── main.py           # Training entrypoint
│       ├── data.py           # DataLoaders & augmentation
│       ├── loss.py           # Loss functions
│       ├── optim.py          # Optimizers & schedulers
│       ├── utils.py          # Utilities (SLERP, checkpoints)
│       ├── latent_metrics.py # Latent space analysis
│       └── models/
│           ├── vae_models.py # VAE architecture
│           └── base_models.py
├── tests/                    # Unit & integration tests
├── AUDIT.md                  # Detailed code audit
├── QUICK_FIXES.md           # Immediate improvements
├── README.md                 # This file
└── pyproject.toml           # Dependencies
```

---

## 🎓 Learning Resources

- **VAE Theory**: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- **Spatial Pyramid Pooling**: [SPP-Net Paper](https://arxiv.org/abs/1406.4729)
- **CBAM Attention**: [CBAM Paper](https://arxiv.org/abs/1807.06521)
- **PyTorch Best Practices**: [PyTorch Docs](https://pytorch.org/)

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open Pull Request

Please ensure:
- ✅ Tests pass (`pytest tests/`)
- ✅ Code is formatted (`black` & `flake8`)
- ✅ Type hints added (`mypy`)
- ✅ Docstrings included

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{ux_key_pfe_2026,
  title={UX-Key-PFE: VAE for UI/UX Wireframe Design Space},
  author={Your Name},
  year={2026},
  url={https://github.com/your-org/UX-Key-PFE}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙋 Support

- **Questions?** Open an [Issue](https://github.com/your-org/UX-Key-PFE/issues)
- **Bugs?** Submit a [Bug Report](https://github.com/your-org/UX-Key-PFE/issues)
- **Ideas?** Start a [Discussion](https://github.com/your-org/UX-Key-PFE/discussions)

---

## 📞 Contact

- **Author**: [Your Name]
- **Email**: your.email@example.com
- **LinkedIn**: [Your Profile]

---

**Last Updated**: 17 janvier 2026  
**Status**: Active Research ✨
