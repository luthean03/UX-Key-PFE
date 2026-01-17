# Quick Fixes - 30 minutes d'améliorations immédiatement applicables

## ✅ QUICK FIX #1: Ajouter Type Hints (15 min)

### src/torchtmpl/data.py - Ligne 28

**AVANT:**
```python
def __init__(self, root_dir, noise_level=0.0, max_height=2048, augment=False, files_list=None, 
             sp_prob=0.02, perspective_p=0.3, perspective_distortion_scale=0.08, random_erasing_prob=0.5,
             rotation_degrees=0, brightness_jitter=0.0, contrast_jitter=0.0):
```

**APRÈS:**
```python
from typing import List, Optional

def __init__(self, 
    root_dir: str,
    noise_level: float = 0.0,
    max_height: int = 2048,
    augment: bool = False,
    files_list: Optional[List[str]] = None,
    sp_prob: float = 0.02,
    perspective_p: float = 0.3,
    perspective_distortion_scale: float = 0.08,
    random_erasing_prob: float = 0.5,
    rotation_degrees: float = 0,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0
) -> None:
```

### src/torchtmpl/models/vae_models.py - Ligne 189

**AVANT:**
```python
def forward(self, x, mask=None):
```

**APRÈS:**
```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
```

---

## ✅ QUICK FIX #2: Input Validation (10 min)

### src/torchtmpl/data.py - __getitem__ method (ligne ~140)

**AJOUTER après line 136:**
```python
def __getitem__(self, idx):
    if not (0 <= idx < len(self.files)):
        raise IndexError(f"Index {idx} out of range [0, {len(self.files)})")
    
    img_path = os.path.join(self.root_dir, self.files[idx])
    
    # Validation
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    try:
        clean_image = Image.open(img_path).convert('L')
    except Exception as e:
        raise RuntimeError(f"Failed to load image {img_path}: {e}")
    
    # ... reste du code
```

### src/torchtmpl/models/vae_models.py - VAE.forward() (ligne ~350)

**AJOUTER au début de la méthode:**
```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Input validation
    assert x.dim() == 4, f"Expected 4D input (B,C,H,W), got {x.dim()}D shape {x.shape}"
    assert x.shape[1] == 1, f"Expected 1 channel, got {x.shape[1]}"
    assert x.shape[0] > 0, "Batch size must be positive"
    
    if mask is not None:
        assert mask.shape == x.shape, f"Mask shape {mask.shape} != input shape {x.shape}"
        assert mask.dtype == torch.float32, f"Mask dtype should be float32, got {mask.dtype}"
    
    orig_h, orig_w = x.shape[2], x.shape[3]
    # ... reste du code
```

---

## ✅ QUICK FIX #3: Centralized Seeding (5 min)

### src/torchtmpl/utils.py - AJOUTER nouvelle fonction

```python
def set_reproducibility(seed: int) -> None:
    """Set random seeds for reproducibility across numpy, torch, and CUDA.
    
    Args:
        seed: Random seed value (typically from config)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Prevent non-deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### src/torchtmpl/main.py - ligne 145 (début de train())

**AJOUTER:**
```python
def train(config):
    # Set reproducibility
    seed = config.get("data", {}).get("seed", 42)
    utils.set_reproducibility(seed)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    # ... reste
```

---

## ✅ QUICK FIX #4: Hardcoded Paths → Relative (5 min)

### config/config-vae.yaml - Ligne 8-9

**AVANT:**
```yaml
data_dir: "/usr/users/sdim/sdim_31/UX-Key-PFE/dataset/vae_dataset_scaled"
archetypes_dir: "/usr/users/sdim/sdim_31/UX-Key-PFE/dataset/archetypes_png_scaled"
```

**APRÈS:**
```yaml
data_dir: "dataset/vae_dataset_scaled"
archetypes_dir: "dataset/archetypes_png_scaled"
```

### src/torchtmpl/main.py - Resolve relative paths (ligne ~170)

**AJOUTER:**
```python
def _resolve_config_paths(config: dict, base_dir: pathlib.Path) -> dict:
    """Convert relative paths in config to absolute paths."""
    data_cfg = config.get("data", {})
    
    for key in ["data_dir", "archetypes_dir"]:
        if key in data_cfg:
            path = data_cfg[key]
            if not os.path.isabs(path):
                data_cfg[key] = str(base_dir / path)
    
    return config

# Dans train():
config = _resolve_config_paths(config, pathlib.Path(__file__).parent.parent.parent)
```

---

## ✅ QUICK FIX #5: Better Error Messages (5 min)

### src/torchtmpl/optim.py - ligne 30

**AVANT:**
```python
if name == "ReduceLROnPlateau":
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
if name == "CosineAnnealingLR":
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
if name == "CosineAnnealingWarmRestarts":
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)

logging.warning(f"Unknown scheduler: {name} (scheduler disabled)")
return None
```

**APRÈS:**
```python
available_schedulers = {
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}

if name not in available_schedulers:
    raise ValueError(
        f"Unknown scheduler: {name}\n"
        f"Available: {list(available_schedulers.keys())}"
    )

scheduler_class = available_schedulers[name]
return scheduler_class(optimizer, **params)
```

---

## 📋 Summary - Copy & Paste Ready

```bash
# 1. Add type hints to all public methods (15 min)
# Edit: data.py, vae_models.py, loss.py, optim.py

# 2. Add assertions at function entry (10 min)
# Edit: data.py __getitem__, vae_models.py forward()

# 3. Add set_reproducibility() to utils.py (5 min)
# Call it at start of train()

# 4. Make paths relative in config-vae.yaml (5 min)
# Add path resolution in main.py

# 5. Improve error messages in optim.py (5 min)
# Use dict lookup instead of if/elif

# TOTAL: ~40 minutes → cleaner, more robust code
```
