# Contributing to UX-Key-PFE

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

Be respectful, inclusive, and professional in all interactions. We're building a collaborative research environment.

---

## Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR-USERNAME/UX-Key-PFE.git
cd UX-Key-PFE
git remote add upstream https://github.com/original-org/UX-Key-PFE.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b bugfix/issue-description
```

---

## Development Workflow

### 1. Make Changes

Keep commits atomic and focused:

```bash
# Good: Small, logical changes
git add src/torchtmpl/data.py
git commit -m "Improve SmartBatchSampler documentation"

# Bad: Too large, mixed concerns
git add .
git commit -m "Refactored everything"
```

### 2. Keep in Sync

```bash
# Update from upstream
git fetch upstream
git rebase upstream/main
```

### 3. Test Before Pushing

```bash
# Run full test suite
pytest tests/ -v

# Check code quality
mypy src/torchtmpl/ --ignore-missing-imports
flake8 src/torchtmpl/ --max-line-length=100
black src/torchtmpl/ --check

# Or format automatically
black src/torchtmpl/
```

---

## Code Standards

### Type Hints (Required)

All public functions must have type hints:

```python
# ❌ BAD
def process_image(img, augment):
    return transformed_img

# ✅ GOOD
from typing import Optional
from PIL import Image

def process_image(
    img: Image.Image,
    augment: bool = False
) -> Image.Image:
    """Process and optionally augment image.
    
    Args:
        img: Input PIL Image
        augment: Whether to apply augmentations
        
    Returns:
        Processed image
    """
    return transformed_img
```

### Docstrings (Required)

Use Google-style docstrings for all public classes/functions:

```python
def compute_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> torch.Tensor:
    """Compute VAE loss (reconstruction + KLD).
    
    Args:
        recon: Reconstructed images (B, C, H, W)
        target: Target images (B, C, H, W)
        mu: Latent means (B, latent_dim)
        logvar: Latent log-variances (B, latent_dim)
        beta: KLD weighting coefficient
        
    Returns:
        Scalar loss value
        
    Raises:
        ValueError: If tensors have incompatible shapes
        
    Example:
        >>> recon = model(x)[3]
        >>> loss = compute_vae_loss(recon, x, mu, logvar, beta=1.0)
    """
    if recon.shape != target.shape:
        raise ValueError(f"Shape mismatch: {recon.shape} vs {target.shape}")
    
    # Implementation
    return loss
```

### Code Style

- **Line length**: Max 100 characters
- **Imports**: Organized in groups (stdlib, external, local)
- **Formatting**: Use `black` for automatic formatting

```python
# ❌ BAD
import torch,numpy,os
from src.torchtmpl.data import VariableSizeDataset
import sys

# ✅ GOOD
import os
import sys
from typing import List, Optional

import numpy as np
import torch

from torchtmpl.data import VariableSizeDataset
```

### Naming Conventions

- Classes: `PascalCase` (e.g., `VariableSizeDataset`)
- Functions: `snake_case` (e.g., `compute_loss`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_BATCH_SIZE`)
- Private: prefix with `_` (e.g., `_internal_helper`)

### Comments

- Use for "why", not "what"
- Keep comments up-to-date with code

```python
# ❌ BAD - says what code already says
x = x + 1  # Add 1 to x

# ✅ GOOD - explains intent
# Skip first sample due to data leakage in preprocessing
x = x[1:]
```

---

## Testing

### Writing Tests

Create tests in `tests/` directory following naming convention:

```python
# tests/test_data.py
import pytest
from torchtmpl.data import VariableSizeDataset

class TestVariableSizeDataset:
    """Test suite for VariableSizeDataset."""
    
    def test_dataset_initialization(self):
        """Dataset should initialize without errors."""
        dataset = VariableSizeDataset(root_dir=".")
        assert len(dataset) > 0
    
    def test_dataset_getitem_types(self):
        """__getitem__ should return correct types."""
        dataset = VariableSizeDataset(root_dir=".")
        tensor, mask = dataset[0]
        
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific file
pytest tests/test_data.py -v

# Run specific test
pytest tests/test_data.py::TestVariableSizeDataset::test_dataset_initialization -v

# Show coverage
pytest tests/ --cov=src/torchtmpl --cov-report=html
```

### Coverage Standards

- Aim for **>80%** coverage on new code
- Critical paths (VAE forward, loss computation) should be **>95%**

```bash
# Check coverage
pytest tests/ --cov=src/torchtmpl --cov-report=term-missing
```

---

## Submitting Changes

### 1. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 2. Create Pull Request

On GitHub:
- Base: `upstream/main`
- Compare: `your-fork/feature/your-feature-name`
- Title: Clear, concise description
- Description: Include:
  - What changed and why
  - How to test the changes
  - Any breaking changes
  - Closes #issue-number (if applicable)

### 3. PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Added unit tests
- [ ] Manual testing completed
- [ ] All tests passing

## Checklist
- [ ] Type hints added
- [ ] Docstrings added
- [ ] Code formatted (`black`, `flake8`)
- [ ] No breaking changes
- [ ] Updated README if needed

## Related Issues
Closes #issue-number
```

### 4. Code Review

- Be receptive to feedback
- Respond to comments promptly
- Update PR based on reviews
- Don't merge your own PR

---

## Reporting Issues

### Bug Report Template

```markdown
## Description
Clear description of the bug.

## Steps to Reproduce
1. ...
2. ...
3. ...

## Expected Behavior
What should happen.

## Actual Behavior
What actually happened.

## Environment
- OS: Ubuntu 22.04
- Python: 3.9
- PyTorch: 2.1.2
- GPU: CUDA 11.8

## Logs/Error Messages
```
error traceback here
```

## Additional Context
Any other relevant information.
```

### Feature Request Template

```markdown
## Description
Clear description of feature.

## Motivation
Why is this feature needed ?

## Proposed Solution
How would you implement it ?

## Alternatives
Other approaches considered.

## Example Usage
```python
# How would users use this feature ?
```
"""

---

## Resources

- [PyTorch Style Guide](https://github.com/pytorch/pytorch/wiki/Style-Guide)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)

---

## Questions ?

- Check [existing issues](https://github.com/your-org/UX-Key-PFE/issues)
- Start a [discussion](https://github.com/your-org/UX-Key-PFE/discussions)
- Email: your.email@example.com

---

Thank you for contributing! 🎉
