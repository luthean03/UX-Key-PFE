# Debugging & Troubleshooting Guide

Common issues when training VAE models and how to fix them.

---

## 🚨 CRITICAL ERRORS

### 1. CUDA Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions (in order of effort):**

1. **Reduce batch size** (fastest)
   ```yaml
   # config-vae.yaml
   data:
     batch_size: 8  # From 16 → 8
   ```

2. **Reduce image max height**
   ```yaml
   data:
     max_height: 2048  # From 3000 → 2048
   ```

3. **Reduce latent dimension**
   ```yaml
   model:
     latent_dim: 64  # From 128 → 64
   ```

4. **Enable gradient accumulation**
   ```yaml
   optimization:
     accumulation_steps: 2  # Effective batch = 16 * 2 = 32
   ```

5. **Clear cache and check GPU**
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # In Python:
   torch.cuda.empty_cache()
   ```

6. **Last resort: Reduce model size**
   ```yaml
   model:
     dropout_p: 0.2  # Add regularization
     # Encoder will have fewer filters (edit vae_models.py)
   ```

**Prevention:**
```python
# Add to training script
def estimate_memory_usage(batch_size, height, width):
    params = sum(p.numel() for p in model.parameters())
    activation_memory = batch_size * 4 * height * width * 32  # bytes
    param_memory = params * 4  # float32
    
    total_gb = (activation_memory + param_memory) / 1e9
    print(f"Estimated GPU memory: {total_gb:.2f} GB")
    
    if total_gb > 20:
        print("⚠️  WARNING: Might be too large for typical 24GB GPUs")

estimate_memory_usage(16, 256, 256)
```

---

### 2. Training Loss Not Decreasing

**Problem:** Loss plateaus or increases

**Root Causes:**

1. **Learning rate too high**
   ```yaml
   optim:
     params:
       lr: 0.0003  # Try 0.0001 or 0.00003
   ```

2. **Learning rate too low**
   ```yaml
   optim:
     params:
       lr: 0.0001  # Try 0.001
   ```

3. **Data loading issue**
   ```python
   # Check first batch
   for batch in train_loader:
       x, mask = batch
       print(f"Batch shape: {x.shape}")
       print(f"Batch range: [{x.min()}, {x.max()}]")
       print(f"NaN/Inf: {torch.isnan(x).sum()}, {torch.isinf(x).sum()}")
       break
   ```

4. **Model not training (no gradient flow)**
   ```python
   # Check gradients exist
   optimizer.zero_grad()
   loss.backward()
   
   grad_count = 0
   for name, param in model.named_parameters():
       if param.grad is not None and param.grad.abs().sum() > 0:
           grad_count += 1
   
   print(f"Params with non-zero gradients: {grad_count} / {sum(1 for _ in model.parameters())}")
   ```

**Fixes:**

```python
# Add comprehensive logging
def debug_training_step(model, x, target):
    """Debug a single training step."""
    model.train()
    
    # Forward
    print("1. Forward pass...")
    mu, logvar, z, recon = model(x)
    
    print(f"   mu shape: {mu.shape}, range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"   recon shape: {recon.shape}, range: [{recon.min():.3f}, {recon.max():.3f}]")
    
    # Check for NaN
    if torch.isnan(recon).any():
        print("   ⚠️  ALERT: NaN in reconstruction!")
        return
    
    # Loss
    print("2. Computing loss...")
    loss = loss_fn(recon, target, mu, logvar)
    print(f"   Loss: {loss.item():.6f}")
    
    if torch.isnan(loss):
        print("   ⚠️  ALERT: NaN in loss!")
        return
    
    # Backward
    print("3. Backward pass...")
    model.zero_grad()
    loss.backward()
    
    total_grad_norm = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"   Total grad norm: {total_grad_norm:.6f}")
    
    if total_grad_norm == 0:
        print("   ⚠️  ALERT: Zero gradients!")
        return
    
    print("✅ Training step OK")

# Usage:
debug_training_step(model, x_sample, x_sample)
```

---

### 3. Blurry Reconstructions

**Problem:** Model outputs are always blurry/averaged

**Root Causes:**

1. **Loss weight imbalance** - KLD dominates
   ```python
   # Check in validation:
   recon_loss_val = 0.5
   kld_loss_val = 0.01
   
   # KLD is too small! Increase beta
   print(f"Ratio: {kld_loss_val / recon_loss_val:.3f}")
   # Should be ~0.1-0.5, not 0.02
   ```

2. **Reconstruction loss weight too low**
   ```yaml
   loss:
     beta_kld: 10.0  # Try increasing KLD penalty to force learning
   ```

3. **Model capacity too low**
   ```yaml
   model:
     latent_dim: 128  # Try 256 for more capacity
   ```

**Debugging:**

```python
# Check loss components during validation
def detailed_validation(model, val_loader, loss_fn, device):
    model.eval()
    
    recon_losses = []
    kld_losses = []
    ssims = []
    
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            mu, logvar, z, recon = model(x)
            
            # Decompose loss
            recon_loss = F.mse_loss(recon, x)
            kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
            
            recon_losses.append(recon_loss.item())
            kld_losses.append(kld.item())
            
            ssim_val = ssim_metric(recon, x).item()
            ssims.append(ssim_val)
    
    print(f"Recon Loss: {np.mean(recon_losses):.4f} ± {np.std(recon_losses):.4f}")
    print(f"KLD Loss:   {np.mean(kld_losses):.4f} ± {np.std(kld_losses):.4f}")
    print(f"SSIM:       {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    
    # Print ratio
    ratio = np.mean(recon_losses) / np.mean(kld_losses)
    print(f"Recon/KLD ratio: {ratio:.2f}")
```

---

## ⚠️ WARNINGS

### 4. NaN/Inf Gradients

**Error:**
```
Loss is NaN/Inf
```

**Causes & Fixes:**

1. **Learning rate too high**
   ```yaml
   optim:
     params:
       lr: 0.0001  # Reduce by 10x
   ```

2. **Exploding gradients**
   ```python
   # Add gradient clipping to training loop
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   optimizer.step()
   ```

3. **Division by zero in loss**
   ```python
   # In loss.py, add epsilon:
   loss = 1.0 - ssim_map.sum() / (mask_small.sum() + 1e-8)  # Add 1e-8!
   ```

4. **Log of negative or zero**
   ```python
   # In VAE reparameterize:
   std = torch.exp(0.5 * logvar)  # OK, always > 0
   
   # But check logvar doesn't go to -inf:
   assert logvar.min() > -100, "logvar too negative"
   ```

**Debugging:**

```python
# Hook to catch NaN
def add_nan_checker(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(lambda grad, n=name: 
                print(f"⚠️  NaN gradient in {n}") if torch.isnan(grad).any() else None
            )

add_nan_checker(model)
```

---

### 5. Model Not Learning (SSIM plateaus)

**Problem:** SSIM ≈ constant (e.g., 0.85) across epochs

**Diagnosis:**

```python
# Check if model parameters are updating
initial_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

train_epoch(model, train_loader)

# Check if anything changed
changes = []
for k, v in model.state_dict().items():
    change = (v - initial_state[k]).abs().max().item()
    if change > 0:
        changes.append((k, change))

if not changes:
    print("⚠️  Model not updating at all!")
elif max(c[1] for c in changes) < 1e-6:
    print("⚠️  Model updating very slowly (maybe learning rate too low)")
else:
    print(f"✅ Model updating: {len(changes)} params changed")
```

**Solutions:**

1. **Check learning rate**
   ```python
   for param_group in optimizer.param_groups:
       print(f"Current LR: {param_group['lr']}")
   ```

2. **Verify optimizer is actually used**
   ```python
   # ✅ Correct:
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()  # Must call this!
   
   # ❌ Wrong:
   optimizer.zero_grad()
   loss.backward()
   # Missing: optimizer.step()
   ```

3. **Check data isn't frozen**
   ```python
   x, _ = next(iter(train_loader))
   x2, _ = next(iter(train_loader))
   
   print(f"Batches are different: {not torch.allclose(x, x2)}")
   # Should be True (different random samples)
   ```

---

### 6. Training Slow (Very Long Epochs)

**Problem:** Each epoch takes > 1 hour

**Diagnosis:**

```python
import time

epoch_times = []
for epoch in range(3):  # Sample 3 epochs
    start = time.time()
    
    data_time = 0
    compute_time = 0
    
    for batch_idx, batch in enumerate(train_loader):
        t0 = time.time()
        x, mask = batch
        x = x.to(device)
        data_time += time.time() - t0
        
        t1 = time.time()
        # Forward/backward
        optimizer.zero_grad()
        mu, logvar, z, recon = model(x)
        loss = loss_fn(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        compute_time += time.time() - t1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}: data={data_time:.1f}s, compute={compute_time:.1f}s")
    
    epoch_times.append(time.time() - start)
    print(f"Epoch time: {epoch_times[-1]:.1f}s")

print(f"\nAverage epoch: {np.mean(epoch_times):.1f}s")
print(f"Data loading: {data_time / compute_time * 100:.1f}% of time")
```

**Fixes:**

1. **If data loading is slow:**
   ```yaml
   data:
     num_workers: 16  # Increase from 8
     prefetch_factor: 2  # Add prefetching
     persistent_workers: true
   ```

2. **If compute is slow:**
   ```bash
   # Check GPU utilization
   nvidia-smi -l 1  # Monitor GPU
   # If <50% utilization, might be I/O starved
   ```

3. **Profile with PyTorch profiler**
   ```python
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
   ) as prof:
       train_epoch(model, train_loader)
   
   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
   ```

---

## 📊 VALIDATION

### 7. Unrealistic Validation Results

**Problem:** Training loss decreases but validation loss increases

**Sign:** Overfitting

**Fixes:**

1. **Increase regularization**
   ```yaml
   optim:
     params:
       weight_decay: 0.001  # Increase L2 regularization
   
   model:
     dropout_p: 0.2  # Increase dropout
   
   data:
     augment: true
     random_erasing_prob: 0.5  # Aggressive masking
   ```

2. **Use early stopping**
   ```python
   best_val_loss = float('inf')
   patience = 20
   patience_counter = 0
   
   for epoch in range(num_epochs):
       train_loss = train_epoch(...)
       val_loss = validate_epoch(...)
       
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           patience_counter = 0
           torch.save(model.state_dict(), 'best_model.pt')
       else:
           patience_counter += 1
           if patience_counter >= patience:
               print(f"Early stopping at epoch {epoch}")
               break
   ```

3. **Reduce model complexity**
   ```yaml
   model:
     latent_dim: 64  # Reduce capacity
   ```

---

## 🔧 PERFORMANCE OPTIMIZATION

### 8. GPU Not Fully Utilized

**Check:**
```bash
nvidia-smi
# If GPU-Util < 70%, something is wrong
```

**Fixes:**

1. **Increase batch size**
   ```yaml
   data:
     batch_size: 32  # From 16
   ```

2. **Use torch.cuda.stream() for pipelining**
   ```python
   stream = torch.cuda.Stream()
   with torch.cuda.stream(stream):
       x = x.to(device)  # Async transfer
   ```

3. **Compile model (PyTorch 2.0+)**
   ```python
   model = torch.compile(model)  # Enable backend optimizations
   ```

---

### 9. Model Too Slow for Inference

**Latency issue:** Decoding takes > 100ms

**Fixes:**

1. **Reduce latent dim**
   ```python
   model_small = VAE({'latent_dim': 64, ...})
   ```

2. **Export to ONNX**
   ```python
   torch.onnx.export(model, dummy_input, 'model.onnx')
   # Use ONNX Runtime (often 2-3x faster)
   ```

3. **Quantize model**
   ```python
   quantized = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)
   ```

---

## 📋 CHECKLIST

Quick verification before training:

- [ ] Data loads without errors
- [ ] Model forward pass works
- [ ] Loss is decreasing (initial epochs)
- [ ] GPU memory < 90% capacity
- [ ] Gradients are non-zero and not NaN
- [ ] Batch size matches config
- [ ] Learning rate not too high (loss doesn't explode)
- [ ] Validation set is separate from training
- [ ] Checkpoint saving works
- [ ] TensorBoard logging enabled

---

## 🆘 If All Else Fails

1. **Start from scratch with minimal example**
   ```python
   # Simplest possible VAE
   model = VAE({'latent_dim': 32})
   model.to(device)
   
   x = torch.randn(4, 1, 128, 128).to(device)
   mu, logvar, z, recon = model(x)
   
   # Does this work? If not, debug model architecture
   ```

2. **Use Weights & Biases for remote debugging**
   ```python
   import wandb
   wandb.init(project="vae-debug")
   wandb.log({"debug_value": value})
   # View in W&B dashboard
   ```

3. **Enable PyTorch debug mode**
   ```python
   torch.autograd.set_detect_anomaly(True)
   # Will catch NaN/Inf earlier
   ```

4. **Contact maintainers** with:
   - Full error message
   - Config file used
   - Python/PyTorch versions
   - GPU info (`nvidia-smi`)
   - Minimal reproducible example

---

**Last Updated:** 17 janvier 2026
