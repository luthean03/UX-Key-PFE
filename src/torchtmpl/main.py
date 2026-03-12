"""Training entrypoint for torchtmpl."""

import logging
import os
import pathlib
import sys

# Set PyTorch memory allocator configuration for better memory management
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import yaml
import torch
import torch.nn.functional as F
import torchinfo.torchinfo as torchinfo
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from PIL import Image
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None


from . import data
from . import loss as loss_module
from . import models
from . import optim
from . import utils
from . import vizualisation


def _setup_tensorboard(config: dict, logdir: pathlib.Path):
    logging_cfg = config.get("logging", {})
    if isinstance(logging_cfg, dict) and "tensorboard" in logging_cfg:
        try:
            writer = SummaryWriter(log_dir=str(logdir))
            logging.info(f"TensorBoard enabled (logs -> {logdir})")
            return writer
        except Exception:
            logging.warning("Could not initialize TensorBoard SummaryWriter")
    return None


def _resolve_config_paths(config: dict, base_dir: pathlib.Path) -> dict:
    """Convert relative paths in config to absolute paths."""
    data_cfg = config.get("data", {})
    
    for key in ["data_dir", "archetypes_dir", "train_dir", "valid_dir"]:
        if key in data_cfg:
            path = data_cfg[key]
            if not os.path.isabs(path):
                data_cfg[key] = str(base_dir / path)
    
    # Resolve test paths
    test_cfg = config.get("test", {})
    for key in ["test_input_dir", "test_output_dir", "model_path"]:
        if key in test_cfg:
            path = test_cfg[key]
            if not os.path.isabs(path):
                test_cfg[key] = str(base_dir / path)
    
    # Resolve interpolate paths
    interpolate_cfg = config.get("interpolate", {})
    for key in ["image1_path", "image2_path", "output_dir", "model_path"]:
        if key in interpolate_cfg:
            path = interpolate_cfg[key]
            if not os.path.isabs(path):
                interpolate_cfg[key] = str(base_dir / path)
    
    return config


def train(config):
    # Set reproducibility
    seed = config.get("data", {}).get("seed", 42)
    utils.set_reproducibility(seed)
    
    # Resolve relative paths in config
    config = _resolve_config_paths(config, pathlib.Path(__file__).parent.parent.parent)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    try:
        if use_cuda:
            try:
                dev_idx = torch.cuda.current_device()
                dev_name = torch.cuda.get_device_name(dev_idx)
            except Exception:
                dev_name = str(device)
            logging.info(f"Using GPU device: {device} ({dev_name})")
        else:
            logging.info(f"Using CPU device: {device}")
    except Exception:
        logging.info(f"Using device: {device}")

    # Data
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(data_config, use_cuda)

    # Model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Resume from checkpoint if specified in config
    resume_path = config.get("resume")
    if resume_path:
        logging.info(f"Attempting to resume from: {resume_path}")
        if os.path.isfile(resume_path):
            try:
                # Load state dict
                state_dict = torch.load(resume_path, map_location=device)
                
                # Handle potential key mismatches (e.g. if model was saved with DataParallel)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k 
                    new_state_dict[name] = v
                    
                model.load_state_dict(new_state_dict)
                logging.info(f"Successfully loaded checkpoint: {resume_path}")
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
        else:
            logging.warning(f"Checkpoint file not found: {resume_path}")

    # NOTE: torch.compile is intentionally NOT used here.
    # Variable-size inputs cause constant recompilation, making it slower.

    # Loss
    logging.info("= Loss")
    loss_display = config.get("loss", {"name": "l1", "beta_kld": config.get("optim", {}).get("beta_kld", 0.001)})

    # Optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Scheduler (created later for OneCycleLR which needs steps_per_epoch)
    scheduler = None
    _sched_deferred = True  # resolved once we know len(train_loader)

    # Logdir
    logging_config = config["logging"]
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    os.makedirs(logdir, exist_ok=True)
    logdir = pathlib.Path(logdir)
    logging.info(f"Will be logging into {logdir}")

    writer = _setup_tensorboard(config, logdir)

    # Save config
    (logdir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    # Summary
    try:
        example_input_size = next(iter(train_loader))[0].shape
        arch_summary = str(torchinfo.summary(model, input_size=example_input_size))
    except Exception:
        arch_summary = "<no input to summarize>"

    def _ds_desc(loader):
        ds = getattr(loader, "dataset", None)
        if ds is None:
            return "<unknown>"
        inner = getattr(ds, "dataset", None)
        return str(inner) if inner is not None else str(ds)

    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + "## Summary of the model architecture\n"
        + f"{arch_summary}\n\n"
        + "## Loss\n\n"
        + f"{loss_display}\n\n"
        + "## Datasets : \n"
        + f"Train : {_ds_desc(train_loader)}\n"
        + f"Validation : {_ds_desc(valid_loader)}"
    )
    (logdir / "summary.txt").write_text(summary_text, encoding="utf-8")
    logging.info(summary_text)
    if writer is not None:
        try:
            writer.add_text("summary", summary_text)
        except Exception:
            pass

    # Checkpoints
    model_checkpoint = utils.ModelCheckpoint(model, str(logdir / "best_model.pt"), min_is_best=True)

    # --- Training loop ---
    loss_config = config.get("loss", {"name": "l1", "beta_kld": config.get("optim", {}).get("beta_kld", 0.001)})
    criterion = loss_module.get_vae_loss(loss_config).to(device)

    target_beta = float(loss_config.get("beta_kld", 0.001))
    warmup_epochs = int(loss_config.get("warmup_epochs", 20))

    # Reconstruction loss name for logging
    recon_loss_name = loss_config.get("name", "l1").lower()
    if recon_loss_name not in ["l1", "mse"]:
        recon_loss_name = "recon"

    optim_conf = config.get("optimization", {})
    grad_accumulation_steps = int(optim_conf.get("accumulation_steps", int(config.get("grad_accumulation_steps", 64))))
    grad_clip_max_norm = float(optim_conf.get("grad_clip_max_norm", 1.0))
    logging.info(f"Training with Gradient Accumulation Steps: {grad_accumulation_steps}")
    logging.info(f"Gradient clipping max_norm: {grad_clip_max_norm}")

    # Setup mixed precision training if enabled
    use_amp = bool(optim_conf.get("mixed_precision", False)) and use_cuda
    if use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler('cuda')
        logging.info("Mixed Precision Training (AMP) enabled")
    else:
        scaler = None
        logging.info("Mixed Precision Training (AMP) disabled")

    # SSIM metric (instantiated once, reused every epoch)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Deferred scheduler creation (needs len(train_loader) for OneCycleLR)
    if _sched_deferred:
        _accum = int(config.get("optimization", {}).get(
            "accumulation_steps", config.get("grad_accumulation_steps", 64)))
        _steps_per_epoch = max(1, len(train_loader) // _accum)
        scheduler = optim.get_scheduler(
            optimizer, optim_config,
            steps_per_epoch=_steps_per_epoch,
            epochs=config["nepochs"],
        )
        _is_batch_scheduler = isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ) if scheduler is not None else False
        if _is_batch_scheduler:
            logging.info("Scheduler will step every optimizer update (per-batch).")
        _sched_deferred = False

    optimizer.zero_grad()
    for e in range(config["nepochs"]):
        # KL annealing
        if warmup_epochs > 0 and e < warmup_epochs:
            current_beta = target_beta * (e / warmup_epochs)
        else:
            current_beta = target_beta
        try:
            criterion.beta = current_beta
        except Exception:
            pass

        if e % 5 == 0:
            logging.info(f"Epoch {e}: Current KL Beta = {current_beta:.6f}")
            if writer is not None:
                try:
                    writer.add_scalar("kl_beta", current_beta, e)
                except Exception:
                    pass
        model.train()
        train_total = 0.0
        train_samples = 0
        # Loss component accumulators
        train_recon_total = 0.0
        train_kld_total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {e}/{config['nepochs']}", dynamic_ncols=True)
        for i, (inputs, targets, masks) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Forward pass with automatic mixed precision if enabled
            if use_amp:
                with autocast('cuda'):
                    recon, mu, logvar = model(inputs, mask=masks)
                    loss_result = criterion(recon, targets, mu, logvar, mask=masks)
            else:
                recon, mu, logvar = model(inputs, mask=masks)
                loss_result = criterion(recon, targets, mu, logvar, mask=masks)

            # Unpack loss components
            total_loss, recon_loss, kld_loss = loss_result
            train_recon_total += recon_loss.detach().item()
            train_kld_total += kld_loss.detach().item()

            # Backward with gradient accumulation
            loss_for_backward = total_loss / grad_accumulation_steps
            if use_amp:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if (i + 1) % grad_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                # Per-batch scheduler step (OneCycleLR)
                if _is_batch_scheduler and scheduler is not None:
                    scheduler.step()

                try:
                    pbar.set_postfix(loss=f"{total_loss.detach().item():.4f}", beta=f"{current_beta:.4f}")
                except Exception:
                    pass

            bs = inputs.shape[0]
            train_total += total_loss.detach().item()
            train_samples += bs

            # Free memory
            del total_loss, recon_loss, kld_loss, loss_for_backward, recon, mu, logvar, inputs, targets, masks

        if (i + 1) % grad_accumulation_steps != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Per-batch scheduler step for the residual accumulation block
            if _is_batch_scheduler and scheduler is not None:
                scheduler.step()

        train_loss = train_total / max(1, train_samples)

        # Validation (loss + SSIM)
        model.eval()
        val_total = 0.0
        val_samples = 0
        # Validation loss component accumulators
        val_recon_total = 0.0
        val_kld_total = 0.0

        total_ssim = 0.0
        ssim_count = 0
        with torch.no_grad():
            for inputs, targets, masks in valid_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                # Use clean targets for validation (not augmented/noisy inputs)
                recon, mu, logvar = model(targets, mask=masks)

                # Unpack loss components
                total_loss, recon_loss, kld_loss = criterion(recon, targets, mu, logvar, mask=masks)
                val_recon_total += recon_loss.item()
                val_kld_total += kld_loss.item()
                val_total += total_loss.item()
                val_samples += inputs.shape[0]
                try:
                    batch_val = ssim_metric(recon, targets)
                    bs = inputs.shape[0]
                    total_ssim += float(batch_val) * bs
                    ssim_count += bs
                except Exception:
                    pass
        test_loss = val_total / max(1, val_samples)
        avg_ssim = total_ssim / ssim_count if ssim_count > 0 else 0.0
        logging.info(f"Validation SSIM: {avg_ssim:.4f}")

        updated = model_checkpoint.update(test_loss)

        # === LR logging + scheduler step ===
        current_lr = optimizer.param_groups[0].get("lr", None)
        if current_lr is not None:
            logging.info(f"Current Learning Rate: {current_lr}")
            if writer is not None:
                try:
                    writer.add_scalar("Learning Rate", current_lr, e)
                except Exception:
                    pass

        if scheduler is not None and not _is_batch_scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        try:
            torch.save(model.state_dict(), str(logdir / "last_model.pt"))
        except Exception:
            logging.debug("Could not save last_model.pt")

        logging.info(
            "[%d/%d] Train loss: %.6f  Test loss : %.6f %s"
            % (e, config["nepochs"], train_loss, test_loss, "[>> BETTER <<]" if updated else "")
        )

        # Recon preview
        try:
            with torch.no_grad():
                sample_inputs, sample_targets, sample_masks = next(iter(valid_loader))
                sample_inputs, sample_targets = sample_inputs.to(device), sample_targets.to(device)
                sample_masks = sample_masks.to(device)

                # Use clean targets for reconstruction preview (match test() behaviour)
                sample_recon, _, _ = model(sample_targets, mask=sample_masks)

                sample_recon = sample_recon * sample_masks
                sample_targets = sample_targets * sample_masks

                # Select first image in batch and crop to valid mask area
                idx = 0
                tgt = sample_targets[idx].cpu()  # (C, H, W)
                rec = sample_recon[idx].cpu()    # (C, H, W)
                msk = sample_masks[idx].cpu()    # (1, H, W)

                # Crop using mask logic
                non_zero = msk[0].nonzero()
                if non_zero.size(0) > 0:
                    h_end = non_zero[:, 0].max().item() + 1
                    w_end = non_zero[:, 1].max().item() + 1
                    tgt = tgt[:, :h_end, :w_end]
                    rec = rec[:, :h_end, :w_end]

                from torchvision.utils import save_image, make_grid

                # Stack target and reconstruction for side-by-side display
                comparison = torch.stack([tgt, rec])  # (2, C, H, W)

                # nrow=2 => Input | Recon (side by side)
                save_image(comparison, logdir / f"reconstruction_epoch_{e}.png", nrow=2)

                if writer is not None:
                    try:
                        writer.add_image("reconstructions", make_grid(comparison, nrow=2), e)
                    except Exception:
                        pass
        except Exception:
            logging.debug("Could not save reconstruction image for epoch %d", e)

        # Log loss components
        train_recon_avg = train_recon_total / max(1, train_samples)
        train_kld_avg = train_kld_total / max(1, train_samples)
        val_recon_avg = val_recon_total / max(1, val_samples)
        val_kld_avg = val_kld_total / max(1, val_samples)

        metrics = {
            "train_ELBO": train_loss,
            "test_ELBO": test_loss,
            "test_SSIM": avg_ssim,
            f"train_{recon_loss_name}_loss": train_recon_avg,
            "train_kld_loss": train_kld_avg,
            f"test_{recon_loss_name}_loss": val_recon_avg,
            "test_kld_loss": val_kld_avg,
        }

        if writer is not None:
            writer.add_scalar("train_ELBO", train_loss, e)
            writer.add_scalar("test_ELBO", test_loss, e)
            try:
                writer.add_scalar("test_SSIM", avg_ssim, e)
            except Exception:
                pass

            # Log loss components with dynamic names
            try:
                writer.add_scalar(f"train/{recon_loss_name}_loss", train_recon_avg, e)
                writer.add_scalar("train/kld_loss", train_kld_avg, e)
                writer.add_scalar(f"test/{recon_loss_name}_loss", val_recon_avg, e)
                writer.add_scalar("test/kld_loss", val_kld_avg, e)
            except Exception:
                pass


    try:
        if writer is not None:
            writer.close()
    except Exception:
        pass


def test(config):
    """Encode/decode test images and save side-by-side comparison grids."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"Using device: {device}")
    
    # Get test config
    test_config = config.get("test", {})
    test_input_dir = test_config.get("test_input_dir", "test_input")
    test_output_dir = test_config.get("test_output_dir", "test_output")
    model_path = test_config.get("model_path")
    
    if not model_path:
        raise ValueError("model_path must be specified in config['test']")
    
    # Create output directory
    os.makedirs(test_output_dir, exist_ok=True)
    logging.info(f"Output directory: {test_output_dir}")
    
    # Load model
    logging.info("= Loading Model")
    model_config = config["model"]
    input_size = (1, 128, 1024)  # Default: (C, H, W) for single-channel wireframes
    num_classes = 0
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    # Use strict=False to handle checkpoints from older model versions
    model.load_state_dict(checkpoint, strict=False)
    logging.info(f"Loaded model from: {model_path}")
    model.eval()
    
    input_path = pathlib.Path(test_input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {test_input_dir}")
    
    # Find PNG files
    image_files = sorted(list(input_path.glob("*.png"))) + sorted(list(input_path.glob("*.jpg")))
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {test_input_dir}")
    
    logging.info(f"Found {len(image_files)} images in {test_input_dir}")
    
    # Process each image
    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                # Load image
                img = Image.open(img_path).convert('L')
                w, h = img.size
                
                # Convert to tensor
                img_tensor = TF.to_tensor(img).unsqueeze(0)  # (1, 1, H, W)
                
                # Pad to multiple of 32
                stride = 32
                pad_h = ((h + stride - 1) // stride) * stride
                pad_w = ((w + stride - 1) // stride) * stride
                
                padded_img = torch.zeros(1, 1, pad_h, pad_w)
                mask = torch.zeros(1, 1, pad_h, pad_w)
                padded_img[0, :, :h, :w] = img_tensor[0]
                mask[0, 0, :h, :w] = 1.0
                
                # Encode-decode
                padded_img = padded_img.to(device)
                mask = mask.to(device)
                
                recon, _, _ = model(padded_img, mask=mask)
                
                # Crop reconstruction to original size
                recon = recon[0, 0, :h, :w].cpu()
                
                # Convert to PIL images
                original_pil = TF.to_pil_image(img_tensor[0, 0, :h, :w])
                recon_np = (recon.clamp(0, 1).numpy() * 255).astype(np.uint8)
                recon_pil = Image.fromarray(recon_np, mode='L')
                
                # Create side-by-side grid
                # Both images should have same height
                grid_width = original_pil.width + recon_pil.width + 20  # 20px gap
                grid_height = max(original_pil.height, recon_pil.height)
                
                grid_img = Image.new('L', (grid_width, grid_height), color=255)
                
                # Paste original on the left
                grid_img.paste(original_pil, (0, 0))
                
                # Paste reconstruction on the right (with 20px gap)
                grid_img.paste(recon_pil, (original_pil.width + 20, 0))
                
                # Save the grid
                output_filename = img_path.stem + "_comparison.png"
                output_path = os.path.join(test_output_dir, output_filename)
                grid_img.save(output_path)
                
                logging.info(f"Saved: {output_filename}")
                
            except Exception as e:
                logging.error(f"Error processing {img_path.name}: {e}")
                continue
    
    logging.info(f"Test completed. Results saved to {test_output_dir}")


def interpolate(config):
    """Interpolate between two images in latent space using SLERP."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"Using device: {device}")
    
    # Get interpolate config
    interp_config = config.get("interpolate", {})
    image1_path = interp_config.get("image1_path")
    image2_path = interp_config.get("image2_path")
    output_dir = interp_config.get("output_dir", "interpolate_output")
    num_steps = int(interp_config.get("num_steps", 10))
    model_path = interp_config.get("model_path")
    
    if not image1_path or not image2_path:
        raise ValueError("image1_path and image2_path must be specified in config['interpolate']")
    
    if not model_path:
        raise ValueError("model_path must be specified in config['interpolate']")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Load model
    logging.info("= Loading Model")
    model_config = config["model"]
    input_size = (1, 128, 1024)
    num_classes = 0
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    logging.info(f"Loaded model from: {model_path}")
    model.eval()
    
    images_pil = []
    latents = []
    orig_dims = []
    
    image_paths = [image1_path, image2_path]
    
    # 1. Find the bounding box across both images so z1 and z2 share the same shape
    max_h, max_w = 0, 0
    for img_path in image_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path)
        max_h = max(max_h, img.size[1])
        max_w = max(max_w, img.size[0])
    
    stride = 32
    pad_h_common = ((max_h + stride - 1) // stride) * stride
    pad_w_common = ((max_w + stride - 1) // stride) * stride
    
    with torch.no_grad():
        for img_idx, img_path in enumerate(image_paths):
            logging.info(f"Loading image {img_idx + 1}: {img_path}")
            
            # Load image
            img = Image.open(img_path).convert('L')
            w, h = img.size
            orig_dims.append((h, w))
            
            # Convert to tensor
            img_tensor = TF.to_tensor(img).unsqueeze(0)  # (1, 1, H, W)
            
            # 2. Use the COMMON padding so z1 and z2 have the same spatial dimensions
            padded_img = torch.zeros(1, 1, pad_h_common, pad_w_common)
            mask = torch.zeros(1, 1, pad_h_common, pad_w_common)
            padded_img[0, :, :h, :w] = img_tensor[0]
            mask[0, 0, :h, :w] = 1.0
            
            # Encode (z1 and z2 now guaranteed to have the same shape)
            _, mu, _ = model(padded_img.to(device), mask=mask.to(device))
            
            # Store latent and original image
            latents.append(mu[0].cpu().numpy())
            images_pil.append(img)
    
    z1, z2 = latents[0], latents[1]
    orig1_pil, orig2_pil = images_pil[0], images_pil[1]
    orig_h1, orig_w1 = orig_dims[0]
    orig_h2, orig_w2 = orig_dims[1]
    
    logging.info(f"Image 1 size: {orig_w1}x{orig_h1}, Image 2 size: {orig_w2}x{orig_h2}")
    logging.info(f"Latent shapes: z1={z1.shape}, z2={z2.shape}")
    
    # Build interpolation frames with geometric interpolation
    interp_frames = []
    
    with torch.no_grad():
        # 1. Original image 1
        interp_frames.append(orig1_pil)
        
        # 2. Reconstruction of image 1 (using full forward pass like test())
        img_tensor_1 = TF.to_tensor(orig1_pil).unsqueeze(0)
        stride = 32
        pad_h = ((orig_h1 + stride - 1) // stride) * stride
        pad_w = ((orig_w1 + stride - 1) // stride) * stride
        
        padded_img_1 = torch.zeros(1, 1, pad_h, pad_w)
        mask_1 = torch.zeros(1, 1, pad_h, pad_w)
        padded_img_1[0, :, :orig_h1, :orig_w1] = img_tensor_1[0]
        mask_1[0, 0, :orig_h1, :orig_w1] = 1.0
        
        recon1, _, _ = model(padded_img_1.to(device), mask=mask_1.to(device))
        recon1 = recon1[0, 0, :orig_h1, :orig_w1].cpu()
        recon1_np = (recon1.numpy() * 255).astype(np.uint8)
        recon1_pil = Image.fromarray(recon1_np, mode='L')
        interp_frames.append(recon1_pil)
        
        # SLERP interpolation in latent space
        logging.info(f"Generating {num_steps} interpolation steps with SLERP...")
        for step in range(num_steps):
            alpha = step / (num_steps - 1) if num_steps > 1 else 0.5
            
            # SLERP in latent space (creates new points between z1 and z2)
            z_interp = utils.slerp_numpy(z1, z2, alpha)
            
            # Log endpoint distances for verification
            if step == 0 or step == num_steps - 1:
                dist_to_z1 = np.linalg.norm(z_interp - z1)
                dist_to_z2 = np.linalg.norm(z_interp - z2)
                logging.info(f"  Step {step} (alpha={alpha:.2f}): dist to z1={dist_to_z1:.4f}, dist to z2={dist_to_z2:.4f}")
            
            # Geometric interpolation of output dimensions
            h_interp = int((1 - alpha) * orig_h1 + alpha * orig_h2)
            w_interp = int((1 - alpha) * orig_w1 + alpha * orig_w2)
            
            # Decode interpolated latent at the interpolated resolution
            z_interp_torch = torch.from_numpy(z_interp).float().unsqueeze(0).to(device)
            recon = model.decode(z_interp_torch, target_size=(h_interp, w_interp))[0, 0].cpu()
            recon_np = (recon.numpy() * 255).astype(np.uint8)
            recon_pil = Image.fromarray(recon_np, mode='L')
            interp_frames.append(recon_pil)
        
        # 4. Reconstruction of image 2 (using full forward pass like test())
        img_tensor_2 = TF.to_tensor(orig2_pil).unsqueeze(0)
        pad_h = ((orig_h2 + stride - 1) // stride) * stride
        pad_w = ((orig_w2 + stride - 1) // stride) * stride
        
        padded_img_2 = torch.zeros(1, 1, pad_h, pad_w)
        mask_2 = torch.zeros(1, 1, pad_h, pad_w)
        padded_img_2[0, :, :orig_h2, :orig_w2] = img_tensor_2[0]
        mask_2[0, 0, :orig_h2, :orig_w2] = 1.0
        
        recon2, _, _ = model(padded_img_2.to(device), mask=mask_2.to(device))
        recon2 = recon2[0, 0, :orig_h2, :orig_w2].cpu()
        recon2_np = (recon2.numpy() * 255).astype(np.uint8)
        recon2_pil = Image.fromarray(recon2_np, mode='L')
        interp_frames.append(recon2_pil)
        
        # 5. Original image 2
        interp_frames.append(orig2_pil)
    
    logging.info(f"Created {len(interp_frames)} frames for grid")
    
    # Create horizontal grid with 10px gap between images (keep original sizes)
    gap = 10
    
    # Calculate grid dimensions (max height, sum of widths)
    grid_height = max([f.height for f in interp_frames])
    total_width = sum(f.width for f in interp_frames) + gap * (len(interp_frames) - 1)
    
    logging.info(f"Grid dimensions: {total_width} x {grid_height}")
    
    # Create and populate grid
    grid_img = Image.new('L', (total_width, grid_height), color=255)
    
    x_offset = 0
    for frame in interp_frames:
        # Center vertically if frame is shorter than grid height
        y_offset = (grid_height - frame.height) // 2
        grid_img.paste(frame, (x_offset, y_offset))
        x_offset += frame.width + gap
    
    # Save grid
    output_filename = "interpolation.png"
    output_path = os.path.join(output_dir, output_filename)
    grid_img.save(output_path)
    
    logging.info(f"Interpolation completed. Grid saved to: {output_path}")
    logging.info(f"Grid size: {grid_img.size[0]} x {grid_img.size[1]} pixels")


def clustering(config):
    """Generate interactive latent-space clustering visualizations (PCA / t-SNE)."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"Using device: {device}")
    
    # Get clustering config
    cluster_config = config.get("clustering", {})
    model_path = cluster_config.get("model_path")
    data_dir = cluster_config.get("data_dir")
    output_dir = cluster_config.get("output_dir")
    max_samples = int(cluster_config.get("max_samples", 1000))
    n_clusters = int(cluster_config.get("n_clusters", 15))
    viz_method = cluster_config.get("viz_method", "both").lower()
    archetypes_dir = cluster_config.get("archetypes_dir")
    # Optional: directory of session JSON files used to derive ground-truth urlId labels
    gt_json_dir = cluster_config.get("gt_json_dir", None)
    
    if not model_path:
        raise ValueError("clustering.model_path must be specified in config")
    
    if not data_dir:
        raise ValueError("clustering.data_dir must be specified in config")
    
    # If output_dir is null, save in model directory
    if output_dir is None:
        model_dir = pathlib.Path(model_path).parent
        output_dir = str(model_dir / "clustering_viz")
        logging.info(f"Output directory not specified, using model directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Validate viz_method
    if viz_method not in ["pca", "tsne", "both"]:
        raise ValueError(f"viz_method must be 'pca', 'tsne', or 'both', got: {viz_method}")
    
    # Load model
    logging.info("= Loading Model")
    model_config = config["model"]
    input_size = (1, 128, 1024)
    num_classes = 0
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    logging.info(f"Loaded model from: {model_path}")
    model.eval()
    
    # Load and encode images from data_dir
    logging.info(f"Loading images from: {data_dir}")
    data_path = pathlib.Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    image_files = sorted(list(data_path.glob("*.png")) + list(data_path.glob("*.jpg")))
    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    logging.info(f"Encoding{len(image_files)} images...")
    
    # Encode images
    latents = []
    images = []
    image_names = []  # Store real filenames
    
    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Encoding images"):
            try:
                # Load image
                img = Image.open(img_path).convert('L')
                w, h = img.size
                
                # Convert to tensor
                import torchvision.transforms.functional as TF
                img_tensor = TF.to_tensor(img).unsqueeze(0)  # (1, 1, H, W)
                
                # Pad to multiple of 32 (same as training/validation)
                stride = 32
                pad_h = ((h + stride - 1) // stride) * stride
                pad_w = ((w + stride - 1) // stride) * stride
                
                padded_img = torch.zeros(1, 1, pad_h, pad_w)
                mask = torch.zeros(1, 1, pad_h, pad_w)
                padded_img[0, :, :h, :w] = img_tensor[0]
                mask[0, 0, :h, :w] = 1.0
                
                # Encode using the padded tensor and mask
                padded_img = padded_img.to(device)
                mask = mask.to(device)
                _, mu, _ = model(padded_img, mask=mask)

                # Adaptive Average Pooling on VALID region only (crop before pool)
                if mu.dim() == 4:
                    mask_latent = F.interpolate(mask, size=mu.shape[2:], mode='nearest')
                    valid_h = int(mask_latent[0, 0, :, 0].sum().item())
                    valid_w = int(mask_latent[0, 0, 0, :].sum().item())
                    valid_h = max(valid_h, 1)
                    valid_w = max(valid_w, 1)
                    mu_valid = mu[:, :, :valid_h, :valid_w]
                    mu_pooled = F.adaptive_avg_pool2d(mu_valid, output_size=(16, 1)).flatten(start_dim=1)
                    latents.append(mu_pooled.cpu().numpy())
                else:
                    latents.append(mu.cpu().numpy())
                images.append(img_tensor[0].cpu())  # Store original (unpadded) image
                image_names.append(img_path.stem)  # Store filename without extension
                
            except Exception as e:
                logging.warning(f"Failed to encode {img_path}: {e}")
                continue
    
    if len(latents) == 0:
        raise ValueError("No images were successfully encoded")
    
    latents = np.concatenate(latents, axis=0)  # (N, latent_dim)
    n_total = len(latents)
    latent_dim = latents.shape[1]
    
    logging.info(f"Encoded {n_total} images (latent_dim={latent_dim})")
    
    # Load archetypes if specified
    archetype_latents = None
    archetype_names = None
    archetype_images = None
    
    if archetypes_dir:
        logging.info(f"Loading archetypes from: {archetypes_dir}")
        archetype_latents, _, archetype_names, archetype_images = vizualisation.load_archetypes(
            archetypes_dir, model, device, max_height=2048
        )
        if archetype_latents is not None:
            logging.info(f"Loaded {len(archetype_latents)} archetypes")

    # Build ground-truth label map from supervised session JSONs (optional)
    # Mapping: PNG stem -> urlId  (e.g. "abc_3" -> "about-us")
    gt_label_map = {}
    if gt_json_dir:
        gt_label_map = vizualisation.build_ground_truth_label_map(gt_json_dir)

    # Attach a GT label to each encoded image (None when not in map)
    gt_labels_all = [gt_label_map.get(stem) for stem in image_names]
    has_gt = any(lbl is not None for lbl in gt_labels_all)
    if has_gt:
        n_labeled = sum(1 for lbl in gt_labels_all if lbl is not None)
        logging.info(f"Ground-truth labels found for {n_labeled}/{n_total} images")
    else:
        logging.info("No ground-truth labels found — skipping external clustering metrics")

    # Perform clustering on full latent space (K-Means + GMM + DBSCAN)
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    logging.info(f"Performing K-Means clustering (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(latents)

    logging.info(f"Performing GMM clustering (k={n_clusters})...")
    # Use float64 + reg_covar for numerical stability (high-dim latents with limited samples
    # make full covariance ill-conditioned; 'diag' is a robust fallback).
    latents_f64 = latents.astype(np.float64)
    gmm = GaussianMixture(
        n_components=n_clusters, random_state=42,
        covariance_type='diag', reg_covar=1e-3, n_init=3,
    )
    gmm_labels = gmm.fit_predict(latents_f64)

    # DBSCAN — auto-detect eps via nearest-neighbour heuristic
    logging.info("Performing DBSCAN clustering...")
    dbscan_cfg = cluster_config.get("dbscan", {})
    dbscan_eps = dbscan_cfg.get("eps", None)
    dbscan_min_samples = int(dbscan_cfg.get("min_samples", 5))

    latents_scaled = StandardScaler().fit_transform(latents)
    if dbscan_eps is None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=dbscan_min_samples)
        nn.fit(latents_scaled)
        distances, _ = nn.kneighbors(latents_scaled)
        dbscan_eps = float(np.percentile(distances[:, -1], 90))
        logging.info(f"DBSCAN auto eps={dbscan_eps:.4f} (90th-percentile of {dbscan_min_samples}-NN distances)")

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    dbscan_labels = dbscan.fit_predict(latents_scaled)
    # Remap noise label (-1) to its own cluster id so every point has a colour
    if np.any(dbscan_labels == -1):
        noise_label = dbscan_labels.max() + 1
        dbscan_labels[dbscan_labels == -1] = noise_label
    dbscan_n_clusters = int(dbscan_labels.max() + 1)
    logging.info(f"DBSCAN found {dbscan_n_clusters} clusters (eps={dbscan_eps:.4f}, min_samples={dbscan_min_samples})")

    # Assign archetypes to clusters if available
    kmeans_archetype_labels = None
    gmm_archetype_labels = None
    dbscan_archetype_labels = None
    if archetype_latents is not None:
        kmeans_archetype_labels = kmeans.predict(archetype_latents)
        gmm_archetype_labels = gmm.predict(archetype_latents)
        # DBSCAN has no predict(); assign each archetype to its nearest cluster centroid
        from sklearn.neighbors import NearestNeighbors
        archetype_scaled = StandardScaler().fit_transform(
            np.vstack([latents, archetype_latents])
        )[len(latents):]  # scale with same distribution
        nn_db = NearestNeighbors(n_neighbors=1).fit(latents_scaled)
        _, nn_idx = nn_db.kneighbors(archetype_scaled)
        dbscan_archetype_labels = dbscan_labels[nn_idx.flatten()]

    # Build clustering_results dict for the interactive visualization
    clustering_results = {
        "K-Means": {
            "labels": kmeans_labels,
            "archetype_labels": kmeans_archetype_labels,
            "n_clusters": n_clusters,
        },
        "GMM": {
            "labels": gmm_labels,
            "archetype_labels": gmm_archetype_labels,
            "n_clusters": n_clusters,
        },
        "DBSCAN": {
            "labels": dbscan_labels,
            "archetype_labels": dbscan_archetype_labels,
            "n_clusters": dbscan_n_clusters,
        },
    }

    # ------------------------------------------------------------------ #
    #  External clustering metrics (requires ground-truth urlId labels)   #
    # ------------------------------------------------------------------ #
    all_metrics = {}
    if has_gt:
        # Only evaluate on images that have a GT label
        gt_arr = np.array(gt_labels_all)
        labeled_mask = np.array([lbl is not None for lbl in gt_labels_all])

        for method_name, result in clustering_results.items():
            pred_labeled = result["labels"][labeled_mask]
            true_labeled = gt_arr[labeled_mask]
            all_metrics[method_name] = vizualisation.compute_clustering_metrics(
                true_labeled, pred_labeled, method_name
            )

    # Subsample for visualization (clustering already done on full dataset)
    if n_total > max_samples:
        logging.info(f"Subsampling {max_samples} points out of {n_total} for visualization (clustering was on all {n_total})...")
        rng = np.random.RandomState(42)
        viz_indices = rng.choice(n_total, size=max_samples, replace=False)
        viz_indices.sort()
        viz_latents = latents[viz_indices]
        viz_images = [images[i] for i in viz_indices]
        viz_image_names = [image_names[i] for i in viz_indices]
        viz_gt_labels = [gt_labels_all[i] for i in viz_indices] if has_gt else None
        viz_clustering_results = {}
        for method_name, result in clustering_results.items():
            viz_clustering_results[method_name] = {
                "labels": result["labels"][viz_indices],
                "archetype_labels": result["archetype_labels"],
                "n_clusters": result["n_clusters"],
            }
        n_samples = max_samples
    else:
        viz_latents = latents
        viz_images = images
        viz_image_names = image_names
        viz_gt_labels = gt_labels_all if has_gt else None
        viz_clustering_results = clustering_results
        n_samples = n_total

    logging.info(f"Visualization will display {n_samples} points (clustered on {n_total})")
    
    # Generate visualizations (PCA/t-SNE on subsampled data for speed)
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    # Determine which visualizations to generate
    viz_methods = []
    if viz_method in ["pca", "both"]:
        viz_methods.append("pca")
    if viz_method in ["tsne", "both"]:
        viz_methods.append("tsne")
    
    for method in viz_methods:
        logging.info(f"Generating {method.upper()} visualization on {n_samples} samples...")
        
        if method == "pca":
            # PCA projection on subsampled data
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3, random_state=42)
            z_embedded = pca.fit_transform(viz_latents)
            logging.info(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}, "
                        f"{pca.explained_variance_ratio_[1]:.3f}, {pca.explained_variance_ratio_[2]:.3f}")
            
            # Project archetypes
            archetype_embedded = None
            if archetype_latents is not None:
                archetype_embedded = pca.transform(archetype_latents)
        
        elif method == "tsne":
            # t-SNE projection on subsampled data
            if n_samples < 50:
                logging.warning(f"Skipping t-SNE: not enough samples ({n_samples} < 50)")
                continue
            
            from sklearn.manifold import TSNE
            perplexity = min(30.0, max(5.0, n_samples / 3))
            
            if archetype_latents is not None:
                # t-SNE doesn't have transform(), fit on combined data
                combined = np.vstack([viz_latents, archetype_latents])
                tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity,
                           init="pca", learning_rate="auto")
                z_combined = tsne.fit_transform(combined)
                z_embedded = z_combined[:n_samples]
                archetype_embedded = z_combined[n_samples:]
            else:
                tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity,
                           init="pca", learning_rate="auto")
                z_embedded = tsne.fit_transform(viz_latents)
                archetype_embedded = None
            logging.info(f"t-SNE completed with perplexity={perplexity}")
        
        # Create interactive 3D visualization
        output_filename = f"clustering_{method}_interactive.html"
        output_path = pathlib.Path(output_dir) / output_filename
        
        vizualisation.create_interactive_3d_visualization(
            z_embedded_3d=z_embedded,
            clustering_results=viz_clustering_results,
            archetype_embedded=archetype_embedded,
            archetype_names=archetype_names if archetype_names else [],
            train_images=viz_images,
            train_image_names=viz_image_names,
            archetype_images=archetype_images,
            viz_method=method.upper(),
            epoch=0,
            n_samples=n_samples,
            latent_dim=latent_dim,
            output_path=output_path,
            gt_labels=viz_gt_labels,
            metrics=all_metrics if all_metrics else None,
        )
        
        logging.info(f"Saved {method.upper()} visualization to: {output_path}")
    
    logging.info(f"Clustering completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test|interpolate|clustering>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    with open(sys.argv[1], "r", encoding="utf-8") as cf:
        config = yaml.safe_load(cf)

    command = sys.argv[2]
    _COMMANDS = {
        "train": train,
        "test": test,
        "interpolate": interpolate,
        "clustering": clustering,
    }
    if command not in _COMMANDS:
        logging.error(f"Unknown command '{command}'. Choose from: {', '.join(_COMMANDS)}")
        sys.exit(-1)
    _COMMANDS[command](config)
