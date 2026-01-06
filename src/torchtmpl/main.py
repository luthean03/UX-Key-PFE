"""Training entrypoint for torchtmpl.

Contains VAE training (with gradient accumulation, KL annealing, SSIM monitoring)
and classic model training.
"""

# coding: utf-8

# Standard imports
import logging
import os
import pathlib
import sys

# External imports
import yaml
import torch
import wandb
import torchinfo.torchinfo as torchinfo
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Optional visualization dependencies (imported lazily)
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None


# ==================== MIXUP / CUTMIX ====================
def mixup_data(x, y, alpha=0.2):
    """Apply Mixup augmentation.
    
    Args:
        x: Input images (batch)
        y: Target images (batch)
        alpha: Mixup parameter (higher = more mixing)
    
    Returns:
        Mixed inputs, mixed targets, lambda coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation.
    
    Args:
        x: Input images (batch)
        y: Target images (batch)
        alpha: CutMix parameter
    
    Returns:
        CutMix inputs, CutMix targets, lambda coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get random bounding box
    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    x_cutmix = x.clone()
    x_cutmix[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    y_cutmix = y.clone()
    y_cutmix[:, :, bby1:bby2, bbx1:bbx2] = y[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to reflect actual area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x_cutmix, y_cutmix, lam

# Local imports
from . import data
from . import loss as loss_module
from . import models
from . import optim
from . import utils
from . import latent_metrics


def _setup_wandb(config: dict):
    logging_cfg = config.get("logging", {})
    if isinstance(logging_cfg, dict) and "wandb" in logging_cfg:
        wandb_config = logging_cfg["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb.log(config)
        logging.info("Will be recording in wandb")
        return wandb.log
    return None


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


def _avg_ssim_on_loader(model, loader, device):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    total_ssim = 0.0
    ssim_count = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            recon = out[0] if isinstance(out, (tuple, list)) else out
            try:
                batch_val = ssim_metric(recon, targets)
                bs = inputs.shape[0]
                total_ssim += float(batch_val) * bs
                ssim_count += bs
            except Exception:
                pass
    return total_ssim / ssim_count if ssim_count > 0 else 0.0


def train(config):
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

    wandb_log = _setup_wandb(config)

    # Data
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(data_config, use_cuda)

    # Model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Loss
    logging.info("= Loss")
    model_is_vae = model_config.get("class", "").lower() == "vae" or model_config.get("type", "").lower() == "vae"
    loss = None
    if not model_is_vae:
        loss_cfg = config["loss"]
        loss_name = loss_cfg.get("name") if isinstance(loss_cfg, dict) else loss_cfg
        loss = optim.get_loss(loss_name)
    loss_display = (
        config.get("loss", {"name": "l1", "beta_kld": config.get("optim", {}).get("beta_kld", 0.001)})
        if model_is_vae
        else loss
    )

    # Optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    scheduler = optim.get_scheduler(optimizer, optim_config)

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

    wandb_name = None
    if wandb_log is not None and getattr(wandb, "run", None) is not None:
        wandb_name = getattr(wandb.run, "name", None)

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
        + (f" Wandb run name : {wandb_name}\n\n" if wandb_name else "")
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
    if wandb_log is not None:
        wandb.log({"summary": summary_text})
    if writer is not None:
        try:
            writer.add_text("summary", summary_text)
        except Exception:
            pass

    # Checkpoints
    model_checkpoint = utils.ModelCheckpoint(model, str(logdir / "best_model.pt"), min_is_best=True)

    # --- Training loop ---
    if model_is_vae:
        loss_config = config.get("loss", {"name": "l1", "beta_kld": config.get("optim", {}).get("beta_kld", 0.001)})
        criterion = loss_module.get_vae_loss(loss_config).to(device)

        target_beta = float(loss_config.get("beta_kld", 0.001))
        warmup_epochs = int(loss_config.get("warmup_epochs", 20))
        
        # AMÉLIORATION: Perceptual Weight Warmup (Option 2)
        target_perceptual_weight = float(loss_config.get("perceptual_weight", 0.08))
        perceptual_warmup_epochs = int(loss_config.get("perceptual_warmup_epochs", 30))
        min_perceptual_weight = float(loss_config.get("min_perceptual_weight", 0.01))
        logging.info(f"Perceptual Weight Warmup: {min_perceptual_weight:.4f} → {target_perceptual_weight:.4f} over {perceptual_warmup_epochs} epochs")

        optim_conf = config.get("optimization", {})
        grad_accumulation_steps = int(optim_conf.get("accumulation_steps", int(config.get("grad_accumulation_steps", 64))))
        logging.info(f"Training with Gradient Accumulation Steps: {grad_accumulation_steps}")
        
        # AMÉLIORATION: Mixed Precision Training (AMP)
        use_amp = bool(optim_conf.get("mixed_precision", False)) and use_cuda
        if use_amp:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            logging.info("Mixed Precision Training (AMP) enabled")
        else:
            scaler = None
            logging.info("Mixed Precision Training (AMP) disabled")

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
            
            # AMÉLIORATION: Perceptual Weight Annealing
            if perceptual_warmup_epochs > 0 and e < perceptual_warmup_epochs:
                current_perceptual_weight = min_perceptual_weight + (target_perceptual_weight - min_perceptual_weight) * (e / perceptual_warmup_epochs)
            else:
                current_perceptual_weight = target_perceptual_weight
            try:
                criterion.perceptual_weight = current_perceptual_weight
            except Exception:
                pass

            if e % 5 == 0:
                logging.info(f"Epoch {e}: Current KL Beta = {current_beta:.6f}, Perceptual Weight = {current_perceptual_weight:.6f}")
                if writer is not None:
                    try:
                        writer.add_scalar("kl_beta", current_beta, e)
                        writer.add_scalar("perceptual_weight", current_perceptual_weight, e)
                    except Exception:
                        pass
                if wandb_log is not None:
                    try:
                        wandb_log({"kl_beta": current_beta, "perceptual_weight": current_perceptual_weight})
                    except Exception:
                        pass

            model.train()
            train_total = 0.0
            train_samples = 0
            # AMÉLIORATION: Accumulateurs pour composantes de loss
            train_pixel_total = 0.0
            train_kld_total = 0.0
            train_vgg_total = 0.0
            
            # Récupère config Mixup/CutMix
            use_mixup = optim_conf.get("use_mixup", False)
            use_cutmix = optim_conf.get("use_cutmix", False)
            mixup_alpha = optim_conf.get("mixup_alpha", 0.2)
            cutmix_alpha = optim_conf.get("cutmix_alpha", 1.0)
            mix_prob = optim_conf.get("mix_prob", 0.5)  # Probabilité d'appliquer mix
            
            pbar = tqdm(train_loader, desc=f"Epoch {e}/{config['nepochs']}", dynamic_ncols=True)
            for i, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # AMÉLIORATION: Apply Mixup/CutMix avec probabilité
                apply_mix = (use_mixup or use_cutmix) and np.random.rand() < mix_prob
                if apply_mix:
                    if use_mixup and (not use_cutmix or np.random.rand() < 0.5):
                        inputs, targets, lam = mixup_data(inputs, targets, mixup_alpha)
                    elif use_cutmix:
                        inputs, targets, lam = cutmix_data(inputs, targets, cutmix_alpha)
                
                # AMÉLIORATION: Forward pass avec AMP si activé
                if use_amp:
                    with autocast():
                        recon, mu, logvar = model(inputs)
                        loss_result = criterion(recon, targets, mu, logvar)
                else:
                    recon, mu, logvar = model(inputs)
                    loss_result = criterion(recon, targets, mu, logvar)
                
                # Récupère toutes les composantes
                if len(loss_result) == 4:
                    total_loss, pixel_loss, kld_loss, vgg_loss = loss_result
                    train_pixel_total += pixel_loss.item()
                    train_kld_total += kld_loss.item()
                    train_vgg_total += vgg_loss.item()
                else:
                    total_loss, pixel_loss, kld_loss = loss_result
                    train_pixel_total += pixel_loss.item()
                    train_kld_total += kld_loss.item()

                # AMÉLIORATION: Backward pass avec AMP si activé
                if use_amp:
                    scaler.scale(total_loss / grad_accumulation_steps).backward()
                else:
                    (total_loss / grad_accumulation_steps).backward()
                
                if (i + 1) % grad_accumulation_steps == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    try:
                        pbar.set_postfix(loss=f"{total_loss.item():.4f}", beta=f"{current_beta:.4f}")
                    except Exception:
                        pass

                bs = inputs.shape[0]
                train_total += total_loss.item()
                train_samples += bs

            if (i + 1) % grad_accumulation_steps != 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss = train_total / max(1, train_samples)

            # Validation (loss + SSIM)
            model.eval()
            val_total = 0.0
            val_samples = 0
            # AMÉLIORATION: Accumulateurs validation
            val_pixel_total = 0.0
            val_kld_total = 0.0
            val_vgg_total = 0.0
            
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            total_ssim = 0.0
            ssim_count = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    recon, mu, logvar = model(inputs)
                    
                    # AMÉLIORATION: Récupère composantes validation
                    loss_result = criterion(recon, targets, mu, logvar)
                    if len(loss_result) == 4:
                        total_loss, pixel_loss, kld_loss, vgg_loss = loss_result
                        val_pixel_total += pixel_loss.item()
                        val_kld_total += kld_loss.item()
                        val_vgg_total += vgg_loss.item()
                    else:
                        total_loss, pixel_loss, kld_loss = loss_result
                        val_pixel_total += pixel_loss.item()
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
                if wandb_log is not None:
                    try:
                        wandb_log({"lr": current_lr})
                    except Exception:
                        pass

            if scheduler is not None:
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
                    sample_inputs, sample_targets = next(iter(valid_loader))
                    sample_inputs, sample_targets = sample_inputs.to(device), sample_targets.to(device)
                    sample_recon, _, _ = model(sample_inputs)
                    comparison = torch.cat([sample_targets.cpu(), sample_recon.cpu()])
                    from torchvision.utils import save_image, make_grid

                    save_image(comparison, logdir / f"reconstruction_epoch_{e}.png", nrow=1)
                    if writer is not None:
                        try:
                            writer.add_image("reconstructions", make_grid(comparison, nrow=1), e)
                        except Exception:
                            pass
            except Exception:
                logging.debug("Could not save reconstruction image for epoch %d", e)

            # AMÉLIORATION: Log des composantes détaillées
            train_pixel_avg = train_pixel_total / max(1, train_samples)
            train_kld_avg = train_kld_total / max(1, train_samples)
            val_pixel_avg = val_pixel_total / max(1, val_samples)
            val_kld_avg = val_kld_total / max(1, val_samples)
            
            metrics = {"train_ELBO": train_loss, "test_ELBO": test_loss, "test_SSIM": avg_ssim}
            
            # Ajout composantes dans metrics
            if train_vgg_total > 0:
                train_vgg_avg = train_vgg_total / max(1, train_samples)
                val_vgg_avg = val_vgg_total / max(1, val_samples)
                metrics.update({
                    "train_pixel_loss": train_pixel_avg,
                    "train_vgg_loss": train_vgg_avg,
                    "train_kld_loss": train_kld_avg,
                    "test_pixel_loss": val_pixel_avg,
                    "test_vgg_loss": val_vgg_avg,
                    "test_kld_loss": val_kld_avg,
                })
            else:
                metrics.update({
                    "train_pixel_loss": train_pixel_avg,
                    "train_kld_loss": train_kld_avg,
                    "test_pixel_loss": val_pixel_avg,
                    "test_kld_loss": val_kld_avg,
                })
            
            if wandb_log is not None:
                wandb_log(metrics)
            if writer is not None:
                writer.add_scalar("train_ELBO", train_loss, e)
                writer.add_scalar("test_ELBO", test_loss, e)
                try:
                    writer.add_scalar("test_SSIM", avg_ssim, e)
                except Exception:
                    pass
                
                # AMÉLIORATION: Log composantes séparées
                try:
                    writer.add_scalar("train/pixel_loss", train_pixel_avg, e)
                    writer.add_scalar("train/kld_loss", train_kld_avg, e)
                    writer.add_scalar("test/pixel_loss", val_pixel_avg, e)
                    writer.add_scalar("test/kld_loss", val_kld_avg, e)
                    if train_vgg_total > 0:
                        writer.add_scalar("train/vgg_loss", train_vgg_avg, e)
                        writer.add_scalar("test/vgg_loss", val_vgg_avg, e)
                except Exception:
                    pass
            
            # AMÉLIORATION: Métriques d'espace latent (tous les 10 epochs)
            if e % 10 == 0 or e == config["nepochs"] - 1:
                try:
                    archetypes_dir = data_config.get('archetypes_dir')
                    if archetypes_dir and pathlib.Path(archetypes_dir).exists() and writer is not None:
                        logging.info(f"Computing latent space metrics (epoch {e}) from {archetypes_dir}...")
                        latent_metrics.log_latent_space_visualization(
                            model, train_loader, archetypes_dir, device, writer, e
                        )
                    elif not archetypes_dir:
                        logging.debug("Skipping latent metrics: archetypes_dir not specified in config")
                except Exception as ex:
                    logging.warning(f"Latent metrics failed: {ex}")

        # Latent visualization
        if TSNE is not None:
            try:
                logging.info("Generating Latent Space Visualization...")
                model.eval()
                latents = []
                with torch.no_grad():
                    for i, (inputs, _) in enumerate(valid_loader):
                        if i >= 1000:
                            break
                        inputs = inputs.to(device)
                        _, mu, _ = model(inputs)
                        latents.append(mu.cpu().numpy())
                if latents:
                    latents = np.concatenate(latents, axis=0)
                    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
                    z_embedded = tsne.fit_transform(latents)
                    plt.figure(figsize=(8, 8))
                    plt.scatter(z_embedded[:, 0], z_embedded[:, 1], alpha=0.6, s=8)
                    plt.title("Latent Space Visualization (t-SNE)")
                    plt.grid(True, alpha=0.3)
                    save_path = logdir / "latent_space_tsne.png"
                    plt.savefig(save_path)
                    plt.close()
                    logging.info(f"Latent plot saved to {save_path}")
            except Exception as ex:
                logging.warning(f"Latent visualization failed: {ex}")
        else:
            logging.warning("TSNE not available (scikit-learn not installed); skipping latent visualization")

    else:
        # Cas des modèles classiques (CNN, ResNet, etc.)
        for e in range(config["nepochs"]):
            train_loss = utils.train(model, train_loader, loss, optimizer, device)
            test_loss = utils.test(model, valid_loader, loss, device)

            avg_ssim = None
            try:
                avg_ssim = _avg_ssim_on_loader(model, valid_loader, device)
                logging.info(f"Validation SSIM: {avg_ssim:.4f}")
            except Exception as ex:
                logging.warning(f"SSIM computation failed: {ex}")
                avg_ssim = None

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
                if wandb_log is not None:
                    try:
                        wandb_log({"lr": current_lr})
                    except Exception:
                        pass

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_loss)
                else:
                    scheduler.step()

            try:
                torch.save(model.state_dict(), str(logdir / "last_model.pt"))
            except Exception:
                logging.debug("Could not save last_model.pt")

            logging.info(
                "[%d/%d] Test loss : %.3f %s"
                % (e, config["nepochs"], test_loss, "[>> BETTER <<]" if updated else "")
            )

            metrics = {"train_CE": train_loss, "test_CE": test_loss}
            if avg_ssim is not None:
                metrics["test_SSIM"] = avg_ssim
            if wandb_log is not None:
                logging.info("Logging on wandb")
                wandb_log(metrics)
            if writer is not None:
                writer.add_scalar("train_CE", train_loss, e)
                writer.add_scalar("test_CE", test_loss, e)
                if avg_ssim is not None:
                    try:
                        writer.add_scalar("test_SSIM", avg_ssim, e)
                    except Exception:
                        pass

    try:
        if writer is not None:
            writer.close()
    except Exception:
        pass


def test(config):
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    with open(sys.argv[1], "r", encoding="utf-8") as cf:
        config = yaml.safe_load(cf)

    command = sys.argv[2]
    eval(f"{command}(config)")
