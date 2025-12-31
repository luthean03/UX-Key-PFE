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

# Local imports
from . import data
from . import loss as loss_module
from . import models
from . import optim
from . import utils


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
        loss = optim.get_loss(config["loss"])
    loss_display = (
        config.get("loss", {"name": "l1", "beta_kld": config.get("optim", {}).get("beta_kld", 0.001)})
        if model_is_vae
        else loss
    )

    # Optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

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

        optim_conf = config.get("optimization", {})
        grad_accumulation_steps = int(optim_conf.get("accumulation_steps", int(config.get("grad_accumulation_steps", 64))))
        logging.info(f"Training with Gradient Accumulation Steps: {grad_accumulation_steps}")

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
                if wandb_log is not None:
                    try:
                        wandb_log({"kl_beta": current_beta})
                    except Exception:
                        pass

            model.train()
            train_total = 0.0
            train_samples = 0
            pbar = tqdm(train_loader, desc=f"Epoch {e}/{config['nepochs']}", dynamic_ncols=True)
            for i, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                recon, mu, logvar = model(inputs)
                total_loss, _, _ = criterion(recon, targets, mu, logvar)

                (total_loss / grad_accumulation_steps).backward()
                if (i + 1) % grad_accumulation_steps == 0:
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss = train_total / max(1, train_samples)

            # Validation (loss + SSIM)
            model.eval()
            val_total = 0.0
            val_samples = 0
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            total_ssim = 0.0
            ssim_count = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    recon, mu, logvar = model(inputs)
                    total_loss, _, _ = criterion(recon, targets, mu, logvar)
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

            metrics = {"train_ELBO": train_loss, "test_ELBO": test_loss, "test_SSIM": avg_ssim}
            if wandb_log is not None:
                wandb_log(metrics)
            if writer is not None:
                writer.add_scalar("train_ELBO", train_loss, e)
                writer.add_scalar("test_ELBO", test_loss, e)
                try:
                    writer.add_scalar("test_SSIM", avg_ssim, e)
                except Exception:
                    pass

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
