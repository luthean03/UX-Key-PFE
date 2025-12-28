# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

# Optional visualization dependencies (imported lazily)
import matplotlib.pyplot as plt
import numpy as np
try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None

# Local imports
from . import data
from . import models
from . import optim
from . import utils
from . import loss as loss_module


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Log device information (CPU vs GPU) and device name
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

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info("Will be recording in wandb")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss (only for non-VAE models)
    logging.info("= Loss")
    model_is_vae = model_config.get("class", "").lower() == "vae" or model_config.get("type", "").lower() == "vae"
    loss = None
    if not model_is_vae:
        loss = optim.get_loss(config["loss"])

    # Prepare a human-friendly loss description for the summary (VAE uses config)
    if model_is_vae:
        loss_display = config.get("loss", {"name": "l1", "beta_kld": config.get("optim", {}).get("beta_kld", 0.001)})
    else:
        loss_display = loss

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # TensorBoard writer (optional)
    writer = None
    if "tensorboard" in logging_config:
        try:
            writer = SummaryWriter(log_dir=str(logdir))
            logging.info(f"TensorBoard enabled (logs -> {logdir})")
        except Exception:
            logging.warning("Could not initialize TensorBoard SummaryWriter")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    input_size = next(iter(train_loader))[0].shape
    # Safe wandb name and dataset descriptions for the summary
    wandb_name = None
    if wandb_log is not None and getattr(wandb, 'run', None) is not None:
        wandb_name = getattr(wandb.run, 'name', None)

    def _ds_desc(loader):
        ds = getattr(loader, 'dataset', None)
        if ds is None:
            return "<unknown>"
        # For Subset, show the wrapped dataset
        inner = getattr(ds, 'dataset', None)
        return str(inner) if inner is not None else str(ds)

    train_desc = _ds_desc(train_loader)
    valid_desc = _ds_desc(valid_loader)

    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb_name}\n\n" if wandb_name else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss_display}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_desc}\n"
        + f"Validation : {valid_desc}"
    )
    with open(logdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})
    if writer is not None:
        # also write the summary as text to TB
        try:
            writer.add_text("summary", summary_text)
        except Exception:
            pass

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    # --- Training loop ---
    if model_is_vae:
        # Load loss config from YAML (falls back to l1 + small beta)
        loss_config = config.get("loss", {"name": "l1", "beta_kld": config.get("optim", {}).get("beta_kld", 0.001)})
        criterion = loss_module.get_vae_loss(loss_config).to(device)

        # KL Annealing parameters (can be tuned from config)
        target_beta = float(loss_config.get("beta_kld", 0.001))
        warmup_epochs = int(loss_config.get("warmup_epochs", 20))

        logging.info(f"Using VAE Loss: {loss_config.get('name')} with target_beta={target_beta} and warmup_epochs={warmup_epochs}")

        # Gradient accumulation: simulate large batch size with batch_size=1 loaders
        grad_accumulation_steps = int(config.get("grad_accumulation_steps", 64))
        logging.info(f"Using gradient accumulation steps: {grad_accumulation_steps}")

        optimizer.zero_grad()
        for e in range(config["nepochs"]):
            # === CALCUL DYNAMIQUE DU BETA (ANNEALING) ===
            if e < warmup_epochs and warmup_epochs > 0:
                current_beta = target_beta * (e / warmup_epochs)
            else:
                current_beta = target_beta

            # Update the criterion internal beta (both SimpleVAELoss and VGGPerceptualLoss expose .beta)
            try:
                criterion.beta = current_beta
            except Exception:
                pass

            # log beta to tensorboard/wandb occasionally
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

            
            
            # Train one epoch (custom VAE loop) with gradient accumulation
            model.train()
            train_total = 0.0
            train_samples = 0

            # === MODIFICATION TQDM ===
            # On enveloppe le loader dans tqdm pour avoir la barre
            pbar = tqdm(train_loader, desc=f"Epoch {e}/{config['nepochs']}", dynamic_ncols=True)

            for i, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(device)
                targets = targets.to(device)
                recon, mu, logvar = model(inputs)

                # Compare reconstruction to the CLEAN target (denoising)
                total_loss, recon_loss, kld_loss = criterion(recon, targets, mu, logvar)

                # Normalize loss for accumulation and backprop
                loss_normalized = total_loss / grad_accumulation_steps
                loss_normalized.backward()

                # === GRADIENT CLIPPING ===
                if (i + 1) % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    # === AJOUT : MISE A JOUR DES INFOS SUR LA BARRE ===
                    # Affiche la loss courante et le beta KL à côté de la barre
                    try:
                        pbar.set_postfix(loss=f"{total_loss.item():.4f}", beta=f"{current_beta:.4f}")
                    except Exception:
                        pass
                    # ==================================================

                batch_size = inputs.shape[0]
                train_total += total_loss.item()
                train_samples += batch_size

            # If number of batches is not divisible by accumulation steps, do a final step
            if (i + 1) % grad_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss = train_total / max(1, train_samples)

            # Validation
            model.eval()
            val_total = 0.0
            val_samples = 0
            with torch.no_grad():
                for (inputs, targets) in valid_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    recon, mu, logvar = model(inputs)
                    total_loss, _, _ = criterion(recon, targets, mu, logvar)
                    val_total += total_loss.item()
                    val_samples += inputs.shape[0]

            test_loss = val_total / max(1, val_samples)

            updated = model_checkpoint.update(test_loss)
            logging.info(
                "[%d/%d] Train loss: %.6f  Test loss : %.6f %s"
                % (
                    e,
                    config["nepochs"],
                    train_loss,
                    test_loss,
                    "[>> BETTER <<]" if updated else "",
                )
            )

            # === AJOUT VISUALISATION ===
            # Save a small grid of original vs reconstruction for quick checks
            try:
                with torch.no_grad():
                    sample_inputs, _ = next(iter(valid_loader))
                    sample_inputs = sample_inputs.to(device)
                    sample_recon, _, _ = model(sample_inputs)

                    # Save the original and reconstruction side-by-side (batch_size=1)
                    comparison = torch.cat([sample_inputs.cpu(), sample_recon.cpu()])

                    from torchvision.utils import save_image, make_grid

                    save_image(comparison, logdir / f"reconstruction_epoch_{e}.png", nrow=1)
                    # Also log to TensorBoard (if available)
                    if writer is not None:
                        try:
                            grid = make_grid(comparison, nrow=1)
                            writer.add_image("reconstructions", grid, e)
                        except Exception:
                            pass
            except Exception:
                logging.debug("Could not save reconstruction image for epoch %d", e)
            # ===========================

            metrics = {"train_ELBO": train_loss, "test_ELBO": test_loss}
            if wandb_log is not None:
                wandb_log(metrics)
            if writer is not None:
                writer.add_scalar("train_ELBO", train_loss, e)
                writer.add_scalar("test_ELBO", test_loss, e)

    else:
        for e in range(config["nepochs"]):
            # Train 1 epoch
            train_loss = utils.train(model, train_loader, loss, optimizer, device)

            # Test
            test_loss = utils.test(model, valid_loader, loss, device)

            updated = model_checkpoint.update(test_loss)
            logging.info(
                "[%d/%d] Test loss : %.3f %s"
                % (
                    e,
                    config["nepochs"],
                    test_loss,
                    "[>> BETTER <<]" if updated else "",
                )
            )

            # Update the dashboard
            metrics = {"train_CE": train_loss, "test_CE": test_loss}
            if wandb_log is not None:
                logging.info("Logging on wandb")
                wandb_log(metrics)
            if writer is not None:
                writer.add_scalar("train_CE", train_loss, e)
                writer.add_scalar("test_CE", test_loss, e)

    # === Latent space visualization helper ===
    def generate_latent_plot(model, dataloader, device, logdir, max_points=1000):
        if TSNE is None:
            logging.warning("TSNE not available (scikit-learn not installed); skipping latent visualization")
            return
        model.eval()
        latents = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                if i >= max_points:
                    break
                inputs = inputs.to(device)
                # For most VAE implementations forward returns (recon, mu, logvar)
                _, mu, _ = model(inputs)
                latents.append(mu.cpu().numpy())

        if len(latents) == 0:
            logging.warning("No latents collected for visualization")
            return

        latents = np.concatenate(latents, axis=0)
        # Run t-SNE to 2D
        try:
            tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
            z_embedded = tsne.fit_transform(latents)
        except Exception as ex:
            logging.warning(f"t-SNE failed: {ex}")
            return

        plt.figure(figsize=(8, 8))
        plt.scatter(z_embedded[:, 0], z_embedded[:, 1], alpha=0.6, s=8)
        plt.title("Latent Space Visualization (t-SNE)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(True, alpha=0.3)
        save_path = pathlib.Path(logdir) / "latent_space_tsne.png"
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Latent plot saved to {save_path}")

    # Close TensorBoard writer if opened (no-op otherwise)
    # Optionally generate latent visualization for VAE models
    try:
        if model_is_vae:
            logging.info("Generating Latent Space Visualization...")
            try:
                generate_latent_plot(model, valid_loader, device, logdir)
            except Exception as ex:
                logging.warning(f"Latent visualization failed: {ex}")
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
