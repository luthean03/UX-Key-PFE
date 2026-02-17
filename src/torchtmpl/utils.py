"""General utilities: reproducibility, checkpointing, SLERP, train/test loops."""

import os
import random

import numpy as np
import torch
import torch.nn
import tqdm


def set_reproducibility(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enforce deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def slerp_numpy(z1: np.ndarray, z2: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two latent codes.

    Preserves the norm of latent vectors for smoother interpolation
    than linear lerp.
    """
    # Normalise to unit vectors
    z1_norm = z1 / (np.linalg.norm(z1) + 1e-8)
    z2_norm = z2 / (np.linalg.norm(z2) + 1e-8)
    
    # Compute angle
    dot = np.clip(np.dot(z1_norm, z2_norm), -1.0 + 1e-6, 1.0 - 1e-6)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    
    # Nearly parallel: fall back to lerp
    if np.abs(sin_omega) < 1e-6:
        return (1 - alpha) * z1 + alpha * z2
    
    # Interpolate and rescale
    scale1 = np.linalg.norm(z1)
    scale2 = np.linalg.norm(z2)
    scale = (1 - alpha) * scale1 + alpha * scale2
    
    z_interp = (np.sin((1 - alpha) * omega) / sin_omega) * z1_norm + \
               (np.sin(alpha * omega) / sin_omega) * z2_norm
    
    return z_interp * scale


def generate_unique_logpath(logdir, raw_run_name):
    """Return a non-existent path ``logdir/raw_run_name_N``."""
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint:
    """Save model weights when a new best score is achieved."""

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False


def train(model, loader, f_loss, optimizer, device, dynamic_display=True):
    """Train a model for one epoch and return the average loss."""

    model.train()

    total_loss = 0
    num_samples = 0
    for i, batch in (pbar := tqdm.tqdm(enumerate(loader))):
        inputs, targets = batch[0], batch[1]
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = f_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]
        pbar.set_description(f"Train loss : {total_loss/num_samples:.2f}")

    return total_loss / num_samples


def test(model, loader, f_loss, device):
    """Evaluate a model on the given loader and return the average loss."""

    model.eval()

    total_loss = 0
    num_samples = 0
    for batch in loader:
        inputs, targets = batch[0], batch[1]
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = f_loss(outputs, targets)

        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]

    return total_loss / num_samples
