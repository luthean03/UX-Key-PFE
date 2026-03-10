"""General utilities: reproducibility, checkpointing, SLERP."""

import os
import random

import numpy as np
import torch
import torch.nn


def set_reproducibility(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic=False allows faster non-deterministic cuDNN kernels.
    # benchmark=False because input sizes vary per batch (SmartBatchSampler);
    # benchmark=True would re-profile algorithms at every new (H,W) shape.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def slerp_numpy(z1: np.ndarray, z2: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two latent codes.

    Preserves the norm of latent vectors for smoother interpolation
    than linear lerp. Works with tensors of any shape by flattening first.
    """
    orig_shape = z1.shape
    # Aplatir pour le calcul géométrique (supporte tenseurs 1D, 3D, etc.)
    z1_flat = z1.flatten()
    z2_flat = z2.flatten()

    z1_norm = z1_flat / (np.linalg.norm(z1_flat) + 1e-8)
    z2_norm = z2_flat / (np.linalg.norm(z2_flat) + 1e-8)

    # Compute angle
    dot = np.clip(np.dot(z1_norm, z2_norm), -1.0 + 1e-6, 1.0 - 1e-6)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    # Nearly parallel: fall back to lerp
    if np.abs(sin_omega) < 1e-6:
        res = (1 - alpha) * z1_flat + alpha * z2_flat
        return res.reshape(orig_shape)

    # Interpolate and rescale
    scale1 = np.linalg.norm(z1_flat)
    scale2 = np.linalg.norm(z2_flat)
    scale = (1 - alpha) * scale1 + alpha * scale2

    z_interp = (np.sin((1 - alpha) * omega) / sin_omega) * z1_norm + \
               (np.sin(alpha * omega) / sin_omega) * z2_norm

    return (z_interp * scale).reshape(orig_shape)


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
