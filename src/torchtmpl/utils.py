# coding: utf-8

# Standard imports
import os

# External imports
import numpy as np
import torch
import torch.nn
import tqdm


# ===================== SLERP (Centralisé) =====================
def slerp_numpy(z1: np.ndarray, z2: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical Linear Interpolation (SLERP) between two latent codes (numpy).
    
    SLERP preserves the norm of latent vectors, providing smoother
    interpolations than linear interpolation.
    
    Args:
        z1: (latent_dim,) first latent code
        z2: (latent_dim,) second latent code  
        alpha: interpolation factor in [0, 1]
        
    Returns:
        z_interp: interpolated latent code with interpolated magnitude
    """
    # Normalize to unit vectors
    z1_norm = z1 / (np.linalg.norm(z1) + 1e-8)
    z2_norm = z2 / (np.linalg.norm(z2) + 1e-8)
    
    # Compute angle between vectors
    dot = np.clip(np.dot(z1_norm, z2_norm), -1.0 + 1e-6, 1.0 - 1e-6)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    
    # Handle nearly parallel vectors (use lerp instead)
    if np.abs(sin_omega) < 1e-6:
        return (1 - alpha) * z1 + alpha * z2
    
    # Scale back to interpolated magnitude
    scale1 = np.linalg.norm(z1)
    scale2 = np.linalg.norm(z2)
    scale = (1 - alpha) * scale1 + alpha * scale2
    
    z_interp = (np.sin((1 - alpha) * omega) / sin_omega) * z1_norm + \
               (np.sin(alpha * omega) / sin_omega) * z2_norm
    
    return z_interp * scale


def slerp_torch(z1: torch.Tensor, z2: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical Linear Interpolation (SLERP) between two latent codes (torch).
    
    SLERP preserves the norm of latent vectors, providing smoother
    interpolations than linear interpolation.
    
    Args:
        z1: (latent_dim,) or (B, latent_dim) first latent code
        z2: (latent_dim,) or (B, latent_dim) second latent code
        alpha: interpolation factor in [0, 1]
        
    Returns:
        z_interp: interpolated latent code with interpolated magnitude
    """
    # Normalize to unit vectors
    z1_norm = z1 / (z1.norm(dim=-1, keepdim=True) + 1e-8)
    z2_norm = z2 / (z2.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Compute angle between vectors
    dot = (z1_norm * z2_norm).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)  # Numerical stability
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    
    # Handle nearly parallel vectors (use lerp instead)
    if sin_omega.abs().min() < 1e-6:
        return (1 - alpha) * z1 + alpha * z2
    
    # Scale back to interpolated magnitude
    scale1 = z1.norm(dim=-1, keepdim=True)
    scale2 = z2.norm(dim=-1, keepdim=True)
    scale = (1 - alpha) * scale1 + alpha * scale2
    
    z_interp = (torch.sin((1 - alpha) * omega) / sin_omega) * z1_norm + \
               (torch.sin(alpha * omega) / sin_omega) * z2_norm
    
    return z_interp * scale


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

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
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0
    for i, (inputs, targets) in (pbar := tqdm.tqdm(enumerate(loader))):

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]
        pbar.set_description(f"Train loss : {total_loss/num_samples:.2f}")

    return total_loss / num_samples


def test(model, loader, f_loss, device):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0
    num_samples = 0
    for (inputs, targets) in loader:

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]

    return total_loss / num_samples
