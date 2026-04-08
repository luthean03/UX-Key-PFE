# Instantiate optimizers and schedulers from configuration with scheduler-specific safeguards.

"""Optimizer and scheduler factory functions."""

import torch
import logging


from typing import Optional


def get_optimizer(cfg, parameters):
    """Create an optimizer from a config dict."""
    algo = cfg.get("algo", "Adam")
    params = dict(cfg.get("params", {}))

    # Dynamically resolve optimizer class from torch.optim.
    OptimClass = getattr(torch.optim, algo)
    optimizer = OptimClass(parameters, **params)
    return optimizer


def get_scheduler(optimizer, cfg: dict, steps_per_epoch: Optional[int] = None, epochs: Optional[int] = None):
    """Create an LR scheduler from an optimizer config dict.

    Args:
        optimizer: The optimizer instance.
        cfg: The full ``optim`` config dict (must contain a ``scheduler`` key).
        steps_per_epoch: Number of *optimizer steps* per epoch (required for
            ``OneCycleLR``).  Equals ``ceil(len(loader) / accumulation_steps)``.
        epochs: Total number of training epochs (required for ``OneCycleLR``).

    Returns:
        A scheduler instance, or *None* if the config has no ``scheduler`` key.
    """

    if not isinstance(cfg, dict) or "scheduler" not in cfg:
        # Scheduler is optional; return None when not configured.
        return None

    sched_config = cfg.get("scheduler", {}) or {}
    name = str(sched_config.get("name", "ReduceLROnPlateau"))
    params = dict(sched_config.get("params", {}) or {})


    # Normalize numeric strings that can come from YAML parsing edge cases.
    for key in ['eta_min', 'min_lr', 'lr', 'T_0', 'T_mult',
                'max_lr', 'pct_start', 'div_factor', 'final_div_factor']:
        if key in params and isinstance(params[key], str):
            try:
                params[key] = float(params[key])
            except ValueError:
                pass


    # Remove unsupported args for schedulers that do not accept them.
    if name in ("CosineAnnealingWarmRestarts", "OneCycleLR") and "verbose" in params:
        params.pop("verbose")

    logging.info(f"Using Scheduler: {name} with params {params}")




    if name == "OneCycleLR":
        # OneCycle is step-based, so we need global step count up front.
        if steps_per_epoch is None or epochs is None:
            raise ValueError(
                "OneCycleLR requires steps_per_epoch and epochs to be "
                "passed to get_scheduler()."
            )

        # Use optimizer LR as fallback when max_lr is omitted.
        if "max_lr" not in params:
            params["max_lr"] = optimizer.param_groups[0]["lr"]
        params.setdefault("pct_start", 0.3)
        params.setdefault("div_factor", 25.0)
        params.setdefault("final_div_factor", 1e4)
        params.setdefault("anneal_strategy", "cos")

        total_steps = steps_per_epoch * epochs
        params["total_steps"] = total_steps
        logging.info(
            f"OneCycleLR: {steps_per_epoch} steps/epoch × {epochs} epochs "
            f"= {total_steps} total steps"
        )
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, **params)




    # Epoch-based schedulers supported by this project.
    available_schedulers = {
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    }

    if name not in available_schedulers:
        raise ValueError(
            f"Unknown scheduler: {name}\n"
            f"Available: {list(available_schedulers.keys()) + ['OneCycleLR']}"
        )

    scheduler_class = available_schedulers[name]
    return scheduler_class(optimizer, **params)
