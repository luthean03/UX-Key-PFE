"""Optimizer and scheduler factory functions."""

import torch
import logging


def get_loss(lossname):
    return getattr(torch.nn, lossname)()


def get_optimizer(cfg, parameters):
    """Create an optimizer from a config dict."""
    algo = cfg.get("algo", "Adam")
    params = dict(cfg.get("params", {}))

    OptimClass = getattr(torch.optim, algo)
    optimizer = OptimClass(parameters, **params)
    return optimizer


def get_scheduler(optimizer, cfg: dict):
    """Create an LR scheduler from an optimizer config dict (returns None if absent)."""

    if not isinstance(cfg, dict) or "scheduler" not in cfg:
        return None

    sched_config = cfg.get("scheduler", {}) or {}
    name = str(sched_config.get("name", "ReduceLROnPlateau"))
    params = dict(sched_config.get("params", {}) or {})
    
    # Convert numeric strings to float (YAML parsing edge-case)
    for key in ['eta_min', 'min_lr', 'lr', 'T_0', 'T_mult']:
        if key in params and isinstance(params[key], str):
            try:
                params[key] = float(params[key])
            except ValueError:
                pass
    
    # Remove unsupported params for specific schedulers
    if name == "CosineAnnealingWarmRestarts" and "verbose" in params:
        params.pop("verbose")
    
    logging.info(f"Using Scheduler: {name} with params {params}")

    available_schedulers = {
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    }

    if name not in available_schedulers:
        raise ValueError(
            f"Unknown scheduler: {name}\n"
            f"Available: {list(available_schedulers.keys())}"
        )

    scheduler_class = available_schedulers[name]
    return scheduler_class(optimizer, **params)
