# coding: utf-8

# External imports
import torch
import logging


def get_loss(lossname):
    return getattr(torch.nn, lossname)()


def get_optimizer(cfg, parameters):
    """
    Create optimizer from `cfg` dict.

    Expected cfg shape:
    {
       'algo': 'Adam',
       'params': { 'lr': 1e-3, 'weight_decay': 0.0, ... }
    }
    """
    algo = cfg.get("algo", "Adam")
    params = dict(cfg.get("params", {}))

    # Allow weight_decay to be specified (default 0.0)
    # Torch will ignore unknown keys for the optimizer constructor, so pass params as-is
    OptimClass = getattr(torch.optim, algo)
    optimizer = OptimClass(parameters, **params)
    return optimizer


def get_scheduler(optimizer, cfg: dict):
    """Create LR scheduler from optimizer config.

    Expected config shape:
    {
      'scheduler': { 'name': 'ReduceLROnPlateau', 'params': {...} }
    }
    """

    if not isinstance(cfg, dict) or "scheduler" not in cfg:
        return None

    sched_config = cfg.get("scheduler", {}) or {}
    name = str(sched_config.get("name", "ReduceLROnPlateau"))
    params = dict(sched_config.get("params", {}) or {})
    logging.info(f"Using Scheduler: {name} with params {params}")

    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)

    logging.warning(f"Unknown scheduler: {name} (scheduler disabled)")
    return None
