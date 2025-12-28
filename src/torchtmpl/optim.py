# coding: utf-8

# External imports
import torch


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
