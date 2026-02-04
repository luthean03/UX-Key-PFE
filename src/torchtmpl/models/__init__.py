# coding: utf-8

# External imports
import torch

# Local imports
from .vae_models import *


def build_model(cfg, input_size, num_classes):
    name = cfg.get("class")
    if not name:
        raise ValueError("Model config must include key 'class'")

    obj = globals().get(name)
    if obj is None:
        raise ValueError(f"Unknown model class: {name}")

    return obj(cfg, input_size, num_classes)
