# Support module execution for model package utilities.

"""Quick smoke-test for VAE model building."""

import torch

from . import build_model


def test_vae():
    cfg = {
        "class": "VAE",
        "latent_dim": 32,
        "input_channels": 1,
        "dropout_p": 0.1,
    }
    input_size = (1, 256, 256)
    num_classes = 0
    model = build_model(cfg, input_size, num_classes)

    batch_size = 2
    x = torch.randn(batch_size, *input_size)
    mask = torch.ones(batch_size, 1, *input_size[1:])

    recon, mu, logvar = model(x, mask=mask)
    print(f"Input:  {x.shape}")
    print(f"Recon:  {recon.shape}")
    print(f"Mu:     {mu.shape}")
    print(f"Logvar: {logvar.shape}")


if __name__ == "__main__":
    test_vae()
