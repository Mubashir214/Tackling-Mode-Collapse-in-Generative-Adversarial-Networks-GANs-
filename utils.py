import torch
import numpy as np
from torchvision import utils

def generate_images(model, z_dim=100, device="cpu", n=16):
    noise = torch.randn(n, z_dim, 1, 1).to(device)

    with torch.no_grad():
        fake = model(noise).cpu()

    grid = utils.make_grid(fake, normalize=True)
    img = np.transpose(grid.numpy(), (1, 2, 0))

    return img
