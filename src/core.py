"""CPU-first physics, synthetic data, and small residual networks."""
import numpy as np
import torch
from torch import nn
from deepinv.physics import Tomography
from deepinv.utils import RandomPhantomDataset

PHANTOM_GENERATOR = "deepinv.RandomPhantomDataset"


def physics(size=64, views=30):
    return Tomography(
        angles=torch.arange(views, dtype=torch.float32) * (180.0 / views),
        img_width=size, circle=False, normalize=False,
        adjoint_via_backprop=True, parallel_computation=True, device="cpu",
    )


def phantoms(count, size, seed):
    """Generate a repeatable batch of DeepInv phantoms, shaped [N, 1, H, W]."""
    # DeepInv uses NumPy's global RNG. Restore it afterwards so generating
    # images does not change random choices elsewhere in the program.
    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        dataset = RandomPhantomDataset(
            length=count, size=size, n_data=1, use_dict_output=True,
        )
        # Each access generates a new image: sample once and keep the tensors.
        images = [dataset[index]["x"] for index in range(count)]
        return torch.stack(images)
    finally:
        np.random.set_state(random_state)


def psnr(pred, target):
    """Per-image PSNR, data range 1, full square image, no output clipping."""
    mse = (pred - target).square().flatten(1).mean(1)
    return -10 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).tiny))


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())


class MultiScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2 = block(1, 8), block(8, 16)
        self.bottom = block(16, 32)
        self.up2, self.up1 = nn.ConvTranspose2d(32, 16, 2, 2), nn.ConvTranspose2d(16, 8, 2, 2)
        self.dec2, self.dec1 = block(32, 16), block(16, 8)
        self.out = nn.Conv2d(8, 1, 1)
        self.pool = nn.MaxPool2d(2)
        # Both networks start at the FBP baseline.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.bottom(self.pool(b))
        d = self.dec2(torch.cat([self.up2(c), b], dim=1))
        e = self.dec1(torch.cat([self.up1(d), a], dim=1))
        return x + self.out(e)


class SingleScale(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(1, 26, 3, padding=1), nn.ReLU()]
        for _ in range(5):
            layers += [nn.Conv2d(26, 26, 3, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(26, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def model(name):
    return {"multiscale": MultiScale, "single_scale": SingleScale}[name]()
