"""Synthetic 3D ellipsoid dataset with guaranteed nonzero local variance."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from monai.data import Dataset

from .registry import register_dataset


# -------------------------------------------------------------------------
# Synthetic 3D ellipsoid generator with guaranteed nonzero local variance
# -------------------------------------------------------------------------

class SyntheticEllipsoidsGenerator:
    """
    Produces synthetic ellipsoids with guaranteed nonzero local variance:
    - Main ellipsoid defines shape
    - Soft boundary attenuation preserves shape
    - A spatially smooth random field adds guaranteed textural variation
    """

    def __init__(
        self,
        num_samples: int,
        image_size: Tuple[int, int, int],
        noise_std: float,
        smooth: bool,
        seed: int,
    ):
        self.num_samples = int(num_samples)
        self.image_size = tuple(int(s) for s in image_size)
        self.noise_std = float(noise_std)
        self.smooth = bool(smooth)

        # coordinate grid
        coords = [
            torch.linspace(-1.0, 1.0, steps=s, dtype=torch.float32)
            for s in self.image_size
        ]
        zz, yy, xx = torch.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
        self.grid = torch.stack([zz, yy, xx], dim=0)

        # random state
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    # ----------------------------------------------------------
    # Main ellipsoid
    # ----------------------------------------------------------
    def _one_ellipsoid(self) -> torch.Tensor:
        g = self.grid

        cx, cy, cz = self.rng.uniform(-0.25, 0.25, 3)
        rx, ry, rz = self.rng.uniform(0.35, 0.55, 3)

        val = ((g[0] - cx) / rz)**2 + ((g[1] - cy) / ry)**2 + ((g[2] - cz) / rx)**2
        mask = (val <= 1.0).float()

        if self.smooth:
            mask = F.avg_pool3d(mask[None, None], 3, stride=1, padding=1)[0, 0]

        return mask

    # ----------------------------------------------------------
    # Smooth random field guaranteeing local nonzero variance
    # ----------------------------------------------------------
    def _smooth_random_field(self) -> torch.Tensor:
        rnd = torch.rand(self.image_size, dtype=torch.float32)

        # multiple smoothing passes = Gaussian-like effect
        for _ in range(3):
            rnd = F.avg_pool3d(rnd[None, None], 5, stride=1, padding=2)[0, 0]

        # normalize to [0,1]
        rnd -= rnd.min()
        rnd /= rnd.max() + 1e-8

        return rnd

    # ----------------------------------------------------------
    # Final sample
    # ----------------------------------------------------------
    def get_sample(self) -> dict:
        base = self._one_ellipsoid()
        texture = self._smooth_random_field()

        # amplify texture near boundaries to guarantee LNCC signal
        boundary = F.conv3d(
            base[None, None],
            weight=torch.ones(1, 1, 5, 5, 5),
            padding=2
        )[0, 0]
        boundary = (boundary > 0) & (boundary < 125)  # inside 5×5×5 window
        boundary = boundary.float()

        # mix components:
        # - inside ellipsoid: mild randomness
        # - boundary: stronger randomness
        # - outside: weak randomness
        img = (
            base * (0.7 + 0.3 * texture) +
            boundary * 0.4 * texture +
            (1 - base) * 0.1 * texture
        )

        # additional global noise
        img = img + torch.randn_like(img) * self.noise_std
        img = img.clamp(0.0, 1.0)

        return {
            "moving": img[None],
            "fixed": img[None].clone(),  # symmetrically defined; model must warp
        }


# -------------------------------------------------------------------------
# MONAI Dataset wrapper
# -------------------------------------------------------------------------

class SyntheticEllipsoidsMonaiDataset(Dataset):
    def __init__(self, generator, transforms=None):
        self.generator = generator
        self.transforms = transforms
        self.data = [{"idx": i} for i in range(len(generator))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.generator.get_sample()
        if self.transforms:
            sample = self.transforms(sample)
        return sample


# -------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------

@register_dataset("synthetic_ellipsoids")
def create_synthetic_ellipsoids(
    split: str,
    num_samples: int,
    image_size=(64, 64, 64),
    noise_std: float = 0.03,
    smooth: bool = True,
    seed: int = 123,
    transforms=None,
):
    """
    Factory for Synthetic Ellipsoids dataset.

    Config example:
        train_dataset:
          name: synthetic_ellipsoids
          image_size: [64, 64, 64]
          num_samples: 4000
          noise_std: 0.03
          smooth: true
          seed: 123
    """
    if split.lower() == "train":
        s = seed
    elif split.lower() == "val":
        s = seed + 1
    else:
        s = seed + 2

    generator = SyntheticEllipsoidsGenerator(
        num_samples=num_samples,
        image_size=tuple(image_size),
        noise_std=noise_std,
        smooth=smooth,
        seed=s,
    )

    return SyntheticEllipsoidsMonaiDataset(generator, transforms=transforms)
