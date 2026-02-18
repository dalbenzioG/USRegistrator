"""DeepReg-style synthetic DVF dataset for 3D registration."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .registry import register_dataset
from .synthetic_ellipsoids import (
    SyntheticEllipsoidsGenerator,
    SyntheticEllipsoidsMonaiDataset,
)


# -------------------------------------------------------------------------
# DeepReg-style synthetic DVF dataset
# -------------------------------------------------------------------------

class DeepRegLikeDVFSyntheticGenerator:
    """
       DeepReg-style synthetic DVF generator (PyTorch implementation).

       - Reuses SyntheticEllipsoidsGenerator to provide the fixed anatomy
       - Generates a Gaussian random field on a coarse grid, scaled by U(0, max_disp)
       - Upsamples to full resolution via trilinear interpolation to obtain a smooth DVF
       - Uses identity_grid + DVF as sampling grid to generate the moving image via grid_sample
       - Returns a dict: {"moving", "fixed", "dvf"}
    """

    def __init__(
        self,
        num_samples: int,
        image_size: Tuple[int, int, int],
        max_disp: float = 0.2,
        cp_spacing: int = 8,
        noise_std: float = 0.03,
        smooth: bool = True,
        seed: int = 123,
    ):
        self.num_samples = int(num_samples)
        self.image_size = tuple(int(s) for s in image_size)
        self.max_disp = float(max_disp)
        self.cp_spacing = int(cp_spacing)

        # Ellipsoid generator: provides base anatomy (fixed image)
        self.base_generator = SyntheticEllipsoidsGenerator(
            num_samples=num_samples,
            image_size=self.image_size,
            noise_std=noise_std,
            smooth=smooth,
            seed=seed,
        )

        self.rng = np.random.RandomState(seed)

        # Precompute normalized identity grid (1, D, H, W, 3)
        D, H, W = self.image_size
        zz = torch.linspace(-1.0, 1.0, steps=D, dtype=torch.float32)
        yy = torch.linspace(-1.0, 1.0, steps=H, dtype=torch.float32)
        xx = torch.linspace(-1.0, 1.0, steps=W, dtype=torch.float32)
        z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
        self.identity_grid = torch.stack([z, y, x], dim=-1)[None]

    def _random_dvf(self) -> torch.Tensor:
        """
                Random DVF generation inspired by DeepReg's gen_rand_ddf:

                - Generate a Gaussian random field on a coarse grid
                - Scale by a random strength sampled from U(0, max_disp)
                - Upsample to full resolution using trilinear interpolation
                - Return a DVF in normalized grid_sample coordinates: (1, D, H, W, 3)
        """
        D, H, W = self.image_size

        # Coarse grid size
        Dc = max(1, D // self.cp_spacing)
        Hc = max(1, H // self.cp_spacing)
        Wc = max(1, W // self.cp_spacing)

        # Random strength U(0, max_disp), one per channel
        low_res_strength = self.rng.uniform(
            low=0.0,
            high=self.max_disp,
            size=(1, 1, 1, 1, 3),
        ).astype(np.float32)

        # Low-resolution Gaussian field (1, Dc, Hc, Wc, 3)
        low_res_field = self.rng.randn(1, Dc, Hc, Wc, 3).astype(np.float32)
        low_res_field = low_res_field * low_res_strength

        # → tensor: (1, 3, Dc, Hc, Wc)
        low_res_field = torch.from_numpy(low_res_field)
        low_res_field = low_res_field.permute(0, 4, 1, 2, 3)

        # Upsample to full resolution: (1, 3, D, H, W)
        dvf_full = F.interpolate(
            low_res_field,
            size=(D, H, W),
            mode="trilinear",
            align_corners=True,
        )

        # grid_sample expects (1, D, H, W, 3)
        dvf_grid = dvf_full.permute(0, 2, 3, 4, 1)
        return dvf_grid

    def __len__(self) -> int:
        return self.num_samples

    def get_sample(self) -> dict:
        # Fixed image: ellipsoid anatomy
        base = self.base_generator.get_sample()
        fixed = base["fixed"].unsqueeze(0)   # (1,1,D,H,W)

        dvf_grid = self._random_dvf()        # (1,D,H,W,3)
        grid = self.identity_grid + dvf_grid

        # Warp fixed to obtain moving
        moving = F.grid_sample(
            fixed,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (1,1,D,H,W)

        # DVF in channel-first format (3,D,H,W)
        dvf_ch_first = dvf_grid[0].permute(3, 0, 1, 2)

        return {
            "moving": moving[0],   # (1,D,H,W)
            "fixed": fixed[0],     # (1,D,H,W)
            "dvf": dvf_ch_first,   # (3,D,H,W)
        }


@register_dataset("deepreg_synthetic")
def create_deepreg_synthetic(
    split: str,
    num_samples: int,
    image_size=(64, 64, 64),
    max_disp: float = 0.2,
    cp_spacing: int = 8,
    noise_std: float = 0.03,
    smooth: bool = True,
    seed: int = 123,
    transforms=None,
):
    """
    Factory for DeepReg-style synthetic DVF dataset.

    Config example:
        train_dataset:
          name: deepreg_synthetic
          image_size: [64, 64, 64]
          num_samples: 4000
          max_disp: 0.2
          cp_spacing: 8
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

    generator = DeepRegLikeDVFSyntheticGenerator(
        num_samples=num_samples,
        image_size=tuple(image_size),
        max_disp=max_disp,
        cp_spacing=cp_spacing,
        noise_std=noise_std,
        smooth=smooth,
        seed=s,
    )

    return SyntheticEllipsoidsMonaiDataset(generator, transforms=transforms)
