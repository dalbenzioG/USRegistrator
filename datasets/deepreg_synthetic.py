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


def normalized_xyz_to_monai_ddf(
    dvf_grid: torch.Tensor,
    image_size: tuple[int, int, int],
) -> torch.Tensor:
    """
    Convert normalized grid_sample displacement field
    from (..., 3) with channel order (x, y, z)
    into channel-first DDF tensor (3, D, H, W)
    in MONAI-side channel order (z, y, x).

    Input:
        dvf_grid: (1, D, H, W, 3), normalized displacement, order = (x, y, z)

    Output:
        ddf_monai: (3, D, H, W), channel order = (z, y, x)
    """
    D, H, W = image_size

    dx = dvf_grid[0, ..., 0] * ((W - 1) / 2.0)
    dy = dvf_grid[0, ..., 1] * ((H - 1) / 2.0)
    dz = dvf_grid[0, ..., 2] * ((D - 1) / 2.0)

    # reorder to (z, y, x) and apply extra scale correction
    ddf_monai = 0.5 * torch.stack([dz, dy, dx], dim=0)
    return ddf_monai



class DeepRegLikeDVFSyntheticGenerator:
    """
    DeepReg-style synthetic DVF generator (PyTorch implementation).

    Produces:
      - fixed: base ellipsoid anatomy
      - moving: fixed warped by a random smooth forward DVF
      - dvf: ground-truth DVF for warping moving -> fixed

    Conventions:
      - internal generation uses grid_sample normalized coordinates
      - returned dvf is converted to voxel-like displacement magnitude
      - channel order returned = (x, y, z), shape = (3, D, H, W)
      - moving -> fixed GT field is approximated as negative of the forward field
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

        self.base_generator = SyntheticEllipsoidsGenerator(
            num_samples=num_samples,
            image_size=self.image_size,
            noise_std=noise_std,
            smooth=smooth,
            seed=seed,
        )

        self.rng = np.random.RandomState(seed)

        # grid_sample 3D expects last dim order = (x, y, z)
        D, H, W = self.image_size
        zz = torch.linspace(-1.0, 1.0, steps=D, dtype=torch.float32)
        yy = torch.linspace(-1.0, 1.0, steps=H, dtype=torch.float32)
        xx = torch.linspace(-1.0, 1.0, steps=W, dtype=torch.float32)

        z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
        self.identity_grid = torch.stack([x, y, z], dim=-1)[None]  # (1, D, H, W, 3)

    def _random_dvf(self) -> torch.Tensor:
        """
        Generate a random smooth DVF in normalized grid coordinates.

        Returns:
            dvf_grid: (1, D, H, W, 3), last-dim order = (x, y, z)
        """
        D, H, W = self.image_size

        Dc = max(1, D // self.cp_spacing)
        Hc = max(1, H // self.cp_spacing)
        Wc = max(1, W // self.cp_spacing)

        # random amplitude per channel in normalized coordinates
        low_res_strength = self.rng.uniform(
            low=0.0,
            high=self.max_disp,
            size=(1, 1, 1, 1, 3),
        ).astype(np.float32)

        # low-resolution Gaussian random field
        low_res_field = self.rng.randn(1, Dc, Hc, Wc, 3).astype(np.float32)
        low_res_field = low_res_field * low_res_strength

        # to channel-first: (1, 3, Dc, Hc, Wc)
        low_res_field = torch.from_numpy(low_res_field).permute(0, 4, 1, 2, 3)

        # upsample to full resolution: (1, 3, D, H, W)
        dvf_full = F.interpolate(
            low_res_field,
            size=(D, H, W),
            mode="trilinear",
            align_corners=False,
        )

        # back to grid format: (1, D, H, W, 3), order = (x, y, z)
        dvf_grid = dvf_full.permute(0, 2, 3, 4, 1)
        return dvf_grid

    def __len__(self) -> int:
        return self.num_samples

    def get_sample(self) -> dict:
        # fixed image
        base = self.base_generator.get_sample()
        fixed = base["fixed"].unsqueeze(0)  # (1, 1, D, H, W)

        # forward field used to synthesize moving from fixed
        forward_dvf_grid = self._random_dvf()  # (1, D, H, W, 3), (x,y,z)

        # moving = fixed warped by forward field
        forward_grid = self.identity_grid + forward_dvf_grid
        moving = F.grid_sample(
            fixed,
            forward_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # (1, 1, D, H, W)

        # approximate inverse field for moving -> fixed
        gt_dvf_grid = -forward_dvf_grid

        # convert normalized grid displacement to MONAI-side supervised DDF
        gt_dvf = normalized_xyz_to_monai_ddf(gt_dvf_grid, self.image_size)

        return {
            "moving": moving[0],   # (1, D, H, W)
            "fixed": fixed[0],     # (1, D, H, W)
            "dvf": gt_dvf,         # (3, D, H, W), channel order = (x, y, z)
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
