"""DeepReg-style synthetic DVF dataset for 3D registration."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.blocks import DVF2DDF, Warp

from .registry import register_dataset
from .synthetic_ellipsoids import (
    SyntheticEllipsoidsGenerator,
    SyntheticEllipsoidsMonaiDataset,
)

def sample_foreground_points(mask: torch.Tensor, num_points: int = 8) -> torch.Tensor:
    """
    mask: (1, D, H, W) or (D, H, W)
    return: (N, 3) voxel coords in (z, y, x)
    """
    if mask.dim() == 4:
        mask = mask[0]

    coords = torch.nonzero(mask > 0.5, as_tuple=False)

    if coords.shape[0] == 0:
        raise ValueError("No foreground voxels found in mask when sampling points.")

    if coords.shape[0] >= num_points:
        idx = torch.randperm(coords.shape[0])[:num_points]
        pts = coords[idx]
    else:
        idx = torch.randint(0, coords.shape[0], (num_points,))
        pts = coords[idx]

    return pts.float()


def warp_fixed_points_to_moving(
    points_zyx: torch.Tensor,
    ddf_fixed_to_moving: torch.Tensor,
) -> torch.Tensor:
    """
    Map fixed-grid points to moving space using the registration pullback DDF.

    points_zyx: (N, 3) voxel coordinates in (z, y, x).
    ddf_fixed_to_moving: (3, D, H, W), voxel displacement in (z, y, x).
    Returns corresponding moving-space coordinates in (z, y, x).
    """
    _, depth, height, width = ddf_fixed_to_moving.shape
    points_z, points_y, points_x = points_zyx.unbind(dim=-1)
    grid = torch.stack(
        [
            2.0 * points_x / max(width - 1, 1) - 1.0,
            2.0 * points_y / max(height - 1, 1) - 1.0,
            2.0 * points_z / max(depth - 1, 1) - 1.0,
        ],
        dim=-1,
    ).view(1, 1, 1, -1, 3)
    sampled = F.grid_sample(
        ddf_fixed_to_moving.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled.squeeze(2).squeeze(2).permute(0, 2, 1)[0]
    return (points_zyx + sampled).float()

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

    ddf_monai = torch.stack([dz, dy, dx], dim=0)
    return ddf_monai


def warp_points_with_forward_grid(points_zyx, dvf_grid, image_size):
    """Compatibility helper: sample normalized xyz displacements at voxel points."""
    return warp_fixed_points_to_moving(
        points_zyx, normalized_xyz_to_monai_ddf(dvf_grid, image_size)
    )


class DeepRegLikeDVFSyntheticGenerator:
    """
    DeepReg-style synthetic DVF generator (PyTorch implementation).

    Produces:
      - fixed: base ellipsoid anatomy
      - moving: fixed sampled through exp(+velocity)
      - dvf: exp(-velocity), a fixed-grid pullback DDF for sampling moving

    Conventions:
      - internal generation uses grid_sample normalized coordinates
      - returned dvf has voxel displacement units
      - returned dvf is in MONAI-side channel order (z, y, x), shape = (3, D, H, W)
      - both directions integrate opposite stationary velocity fields
      - finite resolution/interpolation means inverse consistency is approximate
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
        self.warp = Warp(mode="bilinear", padding_mode="border")
        self.warp_nearest = Warp(mode="nearest", padding_mode="border")
        self.integrate = DVF2DDF(num_steps=7, mode="bilinear", padding_mode="border")

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
        # Preserve upstream streaming samples; this PR changes geometry, not RNG policy.
        base = self.base_generator.get_sample()
        fixed = base["fixed"].unsqueeze(0)
        fixed_mask = base.get("fixed_mask")
        if fixed_mask is not None:
            fixed_mask = fixed_mask.unsqueeze(0)

        velocity_grid = self._random_dvf()
        velocity = normalized_xyz_to_monai_ddf(
            velocity_grid, self.image_size
        ).unsqueeze(0)
        moving_to_fixed = self.integrate(velocity)
        moving = self.warp(fixed, moving_to_fixed)
        gt_dvf = self.integrate(-velocity)

        moving_mask = None
        if fixed_mask is not None:
            moving_mask = self.warp_nearest(fixed_mask.float(), moving_to_fixed)

        fixed_points = None
        moving_points = None
        if fixed_mask is not None:
            fixed_points = sample_foreground_points(
                base["fixed_mask"], num_points=8
            )
            moving_points = warp_fixed_points_to_moving(fixed_points, gt_dvf[0])

        sample = {"moving": moving[0], "fixed": fixed[0], "dvf": gt_dvf[0]}
        if fixed_mask is not None and moving_mask is not None:
            sample["moving_mask"] = moving_mask[0]
            sample["fixed_mask"] = fixed_mask[0]
        if fixed_points is not None and moving_points is not None:
            sample["fixed_points"] = fixed_points
            sample["moving_points"] = moving_points
        return sample


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
