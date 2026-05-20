# tre.py
# NOTE:
# ddf channel order is assumed to be (z, y, x)
# point coordinates are also assumed to be (z, y, x) in voxel space

import torch
import torch.nn.functional as F


def mean_tre(
    ddf: torch.Tensor,
    moving_points: torch.Tensor,
    fixed_points: torch.Tensor,
) -> float:
    """
    ddf: (B, 3, D, H, W)
    moving_points: (B, N, 3), voxel coordinates, assumed order (z, y, x)
    fixed_points: (B, N, 3), voxel coordinates, assumed order (z, y, x)
    """
    B, _, D, H, W = ddf.shape

    z = moving_points[..., 0]
    y = moving_points[..., 1]
    x = moving_points[..., 2]

    x_norm = 2.0 * x / max(W - 1, 1) - 1.0
    y_norm = 2.0 * y / max(H - 1, 1) - 1.0
    z_norm = 2.0 * z / max(D - 1, 1) - 1.0

    grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)   # (B, N, 3)
    grid = grid.view(B, 1, 1, -1, 3)

    sampled_disp = F.grid_sample(
        ddf,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )  # (B, 3, 1, 1, N)

    sampled_disp = sampled_disp.squeeze(2).squeeze(2).permute(0, 2, 1)  # (B, N, 3)

    warped_points = moving_points + sampled_disp
    tre = torch.norm(warped_points - fixed_points, dim=-1)  # (B, N)

    return float(tre.mean().item())