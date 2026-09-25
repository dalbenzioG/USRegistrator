"""Target Registration Error (TRE) from a dense displacement field + landmarks.

Convention (Learn2Reg-style): the predicted DDF is the *backward* / pull map from
the fixed image (US) to the moving image (CT) — `warped_CT[x] = CT[x + ddf(x)]`.
So a fixed-space (US) landmark at index x corresponds to the moving (CT) location
`x + ddf(x)`. With CT landmarks already rigidly aligned into US space, TRE after
deformable registration is:

    TRE = mean_i || (p_US_i + ddf(p_US_i)) - p_CT_i ||   (in mm)

Coordinate handling: landmarks are LPS world mm (order x, y, z). The resampled cube
has identity direction, origin `bbox_lo` (x, y, z) and `spacing` (x, y, z), so
    index = (world - bbox_lo) / spacing.
The DDF tensor is (1, 3, D, H, W) with channels (dz, dy, dx) over array axes
(z=D, y=H, x=W), in voxel units.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _sample_ddf(ddf: torch.Tensor, idx_xyz: torch.Tensor, axis_order: str = "zyx") -> torch.Tensor:
    """Trilinearly sample the DDF at continuous indices, returning (dx, dy, dz) voxels.

    Args:
        ddf: (1, 3, A0, A1, A2) in voxel units. Channel i displaces along tensor spatial
            axis i (the MONAI `Warp` convention).
        idx_xyz: (N, 3) continuous indices in (ix, iy, iz) order.
        axis_order: which world axis each tensor spatial axis is.
            "zyx" — tensor is (z, y, x), so channels are (dz, dy, dx). SimpleITK-derived
                arrays, e.g. `datasets/kidney.py`.
            "xyz" — tensor is (x, y, z), so channels are (dx, dy, dz). MONAI arrays after
                `Orientationd(axcodes="RAS")`, e.g. `datasets/custom_dataset.py`.
    Returns:
        (N, 3) displacement in (dx, dy, dz) order (voxels).
    """
    if axis_order not in ("zyx", "xyz"):
        raise ValueError(f"axis_order must be 'zyx' or 'xyz', got {axis_order!r}")

    sizes = ddf.shape[2:]  # extents along tensor axes 0, 1, 2
    # grid_sample's last dim is (u, v, w) addressing tensor axes (2, 1, 0) — always the
    # reverse of the tensor's own spatial order.
    idx_per_axis = idx_xyz[:, [2, 1, 0]] if axis_order == "zyx" else idx_xyz  # -> per tensor axis
    normalized = [
        2.0 * idx_per_axis[:, axis] / max(sizes[axis] - 1, 1) - 1.0 for axis in range(3)
    ]
    grid = torch.stack(normalized[::-1], dim=-1).view(1, -1, 1, 1, 3).to(ddf.dtype)
    sampled = F.grid_sample(ddf, grid, mode="bilinear", padding_mode="border", align_corners=True)
    disp = sampled.view(3, -1).T  # (N, 3), channel i = displacement along tensor axis i
    return disp[:, [2, 1, 0]] if axis_order == "zyx" else disp  # -> (dx, dy, dz)


def tre_mm(
    ddf: torch.Tensor,
    ct_landmarks_world: np.ndarray,
    us_landmarks_world: np.ndarray,
    bbox_lo_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
    landmark_indices: list[int] | None = None,
    axis_order: str = "zyx",
) -> dict:
    """Mean TRE (mm) before (rigid only) and after applying the DDF.

    If landmark_indices is provided (e.g., [3, 4, 5]), only those rows
    (0-indexed) are used. Returns {"tre_after", "tre_before", "n": N,
    "per_point_after": [...]}. A zero DDF makes tre_after == tre_before
    == ||p_US - p_CT||.
    """
    ct = np.asarray(ct_landmarks_world, dtype=np.float64).reshape(-1, 3)
    us = np.asarray(us_landmarks_world, dtype=np.float64).reshape(-1, 3)
    if landmark_indices is not None:
        ct = ct[landmark_indices]
        us = us[landmark_indices]
    bbox_lo = np.asarray(bbox_lo_xyz, dtype=np.float64).reshape(3)
    spacing = np.asarray(spacing_xyz, dtype=np.float64).reshape(3)
    if ct.shape[0] == 0 or ct.shape != us.shape:
        return {"tre_after": float("nan"), "tre_before": float("nan"), "n": 0, "per_point_after": []}

    # Rigid-only (no warp) TRE.
    tre_before = np.linalg.norm(us - ct, axis=1)

    # US landmark -> continuous index (ix, iy, iz).
    idx = (us - bbox_lo) / spacing
    idx_t = torch.as_tensor(idx, dtype=ddf.dtype, device=ddf.device)
    disp = _sample_ddf(ddf, idx_t, axis_order=axis_order)
    disp = disp.detach().cpu().numpy().astype(np.float64)  # (N, 3) as (dx, dy, dz)

    idx_displaced = idx + disp
    world_displaced = bbox_lo + idx_displaced * spacing
    tre_after = np.linalg.norm(world_displaced - ct, axis=1)

    return {
        "tre_after": float(tre_after.mean()),
        "tre_before": float(tre_before.mean()),
        "n": int(ct.shape[0]),
        "per_point_after": tre_after.tolist(),
    }
