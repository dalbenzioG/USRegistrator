import torch
from torch import Tensor
from monai.metrics import Metric


def _gradient_central_3d(ddf: Tensor):
    """
    Central differences for 3D displacement field.

    Args:
        ddf: (B, 3, D, H, W)

    Returns:
        du_dz, du_dy, du_dx
        each has shape (B, 3, D-2, H-2, W-2)
    """
    if ddf.ndim != 5 or ddf.shape[1] != 3:
        raise ValueError(f"Expected ddf shape (B, 3, D, H, W), got {ddf.shape}")

    du_dz = (ddf[:, :, 2:, 1:-1, 1:-1] - ddf[:, :, :-2, 1:-1, 1:-1]) * 0.5
    du_dy = (ddf[:, :, 1:-1, 2:, 1:-1] - ddf[:, :, 1:-1, :-2, 1:-1]) * 0.5
    du_dx = (ddf[:, :, 1:-1, 1:-1, 2:] - ddf[:, :, 1:-1, 1:-1, :-2]) * 0.5
    return du_dz, du_dy, du_dx


def jacobian_determinant(ddf: Tensor) -> Tensor:
    """
    Compute Jacobian determinant of deformation field phi(x) = x + u(x),
    where u is the displacement field ddf.

    Assumes channel order is (z, y, x) for a tensor shaped (B, 3, D, H, W).

    Args:
        ddf: (B, 3, D, H, W)

    Returns:
        jac_det: (B, D-2, H-2, W-2)
    """
    du_dz, du_dy, du_dx = _gradient_central_3d(ddf)

    # J = I + grad(u)
    J00 = 1.0 + du_dz[:, 0]
    J01 =       du_dy[:, 0]
    J02 =       du_dx[:, 0]

    J10 =       du_dz[:, 1]
    J11 = 1.0 + du_dy[:, 1]
    J12 =       du_dx[:, 1]

    J20 =       du_dz[:, 2]
    J21 =       du_dy[:, 2]
    J22 = 1.0 + du_dx[:, 2]

    jac_det = (
        J00 * (J11 * J22 - J12 * J21)
        - J01 * (J10 * J22 - J12 * J20)
        + J02 * (J10 * J21 - J11 * J20)
    )
    return jac_det


def neg_jac_ratio(ddf: Tensor) -> float:
    """
    Percentage of voxels with negative Jacobian determinant.
    """
    jac = jacobian_determinant(ddf)
    return (jac < 0).float().mean().item()


def jac_det_mean(ddf: Tensor) -> float:
    jac = jacobian_determinant(ddf)
    return jac.mean().item()


def jac_det_min(ddf: Tensor) -> float:
    jac = jacobian_determinant(ddf)
    return jac.min().item()


def log_jac_std(ddf: Tensor, eps: float = 1e-6) -> float:
    """
    Std of log-Jacobian determinant over non-folding voxels (Learn2Reg SDlogJ).

    Folding voxels (jac <= 0) have no real log-Jacobian and are *excluded*, which is the
    standard definition and the one documented for this repo. Clamping them to `eps`
    instead — as this did previously — gave each folded voxel log(1e-6) = -13.8, so the
    metric was dominated by the folding fraction that `neg_jac_ratio` already reports:
    a white-noise field with |u| = 0.36 vox and 1% folding scored 1.55 clamped vs 0.73
    excluded, while a *smooth* field of the same magnitude scores 0.15 either way.
    Report SDlogJ alongside `neg_jac_ratio`; neither substitutes for the other.
    """
    jac = jacobian_determinant(ddf)
    valid = jac > eps
    if not bool(valid.any()):
        return float("nan")
    return torch.log(jac[valid]).std().item()


class NegJacRatio(Metric):
    def __call__(self, ddf: Tensor) -> Tensor:
        return torch.tensor(neg_jac_ratio(ddf), device=ddf.device)


class JacDetMean(Metric):
    def __call__(self, ddf: Tensor) -> Tensor:
        return torch.tensor(jac_det_mean(ddf), device=ddf.device)


class JacDetMin(Metric):
    def __call__(self, ddf: Tensor) -> Tensor:
        return torch.tensor(jac_det_min(ddf), device=ddf.device)


class LogJacStd(Metric):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, ddf: Tensor) -> Tensor:
        return torch.tensor(log_jac_std(ddf, self.eps), device=ddf.device)