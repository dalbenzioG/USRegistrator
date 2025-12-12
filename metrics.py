import torch
import torch.nn.functional as F

def mse(warped, fixed):
    return F.mse_loss(warped, fixed).item()

def mae(warped, fixed):
    return F.l1_loss(warped, fixed).item()

def global_ncc(warped, fixed, eps=1e-5):
    w = warped - warped.mean()
    f = fixed - fixed.mean()
    num = (w * f).sum()
    den = torch.sqrt((w*w).sum() * (f*f).sum() + eps)
    return (num / den).item()

def gradient_l2(ddf):
    # ddf: (B,3,D,H,W)
    dz = torch.diff(ddf, dim=2).pow(2)
    dy = torch.diff(ddf, dim=3).pow(2)
    dx = torch.diff(ddf, dim=4).pow(2)
    return (dz.mean() + dy.mean() + dx.mean()).item()

def epe(ddf, gt_dvf, eps: float = 1e-9):
    """
    Endpoint Error (EPE) between predicted ddf and ground-truth dvf.

    ddf, gt_dvf: (B, 3, D, H, W)
    Returns: Euclidean distance averaged over all batch and spatial dimensions.
    """
    if ddf.shape != gt_dvf.shape:
        raise ValueError(f"EPE: shape mismatch {ddf.shape} vs {gt_dvf.shape}")
    diff = ddf - gt_dvf              # (B,3,D,H,W)
    sq = diff.pow(2).sum(dim=1)      # (B,D,H,W)
    dist = torch.sqrt(sq + eps)      # (B,D,H,W)
    return dist.mean().item()


METRICS = {
    "mse": mse,
    "mae": mae,
    "ncc": global_ncc,
    "grad_l2": gradient_l2,
    "epe": epe,
}
