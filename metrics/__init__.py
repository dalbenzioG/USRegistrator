from .regression import MSEMetric, MAEMetric, mse, mae
from .ncc import NCC, global_ncc
from .smoothness import GradientL2, gradient_l2
from .epe import EPE, epe
from .jacobian import (
    jacobian_determinant,
    neg_jac_ratio,
    jac_det_mean,
    jac_det_min,
    log_jac_std,
    NegJacRatio,
    JacDetMean,
    JacDetMin,
    LogJacStd,
)

METRICS = {
    "mse": mse,
    "mae": mae,
    "ncc": global_ncc,
    "grad_l2": gradient_l2,
    "epe": epe,
    "neg_jac_ratio": neg_jac_ratio,
    "jac_det_mean": jac_det_mean,
    "jac_det_min": jac_det_min,
    "log_jac_std": log_jac_std,
}

# Also expose class-based mapping if needed, or let user access classes directly