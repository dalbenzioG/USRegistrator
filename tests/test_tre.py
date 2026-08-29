"""Tests for metrics.tre.mean_tre.

mean_tre is not in the METRICS registry, so it is not covered by
test_transmorph3d.py's test_metrics_compatibility. train.py calls it
directly in the validation loop.
"""

from __future__ import annotations

import pytest
import torch

from metrics.tre import mean_tre


D = H = W = 16
POINTS = torch.tensor(
    [[[4.0, 5.0, 6.0], [8.0, 8.0, 8.0], [11.0, 3.0, 9.0]]]
)  # (1, 3, 3), voxel coords in (z, y, x)


def test_zero_ddf_gives_distance_between_point_sets():
    """With no displacement, TRE is just the mean moving-to-fixed distance."""
    ddf = torch.zeros(1, 3, D, H, W)
    fixed = POINTS + torch.tensor([0.0, 0.0, 2.0])  # shift 2 voxels in x
    assert mean_tre(ddf, POINTS, fixed) == pytest.approx(2.0, abs=1e-4)


def test_constant_ddf_is_applied_in_zyx_order():
    """A constant field must move the points by exactly that vector, with
    ddf channel c matching point coordinate c (both (z, y, x))."""
    disp = [1.0, -2.0, 3.0]
    ddf = torch.zeros(1, 3, D, H, W)
    for c, v in enumerate(disp):
        ddf[:, c] = v
    fixed = POINTS + torch.tensor(disp)
    assert mean_tre(ddf, POINTS, fixed) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("ddf_dtype", [torch.float16, torch.bfloat16])
def test_accepts_low_precision_ddf(ddf_dtype):
    """Regression: under AMP train.py hands mean_tre a low-precision ddf
    while the landmarks stay float32. grid_sample requires both operands
    to share a dtype, so mean_tre must normalise them itself."""
    ddf = torch.zeros(1, 3, D, H, W, dtype=ddf_dtype)
    fixed = POINTS + torch.tensor([0.0, 0.0, 2.0])
    value = mean_tre(ddf, POINTS, fixed)
    assert isinstance(value, float)
    assert value == pytest.approx(2.0, abs=1e-2)


def test_returns_plain_float():
    """train.py accumulates the result as a python float."""
    ddf = torch.zeros(1, 3, D, H, W)
    value = mean_tre(ddf, POINTS, POINTS)
    assert isinstance(value, float)
    assert value == pytest.approx(0.0, abs=1e-6)
