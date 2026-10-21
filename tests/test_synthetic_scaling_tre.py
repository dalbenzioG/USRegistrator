"""Regression tests for normalized DDF scaling and fixed-grid TRE sampling."""

import pytest
import torch

from datasets.deepreg_synthetic import normalized_xyz_to_monai_ddf
from metrics.tre import mean_tre


def test_normalized_displacement_has_full_voxel_scale_and_zyx_order():
    image_size = (5, 7, 9)
    normalized_xyz = torch.tensor([0.5, -0.5, 0.5])
    field = normalized_xyz.view(1, 1, 1, 1, 3).expand(1, *image_size, 3)

    actual = normalized_xyz_to_monai_ddf(field, image_size)

    # (D-1)/2, (H-1)/2, (W-1)/2 convert normalized displacement to voxels.
    expected = torch.tensor([1.0, -1.5, 2.0])[:, None, None, None]
    torch.testing.assert_close(actual, expected.expand(3, *image_size))


@pytest.mark.parametrize(
    "landmark_offset,expected_tre",
    [((0.0, 0.0, 0.0), 0.0), ((0.3, 0.4, 0.0), 0.5)],
    ids=["exact-correspondence", "known-half-voxel-error"],
)
def test_tre_samples_spatially_varying_ddf_at_fixed_subvoxel_landmarks(
    landmark_offset, expected_tre
):
    z, y, x = torch.meshgrid(
        torch.arange(5, dtype=torch.float32),
        torch.arange(7, dtype=torch.float32),
        torch.arange(9, dtype=torch.float32),
        indexing="ij",
    )
    ddf = torch.stack([z / 8, y / 16, -x / 8])[None]
    fixed = torch.tensor([[[1.25, 2.5, 3.75], [2.5, 4.25, 5.5]]])
    # Analytic fixed + DDF(fixed), not computed through the sampler under test.
    moving = torch.tensor(
        [[[1.40625, 2.65625, 3.28125], [2.8125, 4.515625, 4.8125]]]
    )
    moving = moving + torch.tensor(landmark_offset)

    assert mean_tre(ddf, moving, fixed) == pytest.approx(expected_tre, abs=1e-6)
