"""Analytic geometry checks required by the supervised TransMorph example."""

import pytest
import torch
from monai.networks.blocks import Warp
from datasets import build_dataset
from datasets.deepreg_synthetic import normalized_xyz_to_monai_ddf, warp_fixed_points_to_moving
from metrics.tre import mean_tre


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_dataset(name="deepreg_synthetic"):
    config = {"name": name, "image_size": [16, 20, 24], "num_samples": 3,
              "seed": 456, "noise_std": 0.03, "smooth": True}
    if name == "deepreg_synthetic":
        config.update(max_disp=0.05, cp_spacing=4)
    return build_dataset(config, split="val")


@pytest.mark.parametrize("name", ["synthetic_ellipsoids", "deepreg_synthetic"])
def test_upstream_streaming_policy_is_preserved(name):
    dataset = make_dataset(name)
    first, second = dataset[0], dataset[0]
    assert not torch.equal(first["fixed"], second["fixed"])
    assert not hasattr(dataset, "cache_generated")


def test_normalized_displacements_have_correct_axis_order_and_scale():
    shape = (9, 11, 13)
    displacement_zyx = torch.tensor([1.5, -2.0, 0.5])
    grid = torch.empty(1, *shape, 3)
    grid[..., 0] = 2 * displacement_zyx[2] / (shape[2] - 1)
    grid[..., 1] = 2 * displacement_zyx[1] / (shape[1] - 1)
    grid[..., 2] = 2 * displacement_zyx[0] / (shape[0] - 1)
    actual = normalized_xyz_to_monai_ddf(grid, shape)
    expected = displacement_zyx[:, None, None, None].expand(3, *shape)
    torch.testing.assert_close(actual, expected)


def test_spatially_varying_field_is_sampled_at_fixed_subvoxel_points():
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.arange(9), torch.arange(11), torch.arange(13), indexing="ij"
    )
    field = torch.stack(
        [0.1 * grid_z, -0.05 * grid_y, 0.2 * grid_x], dim=0
    )
    fixed = torch.tensor([[1.25, 2.5, 3.75], [4.5, 5.25, 6.5]])
    moving = fixed * torch.tensor([1.1, 0.95, 1.2])
    torch.testing.assert_close(warp_fixed_points_to_moving(fixed, field), moving)
    assert mean_tre(field[None], moving[None], fixed[None]) < 1e-5


def test_identity_and_known_translation_agree_for_images_and_points():
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.arange(9), torch.arange(11), torch.arange(13), indexing="ij"
    )
    fixed_image = (grid_z + 10 * grid_y + 100 * grid_x).float()[None, None]
    translation = torch.tensor([1.0, -1.0, 2.0])
    forward = translation[None, :, None, None, None].expand(1, 3, 9, 11, 13)
    warp = Warp(mode="bilinear", padding_mode="border")
    moving_image = warp(fixed_image, forward)
    restored = warp(moving_image, -forward)
    torch.testing.assert_close(
        restored[..., 2:-2, 2:-2, 3:-3],
        fixed_image[..., 2:-2, 2:-2, 3:-3],
        atol=2e-4,
        rtol=1e-6,
    )
    fixed_points = torch.tensor([[[4.0, 5.0, 6.0]]])
    moving_points = fixed_points - translation
    assert mean_tre(-forward, moving_points, fixed_points) < 1e-6
    assert mean_tre(torch.zeros_like(forward), fixed_points, fixed_points) == 0
    assert mean_tre(torch.zeros_like(forward), moving_points, fixed_points) == pytest.approx(
        float(torch.linalg.vector_norm(translation))
    )


def test_zero_velocity_preserves_images_masks_and_landmarks():
    dataset = build_dataset(
        {
            "name": "deepreg_synthetic",
            "image_size": [16, 20, 24],
            "num_samples": 1,
            "max_disp": 0.0,
        },
        split="val",
    )
    sample = dataset[0]
    assert torch.count_nonzero(sample["dvf"]) == 0
    torch.testing.assert_close(sample["moving"], sample["fixed"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(sample["moving_mask"], sample["fixed_mask"], rtol=0, atol=0)
    torch.testing.assert_close(sample["moving_points"], sample["fixed_points"], rtol=0, atol=0)


def test_generated_landmarks_agree_with_ground_truth_field():
    sample = make_dataset()[0]
    assert mean_tre(
        sample["dvf"][None], sample["moving_points"][None], sample["fixed_points"][None]
    ) < 1e-6


def test_tre_accepts_half_precision_model_output_on_cpu():
    field = torch.ones(1, 3, 9, 11, 13, dtype=torch.float16)
    fixed = torch.tensor([[[3.5, 4.5, 5.5]]])
    assert mean_tre(field, fixed + 1, fixed) < 1e-6
