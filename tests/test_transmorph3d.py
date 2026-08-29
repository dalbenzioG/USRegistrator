"""Tests for the TransMorph3D registration model.

Covers registry integration, constructor validation, forward/backward
behaviour, the near-identity initialization, warp correctness, config-driven
construction (including the shipped YAML config), checkpoint round-trips,
and loss/metric compatibility with the training pipeline.

Small inputs (64^3, the minimum valid size) and feature_size=12 keep the CPU
tests reasonably fast; the GPU/AMP tests are skipped automatically when CUDA
is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from models import MODEL_REGISTRY, TransMorph3D, build_model, create_transmorph3d

# Smallest valid volume: every dim divisible by 32 and >= 64 (a 32-voxel dim
# collapses to 1 at the SwinUNETR bottleneck and breaks InstanceNorm).
IMAGE_SIZE = (64, 64, 64)
# feature_size must be divisible by 12 for SwinUNETR.
SMALL_KWARGS = dict(feature_size=12)


def make_model(**overrides) -> TransMorph3D:
    kwargs = {**SMALL_KWARGS, **overrides}
    return TransMorph3D(image_size=IMAGE_SIZE, **kwargs)


def make_inputs(batch=1, size=IMAGE_SIZE, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    moving = torch.rand(batch, 1, *size, generator=g).to(device)
    fixed = torch.rand(batch, 1, *size, generator=g).to(device)
    return moving, fixed


# -------------------------------------------------------------------------
# Registry / factory
# -------------------------------------------------------------------------

class TestRegistry:
    def test_registered(self):
        assert "transmorph3d" in MODEL_REGISTRY

    def test_factory_returns_module(self):
        model = create_transmorph3d(image_size=IMAGE_SIZE, **SMALL_KWARGS)
        assert isinstance(model, TransMorph3D)
        assert isinstance(model, nn.Module)

    def test_build_model_from_config(self):
        cfg = {"name": "transmorph3d", "feature_size": 12, "window_size": 7}
        model = build_model(cfg, image_size=IMAGE_SIZE)
        assert isinstance(model, TransMorph3D)

    def test_shipped_yaml_config_builds(self, repo_root: Path):
        """The checked-in config must construct without errors."""
        cfg_path = repo_root / "configs" / "deepreg_synth_transmorph3d.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        assert cfg["model"]["name"] == "transmorph3d"
        model = build_model(cfg["model"], image_size=cfg["image_size"])
        assert isinstance(model, TransMorph3D)
        assert model.image_size == list(cfg["image_size"])


# -------------------------------------------------------------------------
# Constructor validation
# -------------------------------------------------------------------------

class TestValidation:
    @pytest.mark.parametrize("bad_size", [(64, 64), (64,), (64, 64, 64, 64)])
    def test_rejects_wrong_rank(self, bad_size):
        with pytest.raises(ValueError, match="length 3"):
            TransMorph3D(image_size=bad_size, **SMALL_KWARGS)

    @pytest.mark.parametrize("bad_size", [(48, 64, 64), (64, 63, 64), (64, 64, 30)])
    def test_rejects_non_divisible_by_32(self, bad_size):
        with pytest.raises(ValueError, match="divisible"):
            TransMorph3D(image_size=bad_size, **SMALL_KWARGS)

    @pytest.mark.parametrize("bad_size", [(32, 32, 32), (32, 64, 96)])
    def test_rejects_too_small_dims(self, bad_size):
        """32 is divisible by 32 but collapses to 1 at the bottleneck and
        crashes InstanceNorm, so it must be rejected up front."""
        with pytest.raises(ValueError, match="at least 64"):
            TransMorph3D(image_size=bad_size, **SMALL_KWARGS)

    @pytest.mark.parametrize("good_size", [(64, 64, 64), (64, 96, 128)])
    def test_accepts_valid_sizes(self, good_size):
        model = TransMorph3D(image_size=good_size, **SMALL_KWARGS)
        assert model.image_size == list(good_size)


# -------------------------------------------------------------------------
# Forward pass
# -------------------------------------------------------------------------

class TestForward:
    @pytest.mark.parametrize("batch", [1, 2])
    def test_output_shapes(self, batch):
        model = make_model().eval()
        moving, fixed = make_inputs(batch=batch)
        with torch.no_grad():
            warped, ddf = model(moving, fixed)
        assert warped.shape == (batch, 1, *IMAGE_SIZE)
        assert ddf.shape == (batch, 3, *IMAGE_SIZE)

    def test_anisotropic_volume(self):
        size = (64, 96, 128)
        model = TransMorph3D(image_size=size, **SMALL_KWARGS).eval()
        moving, fixed = make_inputs(size=size)
        with torch.no_grad():
            warped, ddf = model(moving, fixed)
        assert warped.shape == (1, 1, *size)
        assert ddf.shape == (1, 3, *size)

    def test_outputs_finite(self):
        model = make_model().eval()
        moving, fixed = make_inputs()
        with torch.no_grad():
            warped, ddf = model(moving, fixed)
        assert torch.isfinite(warped).all()
        assert torch.isfinite(ddf).all()

    def test_near_identity_init(self):
        """With the near-zero ddf-head init, the initial transform should be
        close to identity: tiny ddf and warped ~= moving."""
        model = make_model().eval()
        moving, fixed = make_inputs()
        with torch.no_grad():
            warped, ddf = model(moving, fixed)
        assert ddf.abs().max().item() < 1e-2
        assert (warped - moving).abs().max().item() < 1e-2

    def test_zero_ddf_warp_is_identity(self):
        """Sanity-check the Warp component: zero displacement returns moving."""
        model = make_model().eval()
        moving, _ = make_inputs()
        zero_ddf = torch.zeros(1, 3, *IMAGE_SIZE)
        warped = model.warp(moving, zero_ddf)
        assert torch.allclose(warped, moving, atol=1e-5)

    def test_forward_depends_on_both_inputs(self):
        """The ddf must react to changes in either the moving or the fixed
        image (both are concatenated into the network input)."""
        model = make_model().eval()
        # Use non-degenerate weights so the head is not effectively zero.
        for m in model.net.out.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=1e-2)
        moving, fixed = make_inputs(seed=1)
        moving2, fixed2 = make_inputs(seed=2)
        with torch.no_grad():
            _, ddf_ref = model(moving, fixed)
            _, ddf_mov = model(moving2, fixed)
            _, ddf_fix = model(moving, fixed2)
        assert not torch.allclose(ddf_ref, ddf_mov)
        assert not torch.allclose(ddf_ref, ddf_fix)


# -------------------------------------------------------------------------
# Backward pass / training step
# -------------------------------------------------------------------------

class TestBackward:
    def test_gradients_flow(self):
        model = make_model().train()
        moving, fixed = make_inputs()
        warped, ddf = model(moving, fixed)
        loss = torch.nn.functional.mse_loss(warped, fixed) + ddf.pow(2).mean()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum().item() > 0 for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)

    def test_optimizer_step_changes_ddf_head(self):
        model = make_model().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        head_params_before = [
            p.detach().clone()
            for m in model.net.out.modules()
            if isinstance(m, nn.Conv3d)
            for p in m.parameters()
        ]

        moving, fixed = make_inputs()
        warped, _ = model(moving, fixed)
        loss = torch.nn.functional.mse_loss(warped, fixed)
        loss.backward()
        optimizer.step()

        head_params_after = [
            p.detach().clone()
            for m in model.net.out.modules()
            if isinstance(m, nn.Conv3d)
            for p in m.parameters()
        ]
        assert any(
            not torch.allclose(before, after)
            for before, after in zip(head_params_before, head_params_after)
        )

    def test_lncc_dvf_loss_compatibility(self):
        """The loss used by the shipped config must accept the model outputs."""
        from losses import build_loss

        loss_fn = build_loss(
            {
                "name": "lncc_dvf",
                "kernel_size": 9,
                "smooth_nr": 1e-4,
                "smooth_dr": 1e-2,
                "image_weight": 1.0,
                "dvf_weight": 0.5,
            }
        )
        model = make_model().train()
        moving, fixed = make_inputs()
        gt_dvf = torch.zeros(1, 3, *IMAGE_SIZE)
        warped, ddf = model(moving, fixed)
        loss = loss_fn(warped, fixed, ddf, gt_dvf)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()

    def test_metrics_compatibility(self):
        """All registered metrics must run on the model outputs (same call
        pattern train.py uses), except dice which needs labels."""
        from metrics import METRICS

        model = make_model().eval()
        moving, fixed = make_inputs()
        gt_dvf = torch.zeros(1, 3, *IMAGE_SIZE)
        with torch.no_grad():
            warped, ddf = model(moving, fixed)

        for name, fn in METRICS.items():
            if name in {"grad_l2", "neg_jac_ratio", "jac_det_mean", "jac_det_min", "log_jac_std"}:
                value = fn(ddf)
            elif name == "epe":
                value = fn(ddf, gt_dvf)
            elif name == "dice":
                continue  # needs labels; synthetic DVF dataset has none
            else:
                value = fn(warped, fixed)
            assert value == value, f"metric {name} returned NaN"  # NaN check


# -------------------------------------------------------------------------
# Checkpointing
# -------------------------------------------------------------------------

class TestCheckpoint:
    def test_state_dict_roundtrip(self, tmp_path: Path):
        model = make_model().eval()
        # Perturb the head so we are not just checking near-zero weights.
        for m in model.net.out.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=1e-2)

        ckpt_path = tmp_path / "transmorph3d.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        restored = make_model().eval()
        restored.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)["model_state_dict"]
        )

        moving, fixed = make_inputs()
        with torch.no_grad():
            warped_a, ddf_a = model(moving, fixed)
            warped_b, ddf_b = restored(moving, fixed)
        assert torch.allclose(warped_a, warped_b, atol=1e-6)
        assert torch.allclose(ddf_a, ddf_b, atol=1e-6)


# -------------------------------------------------------------------------
# GPU / AMP
# -------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCuda:
    def test_forward_backward_cuda(self):
        model = make_model().cuda().train()
        moving, fixed = make_inputs(device="cuda")
        warped, ddf = model(moving, fixed)
        assert warped.is_cuda and ddf.is_cuda
        loss = torch.nn.functional.mse_loss(warped, fixed)
        loss.backward()
        assert torch.isfinite(loss)

    def test_amp_autocast_forward(self):
        """train.py wraps the forward in autocast; make sure that path works."""
        model = make_model().cuda().train()
        moving, fixed = make_inputs(device="cuda")
        with torch.amp.autocast("cuda", enabled=True):
            warped, ddf = model(moving, fixed)
        assert torch.isfinite(warped).all()
        assert torch.isfinite(ddf).all()
