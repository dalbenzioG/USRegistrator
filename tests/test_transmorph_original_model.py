"""Tensor-only checks of the pinned original Tiny model and its own warper."""

from importlib.util import find_spec
import io

import pytest
import torch

from models import build_model
from models.transmorph_original3d import resolve_transmorph_root


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def original_tiny_model():
    source_root = resolve_transmorph_root()
    if not (source_root / "models" / "TransMorph.py").is_file():
        pytest.skip("Optional original TransMorph checkout is not installed")
    missing = [name for name in ("timm", "ml_collections") if find_spec(name) is None]
    if missing:
        pytest.skip(f"Missing optional TransMorph dependencies: {', '.join(missing)}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = build_model(
            {"name": "transmorph_original3d", "variant": "tiny",
             "transmorph_root": str(source_root)},
            [64, 64, 64],
        )
    return model.eval()


@pytest.fixture(scope="module")
def registration_inputs():
    generator = torch.Generator(device="cpu").manual_seed(456)
    moving = torch.rand(1, 1, 64, 64, 64, generator=generator)
    fixed = torch.rand(1, 1, 64, 64, 64, generator=generator)
    return moving, fixed


def test_original_tiny_forward_contract(original_tiny_model, registration_inputs):
    assert sum(parameter.numel() for parameter in original_tiny_model.parameters()) == 244527
    with torch.no_grad():
        result = original_tiny_model(*registration_inputs)
    assert isinstance(result, tuple) and len(result) == 2
    warped, ddf = result
    assert warped.shape == (1, 1, 64, 64, 64)
    assert ddf.shape == (1, 3, 64, 64, 64)
    assert torch.isfinite(warped).all()
    assert torch.isfinite(ddf).all()


def test_original_tiny_state_dict_roundtrip(original_tiny_model, registration_inputs):
    checkpoint = io.BytesIO()
    torch.save(original_tiny_model.state_dict(), checkpoint)
    checkpoint.seek(0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(789)
        restored = build_model(
            {"name": "transmorph_original3d", "variant": "tiny",
             "transmorph_root": str(resolve_transmorph_root())},
            [64, 64, 64],
        ).eval()
    loaded = restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True,
    )
    assert loaded.missing_keys == []
    assert loaded.unexpected_keys == []
    with torch.no_grad():
        expected = original_tiny_model(*registration_inputs)
        actual = restored(*registration_inputs)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-6, rtol=1e-5)


@pytest.fixture(scope="module")
def original_transmorph_warper(original_tiny_model):
    return original_tiny_model.net.spatial_trans


@pytest.mark.parametrize("channel", [0, 1, 2], ids=["z", "y", "x"])
@pytest.mark.parametrize("offset", [-1.0, 1.0], ids=["minus-one", "plus-one"])
def test_original_warper_signed_axis_convention(original_transmorph_warper, channel, offset):
    shape = (64, 64, 64)
    grid_z, grid_y, grid_x = torch.meshgrid(
        *(torch.arange(size, dtype=torch.float32) for size in shape), indexing="ij"
    )
    source = (0.01 * grid_z + 0.001 * grid_y + 0.0001 * grid_x)[None, None]
    field = torch.zeros(1, 3, *shape)
    field[:, channel] = offset
    actual = original_transmorph_warper(source, field)
    expected = torch.roll(source, shifts=-int(offset), dims=channel + 2)
    interior = (..., slice(1, -1), slice(1, -1), slice(1, -1))
    torch.testing.assert_close(actual[interior], expected[interior], atol=2e-6, rtol=1e-5)
