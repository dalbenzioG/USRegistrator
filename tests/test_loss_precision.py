"""Keep LNCC reductions in float32 even when the network uses autocast."""
import pytest
import torch
from losses.lncc_dvf import LNCCWithDVFSupervision


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inputs(dtype=torch.float32):
    generator = torch.Generator().manual_seed(53)
    fixed = torch.rand(1, 1, 12, 12, 12, generator=generator).to(dtype)
    warped = (0.9 * fixed + 0.05).detach().requires_grad_(True)
    field = torch.full((1, 3, 12, 12, 12), 0.1, dtype=dtype, requires_grad=True)
    return warped, fixed, field, torch.zeros_like(field)


def test_outer_autocast_does_not_lower_loss_precision():
    criterion = LNCCWithDVFSupervision(kernel_size=9)
    args = inputs()
    expected = criterion(*args)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = criterion(*args)
        assert torch.is_autocast_enabled("cpu")
    assert actual.dtype == torch.float32 and torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.backward()
    assert torch.isfinite(args[0].grad).all()
    assert torch.isfinite(args[2].grad).all()


def test_half_precision_inputs_have_float32_loss_and_finite_gradients():
    criterion = LNCCWithDVFSupervision(kernel_size=9)
    args = inputs(torch.float16)
    expected = criterion(*[arg.float() for arg in args])
    actual = criterion(*args)
    assert actual.dtype == torch.float32 and torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.backward()
    assert torch.isfinite(args[0].grad).all()
    assert torch.isfinite(args[2].grad).all()
