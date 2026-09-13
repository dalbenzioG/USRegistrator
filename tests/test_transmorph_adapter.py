from pathlib import Path
import hashlib
import sys
import subprocess
from types import SimpleNamespace

import pytest
import torch

import models
from models import build_model
from models import transmorph_original3d as adapter
from models.transmorph_original3d import _load_transmorph


def test_optional_model_is_registered_without_loading_its_dependencies():
    assert models.MODEL_REGISTRY["transmorph_original3d"] is adapter.create_transmorph_original3d
    assert models.MODEL_REGISTRY.get("transmorph3d") is not adapter.create_transmorph_original3d
    model = build_model({"name": "localnet3d", "num_channel_initial": 4}, [32, 32, 32])
    assert isinstance(model, models.LocalNet3D)


def test_unknown_variant_and_missing_source_have_actionable_errors(tmp_path):
    with pytest.raises(ValueError, match="Unknown TransMorph variant"):
        build_model({"name": "transmorph_original3d", "variant": "invalid"}, [64, 64, 64])
    with pytest.raises(FileNotFoundError, match="Missing TransMorph source"):
        build_model({"name": "transmorph_original3d", "transmorph_root": str(tmp_path)}, [64, 64, 64])


def test_invalid_image_size_fails_before_optional_imports():
    with pytest.raises(ValueError, match="multiples of 32"):
        build_model({"name": "transmorph_original3d"}, [63, 64, 64])


def test_loader_restores_package_and_import_path_on_failure(tmp_path, monkeypatch):
    source = tmp_path / "models"
    source.mkdir()
    (source / "configs_TransMorph.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "TransMorph.py").write_text(
        "import models.configs_TransMorph as configs\nraise RuntimeError('deliberate failure')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "verify_transmorph_source", lambda root: {})
    previous_package = sys.modules["models"]
    previous_path = list(sys.path)
    previous_modules = {name for name in sys.modules if name.startswith("models.")}
    with pytest.raises(RuntimeError, match="deliberate failure"):
        _load_transmorph(str(tmp_path))
    assert sys.modules["models"] is previous_package
    assert sys.path == previous_path
    assert {name for name in sys.modules if name.startswith("models.")} == previous_modules
    assert not hasattr(models, "configs_TransMorph")


def test_source_pin_accepts_newline_conversion_but_rejects_changed_code(tmp_path, monkeypatch):
    relative = "models/TransMorph.py"
    path = tmp_path / relative
    path.parent.mkdir()
    expected = hashlib.sha256(b"VALUE = 1\n").hexdigest()
    monkeypatch.setattr(adapter, "SOURCE_SHA256", {relative: expected})
    path.write_bytes(b"VALUE = 1\r\n")
    assert adapter.verify_transmorph_source(tmp_path) == {relative: expected}
    path.write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError, match="source mismatch"):
        adapter.verify_transmorph_source(tmp_path)


def test_source_checked_on_every_load_before_cached_import(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(adapter, "verify_transmorph_source", lambda root: calls.append(root))
    monkeypatch.setattr(adapter, "_load_verified_transmorph", lambda root: ("model", "config"))
    assert adapter._load_transmorph(str(tmp_path)) == ("model", "config")
    assert adapter._load_transmorph(str(tmp_path)) == ("model", "config")
    assert calls == [str(tmp_path), str(tmp_path)]


def test_optional_import_failure_has_setup_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "verify_transmorph_source", lambda root: {})

    def unavailable(root):
        raise ModuleNotFoundError("ml_collections")

    monkeypatch.setattr(adapter, "_load_verified_transmorph", unavailable)
    with pytest.raises(ImportError, match="requirements-transmorph.txt"):
        adapter._load_transmorph(str(tmp_path))


def test_adapter_preserves_moving_fixed_order_and_upstream_outputs(monkeypatch):
    seen = []

    class StubNetwork(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.img_size == (64, 64, 64)

        def forward(self, inputs):
            seen.append(inputs)
            return inputs[:, :1], torch.ones(inputs.shape[0], 3, *inputs.shape[2:])

    factories = SimpleNamespace(get_3DTransMorphTiny_config=lambda: SimpleNamespace())
    monkeypatch.setattr(adapter, "_load_transmorph", lambda root: (StubNetwork, factories))
    model = adapter.TransMorphOriginal3D([64, 64, 64])
    moving, fixed = torch.zeros(1, 1, 4, 5, 6), torch.ones(1, 1, 4, 5, 6)
    warped, ddf = model(moving, fixed)
    torch.testing.assert_close(seen[0][:, :1], moving)
    torch.testing.assert_close(seen[0][:, 1:], fixed)
    torch.testing.assert_close(warped, moving)
    assert ddf.shape == (1, 3, 4, 5, 6)


def test_core_import_works_when_optional_dependencies_are_unavailable():
    program = """
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'timm', 'ml_collections'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockOptional())
import models
assert 'transmorph_original3d' in models.MODEL_REGISTRY
assert not any(name.startswith('_usregistrator_tm_') for name in sys.modules)
model = models.build_model({'name': 'localnet3d', 'num_channel_initial': 4}, [32, 32, 32])
assert isinstance(model, models.LocalNet3D)
"""
    result = subprocess.run([sys.executable, "-c", program],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
