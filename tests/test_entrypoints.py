"""Both supported CLIs must import and dispatch to the same training API."""

import importlib
from pathlib import Path
import subprocess
import sys

import pytest
import train


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entrypoint", ["main.py", "train.py"])
def test_cli_help_imports_without_starting_training(entrypoint, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / entrypoint), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--config" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("module_name", ["main", "train"])
@pytest.mark.parametrize("config_path", [None, "path with spaces/experiment.yaml"])
def test_cli_passes_default_or_explicit_config(monkeypatch, module_name, config_path):
    module = importlib.import_module(module_name)
    calls = []
    monkeypatch.setattr(module, "run_training", calls.append)
    arguments = [f"{module_name}.py"]
    if config_path is not None:
        arguments += ["--config", config_path]
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    assert calls == [config_path or "configs/deepreg_synth.yaml"]


def test_public_api_does_not_parse_host_process_arguments(monkeypatch):
    seen = []

    def load_config(path):
        seen.append(path)
        raise RuntimeError("reached config loader")

    monkeypatch.setattr(sys, "argv", ["notebook-kernel", "--unrelated-kernel-argument"])
    monkeypatch.setattr(train, "load_config", load_config)
    with pytest.raises(RuntimeError, match="reached config loader"):
        train.run_training("experiment.yaml")
    assert seen == ["experiment.yaml"]
