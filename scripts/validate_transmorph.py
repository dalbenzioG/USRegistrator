"""Focused train/save/reload smoke; not a convergence or accuracy benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from monai.networks.blocks import Warp
from torch.utils.data import DataLoader
from datasets import build_dataset
from losses import build_loss
from models import build_model
from train import evaluate, set_seed


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--models", nargs="+", default=["transmorph_original_tiny", "localnet3d"],
                        choices=["transmorph_original_tiny", "localnet3d"])
    parser.add_argument("--entrypoint", choices=["main.py", "train.py"], default="main.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; preserve existing runs")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    report = {"status": "RUNNING", "purpose": "focused engineering smoke, not model ranking",
              "device": str(device), "amp": device.type == "cuda", "entrypoint": args.entrypoint,
              "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "python": sys.version, "packages": {name: importlib.metadata.version(name)
              for name in ("torch", "monai", "numpy", "PyYAML", "wandb")}, "models": {}}
    summary_path = output / "summary.json"
    template = yaml.safe_load((ROOT / "configs/deepreg_synth_transmorph_original_tiny.yaml").read_text())
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", WANDB_MODE="disabled")
    try:
        for name in args.models:
            started = time.perf_counter()
            folder = output / name
            folder.mkdir()
            cfg = json.loads(json.dumps(template))
            cfg["device"] = str(device)
            cfg["training"].update(batch_size=2, num_workers=0, seed=42,
                                   amp=device.type == "cuda", save_dir=str(folder / "checkpoints"))
            if name == "localnet3d":
                cfg["model"] = {"name": "localnet3d", "num_channel_initial": 4}
            config_path = folder / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            set_seed(42)
            # Capture fresh cases only for this smoke's post-training roundtrip.
            # Production train/validation datasets retain upstream streaming access.
            stream = build_dataset(cfg["val_dataset"], split="val")
            cases = [stream[i] for i in range(len(stream))]
            batch = next(iter(DataLoader(cases, batch_size=1)))
            model = build_model(cfg["model"], cfg["image_size"]).to(device)
            criterion = build_loss(cfg["loss"]).to(device)
            moving, fixed = batch["moving"].to(device), batch["fixed"].to(device)
            warped, field = model(moving, fixed)
            assert field.shape == (1, 3, *cfg["image_size"])
            assert warped.shape == fixed.shape
            if name == "transmorph_original_tiny":
                # Original TransMorph uses zero padding, whereas some core models
                # use border padding. Compare the same sampler settings explicitly.
                expected = Warp(mode="bilinear", padding_mode="zeros")(moving, field)
                torch.testing.assert_close(warped, expected, atol=2e-4, rtol=1e-4)
            loss = criterion(warped, fixed, field, batch["dvf"].to(device))
            assert torch.isfinite(loss)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert any(torch.count_nonzero(g).item() for g in grads)
            parameters = sum(p.numel() for p in model.parameters())
            del model, grads, warped, field, loss
            print(f"Training smoke: {name}", flush=True)
            with (folder / "train.log").open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-u", args.entrypoint, "--config", str(config_path)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            hashes = {filename: sha(folder / "checkpoints" / filename)
                      for filename in ("best_epe.pt", "best_val_loss.pt", "last.pt")}
            # Only deserialize trusted checkpoints produced by the subprocess above.
            checkpoint = torch.load(folder / "checkpoints/best_epe.pt", map_location="cpu", weights_only=False)
            assert checkpoint["epoch"] == 1 and math.isfinite(checkpoint["train_loss"])
            assert all(math.isfinite(checkpoint["metrics"][key]) for key in ("epe", "mtre"))
            optimizer_states = checkpoint["optimizer_state_dict"]["state"].values()
            optimizer_steps = [float(state.get("step", 0)) for state in optimizer_states]
            max_step = max(optimizer_steps, default=0)
            scaler_state = checkpoint.get("scaler_state_dict")
            assert max_step > 0, (
                f"No optimizer updates for {name}; AMP may have skipped every step. "
                f"Scaler state: {scaler_state}"
            )
            restored = build_model(cfg["model"], cfg["image_size"]).to(device)
            restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
            before_loss, before, _ = evaluate(restored, criterion, DataLoader(cases, batch_size=2),
                                              device, use_amp=cfg["training"]["amp"])
            roundtrip_path = folder / "roundtrip_state.pt"
            torch.save(restored.state_dict(), roundtrip_path)
            roundtrip = build_model(cfg["model"], cfg["image_size"]).to(device)
            roundtrip.load_state_dict(torch.load(roundtrip_path, map_location=device, weights_only=True), strict=True)
            after_loss, after, _ = evaluate(roundtrip, criterion, DataLoader(cases, batch_size=2),
                                            device, use_amp=cfg["training"]["amp"])
            tolerance = 1e-3 if device.type == "cuda" else 1e-5
            assert math.isfinite(after_loss) and abs(after_loss - before_loss) <= tolerance
            for key in ("epe", "mtre"):
                assert math.isfinite(after[key]) and abs(after[key] - before[key]) <= tolerance
            report["models"][name] = {
                "status": "PASS", "parameters": parameters, "epochs": 1,
                "max_optimizer_step": max_step, "amp_scaler_state": scaler_state,
                "train_samples": 4, "val_samples": 2, "checkpoint_sha256": hashes,
                "roundtrip_state_sha256": sha(roundtrip_path),
                "training_validation_metrics": {k: checkpoint["metrics"][k] for k in ("epe", "mtre")},
                "roundtrip_fixture_metrics": {k: after[k] for k in ("epe", "mtre")},
                "roundtrip_tolerance": tolerance,
                "roundtrip_note": "Same captured cases before/after serialization; not a replay of streaming training-validation cases",
                "seconds": time.perf_counter() - started,
            }
            del restored, roundtrip, checkpoint, criterion, cases
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
