# train.py

from __future__ import annotations
import os
import argparse
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import wandb

from datasets import build_dataset
from models import build_model
from losses import build_loss
from metrics import METRICS, jacobian_determinant


# -------------------------------------------------------------------------
# Config / utilities
# -------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def visualize_slices(moving: torch.Tensor, fixed: torch.Tensor, warped: torch.Tensor):
    """
    Log two visualization groups:
    1) same-z comparison using fixed best z
    2) each volume's own best z

    Inputs:
        moving, fixed, warped: (B, 1, D, H, W)
    """
    def norm_slice(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - x.min()
        xmax = x.max()
        if xmax > 1e-8:
            x = x / xmax
        return x

    images = []

    moving_3d = moving[0, 0].detach().cpu()
    fixed_3d = fixed[0, 0].detach().cpu()
    warped_3d = warped[0, 0].detach().cpu()

    moving_best_z = int(torch.argmax(moving_3d.sum(dim=(1, 2))).item())
    fixed_best_z = int(torch.argmax(fixed_3d.sum(dim=(1, 2))).item())
    warped_best_z = int(torch.argmax(warped_3d.sum(dim=(1, 2))).item())

    # Group 1: same-z comparison
    same_z = fixed_best_z
    for name, vol_3d in [("moving", moving_3d), ("fixed", fixed_3d), ("warped", warped_3d)]:
        slice_img = vol_3d[same_z].numpy()
        slice_img = norm_slice(slice_img)
        images.append(
            wandb.Image(slice_img, caption=f"[same z={same_z}] {name}")
        )

    # Group 2: each volume's own best z
    own_best = [
        ("moving", moving_3d, moving_best_z),
        ("fixed", fixed_3d, fixed_best_z),
        ("warped", warped_3d, warped_best_z),
    ]
    for name, vol_3d, z in own_best:
        slice_img = vol_3d[z].numpy()
        slice_img = norm_slice(slice_img)
        images.append(
            wandb.Image(slice_img, caption=f"[own best z={z}] {name}")
        )

    return images

def maybe_wandb_log(enabled: bool, data: dict):
    if enabled:
        wandb.log(data)


# -------------------------------------------------------------------------
# Training / evaluation loops
# -------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    """
    Single training epoch.

    Supports both:
    - image-only loss: loss(warped, fixed)
    - DVF-aware loss:  loss(warped, fixed, ddf, gt_dvf)
    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        moving = batch["moving"].to(device, non_blocking=True)
        fixed = batch["fixed"].to(device, non_blocking=True)

        gt_dvf = batch.get("dvf", None)
        if gt_dvf is not None:
            gt_dvf = gt_dvf.to(device, non_blocking=True)

        if torch.isnan(moving).any() or torch.isinf(moving).any():
            print("Warning: NaN/Inf detected in moving image. Skipping batch.")
            continue
        if torch.isnan(fixed).any() or torch.isinf(fixed).any():
            print("Warning: NaN/Inf detected in fixed image. Skipping batch.")
            continue

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            warped, ddf = model(moving, fixed)

            if torch.isnan(warped).any() or torch.isinf(warped).any():
                print("Warning: NaN/Inf detected in warped output. Skipping batch.")
                continue
            if torch.isnan(ddf).any() or torch.isinf(ddf).any():
                print("Warning: NaN/Inf detected in ddf output. Skipping batch.")
                continue

            try:
                loss = loss_fn(warped, fixed, ddf, gt_dvf)
            except TypeError:
                loss = loss_fn(warped, fixed)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = moving.shape[0]
        running_loss += loss.item() * bs
        n_samples += bs

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
):
    """
    Validation loop.

    Computes:
      - loss
      - metrics defined in METRICS
      - one visualization batch for W&B
      - detailed debug prints for batch_idx == 0 only
    """
    model.eval()
    running_loss = 0.0
    n_samples = 0
    metric_totals = {name: 0.0 for name in METRICS}
    visuals = None

    for batch_idx, batch in enumerate(dataloader):
        moving = batch["moving"].to(device, non_blocking=True)
        fixed = batch["fixed"].to(device, non_blocking=True)

        gt_dvf = batch.get("dvf", None)
        if gt_dvf is not None:
            gt_dvf = gt_dvf.to(device, non_blocking=True)

        if torch.isnan(moving).any() or torch.isinf(moving).any():
            print("Warning: NaN/Inf detected in moving image (eval). Skipping batch.")
            continue
        if torch.isnan(fixed).any() or torch.isinf(fixed).any():
            print("Warning: NaN/Inf detected in fixed image (eval). Skipping batch.")
            continue

        with autocast("cuda", enabled=use_amp):
            warped, ddf = model(moving, fixed)

            if torch.isnan(warped).any() or torch.isinf(warped).any():
                print("Warning: NaN/Inf detected in warped output (eval). Skipping batch.")
                continue
            if torch.isnan(ddf).any() or torch.isinf(ddf).any():
                print("Warning: NaN/Inf detected in ddf output (eval). Skipping batch.")
                continue

            try:
                loss = loss_fn(warped, fixed, ddf, gt_dvf)
            except TypeError:
                loss = loss_fn(warped, fixed)

        bs = moving.shape[0]
        running_loss += loss.item() * bs
        n_samples += bs

        for name, fn in METRICS.items():
            if name in {"grad_l2", "neg_jac_ratio", "jac_det_mean", "jac_det_min", "log_jac_std"}:
                metric_totals[name] += fn(ddf) * bs
            elif name == "epe":
                if gt_dvf is not None:
                    metric_totals[name] += fn(ddf, gt_dvf) * bs
            else:
                metric_totals[name] += fn(warped, fixed) * bs

        if batch_idx == 0:
            visuals = visualize_slices(moving, fixed, warped)

    avg_loss = running_loss / max(n_samples, 1)
    avg_metrics = {
        name: metric_totals[name] / max(n_samples, 1) for name in METRICS
    }
    return avg_loss, avg_metrics, visuals


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment1.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config path: {args.config}")
    print("---- Full Config ----")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print("---------------------")

    wandb_enabled = bool(cfg.get("wandb", {}).get("enabled", False))
    if wandb_enabled:
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"]["run_name"],
            config=cfg,
            mode="offline" if cfg["wandb"]["offline"] else "online",
        )

    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    use_amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")

    set_seed(cfg["training"]["seed"])

    # -------------------------
    # checkpoint save settings
    # -------------------------
    save_dir = cfg["training"].get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    best_epe = float("inf")
    best_val_loss = float("inf")

    # Datasets / loaders
    train_ds = build_dataset(cfg["train_dataset"], split="train")
    val_ds = build_dataset(cfg["val_dataset"], split="val")

    print("---- Dataset sanity check ----")
    item = val_ds[0]
    print("moving:", float(item["moving"].min()), float(item["moving"].max()))
    print("fixed:", float(item["fixed"].min()), float(item["fixed"].max()))
    print("any NaN:", torch.isnan(item["moving"]).any().item(), torch.isnan(item["fixed"]).any().item())
    print("any Inf:", torch.isinf(item["moving"]).any().item(), torch.isinf(item["fixed"]).any().item())
    print("--------------------------------")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # Model / loss / optimizer
    model = build_model(cfg["model"], image_size=cfg["image_size"]).to(device)
    loss_fn = build_loss(cfg["loss"]).to(device)

    optimizer_name = cfg["optimizer"].get("name", "Adam").lower()
    lr = float(cfg["optimizer"]["lr"])
    weight_decay = float(cfg["optimizer"].get("weight_decay", 0.0))

    if lr > 1e-3:
        print(f"Warning: Learning rate {lr} is very high. Consider using a lower value (e.g., 1e-4 or 1e-5).")

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'")

    scaler = GradScaler(enabled=use_amp)

    epochs = cfg["training"]["epochs"]
    val_every = cfg["training"]["val_every"]

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Checkpoint dir: {save_dir}")

    last_val_loss = float("nan")
    last_metrics = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        log_dict = {"train/loss": train_loss, "epoch": epoch}

        if val_every > 0 and (epoch % val_every == 0):
            val_loss, metrics, visuals = evaluate(
                model=model,
                loss_fn=loss_fn,
                dataloader=val_loader,
                device=device,
                use_amp=use_amp,
            )
            last_val_loss = val_loss
            last_metrics = metrics

            log_dict["val/loss"] = val_loss
            for name, value in metrics.items():
                log_dict[f"val/{name}"] = value

            if visuals is not None:
                maybe_wandb_log(wandb_enabled, {"val/slices": visuals, "epoch": epoch})

            # -------------------------
            # save best by EPE
            # -------------------------
            if metrics is not None and "epe" in metrics:
                if metrics["epe"] < best_epe:
                    best_epe = metrics["epe"]
                    best_epe_path = os.path.join(save_dir, "best_epe.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict() if use_amp else None,
                            "config": cfg,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "metrics": metrics,
                            "best_epe": best_epe,
                        },
                        best_epe_path,
                    )
                    print(
                        f"[Checkpoint] Saved best EPE model to {best_epe_path} "
                        f"(epoch={epoch}, val_epe={best_epe:.6f})"
                    )

            # -------------------------
            # save best by val_loss
            # -------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_loss_path = os.path.join(save_dir, "best_val_loss.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if use_amp else None,
                        "config": cfg,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "metrics": metrics,
                        "best_val_loss": best_val_loss,
                    },
                    best_loss_path,
                )
                print(
                    f"[Checkpoint] Saved best val-loss model to {best_loss_path} "
                    f"(epoch={epoch}, val_loss={best_val_loss:.6f})"
                )

        maybe_wandb_log(wandb_enabled, log_dict)

        # -------------------------
        # save last checkpoint every epoch
        # -------------------------
        last_path = os.path.join(save_dir, "last.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "config": cfg,
                "train_loss": train_loss,
                "last_val_loss": last_val_loss,
                "last_metrics": last_metrics,
            },
            last_path,
        )

        metric_str = ""
        if last_metrics is not None:
            if "ncc" in last_metrics:
                metric_str += f", val_ncc = {last_metrics['ncc']:.4f}"
            if "epe" in last_metrics:
                metric_str += f", val_epe = {last_metrics['epe']:.4f}"
            if "grad_l2" in last_metrics:
                metric_str += f", val_grad_l2 = {last_metrics['grad_l2']:.4f}"
            if "neg_jac_ratio" in last_metrics:
                metric_str += f", val_neg_jac = {last_metrics['neg_jac_ratio']:.4f}"
            if "jac_det_mean" in last_metrics:
                metric_str += f", val_jac_mean = {last_metrics['jac_det_mean']:.4f}"
            if "log_jac_std" in last_metrics:
                metric_str += f", val_logjac_std = {last_metrics['log_jac_std']:.4f}"

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_loss = {train_loss:.4f}, "
            f"val_loss = {last_val_loss:.4f}"
            f"{metric_str}"
        )

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()