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
from monai.networks.blocks import Warp

from datasets import build_dataset
from models import build_model
from losses import build_loss
from metrics import METRICS, jacobian_determinant
from metrics.tre import mean_tre


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


def _mask_contour(mask_2d: np.ndarray) -> np.ndarray:
    """Approximate mask contour without extra dependencies."""
    mask = mask_2d.astype(bool)
    if not np.any(mask):
        return mask

    neighbors = (
        np.roll(mask, 1, axis=0)
        & np.roll(mask, -1, axis=0)
        & np.roll(mask, 1, axis=1)
        & np.roll(mask, -1, axis=1)
    )
    return mask & (~neighbors)


def _overlay_labels(
    base_slice: np.ndarray,
    primary_mask: np.ndarray | None = None,
    secondary_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Create RGB overlay from grayscale slice and optional masks."""
    base_rgb = np.stack([base_slice, base_slice, base_slice], axis=-1)

    if primary_mask is not None and np.any(primary_mask):
        primary = primary_mask.astype(bool)
        base_rgb[primary, 1] = np.clip(base_rgb[primary, 1] * 0.6 + 0.4, 0.0, 1.0)
        base_rgb[primary, 0] *= 0.6
        base_rgb[primary, 2] *= 0.6

    if secondary_mask is not None and np.any(secondary_mask):
        contour = _mask_contour(secondary_mask)
        base_rgb[contour, 0] = 1.0
        base_rgb[contour, 1] = 0.0
        base_rgb[contour, 2] = 0.0

    return base_rgb


def visualize_slices(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    warped: torch.Tensor,
    moving_label: torch.Tensor | None = None,
    fixed_label: torch.Tensor | None = None,
    warped_moving_label: torch.Tensor | None = None,
):
    """
    Log overlay triplet for easier label-alignment inspection.

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

    moving_3d = moving[0, 0].detach().cpu().numpy()
    fixed_3d = fixed[0, 0].detach().cpu().numpy()
    warped_3d = warped[0, 0].detach().cpu().numpy()

    moving_label_3d = (
        moving_label[0, 0].detach().cpu().numpy()
        if moving_label is not None else None
    )
    fixed_label_3d = (
        fixed_label[0, 0].detach().cpu().numpy()
        if fixed_label is not None else None
    )
    warped_label_3d = (
        warped_moving_label[0, 0].detach().cpu().numpy()
        if warped_moving_label is not None else None
    )

    moving_best_z = int(np.argmax(moving_3d.sum(axis=(1, 2))))
    fixed_best_z = int(np.argmax(fixed_3d.sum(axis=(1, 2))))
    warped_best_z = int(np.argmax(warped_3d.sum(axis=(1, 2))))

    # Moving panel with moving-label overlay.
    moving_img = norm_slice(moving_3d[moving_best_z])
    moving_mask = None if moving_label_3d is None else (moving_label_3d[moving_best_z] > 0.5)
    images.append(
        wandb.Image(
            _overlay_labels(moving_img, primary_mask=moving_mask),
            caption=f"[overlay z={moving_best_z}] moving + moving_label",
        )
    )

    # Fixed panel with fixed-label overlay.
    fixed_img = norm_slice(fixed_3d[fixed_best_z])
    fixed_mask = None if fixed_label_3d is None else (fixed_label_3d[fixed_best_z] > 0.5)
    images.append(
        wandb.Image(
            _overlay_labels(fixed_img, primary_mask=fixed_mask),
            caption=f"[overlay z={fixed_best_z}] fixed + fixed_label",
        )
    )

    # Warped panel with warped-label fill + fixed-label contour for alignment.
    warped_img = norm_slice(warped_3d[warped_best_z])
    warped_mask = None if warped_label_3d is None else (warped_label_3d[warped_best_z] > 0.5)
    fixed_contour_mask = None if fixed_label_3d is None else (fixed_label_3d[warped_best_z] > 0.5)
    images.append(
        wandb.Image(
            _overlay_labels(
                warped_img,
                primary_mask=warped_mask,
                secondary_mask=fixed_contour_mask,
            ),
            caption=f"[overlay z={warped_best_z}] warped + warped_label (green), fixed contour (red)",
        )
    )

    return images

def maybe_wandb_log(enabled: bool, data: dict):
    if enabled:
        wandb.log(data)


def validate_dataset_config(dataset_cfg: dict, split_name: str):
    """Validate dataset-specific config fields before instantiation."""
    dataset_name = dataset_cfg.get("name")
    if dataset_name == "custom_dataset" and not dataset_cfg.get("json_file"):
        raise ValueError(
            f"{split_name}_dataset uses name='custom_dataset' but is missing 'json_file'. "
            "Add json_file: /path/to/manifest.json in the YAML config."
        )


def _get_optional_tensor(
    batch: dict,
    device: torch.device,
    keys: tuple[str, ...],
) -> torch.Tensor | None:
    """Return the first tensor found for the provided keys, moved to device."""
    for key in keys:
        value = batch.get(key, None)
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
    return None


def _compute_registration_loss(
    loss_fn: torch.nn.Module,
    warped: torch.Tensor,
    fixed: torch.Tensor,
    ddf: torch.Tensor,
    gt_dvf: torch.Tensor | None,
    fixed_label: torch.Tensor | None,
    moving_label: torch.Tensor | None,
    warped_moving_label: torch.Tensor | None,
) -> torch.Tensor:
    """
    Compute loss while remaining compatible with existing loss signatures.

    Tries richer signatures first to support label-aware losses.
    """
    try:
        return loss_fn(
            warped,
            fixed,
            ddf,
            gt_dvf,
            fixed_label=fixed_label,
            moving_label=moving_label,
            warped_moving_label=warped_moving_label,
        )
    except TypeError:
        pass

    try:
        return loss_fn(warped, fixed, ddf, gt_dvf)
    except TypeError:
        pass

    try:
        return loss_fn(
            warped,
            fixed,
            fixed_label=fixed_label,
            moving_label=moving_label,
            warped_moving_label=warped_moving_label,
        )
    except TypeError:
        return loss_fn(warped, fixed)


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
    - label-aware loss: loss(..., fixed_label=..., moving_label=..., warped_moving_label=...)
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
        fixed_label = _get_optional_tensor(batch, device, keys=("fixed_label",))
        moving_label = _get_optional_tensor(batch, device, keys=("moving_label",))
        warped_moving_label = _get_optional_tensor(
            batch,
            device,
            keys=("warped_moving_label", "warped_label"),
        )

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

            loss = _compute_registration_loss(
                loss_fn=loss_fn,
                warped=warped,
                fixed=fixed,
                ddf=ddf,
                gt_dvf=gt_dvf,
                fixed_label=fixed_label,
                moving_label=moving_label,
                warped_moving_label=warped_moving_label,
            )

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
    ddf_abs_mean_total = 0.0
    ddf_l2_mean_total = 0.0
    metric_totals = {name: 0.0 for name in METRICS}
    metric_counts = {name: 0 for name in METRICS}
    mtre_total = 0.0
    mtre_count = 0
    visuals = None

    for batch_idx, batch in enumerate(dataloader):
        moving = batch["moving"].to(device, non_blocking=True)
        fixed = batch["fixed"].to(device, non_blocking=True)

        gt_dvf = batch.get("dvf", None)
        if gt_dvf is not None:
            gt_dvf = gt_dvf.to(device, non_blocking=True)
        fixed_label = _get_optional_tensor(batch, device, keys=("fixed_label",))
        moving_label = _get_optional_tensor(batch, device, keys=("moving_label",))
        warped_moving_label = _get_optional_tensor(
            batch,
            device,
            keys=("warped_moving_label", "warped_label"),
        )
        moving_points = _get_optional_tensor(batch, device, keys=("moving_points",))
        fixed_points = _get_optional_tensor(batch, device, keys=("fixed_points",))
        has_points = moving_points is not None and fixed_points is not None

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

            loss = _compute_registration_loss(
                loss_fn=loss_fn,
                warped=warped,
                fixed=fixed,
                ddf=ddf,
                gt_dvf=gt_dvf,
                fixed_label=fixed_label,
                moving_label=moving_label,
                warped_moving_label=warped_moving_label,
            )

        bs = moving.shape[0]
        running_loss += loss.item() * bs
        n_samples += bs
        ddf_abs_mean_total += float(ddf.abs().mean().item()) * bs
        ddf_l2_mean_total += float(torch.sqrt((ddf ** 2).sum(dim=1)).mean().item()) * bs

        for name, fn in METRICS.items():
            if name in {"grad_l2", "neg_jac_ratio", "jac_det_mean", "jac_det_min", "log_jac_std"}:
                metric_totals[name] += fn(ddf) * bs
                metric_counts[name] += bs
            elif name == "epe":
                if gt_dvf is not None:
                    metric_totals[name] += fn(ddf, gt_dvf) * bs
                    metric_counts[name] += bs
            elif name == "dice":
                try:
                    metric_totals[name] += fn(
                        fixed_label=fixed_label,
                        moving_label=moving_label,
                        warped_moving_label=warped_moving_label,
                        ddf=ddf,
                    ) * bs
                    metric_counts[name] += bs
                except ValueError:
                    # Dataset may not provide labels for this batch/split.
                    pass
            else:
                metric_totals[name] += fn(warped, fixed) * bs
                metric_counts[name] += bs

        if has_points:
            mtre_value = mean_tre(ddf, moving_points.float(), fixed_points.float())
            mtre_total += mtre_value * bs
            mtre_count += bs
            
        if batch_idx == 0:
            warped_moving_label_for_vis = warped_moving_label
            if warped_moving_label_for_vis is None and moving_label is not None:
                warped_moving_label_for_vis = Warp(mode="nearest", padding_mode="border")(
                    moving_label.float(),
                    ddf,
                )
            visuals = visualize_slices(
                moving=moving,
                fixed=fixed,
                warped=warped,
                moving_label=moving_label,
                fixed_label=fixed_label,
                warped_moving_label=warped_moving_label_for_vis,
            )

    avg_loss = running_loss / max(n_samples, 1)
    avg_metrics = {}
    for name in METRICS:
        count = metric_counts[name]
        if count > 0:
            avg_metrics[name] = metric_totals[name] / count
        else:
            avg_metrics[name] = float("nan")
    if n_samples > 0:
        avg_metrics["ddf_abs_mean"] = ddf_abs_mean_total / n_samples
        avg_metrics["ddf_l2_mean"] = ddf_l2_mean_total / n_samples
    else:
        avg_metrics["ddf_abs_mean"] = float("nan")
        avg_metrics["ddf_l2_mean"] = float("nan")
    if mtre_count > 0:
        avg_metrics["mtre"] = mtre_total / mtre_count
    else:
        avg_metrics["mtre"] = float("nan")
    return avg_loss, avg_metrics, visuals


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------

def run_training(config_path: str):
    """Run the full training pipeline from a YAML config path."""
    cfg = load_config(config_path)
    print(f"Config path: {config_path}")
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
    validate_dataset_config(cfg["train_dataset"], split_name="train")
    validate_dataset_config(cfg["val_dataset"], split_name="val")
    train_ds = build_dataset(cfg["train_dataset"], split="train")
    val_ds = build_dataset(cfg["val_dataset"], split="val")

    print("---- Dataset sanity check ----")
    item = val_ds[0]
    print("moving:", float(item["moving"].min()), float(item["moving"].max()))
    print("fixed:", float(item["fixed"].min()), float(item["fixed"].max()))
    print("any NaN:", torch.isnan(item["moving"]).any().item(), torch.isnan(item["fixed"]).any().item())
    print("any Inf:", torch.isinf(item["moving"]).any().item(), torch.isinf(item["fixed"]).any().item())
    optional_keys = [
        key for key in (
            "moving_label",
            "fixed_label",
            "moving_mask",
            "fixed_mask",
            "moving_landmarks",
            "fixed_landmarks",
            "moving_points",
            "fixed_points",
        )
        if key in item
    ]
    if "moving_points" in item:
        print("moving_points shape:", tuple(item["moving_points"].shape))
    if "fixed_points" in item:
        print("fixed_points shape:", tuple(item["fixed_points"].shape))
    if optional_keys:
        print("optional fields:", ", ".join(optional_keys))
    if "dvf" not in item:
        print("note: dataset does not provide 'dvf'; EPE metric will be unavailable (NaN).")
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
            if (
                metrics is not None
                and "epe" in metrics
                and np.isfinite(metrics["epe"])
            ):
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
            if "dice" in last_metrics and np.isfinite(last_metrics["dice"]):
                metric_str += f", val_dice = {last_metrics['dice']:.4f}"
            if "epe" in last_metrics and np.isfinite(last_metrics["epe"]):
                metric_str += f", val_epe = {last_metrics['epe']:.4f}"
            if "grad_l2" in last_metrics:
                metric_str += f", val_grad_l2 = {last_metrics['grad_l2']:.4f}"
            if "neg_jac_ratio" in last_metrics:
                metric_str += f", val_neg_jac = {last_metrics['neg_jac_ratio']:.4f}"
            if "jac_det_mean" in last_metrics:
                metric_str += f", val_jac_mean = {last_metrics['jac_det_mean']:.4f}"
            if "log_jac_std" in last_metrics:
                metric_str += f", val_logjac_std = {last_metrics['log_jac_std']:.4f}"
            if "ddf_abs_mean" in last_metrics:
                metric_str += f", val_ddf_abs = {last_metrics['ddf_abs_mean']:.4f}"
            if "ddf_l2_mean" in last_metrics:
                metric_str += f", val_ddf_l2 = {last_metrics['ddf_l2_mean']:.4f}"
            if "mtre" in last_metrics and np.isfinite(last_metrics["mtre"]):
                metric_str += f", val_mtre = {last_metrics['mtre']:.4f}"

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_loss = {train_loss:.4f}, "
            f"val_loss = {last_val_loss:.4f}"
            f"{metric_str}"
        )

    if wandb_enabled:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/deepreg_synth.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
