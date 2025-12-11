# models.py

from __future__ import annotations

from typing import Dict, Callable, Sequence, Any

import torch
from torch import nn

from monai.networks.nets import GlobalNet, LocalNet, UNet
from monai.networks.blocks import Warp


# -------------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """
    Decorator to register a model factory under a string key.

    Usage:
        @register_model("globalnet3d")
        def create_globalnet3d(...):
            ...
    """
    def decorator(fn: Callable[..., nn.Module]):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def build_model(cfg: dict, image_size: Sequence[int]) -> nn.Module:
    """
    Build a model from a config dict.

    Expected config format:
        cfg = {
            "name": "globalnet3d",
            ... other kwargs passed to the factory ...
        }

    Args:
        cfg: model configuration dictionary.
        image_size: (D, H, W) of the 3D volume.

    Returns:
        Instantiated nn.Module.
    """
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return MODEL_REGISTRY[name](image_size=image_size, **kwargs)


# -------------------------------------------------------------------------
# GlobalNet3D
# -------------------------------------------------------------------------

class GlobalNet3D(nn.Module):
    """
    MONAI GlobalNet + Warp for 3D registration.

    Components:
      - GlobalNet: predicts 3D displacement field (ddf)
      - Warp: applies ddf to the moving image

    Inputs:
      moving: (B, 1, D, H, W)
      fixed:  (B, 1, D, H, W)

    Outputs:
      warped: (B, 1, D, H, W)  -- moving warped into fixed space
      ddf:    (B, 3, D, H, W)  -- displacement field
    """

    def __init__(
        self,
        image_size: Sequence[int],
        num_channel_initial: int = 16,
        depth: int = 3,
        warp_mode: str = "bilinear",
        warp_padding_mode: str = "border",
    ):
        super().__init__()

        if len(image_size) != 3:
            raise ValueError(
                f"image_size must be length 3 (D, H, W), got {image_size}"
            )

        self.image_size = [int(s) for s in image_size]

        self.net = GlobalNet(
            image_size=self.image_size,
            spatial_dims=3,
            in_channels=2,  # moving + fixed concatenated along channel dim
            num_channel_initial=num_channel_initial,
            depth=depth,
        )

        self.warp = Warp(mode=warp_mode, padding_mode=warp_padding_mode)

        # Optional but strongly recommended:
        # Zero-init the last conv layer inside GlobalNet
        for m in self.net.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=1e-5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        """
        Args:
            moving: (B, 1, D, H, W)
            fixed:  (B, 1, D, H, W)

        Returns:
            warped: (B, 1, D, H, W)
            ddf:    (B, 3, D, H, W)
        """
        x = torch.cat([moving, fixed], dim=1)  # (B, 2, D, H, W)
        # Predict raw displacement
        ddf_raw = self.net(x)  # (B,3,D,H,W)

        # ---- ① Restrict DVF into reasonable numeric range ----
        # Use tanh + scaling
        max_disp = 0.2  # MUST match synthetic DVF max_disp
        ddf = torch.tanh(ddf_raw) * max_disp

        # ---- ② Warp ----
        warped = self.warp(moving, ddf)

        return warped, ddf


@register_model("globalnet3d")
def create_globalnet3d(
    image_size: Sequence[int],
    num_channel_initial: int = 16,
    depth: int = 3,
    warp_mode: str = "bilinear",
    warp_padding_mode: str = "border",
) -> nn.Module:
    """
    Factory for GlobalNet3D.

    Config example:
        model:
          name: globalnet3d
          num_channel_initial: 16
          depth: 3
          warp_mode: bilinear
          warp_padding_mode: border
    """
    return GlobalNet3D(
        image_size=image_size,
        num_channel_initial=num_channel_initial,
        depth=depth,
        warp_mode=warp_mode,
        warp_padding_mode=warp_padding_mode,
    )


# -------------------------------------------------------------------------
# Optional extra models (same interface) – can be used later
# -------------------------------------------------------------------------

class LocalNet3D(nn.Module):
    """
    MONAI LocalNet + Warp for 3D registration.
    """

    def __init__(
        self,
        image_size: Sequence[int],
        in_channels: int = 2,
        num_channel_initial: int = 16,
        depth: int = 3,
        warp_mode: str = "bilinear",
        warp_padding_mode: str = "border",
    ):
        super().__init__()

        self.net = LocalNet(
            spatial_dims=3,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            depth=depth,
        )
        self.warp = Warp(mode=warp_mode, padding_mode=warp_padding_mode)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        x = torch.cat([moving, fixed], dim=1)  # (B, 2, D, H, W)
        ddf = self.net(x)                      # (B, 3, D, H, W)
        warped = self.warp(moving, ddf)
        return warped, ddf


@register_model("localnet3d")
def create_localnet3d(
    image_size: Sequence[int],
    in_channels: int = 2,
    num_channel_initial: int = 16,
    depth: int = 3,
    warp_mode: str = "bilinear",
    warp_padding_mode: str = "border",
) -> nn.Module:
    return LocalNet3D(
        image_size=image_size,
        in_channels=in_channels,
        num_channel_initial=num_channel_initial,
        depth=depth,
        warp_mode=warp_mode,
        warp_padding_mode=warp_padding_mode,
    )


class UNetReg3D(nn.Module):
    """
    UNet-based 3D registration network (MONAI UNet + Warp).

    This predicts a dense displacement field with a standard UNet.
    """

    def __init__(
        self,
        image_size: Sequence[int],
        in_channels: int = 2,
        out_channels: int = 3,
        channels: Sequence[int] = (16, 32, 64, 128, 256),
        strides: Sequence[int] = (2, 2, 2, 2),
        num_res_units: int = 2,
        warp_mode: str = "bilinear",
        warp_padding_mode: str = "border",
    ):
        super().__init__()

        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=list(channels),
            strides=list(strides),
            num_res_units=num_res_units,
        )
        self.warp = Warp(mode=warp_mode, padding_mode=warp_padding_mode)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        x = torch.cat([moving, fixed], dim=1)
        ddf = self.net(x)  # (B,3,D,H,W)
        warped = self.warp(moving, ddf)
        return warped, ddf


@register_model("unetreg3d")
def create_unetreg3d(
    image_size: Sequence[int],
    in_channels: int = 2,
    out_channels: int = 3,
    channels: Sequence[int] = (16, 32, 64, 128, 256),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
    warp_mode: str = "bilinear",
    warp_padding_mode: str = "border",
) -> nn.Module:
    return UNetReg3D(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        warp_mode=warp_mode,
        warp_padding_mode=warp_padding_mode,
    )
