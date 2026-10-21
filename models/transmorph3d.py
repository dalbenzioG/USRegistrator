"""TransMorph3D - TransMorph (Chen et al., 2022) built from MONAI Swin blocks."""

from __future__ import annotations

import warnings
from typing import Sequence

import torch
from torch import nn

from monai.networks.blocks import Convolution, UpSample, Warp
from monai.networks.nets.swin_unetr import BasicLayer, PatchEmbed, PatchMerging

from .registry import register_model


# Architecture presets from the reference implementation
# (junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration,
# TransMorph/models/configs_TransMorph.py). Every preset shares
# patch_size=4, window_size=(5, 6, 7), mlp_ratio=4, qkv_bias=False,
# drop_path_rate=0.3, patch_norm=True and reg_head_chan=16.
TRANSMORPH_VARIANTS: dict[str, dict] = {
    "tiny": {"embed_dim": 6, "depths": (2, 2, 4, 2), "num_heads": (2, 2, 4, 4)},
    "small": {"embed_dim": 48, "depths": (2, 2, 4, 2), "num_heads": (4, 4, 4, 4)},
    "base": {"embed_dim": 96, "depths": (2, 2, 4, 2), "num_heads": (4, 4, 8, 8)},
    "large": {"embed_dim": 128, "depths": (2, 2, 12, 2), "num_heads": (4, 4, 8, 16)},
}

PATCH_SIZE = 4
NUM_STAGES = 4
# patch embedding (4) x three patch-merging steps (2^3) = 32
SIZE_DIVISOR = PATCH_SIZE * 2 ** (NUM_STAGES - 1)


def _conv_block(in_channels: int, out_channels: int) -> Convolution:
    """Conv3d -> InstanceNorm3d -> LeakyReLU, matching TransMorph's Conv3dReLU."""
    return Convolution(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=1,
        norm="instance",
        act=("leakyrelu", {"inplace": True}),
        bias=False,
    )


class _SwinEncoder(nn.Module):
    """TransMorph's Swin encoder, assembled from MONAI Swin blocks.

    Each stage runs its Swin blocks and is tapped *before* patch merging,
    then normalised. This reproduces the reference encoder's ``out_indices
    = (0, 1, 2, 3)`` skip taps, which MONAI's ``SwinTransformer`` does not
    expose (it taps after merging).

    Returns features with ``embed_dim * 2**i`` channels at stride
    ``4 * 2**i``, for i in 0..3.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depths: Sequence[int],
        num_heads: Sequence[int],
        window_size: Sequence[int],
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        patch_norm: bool,
        use_checkpoint: bool,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            patch_size=(PATCH_SIZE,) * 3,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None,  # type: ignore[arg-type]
            spatial_dims=3,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decays linearly over all blocks, as upstream.
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.merges = nn.ModuleList()
        for i in range(NUM_STAGES):
            dim = embed_dim * 2**i
            start = sum(depths[:i])
            self.layers.append(
                BasicLayer(
                    dim=dim,
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    drop_path=dpr[start : start + depths[i]],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=nn.LayerNorm,
                    downsample=None,  # tap before merging, then merge below
                    use_checkpoint=use_checkpoint,
                )
            )
            self.norms.append(nn.LayerNorm(dim))
            if i < NUM_STAGES - 1:
                self.merges.append(PatchMerging(dim=dim, spatial_dims=3))

    @staticmethod
    def _channels_last(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 4, 1)

    @staticmethod
    def _channels_first(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 4, 1, 2, 3)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.pos_drop(self.patch_embed(x))

        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x.contiguous())
            # Normalise the skip only; the un-normalised tensor flows on.
            features.append(
                self._channels_first(self.norms[i](self._channels_last(x)))
            )
            if i < len(self.merges):
                x = self._channels_first(self.merges[i](self._channels_last(x)))
        return features


class _DecoderBlock(nn.Module):
    """Trilinear upsample, concatenate skip, then two conv blocks."""

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.up = UpSample(
            spatial_dims=3,
            scale_factor=2,
            mode="nontrainable",
            interp_mode="trilinear",
            align_corners=False,
        )
        self.conv1 = _conv_block(in_channels + skip_channels, out_channels)
        self.conv2 = _conv_block(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class TransMorph3D(nn.Module):
    """TransMorph: Transformer for unsupervised medical image registration.

    Faithful reimplementation of Chen et al., *Medical Image Analysis* 2022,
    assembled from MONAI building blocks. A Swin Transformer encoder and a
    convolutional decoder predict a dense displacement field from the
    concatenated moving/fixed pair; MONAI's ``Warp`` resamples the moving
    image, replacing the reference ``SpatialTransformer``.

    Two full-resolution convolutional skip paths run straight off the input
    (``if_convskip``), and the four encoder stages supply transformer skips
    (``if_transskip``).

    Components (all from MONAI):
      - ``PatchEmbed`` + ``BasicLayer`` + ``PatchMerging``: Swin encoder
      - ``Convolution`` + ``UpSample``: convolutional decoder
      - ``Warp``: applies the ddf to the moving image

    Note: patch embedding (4) and three patch-merging steps (2^3) mean every
    spatial dimension must be divisible by 32.

    Known deviation from the reference implementation. Where a stage's feature
    map is smaller than ``window_size``, MONAI clamps the window to the feature
    map and drops the cyclic shift, following the official Swin Transformer
    implementation. The reference instead keeps the configured window and
    zero-pads the feature map up to a multiple of it, so attention there also
    covers padding. Outputs are bit-identical in every stage whose resolution
    exceeds ``window_size`` (verified by transferring reference weights); they
    differ in the deeper stages. The default ``window_size=(5, 6, 7)`` is tuned
    for the reference input size of 160x192x224, whose final stage is exactly
    5x6x7, so smaller inputs always clamp somewhere. ``__init__`` warns with the
    affected stages.

    Inputs:
      moving: (B, 1, D, H, W)
      fixed:  (B, 1, D, H, W)

    Outputs:
      warped: (B, 1, D, H, W)  -- moving warped into fixed space
      ddf:    (B, 3, D, H, W)  -- displacement field, voxel units, (z, y, x)
    """

    def __init__(
        self,
        image_size: Sequence[int],
        variant: str = "base",
        embed_dim: int | None = None,
        depths: Sequence[int] | None = None,
        num_heads: Sequence[int] | None = None,
        window_size: Sequence[int] = (5, 6, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.3,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        reg_head_chan: int = 16,
        if_transskip: bool = True,
        if_convskip: bool = True,
        warp_mode: str = "bilinear",
        warp_padding_mode: str = "zeros",
    ):
        super().__init__()

        if len(image_size) != 3:
            raise ValueError(
                f"image_size must be length 3 (D, H, W), got {image_size}"
            )

        self.image_size = [int(s) for s in image_size]
        for dim in self.image_size:
            if dim < SIZE_DIVISOR or dim % SIZE_DIVISOR != 0:
                raise ValueError(
                    f"TransMorph3D requires each spatial dimension to be a positive "
                    f"multiple of {SIZE_DIVISOR} (patch size {PATCH_SIZE} and "
                    f"{NUM_STAGES - 1} patch-merging steps), got "
                    f"image_size={self.image_size}."
                )

        variant = str(variant).lower()
        if variant not in TRANSMORPH_VARIANTS:
            raise ValueError(
                f"Unknown TransMorph variant '{variant}'. "
                f"Available: {sorted(TRANSMORPH_VARIANTS)}"
            )
        preset = TRANSMORPH_VARIANTS[variant]
        embed_dim = preset["embed_dim"] if embed_dim is None else int(embed_dim)
        depths = tuple(preset["depths"] if depths is None else depths)
        num_heads = tuple(preset["num_heads"] if num_heads is None else num_heads)

        if len(depths) != NUM_STAGES or len(num_heads) != NUM_STAGES:
            raise ValueError(
                f"depths and num_heads must both have {NUM_STAGES} entries, "
                f"got depths={depths}, num_heads={num_heads}"
            )
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")

        self.variant = variant
        self.if_transskip = if_transskip
        self.if_convskip = if_convskip

        self.transformer = _SwinEncoder(
            in_channels=2,  # moving + fixed concatenated along channel dim
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=tuple(window_size),
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
        )

        self.up0 = _DecoderBlock(
            embed_dim * 8, embed_dim * 4, embed_dim * 4 if if_transskip else 0
        )
        self.up1 = _DecoderBlock(
            embed_dim * 4, embed_dim * 2, embed_dim * 2 if if_transskip else 0
        )
        self.up2 = _DecoderBlock(
            embed_dim * 2, embed_dim, embed_dim if if_transskip else 0
        )
        self.up3 = _DecoderBlock(
            embed_dim, embed_dim // 2, embed_dim // 2 if if_convskip else 0
        )
        self.up4 = _DecoderBlock(
            embed_dim // 2, reg_head_chan, reg_head_chan if if_convskip else 0
        )

        # Convolutional skip paths taken directly from the input pair.
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = _conv_block(2, embed_dim // 2)
        self.c2 = _conv_block(2, reg_head_chan)

        self.reg_head = nn.Conv3d(reg_head_chan, 3, kernel_size=3, padding=1)
        # Near-zero init so training starts close to the identity transform.
        nn.init.normal_(self.reg_head.weight, std=1e-5)
        nn.init.zeros_(self.reg_head.bias)

        self.warp = Warp(mode=warp_mode, padding_mode=warp_padding_mode)

        clamped = [
            (i, res)
            for i in range(NUM_STAGES)
            for res in [tuple(s // (PATCH_SIZE * 2**i) for s in self.image_size)]
            if any(r <= w for r, w in zip(res, window_size))
        ]
        if clamped:
            warnings.warn(
                f"TransMorph3D: window_size={tuple(window_size)} does not fit "
                f"stage(s) {[i for i, _ in clamped]} at image_size="
                f"{self.image_size} (stage resolutions {[r for _, r in clamped]}). "
                "MONAI clamps the window and drops the shift there, so those "
                "stages differ from the reference implementation. Reduce "
                "window_size or enlarge image_size to avoid this.",
                stacklevel=2,
            )

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

        if self.if_convskip:
            f4 = self.c1(self.avg_pool(x))  # (B, embed_dim // 2, D/2, ...)
            f5 = self.c2(x)                 # (B, reg_head_chan, D, ...)
        else:
            f4 = f5 = None

        feats = self.transformer(x)
        f1, f2, f3 = (feats[2], feats[1], feats[0]) if self.if_transskip else (None,) * 3

        y = self.up0(feats[3], f1)
        y = self.up1(y, f2)
        y = self.up2(y, f3)
        y = self.up3(y, f4)
        y = self.up4(y, f5)

        ddf = self.reg_head(y)  # (B, 3, D, H, W)
        warped = self.warp(moving, ddf)
        return warped, ddf


@register_model("transmorph3d")
def create_transmorph3d(
    image_size: Sequence[int],
    variant: str = "base",
    embed_dim: int | None = None,
    depths: Sequence[int] | None = None,
    num_heads: Sequence[int] | None = None,
    window_size: Sequence[int] = (5, 6, 7),
    mlp_ratio: float = 4.0,
    qkv_bias: bool = False,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.3,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    reg_head_chan: int = 16,
    if_transskip: bool = True,
    if_convskip: bool = True,
    warp_mode: str = "bilinear",
    warp_padding_mode: str = "zeros",
) -> nn.Module:
    """
    Factory for TransMorph3D.

    Config example:
        model:
          name: transmorph3d
          variant: base          # tiny | small | base | large
          drop_path_rate: 0.3
          warp_mode: bilinear
          warp_padding_mode: zeros

    Set `embed_dim`, `depths` or `num_heads` to override the preset.
    """
    return TransMorph3D(
        image_size=image_size,
        variant=variant,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        reg_head_chan=reg_head_chan,
        if_transskip=if_transskip,
        if_convskip=if_convskip,
        warp_mode=warp_mode,
        warp_padding_mode=warp_padding_mode,
    )
