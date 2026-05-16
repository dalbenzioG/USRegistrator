"""Custom medical registration dataset driven by a JSON manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    Spacingd,
)

from .registry import register_dataset


REQUIRED_IMAGE_KEYS = ("moving", "fixed")
OPTIONAL_IMAGE_KEYS = ("moving_label", "fixed_label", "moving_mask", "fixed_mask")
OPTIONAL_LANDMARK_KEYS = ("moving_landmarks", "fixed_landmarks")


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _load_manifest_cases(json_file: str, split: str) -> list[dict[str, Any]]:
    manifest_path = Path(json_file).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset json_file not found: '{manifest_path}'. "
            "Set train_dataset/val_dataset json_file to a valid path."
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError("Dataset JSON must be an object with top-level 'train' and 'val' arrays.")

    required_splits = ("train", "val")
    for split_name in required_splits:
        if split_name not in manifest:
            raise ValueError(
                f"Dataset JSON is missing top-level '{split_name}' list. "
                "Expected format: {'train': [...], 'val': [...]}."
            )
        if not isinstance(manifest[split_name], list):
            raise ValueError(f"Dataset JSON field '{split_name}' must be a list of case dictionaries.")

    if split not in manifest:
        raise ValueError(f"Unknown split '{split}'. Expected one of {required_splits}.")

    base_dir = manifest_path.parent
    normalized_cases: list[dict[str, Any]] = []
    for idx, case in enumerate(manifest[split]):
        if not isinstance(case, dict):
            raise ValueError(f"Case #{idx} in split '{split}' must be an object/dictionary.")

        normalized: dict[str, Any] = {}

        for key in REQUIRED_IMAGE_KEYS:
            value = case.get(key, None)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Case #{idx} in split '{split}' must define a non-empty string '{key}'."
                )
            resolved = _resolve_path(value, base_dir)
            if not Path(resolved).exists():
                raise FileNotFoundError(
                    f"Case #{idx} in split '{split}' references missing file for '{key}': '{resolved}'."
                )
            normalized[key] = resolved

        for key in OPTIONAL_IMAGE_KEYS:
            value = case.get(key, None)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Case #{idx} in split '{split}' has invalid '{key}'. "
                    "Use a file path string or omit the field."
                )
            resolved = _resolve_path(value, base_dir)
            if not Path(resolved).exists():
                raise FileNotFoundError(
                    f"Case #{idx} in split '{split}' references missing file for '{key}': '{resolved}'."
                )
            normalized[key] = resolved

        for key in OPTIONAL_LANDMARK_KEYS:
            value = case.get(key, None)
            if value is None:
                continue
            if isinstance(value, str):
                resolved = _resolve_path(value, base_dir)
                if not Path(resolved).exists():
                    raise FileNotFoundError(
                        f"Case #{idx} in split '{split}' references missing file for '{key}': '{resolved}'."
                    )
                normalized[key] = resolved
            elif isinstance(value, list):
                normalized[key] = value
            else:
                raise ValueError(
                    f"Case #{idx} in split '{split}' has invalid '{key}'. "
                    "Use a path string, list of coordinates, or omit the field."
                )

        # Keep any user metadata fields untouched so the manifest stays extensible.
        for key, value in case.items():
            if key in normalized:
                continue
            if key in REQUIRED_IMAGE_KEYS or key in OPTIONAL_IMAGE_KEYS or key in OPTIONAL_LANDMARK_KEYS:
                continue
            normalized[key] = value

        normalized_cases.append(normalized)

    if not normalized_cases:
        raise ValueError(
            f"No cases were found for split '{split}' in '{manifest_path}'. "
            "Add at least one case with 'moving' and 'fixed'."
        )

    return normalized_cases


def _load_landmarks_file(path: str) -> np.ndarray:
    landmark_path = Path(path)
    suffix = landmark_path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(landmark_path)
    elif suffix == ".npz":
        with np.load(landmark_path) as npz:
            if not npz.files:
                raise ValueError(f"Landmark file '{landmark_path}' is empty.")
            arr = npz[npz.files[0]]
    elif suffix in {".txt", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        arr = np.loadtxt(landmark_path, delimiter=delimiter)
    else:
        raise ValueError(
            f"Unsupported landmarks file extension for '{landmark_path}'. "
            "Supported: .npy, .npz, .txt, .csv"
        )

    return np.asarray(arr, dtype=np.float32)


class LoadLandmarksd(MapTransform):
    """Load landmarks from a file path or inline coordinate list."""

    def __init__(self, keys: Iterable[str]):
        super().__init__(keys)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            value = d.get(key, None)
            if value is None:
                continue

            if isinstance(value, str):
                arr = _load_landmarks_file(value)
            else:
                arr = np.asarray(value, dtype=np.float32)

            if arr.ndim == 1:
                if arr.size % 3 != 0:
                    raise ValueError(
                        f"Landmarks in '{key}' must have 3 values per point. "
                        f"Got {arr.size} values."
                    )
                arr = arr.reshape(-1, 3)

            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(
                    f"Landmarks in '{key}' must have shape (N, 3). Got shape {arr.shape}."
                )

            d[key] = arr
        return d


class InterpolateToSized(MapTransform):
    """Resize image-like tensors to a fixed spatial size via interpolation."""

    def __init__(self, keys: Iterable[str], spatial_size: tuple[int, int, int]):
        super().__init__(keys)
        self.spatial_size = tuple(int(x) for x in spatial_size)
        if len(self.spatial_size) != 3:
            raise ValueError(f"spatial_size must have 3 values, got {self.spatial_size}")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            value = d.get(key, None)
            if value is None:
                continue

            tensor = torch.as_tensor(value).float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4:
                continue

            mode = "nearest" if key.endswith("_label") or key.endswith("_mask") else "trilinear"
            if mode == "nearest":
                resized = F.interpolate(tensor.unsqueeze(0), size=self.spatial_size, mode=mode)
            else:
                resized = F.interpolate(
                    tensor.unsqueeze(0),
                    size=self.spatial_size,
                    mode=mode,
                    align_corners=True,
                )
            d[key] = resized.squeeze(0)

        return d


class ModalityAwareIntensityNormd(MapTransform):
    """Apply CT-window or quantile normalization based on modality metadata."""

    def __init__(
        self,
        keys: Iterable[str],
        ct_window: tuple[float, float] = (-1000.0, 1000.0),
        quantile_range: tuple[float, float] = (0.01, 0.99),
        default_is_ct: bool = False,
    ):
        super().__init__(keys)
        self.ct_window = (float(ct_window[0]), float(ct_window[1]))
        self.quantile_range = (float(quantile_range[0]), float(quantile_range[1]))
        self.default_is_ct = bool(default_is_ct)

    @staticmethod
    def _infer_ct_from_modality(value: Any, fallback: bool) -> bool:
        if value is None:
            return fallback
        text = str(value).strip().lower()
        if not text:
            return fallback
        if "ct" in text:
            return True
        if "us" in text or "ultrasound" in text:
            return False
        return fallback

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            value = d.get(key, None)
            if value is None:
                continue

            tensor = torch.as_tensor(value).float()
            modality_key = f"{key}_modality"
            is_ct = self._infer_ct_from_modality(d.get(modality_key, None), self.default_is_ct)

            if is_ct:
                im_min, im_max = self.ct_window
            else:
                flat = tensor.reshape(-1)
                q_low, q_high = self.quantile_range
                im_min = float(torch.quantile(flat, q_low).item())
                im_max = float(torch.quantile(flat, q_high).item())

            normalized = torch.clamp(tensor, min=im_min, max=im_max)
            normalized = normalized - im_min
            intensity_range = im_max - im_min
            if intensity_range > 0.0:
                normalized = normalized / intensity_range

            d[key] = normalized
        return d


def _collect_present_keys(cases: list[dict[str, Any]], candidate_keys: tuple[str, ...]) -> list[str]:
    present: list[str] = []
    for key in candidate_keys:
        if any(key in case for case in cases):
            present.append(key)
    return present


def _spacing_mode_for_key(key: str) -> str:
    if key.endswith("_label") or key.endswith("_mask"):
        return "nearest"
    return "bilinear"


def _build_transforms(
    cases: list[dict[str, Any]],
    image_size: tuple[int, int, int] | None,
    orientation: str | None,
    spacing: tuple[float, float, float] | None,
    ct_window: tuple[float, float],
    quantile_range: tuple[float, float],
    default_is_ct: bool,
    normalize_intensity: bool,
):
    image_keys = _collect_present_keys(cases, REQUIRED_IMAGE_KEYS + OPTIONAL_IMAGE_KEYS)
    landmark_keys = _collect_present_keys(cases, OPTIONAL_LANDMARK_KEYS)

    transforms = [
        LoadImaged(keys=image_keys, image_only=True),
        EnsureChannelFirstd(keys=image_keys),
    ]

    if orientation:
        transforms.append(Orientationd(keys=image_keys, axcodes=orientation))

    if spacing is not None:
        modes = [_spacing_mode_for_key(key) for key in image_keys]
        transforms.append(Spacingd(keys=image_keys, pixdim=spacing, mode=modes))

    # Multigradicon-style preprocessing path:
    # 1) interpolation to target shape
    # 2) modality-aware intensity normalization
    if image_size is not None:
        transforms.append(InterpolateToSized(keys=image_keys, spatial_size=image_size))
    if normalize_intensity:
        transforms.append(
            ModalityAwareIntensityNormd(
                keys=["moving", "fixed"],
                ct_window=ct_window,
                quantile_range=quantile_range,
                default_is_ct=default_is_ct,
            )
        )

    if landmark_keys:
        transforms.append(LoadLandmarksd(keys=landmark_keys))

    transforms.append(EnsureTyped(keys=image_keys + landmark_keys, dtype=torch.float32, track_meta=False))
    return Compose(transforms)


@register_dataset("custom_dataset")
def create_custom_dataset(
    split: str,
    json_file: str,
    image_size=(64, 64, 64),
    orientation: str | None = "RAS",
    spacing: tuple[float, float, float] | list[float] | None = None,
    preprocess_style: str = "multigradicon",
    ct_window: tuple[float, float] | list[float] = (-1000.0, 1000.0),
    quantile_range: tuple[float, float] | list[float] = (0.01, 0.99),
    default_is_ct: bool = False,
    # Legacy options retained for config compatibility (unused in multigradicon mode).
    roi_from_labels: bool | None = None,
    roi_margin: int | tuple[int, int, int] | list[int] | None = None,
    roi_reference: str | None = None,
    normalize_intensity: bool = True,
    intensity_lower: float | None = None,
    intensity_upper: float | None = None,
    intensity_clip: bool | None = None,
    transforms=None,
):
    """
    Factory for loading user-provided registration datasets from JSON.

    Expected JSON structure:
    {
      "train": [{"moving": "...", "fixed": "...", ...}],
      "val":   [{"moving": "...", "fixed": "...", ...}]
    }

    Required per case:
      - moving: path to moving image
      - fixed: path to fixed image

    Optional per case:
      - moving_label, fixed_label
      - moving_mask, fixed_mask
      - moving_landmarks, fixed_landmarks
      - any metadata fields (kept as-is)
    """
    split_name = split.lower()
    if split_name not in {"train", "val"}:
        raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'val'.")

    cases = _load_manifest_cases(json_file=json_file, split=split_name)

    if transforms is None:
        del roi_from_labels, roi_margin, roi_reference, intensity_lower, intensity_upper, intensity_clip
        preprocess_style_norm = str(preprocess_style).lower()
        if preprocess_style_norm != "multigradicon":
            raise ValueError(
                "custom_dataset now uses only multigradicon-style preprocessing. "
                f"Set preprocess_style: multigradicon (got '{preprocess_style}')."
            )
        spacing_tuple = None if spacing is None else tuple(float(x) for x in spacing)
        size_tuple = None if image_size is None else tuple(int(x) for x in image_size)
        ct_window_tuple = tuple(float(x) for x in ct_window)
        quantile_tuple = tuple(float(x) for x in quantile_range)
        if len(ct_window_tuple) != 2:
            raise ValueError(f"ct_window must have 2 values. Got {ct_window}.")
        if len(quantile_tuple) != 2:
            raise ValueError(f"quantile_range must have 2 values. Got {quantile_range}.")
        transforms = _build_transforms(
            cases=cases,
            image_size=size_tuple,
            orientation=orientation,
            spacing=spacing_tuple,
            ct_window=ct_window_tuple,
            quantile_range=quantile_tuple,
            default_is_ct=bool(default_is_ct),
            normalize_intensity=normalize_intensity,
        )

    return Dataset(data=cases, transform=transforms)
