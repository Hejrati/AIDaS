"""Inference adapter for the PyTorch AI_ForAIDAS OCT models.

The Step 2 UI imports this module only when the AI_ForAIDAS backend is used, so
PyTorch remains an optional runtime dependency for the rest of AIDaS.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import numpy as np


DEFAULT_FEATURES = (32, 64, 128, 256)
NUM_BOUNDARIES = 6


@dataclass(frozen=True)
class PredictionResult:
    boundaries: np.ndarray
    fovea_x: Optional[int]
    device: str
    boundary_model_path: str
    vline_model_path: Optional[str]


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "AI_ForAIDAS requires PyTorch. Install torch in the AIDaS Python "
            "environment before using this AI version."
        ) from exc
    return torch, nn, F


def _normalize_for_model(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"AI_ForAIDAS expects a 2-D grayscale image, got shape {arr.shape}.")
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
    arr = arr.astype(np.float32, copy=False)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.ascontiguousarray((arr - lo) / (hi - lo + 1e-8), dtype=np.float32)


def _build_unet_class(torch, nn, F):
    class DoubleConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self, in_ch: int = 1, num_bnd: int = NUM_BOUNDARIES, features=DEFAULT_FEATURES):
            super().__init__()
            self.features = tuple(features)
            self.pool = nn.MaxPool2d(2)

            self.encoders = nn.ModuleList()
            ch = in_ch
            for f in self.features:
                self.encoders.append(DoubleConv(ch, f))
                ch = f

            self.bottleneck = DoubleConv(ch, ch * 2)
            ch *= 2

            self.upconvs = nn.ModuleList()
            self.decoders = nn.ModuleList()
            for f in reversed(self.features):
                self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
                self.decoders.append(DoubleConv(f * 2, f))
                ch = f

            self.head = nn.Conv2d(ch, num_bnd, kernel_size=1)

        def forward(self, x):
            h, w = x.shape[-2:]
            factor = 2 ** len(self.features)
            x = F.pad(x, (0, (-w) % factor, 0, (-h) % factor), mode="reflect")

            skips = []
            t = x
            for enc in self.encoders:
                t = enc(t)
                skips.append(t)
                t = self.pool(t)

            t = self.bottleneck(t)

            for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
                t = up(t)
                if t.shape[-2:] != skip.shape[-2:]:
                    t = F.interpolate(t, size=skip.shape[-2:])
                t = dec(torch.cat((t, skip), dim=1))

            return self.head(t)[..., :h, :w]

    return UNet


def _build_vline_class(nn):
    class VLineNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.head(self.encoder(x))

    return VLineNet


def _state_dict_from_checkpoint(ckpt):
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(str(key).startswith("module.") for key in state_dict):
        return state_dict
    return {
        str(key)[7:] if str(key).startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _resolve_device(torch, requested: str):
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("AI_ForAIDAS device is set to CUDA, but CUDA is not available.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported AI_ForAIDAS device: {requested}")
    return torch.device(requested)


def _load_boundary_model(model_path: str, device, torch, nn, F):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"AI_ForAIDAS boundary model not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    num_bnd = int(ckpt.get("num_bnd", NUM_BOUNDARIES)) if isinstance(ckpt, dict) else NUM_BOUNDARIES
    features = tuple(ckpt.get("features", DEFAULT_FEATURES)) if isinstance(ckpt, dict) else DEFAULT_FEATURES

    UNet = _build_unet_class(torch, nn, F)
    model = UNet(num_bnd=num_bnd, features=features)
    model.load_state_dict(_strip_module_prefix(_state_dict_from_checkpoint(ckpt)))
    model.to(device).eval()
    return model


def _load_vline_model(model_path: str, device, torch, nn):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"AI_ForAIDAS vline model not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    VLineNet = _build_vline_class(nn)
    model = VLineNet()
    model.load_state_dict(_strip_module_prefix(_state_dict_from_checkpoint(ckpt)))
    model.to(device).eval()
    return model


def _soft_argmax_y(logits, torch):
    height = logits.shape[2]
    y_idx = torch.arange(height, dtype=torch.float32, device=logits.device)
    probs = torch.softmax(logits, dim=2)
    return (probs * y_idx[None, None, :, None]).sum(dim=2)


def predict_boundaries_and_fovea(
    image: np.ndarray,
    *,
    boundary_model_path: str,
    vline_model_path: Optional[str] = None,
    predict_fovea: bool = True,
    device_name: str = "auto",
) -> PredictionResult:
    """Run AI_ForAIDAS inference on a 2-D OCT image."""
    torch, nn, F = _import_torch()

    device = _resolve_device(torch, device_name)
    image_norm = _normalize_for_model(image)
    height, width = image_norm.shape

    boundary_model = _load_boundary_model(boundary_model_path, device, torch, nn, F)
    x = torch.from_numpy(image_norm[np.newaxis, np.newaxis]).to(device)

    with torch.inference_mode():
        pred_y = _soft_argmax_y(boundary_model(x), torch)[0].detach().cpu().numpy()

    if pred_y.shape[0] < NUM_BOUNDARIES:
        raise RuntimeError(
            f"AI_ForAIDAS returned {pred_y.shape[0]} boundary rows; expected {NUM_BOUNDARIES}."
        )

    boundaries = np.sort(pred_y[:NUM_BOUNDARIES], axis=0).astype(np.float32, copy=False)
    boundaries = np.clip(boundaries, 0, height - 1)

    fovea_x = None
    resolved_vline_path = None
    if predict_fovea and vline_model_path:
        resolved_vline_path = vline_model_path
        vline_model = _load_vline_model(vline_model_path, device, torch, nn)
        with torch.inference_mode():
            x_norm = float(vline_model(x).item())
        fovea_x = int(np.clip(round(x_norm * width), 0, width - 1))

    return PredictionResult(
        boundaries=boundaries,
        fovea_x=fovea_x,
        device=str(device),
        boundary_model_path=boundary_model_path,
        vline_model_path=resolved_vline_path,
    )
