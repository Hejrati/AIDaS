"""PyTorch model definition used only for training and ONNX export.

The installed AIDaS application does not import this module.  Keeping the
checkpoint loader separate from :mod:`aidas.ai.inference` prevents PyInstaller
from pulling the large PyTorch runtime into the Windows application.
"""

from __future__ import annotations

import os


DEFAULT_FEATURES = (32, 64, 128, 256)
NUM_BOUNDARIES = 6


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except (ImportError, OSError) as exc:  # pragma: no cover - depends on developer setup
        raise RuntimeError(
            "Exporting or training AI_ForAIDAS requires PyTorch. Install the "
            "developer tools with 'python -m pip install torch onnx'.\n\n"
            f"Original error: {exc}"
        ) from exc
    return torch, nn, F


def build_unet_class(torch, nn, F):
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
            for feature_count in self.features:
                self.encoders.append(DoubleConv(ch, feature_count))
                ch = feature_count

            self.bottleneck = DoubleConv(ch, ch * 2)
            ch *= 2

            self.upconvs = nn.ModuleList()
            self.decoders = nn.ModuleList()
            for feature_count in reversed(self.features):
                self.upconvs.append(nn.ConvTranspose2d(ch, feature_count, kernel_size=2, stride=2))
                self.decoders.append(DoubleConv(feature_count * 2, feature_count))
                ch = feature_count

            self.head = nn.Conv2d(ch, num_bnd, kernel_size=1)

        def forward(self, x):
            height, width = x.shape[-2:]
            factor = 2 ** len(self.features)
            x = F.pad(
                x,
                (0, (-width) % factor, 0, (-height) % factor),
                mode="reflect",
            )

            skips = []
            tensor = x
            for encoder in self.encoders:
                tensor = encoder(tensor)
                skips.append(tensor)
                tensor = self.pool(tensor)

            tensor = self.bottleneck(tensor)

            for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
                tensor = upconv(tensor)
                if tensor.shape[-2:] != skip.shape[-2:]:
                    tensor = F.interpolate(tensor, size=skip.shape[-2:])
                tensor = decoder(torch.cat((tensor, skip), dim=1))

            return self.head(tensor)[..., :height, :width]

    return UNet


def state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(str(key).startswith("module.") for key in state_dict):
        return state_dict
    return {
        str(key)[7:] if str(key).startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_boundary_model(model_path: str, *, device="cpu"):
    """Load the training checkpoint and return an evaluated U-Net."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"AI_ForAIDAS boundary checkpoint not found: {model_path}")

    torch, nn, F = import_torch()
    torch_device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=torch_device)
    num_boundaries = (
        int(checkpoint.get("num_bnd", NUM_BOUNDARIES))
        if isinstance(checkpoint, dict)
        else NUM_BOUNDARIES
    )
    features = (
        tuple(checkpoint.get("features", DEFAULT_FEATURES))
        if isinstance(checkpoint, dict)
        else DEFAULT_FEATURES
    )

    unet_class = build_unet_class(torch, nn, F)
    model = unet_class(num_bnd=num_boundaries, features=features)
    model.load_state_dict(strip_module_prefix(state_dict_from_checkpoint(checkpoint)))
    model.to(torch_device).eval()
    return model, checkpoint, torch
