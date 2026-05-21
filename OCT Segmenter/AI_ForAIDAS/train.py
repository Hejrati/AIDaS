"""
OCT Retinal Boundary Segmentation — Training
Usage: python train.py [--epochs N] [--lr LR] [--save-path model.pth]

Ground truth is extracted from the MARKED TIFFs (which encode 6 retinal
boundaries as specific intensity values 249-255).  The U-Net is trained
on the paired regular TIFFs at native resolution — images are never resized.
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from scipy.ndimage import gaussian_filter1d

DATA_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test", "gathered_data")
NUM_BOUNDARIES = 6
# Pixel values used to mark each boundary in the MARKED files (consistent across all images)
BOUNDARY_VALS  = [249, 250, 252, 253, 254, 255]
BOUNDARY_NAMES = ['ILM', 'NFL/GCL', 'IPL/INL', 'INL/OPL', 'IS/OS', 'RPE']
SIGMA          = 1.5   # Gaussian soft-label width (pixels)
CROP_W         = None  # Width crop during training (None = full width; set to e.g. 512 for CPU)


# ── Analyze .img loader ────────────────────────────────────────────────────────

import struct as _struct

def _read_analyze(base):
    """Return (array [nz,H,W], endian) for an Analyze 7.5 .img/.hdr pair."""
    with open(base + '.hdr', 'rb') as f:
        hdr = f.read()
    e = '<' if _struct.unpack_from('<i', hdr, 0)[0] == 348 else '>'
    dims  = _struct.unpack_from(f'{e}8h', hdr, 40)
    dt_id = _struct.unpack_from(f'{e}h',  hdr, 70)[0]
    nx, ny, nz = dims[1], dims[2], dims[3]
    dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32}
    dt  = np.dtype(dtype_map[dt_id]).newbyteorder(e)
    raw = np.frombuffer(open(base + '.img', 'rb').read(), dtype=dt)
    return raw.reshape(nz, ny, nx), e


# ── Ground-truth extraction ────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load any TIFF as float32 grayscale, normalised to [0, 1]."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def extract_boundaries(marked: np.ndarray) -> np.ndarray:
    """
    Extract 6 boundary y-coordinates from a MARKED TIFF.

    Each boundary is encoded as a specific pixel value (BOUNDARY_VALS).
    Returns shape (6, W) float32 — y-position per column for each boundary.
    """
    H, W  = marked.shape
    y_idx = np.arange(H, dtype=np.float32).reshape(-1, 1)   # (H, 1)

    out = np.zeros((NUM_BOUNDARIES, W), dtype=np.float32)
    for b, val in enumerate(BOUNDARY_VALS):
        mask      = (marked == val).astype(np.float32)       # (H, W)
        col_count = mask.sum(axis=0)                         # (W,)
        col_ysum  = (mask * y_idx).sum(axis=0)               # (W,)

        valid = col_count > 0
        out[b, valid] = col_ysum[valid] / col_count[valid]

        # Interpolate across any missing columns (rare edge case)
        if not valid.all() and valid.any():
            xs = np.where(valid)[0]
            out[b] = np.interp(np.arange(W), xs, out[b, valid])

    return out


# ── Dataset ────────────────────────────────────────────────────────────────────

class OCTDataset(Dataset):
    def __init__(self, root_dir: str = DATA_DIR):
        self.samples = []  # list of (raw_base, marked_base, slice_idx)

        # .img mode: scan_*.img / scan_*_MARKED.img pairs (each file has multiple slices)
        img_files = sorted([
            f for f in glob.glob(os.path.join(root_dir, 'scan_*.img'))
            if 'MARKED' not in f
        ])
        if img_files:
            for raw_path in img_files:
                base         = os.path.splitext(raw_path)[0]
                marked_path  = base + '_MARKED.img'
                if not os.path.exists(marked_path):
                    continue
                # Use only slice 0 — slices within a file are identical B-scans
                self.samples.append((base, base + '_MARKED', 0))
            print(f"Dataset (.img): {len(self.samples)} unique images from {len(img_files)} paired files")
        else:
            # Fallback: original TIFF mode
            for reg_path in sorted(glob.glob(os.path.join(root_dir, 'Light_[0-9]*.tiff'))):
                name        = os.path.splitext(os.path.basename(reg_path))[0]
                marked_path = os.path.join(root_dir, 'Light_MARKED_' + name.split('_', 1)[1] + '.tiff')
                if os.path.exists(marked_path):
                    self.samples.append((os.path.splitext(reg_path)[0],
                                         os.path.splitext(marked_path)[0], None))
            print(f"Dataset (.tiff): {len(self.samples)} paired TIFFs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_base, marked_base, slice_idx = self.samples[idx]

        if slice_idx is not None:
            # .img mode
            raw_arr, _    = _read_analyze(raw_base)
            marked_arr, _ = _read_analyze(marked_base)
            sl  = raw_arr[slice_idx].astype(np.float32)
            lo, hi = sl.min(), sl.max()
            img    = (sl - lo) / (hi - lo + 1e-8)          # (H, W) float32
            marked = marked_arr[slice_idx]                   # (H, W) uint8
        else:
            # TIFF mode
            img    = load_image(raw_base + '.tiff')         # (H, W) float32
            marked = np.array(Image.open(marked_base + '.tiff'))  # (H, W) uint8

        H, W = img.shape
        bnds = extract_boundaries(marked)                    # (6, W) float32

        # Random width crop during training to keep computation manageable.
        # The model is fully convolutional, so inference runs at any width.
        if CROP_W is not None and W > CROP_W:
            x0   = np.random.randint(0, W - CROP_W)
            img  = img[:, x0:x0 + CROP_W]
            bnds = bnds[:, x0:x0 + CROP_W]
            W    = CROP_W

        # Build soft heatmap targets: (6, H, W) — vectorised, no per-column loop
        xs       = np.arange(W)
        ys       = np.clip(np.round(bnds).astype(int), 0, H - 1)  # (6, W)
        heatmaps = np.zeros((NUM_BOUNDARIES, H, W), dtype=np.float32)
        for b in range(NUM_BOUNDARIES):
            heatmaps[b, ys[b], xs] = 1.0
            heatmaps[b] = gaussian_filter1d(heatmaps[b], sigma=SIGMA, axis=0)
            col_sums    = heatmaps[b].sum(axis=0, keepdims=True).clip(min=1e-8)
            heatmaps[b] /= col_sums

        return {
            'image':    torch.from_numpy(img[np.newaxis]),    # (1, H, W)
            'heatmaps': torch.from_numpy(heatmaps),            # (6, H, W)
            'gt_y':     torch.from_numpy(bnds),                # (6, W)
        }


# ── Model ──────────────────────────────────────────────────────────────────────

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
    """
    Fully-convolutional U-Net for retinal boundary heatmap prediction.

    Input:  (B, 1,            H, W) — grayscale OCT at any native resolution
    Output: (B, NUM_BOUNDARIES, H, W) — per-boundary logits (softmax along H)
    """

    def __init__(self,
                 in_ch:    int   = 1,
                 num_bnd:  int   = NUM_BOUNDARIES,
                 features: tuple = (32, 64, 128, 256)):
        super().__init__()
        self.features = tuple(features)
        self.pool     = nn.MaxPool2d(2)

        self.encoders = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(ch, ch * 2)
        ch *= 2

        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))
            ch = f

        self.head = nn.Conv2d(ch, num_bnd, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w    = x.shape[-2:]
        factor  = 2 ** len(self.features)
        # Pad to next multiple of factor so pooling/upsampling sizes match
        x = F.pad(x, (0, (-w) % factor, 0, (-h) % factor), mode='reflect')

        skips, t = [], x
        for enc in self.encoders:
            t = enc(t)
            skips.append(t)
            t = self.pool(t)

        t = self.bottleneck(t)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            t = up(t)
            if t.shape[-2:] != skip.shape[-2:]:
                t = F.interpolate(t, size=skip.shape[-2:])
            t = dec(torch.cat([t, skip], dim=1))

        # Crop back to original (unpadded) size
        return self.head(t)[..., :h, :w]


# ── Loss & metrics ─────────────────────────────────────────────────────────────

def soft_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between per-column y-softmax and Gaussian soft targets."""
    return -(targets * F.log_softmax(logits, dim=2)).sum(dim=2).mean()


def soft_argmax_y(logits: torch.Tensor) -> torch.Tensor:
    """Differentiable argmax: weighted mean of y-indices. Returns (B, num_bnd, W)."""
    H     = logits.shape[2]
    y_idx = torch.arange(H, dtype=torch.float32, device=logits.device)
    probs = torch.softmax(logits, dim=2)
    return (probs * y_idx[None, None, :, None]).sum(dim=2)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = mae_sum = n = 0
    for batch in loader:
        img  = batch['image'].to(device)
        hmps = batch['heatmaps'].to(device)
        gt_y = batch['gt_y'].to(device)
        pred = model(img)
        loss_sum += soft_ce_loss(pred, hmps).item()
        mae_sum  += (soft_argmax_y(pred) - gt_y).abs().mean().item()
        n += 1
    return loss_sum / max(n, 1), mae_sum / max(n, 1)


# ── Training entry point ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Train OCT retinal segmentation U-Net')
    ap.add_argument('--epochs',    type=int,   default=60,
                    help='Number of training epochs (default: 60)')
    ap.add_argument('--lr',        type=float, default=1e-3,
                    help='Initial learning rate (default: 1e-3)')
    ap.add_argument('--save-path', type=str,   default='model.pth',
                    help='Where to save the best checkpoint (default: model.pth)')
    ap.add_argument('--data-dir',  type=str,   default=DATA_DIR,
                    help='Folder containing Light_*.tiff and Light_MARKED_*.tiff')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    dataset = OCTDataset(args.data_dir)
    if len(dataset) == 0:
        sys.exit("No paired (regular + MARKED) TIFFs found. Check --data-dir.")

    n_val   = max(2, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    print(f"Train : {n_train}   Val : {n_val}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0)

    model   = UNet().to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Model : {n_param:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-5)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_path)
    best_val  = float('inf')

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_mae = 0.0
        for batch in train_loader:
            img  = batch['image'].to(device)
            hmps = batch['heatmaps'].to(device)
            gt_y = batch['gt_y'].to(device)

            pred = model(img)
            loss = soft_ce_loss(pred, hmps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tr_mae += (soft_argmax_y(pred) - gt_y).abs().mean().item()
            tr_loss += loss.item()

        tr_loss /= len(train_loader)
        tr_mae  /= len(train_loader)

        # ── Validate ────────────────────────────────────────────────────────
        vl_loss, vl_mae = evaluate(model, val_loader, device)
        scheduler.step(vl_loss)

        print(f"Epoch {epoch:3d}/{args.epochs}  |  "
              f"train  loss={tr_loss:.4f}  mae={tr_mae:.2f}px  |  "
              f"val    loss={vl_loss:.4f}  mae={vl_mae:.2f}px")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({
                'model_state': model.state_dict(),
                'num_bnd':     NUM_BOUNDARIES,
                'features':    list(model.features),
            }, save_path)
            print(f"  -> checkpoint saved to {save_path}")

    print(f"\nTraining complete.  Best val loss : {best_val:.4f}")


if __name__ == '__main__':
    main()
