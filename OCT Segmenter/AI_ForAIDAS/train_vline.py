"""
Vertical Line (Foveal Centre) Predictor -- Training
Usage: python train_vline.py [--epochs N] [--save-path vline_model.pth]

Trains a small CNN that takes a normalised OCT image and predicts the
x-coordinate of the vertical reference line (value 243 in MARKED files),
expressed as a fraction of image width so it works at any resolution.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import DATA_DIR, _read_analyze

VLINE_VAL   = 243          # pixel value used for the vertical line in MARKED files
MIN_COL_COVERAGE = 0.5     # fraction of rows that must be 243 to count as the vline


# ── Ground-truth extraction ────────────────────────────────────────────────────

def extract_vline_x(marked_slice: np.ndarray) -> float | None:
    """
    Return the vertical line x-coordinate as a fraction of image width,
    or None if no vertical line is found.
    """
    H, W  = marked_slice.shape
    col_counts = (marked_slice == VLINE_VAL).sum(axis=0)
    cols = np.where(col_counts > H * MIN_COL_COVERAGE)[0]
    if len(cols) == 0:
        return None
    return float(cols[0]) / W   # normalised [0, 1]


# ── Dataset ────────────────────────────────────────────────────────────────────

class VLineDataset(Dataset):
    def __init__(self, root_dir: str = DATA_DIR):
        self.samples = []   # (raw_base, marked_base, slice_idx, x_norm)

        img_files = sorted([
            f for f in glob.glob(os.path.join(root_dir, 'scan_*.img'))
            if 'MARKED' not in f
        ])
        skipped = 0
        for raw_path in img_files:
            base        = os.path.splitext(raw_path)[0]
            marked_path = base + '_MARKED.img'
            if not os.path.exists(marked_path):
                continue
            raw_arr,    _ = _read_analyze(base)
            marked_arr, _ = _read_analyze(base + '_MARKED')
            # Use only slice 0 — slices within a file are identical B-scans
            x_norm = extract_vline_x(marked_arr[0])
            if x_norm is None:
                skipped += 1
                continue
            self.samples.append((base, base + '_MARKED', 0, x_norm))

        print(f"VLineDataset: {len(self.samples)} unique images  ({skipped} skipped — no vline found)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_base, _, slice_idx, x_norm = self.samples[idx]
        raw_arr, _ = _read_analyze(raw_base)
        sl = raw_arr[slice_idx].astype(np.float32)
        lo, hi = sl.min(), sl.max()
        img = (sl - lo) / (hi - lo + 1e-8)   # (H, W) float32 [0, 1]
        return {
            'image':  torch.from_numpy(img[np.newaxis]),   # (1, H, W)
            'x_norm': torch.tensor(x_norm, dtype=torch.float32),
        }


# ── Model ──────────────────────────────────────────────────────────────────────

class VLineNet(nn.Module):
    """
    Lightweight CNN: predicts foveal centre x as a fraction of image width.
    Input:  (B, 1, H, W)
    Output: (B, 1)  in [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  16, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(16),  nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))   # (B, 1)


# ── Training ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae_sum = n = 0
    for batch in loader:
        img    = batch['image'].to(device)
        x_norm = batch['x_norm'].to(device)
        pred   = model(img).squeeze(1)
        # MAE in pixels: need image width — approximate from input tensor
        W      = img.shape[-1]
        mae_sum += (pred - x_norm).abs().mean().item() * W
        n += 1
    return mae_sum / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description='Train foveal vertical-line predictor')
    ap.add_argument('--epochs',    type=int,   default=60)
    ap.add_argument('--lr',        type=float, default=1e-3)
    ap.add_argument('--save-path', type=str,   default='vline_model.pth')
    ap.add_argument('--data-dir',  type=str,   default=DATA_DIR)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    dataset = VLineDataset(args.data_dir)
    if len(dataset) == 0:
        sys.exit("No samples with a vertical line found.")

    n_val   = max(2, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    print(f"Train : {n_train}   Val : {n_val}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0)

    model   = VLineNet().to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Model : {n_param:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-5)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_path)
    best_mae  = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_mae = 0.0
        for batch in train_loader:
            img    = batch['image'].to(device)
            x_norm = batch['x_norm'].to(device)
            pred   = model(img).squeeze(1)
            loss   = F.mse_loss(pred, x_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                W = img.shape[-1]
                tr_mae  += (pred - x_norm).abs().mean().item() * W
            tr_loss += loss.item()

        tr_loss /= len(train_loader)
        tr_mae  /= len(train_loader)
        vl_mae   = evaluate(model, val_loader, device)
        scheduler.step(vl_mae)

        print(f"Epoch {epoch:3d}/{args.epochs}  |  "
              f"train  loss={tr_loss:.5f}  mae={tr_mae:.1f}px  |  "
              f"val    mae={vl_mae:.1f}px")

        if vl_mae < best_mae:
            best_mae = vl_mae
            torch.save({'model_state': model.state_dict()}, save_path)
            print(f"  -> checkpoint saved ({save_path})  best val mae={best_mae:.1f}px")

    print(f"\nTraining complete.  Best val MAE : {best_mae:.1f}px")


if __name__ == '__main__':
    main()
