"""
OCT Retinal Boundary Segmentation — Inference
Usage: python segment.py <image.tiff> [--model model.pth] [--output out.png]

Loads a trained model and draws the 6 predicted retinal boundaries on the image.
Also saves a CSV of predicted y-coordinates in the same format as the training data.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import UNet, load_image, soft_argmax_y, NUM_BOUNDARIES, BOUNDARY_NAMES

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pth')
# Exact grayscale values and line widths used in the MARKED TIFFs
BOUNDARY_STYLES = [
    (255, 5),   # ILM    (thick)
    (254, 5),   # NFL/GCL (thick)
    (253, 1),   # IPL/INL
    (252, 1),   # INL/OPL
    (250, 1),   # IS/OS
    (249, 1),   # RPE
]


def predict(img_norm: np.ndarray, model, device) -> np.ndarray:
    """Run inference; returns (num_bnd, W) boundary y-coordinates."""
    x = torch.from_numpy(img_norm[np.newaxis, np.newaxis]).to(device)
    with torch.no_grad():
        pred_y = soft_argmax_y(model(x))[0].cpu().numpy()  # (num_bnd, W)
    # Enforce physical ordering: boundaries never cross
    return np.sort(pred_y, axis=0)


def save_overlay(img_raw: np.ndarray, pred_y: np.ndarray, output_path: str, title: str = ''):
    """Draw boundary lines on the image and save at exact original pixel dimensions."""
    H, W = img_raw.shape

    # Match the MARKED TIFF scaling: pixel / 137 + 5.7, clipped to [0, 255]
    gray_u8 = (img_raw / 137.0 + 5.7).clip(0, 255).astype(np.uint8)
    out = Image.fromarray(gray_u8, mode='L')
    draw = ImageDraw.Draw(out)

    for b in range(pred_y.shape[0]):
        val, width = BOUNDARY_STYLES[b % len(BOUNDARY_STYLES)]
        pts = [(x, int(round(pred_y[b, x]))) for x in range(W)]
        draw.line(pts, fill=val, width=width)

    out.save(output_path)
    print(f"Overlay  -> {output_path}  ({W}x{H})")


def save_csv(pred_y: np.ndarray, output_path: str):
    with open(output_path, 'w') as f:
        for b, row in enumerate(pred_y):
            f.write(f'{b+1}\t' + ','.join(f'{v:.1f}' for v in row) + '\n')
    print(f"CSV      -> {output_path}")


def main():
    ap = argparse.ArgumentParser(description='Segment retinal layers in an OCT TIFF')
    ap.add_argument('image',      help='Input TIFF path')
    ap.add_argument('--model',    default=DEFAULT_MODEL)
    ap.add_argument('--output',   default=None, help='Output PNG path')
    ap.add_argument('--no-csv',   action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"Image not found: {args.image}")
    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}  —  run train.py first")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt  = torch.load(args.model, map_location=device)
    model = UNet(num_bnd=ckpt.get('num_bnd', NUM_BOUNDARIES),
                 features=tuple(ckpt.get('features', (16, 32, 64, 128))))
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    img_raw  = np.array(Image.open(args.image))   # raw 16-bit for display
    img_norm = load_image(args.image)              # normalised for model input
    print(f"Image : {os.path.basename(args.image)}  {img_norm.shape[0]}x{img_norm.shape[1]}")

    pred_y = predict(img_norm, model, device)

    stem    = os.path.splitext(args.image)[0]
    outfile = args.output or stem + '_segmented.tiff'
    save_overlay(img_raw, pred_y, outfile, title=os.path.basename(args.image))

    if not args.no_csv:
        save_csv(pred_y, stem + '_pred_boundaries.csv')


if __name__ == '__main__':
    main()
