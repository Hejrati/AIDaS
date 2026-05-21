"""
OCT Retinal Boundary Segmentation -- Analyze 7.5 (.img/.hdr) Inference
Usage: python segment_analyze.py test/Light.img test/Dark.img [--model model.pth]

Loads a trained model and segments each 2D slice in an Analyze .img file.
Outputs a uint8 Analyze .img/.hdr with 6 retinal boundaries overlaid,
matching the style of the human-annotated MARKED files.
"""

import os
import sys
import argparse
import struct
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import UNet, soft_argmax_y, NUM_BOUNDARIES

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pth')

BOUNDARY_STYLES = [
    (255, 5),   # ILM    (thick)
    (254, 5),   # NFL/GCL (thick)
    (253, 1),   # IPL/INL
    (252, 1),   # INL/OPL
    (250, 1),   # IS/OS
    (249, 1),   # RPE
]


def load_analyze(base):
    """Load an Analyze 7.5 file pair. Returns (array [nz,H,W], endian char, raw header bytes)."""
    with open(base + '.hdr', 'rb') as f:
        hdr_bytes = f.read()
    e = '<' if struct.unpack_from('<i', hdr_bytes, 0)[0] == 348 else '>'
    dims  = struct.unpack_from(f'{e}8h', hdr_bytes, 40)
    dt_id = struct.unpack_from(f'{e}h',  hdr_bytes, 70)[0]
    nx, ny, nz = dims[1], dims[2], dims[3]
    dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32}
    dt  = np.dtype(dtype_map[dt_id]).newbyteorder(e)
    raw = np.frombuffer(open(base + '.img', 'rb').read(), dtype=dt)
    return raw.reshape(nz, ny, nx), e, hdr_bytes


def write_analyze_header(path, nx, ny, nz):
    """Write a minimal big-endian uint8 Analyze 7.5 header."""
    e = '>'
    hdr = bytearray(348)
    struct.pack_into(f'{e}i', hdr,  0, 348)                              # sizeof_hdr
    hdr[38] = ord('r')                                                    # regular
    struct.pack_into(f'{e}8h', hdr, 40, 4, nx, ny, nz, 1, 0, 0, 0)     # dim
    struct.pack_into(f'{e}h',  hdr, 70, 2)                               # datatype = uint8
    struct.pack_into(f'{e}h',  hdr, 72, 8)                               # bitpix
    struct.pack_into(f'{e}8f', hdr, 76, 1.0, 1.0, 1.0, 1.0, 0., 0., 0., 0.)  # pixdim
    struct.pack_into(f'{e}f',  hdr, 124, 255.0)                          # cal_max
    struct.pack_into(f'{e}f',  hdr, 128,   0.0)                          # cal_min
    struct.pack_into(f'{e}i',  hdr, 140, 255)                            # glmax
    struct.pack_into(f'{e}i',  hdr, 144,   0)                            # glmin
    with open(path, 'wb') as f:
        f.write(hdr)


def slice_to_uint8(raw_slice):
    """Scale a raw slice to uint8 [0, 243], keeping 249-255 free for boundary markers."""
    s = raw_slice.astype(np.float32)
    lo, hi = s.min(), s.max()
    return ((s - lo) / (hi - lo + 1e-8) * 243).clip(0, 243).astype(np.uint8)


def normalize_for_model(raw_slice):
    """Normalize a raw slice to float32 [0, 1] for model input."""
    s = raw_slice.astype(np.float32)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-8)


def predict(img_norm, model, device):
    """Run inference; returns (num_bnd, W) sorted boundary y-coordinates."""
    x = torch.from_numpy(img_norm[np.newaxis, np.newaxis]).to(device)
    with torch.no_grad():
        pred_y = soft_argmax_y(model(x))[0].cpu().numpy()
    return np.sort(pred_y, axis=0)


def overlay_boundaries(uint8_slice, pred_y):
    """Draw boundary lines on a uint8 HxW array; returns uint8 array."""
    img  = Image.fromarray(uint8_slice, mode='L')
    draw = ImageDraw.Draw(img)
    W    = uint8_slice.shape[1]
    for b in range(pred_y.shape[0]):
        val, width = BOUNDARY_STYLES[b % len(BOUNDARY_STYLES)]
        pts = [(x, int(round(pred_y[b, x]))) for x in range(W)]
        draw.line(pts, fill=val, width=width)
    return np.array(img)


def save_csv(pred_y, csv_path, slice_idx, mode='a'):
    with open(csv_path, mode) as f:
        f.write(f'# slice {slice_idx}\n')
        for b, row in enumerate(pred_y):
            f.write(f'{b+1}\t' + ','.join(f'{v:.1f}' for v in row) + '\n')


def process_file(img_path, model, device, no_csv=False):
    base           = os.path.splitext(img_path)[0]
    arr, _, _      = load_analyze(base)
    nz, H, W       = arr.shape
    print(f"\nInput : {os.path.basename(img_path)}  {nz} slice(s)  H={H} W={W}")

    csv_path   = base + '_pred_boundaries.csv'
    out_slices = []

    for sl in range(nz):
        raw_sl   = arr[sl]
        img_norm = normalize_for_model(raw_sl)
        pred_y   = predict(img_norm, model, device)
        uint8_sl = slice_to_uint8(raw_sl)
        out_sl   = overlay_boundaries(uint8_sl, pred_y)
        out_slices.append(out_sl)
        print(f"  Slice {sl}: boundaries y=[{pred_y.min():.1f}, {pred_y.max():.1f}]")
        if not no_csv:
            save_csv(pred_y, csv_path, sl, mode='w' if sl == 0 else 'a')

    out_arr  = np.stack(out_slices, axis=0)   # (nz, H, W) uint8
    out_base = base + '_segmented'
    with open(out_base + '.img', 'wb') as f:
        f.write(out_arr.tobytes())
    write_analyze_header(out_base + '.hdr', W, H, nz)

    print(f"Output: {out_base}.img + .hdr  ({W}x{H}, {nz} slice(s), uint8)")
    if not no_csv:
        print(f"CSV   : {csv_path}")


def main():
    ap = argparse.ArgumentParser(description='Segment retinal layers in Analyze .img/.hdr files')
    ap.add_argument('images', nargs='+', help='Input .img file path(s)')
    ap.add_argument('--model',  default=DEFAULT_MODEL)
    ap.add_argument('--no-csv', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}  -- run train.py first")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt  = torch.load(args.model, map_location=device)
    model = UNet(num_bnd=ckpt.get('num_bnd', NUM_BOUNDARIES),
                 features=tuple(ckpt.get('features', (32, 64, 128, 256))))
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    print(f"Model : {args.model}")

    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Skipping (not found): {img_path}")
            continue
        process_file(img_path, model, device, no_csv=args.no_csv)

    print("\nDone.")


if __name__ == '__main__':
    main()
