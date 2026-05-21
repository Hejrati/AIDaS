"""
OCT Retinal Boundary Segmentation -- .img via TIFF round-trip
Usage: python segment_img_via_tiff.py test/Light.img test/Dark.img [--model model.pth]

Pipeline per slice:
  1. Extract int16 slice from Analyze .img  ->  uint16 TIFF  (lossless offset)
  2. Run the standard segment.py pipeline on that TIFF
  3. Read the segmented uint8 TIFF output
  4. Stack all slices and save as uint8 Analyze .img / .hdr
"""

import os
import sys
import struct
import argparse
import tempfile
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import UNet, load_image, soft_argmax_y, NUM_BOUNDARIES
from train_vline import VLineNet
from segment import predict, save_overlay, BOUNDARY_STYLES, DEFAULT_MODEL

DEFAULT_VLINE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vline_model.pth')


# ── Analyze I/O ────────────────────────────────────────────────────────────────

def load_analyze(base):
    """Return (array [nz,H,W], endian char).  Handles both LE and BE headers."""
    with open(base + '.hdr', 'rb') as f:
        hdr = f.read()
    e = '<' if struct.unpack_from('<i', hdr, 0)[0] == 348 else '>'
    dims  = struct.unpack_from(f'{e}8h', hdr, 40)
    dt_id = struct.unpack_from(f'{e}h',  hdr, 70)[0]
    nx, ny, nz = dims[1], dims[2], dims[3]
    dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32}
    dt  = np.dtype(dtype_map[dt_id]).newbyteorder(e)
    raw = np.frombuffer(open(base + '.img', 'rb').read(), dtype=dt)
    return raw.reshape(nz, ny, nx), e


def write_analyze_header(path, nx, ny, nz):
    """Write a big-endian uint8 Analyze 7.5 header."""
    e   = '>'
    hdr = bytearray(348)
    struct.pack_into(f'{e}i', hdr,  0, 348)
    hdr[38] = ord('r')
    struct.pack_into(f'{e}8h', hdr, 40, 4, nx, ny, nz, 1, 0, 0, 0)
    struct.pack_into(f'{e}h',  hdr, 70, 2)           # datatype = uint8
    struct.pack_into(f'{e}h',  hdr, 72, 8)           # bitpix = 8
    struct.pack_into(f'{e}8f', hdr, 76, 1.0, 1.0, 1.0, 1.0, 0., 0., 0., 0.)
    struct.pack_into(f'{e}f',  hdr, 124, 255.0)
    struct.pack_into(f'{e}f',  hdr, 128,   0.0)
    struct.pack_into(f'{e}i',  hdr, 140, 255)
    struct.pack_into(f'{e}i',  hdr, 144,   0)
    with open(path, 'wb') as f:
        f.write(hdr)


# ── Slice conversion ───────────────────────────────────────────────────────────

def slice_to_tiff(slice_arr, path):
    """
    Save a raw int16/int32 slice as a uint16 TIFF.
    Offsets values so the minimum becomes 0 (lossless up to relative values,
    which is all that matters since load_image() uses min-max normalization).
    """
    s      = slice_arr.astype(np.int32)
    offset = int(-s.min()) if s.min() < 0 else 0
    u16    = (s + offset).astype(np.uint16)
    Image.fromarray(u16).save(path)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def predict_vline(img_norm, vline_model, device):
    """Return the predicted foveal x-column, or None if no vline model loaded."""
    if vline_model is None:
        return None
    W = img_norm.shape[1]
    x = torch.from_numpy(img_norm[np.newaxis, np.newaxis]).to(device)
    with torch.no_grad():
        x_norm = vline_model(x).item()
    return int(round(x_norm * W))


def process_file(img_path, model, device, vline_model=None, vline_col=None, out_suffix='_segmented', no_csv=False):
    base       = os.path.splitext(img_path)[0]
    arr, _     = load_analyze(base)
    nz, H, W   = arr.shape
    print(f"\nInput : {os.path.basename(img_path)}  {nz} slice(s)  H={H}  W={W}")

    csv_path   = base + '_pred_boundaries.csv'
    out_slices = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for sl in range(nz):
            # 1. int16 slice -> uint16 TIFF
            tiff_in  = os.path.join(tmpdir, f'slice_{sl}.tiff')
            tiff_out = os.path.join(tmpdir, f'slice_{sl}_segmented.tiff')
            slice_to_tiff(arr[sl], tiff_in)

            # 2. Inference (exactly as segment.py does for a TIFF)
            img_raw  = np.array(Image.open(tiff_in))   # uint16, for overlay
            img_norm = load_image(tiff_in)              # float32 [0,1], for model
            pred_y   = predict(img_norm, model, device)

            # 3. Draw overlay -> segmented TIFF
            save_overlay(img_raw, pred_y, tiff_out)

            # 4. Read back; vertical line drawn separately if vline_col provided
            seg = np.array(Image.open(tiff_out))
            out_slices.append(seg)

            print(f"  Slice {sl}: y=[{pred_y.min():.1f}, {pred_y.max():.1f}]")

            # Optional CSV
            if not no_csv:
                mode = 'w' if sl == 0 else 'a'
                with open(csv_path, mode) as f:
                    f.write(f'# slice {sl}\n')
                    for b, row in enumerate(pred_y):
                        f.write(f'{b+1}\t' + ','.join(f'{v:.1f}' for v in row) + '\n')

    # 5. Draw vertical line if column provided
    if vline_col is not None:
        col = int(np.clip(vline_col, 0, W - 1))
        for sl_arr in out_slices:
            sl_arr[:, col] = 243
        print(f"  Vertical line drawn at column {col}")

    # 6. Stack and save as Analyze uint8
    out_arr  = np.stack(out_slices, axis=0)   # (nz, H, W)
    out_base = base + out_suffix
    with open(out_base + '.img', 'wb') as f:
        f.write(out_arr.tobytes())
    write_analyze_header(out_base + '.hdr', W, H, nz)

    print(f"Output: {out_base}.img + .hdr  ({W}x{H}, {nz} slice(s), uint8)")
    if not no_csv:
        print(f"CSV   : {csv_path}")


def main():
    ap = argparse.ArgumentParser(
        description='Segment Analyze .img files via TIFF round-trip using the trained model')
    ap.add_argument('images', nargs='+', help='Input .img file path(s)')
    ap.add_argument('--model',       default=DEFAULT_MODEL)
    ap.add_argument('--vline-model', default=DEFAULT_VLINE_MODEL,
                    help='Vertical line model (default: vline_model.pth). '
                         'Pass "none" to use centre fallback.')
    ap.add_argument('--no-csv', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}  -- run train.py first")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Boundary model
    ckpt  = torch.load(args.model, map_location=device)
    model = UNet(num_bnd=ckpt.get('num_bnd', NUM_BOUNDARIES),
                 features=tuple(ckpt.get('features', (32, 64, 128, 256))))
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    print(f"Boundary model : {args.model}")

    # Vertical line model (optional)
    vline_model = None
    vline_path  = args.vline_model
    if vline_path.lower() != 'none' and os.path.exists(vline_path):
        vl_ckpt     = torch.load(vline_path, map_location=device)
        vline_model = VLineNet()
        vline_model.load_state_dict(vl_ckpt['model_state'])
        vline_model.to(device).eval()
        print(f"VLine model    : {vline_path}")
    else:
        print(f"VLine model    : not found — using centre fallback  (run train_vline.py to train)")

    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Skipping (not found): {img_path}")
            continue
        process_file(img_path, model, device, vline_model=vline_model, no_csv=args.no_csv)

    print("\nDone.")


if __name__ == '__main__':
    main()
