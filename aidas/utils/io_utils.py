"""I/O utilities for OCT image processing.

Provides:
    - read_raw_oct()    — read a raw binary OCT file (like ImageJ "Raw…" import)
    - read_tiff()       — read TIFF/image stacks (including ImageJ big-endian 16-bit)
    - scale_image()     — pixel-replicate scaling (no interpolation)
    - write_analyze()   — write Analyze 7.5 format (.hdr + .img)
    - read_analyze()    — read  Analyze 7.5 format
    - save_tiff()       — save 8/16-bit TIFF via Pillow
"""

import os
import struct

import numpy as np
from PIL import Image


# ════════════════════════════════════════════════════════════════════════════
#  Raw OCT reader
# ════════════════════════════════════════════════════════════════════════════

def read_raw_oct(filepath, width=768, height=1200, offset=1050,
                 bit_depth=16, little_endian=True):
    """Read a raw binary OCT image — mirrors ImageJ's *Raw…* import.

    Parameters
    ----------
    filepath : str
        Path to the binary file.
    width, height : int
        Image dimensions in pixels.
    offset : int
        Header bytes to skip before pixel data.
    bit_depth : int
        8 or 16.
    little_endian : bool
        Byte-order flag (ignored for 8-bit).

    Returns
    -------
    np.ndarray  — shape (height, width), dtype uint8 or uint16.
    """
    if bit_depth == 16:
        dtype = "<u2" if little_endian else ">u2"
        bpp = 2
    elif bit_depth == 8:
        dtype = "u1"
        bpp = 1
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    expected_bytes = width * height * bpp
    actual = os.path.getsize(filepath)
    available = max(0, actual - offset)
    bytes_to_read = min(expected_bytes, available)

    with open(filepath, "rb") as fh:
        fh.seek(offset)
        raw = fh.read(bytes_to_read)

    # If data is truncated, pad with zeros so missing pixels appear at the bottom.
    if len(raw) < expected_bytes:
        raw = raw + (b"\x00" * (expected_bytes - len(raw)))

    data = np.frombuffer(raw, dtype=dtype).reshape((height, width))
    # Ensure native byte-order for downstream processing
    if not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder("="))
    return data


# ════════════════════════════════════════════════════════════════════════════
#  TIFF reader (handles ImageJ big-endian 16-bit, multi-frame stacks, etc.)
# ════════════════════════════════════════════════════════════════════════════

_NATIVE_ENDIAN = "<" if np.little_endian else ">"


def read_tiff(filepath):
    """Read a TIFF image file → numpy array.

    Handles ImageJ big-endian 16-bit TIFFs (mode ``I;16B``) and standard
    8/16/32-bit grayscale or RGB images.  Multi-frame TIFFs are returned
    as a 3-D array (N, H, W).

    Returns
    -------
    np.ndarray — shape (H, W) for single-frame, (N, H, W) for stacks.
    """
    img = Image.open(filepath)
    n_frames = getattr(img, "n_frames", 1)

    frames = []
    for i in range(n_frames):
        img.seek(i)
        arr = np.array(img)
        # Pillow may return big-endian dtype ('>u2') — convert to native
        if arr.dtype.byteorder not in ("=", "|", _NATIVE_ENDIAN):
            arr = arr.astype(arr.dtype.newbyteorder("="))
        frames.append(arr)

    if len(frames) == 1:
        return frames[0]
    return np.stack(frames, axis=0)


# ════════════════════════════════════════════════════════════════════════════
#  Image scaling (pixel replication, no interpolation)
# ════════════════════════════════════════════════════════════════════════════

def scale_image(image, sx=3, sy=1):
    """Scale *image* by integer factors via pixel replication.

    Equivalent to ImageJ ``Scale…`` with ``interpolation=None``.
    """
    out = image
    if sy > 1:
        out = np.repeat(out, sy, axis=0)
    if sx > 1:
        out = np.repeat(out, sx, axis=-1)
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Analyze 7.5 writer / reader
# ════════════════════════════════════════════════════════════════════════════

_ANALYZE_DT = {
    np.dtype("uint8"):   (2,  8),
    np.dtype("int16"):   (4,  16),
    np.dtype("uint16"):  (4,  16),   # no unsigned-16 in Analyze; store raw bytes
    np.dtype("int32"):   (8,  32),
    np.dtype("float32"): (16, 32),
    np.dtype("float64"): (64, 64),
}


def write_analyze(filepath, data):
    """Write a 2-D or 3-D image as Analyze 7.5  (.hdr + .img).

    Parameters
    ----------
    filepath : str
        Base path (extension is replaced).  Creates ``<base>.hdr`` and ``<base>.img``.
    data : np.ndarray
        Shape (H, W) for a single slice, or (N, H, W) for a stack.
    """
    base = _strip_ext(filepath)

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    nslices, height, width = data.shape

    dt_code, bitpix = _ANALYZE_DT.get(data.dtype, (4, 16))

    # ---------- build 348-byte header (little-endian) ----------
    hdr = bytearray(348)

    # header_key  (bytes 0-39)
    struct.pack_into("<i", hdr, 0,  348)        # sizeof_hdr
    struct.pack_into("<i", hdr, 32, 16384)      # extents
    hdr[38] = ord("r")                          # regular

    # image_dimension  (bytes 40-147)
    struct.pack_into("<h", hdr, 40, 4)          # dim[0] — number of dims
    struct.pack_into("<h", hdr, 42, width)      # dim[1] — x
    struct.pack_into("<h", hdr, 44, height)     # dim[2] — y
    struct.pack_into("<h", hdr, 46, nslices)    # dim[3] — z
    struct.pack_into("<h", hdr, 48, 1)          # dim[4] — t

    struct.pack_into("<h", hdr, 70, dt_code)    # datatype
    struct.pack_into("<h", hdr, 72, bitpix)     # bitpix

    for off in (76, 80, 84, 88):                # pixdim[0..3]
        struct.pack_into("<f", hdr, off, 1.0)

    struct.pack_into("<i", hdr, 140, int(data.max()))   # glmax
    struct.pack_into("<i", hdr, 144, int(data.min()))   # glmin

    # description
    desc = b"AIDaS OCT Processing"
    hdr[148:148 + len(desc)] = desc

    # ---------- write files ----------
    with open(base + ".hdr", "wb") as fh:
        fh.write(hdr)

    with open(base + ".img", "wb") as fh:
        # Analyze stores x-fastest, then y, then z — same as C-contiguous (row-major)
        for s in range(nslices):
            slice_arr = np.ascontiguousarray(np.flipud(data[s]))
            fh.write(slice_arr.tobytes())

    return base + ".hdr", base + ".img"


def read_analyze(filepath):
    """Read an Analyze 7.5 file pair (.hdr + .img) → np.ndarray."""
    base = _strip_ext(filepath)

    with open(base + ".hdr", "rb") as fh:
        hdr = fh.read(348)

    # Detect endianness
    sz = struct.unpack_from("<i", hdr, 0)[0]
    end = "<" if sz == 348 else ">"
    if end == ">" and struct.unpack_from(">i", hdr, 0)[0] != 348:
        raise ValueError("Invalid Analyze header")

    width   = struct.unpack_from(f"{end}h", hdr, 42)[0]
    height  = struct.unpack_from(f"{end}h", hdr, 44)[0]
    nslices = struct.unpack_from(f"{end}h", hdr, 46)[0]
    dt_code = struct.unpack_from(f"{end}h", hdr, 70)[0]
    bitpix  = struct.unpack_from(f"{end}h", hdr, 72)[0]

    dt_map = {(2, 8): np.uint8, (4, 16): np.int16,
              (8, 32): np.int32, (16, 32): np.float32, (64, 64): np.float64}
    dtype = dt_map.get((dt_code, bitpix), np.int16)

    with open(base + ".img", "rb") as fh:
        raw = np.frombuffer(fh.read(), dtype=dtype)

    if nslices > 1:
        arr = raw.reshape((nslices, height, width))
        # When writing we stored flipped slices (flipud). Flip them back to upright.
        return np.stack([np.flipud(arr[s]) for s in range(arr.shape[0])], axis=0)

    arr = raw.reshape((height, width))
    return np.flipud(arr)


# ════════════════════════════════════════════════════════════════════════════
#  TIFF utility
# ════════════════════════════════════════════════════════════════════════════

def save_tiff(filepath, data):
    """Save a numpy array as TIFF (8 or 16-bit)."""
    if data.dtype == np.uint16:
        img = Image.fromarray(data.astype(np.uint16))
    elif data.dtype == np.int16:
        img = Image.fromarray(data.astype(np.int16))
    else:
        img = Image.fromarray(data)
    img.save(filepath)


# ════════════════════════════════════════════════════════════════════════════
#  helpers
# ════════════════════════════════════════════════════════════════════════════

def _strip_ext(path):
    base = str(path)
    for ext in (".hdr", ".img", ".HDR", ".IMG"):
        if base.endswith(ext):
            return base[:-4]
    # Also strip .tif/.tiff just in case
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        if base.endswith(ext):
            return base[:-len(ext)]
    return base
