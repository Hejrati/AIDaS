"""Byte-for-byte regression against Analyze export before copy consolidation."""

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from aidas.utils.io_utils import _strip_ext, write_analyze


# Frozen storage mapping and writer from the pre-optimization implementation.
_ANALYZE_DT = {
    np.dtype("uint8"): (2, 8), np.dtype("int16"): (4, 16),
    np.dtype("uint16"): (4, 16), np.dtype("int32"): (8, 32),
    np.dtype("float32"): (16, 32), np.dtype("float64"): (64, 64),
}


def _legacy_write_analyze(filepath, data):
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
    end = '>'
    hdr = bytearray(348)
    struct.pack_into(f'{end}i', hdr, 0, 348)
    struct.pack_into(f'{end}i', hdr, 32, 16384)
    hdr[38] = ord('r')
    struct.pack_into(f'{end}h', hdr, 40, 4)
    struct.pack_into(f'{end}h', hdr, 42, width)
    struct.pack_into(f'{end}h', hdr, 44, height)
    struct.pack_into(f'{end}h', hdr, 46, nslices)
    struct.pack_into(f'{end}h', hdr, 48, 1)
    hdr[56:60] = b'mm\x00\x00'
    struct.pack_into(f'{end}h', hdr, 70, dt_code)
    struct.pack_into(f'{end}h', hdr, 72, bitpix)
    for off in (80, 84, 88):
        struct.pack_into(f'{end}f', hdr, off, 1.0)
    struct.pack_into(f'{end}f', hdr, 112, 1.0)
    if dt_code == 4 and bitpix == 16:
        data_for_header = np.asarray(data, dtype=np.int16)
    else:
        data_for_header = data
    struct.pack_into(f'{end}i', hdr, 140, int(data_for_header.max()))
    struct.pack_into(f'{end}i', hdr, 144, int(data_for_header.min()))
    desc = b'AIDaS OCT Processing'
    hdr[148:148 + len(desc)] = desc
    with open(base + '.hdr', 'wb') as fh:
        fh.write(hdr)
    with open(base + '.img', 'wb') as fh:
        for s in range(nslices):
            slice_arr = np.ascontiguousarray(np.flipud(data[s]))
            if (dt_code, bitpix) == (2, 8):
                target_dtype = np.dtype('u1')
            elif (dt_code, bitpix) == (4, 16):
                target_dtype = np.dtype(f'{end}i2')
            elif (dt_code, bitpix) == (8, 32):
                target_dtype = np.dtype(f'{end}i4')
            elif (dt_code, bitpix) == (16, 32):
                target_dtype = np.dtype(f'{end}f4')
            elif (dt_code, bitpix) == (64, 64):
                target_dtype = np.dtype(f'{end}f8')
            else:
                target_dtype = np.dtype(f'{end}i2')
            slice_arr = slice_arr.astype(target_dtype, copy=False)
            fh.write(slice_arr.tobytes())
    return (base + '.hdr', base + '.img')


class AnalyzeExportTests(unittest.TestCase):
    def _assert_same_export(self, source, directory):
        before = source.tobytes()
        source.setflags(write=False)
        baseline = str(directory / "baseline")
        actual = str(directory / "actual")
        expected_paths = _legacy_write_analyze(baseline, source)
        actual_paths = write_analyze(actual, source)
        self.assertEqual(actual_paths, (actual + ".hdr", actual + ".img"))
        for expected_path, actual_path in zip(expected_paths, actual_paths):
            self.assertEqual(Path(expected_path).read_bytes(), Path(actual_path).read_bytes())
        self.assertEqual(source.tobytes(), before)
        self.assertFalse(source.flags.writeable)

    def test_headers_and_pixels_match_for_dtypes_byte_orders_and_memory_layouts(self):
        rng = np.random.default_rng(219)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for dtype in ("uint8", "uint16", "int16", "int32", "float32", "float64",
                          ">i2", ">u2", ">i4", ">f4", ">f8", "int8", "uint32", "float16"):
                array = (rng.normal(size=(2, 23, 41)) * 1000).astype(dtype)
                layouts = (array, array[0], array[:, :, ::-1], array[:, ::2, ::3],
                           np.asfortranarray(array))
                for index, source in enumerate(layouts):
                    with self.subTest(dtype=dtype, layout=index):
                        self._assert_same_export(source, directory)

    def test_signed_zero_and_integer_extremes_preserve_storage_bits(self):
        arrays = [np.array([[-0.0, 0.0], [-1.0, 1.0]], dtype=dtype)
                  for dtype in (np.float32, np.float64)]
        for dtype in (np.uint8, np.uint16, np.int16, np.int32):
            limits = np.iinfo(dtype)
            arrays.append(np.array([[limits.min, limits.max], [0, 1]], dtype=dtype))
        with tempfile.TemporaryDirectory() as temporary_directory:
            for array in arrays:
                with self.subTest(dtype=array.dtype):
                    self._assert_same_export(array, Path(temporary_directory))

    def test_representative_step1_and_step2_volumes_match_in_full(self):
        rng = np.random.default_rng(731)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for shape, dtype in (((2, 177, 2133), np.int16),
                                 ((2, 177, 2133), np.uint8),
                                 ((2, 461, 2851), np.float64)):
                image = rng.integers(0, 230, size=shape).astype(dtype)
                if dtype is np.uint8:
                    image[:, 50, :] = 255
                    image[:, 70, :] = 254
                    image[:, :, 1066] = 243
                with self.subTest(shape=shape, dtype=dtype):
                    self._assert_same_export(image, directory)
