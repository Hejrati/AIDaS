"""Exact reference for TIFF values/layout and bounded temporary allocations."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from aidas.utils import io_utils


def _legacy_read_tiff(path):
    """Frozen reader from before uniform-stack allocation optimization."""
    image = Image.open(path)
    frames = []
    for index in range(getattr(image, "n_frames", 1)):
        image.seek(index)
        array = np.array(image)
        if array.dtype.byteorder not in ("=", "|", io_utils._NATIVE_ENDIAN):
            array = array.astype(array.dtype.newbyteorder("="))
        frames.append(array)
    if len(frames) == 1:
        return frames[0]
    return np.stack(frames, axis=0)


class _Pages:
    def __init__(self, pages):
        self.pages = pages
        self.n_frames = len(pages)
        self.index = 0

    def seek(self, index):
        self.index = index

    def __array__(self, dtype=None):
        return np.asarray(self.pages[self.index], dtype=dtype)


class TiffStackAllocationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pixels = np.arange(77).reshape(7, 11)

    def assert_exact(self, expected, actual):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual(actual.strides, expected.strides)
        self.assertEqual(actual.flags.writeable, expected.flags.writeable)
        self.assertEqual(actual.flags.owndata, expected.flags.owndata)

    def _save(self, name, data, count):
        path = self.root / name
        images = [Image.fromarray(data) for _ in range(count)]
        try:
            images[0].save(path, save_all=count > 1, append_images=images[1:])
        finally:
            for image in images:
                image.close()
        return path

    def test_real_tiff_modes_and_float_payloads_match_frozen_reader(self):
        cases = {name: (self.pixels - 35).astype(name)
                 for name in ("uint8", "uint16", ">u2", "int16", "int32", "float32")}
        cases["bool"] = self.pixels % 2 == 0
        cases["RGB"] = np.stack([self.pixels.astype(np.uint8)] * 3, axis=-1)
        cases["RGBA"] = np.stack([self.pixels.astype(np.uint8)] * 4, axis=-1)
        bits = np.asarray([0x7FC00001, 0xFFC00002, 0x80000000, 0, 0x7F800000,
                           0xFF800000, 0x3EAAAAAB], dtype=np.uint32)
        cases["float_payloads"] = np.tile(bits.view(np.float32), (5, 1))
        for name, data in cases.items():
            for count in (1, 3):
                with self.subTest(mode=name, pages=count):
                    path = self._save(f"{name.replace('>', 'big_')}_{count}.tif", data, count)
                    original = path.read_bytes()
                    self.assert_exact(_legacy_read_tiff(path), io_utils.read_tiff(path))
                    self.assertEqual(path.read_bytes(), original)

    def test_heterogeneous_page_promotion_and_errors_remain_identical(self):
        cases = {
            "numeric_promotion": [self.pixels.astype(np.uint16), self.pixels.astype(np.float32), self.pixels.astype(np.int32)],
            "shape_error": [self.pixels.astype(np.uint16), self.pixels.T.astype(np.uint16)],
            "empty_error": [],
        }
        for name, pages in cases.items():
            with self.subTest(name=name):
                source_bytes = [array.tobytes() for array in pages]
                for array in pages:
                    array.setflags(write=False)
                observed = []
                for function in (_legacy_read_tiff, io_utils.read_tiff):
                    with mock.patch.object(io_utils.Image, "open", side_effect=lambda _path: _Pages(pages)):
                        try:
                            observed.append(function("pages"))
                        except Exception as error:
                            observed.append((type(error), str(error)))
                if isinstance(observed[0], np.ndarray):
                    self.assert_exact(*observed)
                else:
                    self.assertEqual(*observed)
                self.assertEqual([array.tobytes() for array in pages], source_bytes)
                self.assertTrue(all(not array.flags.writeable for array in pages))

    def test_preallocation_failure_uses_original_page_collection(self):
        pages = [self.pixels.astype(np.uint16)] * 3
        with mock.patch.object(io_utils.Image, "open", side_effect=lambda _path: _Pages(pages)):
            expected = _legacy_read_tiff("pages")
            with mock.patch.object(io_utils.np, "empty", side_effect=MemoryError("allocation")):
                actual = io_utils.read_tiff("pages")
        self.assert_exact(expected, actual)

    def test_later_page_error_releases_preallocation_before_reraising(self):
        class FailingPages(_Pages):
            def seek(self, index):
                if index == 1:
                    raise OSError("corrupt later page")
                super().seek(index)

        pages = [self.pixels.astype(np.uint16)] * 3
        errors = []
        released = False
        for function in (_legacy_read_tiff, io_utils.read_tiff):
            with mock.patch.object(io_utils.Image, "open", side_effect=lambda _path: FailingPages(pages)):
                try:
                    function("pages")
                except OSError as error:
                    errors.append((type(error), str(error)))
                    trace = error.__traceback__
                    while trace is not None:
                        if trace.tb_frame.f_code is io_utils.read_tiff.__code__:
                            values = trace.tb_frame.f_locals
                            released = all(values.get(name) is None for name in ("stack", "frames", "arr"))
                        trace = trace.tb_next
        self.assertEqual(errors[0], errors[1])
        self.assertTrue(released)

    def test_caller_owned_stream_stays_open(self):
        path = self._save("stream.tif", self.pixels.astype(np.uint16), 3)
        streams = [BytesIO(path.read_bytes()) for _ in range(2)]
        try:
            self.assert_exact(_legacy_read_tiff(streams[0]), io_utils.read_tiff(streams[1]))
            self.assertTrue(all(not stream.closed for stream in streams))
        finally:
            for stream in streams:
                stream.close()


if __name__ == "__main__":
    unittest.main()
