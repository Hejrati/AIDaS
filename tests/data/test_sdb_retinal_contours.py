from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aidas.utils.sdb_reader import (
    detect_sdb_image,
    read_sdb_image,
    read_sdb_retinal_contours,
)

from tests.support.sdb_fixture import build_cmdb, metadata_chunk, oct_chunk


def contour_chunk(layer_id, values, *, kind=10019, padding=0, **identity):
    values = np.asarray(values, dtype="<f4")
    payload = struct.pack("<4I", 2, layer_id, 5, values.size)
    payload += b"\0" * padding + values.tobytes()
    return {"kind": kind, "ind": 1, "payload": payload, **identity}


class SdbRetinalContourTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "scan.sdb"
        self.edb = self.directory / "scan.edb"
        self.width, self.height = 100, 80
        xs = np.linspace(-1, 1, self.width, dtype=np.float32)
        self.ilm = 15 + 8 * xs**2
        self.bm = self.ilm + 20
        self.words = np.arange(self.width * self.height, dtype=np.uint16)
        self.path.write_bytes(build_cmdb([
            oct_chunk(self.width, self.height, words=self.words),
        ]))
        self.info = detect_sdb_image(self.path)

    def write_edb(self, contours=None, metadata=None):
        if contours is None:
            contours = [contour_chunk(0, self.ilm), contour_chunk(1, self.bm)]
        if metadata is None:
            metadata = [metadata_chunk(self.width, self.height)]
        self.edb.write_bytes(build_cmdb([*metadata, *contours]))

    def test_outer_retinal_coordinates_support_both_observed_header_lengths(self):
        original = self.path.read_bytes()
        for padding in (0, 20):
            with self.subTest(padding=padding):
                self.write_edb([
                    contour_chunk(0, self.ilm, padding=padding),
                    contour_chunk(1, self.bm, padding=padding),
                    contour_chunk(2, np.full(self.width, np.finfo(np.float32).max)),
                ])
                contours = read_sdb_retinal_contours(self.path, self.info)
                self.assertEqual(contours.shape, (2, self.width))
                np.testing.assert_array_equal(contours[0], self.ilm)
                np.testing.assert_array_equal(contours[1], self.bm)
                self.assertEqual(self.path.read_bytes(), original)
                np.testing.assert_array_equal(
                    read_sdb_image(self.path, self.info).ravel(), self.words,
                )

    def test_current_contours_supersede_legacy_for_each_outer_boundary(self):
        self.write_edb([
            contour_chunk(0, np.zeros(self.width)),
            contour_chunk(1, np.zeros(self.width - 1)),
            contour_chunk(0, self.ilm, kind=0x2723, padding=20),
            contour_chunk(1, self.bm, kind=0x2723, padding=20),
        ])
        contours = read_sdb_retinal_contours(self.path, self.info)
        np.testing.assert_array_equal(contours, np.stack([self.ilm, self.bm]))

    def test_missing_legacy_annotation_points_remain_nan_at_original_columns(self):
        top = self.ilm.copy()
        top[:7] = [0, -1, np.nan, np.inf, np.finfo(np.float32).max, 80, 300]
        bottom = self.bm.copy()
        bottom[-8:] = np.finfo(np.float32).max
        self.write_edb([contour_chunk(0, top), contour_chunk(1, bottom)])
        contours = read_sdb_retinal_contours(self.path, self.info)
        self.assertEqual(contours.shape, (2, self.width))
        self.assertTrue(np.isnan(contours[0, :7]).all())
        self.assertTrue(np.isnan(contours[1, -8:]).all())
        np.testing.assert_array_equal(contours[0, 7:], self.ilm[7:])
        np.testing.assert_array_equal(contours[1, :-8], self.bm[:-8])

    def test_zero_is_valid_in_current_contour_coordinates(self):
        top = self.ilm.copy()
        top[0] = 0
        self.write_edb([
            contour_chunk(0, top, kind=0x2723, padding=20),
            contour_chunk(1, self.bm, kind=0x2723, padding=20),
        ])
        self.assertEqual(read_sdb_retinal_contours(self.path, self.info)[0, 0], 0)

    def test_metadata_must_confirm_dimensions_for_the_selected_oct_identity(self):
        for metadata in ([], [metadata_chunk(99, 80)], [metadata_chunk(100, 79)],
                         [metadata_chunk(100, 80, ids=(9, 2, 3))],
                         [metadata_chunk(100, 80, slice_id=2)],
                         [metadata_chunk(100, 80, ind=0)]):
            with self.subTest(metadata=metadata):
                self.write_edb(metadata=metadata)
                self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))

    def test_contours_require_matching_database_series_slice_and_subtype(self):
        for mismatch in ({"ids": (9, 2, 3)}, {"ids": (1, 9, 3)},
                         {"ids": (1, 2, 9)}, {"slice_id": 2}, {"ind": 0}):
            with self.subTest(mismatch=mismatch):
                self.write_edb([
                    contour_chunk(0, self.ilm, **mismatch),
                    contour_chunk(1, self.bm, **mismatch),
                ])
                self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))

    def test_single_inner_or_missing_outer_contours_do_not_define_retina_envelope(self):
        for contours in ([], [contour_chunk(0, self.ilm)], [contour_chunk(1, self.bm)],
                         [contour_chunk(2, self.ilm), contour_chunk(16, self.bm)]):
            with self.subTest(contours=contours):
                self.write_edb(contours)
                self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))

    def test_partial_coverage_and_large_interior_annotation_holes_are_rejected(self):
        for missing in (slice(0, 21), slice(42, 57)):
            with self.subTest(missing=missing):
                top = self.ilm.copy()
                top[missing] = np.nan
                self.write_edb([contour_chunk(0, top), contour_chunk(1, self.bm)])
                self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))

    def test_crossing_or_collapsed_outer_boundaries_are_rejected(self):
        for value in (self.ilm[50] - 1, self.ilm[50], self.ilm[50] + 0.5):
            with self.subTest(value=value):
                bottom = self.bm.copy()
                bottom[50] = value
                self.write_edb([contour_chunk(0, self.ilm), contour_chunk(1, bottom)])
                self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))

    def test_duplicate_outer_contours_must_agree(self):
        different = self.bm.copy()
        different[30] += 1
        self.write_edb([
            contour_chunk(0, self.ilm), contour_chunk(1, self.bm),
            contour_chunk(1, different),
        ])
        self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))
        self.write_edb([
            contour_chunk(0, self.ilm), contour_chunk(1, self.bm),
            contour_chunk(1, self.bm),
        ])
        np.testing.assert_array_equal(
            read_sdb_retinal_contours(self.path, self.info), np.stack([self.ilm, self.bm]),
        )

    def test_unsupported_padding_or_width_is_rejected(self):
        nonzero_padding = contour_chunk(0, self.ilm, padding=20)
        raw = bytearray(nonzero_padding["payload"])
        raw[20] = 1
        nonzero_padding["payload"] = bytes(raw)
        for invalid in (contour_chunk(0, self.ilm, padding=4),
                        contour_chunk(0, self.ilm, padding=24),
                        contour_chunk(0, self.ilm[:-1]),
                        nonzero_padding):
            with self.subTest(invalid=invalid):
                self.write_edb([invalid, contour_chunk(1, self.bm)])
                self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))

    def test_missing_different_stem_ambiguous_or_malformed_companion_provides_no_hint(self):
        self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))
        self.write_edb()
        self.edb.rename(self.directory / "other.edb")
        self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))
        self.write_edb()
        with patch.object(Path, "iterdir", return_value=iter([self.edb, self.edb])):
            self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))
        raw = self.edb.read_bytes()
        for malformed in (b"not CMDb", b"CMDb", raw[:-1]):
            self.edb.write_bytes(malformed)
            self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))

    def test_changed_sdb_or_missing_identity_cannot_reuse_old_companion_coordinates(self):
        self.write_edb()
        self.assertIsNone(read_sdb_retinal_contours(self.path, replace(self.info, _identity=None)))
        self.assertIsNone(read_sdb_retinal_contours(self.path, replace(self.info, offset=self.info.offset + 2)))
        self.assertIsNone(read_sdb_retinal_contours(self.path, replace(self.info, width=99)))
        raw = bytearray(self.path.read_bytes())
        struct.pack_into("<I", raw, self.info.offset - 4, self.height)
        struct.pack_into("<I", raw, self.info.offset - 8, self.width)
        self.path.write_bytes(raw)
        self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))
        self.path.write_bytes(bytes(raw) + b"changed")
        self.assertIsNone(read_sdb_retinal_contours(self.path, self.info))

    def test_duplicate_sdb_identity_cannot_attach_one_metadata_record_to_two_frames(self):
        self.write_edb()
        self.path.write_bytes(build_cmdb([
            oct_chunk(self.width, self.height), oct_chunk(self.width, self.height),
        ]))
        info = detect_sdb_image(self.path)
        self.assertIsNone(read_sdb_retinal_contours(self.path, info))


if __name__ == "__main__":
    unittest.main()
