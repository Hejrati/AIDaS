from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from aidas.utils.io_utils import read_raw_oct
from aidas.utils.sdb_reader import (
    detect_sdb_image,
    read_sdb_image,
    read_sdb_pixels,
)
from sdb_fixture import build_cmdb, ir_chunk, metadata_chunk, oct_chunk


class SdbReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "scan.sdb"

    def write(self, chunks):
        self.path.write_bytes(build_cmdb(chunks))
        return detect_sdb_image(self.path)

    def write_edb(self, chunks, filename="scan.edb"):
        (self.directory / filename).write_bytes(build_cmdb(chunks))

    def test_selects_oct_after_ir_using_native_dimensions(self):
        info = self.write([ir_chunk(9, 5), oct_chunk(7, 3)])
        self.assertEqual((info.width, info.height, info.frame_count), (7, 3, 1))
        self.assertGreater(info.offset, 9 * 5)
        self.assertEqual(info.file_size, self.path.stat().st_size)
        self.assertEqual(info.import_params(), {
            "width": 7, "height": 3, "offset": info.offset,
            "little_endian": True, "bit_depth": 16,
        })
        self.assertEqual(read_sdb_image(self.path, info).shape, (3, 7))
        with self.assertRaises(FrozenInstanceError):
            info.width = 999

    def test_dimensions_belong_to_each_file_not_a_global_default(self):
        for width, height in [(1, 1), (13, 4), (32, 27)]:
            with self.subTest(width=width, height=height):
                info = self.write([oct_chunk(width, height)])
                image = read_sdb_image(self.path, info)
                self.assertEqual(image.shape, (height, width))
                self.assertEqual(image.dtype, np.uint16)

    def test_first_scan_is_selected_in_series_and_slice_order(self):
        info = self.write([
            oct_chunk(2, 4, slice_id=8),
            oct_chunk(6, 3, slice_id=0),
            oct_chunk(9, 1, slice_id=4),
        ])
        self.assertEqual((info.width, info.height, info.frame_count), (6, 3, 3))
        self.assertEqual(read_sdb_image(self.path, info).shape, (3, 6))

    def test_follows_valid_linked_directory_pages(self):
        raw = bytearray(build_cmdb([oct_chunk(2, 4, slice_id=8)]))
        directory_position = len(raw)
        entry_position = directory_position + 52
        start = entry_position + 44
        payload = oct_chunk(6, 3)["payload"]
        raw.extend(struct.pack("<12sI10H4I", b"MDbDir", 100, *([0] * 10),
                               1, directory_position, 88, 0))
        raw.extend(struct.pack("<7IiHHII", entry_position, start, len(payload),
                               0, 1, 2, 3, 0, 1, 0, 0x40000000, 0))
        raw.extend(struct.pack("<12s8IiHHII", b"MDbData", 0, entry_position,
                               start, len(payload), 0, 1, 2, 3, 0, 1, 0, 0x40000000, 0))
        raw.extend(payload)
        struct.pack_into("<I", raw, 76, directory_position)
        self.path.write_bytes(raw)
        info = detect_sdb_image(self.path)
        self.assertEqual((info.width, info.height, info.frame_count), (6, 3, 2))

    def test_reserved_directory_slots_are_not_treated_as_chunks(self):
        self.path.write_bytes(build_cmdb([oct_chunk(7, 3)], directory_capacity=512))
        info = detect_sdb_image(self.path)
        self.assertEqual((info.width, info.height, info.frame_count), (7, 3, 1))

    def test_case_insensitive_same_stem_companion_confirms_dimensions(self):
        self.write_edb([metadata_chunk(7, 3, spacing=0.0038)], filename="SCAN.EDB")
        info = self.write([oct_chunk(7, 3)])
        self.assertEqual(info.edb_status, "SDB/EDB confirmed")
        self.assertAlmostEqual(info.axial_spacing_mm, 0.0038, places=8)

    def test_missing_companion_keeps_native_detection(self):
        info = self.write([oct_chunk(7, 3)])
        self.assertEqual(info.edb_status, "SDB header")
        self.assertIsNone(info.axial_spacing_mm)

    def test_pdb_is_not_used_to_guess_image_dimensions(self):
        (self.directory / "scan.pdb").write_bytes(b"not image dimensions")
        info = self.write([oct_chunk(7, 3)])
        self.assertEqual((info.width, info.height, info.edb_status), (7, 3, "SDB header"))

    def test_companion_mismatch_does_not_replace_sound_sdb_dimensions(self):
        self.write_edb([metadata_chunk(100, 200)])
        info = self.write([oct_chunk(7, 3)])
        self.assertEqual((info.width, info.height), (7, 3))
        self.assertEqual(info.edb_status, "SDB header; EDB mismatch")
        self.assertIsNone(info.axial_spacing_mm)
        self.assertEqual(read_sdb_image(self.path, info).shape, (3, 7))

    def test_companion_requires_matching_database_slice_and_subtype(self):
        for mismatch in ({"ids": (999, 2, 3)}, {"ids": (1, 999, 3)},
                         {"ids": (1, 2, 999)}, {"slice_id": 2}, {"ind": 0}):
            with self.subTest(mismatch=mismatch):
                self.write_edb([metadata_chunk(7, 3, **mismatch)])
                info = self.write([oct_chunk(7, 3)])
                self.assertEqual(info.edb_status, "SDB header; EDB unrelated")
                self.assertIsNone(info.axial_spacing_mm)

    def test_different_stem_edb_is_ignored(self):
        self.write_edb([metadata_chunk(7, 3)], filename="other.edb")
        info = self.write([oct_chunk(7, 3)])
        self.assertEqual(info.edb_status, "SDB header")

    def test_truncated_or_unknown_companion_is_only_a_diagnostic(self):
        for raw in (b"CMDb", b"unknown companion", build_cmdb([metadata_chunk(7, 3)])[:-1]):
            with self.subTest(length=len(raw)):
                (self.directory / "scan.edb").write_bytes(raw)
                info = self.write([oct_chunk(7, 3)])
                self.assertEqual(info.edb_status, "SDB header; EDB unreadable")
                self.assertEqual(read_sdb_image(self.path, info).shape, (3, 7))

    def test_nonfinite_or_nonpositive_spacing_is_not_exposed(self):
        for spacing in (float("nan"), float("inf"), -0.2, 0.0):
            with self.subTest(spacing=spacing):
                self.write_edb([metadata_chunk(7, 3, spacing=spacing)])
                info = self.write([oct_chunk(7, 3)])
                self.assertEqual(info.edb_status, "SDB/EDB confirmed")
                self.assertIsNone(info.axial_spacing_mm)

    def test_unknown_flat_raw_files_return_none(self):
        for raw in (b"", b"flat binary data", b"\x00" * 1024):
            with self.subTest(length=len(raw)):
                self.path.write_bytes(raw)
                self.assertIsNone(detect_sdb_image(self.path))

    def test_recognized_truncated_files_raise_instead_of_raw_fallback(self):
        original = build_cmdb([oct_chunk(7, 3)])
        for length in (4, 12, 36, 87, 140, len(original) - 1):
            with self.subTest(length=length):
                self.path.write_bytes(original[:length])
                with self.assertRaises(ValueError):
                    detect_sdb_image(self.path)

    def test_native_ir_only_file_has_no_oct_and_raises(self):
        self.path.write_bytes(build_cmdb([ir_chunk(7, 3)]))
        with self.assertRaisesRegex(ValueError, "No supported OCT"):
            detect_sdb_image(self.path)

    def test_invalid_directory_and_chunk_references_raise(self):
        original = build_cmdb([oct_chunk(7, 3)])
        start = struct.unpack_from("<I", original, 144)[0]
        changes = {
            "main pointer outside file": (76, len(original) + 100),
            "directory self position": (128, 99),
            "directory cycle": (132, 88),
            "directory count outside file": (124, 0xFFFFFFFF),
            "entry self position": (140, 0),
            "entry wrong database ID": (156, 999),
            "entry wrong type": (176, 10004),
            "entry wrong payload length": (148, 1),
            "chunk self position": (start + 20, start + 2),
            "chunk entry position": (start + 16, 999),
            "image wrong count": (start + 68, 1),
            "image zero height": (start + 72, 0),
            "image wrong size": (start + 60, 22),
            "image unsupported encoding": (start + 64, 0xFFFFFFFF),
        }
        for label, (offset, value) in changes.items():
            with self.subTest(label=label):
                raw = bytearray(original)
                struct.pack_into("<I", raw, offset, value)
                self.path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    detect_sdb_image(self.path)

    def test_does_not_scan_unreferenced_pixel_chunks(self):
        valid = build_cmdb([oct_chunk(7, 3)])
        orphan = build_cmdb([oct_chunk(19, 20)])
        self.path.write_bytes(valid + orphan)
        info = detect_sdb_image(self.path)
        self.assertEqual((info.width, info.height, info.frame_count), (7, 3, 1))

    def test_overlapping_active_chunk_payloads_are_rejected(self):
        raw = bytearray(build_cmdb([
            {"kind": 123, "ind": 65535, "payload": b"\0" * 512},
            oct_chunk(3, 2),
        ]))
        outer_start = struct.unpack_from("<I", raw, 144)[0]
        inner_start = struct.unpack_from("<I", raw, 188)[0]
        inner_record = raw[inner_start:]
        overlap_start = outer_start + 60 + 64
        raw[overlap_start:overlap_start + len(inner_record)] = inner_record
        struct.pack_into("<I", raw, 188, overlap_start)
        struct.pack_into("<I", raw, overlap_start + 20, overlap_start)
        self.path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "Overlapping.*data chunks"):
            detect_sdb_image(self.path)

    def test_manual_parameters_cannot_include_ir_metadata_or_other_frames(self):
        info = self.write([ir_chunk(9, 5), oct_chunk(7, 3), oct_chunk(7, 3, slice_id=2)])
        for params in ({"offset": 0}, {"offset": info.offset - 2},
                       {"offset": info.offset + 1}, {"height": 4},
                       {"width": 0}, {"height": -1}):
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    read_sdb_pixels(self.path, info, **params)

    def test_manual_read_can_reshape_or_select_only_existing_oct_pixels(self):
        info = self.write([oct_chunk(8, 4, words=list(range(32)))])
        image = read_sdb_image(self.path, info)
        np.testing.assert_array_equal(
            read_sdb_pixels(self.path, info, width=4, height=8), image.reshape(8, 4)
        )
        np.testing.assert_array_equal(
            read_sdb_pixels(self.path, info, height=2, offset=info.offset + 8 * 2), image[1:3]
        )

    def test_rejects_native_payload_that_changed_after_detection(self):
        info = self.write([oct_chunk(7, 3)])
        with self.path.open("ab") as stream:
            stream.write(b"extra")
        with self.assertRaisesRegex(ValueError, "changed after"):
            read_sdb_image(self.path, info)

    def test_rejects_same_size_header_change_after_detection(self):
        info = self.write([oct_chunk(7, 3)])
        raw = bytearray(self.path.read_bytes())
        struct.pack_into("<I", raw, info.offset - 4, 3)
        struct.pack_into("<I", raw, info.offset - 8, 7)
        self.path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "changed after"):
            read_sdb_image(self.path, info)

    def test_all_unsigned_words_are_preserved_without_pixel_processing(self):
        words = np.arange(65536, dtype=np.uint16)
        info = self.write([oct_chunk(256, 256, words=words)])
        image = read_sdb_image(self.path, info)
        self.assertEqual(image.dtype, np.uint16)
        self.assertTrue(image.dtype.isnative)
        np.testing.assert_array_equal(image, words.reshape(256, 256))
        self.assertEqual(image.tobytes(), words.tobytes())
        self.assertEqual(int(image[-1, -1]), 65535)

    def test_all_unsigned_words_match_manual_import_for_both_endian_settings(self):
        words = np.arange(65536, dtype=np.uint16)
        info = self.write([ir_chunk(9, 5), oct_chunk(256, 256, words=words)])
        for little_endian in (True, False):
            with self.subTest(little_endian=little_endian):
                manual = read_raw_oct(
                    self.path, width=info.width, height=info.height,
                    offset=info.offset, bit_depth=16, little_endian=little_endian,
                )
                image = read_sdb_image(self.path, info, little_endian=little_endian)
                pixels = read_sdb_pixels(self.path, info, little_endian=little_endian)
                self.assertEqual(image.dtype, np.uint16)
                self.assertTrue(image.dtype.isnative)
                np.testing.assert_array_equal(image, manual)
                np.testing.assert_array_equal(pixels, manual)
                self.assertEqual(image.tobytes(), manual.tobytes())

    def test_endian_override_only_changes_byte_order_of_original_unsigned_words(self):
        words = np.array([0, 0x0001, 0x0100, 0x7FFF, 0x8000, 0xFFFF], dtype=np.uint16)
        info = self.write([oct_chunk(3, 2, words=words)])
        np.testing.assert_array_equal(
            read_sdb_image(self.path, info, little_endian=True), words.reshape(2, 3)
        )
        np.testing.assert_array_equal(
            read_sdb_image(self.path, info, little_endian=False), words.byteswap().reshape(2, 3)
        )


if __name__ == "__main__":
    unittest.main()
