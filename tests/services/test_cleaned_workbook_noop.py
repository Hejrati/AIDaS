"""Saved LIGHT2 values/formats stay exact when redundant exports are skipped."""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Font, NamedStyle
from openpyxl.worksheet.datavalidation import DataValidation

from aidas.services import step4_compiler as compiler
from aidas.services.cleaned_workbook import matches_generated_light_workbook


_TEXT = ("Distance\tWhole\tELM\tONL\n"
         "-1\tlabel\t0.33363333333333334\t1.25\n"
         "0\t0\tNA\tNaN\n"
         "1\t0\t0.33413333333333334\tInfinity\n")


def _parts(path):
    with ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(parts["docProps/core.xml"])
    for element in root:
        if element.tag in {"{http://purl.org/dc/terms/}created", "{http://purl.org/dc/terms/}modified"}:
            element.text = ""
    parts["docProps/core.xml"] = ET.tostring(root)
    return parts


class CleanedWorkbookNoopTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _source(self, name, content=_TEXT):
        folder = self.root / name
        folder.mkdir()
        path = folder / "_thickness_vs_distance_from_fovea_LIGHT.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def _expected_rows(self, source):
        rows = compiler._clean_light_rows(compiler.read_tsv_rows(source))
        return ([list(rows[0])] if rows else []) + [
            [compiler._excel_thickness_value(value) for value in row] for row in rows[1:]]

    def test_fractional_saved_values_are_reused_without_rewriting(self):
        source = self._source("fractional")
        destination = Path(compiler.create_cleaned_light_thickness_workbook(source))
        original = destination.read_bytes()
        source_bytes = source.read_bytes()
        rows = compiler.read_thickness_rows(destination)
        self.assertEqual(rows[1][2].hex(), float("0.3336333333333333").hex())
        self.assertNotEqual(rows[1][2].hex(), float("0.33363333333333334").hex())
        self.assertIsNone(rows[2][2])
        self.assertEqual(rows[2][3], "NaN")
        self.assertEqual(rows[3][3], "Infinity")
        self.assertTrue(matches_generated_light_workbook(destination, self._expected_rows(source)))
        with mock.patch.object(compiler.Workbook, "save", side_effect=AssertionError("Redundant save")):
            self.assertEqual(compiler.create_cleaned_light_thickness_workbook(source), str(destination))
        self.assertEqual(destination.read_bytes(), original)
        self.assertEqual(source.read_bytes(), source_bytes)
        self.assertEqual(compiler.read_thickness_rows(destination), rows)

    def test_modified_layout_metadata_and_values_take_the_original_export_path(self):
        mutations = {
            "number_format": lambda w: setattr(w.active["C2"], "number_format", "0.000000"),
            "font": lambda w: setattr(w.active["C2"], "font", Font(italic=True)),
            "row_height": lambda w: setattr(w.active.row_dimensions[2], "height", 31),
            "hidden_row": lambda w: setattr(w.active.row_dimensions[2], "hidden", True),
            "column_width": lambda w: setattr(w.active.column_dimensions["C"], "width", 80),
            "hidden_column": lambda w: setattr(w.active.column_dimensions["C"], "hidden", True),
            "freeze": lambda w: setattr(w.active, "freeze_panes", "B3"),
            "filter": lambda w: setattr(w.active.auto_filter, "ref", "A1:C2"),
            "sheet_title": lambda w: setattr(w.active, "title", "User title"),
            "extra_sheet": lambda w: w.create_sheet("extra"),
            "formula": lambda w: setattr(w.active["C2"], "value", "=1/3"),
            "comment": lambda w: setattr(w.active["C2"], "comment", Comment("note", "author")),
            "merge": lambda w: w.active.merge_cells("B2:C2"),
            "protection": lambda w: setattr(w.active.protection, "sheet", True),
            "creator": lambda w: setattr(w.properties, "creator", "User author"),
            "margin": lambda w: setattr(w.active.page_margins, "left", 3.0),
            "unused_style": lambda w: w.add_named_style(NamedStyle(name="Unused", font=Font(italic=True))),
            "tab_color": lambda w: setattr(w.active.sheet_properties, "tabColor", "FF0000"),
            "validation": lambda w: w.active.add_data_validation(DataValidation(type="whole", formula1=0, formula2=10)),
            "value": lambda w: setattr(w.active["C2"], "value", 9.5),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                source = self._source(name)
                destination = Path(compiler.create_cleaned_light_thickness_workbook(source))
                expected = _parts(destination)
                workbook = load_workbook(destination)
                try:
                    mutation(workbook)
                    workbook.save(destination)
                finally:
                    workbook.close()
                self.assertFalse(matches_generated_light_workbook(destination, self._expected_rows(source)))
                compiler.create_cleaned_light_thickness_workbook(source)
                self.assertEqual(_parts(destination), expected)

    def test_unknown_archive_parts_force_normal_export(self):
        source = self._source("extra_part")
        destination = Path(compiler.create_cleaned_light_thickness_workbook(source))
        expected = _parts(destination)
        with ZipFile(destination, "a") as archive:
            archive.writestr("custom/extra.xml", "<custom/>")
        self.assertFalse(matches_generated_light_workbook(destination, self._expected_rows(source)))
        compiler.create_cleaned_light_thickness_workbook(source)
        self.assertEqual(_parts(destination), expected)

    def test_original_exact_value_branch_still_preserves_custom_layout(self):
        source = self._source("exact", _TEXT.replace("0.33363333333333334", "0.5").replace("0.33413333333333334", "0.75"))
        destination = Path(compiler.create_cleaned_light_thickness_workbook(source))
        workbook = load_workbook(destination)
        try:
            workbook.active["C2"].number_format = "0.000000"
            workbook.save(destination)
        finally:
            workbook.close()
        before = destination.read_bytes()
        compiler.create_cleaned_light_thickness_workbook(source)
        self.assertEqual(destination.read_bytes(), before)

    def test_xml_comments_and_processing_instructions_force_normal_export(self):
        for name, part, marker in (
            ("sheet_comment", "xl/worksheets/sheet1.xml", b"<!--custom metadata-->"),
            ("sheet_pi", "xl/worksheets/sheet1.xml", b"<?aidas custom?>"),
            ("core_comment", "docProps/core.xml", b"<!--custom metadata-->"),
            ("core_pi", "docProps/core.xml", b"<?aidas custom?>"),
        ):
            with self.subTest(name=name):
                source = self._source(name)
                destination = Path(compiler.create_cleaned_light_thickness_workbook(source))
                expected = _parts(destination)
                buffer = BytesIO()
                with ZipFile(destination) as original, ZipFile(buffer, "w") as changed:
                    for info in original.infolist():
                        raw = original.read(info.filename)
                        if info.filename == part:
                            prefix, suffix = raw.rsplit(b"</", 1)
                            raw = prefix + marker + b"</" + suffix
                        changed.writestr(info, raw)
                destination.write_bytes(buffer.getvalue())
                self.assertFalse(matches_generated_light_workbook(destination, self._expected_rows(source)))
                compiler.create_cleaned_light_thickness_workbook(source)
                self.assertEqual(_parts(destination), expected)

    def test_source_is_reread_even_with_unchanged_size_and_modification_time(self):
        source = self._source("freshness")
        destination = Path(compiler.create_cleaned_light_thickness_workbook(source))
        stamp = source.stat()
        source.write_text(_TEXT.replace("0.33363333333333334", "0.43363333333333334"), encoding="utf-8")
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertEqual(source.stat().st_size, stamp.st_size)
        compiler.create_cleaned_light_thickness_workbook(source)
        self.assertEqual(compiler.read_thickness_rows(destination)[1][2].hex(), float("0.4336333333333333").hex())

    def test_unexpected_xml_container_text_and_prolog_metadata_force_export(self):
        namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        for name in ("data_text", "data_tail", "row_text", "row_tail", "cell_text", "cell_tail",
                     "number_tail", "inline_text", "inline_tail", "text_tail", "prolog_comment", "prolog_pi"):
            with self.subTest(name=name):
                source = self._source(name)
                destination = Path(compiler.create_cleaned_light_thickness_workbook(source))
                expected = _parts(destination)
                buffer = BytesIO()
                with ZipFile(destination) as original, ZipFile(buffer, "w") as changed:
                    for info in original.infolist():
                        raw = original.read(info.filename)
                        if info.filename == "xl/worksheets/sheet1.xml":
                            sheet = ET.fromstring(raw)
                            data = sheet.find(namespace + "sheetData")
                            row = data[1]
                            numeric = next(cell for cell in row if cell.get("r") == "C2")
                            text = next(cell for cell in row if cell.get("r") == "B2")
                            targets = {
                                "data": data, "row": row, "cell": numeric,
                                "number": numeric[0], "inline": text[0], "text": text[0][0],
                            }
                            if name.startswith("prolog"):
                                marker = b"<!--unexpected metadata-->" if name.endswith("comment") else b"<?aidas unexpected?>"
                                raw = marker + raw
                            else:
                                target, attribute = name.split("_")
                                setattr(targets[target], attribute, "unexpected")
                                raw = ET.tostring(sheet, encoding="utf-8")
                        changed.writestr(info, raw)
                destination.write_bytes(buffer.getvalue())
                self.assertFalse(matches_generated_light_workbook(destination, self._expected_rows(source)))
                compiler.create_cleaned_light_thickness_workbook(source)
                self.assertEqual(_parts(destination), expected)

    def test_required_write_failure_preserves_existing_file_and_removes_temporary(self):
        source = self._source("failure")
        destination = Path(compiler.create_cleaned_light_thickness_workbook(source))
        original = destination.read_bytes()
        source.write_text(_TEXT.replace("0.33363333333333334", "2.5"), encoding="utf-8")
        with mock.patch.object(compiler.os, "replace", side_effect=PermissionError("locked destination")):
            with self.assertRaisesRegex(PermissionError, "locked"):
                compiler.create_cleaned_light_thickness_workbook(source)
        self.assertEqual(destination.read_bytes(), original)
        self.assertEqual(list(source.parent.glob(f".{destination.stem}.*.xlsx")), [])

    def test_doctype_entities_and_cdata_are_not_generated_workbooks(self):
        for name in ("doctype", "entity", "cdata"):
            with self.subTest(name=name):
                source = self._source(name)
                destination = Path(compiler.create_cleaned_light_thickness_workbook(source))
                expected = _parts(destination)
                buffer = BytesIO()
                with ZipFile(destination) as original, ZipFile(buffer, "w") as changed:
                    for info in original.infolist():
                        raw = original.read(info.filename)
                        if info.filename == "xl/worksheets/sheet1.xml":
                            if name == "doctype":
                                raw = b"<!DOCTYPE worksheet>" + raw
                            elif name == "entity":
                                raw = b'<!DOCTYPE worksheet [<!ENTITY label "label">]>' + raw
                            else:
                                self.assertIn(b"<t>label</t>", raw)
                                raw = raw.replace(b"<t>label</t>", b"<t><![CDATA[label]]></t>", 1)
                        changed.writestr(info, raw)
                destination.write_bytes(buffer.getvalue())
                self.assertFalse(matches_generated_light_workbook(destination, self._expected_rows(source)))
                if name != "entity":
                    compiler.create_cleaned_light_thickness_workbook(source)
                    self.assertEqual(_parts(destination), expected)
                # Optional hardened XML readers may reject ENTITY declarations
                # before the existing export path reaches this guard.

    def test_corrupt_saved_workbook_is_recreated(self):
        source = self._source("corrupt")
        destination = Path(compiler.create_cleaned_light_thickness_workbook(source))
        expected = _parts(destination)
        destination.write_bytes(b"not an xlsx archive")
        compiler.create_cleaned_light_thickness_workbook(source)
        self.assertEqual(_parts(destination), expected)

    def test_empty_irregular_and_extreme_inputs_keep_exact_export_parts(self):
        contents = {
            "uneven": "Distance\tWhole\tELM\n-1\t0\t0.33363333333333334\t1.25\textra\n0\t0\tNA\n",
            "empty_header": "\n-1\t0\t0.33363333333333334\t1.25\n",
            "header_only": "Distance\tWhole\tELM\tONL\n",
            "empty_source": "",
            "extremes": "Distance\tWhole\tELM\tONL\n-1\t-0.0\t1.2345678901234567e-200\t1.7976931348623157e308\n0\t0\t-1.2345678901234567e100\t-infinity\n",
        }
        for name, content in contents.items():
            with self.subTest(name=name):
                source = self._source(name, content)
                destination = Path(compiler.create_cleaned_light_thickness_workbook(source))
                expected = _parts(destination)
                self.assertEqual(matches_generated_light_workbook(destination, self._expected_rows(source)), name != "empty_source")
                compiler.create_cleaned_light_thickness_workbook(source)
                self.assertEqual(_parts(destination), expected)


if __name__ == "__main__":
    unittest.main()
