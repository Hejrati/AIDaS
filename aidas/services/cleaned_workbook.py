"""Recognize an unchanged LIGHT2 workbook without rewriting its saved values.

The writer's exact cell tokens and complete generated layout must match.
Unknown formatting or metadata falls back to normal export. Only fixed layout
templates are cached; source rows and scientific results are never cached.
"""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import math
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

from openpyxl import Workbook
from openpyxl.compat import safe_string
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter


_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_SHEET = "xl/worksheets/sheet1.xml"
_CORE = "docProps/core.xml"
_DATE_TAGS = {"{http://purl.org/dc/terms/}created", "{http://purl.org/dc/terms/}modified"}


def _parse_xml(raw):
    # Generated XML has no declarations/CDATA/comments/PIs beyond its XML prolog.
    declaration_end = raw.find(b"?>") + 2 if raw.startswith(b"<?xml ") else 0
    if b"\x00" in raw or b"<!" in raw or raw.find(b"<?", declaration_end) >= 0:
        raise ValueError("Unexpected XML metadata in generated workbook")
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
    return ET.fromstring(raw, parser=parser)


def _core_without_dates(raw):
    root = _parse_xml(raw)
    for child in root:
        if child.tag in _DATE_TAGS:
            child.text = ""
    return ET.tostring(root)


@lru_cache(maxsize=64)
def _layout_template(width, height):
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "LIGHT2"
    worksheet.append(["header"] * width)
    worksheet.append([0] * width)
    worksheet.freeze_panes = "A2"
    reference = f"A1:{get_column_letter(width)}{height}"
    worksheet.auto_filter.ref = reference
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
    buffer = BytesIO()
    try:
        workbook.save(buffer)
    finally:
        workbook.close()
    with ZipFile(buffer) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
        metadata = {item.filename: (item.compress_type, item.create_system,
                    item.internal_attr, item.external_attr, item.flag_bits,
                    item.comment, item.extra) for item in archive.infolist()}
    layout = _parse_xml(parts[_SHEET])
    layout.find(_NS + "dimension").set("ref", reference)
    layout.remove(layout.find(_NS + "sheetData"))
    parts[_SHEET] = ET.tostring(layout)
    parts[_CORE] = _core_without_dates(parts[_CORE])
    return parts, metadata


def _canonical_cell(cell, value, column, row):
    if value is not None and not isinstance(value, (str, float)):
        return False
    if cell.text is not None or cell.tail is not None:
        return False
    attributes = {"r": f"{get_column_letter(column)}{row}"}
    if row == 1:
        attributes["s"] = "1"
    attributes["t"] = "inlineStr" if isinstance(value, str) else "n"
    if cell.tag != _NS + "c" or cell.attrib != attributes:
        return False
    children = list(cell)
    if value is None or value == "":
        return not children and cell.text is None
    if isinstance(value, float):
        return (math.isfinite(value) and len(children) == 1
                and children[0].tag == _NS + "v" and not children[0].attrib
                and children[0].text == safe_string(value)
                and children[0].tail is None and not list(children[0]))
    if (len(children) != 1 or children[0].tag != _NS + "is" or children[0].attrib
            or children[0].text is not None or children[0].tail is not None):
        return False
    content = list(children[0])
    expected = {"{http://www.w3.org/XML/1998/namespace}space": "preserve"} if value != value.strip() else {}
    return (len(content) == 1 and content[0].tag == _NS + "t"
            and content[0].attrib == expected and content[0].text == value
            and content[0].tail is None and not list(content[0]))


def matches_generated_light_workbook(path, expected_rows):
    """Require the exact current writer output, preserving non-default exports.

    Core creation/modified dates may remain as on the existing no-op path.
    File values are compared to writer tokens, never rounded for analysis.
    """
    height = len(expected_rows)
    width = max(map(len, expected_rows), default=0)
    if width < 1 or height < 1:
        return False
    parts, metadata = _layout_template(width, height)
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            if archive.comment or len(names) != len(set(names)) or set(names) != set(parts):
                return False
            for item in archive.infolist():
                current = (item.compress_type, item.create_system, item.internal_attr,
                           item.external_attr, item.flag_bits, item.comment, item.extra)
                if current != metadata[item.filename]:
                    return False
            for name, expected in parts.items():
                raw = archive.read(name)
                if name == _CORE:
                    raw = _core_without_dates(raw)
                elif name == _SHEET:
                    sheet = _parse_xml(raw)
                    data = sheet.find(_NS + "sheetData")
                    if (data is None or data.attrib or len(data) != height
                            or data.text is not None or data.tail is not None):
                        return False
                    for number, (row, values) in enumerate(zip(data, expected_rows), start=1):
                        if (row.tag != _NS + "row" or row.attrib != {"r": str(number)}
                                or row.text is not None or row.tail is not None):
                            return False
                        values = values + [None] * (width - len(values))
                        cells = list(row)
                        columns = [(index, value) for index, value in enumerate(values, start=1)
                                   if number == 1 or value is not None]
                        if len(cells) != len(columns):
                            return False
                        if not all(_canonical_cell(cell, value, column, number)
                                   for cell, (column, value) in zip(cells, columns)):
                            return False
                    sheet.remove(data)
                    raw = ET.tostring(sheet)
                if raw != expected:
                    return False
    except (OSError, ValueError, KeyError, BadZipFile, ET.ParseError):
        return False
    return True
