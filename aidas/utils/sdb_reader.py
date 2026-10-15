"""Read directory-referenced Heidelberg CMDb OCT images without raw guesses.

CMDb records use the same documented chunk layout as Heidelberg E2E files.
Image dimensions belong to each image record; companion EDB metadata is only
accepted when its database, series, slice and image-subtype keys also match.
No patient text records are interpreted or exposed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import struct
from typing import BinaryIO

import numpy as np


_DIRECTORY = struct.Struct("<12sI10H4I")
_ENTRY = struct.Struct("<7IiHHII")
_CHUNK = struct.Struct("<12s8IiHHII")
_IMAGE = struct.Struct("<5I")
_IMAGE_TYPE = 0x40000000
_BSCAN_METADATA_TYPE = 10004
_CONTOUR_TYPE = 10019  # 0x2723; both legacy and current header layouts occur.


@dataclass(frozen=True)
class SdbImageInfo:
    width: int
    height: int
    offset: int
    frame_count: int
    file_size: int
    little_endian: bool = True
    bit_depth: int = 16
    edb_status: str = "SDB header"
    axial_spacing_mm: float | None = None
    _identity: tuple[int, ...] | None = field(default=None, repr=False)

    def import_params(self) -> dict:
        """Return the parameters used by Step 1's existing import controls."""
        return {
            "width": self.width,
            "height": self.height,
            "offset": self.offset,
            "bit_depth": self.bit_depth,
            "little_endian": self.little_endian,
        }


@dataclass(frozen=True)
class _Record:
    start: int
    size: int
    kind: int
    identity: tuple[int, ...]

    @property
    def payload(self) -> int:
        return self.start + _CHUNK.size


@dataclass(frozen=True)
class _OctImage:
    width: int
    height: int
    offset: int
    identity: tuple[int, ...]


def _read_at(stream: BinaryIO, file_size: int, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0 or offset + size > file_size:
        raise ValueError("Truncated Heidelberg CMDb record or invalid file pointer.")
    stream.seek(offset)
    raw = stream.read(size)
    if len(raw) != size:
        raise ValueError("Truncated Heidelberg CMDb record.")
    return raw


def _magic(raw: bytes, expected: bytes) -> bool:
    return raw == expected.ljust(12, b"\0")


def _records(stream: BinaryIO, file_size: int) -> list[_Record] | None:
    """Validate the active directory chain and its referenced records only."""
    stream.seek(0)
    if stream.read(4) != b"CMDb":
        return None
    header = _read_at(stream, file_size, 0, 36)
    if not _magic(header[:12], b"CMDb"):
        raise ValueError("Invalid Heidelberg CMDb file header.")
    main = _DIRECTORY.unpack(_read_at(stream, file_size, 36, _DIRECTORY.size))
    if not _magic(main[0], b"MDbMDir"):
        raise ValueError("Invalid Heidelberg CMDb main directory.")
    current = main[-3]
    visited: set[int] = set()
    directory_regions: list[tuple[int, int]] = [(0, 88)]
    entries: list[tuple[int, tuple]] = []
    while current:
        if current in visited:
            raise ValueError("Heidelberg CMDb directory cycle.")
        visited.add(current)
        if current < 88:
            raise ValueError("Invalid Heidelberg CMDb directory pointer.")
        directory = _DIRECTORY.unpack(
            _read_at(stream, file_size, current, _DIRECTORY.size)
        )
        if not _magic(directory[0], b"MDbDir") or directory[-3] != current:
            raise ValueError("Invalid Heidelberg CMDb directory position.")
        count = directory[-4]
        table_start = current + _DIRECTORY.size
        table_size = count * _ENTRY.size
        # Check before allocating or iterating a corrupt directory count.
        if table_start + table_size > file_size:
            raise ValueError("Truncated Heidelberg CMDb directory entries.")
        region = (current, table_start + table_size)
        if any(region[0] < end and region[1] > start for start, end in directory_regions):
            raise ValueError("Overlapping Heidelberg CMDb directories.")
        directory_regions.append(region)
        for index in range(count):
            position = table_start + index * _ENTRY.size
            entry = _ENTRY.unpack(_read_at(stream, file_size, position, _ENTRY.size))
            # Reserved slots and deleted entries do not reference active data.
            if entry[1] <= entry[0]:
                continue
            if entry[0] != position:
                raise ValueError("Invalid Heidelberg CMDb directory entry position.")
            entries.append((position, entry))
        current = directory[-2]

    records: list[_Record] = []
    seen_starts: set[int] = set()
    for position, entry in entries:
        start, size = entry[1], entry[2]
        end = start + _CHUNK.size + size
        if any(start < region_end and end > region_start
               for region_start, region_end in directory_regions):
            raise ValueError("Heidelberg CMDb data overlaps its directory.")
        chunk = _CHUNK.unpack(_read_at(stream, file_size, start, _CHUNK.size))
        if not _magic(chunk[0], b"MDbData"):
            raise ValueError("Invalid Heidelberg CMDb chunk header.")
        # Chunk unknown2 is its referring directory entry; pos is self-position.
        if chunk[2] != position or chunk[3] != start or chunk[4] != size:
            raise ValueError("Heidelberg CMDb chunk position or length mismatch.")
        identity = tuple(chunk[6:11])
        if identity != tuple(entry[4:9]) or chunk[-2] != entry[-2]:
            raise ValueError("Heidelberg CMDb chunk identity or type mismatch.")
        if end > file_size:
            raise ValueError("Truncated Heidelberg CMDb chunk payload.")
        if start in seen_starts:
            raise ValueError("Duplicate Heidelberg CMDb chunk reference.")
        seen_starts.add(start)
        records.append(_Record(start, size, chunk[-2], identity))
    ranges = sorted((record.start, record.payload + record.size) for record in records)
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("Overlapping Heidelberg CMDb data chunks.")
    return records


def _oct_images(stream: BinaryIO, file_size: int, records: list[_Record]) -> list[_OctImage]:
    images = []
    for record in records:
        if record.kind != _IMAGE_TYPE or record.identity[-1] != 1:
            continue
        if record.size < _IMAGE.size:
            raise ValueError("Truncated Heidelberg OCT image header.")
        size, image_type, count, height, width = _IMAGE.unpack(
            _read_at(stream, file_size, record.payload, _IMAGE.size)
        )
        if width == 0 or height == 0:
            raise ValueError("Invalid Heidelberg OCT image dimensions.")
        if image_type != 0x02200201:
            raise ValueError("Unsupported Heidelberg OCT pixel encoding.")
        if count != width * height or size != record.size or size != 20 + count * 2:
            raise ValueError("Heidelberg OCT image dimensions and payload length disagree.")
        images.append(_OctImage(width, height, record.payload + 20, record.identity))
    # Series/slice order is independent of physical chunk and directory order.
    images.sort(key=lambda image: (image.identity, image.offset))
    return images


def _edb_confirmation(path: Path, image: _OctImage) -> tuple[str, float | None]:
    try:
        companions = sorted(
            (item for item in path.parent.iterdir()
             if item.is_file() and item.stem.casefold() == path.stem.casefold()
             and item.suffix.casefold() == ".edb"),
            key=lambda item: item.name,
        )
        if not companions:
            return "SDB header", None
        if len(companions) != 1:
            return "SDB header; EDB ambiguous", None
        companion = companions[0]
        with companion.open("rb") as stream:
            file_size = companion.stat().st_size
            records = _records(stream, file_size)
            if records is None:
                return "SDB header; EDB unreadable", None
            matching = [record for record in records
                        if record.kind == _BSCAN_METADATA_TYPE
                        and record.identity == image.identity]
            if not matching:
                return "SDB header; EDB unrelated", None
            spacing = None
            for record in matching:
                if record.size < 40:
                    return "SDB header; EDB unreadable", None
                raw = _read_at(stream, file_size, record.payload, 40)
                height, width = struct.unpack_from("<2I", raw, 4)
                if (width, height) != (image.width, image.height):
                    return "SDB header; EDB mismatch", None
                axial = struct.unpack_from("<f", raw, 36)[0]
                if math.isfinite(axial) and axial > 0:
                    spacing = axial
            return "SDB/EDB confirmed", spacing
    except (OSError, ValueError, struct.error):
        return "SDB header; EDB unreadable", None


def detect_sdb_image(path: str | Path) -> SdbImageInfo | None:
    """Detect a native OCT image; return None only for unrecognized raw files.

    A recognized but malformed CMDb file raises ValueError, so it cannot silently
    fall through to a guessed raw import. The SDB image header is authoritative;
    a bad or unrelated companion produces a diagnostic instead of changing it.
    """
    path = Path(path)
    with path.open("rb") as stream:
        file_size = path.stat().st_size
        records = _records(stream, file_size)
        if records is None:
            return None
        images = _oct_images(stream, file_size, records)
    if not images:
        raise ValueError("No supported OCT image was found in the Heidelberg SDB file.")
    image = images[0]
    status, spacing = _edb_confirmation(path, image)
    return SdbImageInfo(
        width=image.width, height=image.height, offset=image.offset,
        frame_count=len(images), file_size=file_size,
        edb_status=status, axial_spacing_mm=spacing, _identity=image.identity,
    )


def read_sdb_pixels(
    path: str | Path,
    info: SdbImageInfo,
    *,
    width: int | None = None,
    height: int | None = None,
    offset: int | None = None,
    little_endian: bool | None = None,
) -> np.ndarray:
    """Read original unsigned OCT words using the existing raw import behavior.

    Adjustments must stay entirely inside this OCT frame's pixel payload. They
    cannot include an IR image, metadata, neighboring frames, or zero padding.
    Only byte order is interpreted; no intensity decoding, normalization,
    inversion, or scaling is applied to the stored pixel values.
    """
    width = info.width if width is None else int(width)
    height = info.height if height is None else int(height)
    offset = info.offset if offset is None else int(offset)
    little_endian = info.little_endian if little_endian is None else little_endian
    if width <= 0 or height <= 0:
        raise ValueError("OCT width and height must be positive.")
    with Path(path).open("rb") as stream:
        file_size = stream.seek(0, 2)
        if file_size != info.file_size:
            raise ValueError("The SDB file changed after image detection; reopen it.")
        records = _records(stream, file_size)
        if records is None:
            raise ValueError("This is not a native Heidelberg SDB image.")
        images = _oct_images(stream, file_size, records)
        matching = [image for image in images
                    if image.offset == info.offset
                    and (image.width, image.height) == (info.width, info.height)
                    and (info._identity is None or image.identity == info._identity)]
        if not matching:
            raise ValueError("The SDB image header changed after detection; reopen it.")
        payload_end = info.offset + info.width * info.height * 2
        if offset < info.offset or offset + width * height * 2 > payload_end:
            raise ValueError("Import parameters extend outside the detected OCT pixel payload.")
        if (offset - info.offset) % 2:
            raise ValueError("The OCT pixel offset must align to a 16-bit pixel.")
        raw = _read_at(stream, file_size, offset, width * height * 2)
    words = np.frombuffer(raw, dtype="<u2" if little_endian else ">u2")
    if not words.dtype.isnative:
        words = words.astype(words.dtype.newbyteorder("="))
    return words.reshape(height, width)


def read_sdb_image(
    path: str | Path,
    info: SdbImageInfo,
    *,
    little_endian: bool | None = None,
) -> np.ndarray:
    """Read original pixels using detected dimensions and the chosen byte order."""
    return read_sdb_pixels(path, info, little_endian=little_endian)


def read_sdb_retinal_contours(
    path: str | Path, info: SdbImageInfo,
) -> np.ndarray | None:
    """Return matching EDB ILM/BM coordinates for a conservative crop suggestion.

    Rows are ILM (Heidelberg layer 0) and Bruch's membrane (layer 1), in the
    original SDB image coordinates. Missing annotation points remain NaN and
    must not be treated as evidence that the image's lateral edges are blank.
    No image pixels are read, transformed, or replaced. Unsupported, incomplete,
    conflicting, or unrelated metadata simply provides no crop hint.

    Anatomy IDs follow eyepy's E2E reader and Heidelberg SEG_MAPPING. Contour
    payloads use the 16-byte legacy header or the 36-byte padded header; the
    selected frame must also have matching EDB B-scan dimensions and identity.
    """
    path = Path(path)
    if info._identity is None:
        return None
    try:
        # Revalidate the selected frame before attaching companion coordinates.
        with path.open("rb") as stream:
            file_size = path.stat().st_size
            if file_size != info.file_size:
                return None
            records = _records(stream, file_size)
            if records is None:
                return None
            images = [image for image in _oct_images(stream, file_size, records)
                      if image.identity == info._identity]
            if len(images) != 1:
                return None
            image = images[0]
            if ((image.width, image.height, image.offset)
                    != (info.width, info.height, info.offset)):
                return None

        companions = [item for item in path.parent.iterdir()
                      if item.is_file()
                      and item.stem.casefold() == path.stem.casefold()
                      and item.suffix.casefold() == ".edb"]
        if len(companions) != 1:
            return None
        companion = companions[0]
        with companion.open("rb") as stream:
            file_size = companion.stat().st_size
            records = _records(stream, file_size)
            if records is None:
                return None
            matching = [record for record in records
                        if record.identity == info._identity]
            metadata = [record for record in matching
                        if record.kind == _BSCAN_METADATA_TYPE]
            if not metadata:
                return None
            for record in metadata:
                if record.size < 40:
                    return None
                raw = _read_at(stream, file_size, record.payload, 40)
                height, width = struct.unpack_from("<2I", raw, 4)
                if (width, height) != (info.width, info.height):
                    return None

            by_layer: dict[int, dict[int, list[_Record]]] = {0: {}, 1: {}}
            for record in matching:
                if record.kind != _CONTOUR_TYPE:
                    continue
                if record.size < 16:
                    return None
                header = _read_at(stream, file_size, record.payload, 16)
                _, layer_id, _, width = struct.unpack("<4I", header)
                if layer_id in by_layer:
                    header_size = record.size - width * 4
                    if header_size not in (16, 36):
                        return None
                    by_layer[layer_id].setdefault(header_size, []).append(record)

            contours = []
            for layer_id in (0, 1):
                candidates = by_layer[layer_id]
                # Padded current segmentation supersedes flat legacy data.
                # Both layouts have decimal type 10019 (hexadecimal 0x2723).
                header_size = 36 if 36 in candidates else 16
                chosen = candidates.get(header_size)
                if not chosen:
                    return None
                accepted = None
                for record in chosen:
                    header = _read_at(stream, file_size, record.payload, 16)
                    if struct.unpack_from("<I", header, 12)[0] != info.width:
                        return None
                    if header_size == 36:
                        padding = _read_at(stream, file_size, record.payload + 16, 20)
                        if padding != b"\0" * 20:
                            return None
                    raw = _read_at(stream, file_size, record.payload + header_size,
                                   info.width * 4)
                    contour = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
                    invalid = (~np.isfinite(contour) | (contour < 0)
                               | (contour >= info.height))
                    if header_size == 16:
                        invalid |= contour < 1e-9
                    contour[invalid] = np.nan
                    if accepted is not None and not np.array_equal(
                        accepted, contour, equal_nan=True,
                    ):
                        return None
                    accepted = contour
                contours.append(accepted)
    except (OSError, ValueError, struct.error):
        return None

    result = np.stack(contours)
    valid = np.isfinite(result)
    joint = valid.all(axis=0)
    if (np.any(valid.mean(axis=1) < 0.80) or joint.mean() < 0.75
            or np.any(result[1, joint] - result[0, joint] < 1.0)):
        return None
    # Large holes inside the annotated span leave its outer envelope uncertain.
    columns = np.flatnonzero(joint)
    if columns.size > 1 and np.max(np.diff(columns)) - 1 > max(8, math.ceil(info.width * 0.05)):
        return None
    return result
