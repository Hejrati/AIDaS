"""Minimal, synthetic Heidelberg CMDb records without patient information."""

from __future__ import annotations

import struct

import numpy as np


def build_cmdb(chunks, *, directory_capacity=None):
    """Build one active directory, optionally followed by reserved blank slots."""
    chunks = list(chunks)
    capacity = len(chunks) if directory_capacity is None else directory_capacity
    if capacity < len(chunks):
        raise ValueError("Directory capacity must contain every chunk.")
    file_header = struct.pack("<12sI10H", b"CMDb", 100, *([0] * 10))
    main = struct.pack(
        "<12sI10H4I", b"MDbMDir", 100, *([0] * 10), capacity, 88, 0, 0
    )
    directory = struct.pack(
        "<12sI10H4I", b"MDbDir", 100, *([0] * 10), capacity, 88, 0, 0
    )
    entries = []
    bodies = []
    start = 140 + capacity * 44
    for index, chunk in enumerate(chunks):
        position = 140 + index * 44
        payload = chunk["payload"]
        kind = chunk.get("kind", 0x40000000)
        identity = chunk.get("ids", (1, 2, 3))
        slice_id = chunk.get("slice_id", 0)
        subtype = chunk.get("ind", 1)
        entries.append(struct.pack(
            "<7IiHHII", position, start, len(payload), 0, *identity,
            slice_id, subtype, 0, kind, 0
        ))
        body = struct.pack(
            "<12s8IiHHII", b"MDbData", 0, position, start, len(payload),
            0, *identity, slice_id, subtype, 0, kind, 0
        ) + payload
        bodies.append(body)
        start += len(body)
    entries.extend([b"\0" * 44] * (capacity - len(chunks)))
    return file_header + main + directory + b"".join(entries) + b"".join(bodies)


def oct_chunk(width, height, words=None, **identity):
    if words is None:
        words = np.full((height, width), 58 << 10, dtype=np.uint16)
    data = np.asarray(words, dtype="<u2").tobytes()
    if len(data) != width * height * 2:
        raise ValueError("OCT fixture words must match the requested dimensions.")
    payload = struct.pack(
        "<5I", 20 + len(data), 0x02200201, width * height, height, width
    ) + data
    return {"kind": 0x40000000, "ind": 1, "payload": payload, **identity}


def ir_chunk(width, height, values=None, **identity):
    if values is None:
        values = np.full((height, width), 127, dtype=np.uint8)
    data = np.asarray(values, dtype=np.uint8).tobytes()
    if len(data) != width * height:
        raise ValueError("IR fixture pixels must match the requested dimensions.")
    payload = struct.pack(
        "<5I", 20 + len(data), 0x02010201, width * height, height, width
    ) + data
    return {"kind": 0x40000000, "ind": 0, "payload": payload, **identity}


def metadata_chunk(width, height, spacing=0.00387167, **identity):
    payload = bytearray(40)
    struct.pack_into("<2I", payload, 4, height, width)
    struct.pack_into("<f", payload, 36, spacing)
    return {"kind": 10004, "ind": 1, "payload": bytes(payload), **identity}
