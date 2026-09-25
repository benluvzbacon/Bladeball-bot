"""Minimal PNG encoder (NumPy + zlib) so the web menu can show camera previews
without pulling in Pillow or OpenCV."""

from __future__ import annotations

import struct
import zlib

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def encode_png(rgb: np.ndarray, level: int = 1) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 RGB image (or ``(H, W)`` grayscale) as PNG bytes."""
    img = np.ascontiguousarray(rgb, dtype=np.uint8)
    if img.ndim == 2:
        color_type, channels = 0, 1
        img = img[:, :, None]
    elif img.ndim == 3 and img.shape[2] == 3:
        color_type, channels = 2, 3
    else:
        raise ValueError("expected an (H, W) or (H, W, 3) array")
    h, w = img.shape[:2]
    raw = np.empty((h, 1 + w * channels), dtype=np.uint8)
    raw[:, 0] = 0  # filter type "None" for every scanline
    raw[:, 1:] = img.reshape(h, w * channels)
    header = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw.tobytes(), level))
        + _chunk(b"IEND", b"")
    )
