"""Shared frame primitives for glitch effects.

Frames are ndarray[H, W, 3] uint8 RGB. Effects may mutate in-place or return
a new array; render.py treats the returned value as authoritative.
"""
from __future__ import annotations
import numpy as np


def luminance(frame: np.ndarray) -> np.ndarray:
    """Return (H, W) float32 perceived brightness [0..1]."""
    r = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    b = frame[:, :, 2].astype(np.float32)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
