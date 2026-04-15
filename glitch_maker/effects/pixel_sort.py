"""Pixel-sort — per-row sort of contiguous pixel runs above a brightness threshold.

Classic Kim Asendorf algorithm: walk each row; whenever brightness crosses the
threshold, sort the run of pixels until it drops back. Audio drives threshold
(lower = more aggressive) and can invert the comparison on transients.
"""
from __future__ import annotations
import numpy as np

from ._frame import luminance


class PixelSort:
    def __init__(self):
        self.invert = False

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        if features.get('transient', False):
            self.invert = not self.invert

        # Lower threshold on heavy bass = more pixels get sorted.
        thresh = 0.65 - features.get('bass', 0.0) * 0.35
        thresh = max(0.05, min(0.95, thresh))

        lum = luminance(frame)
        H, W = lum.shape
        mask = (lum < thresh) if self.invert else (lum > thresh)

        out = frame.copy()
        # Sort each row's runs by luminance. Vectorised per-row for speed.
        for y in range(H):
            row_mask = mask[y]
            row_lum = lum[y]
            i = 0
            while i < W:
                if row_mask[i]:
                    j = i
                    while j < W and row_mask[j]:
                        j += 1
                    if j - i > 1:
                        order = np.argsort(row_lum[i:j])
                        out[y, i:j] = out[y, i:j][order]
                    i = j
                else:
                    i += 1
        return out
