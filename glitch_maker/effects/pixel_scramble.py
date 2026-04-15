"""Pixel scramble — shuffle pixel positions within random blocks.

For each random block, pixels are randomly reordered. Each block keeps its
average color but loses all structure — reads as localized "data loss"
against any content, busy or sparse. Audio-reactive block count and size.
"""
from __future__ import annotations
import numpy as np


class PixelScramble:
    def __init__(self, count_pct: float = 3.0, size_pct: float = 15.0):
        self.count_pct = float(np.clip(count_pct, 0.0, 100.0))
        self.size_pct = float(np.clip(size_pct, 0.0, 100.0))
        self.rng = np.random.default_rng()

    def _activity(self, features: dict) -> float:
        if features.get('section_change', False):
            return 1.0
        base = (features.get('rms', 0.0) * 0.5
                + features.get('bass', 0.0) * 0.3
                + features.get('onset', 0.0) * 0.2)
        if features.get('transient', False):
            base += 0.3
        return float(np.clip(base, 0.0, 1.0))

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        H, W, _ = frame.shape
        activity = self._activity(features)
        if activity == 0.0:
            return frame

        axis = min(H, W)
        count = max(0, int(H * self.count_pct / 100.0 * activity))
        max_size = max(2, int(axis * self.size_pct / 100.0 * activity))
        if count == 0:
            return frame

        out = frame.copy()
        for _ in range(count):
            h = int(self.rng.integers(8, max(9, max_size + 1)))
            w = int(self.rng.integers(8, max(9, max_size + 1)))
            y = int(self.rng.integers(0, max(1, H - h)))
            x = int(self.rng.integers(0, max(1, W - w)))
            block = out[y:y + h, x:x + w]
            flat = block.reshape(-1, 3)
            self.rng.shuffle(flat)  # in-place shuffle along axis 0
            out[y:y + h, x:x + w] = flat.reshape(block.shape)
        return out
