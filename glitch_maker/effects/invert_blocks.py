"""Invert blocks — negate color (255 - value) inside random rectangles.

Stark tonal flip against untouched regions. Pairs well with other block-
based effects for layered corruption patches.
"""
from __future__ import annotations
import numpy as np


class InvertBlocks:
    def __init__(self,
                 count_pct: float = 3.0,
                 size_pct: float = 20.0,
                 audio_scale: float = 0.6):
        self.count_pct = float(np.clip(count_pct, 0.0, 100.0))
        self.size_pct = float(np.clip(size_pct, 0.0, 100.0))
        self.audio_scale = float(np.clip(audio_scale, 0.0, 1.0))
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
        activity = self._activity(features)
        intensity = (1.0 - self.audio_scale) + self.audio_scale * activity
        H, W, _ = frame.shape
        count = max(0, int(H * self.count_pct / 100.0 * intensity))
        max_size = int(min(H, W) * self.size_pct / 100.0 * intensity)
        if count == 0 or max_size < 4:
            return frame

        out = frame.copy()
        for _ in range(count):
            h = int(self.rng.integers(4, max_size + 1))
            w = int(self.rng.integers(4, max_size + 1))
            y = int(self.rng.integers(0, max(1, H - h)))
            x = int(self.rng.integers(0, max(1, W - w)))
            out[y:y + h, x:x + w] = 255 - out[y:y + h, x:x + w]
        return out
