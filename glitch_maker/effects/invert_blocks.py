"""Invert blocks — negate color (255 - value) inside random regions."""
from __future__ import annotations
import numpy as np

from ._frame import pick_block_masks, write_block


class InvertBlocks:
    def __init__(self,
                 count_pct: float = 3.0,
                 size_pct: float = 20.0,
                 audio_scale: float = 0.6,
                 shape: str = 'rectangle',
                 orientation: str = 'horizontal',
                 band_length_pct: float = 100.0,
                 delay_frames: int = 1):
        self.count_pct = float(np.clip(count_pct, 0.0, 100.0))
        self.size_pct = float(np.clip(size_pct, 0.0, 100.0))
        self.audio_scale = float(np.clip(audio_scale, 0.0, 1.0))
        self.shape = shape
        self.orientation = orientation
        self.band_length_pct = float(np.clip(band_length_pct, 1.0, 100.0))
        self.delay_frames = max(1, int(delay_frames))
        self.rng = np.random.default_rng()
        self._frame_idx = 0

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
        self._frame_idx += 1
        if self._frame_idx % self.delay_frames != 0:
            return frame

        activity = self._activity(features)
        intensity = (1.0 - self.audio_scale) + self.audio_scale * activity
        H, W, _ = frame.shape
        count = max(0, int(H * self.count_pct / 100.0 * intensity))
        max_size = int(min(H, W) * self.size_pct / 100.0 * intensity)
        blocks = pick_block_masks(self.shape, count, H, W, max_size, self.rng,
                                  orientation=self.orientation,
                                  band_length_pct=self.band_length_pct)
        if not blocks:
            return frame

        out = frame.copy()
        for bbox, mask in blocks:
            y0, y1, x0, x1 = bbox
            inverted = 255 - out[y0:y1, x0:x1]
            write_block(out, bbox, mask, inverted)
        return out
