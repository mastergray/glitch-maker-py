"""Pixel scramble — shuffle pixel positions within random blocks."""
from __future__ import annotations
import numpy as np

from ._frame import pick_block_masks, write_block


class PixelScramble:
    def __init__(self,
                 count_pct: float = 3.0,
                 size_pct: float = 15.0,
                 shape: str = 'rectangle',
                 delay_frames: int = 1):
        self.count_pct = float(np.clip(count_pct, 0.0, 100.0))
        self.size_pct = float(np.clip(size_pct, 0.0, 100.0))
        self.shape = shape
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
        if activity == 0.0:
            return frame
        H, W, _ = frame.shape
        count = max(0, int(H * self.count_pct / 100.0 * activity))
        max_size = int(min(H, W) * self.size_pct / 100.0 * activity)
        blocks = pick_block_masks(self.shape, count, H, W, max_size, self.rng)
        if not blocks:
            return frame

        out = frame.copy()
        for bbox, mask in blocks:
            y0, y1, x0, x1 = bbox
            block = out[y0:y1, x0:x1].copy()
            flat = block.reshape(-1, 3)
            self.rng.shuffle(flat)
            write_block(out, bbox, mask, flat.reshape(block.shape))
        return out
