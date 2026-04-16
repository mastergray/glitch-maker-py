"""Bit crush — quantize color depth in random blocks.

Masks off the low bits of each channel so colors collapse to 2^bits values
per channel. Produces harsh banded color zones against the full-color
untouched regions. Classic destructive retro-digital look.

bits=1 -> 8 total colors, bits=3 -> 512, bits=6 -> 262k (subtle).
"""
from __future__ import annotations
import numpy as np

from ._frame import pick_block_masks, write_block


class BitCrush:
    def __init__(self,
                 count_pct: float = 3.0,
                 size_pct: float = 20.0,
                 bits: int = 2,
                 audio_scale: float = 0.6,
                 shape: str = 'rectangle',
                 orientation: str = 'horizontal',
                 delay_frames: int = 1):
        self.count_pct = float(np.clip(count_pct, 0.0, 100.0))
        self.size_pct = float(np.clip(size_pct, 0.0, 100.0))
        self.bits = int(np.clip(bits, 1, 7))
        self.audio_scale = float(np.clip(audio_scale, 0.0, 1.0))
        self.shape = shape
        self.orientation = orientation
        self.delay_frames = max(1, int(delay_frames))
        keep_bits = 8 - self.bits
        self.mask = np.uint8((0xFF << keep_bits) & 0xFF) if self.bits < 8 else np.uint8(0xFF)
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
                                  orientation=self.orientation)
        if not blocks:
            return frame

        out = frame.copy()
        for bbox, mask in blocks:
            y0, y1, x0, x1 = bbox
            crushed = out[y0:y1, x0:x1] & self.mask
            write_block(out, bbox, mask, crushed)
        return out
