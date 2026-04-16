"""Channel swap — swap or rotate RGB channels within random blocks.

Transforms pixel VALUES rather than positions, so damage stays visible on
busy content (unlike tear which just moves pixels around).

Modes:
  rb      swap red and blue
  rg      swap red and green
  gb      swap green and blue
  rotate  rotate R->G->B->R (color wheel shift)
  random  pick any of the above per block (default)
"""
from __future__ import annotations
import numpy as np


_PERMUTATIONS = {
    'rb': [2, 1, 0],
    'rg': [1, 0, 2],
    'gb': [0, 2, 1],
    'rotate': [1, 2, 0],
}


class ChannelSwap:
    def __init__(self,
                 count_pct: float = 5.0,
                 size_pct: float = 15.0,
                 mode: str = 'random',
                 audio_scale: float = 0.7):
        if mode != 'random' and mode not in _PERMUTATIONS:
            raise ValueError(f'mode must be random or {list(_PERMUTATIONS)}, got {mode!r}')
        self.count_pct = float(np.clip(count_pct, 0.0, 100.0))
        self.size_pct = float(np.clip(size_pct, 0.0, 100.0))
        self.mode = mode
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
        modes = list(_PERMUTATIONS.keys())
        for _ in range(count):
            h = int(self.rng.integers(4, max_size + 1))
            w = int(self.rng.integers(4, max_size + 1))
            y = int(self.rng.integers(0, max(1, H - h)))
            x = int(self.rng.integers(0, max(1, W - w)))
            mode = self.mode if self.mode != 'random' else modes[int(self.rng.integers(0, 4))]
            perm = _PERMUTATIONS[mode]
            out[y:y + h, x:x + w] = out[y:y + h, x:x + w, perm]
        return out
