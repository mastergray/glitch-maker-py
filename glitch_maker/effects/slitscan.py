"""Slitscan — each row (or column) shows content from a different past frame.

Keeps a rolling buffer of the last N frames. For a horizontal slitscan, row y
is sampled from a frame `delay(y)` frames in the past, producing time-warped
motion smearing. Default is linear delay top-to-bottom (top row = newest,
bottom row = oldest).

Requires history, so the first N frames act as passthrough until the buffer
fills. This effect is stateful across frames; be mindful when stacking with
--accumulate.
"""
from __future__ import annotations
from collections import deque

import numpy as np


class Slitscan:
    def __init__(self,
                 direction: str = 'horizontal',
                 max_delay_frames: int = 30,
                 audio_scale: float = 0.5):
        if direction not in ('horizontal', 'vertical'):
            raise ValueError(f'direction must be horizontal/vertical, got {direction!r}')
        self.direction = direction
        self.max_delay_frames = max(1, int(max_delay_frames))
        self.audio_scale = float(np.clip(audio_scale, 0.0, 1.0))
        self.buffer: deque = deque(maxlen=self.max_delay_frames + 1)

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
        self.buffer.append(frame.copy())

        activity = self._activity(features)
        intensity = (1.0 - self.audio_scale) + self.audio_scale * activity
        effective_delay = int(self.max_delay_frames * intensity)
        effective_delay = min(effective_delay, len(self.buffer) - 1)
        if effective_delay <= 0:
            return frame

        H, W, _ = frame.shape
        out = frame.copy()
        frames = list(self.buffer)  # oldest first
        if self.direction == 'horizontal':
            # Top row = newest (delay 0), bottom row = oldest (delay = effective_delay).
            for y in range(H):
                d = int(effective_delay * y / max(1, H - 1))
                src = frames[-1 - d]
                out[y] = src[y]
        else:
            for x in range(W):
                d = int(effective_delay * x / max(1, W - 1))
                src = frames[-1 - d]
                out[:, x] = src[:, x]
        return out
