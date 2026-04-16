"""Warp — sine-wave displacement map across the frame.

Each row (or column) is shifted along the perpendicular axis by a sine wave
of configurable frequency, amplitude, and speed. Produces underwater /
heat-haze / analog-tape wobble.

direction=horizontal: rows shift left-right based on sin(y * freq + t).
direction=vertical:   columns shift up-down based on sin(x * freq + t).
direction=both:       applies both (can be intense).
"""
from __future__ import annotations
import numpy as np


class Warp:
    def __init__(self,
                 amplitude_px: float = 20.0,
                 frequency: float = 0.02,
                 speed: float = 3.0,
                 direction: str = 'horizontal',
                 audio_scale: float = 0.5):
        if direction not in ('horizontal', 'vertical', 'both'):
            raise ValueError(f'direction must be horizontal/vertical/both, got {direction!r}')
        self.amplitude_px = float(max(0.0, amplitude_px))
        self.frequency = float(frequency)
        self.speed = float(speed)
        self.direction = direction
        self.audio_scale = float(np.clip(audio_scale, 0.0, 1.0))
        self.phase = 0.0

    def _activity(self, features: dict) -> float:
        if features.get('section_change', False):
            return 1.0
        base = (features.get('rms', 0.0) * 0.5
                + features.get('bass', 0.0) * 0.3
                + features.get('onset', 0.0) * 0.2)
        return float(np.clip(base, 0.0, 1.0))

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        activity = self._activity(features)
        self.phase += self.speed * 0.03 * (1.0 + 0.5 * activity)
        intensity = (1.0 - self.audio_scale) + self.audio_scale * activity
        amp = self.amplitude_px * intensity
        if amp < 1.0:
            return frame

        H, W, _ = frame.shape
        out = frame

        if self.direction in ('horizontal', 'both'):
            yy = np.arange(H)
            dx = (amp * np.sin(self.frequency * yy + self.phase)).astype(int)
            shifted = out.copy()
            for y in range(H):
                if dx[y]:
                    shifted[y] = np.roll(out[y], dx[y], axis=0)
            out = shifted

        if self.direction in ('vertical', 'both'):
            xx = np.arange(W)
            dy = (amp * np.sin(self.frequency * xx + self.phase * 1.1)).astype(int)
            shifted = out.copy()
            for x in range(W):
                if dy[x]:
                    shifted[:, x] = np.roll(out[:, x], dy[x], axis=0)
            out = shifted

        return out
