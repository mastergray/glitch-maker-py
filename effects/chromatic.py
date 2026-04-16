"""Chromatic aberration — per-channel sub-pixel RGB displacement.

Shifts R and B channels relative to G. Direction of the drift is configurable
(horizontal only / vertical only / circular orbit). Amplitude scales with
audio activity; audio_scale controls how much of the max amplitude is
audio-driven vs. always-on.
"""
from __future__ import annotations
import numpy as np


class Chromatic:
    def __init__(self,
                 amount: float = 16.0,
                 direction: str = 'circular',
                 audio_scale: float = 0.75):
        if direction not in ('horizontal', 'vertical', 'circular'):
            raise ValueError(f'direction must be horizontal/vertical/circular, got {direction!r}')
        self.amount = float(max(0.0, amount))
        self.direction = direction
        self.audio_scale = float(np.clip(audio_scale, 0.0, 1.0))
        self.phase = 0.0
        self.sign = 1

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        if features.get('transient', False):
            self.sign *= -1

        self.phase += 0.12 + features.get('bass', 0.0) * 0.35

        # Audio scales from (1 - audio_scale) minimum up to full amount.
        audio = (features.get('treble', 0.0) * 0.6 + features.get('rms', 0.0) * 0.4)
        intensity = (1.0 - self.audio_scale) + self.audio_scale * audio
        amp = self.amount * intensity

        if self.direction == 'horizontal':
            dx = int(self.sign * amp * np.cos(self.phase))
            dy = 0
        elif self.direction == 'vertical':
            dx = 0
            dy = int(self.sign * amp * np.cos(self.phase))
        else:  # circular
            dx = int(self.sign * amp * np.cos(self.phase))
            dy = int(self.sign * amp * np.sin(self.phase) * 0.5)

        out = frame.copy()
        r = frame[:, :, 0]
        b = frame[:, :, 2]
        out[:, :, 0] = np.roll(r, shift=(dy, dx), axis=(0, 1))
        out[:, :, 2] = np.roll(b, shift=(-dy, -dx), axis=(0, 1))
        return out
