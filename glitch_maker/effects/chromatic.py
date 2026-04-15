"""Chromatic aberration — per-channel sub-pixel RGB displacement.

Shifts R and B channels relative to G, with amounts driven by audio. Treble
energises the offset; bass slows the drift. Mid channel (G) stays put as the
luminance anchor so motion is preserved.
"""
from __future__ import annotations
import numpy as np


class Chromatic:
    def __init__(self):
        self.phase = 0.0
        self.direction = 1

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        if features.get('transient', False):
            self.direction *= -1

        self.phase += 0.12 + features.get('bass', 0.0) * 0.35
        amp = 4 + features.get('treble', 0.0) * 28 + features.get('rms', 0.0) * 8
        dx = int(self.direction * amp * np.cos(self.phase))
        dy = int(self.direction * amp * np.sin(self.phase) * 0.35)

        out = frame.copy()
        r = frame[:, :, 0]
        b = frame[:, :, 2]
        out[:, :, 0] = np.roll(r, shift=(dy, dx), axis=(0, 1))
        out[:, :, 2] = np.roll(b, shift=(-dy, -dx), axis=(0, 1))
        return out
