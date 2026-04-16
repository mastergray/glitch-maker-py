"""Noise — per-pixel random value perturbation across the whole frame.

Unlike block-based corruption effects, noise affects every pixel uniformly.
Used for film grain / analog TV static / subtle texturing.

type=gaussian  zero-mean normal distribution (soft, grain-like)
type=uniform   flat distribution (harsher, punchier)

mono=True      same noise value applied to R, G, B (luminance-only grain)
mono=False     independent per-channel noise (colored TV static)
"""
from __future__ import annotations
import numpy as np


class Noise:
    def __init__(self,
                 amount: float = 15.0,
                 type: str = 'gaussian',
                 mono: bool = True,
                 audio_scale: float = 0.4):
        if type not in ('gaussian', 'uniform'):
            raise ValueError(f'type must be gaussian/uniform, got {type!r}')
        self.amount = float(max(0.0, amount))
        self.type = type
        self.mono = bool(mono)
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
        amp = self.amount * intensity
        if amp < 0.5:
            return frame

        H, W, _ = frame.shape
        shape = (H, W) if self.mono else (H, W, 3)
        if self.type == 'gaussian':
            noise = self.rng.normal(0, amp, size=shape).astype(np.int16)
        else:  # uniform
            ai = int(amp)
            noise = self.rng.integers(-ai, ai + 1, size=shape).astype(np.int16)

        if self.mono:
            noise = noise[:, :, None]  # broadcast across channels

        out = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return out
