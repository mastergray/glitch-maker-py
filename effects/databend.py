"""Databend — raw byte-level corruption of frame data.

Picks N random byte ranges and corrupts them via configurable modes:
  shift — roll the chunk by a random offset
  xor   — XOR the chunk with a random bitmask
  swap  — swap the chunk with another chunk of the same size

Audio drives chunk count; transients widen chunk size for destructive bursts.
"""
from __future__ import annotations
import numpy as np


MODES = ('shift', 'xor', 'swap')


class Databend:
    def __init__(self,
                 count: int = 10,
                 chunk_size: int = 1024,
                 modes: tuple = MODES,
                 audio_scale: float = 0.8):
        self.count = max(1, int(count))
        self.chunk_size = max(16, int(chunk_size))
        if isinstance(modes, str):
            modes = tuple(m.strip() for m in modes.split(','))
        for m in modes:
            if m not in MODES:
                raise ValueError(f'unknown databend mode {m!r}; valid: {MODES}')
        self.modes = tuple(modes) if modes else MODES
        self.audio_scale = float(np.clip(audio_scale, 0.0, 1.0))
        self.rng = np.random.default_rng()

    def _activity(self, features: dict) -> float:
        base = (features.get('rms', 0.0) * 0.5
                + features.get('bass', 0.0) * 0.5)
        if features.get('transient', False):
            base += 0.3
        return float(np.clip(base, 0.0, 1.0))

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        activity = self._activity(features)
        # Count and chunk size scale with audio from (1-audio_scale) min to full.
        intensity = (1.0 - self.audio_scale) + self.audio_scale * activity
        n_chunks = max(1, int(self.count * intensity))
        chunk_size = max(16, int(self.chunk_size * intensity))
        if features.get('transient', False):
            chunk_size *= 2

        flat = frame.reshape(-1).copy()
        total = flat.size

        for _ in range(n_chunks):
            start = int(self.rng.integers(0, max(1, total - chunk_size)))
            end = min(total, start + chunk_size)
            mode = self.modes[int(self.rng.integers(0, len(self.modes)))]
            if mode == 'shift':
                off = int(self.rng.integers(1, 64))
                flat[start:end] = np.roll(flat[start:end], off)
            elif mode == 'xor':
                mask = np.uint8(self.rng.integers(1, 64))
                flat[start:end] ^= mask
            else:  # swap
                start2 = int(self.rng.integers(0, max(1, total - (end - start))))
                end2 = start2 + (end - start)
                tmp = flat[start:end].copy()
                flat[start:end] = flat[start2:end2]
                flat[start2:end2] = tmp

        return flat.reshape(frame.shape)
