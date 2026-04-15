"""Databend — raw byte-level corruption of frame data.

Picks N random byte ranges and either shifts them by a small offset or XORs
them with a mask. Bass drives chunk count; transients widen chunk size for
sudden destructive bursts.
"""
from __future__ import annotations
import numpy as np


class Databend:
    def __init__(self):
        self.rng = np.random.default_rng()

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        flat = frame.reshape(-1).copy()
        total = flat.size

        base_chunks = 2 + int(features.get('bass', 0.0) * 18)
        chunk_size = 128 + int(features.get('rms', 0.0) * 2048)
        if features.get('transient', False):
            chunk_size *= 4

        for _ in range(base_chunks):
            start = int(self.rng.integers(0, max(1, total - chunk_size)))
            end = min(total, start + chunk_size)
            mode = self.rng.integers(0, 3)
            if mode == 0:
                # Shift: roll the chunk by a random offset
                off = int(self.rng.integers(1, 64))
                flat[start:end] = np.roll(flat[start:end], off)
            elif mode == 1:
                # XOR with a random bitmask (keeps high-order bits for recognisability)
                mask = np.uint8(self.rng.integers(1, 64))
                flat[start:end] ^= mask
            else:
                # Swap with a second random chunk of the same size
                start2 = int(self.rng.integers(0, max(1, total - (end - start))))
                end2 = start2 + (end - start)
                tmp = flat[start:end].copy()
                flat[start:end] = flat[start2:end2]
                flat[start2:end2] = tmp

        return flat.reshape(frame.shape)
