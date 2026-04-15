"""JPEG corrupt — re-encode random blocks at very low quality for real DCT artifacts.

Each affected block is extracted, JPEG-encoded in-memory at low quality
(default q=5), decoded, and pasted back. Produces authentic 8x8 DCT
macroblock artifacts — the "real" datamosh / compressed-to-death look.

More expensive than pixel_scramble (needs PIL per block) but the artifacts
are visually distinct and survive on any content.
"""
from __future__ import annotations
import io

import numpy as np
from PIL import Image


class JpegCorrupt:
    def __init__(self, count_pct: float = 3.0, size_pct: float = 20.0,
                 quality: int = 5):
        self.count_pct = float(np.clip(count_pct, 0.0, 100.0))
        self.size_pct = float(np.clip(size_pct, 0.0, 100.0))
        self.quality = int(np.clip(quality, 1, 100))
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

    def _corrupt_block(self, block: np.ndarray) -> np.ndarray:
        img = Image.fromarray(block, mode='RGB')
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=self.quality)
        buf.seek(0)
        return np.array(Image.open(buf).convert('RGB'))

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        H, W, _ = frame.shape
        activity = self._activity(features)
        if activity == 0.0:
            return frame

        axis = min(H, W)
        count = max(0, int(H * self.count_pct / 100.0 * activity))
        max_size = max(16, int(axis * self.size_pct / 100.0 * activity))
        if count == 0:
            return frame

        out = frame.copy()
        for _ in range(count):
            # Snap to multiples of 8 so JPEG's DCT grid aligns cleanly with the
            # block edges (otherwise internal blocks get partially rendered).
            h = int(self.rng.integers(16, max(17, max_size + 1))) & ~7
            w = int(self.rng.integers(16, max(17, max_size + 1))) & ~7
            h = max(8, h)
            w = max(8, w)
            y = int(self.rng.integers(0, max(1, H - h)))
            x = int(self.rng.integers(0, max(1, W - w)))
            block = out[y:y + h, x:x + w]
            out[y:y + h, x:x + w] = self._corrupt_block(block)
        return out
