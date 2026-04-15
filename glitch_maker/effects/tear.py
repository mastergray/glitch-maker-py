"""Tear — spatial signal-tear effect.

Paints strips of either a secondary video or a solid color over the primary
frame. Strip orientation, length, count, and opacity are configurable;
length/count are ceilings that audio scales up to (quiet = no tears,
transient/peak = near-ceiling, section_change = one-frame burst at 100%).
"""
from __future__ import annotations
import numpy as np

from ._frame import ClipReader


class Tear:
    def __init__(self, source: str = 'clip', clip_path: str | None = None,
                 color: tuple[int, int, int] = (255, 0, 0),
                 opacity: float = 1.0,
                 orientation: str = 'horizontal',
                 length_pct: float = 80.0,
                 count_pct: float = 5.0):
        if source not in ('clip', 'color'):
            raise ValueError(f'source must be clip or color, got {source!r}')
        if orientation not in ('horizontal', 'vertical'):
            raise ValueError(f'orientation must be horizontal or vertical, got {orientation!r}')
        if source == 'clip' and clip_path is None:
            raise ValueError('tear source=clip requires clip_path')

        self.source = source
        self.color = np.array(color, dtype=np.uint8)
        self.opacity = float(np.clip(opacity, 0.0, 1.0))
        self.orientation = orientation
        self.length_pct = float(np.clip(length_pct, 0.0, 100.0))
        self.count_pct = float(np.clip(count_pct, 0.0, 100.0))
        self.reader: ClipReader | None = ClipReader(clip_path) if source == 'clip' else None
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
        H, W, _ = frame.shape
        activity = self._activity(features)
        if activity == 0.0:
            return frame

        # Axis sizes: horizontal strips are 1-row tall, length along W.
        if self.orientation == 'horizontal':
            axis_n = H            # how many possible strip positions
            length_axis = W       # strip length runs along width
        else:
            axis_n = W
            length_axis = H

        count_ceiling = int(axis_n * self.count_pct / 100.0)
        count = max(0, int(count_ceiling * activity))
        if count == 0:
            return frame

        max_length = int(length_axis * self.length_pct / 100.0 * activity)
        if max_length <= 0:
            return frame

        # Pre-fetch secondary frame once if using clip source.
        other = self.reader.read(frame.shape) if self.reader is not None else None

        out = frame if self.opacity == 1.0 else frame.copy()
        opacity = self.opacity

        positions = self.rng.integers(0, axis_n, size=count)
        lengths = self.rng.integers(1, max_length + 1, size=count)
        starts = self.rng.integers(0, max(1, length_axis), size=count)

        for pos, ln, st in zip(positions, lengths, starts):
            end = min(length_axis, st + ln)
            if self.orientation == 'horizontal':
                dst = out[pos, st:end]
                src = (other[pos, st:end] if other is not None
                       else np.broadcast_to(self.color, dst.shape))
            else:
                dst = out[st:end, pos]
                src = (other[st:end, pos] if other is not None
                       else np.broadcast_to(self.color, dst.shape))

            if opacity >= 1.0:
                np.copyto(dst, src)
            else:
                # Linear blend: out = src*opacity + dst*(1-opacity)
                blended = src.astype(np.float32) * opacity + dst.astype(np.float32) * (1.0 - opacity)
                np.copyto(dst, blended.astype(np.uint8))

        return out

    def close(self):
        if self.reader is not None:
            self.reader.close()
