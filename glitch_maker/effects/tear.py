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
                 thickness_pct: float = 15.0,
                 count_pct: float = 2.0):
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
        self.thickness_pct = float(np.clip(thickness_pct, 0.0, 100.0))
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

        # For horizontal tears: blocks span along W (length), stacked along H
        # (thickness). Swap roles for vertical.
        if self.orientation == 'horizontal':
            thickness_axis = H
            length_axis = W
        else:
            thickness_axis = W
            length_axis = H

        count = max(0, int(thickness_axis * self.count_pct / 100.0 * activity))
        max_length = int(length_axis * self.length_pct / 100.0 * activity)
        max_thickness = int(thickness_axis * self.thickness_pct / 100.0 * activity)
        if count == 0 or max_length <= 0 or max_thickness <= 0:
            return frame

        other = self.reader.read(frame.shape) if self.reader is not None else None

        out = frame if self.opacity == 1.0 else frame.copy()
        opacity = self.opacity

        thick_starts = self.rng.integers(0, thickness_axis, size=count)
        thicknesses = self.rng.integers(1, max_thickness + 1, size=count)
        len_starts = self.rng.integers(0, max(1, length_axis), size=count)
        lengths = self.rng.integers(1, max_length + 1, size=count)

        for ts, th, ls, ln in zip(thick_starts, thicknesses, len_starts, lengths):
            t_end = min(thickness_axis, ts + th)
            l_end = min(length_axis, ls + ln)
            if self.orientation == 'horizontal':
                dst = out[ts:t_end, ls:l_end]
                src = (other[ts:t_end, ls:l_end] if other is not None
                       else np.broadcast_to(self.color, dst.shape))
            else:
                dst = out[ls:l_end, ts:t_end]
                src = (other[ls:l_end, ts:t_end] if other is not None
                       else np.broadcast_to(self.color, dst.shape))

            if opacity >= 1.0:
                np.copyto(dst, src)
            else:
                blended = src.astype(np.float32) * opacity + dst.astype(np.float32) * (1.0 - opacity)
                np.copyto(dst, blended.astype(np.uint8))

        return out

    def close(self):
        if self.reader is not None:
            self.reader.close()
