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
    def __init__(self, source: str = 'displace', clip_path: str | None = None,
                 color: tuple[int, int, int] = (255, 0, 0),
                 opacity: float = 1.0,
                 orientation: str = 'horizontal',
                 length_pct: float = 100.0,
                 thickness_pct: float = 15.0,
                 count_pct: float = 2.0,
                 displace_max_pct: float = 10.0):
        if source not in ('clip', 'color', 'displace'):
            raise ValueError(f'source must be clip/color/displace, got {source!r}')
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
        self.displace_max_pct = float(np.clip(displace_max_pct, 0.0, 100.0))
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
        base = frame  # read from original base for displace source (before writes)

        out = frame.copy()  # displace needs a separate write target so reads stay clean
        opacity = self.opacity

        thick_starts = self.rng.integers(0, thickness_axis, size=count)
        thicknesses = self.rng.integers(1, max_thickness + 1, size=count)
        len_starts = self.rng.integers(0, max(1, length_axis), size=count)
        lengths = self.rng.integers(1, max_length + 1, size=count)

        # Displace offsets: shift ONLY along the length axis. Horizontal tears
        # shift horizontally; vertical tears shift vertically. Shifting on the
        # thickness axis makes bands look like 2D blocks instead of signal tears.
        if self.source == 'displace':
            if self.orientation == 'horizontal':
                max_off = int(W * self.displace_max_pct / 100.0 * activity)
                dx_arr = (self.rng.integers(-max_off, max_off + 1, size=count)
                          if max_off > 0 else np.zeros(count, dtype=int))
                dy_arr = np.zeros(count, dtype=int)
            else:
                max_off = int(H * self.displace_max_pct / 100.0 * activity)
                dy_arr = (self.rng.integers(-max_off, max_off + 1, size=count)
                          if max_off > 0 else np.zeros(count, dtype=int))
                dx_arr = np.zeros(count, dtype=int)
        else:
            dy_arr = dx_arr = np.zeros(count, dtype=int)

        for k in range(count):
            ts, th, ls, ln = thick_starts[k], thicknesses[k], len_starts[k], lengths[k]
            t_end = min(thickness_axis, ts + th)
            l_end = min(length_axis, ls + ln)
            if self.orientation == 'horizontal':
                y0, y1, x0, x1 = ts, t_end, ls, l_end
            else:
                y0, y1, x0, x1 = ls, l_end, ts, t_end
            dst = out[y0:y1, x0:x1]

            if self.source == 'displace':
                # Pull base pixels from a shifted region, wrapping at frame edges.
                yy = (np.arange(y0, y1) + dy_arr[k]) % H
                xx = (np.arange(x0, x1) + dx_arr[k]) % W
                src = base[yy[:, None], xx[None, :]]
            elif self.source == 'clip':
                src = other[y0:y1, x0:x1]
            else:  # color
                src = np.broadcast_to(self.color, dst.shape)

            if opacity >= 1.0:
                np.copyto(dst, src)
            else:
                blended = src.astype(np.float32) * opacity + dst.astype(np.float32) * (1.0 - opacity)
                np.copyto(dst, blended.astype(np.uint8))

        return out

    def close(self):
        if self.reader is not None:
            self.reader.close()
