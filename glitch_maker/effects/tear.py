"""Tear — spatial signal-tear effect with multiple shape modes.

Shapes:
  band   (default) — axis-aligned strips anchored to an edge (classic signal tear)
  square          — axis-aligned squares at random positions in the frame
  circle          — circular regions at random positions

For band the length/thickness controls the strip dimensions; for square/circle
both collapse into a single size knob (length_pct = max side/diameter as %
of min(W,H)). Count is audio-scaled for all shapes. Source (displace/clip/
color) and opacity work identically across shapes.
"""
from __future__ import annotations
import numpy as np

from ._frame import ClipReader


class Tear:
    def __init__(self, source: str = 'displace', clip_path: str | None = None,
                 color: tuple[int, int, int] = (255, 0, 0),
                 opacity: float = 1.0,
                 orientation: str = 'horizontal',
                 length_pct: float = 80.0,
                 thickness_pct: float = 25.0,
                 count_pct: float = 5.0,
                 displace_max_pct: float = 25.0,
                 delay_frames: int = 1,
                 shape: str = 'band'):
        if source not in ('clip', 'color', 'displace'):
            raise ValueError(f'source must be clip/color/displace, got {source!r}')
        if orientation not in ('horizontal', 'vertical'):
            raise ValueError(f'orientation must be horizontal or vertical, got {orientation!r}')
        if shape not in ('band', 'square', 'circle'):
            raise ValueError(f'shape must be band/square/circle, got {shape!r}')
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
        self.delay_frames = max(1, int(delay_frames))
        self.shape = shape
        self.reader: ClipReader | None = ClipReader(clip_path) if source == 'clip' else None
        self.rng = np.random.default_rng()
        self._frame_idx = 0

    def _activity(self, features: dict) -> float:
        if features.get('section_change', False):
            return 1.0
        base = (features.get('rms', 0.0) * 0.5
                + features.get('bass', 0.0) * 0.3
                + features.get('onset', 0.0) * 0.2)
        if features.get('transient', False):
            base += 0.3
        return float(np.clip(base, 0.0, 1.0))

    def _paint_block(self, out, base, other, bbox, mask, dy, dx):
        """Write source pixels (displace/color/clip) into out[bbox] where mask is True.

        mask=None means the whole bbox is written (rectangular shapes skip the mask check
        for speed). For circle/hex the mask restricts writes to the shape interior.
        """
        y0, y1, x0, x1 = bbox
        H, W = base.shape[:2]
        dst = out[y0:y1, x0:x1]

        if self.source == 'displace':
            yy = (np.arange(y0, y1) + dy) % H
            xx = (np.arange(x0, x1) + dx) % W
            src = base[yy[:, None], xx[None, :]]
        elif self.source == 'clip':
            src = other[y0:y1, x0:x1]
        else:  # color
            src = np.broadcast_to(self.color, dst.shape)

        if self.opacity >= 1.0:
            if mask is None:
                np.copyto(dst, src)
            else:
                dst[mask] = src[mask]
        else:
            op = self.opacity
            if mask is None:
                blended = src.astype(np.float32) * op + dst.astype(np.float32) * (1.0 - op)
                np.copyto(dst, blended.astype(np.uint8))
            else:
                src_m = src[mask].astype(np.float32)
                dst_m = dst[mask].astype(np.float32)
                dst[mask] = (src_m * op + dst_m * (1.0 - op)).astype(np.uint8)

    def _step_band(self, frame, activity):
        """Classic axis-aligned band tears with edge origination."""
        H, W, _ = frame.shape
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
        base = frame
        out = frame.copy()

        thick_starts = self.rng.integers(0, thickness_axis, size=count)
        thicknesses = self.rng.integers(1, max_thickness + 1, size=count)
        lengths = np.maximum(1, (self.rng.uniform(0.5, 1.0, size=count) * max_length).astype(int))
        edges = self.rng.integers(0, 2, size=count)
        len_starts = np.where(edges == 0, 0, np.maximum(0, length_axis - lengths))

        # Band mode displaces only along the length axis (signal-tear look).
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
                bbox = (ts, t_end, ls, l_end)
            else:
                bbox = (ls, l_end, ts, t_end)
            self._paint_block(out, base, other, bbox, None, dy_arr[k], dx_arr[k])

        return out

    def _free_block_params(self, H, W, count, activity, max_size_factor=1.0):
        """Shared helper for square/circle: pick N random positions/sizes + 2D offsets."""
        min_axis = min(H, W)
        max_size = int(min_axis * self.length_pct / 100.0 * activity * max_size_factor)
        if max_size < 4:
            return None
        sizes = np.maximum(4, (self.rng.uniform(0.5, 1.0, size=count) * max_size).astype(int))
        # Displace offsets: both axes (squares/circles are 2D, so 1D shift doesn't fit)
        if self.source == 'displace':
            max_off_y = int(H * self.displace_max_pct / 100.0 * activity)
            max_off_x = int(W * self.displace_max_pct / 100.0 * activity)
            dy_arr = (self.rng.integers(-max_off_y, max_off_y + 1, size=count)
                      if max_off_y > 0 else np.zeros(count, dtype=int))
            dx_arr = (self.rng.integers(-max_off_x, max_off_x + 1, size=count)
                      if max_off_x > 0 else np.zeros(count, dtype=int))
        else:
            dy_arr = dx_arr = np.zeros(count, dtype=int)
        return sizes, dy_arr, dx_arr

    def _step_square(self, frame, activity):
        H, W, _ = frame.shape
        count = max(0, int(min(H, W) * self.count_pct / 100.0 * activity))
        if count == 0:
            return frame
        params = self._free_block_params(H, W, count, activity)
        if params is None:
            return frame
        sizes, dy_arr, dx_arr = params

        other = self.reader.read(frame.shape) if self.reader is not None else None
        base = frame
        out = frame.copy()

        for k in range(count):
            s = int(sizes[k])
            y = int(self.rng.integers(0, max(1, H - s)))
            x = int(self.rng.integers(0, max(1, W - s)))
            self._paint_block(out, base, other, (y, y + s, x, x + s), None,
                              int(dy_arr[k]), int(dx_arr[k]))
        return out

    def _step_circle(self, frame, activity):
        H, W, _ = frame.shape
        count = max(0, int(min(H, W) * self.count_pct / 100.0 * activity))
        if count == 0:
            return frame
        # length_pct here is max diameter, so scale radius by 0.5
        params = self._free_block_params(H, W, count, activity, max_size_factor=0.5)
        if params is None:
            return frame
        radii, dy_arr, dx_arr = params

        other = self.reader.read(frame.shape) if self.reader is not None else None
        base = frame
        out = frame.copy()

        for k in range(count):
            r = int(radii[k])
            cy = int(self.rng.integers(r, max(r + 1, H - r)))
            cx = int(self.rng.integers(r, max(r + 1, W - r)))
            y0, y1 = cy - r, cy + r
            x0, x1 = cx - r, cx + r
            # Build circular mask for the bbox (2r x 2r).
            yy, xx = np.ogrid[:y1 - y0, :x1 - x0]
            mask = (yy - r) ** 2 + (xx - r) ** 2 <= r * r
            self._paint_block(out, base, other, (y0, y1, x0, x1), mask,
                              int(dy_arr[k]), int(dx_arr[k]))
        return out

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        self._frame_idx += 1
        if self._frame_idx % self.delay_frames != 0:
            return frame

        activity = self._activity(features)
        if activity == 0.0:
            return frame

        if self.shape == 'band':
            return self._step_band(frame, activity)
        elif self.shape == 'square':
            return self._step_square(frame, activity)
        else:
            return self._step_circle(frame, activity)

    def close(self):
        if self.reader is not None:
            self.reader.close()
