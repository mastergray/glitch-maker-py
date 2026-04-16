"""Pixel sort — Kim-Asendorf-style sort of contiguous pixel runs within a threshold.

Sorts each row (or column) of pixels by a chosen key (brightness / hue / saturation).
Only pixels whose value falls within [lower, upper] get included in the sort,
producing the characteristic streaky/melted look with the sorted bands sitting
beside untouched regions.

Audio reactivity: activity widens the threshold window from a narrow baseline
toward the user-supplied max range. At silence the sort affects a tight middle
band; at transients/peaks the full user-set range is active, producing the most
dramatic streaking.
"""
from __future__ import annotations
import numpy as np
from PIL import Image


def _brightness(pixels: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 -> (H,W) float32 [0,1]."""
    r = pixels[..., 0].astype(np.float32)
    g = pixels[..., 1].astype(np.float32)
    b = pixels[..., 2].astype(np.float32)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def _hue_saturation(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H,W,3) uint8 -> (hue, saturation) each (H,W) float32 [0,1]."""
    img = Image.fromarray(pixels, mode='RGB').convert('HSV')
    hsv = np.array(img, dtype=np.float32) / 255.0
    return hsv[..., 0], hsv[..., 1]


class PixelSort:
    def __init__(self,
                 sort_by: str = 'brightness',
                 direction: str = 'horizontal',
                 lower_threshold: float = 0.25,
                 upper_threshold: float = 0.8,
                 reverse: bool = False,
                 angle: int = 0,
                 audio_scale: float = 0.5):
        if sort_by not in ('brightness', 'hue', 'saturation'):
            raise ValueError(f'sort_by must be brightness/hue/saturation, got {sort_by!r}')
        if direction not in ('horizontal', 'vertical', 'both'):
            raise ValueError(f'direction must be horizontal/vertical/both, got {direction!r}')

        self.sort_by = sort_by
        self.direction = direction
        self.lower = float(np.clip(lower_threshold, 0.0, 1.0))
        self.upper = float(np.clip(upper_threshold, self.lower, 1.0))
        self.reverse = bool(reverse)
        self.angle = int(angle) % 360
        self.audio_scale = float(np.clip(audio_scale, 0.0, 1.0))

    def _activity(self, features: dict) -> float:
        if features.get('section_change', False):
            return 1.0
        base = (features.get('rms', 0.0) * 0.5
                + features.get('bass', 0.0) * 0.3
                + features.get('onset', 0.0) * 0.2)
        if features.get('transient', False):
            base += 0.3
        return float(np.clip(base, 0.0, 1.0))

    def _value_map(self, pixels: np.ndarray) -> np.ndarray:
        """Return (H, W) float32 [0,1] sort-key values per pixel."""
        if self.sort_by == 'brightness':
            return _brightness(pixels)
        hue, sat = _hue_saturation(pixels)
        return hue if self.sort_by == 'hue' else sat

    def _sort_axis_0(self, pixels: np.ndarray, lower: float, upper: float) -> np.ndarray:
        """Sort contiguous runs within [lower, upper] along the first axis per column.

        pixels: (N, M, 3). Treats each column as an independent "row" — sorts runs
        along axis 0 within each column. Caller transposes if horizontal.
        """
        values = self._value_map(pixels)  # (N, M)
        mask = (values >= lower) & (values <= upper)
        out = pixels.copy()
        N, M, _ = pixels.shape

        for col in range(M):
            m = mask[:, col]
            if not m.any():
                continue
            # Find contiguous True runs
            col_vals = values[:, col]
            col_pix = out[:, col]
            i = 0
            while i < N:
                if m[i]:
                    j = i
                    while j < N and m[j]:
                        j += 1
                    if j - i > 1:
                        run_vals = col_vals[i:j]
                        order = np.argsort(run_vals)
                        if self.reverse:
                            order = order[::-1]
                        col_pix[i:j] = col_pix[i:j][order]
                    i = j
                else:
                    i += 1
        return out

    def _sort_horizontal(self, pixels: np.ndarray, lower: float, upper: float) -> np.ndarray:
        # Treat rows as "columns" by transposing, sort along axis 0, transpose back.
        t = pixels.transpose(1, 0, 2)
        t = self._sort_axis_0(t, lower, upper)
        return t.transpose(1, 0, 2)

    def _sort_vertical(self, pixels: np.ndarray, lower: float, upper: float) -> np.ndarray:
        return self._sort_axis_0(pixels, lower, upper)

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        activity = self._activity(features)
        if activity == 0.0 and self.audio_scale > 0.0:
            return frame

        # Scale thresholds by audio: at low activity the range narrows toward the
        # midpoint (less sort); at high activity the full user range applies.
        if self.audio_scale > 0.0:
            mid = (self.lower + self.upper) * 0.5
            # Shrink factor: 1.0 at activity=1, (1 - audio_scale) at activity=0
            shrink = 1.0 - self.audio_scale * (1.0 - activity)
            half = (self.upper - self.lower) * 0.5 * shrink
            eff_lower = max(0.0, mid - half)
            eff_upper = min(1.0, mid + half)
        else:
            eff_lower = self.lower
            eff_upper = self.upper

        if self.angle != 0:
            img = Image.fromarray(frame, mode='RGB')
            img = img.rotate(self.angle, expand=True, fillcolor=(0, 0, 0))
            work = np.array(img)
        else:
            work = frame

        if self.direction in ('horizontal', 'both'):
            work = self._sort_horizontal(work, eff_lower, eff_upper)
        if self.direction in ('vertical', 'both'):
            work = self._sort_vertical(work, eff_lower, eff_upper)

        if self.angle != 0:
            img = Image.fromarray(work, mode='RGB').rotate(-self.angle, expand=True,
                                                           fillcolor=(0, 0, 0))
            W_rot, H_rot = img.size
            orig_h, orig_w = frame.shape[:2]
            left = (W_rot - orig_w) // 2
            top = (H_rot - orig_h) // 2
            img = img.crop((left, top, left + orig_w, top + orig_h))
            work = np.array(img)

        return work
