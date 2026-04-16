"""Shared frame primitives for glitch effects.

Frames are ndarray[H, W, 3] uint8 RGB. Effects may mutate in-place or return
a new array; render.py treats the returned value as authoritative.
"""
from __future__ import annotations
import numpy as np

from video import VideoMeta, open_reader, probe


def luminance(frame: np.ndarray) -> np.ndarray:
    """Return (H, W) float32 perceived brightness [0..1]."""
    r = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    b = frame[:, :, 2].astype(np.float32)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def pick_block_masks(shape: str, count: int, H: int, W: int, max_size: int,
                     rng: np.random.Generator) -> list:
    """Pick N random block positions with optional per-block masks.

    Returns list of (bbox, mask) pairs where bbox = (y0, y1, x0, x1) and mask
    is either None (full bbox) or a (h, w) bool array restricting the region
    to the shape interior.

    Shapes:
      rectangle  random w x h rectangles (mask=None)
      square     random side, equal w = h (mask=None)
      circle     random radius disks (mask is a circular bool array)
    """
    if shape not in ('rectangle', 'square', 'circle'):
        raise ValueError(f'shape must be rectangle/square/circle, got {shape!r}')
    if max_size < 4 or count <= 0:
        return []

    blocks = []
    for _ in range(count):
        if shape == 'rectangle':
            h = int(rng.integers(4, max_size + 1))
            w = int(rng.integers(4, max_size + 1))
            y = int(rng.integers(0, max(1, H - h)))
            x = int(rng.integers(0, max(1, W - w)))
            blocks.append(((y, y + h, x, x + w), None))
        elif shape == 'square':
            s = int(rng.integers(4, max_size + 1))
            y = int(rng.integers(0, max(1, H - s)))
            x = int(rng.integers(0, max(1, W - s)))
            blocks.append(((y, y + s, x, x + s), None))
        else:  # circle
            r = int(rng.integers(2, max(3, max_size // 2 + 1)))
            cy = int(rng.integers(r, max(r + 1, H - r)))
            cx = int(rng.integers(r, max(r + 1, W - r)))
            y0, y1 = cy - r, cy + r
            x0, x1 = cx - r, cx + r
            yy, xx = np.ogrid[:y1 - y0, :x1 - x0]
            mask = (yy - r) ** 2 + (xx - r) ** 2 <= r * r
            blocks.append(((y0, y1, x0, x1), mask))
    return blocks


def write_block(out: np.ndarray, bbox: tuple, mask, new_values: np.ndarray) -> None:
    """Write new_values into out[bbox] where mask is True (or all if mask is None).

    new_values should match the bbox size (not just the masked pixels). The function
    handles the masked write.
    """
    y0, y1, x0, x1 = bbox
    dst = out[y0:y1, x0:x1]
    if mask is None:
        np.copyto(dst, new_values)
    else:
        dst[mask] = new_values[mask]


class ClipReader:
    """Reads frames from a secondary video, resizing to target shape on demand.

    Used by any effect that composes a second source. Re-opens the ffmpeg
    process on EOF so the secondary loops indefinitely (important for long
    primaries). Callers must invoke close() when done.
    """
    def __init__(self, path: str):
        self.path = path
        self.meta: VideoMeta = probe(path)
        self._open()

    def _open(self):
        self.proc, self.nbytes = open_reader(self.path, self.meta)

    def read(self, target_shape) -> np.ndarray:
        raw = self.proc.stdout.read(self.nbytes)
        if not raw or len(raw) < self.nbytes:
            # EOF — tear down and re-open so the secondary loops
            self.close()
            self._open()
            raw = self.proc.stdout.read(self.nbytes)
            if not raw or len(raw) < self.nbytes:
                # Still empty after reopen — return black as a last resort
                H, W, _ = target_shape
                return np.zeros((H, W, 3), dtype=np.uint8)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.meta.height, self.meta.width, 3)
        H, W, _ = target_shape
        if frame.shape[:2] != (H, W):
            ys = (np.arange(H) * self.meta.height / H).astype(int)
            xs = (np.arange(W) * self.meta.width / W).astype(int)
            frame = frame[ys[:, None], xs[None, :]]
        return frame

    def close(self):
        try:
            self.proc.terminate()
        except ProcessLookupError:
            pass
        if self.proc.stdout:
            try:
                self.proc.stdout.close()
            except BrokenPipeError:
                pass
        self.proc.wait()
