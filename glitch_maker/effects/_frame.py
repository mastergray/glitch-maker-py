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
