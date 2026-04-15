"""Section swap — splice frames from a second video at analyzer-triggered moments.

Reads a second video alongside the primary one. When the audio analyzer
reports a section_change (or peak / downbeat, configurable), the output
switches to the secondary clip for N frames before returning. Creates the
cutup/breakbeat-video editing feel without manual timeline work.

Unlike other effects which are stateless-ish, this one holds its own frame
reader on the secondary clip. Construction happens in render.py where the
second video path is known.
"""
from __future__ import annotations
import numpy as np

from video import VideoMeta, open_reader, probe


class SectionSwap:
    def __init__(self, other_path: str, hold_frames: int = 30):
        self.other_path = other_path
        self.hold_frames = hold_frames
        self.other_meta: VideoMeta = probe(other_path)
        self.other_proc, self.other_nbytes = open_reader(other_path, self.other_meta)
        self.other_exhausted = False
        self.swap_remaining = 0

    def _read_other(self, target_shape):
        if self.other_exhausted:
            return None
        raw = self.other_proc.stdout.read(self.other_nbytes)
        if not raw or len(raw) < self.other_nbytes:
            self.other_exhausted = True
            return None
        other = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.other_meta.height, self.other_meta.width, 3)
        # Resize to match target if dimensions differ — cheap nearest-neighbour.
        H, W, _ = target_shape
        if other.shape[:2] != (H, W):
            ys = (np.arange(H) * self.other_meta.height / H).astype(int)
            xs = (np.arange(W) * self.other_meta.width / W).astype(int)
            other = other[ys[:, None], xs[None, :]]
        return other

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        # Trigger: section_change is the primary signal; on_peak for extra punches.
        if features.get('section_change', False):
            self.swap_remaining = self.hold_frames

        if self.swap_remaining > 0:
            other = self._read_other(frame.shape)
            if other is not None:
                self.swap_remaining -= 1
                return other
            # Secondary clip exhausted — fall through to primary.
            self.swap_remaining = 0

        return frame

    def close(self):
        # Terminate the reader to avoid SIGPIPE noise when it still has frames
        # queued and we're done consuming.
        try:
            self.other_proc.terminate()
        except ProcessLookupError:
            pass
        if self.other_proc.stdout:
            try:
                self.other_proc.stdout.close()
            except BrokenPipeError:
                pass
        self.other_proc.wait()
