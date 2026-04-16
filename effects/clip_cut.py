"""Clip cut — temporal jump-cut to a second video on analyzer-triggered moments.

When the audio analyzer reports a section_change, output switches to the
secondary clip for N frames, then returns to the primary. Creates the
cutup / breakbeat-video editing feel without manual timeline work.

Formerly named section_swap; kept as a distinct effect from `tear` (which
is a spatial mixer, not a temporal cutter).
"""
from __future__ import annotations
import numpy as np

from ._frame import ClipReader


class ClipCut:
    def __init__(self, other_path: str, hold_frames: int = 30):
        self.reader = ClipReader(other_path)
        self.hold_frames = hold_frames
        self.remaining = 0

    def step(self, frame: np.ndarray, features: dict) -> np.ndarray:
        if features.get('section_change', False):
            self.remaining = self.hold_frames
        if self.remaining > 0:
            self.remaining -= 1
            return self.reader.read(frame.shape)
        return frame

    def close(self):
        self.reader.close()
