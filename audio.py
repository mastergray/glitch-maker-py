"""Audio feature extraction for audio-reactive Dazzle port.

The analyzer (vendored as `analyze_audio.py` alongside this file) produces
beats, onsets, section changes, spectral brightness, tension, and segment
labels. We add a supplementary 3-band FFT pass (bass/mid/treble) since the
analyzer only provides overall energy + spectral centroid.

Originally sourced from seamless-disarray/video-generator-py; vendored here
so the dazzle port is self-contained.
"""
from __future__ import annotations
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


EXTERNAL_ANALYZER = Path(__file__).resolve().parent / 'analyze_audio.py'


class AudioFeatures:
    def __init__(self, wav_path: str, fps: int = 60, analysis_json: str | None = None,
                 ffmpeg_bin: str = 'ffmpeg', ffprobe_bin: str = 'ffprobe'):
        self._ffmpeg_bin = ffmpeg_bin
        self._ffprobe_bin = ffprobe_bin

        data, sr = sf.read(wav_path, dtype='float32', always_2d=True)
        self.samples = data.mean(axis=1)
        self.sr = sr
        self.fps = fps
        self.samples_per_frame = int(round(sr / fps))
        self.n_frames = len(self.samples) // self.samples_per_frame
        self.duration_sec = len(self.samples) / sr

        self._compute_bands()
        self._load_external_analysis(wav_path, analysis_json)

    def _compute_bands(self):
        n = self.samples_per_frame
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sr)
        bass_mask = freqs < 200
        mid_mask = (freqs >= 200) & (freqs < 2000)
        treble_mask = freqs >= 2000
        window = np.hanning(n).astype(np.float32)

        rms = np.zeros(self.n_frames, dtype=np.float32)
        bass = np.zeros(self.n_frames, dtype=np.float32)
        mid = np.zeros(self.n_frames, dtype=np.float32)
        treble = np.zeros(self.n_frames, dtype=np.float32)

        for i in range(self.n_frames):
            chunk = self.samples[i * n:(i + 1) * n]
            if len(chunk) < n:
                break
            rms[i] = np.sqrt(np.mean(chunk * chunk))
            spec = np.abs(np.fft.rfft(chunk * window))
            bass[i] = spec[bass_mask].mean()
            mid[i] = spec[mid_mask].mean()
            treble[i] = spec[treble_mask].mean()

        self.rms = _norm(rms)
        self.bass = _norm(bass)
        self.mid = _norm(mid)
        self.treble = _norm(treble)

    def _load_external_analysis(self, wav_path: str, analysis_json: str | None):
        if analysis_json is None:
            tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
            tmp.close()
            analysis_json = tmp.name
            print(f'[audio] running external analyzer -> {analysis_json}')
            cmd = ['python3', str(EXTERNAL_ANALYZER),
                   '--input', str(wav_path), '--output', analysis_json]
            if self._ffmpeg_bin != 'ffmpeg':
                cmd.extend(['--ffmpeg', self._ffmpeg_bin])
            if self._ffprobe_bin != 'ffprobe':
                cmd.extend(['--ffprobe', self._ffprobe_bin])
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
            self._tmp_json = analysis_json
        else:
            self._tmp_json = None

        with open(analysis_json) as f:
            analysis = json.load(f)
        # Keep the raw blob so the renderer can inline it into a profile file.
        self.raw_analysis = analysis

        # Per-frame interpolation: analyzer is at 20Hz (50ms hops), we want `fps` Hz.
        frames = analysis['frames']
        times = np.array([f['t'] for f in frames], dtype=np.float32)
        energy = np.array([f['energy_rel'] for f in frames], dtype=np.float32)
        brightness = np.array([f['brightness_rel'] for f in frames], dtype=np.float32)
        onset = np.array([f['onset_rel'] for f in frames], dtype=np.float32)
        tension = np.array([f['tension_rel'] for f in frames], dtype=np.float32)

        frame_t = np.arange(self.n_frames) / self.fps
        self.energy = _zscore_to_01(np.interp(frame_t, times, energy))
        self.brightness = _zscore_to_01(np.interp(frame_t, times, brightness))
        self.onset = _zscore_to_01(np.interp(frame_t, times, onset))
        self.tension = _zscore_to_01(np.interp(frame_t, times, tension))

        # Build per-frame event flags
        self.beat = np.zeros(self.n_frames, dtype=bool)
        self.downbeat = np.zeros(self.n_frames, dtype=bool)
        self.peak = np.zeros(self.n_frames, dtype=bool)
        self.transient_hit = np.zeros(self.n_frames, dtype=bool)
        self.in_silence = np.zeros(self.n_frames, dtype=bool)
        self.section_change_frames: list[int] = []

        def t_to_frame(t):
            return min(self.n_frames - 1, max(0, int(round(t * self.fps))))

        silence_starts = []
        for ev in analysis['events']:
            t = ev['t']
            fi = t_to_frame(t)
            typ = ev['type']
            if typ == 'beat':
                self.beat[fi] = True
            elif typ == 'downbeat':
                self.downbeat[fi] = True
            elif typ == 'peak':
                self.peak[fi] = True
            elif typ == 'transient_hit':
                self.transient_hit[fi] = True
            elif typ == 'section_change':
                self.section_change_frames.append(fi)
            elif typ == 'silence_start':
                silence_starts.append(fi)
            elif typ == 'silence_end':
                if silence_starts:
                    start = silence_starts.pop(0)
                    self.in_silence[start:fi + 1] = True

        # Segment labels: build per-frame label list
        self.segments = analysis.get('segments', [])

    def features_at(self, frame_idx: int) -> dict:
        i = min(frame_idx, self.n_frames - 1)
        # Composite "transient" flag for effects: real onset or peak
        trans = bool(self.transient_hit[i] or self.peak[i])
        return dict(
            # 3-band (my supplementary pass) — still 0..1 normalized
            rms=float(self.rms[i]),
            bass=float(self.bass[i]),
            mid=float(self.mid[i]),
            treble=float(self.treble[i]),
            # Rich features from external analyzer
            energy=float(self.energy[i]),
            brightness=float(self.brightness[i]),
            onset=float(self.onset[i]),
            tension=float(self.tension[i]),
            # Discrete events
            on_beat=bool(self.beat[i]),
            on_downbeat=bool(self.downbeat[i]),
            on_peak=bool(self.peak[i]),
            in_silence=bool(self.in_silence[i]),
            # Back-compat: many effects read `transient` — now beat-aware
            transient=trans,
        )

    def segment_label_at(self, t_sec: float) -> list[str]:
        for s in self.segments:
            if s['start'] <= t_sec < s['end']:
                return s['labels']
        return []

    def __del__(self):
        if getattr(self, '_tmp_json', None) and os.path.exists(self._tmp_json):
            try:
                os.unlink(self._tmp_json)
            except OSError:
                pass


def _norm(x: np.ndarray) -> np.ndarray:
    p99 = np.percentile(x, 99.0)
    if p99 < 1e-9:
        return np.zeros_like(x)
    return np.clip(x / p99, 0.0, 1.0)


def _zscore_to_01(x: np.ndarray) -> np.ndarray:
    """Map z-score (roughly -3..+3) to 0..1 via a soft sigmoid-ish curve."""
    return np.clip(0.5 + x / 6.0, 0.0, 1.0)
