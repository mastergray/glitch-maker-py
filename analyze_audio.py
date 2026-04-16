#!/usr/bin/env python3
import argparse
import json
import subprocess
from array import array
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


def ffprobe_metadata(audio_path):
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=sample_rate,channels",
            "-select_streams",
            "a:0",
            "-of",
            "json",
            str(audio_path),
        ],
        text=True,
    )
    data = json.loads(out)
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    stream = streams[0] if streams else {}
    duration = float(fmt.get("duration", 0.0))
    source_sr = int(stream.get("sample_rate", 0) or 0)
    channels = int(stream.get("channels", 1) or 1)
    return duration, source_sr, channels


def extract_pcm_mono(audio_path, sample_rate_hz, max_duration_sec=None):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
    ]
    if max_duration_sec is not None:
        cmd.extend(["-t", f"{max_duration_sec:.6f}"])
    cmd.extend(["-f", "s16le", "-"])
    raw = subprocess.check_output(cmd)
    samples = array("h")
    samples.frombytes(raw)
    mono = np.asarray(samples, dtype=np.float32) / 32768.0
    return mono


def zscore(values):
    mean = float(np.mean(values)) if values.size else 0.0
    std = float(np.std(values)) if values.size else 0.0
    if std < 1e-8:
        return np.zeros_like(values), mean, std
    return (values - mean) / std, mean, std


def frame_audio(samples, sample_rate_hz, hop_sec=0.05, frame_size=2048):
    hop = max(1, int(round(hop_sec * sample_rate_hz)))
    if samples.size < frame_size:
        pad = np.zeros(frame_size - samples.size, dtype=np.float32)
        samples = np.concatenate([samples, pad])

    starts = np.arange(0, max(1, samples.size - frame_size + 1), hop, dtype=np.int64)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int64)
    window = np.hanning(frame_size).astype(np.float32)

    times = np.zeros(starts.size, dtype=np.float32)
    energy_raw = np.zeros(starts.size, dtype=np.float32)
    brightness_raw = np.zeros(starts.size, dtype=np.float32)
    onset_raw = np.zeros(starts.size, dtype=np.float32)

    prev_mag = None
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate_hz).astype(np.float32)

    for i, start in enumerate(starts):
        frame = samples[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        times[i] = (start + frame_size * 0.5) / float(sample_rate_hz)
        energy_raw[i] = float(np.sqrt(np.mean(frame * frame) + 1e-12))

        spec = np.fft.rfft(frame * window)
        mag = np.abs(spec).astype(np.float32)
        mag_sum = float(np.sum(mag)) + 1e-12
        brightness_raw[i] = float(np.sum(freqs * mag) / mag_sum)

        if prev_mag is None:
            onset_raw[i] = 0.0
        else:
            diff = mag - prev_mag
            onset_raw[i] = float(np.mean(np.maximum(diff, 0.0)))
        prev_mag = mag

    return times, energy_raw, brightness_raw, onset_raw


def build_frame_metrics(times, energy_raw, brightness_raw, onset_raw):
    # Log-compress energy for better dynamic behavior before normalization.
    energy_comp = 20.0 * np.log10(np.maximum(energy_raw, 1e-8))
    onset_comp = np.log1p(np.maximum(onset_raw, 0.0))

    energy_rel, e_mean, e_std = zscore(energy_comp)
    brightness_rel, b_mean, b_std = zscore(brightness_raw)
    onset_rel, o_mean, o_std = zscore(onset_comp)

    tension_raw = 0.5 * energy_rel + 0.3 * onset_rel + 0.2 * brightness_rel
    tension_rel, t_mean, t_std = zscore(tension_raw)

    stats = {
        "energy": {
            "raw_mean": e_mean,
            "raw_stddev": e_std,
            "raw_min": float(np.min(energy_comp)),
            "raw_max": float(np.max(energy_comp)),
        },
        "spectral_brightness": {
            "raw_mean": b_mean,
            "raw_stddev": b_std,
            "raw_min": float(np.min(brightness_raw)),
            "raw_max": float(np.max(brightness_raw)),
        },
        "onset_strength": {
            "raw_mean": o_mean,
            "raw_stddev": o_std,
            "raw_min": float(np.min(onset_comp)),
            "raw_max": float(np.max(onset_comp)),
        },
        "tension": {
            "raw_mean": t_mean,
            "raw_stddev": t_std,
            "raw_min": float(np.min(tension_raw)),
            "raw_max": float(np.max(tension_raw)),
        },
    }

    frames = []
    for i, t in enumerate(times):
        frames.append(
            {
                "t": float(t),
                "energy_rel": float(energy_rel[i]),
                "brightness_rel": float(brightness_rel[i]),
                "onset_rel": float(onset_rel[i]),
                "tension_rel": float(tension_rel[i]),
            }
        )
    return frames, stats, energy_rel, brightness_rel, onset_rel, tension_rel


def detect_beats(times, onset_rel, min_interval_sec=0.25):
    if times.size < 3:
        return []
    dt = float(np.median(np.diff(times)))
    min_distance = max(1, int(round(min_interval_sec / max(dt, 1e-6))))
    peaks, props = find_peaks(onset_rel, height=0.5, distance=min_distance)
    heights = props.get("peak_heights", np.array([], dtype=np.float32))
    events = []
    for idx, h in zip(peaks, heights):
        strength = float(np.clip((h - 0.5) / 2.0, 0.0, 1.0))
        events.append(
            {
                "t": float(times[idx]),
                "type": "beat",
                "strength": strength,
                "confidence": float(np.clip(0.55 + 0.4 * strength, 0.0, 1.0)),
            }
        )
    return events


def detect_downbeats(beat_events):
    downbeats = []
    for i, beat in enumerate(beat_events):
        if i % 4 != 0:
            continue
        downbeats.append(
            {
                "t": beat["t"],
                "type": "downbeat",
                "strength": float(np.clip(beat.get("strength", 0.0) * 1.1, 0.0, 1.0)),
                "confidence": float(np.clip(beat.get("confidence", 0.0) * 0.95, 0.0, 1.0)),
            }
        )
    return downbeats


def detect_peaks(times, tension_rel):
    peaks, props = find_peaks(tension_rel, height=1.5, distance=2)
    heights = props.get("peak_heights", np.array([], dtype=np.float32))
    out = []
    for idx, h in zip(peaks, heights):
        out.append(
            {
                "t": float(times[idx]),
                "type": "peak",
                "metric": "tension_rel",
                "value": float(h),
                "confidence": float(np.clip((h - 1.5) / 2.0 + 0.6, 0.0, 1.0)),
            }
        )
    return out


def detect_silence_regions(times, energy_rel, threshold=-1.25, min_duration_sec=0.25):
    silent = energy_rel < threshold
    if times.size < 2:
        return []
    dt = float(np.median(np.diff(times)))
    min_len = max(1, int(round(min_duration_sec / max(dt, 1e-6))))
    regions = []
    start = None
    for i, is_silent in enumerate(silent):
        if is_silent and start is None:
            start = i
        elif not is_silent and start is not None:
            if i - start >= min_len:
                regions.append((start, i - 1))
            start = None
    if start is not None and len(silent) - start >= min_len:
        regions.append((start, len(silent) - 1))

    events = []
    for s, e in regions:
        conf = float(np.clip((abs(float(np.mean(energy_rel[s : e + 1]))) - 1.25) / 1.8 + 0.6, 0.0, 1.0))
        events.append({"t": float(times[s]), "type": "silence_start", "confidence": conf})
        events.append({"t": float(times[e]), "type": "silence_end", "confidence": conf})
    return events


def detect_section_changes(times, energy_rel, brightness_rel, onset_rel, bin_sec=1.0):
    if times.size < 2:
        return []
    t_end = float(times[-1])
    bins = np.arange(0.0, t_end + bin_sec, bin_sec)
    if bins.size < 3:
        return []

    binned = []
    binned_t = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (times >= lo) & (times < hi)
        if not np.any(mask):
            continue
        binned_t.append(float(lo))
        binned.append(
            [
                float(np.mean(energy_rel[mask])),
                float(np.mean(brightness_rel[mask])),
                float(np.mean(onset_rel[mask])),
            ]
        )
    if len(binned) < 3:
        return []

    binned = np.asarray(binned, dtype=np.float32)
    novelty = np.linalg.norm(np.diff(binned, axis=0), axis=1)
    mean = float(np.mean(novelty))
    std = float(np.std(novelty))
    threshold = mean + std
    peaks, props = find_peaks(novelty, height=threshold, distance=1)
    heights = props.get("peak_heights", np.array([], dtype=np.float32))

    out = []
    for idx, h in zip(peaks, heights):
        conf = float(np.clip((h - threshold) / (std + 1e-6) * 0.25 + 0.7, 0.0, 1.0))
        out.append({"t": float(binned_t[idx + 1]), "type": "section_change", "confidence": conf})
    return out


def detect_transient_hits(times, onset_rel, beat_events):
    peaks, props = find_peaks(onset_rel, height=1.2, distance=1)
    heights = props.get("peak_heights", np.array([], dtype=np.float32))
    beat_times = np.array([e["t"] for e in beat_events], dtype=np.float32)
    out = []
    for idx, h in zip(peaks, heights):
        t = float(times[idx])
        if beat_times.size > 0 and float(np.min(np.abs(beat_times - t))) <= 0.04:
            continue
        strength = float(np.clip((h - 1.2) / 2.0, 0.0, 1.0))
        out.append(
            {
                "t": t,
                "type": "transient_hit",
                "strength": strength,
                "confidence": float(np.clip(0.6 + 0.35 * strength, 0.0, 1.0)),
            }
        )
    return out


def build_segments(times, energy_rel, brightness_rel, onset_rel, tension_rel, bin_sec=1.0):
    if times.size == 0:
        return []
    t_end = float(times[-1])
    bins = np.arange(0.0, t_end + bin_sec, bin_sec)
    rows = []
    for i in range(len(bins) - 1):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (times >= lo) & (times < hi)
        if not np.any(mask):
            continue
        e = float(np.mean(energy_rel[mask]))
        b = float(np.mean(brightness_rel[mask]))
        o = float(np.mean(onset_rel[mask]))
        t = float(np.mean(tension_rel[mask]))

        labels = []
        if e < -1.25:
            labels.append("silent")
        else:
            if t < -0.5:
                labels.append("calm")
            elif t > 0.8 or e > 0.8:
                labels.append("intense")
            else:
                labels.append("neutral")

            if b > 0.5:
                labels.append("bright")
            elif b < -0.5:
                labels.append("dark")

            if o > 0.6:
                labels.append("percussive")
            elif o < -0.2:
                labels.append("sustained")

        conf = float(np.clip(0.55 + 0.25 * min(1.5, abs(t)) + 0.15 * min(1.5, abs(e)), 0.0, 1.0))
        rows.append((lo, hi, tuple(sorted(set(labels))), conf))

    if not rows:
        return []

    segments = []
    cur_start, cur_end, cur_labels, confs = rows[0][0], rows[0][1], rows[0][2], [rows[0][3]]
    for lo, hi, labels, conf in rows[1:]:
        if labels == cur_labels and abs(lo - cur_end) < 1e-6:
            cur_end = hi
            confs.append(conf)
            continue
        segments.append(
            {
                "start": float(cur_start),
                "end": float(cur_end),
                "labels": list(cur_labels),
                "confidence": float(np.mean(confs)),
            }
        )
        cur_start, cur_end, cur_labels, confs = lo, hi, labels, [conf]

    segments.append(
        {
            "start": float(cur_start),
            "end": float(cur_end),
            "labels": list(cur_labels),
            "confidence": float(np.mean(confs)),
        }
    )
    return segments


def build_output(
    input_path,
    track_id,
    track_title,
    track_artist,
    duration_sec,
    sample_rate_hz,
    channels,
    hop_sec,
    stats,
    frames,
    events,
    segments,
):
    return {
        "schema_version": "1.0.0",
        "track": {
            "id": track_id,
            "title": track_title,
            "artist": track_artist,
            "duration_sec": float(duration_sec),
            "sample_rate_hz": int(sample_rate_hz),
            "channels": int(channels),
        },
        "analysis": {
            "frame_hop_sec": float(hop_sec),
            "normalization": {
                "mode": "relative_track_zscore",
                "center": "mean",
                "scale": "stddev",
            },
            "source_path": str(input_path),
        },
        "global_stats": {"metrics": stats},
        "frames": frames,
        "events": sorted(events, key=lambda e: e["t"]),
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze an audio file and emit Audio Timeline Analysis v1 JSON."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input audio file path (.wav/.mp3/etc)")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    parser.add_argument("--track-id", default=None, help="Track ID for output JSON")
    parser.add_argument("--title", default=None, help="Track title")
    parser.add_argument("--artist", default=None, help="Track artist")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Analysis sample rate")
    parser.add_argument("--hop-sec", type=float, default=0.05, help="Frame hop size in seconds")
    parser.add_argument("--frame-size", type=int, default=2048, help="Frame size in samples")
    parser.add_argument("--max-duration", type=float, default=None, help="Optional limit for analysis duration (seconds)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.hop_sec <= 0:
        raise ValueError("--hop-sec must be > 0")
    if args.frame_size < 256:
        raise ValueError("--frame-size must be >= 256")
    if args.sample_rate < 8000:
        raise ValueError("--sample-rate must be >= 8000")

    duration_sec, source_sr, channels = ffprobe_metadata(args.input)
    mono = extract_pcm_mono(args.input, args.sample_rate, max_duration_sec=args.max_duration)
    if mono.size == 0:
        raise RuntimeError("No PCM samples were extracted from input audio.")

    times, energy_raw, brightness_raw, onset_raw = frame_audio(
        mono, args.sample_rate, hop_sec=args.hop_sec, frame_size=args.frame_size
    )

    frames, stats, energy_rel, brightness_rel, onset_rel, tension_rel = build_frame_metrics(
        times, energy_raw, brightness_raw, onset_raw
    )

    beat_events = detect_beats(times, onset_rel)
    events = []
    events.extend(beat_events)
    events.extend(detect_downbeats(beat_events))
    events.extend(detect_peaks(times, tension_rel))
    events.extend(detect_section_changes(times, energy_rel, brightness_rel, onset_rel))
    events.extend(detect_silence_regions(times, energy_rel))
    events.extend(detect_transient_hits(times, onset_rel, beat_events))

    segments = build_segments(times, energy_rel, brightness_rel, onset_rel, tension_rel)

    effective_duration = min(
        duration_sec,
        float(args.max_duration) if args.max_duration is not None else duration_sec,
    )
    output = build_output(
        input_path=args.input,
        track_id=args.track_id or args.input.stem,
        track_title=args.title or args.input.stem,
        track_artist=args.artist or "unknown",
        duration_sec=effective_duration,
        sample_rate_hz=args.sample_rate,
        channels=channels if channels > 0 else 1,
        hop_sec=args.hop_sec,
        stats=stats,
        frames=frames,
        events=events,
        segments=segments,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(output, f, indent=2)
            f.write("\n")
        else:
            json.dump(output, f, separators=(",", ":"))

    print(f"Wrote {args.output}")
    print(f"Frames: {len(frames)} | Events: {len(events)} | Segments: {len(segments)}")
    print(f"Source duration: {duration_sec:.2f}s | Effective duration: {effective_duration:.2f}s")
    print(f"Source SR: {source_sr} Hz | Analysis SR: {args.sample_rate} Hz")


if __name__ == "__main__":
    main()
