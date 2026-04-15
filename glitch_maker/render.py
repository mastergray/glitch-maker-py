"""Glitch-maker — audio-reactive glitch effects for video.

Reads a source video via ffmpeg (rgb24 frames), pipes each frame through a
stack of effects driven by audio-analysis features (shared with dazzle), and
writes the result to MP4 with the source audio track preserved.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from audio import AudioFeatures
from video import probe, open_reader, open_writer
from effects import REGISTRY
from effects.section_swap import SectionSwap


def _extract_audio(video_path: str) -> str:
    """Extract the source audio track to a temp WAV; returns the WAV path."""
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    subprocess.check_call(
        ['ffmpeg', '-y', '-v', 'error',
         '-i', video_path,
         '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
         tmp.name],
    )
    return tmp.name


def build_stack(names, swap_with):
    stack = []
    for name in names:
        if name == 'section_swap':
            if not swap_with:
                raise SystemExit('section_swap requires --swap-with PATH')
            stack.append(('section_swap', SectionSwap(swap_with)))
        elif name in REGISTRY:
            stack.append((name, REGISTRY[name]()))
        else:
            raise SystemExit(f'unknown effect: {name}')
    return stack


def render(source_path: str, out_path: str, effects: list[str],
           swap_with: str | None = None, audio_path: str | None = None,
           seed: int | None = None, crf: int = 18,
           profile_path: str | None = None):
    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'big')
    np.random.seed(seed)

    meta = probe(source_path)
    print(f'[video] source: {source_path}')
    print(f'  {meta.width}x{meta.height} @ {meta.fps:.2f} fps  ~{meta.n_frames} frames  {meta.duration_sec:.2f}s')

    # Audio: extract from source unless user supplied override.
    audio_src = audio_path or _extract_audio(source_path)
    print(f'[audio] loading features from {audio_src}')
    af = AudioFeatures(audio_src, fps=int(round(meta.fps)))
    section_frames = set(af.section_change_frames)
    print(f'  duration={af.duration_sec:.2f}s sections={len(section_frames)} beats={int(af.beat.sum())}')

    stack = build_stack(effects, swap_with)
    print(f'[stack] {[n for n, _ in stack]}')

    reader, frame_nbytes = open_reader(source_path, meta)
    writer = open_writer(source_path, out_path, meta, crf=crf)

    print(f'[render] writing {out_path}')
    t0 = time.time()
    i = 0
    try:
        while True:
            raw = reader.stdout.read(frame_nbytes)
            if not raw or len(raw) < frame_nbytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(meta.height, meta.width, 3).copy()

            feats = af.features_at(i)
            feats['section_change'] = i in section_frames

            for _, eff in stack:
                frame = eff.step(frame, feats)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)

            writer.stdin.write(frame.tobytes())
            i += 1
            if i % 60 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                pct = i / max(1, meta.n_frames) * 100
                print(f'  {i:5d}/{meta.n_frames}  {pct:5.1f}%  {rate:4.1f} fps', flush=True)
    finally:
        reader.stdout.close()
        reader.wait()
        writer.stdin.close()
        writer.wait()
        for name, eff in stack:
            if hasattr(eff, 'close'):
                eff.close()
        # Clean up any audio tempfile we created ourselves.
        if audio_path is None:
            try:
                os.remove(audio_src)
            except OSError:
                pass

    dt = time.time() - t0
    render_fps = i / dt if dt else 0
    print(f'[render] done in {dt:.1f}s ({render_fps:.1f} fps, {i} frames)')

    profile = {
        'input': {'video': os.path.abspath(source_path),
                  'audio': os.path.abspath(audio_src) if audio_path else 'extracted-from-video',
                  'width': meta.width, 'height': meta.height,
                  'fps': meta.fps, 'n_frames': i},
        'output': {'path': os.path.abspath(out_path), 'crf': crf},
        'settings': {'seed': seed, 'effects': list(effects),
                     'swap_with': swap_with},
        'audio_analysis_summary': {
            'beats': int(af.beat.sum()),
            'downbeats': int(af.downbeat.sum()),
            'peaks': int(af.peak.sum()),
            'sections': len(section_frames),
        },
        'render_stats': {'wall_sec': round(dt, 2), 'render_fps': round(render_fps, 2)},
    }

    print()
    print('=== render summary ===')
    print(f'  input:   {profile["input"]["video"]}')
    print(f'  output:  {profile["output"]["path"]}')
    print(f'  effects: {", ".join(effects)}')
    print(f'  seed:    {seed}')
    asum = profile['audio_analysis_summary']
    print(f'  audio:   beats={asum["beats"]} downbeats={asum["downbeats"]} '
          f'peaks={asum["peaks"]} sections={asum["sections"]}')
    print(f'  render:  {dt:.1f}s at {render_fps:.1f} fps')

    if profile_path is not None:
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        print(f'  profile: {os.path.abspath(profile_path)}')
    else:
        print('  profile: (skipped, --no-profile)')


def main():
    effect_names = sorted(list(REGISTRY.keys()) + ['section_swap'])
    epilog = 'Available effects:\n  ' + ', '.join(effect_names)
    p = argparse.ArgumentParser(
        description='Glitch-maker — audio-reactive glitch effects for video',
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('source', help='input video (MP4 / MKV / ...)')
    p.add_argument('output', help='output MP4')
    p.add_argument('--effects', nargs='+', required=True,
                   help='ordered stack of effects to apply (see list below)')
    p.add_argument('--swap-with', default=None,
                   help='secondary video path; required when "section_swap" is in --effects')
    p.add_argument('--audio', default=None,
                   help='override audio source (default: extract from --source)')
    p.add_argument('--seed', type=int, default=None,
                   help='RNG seed (recorded in profile for reproducibility)')
    p.add_argument('--crf', type=int, default=18,
                   help='H.264 quality (default 18; lower=bigger/sharper)')
    p.add_argument('--profile', default=None,
                   help='path to write render profile JSON (default: <output>.profile.json)')
    p.add_argument('--no-profile', action='store_true',
                   help='skip writing the profile JSON')

    args = p.parse_args()

    unknown = [e for e in args.effects if e not in REGISTRY and e != 'section_swap']
    if unknown:
        p.error(f"unknown effect(s): {', '.join(unknown)}\n  valid: {', '.join(effect_names)}")

    if args.no_profile:
        profile_path = None
    elif args.profile is not None:
        profile_path = args.profile
    else:
        base, _ = os.path.splitext(args.output)
        profile_path = base + '.profile.json'

    render(args.source, args.output, args.effects,
           swap_with=args.swap_with, audio_path=args.audio,
           seed=args.seed, crf=args.crf, profile_path=profile_path)


if __name__ == '__main__':
    main()
