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
from effects import REGISTRY, PARAMETRIC


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


def _parse_color(s: str) -> tuple[int, int, int]:
    parts = s.split(',')
    if len(parts) != 3:
        raise SystemExit(f'--tear-color expects "R,G,B", got {s!r}')
    try:
        vals = tuple(int(p.strip()) for p in parts)
    except ValueError:
        raise SystemExit(f'--tear-color values must be integers, got {s!r}')
    if not all(0 <= v <= 255 for v in vals):
        raise SystemExit(f'--tear-color values must be in [0,255], got {s!r}')
    return vals


def build_stack(names, args):
    stack = []
    for name in names:
        if name == 'clip_cut':
            if not args.clip_cut_with:
                raise SystemExit('clip_cut requires --clip-cut-with PATH')
            stack.append((name, PARAMETRIC['clip_cut'](args.clip_cut_with)))
        elif name == 'tear':
            if args.tear_source == 'clip' and not args.tear_clip:
                raise SystemExit('tear source=clip requires --tear-clip PATH')
            stack.append((name, PARAMETRIC['tear'](
                source=args.tear_source,
                clip_path=args.tear_clip,
                color=_parse_color(args.tear_color),
                opacity=args.tear_opacity,
                orientation=args.tear_orientation,
                length_pct=args.tear_length_pct,
                thickness_pct=args.tear_thickness_pct,
                count_pct=args.tear_count_pct,
                displace_max_pct=args.tear_displace_max_pct,
                delay_frames=args.tear_delay_frames,
                shape=args.tear_shape,
            )))
        elif name == 'pixel_scramble':
            stack.append((name, PARAMETRIC['pixel_scramble'](
                count_pct=args.pscramble_count_pct,
                size_pct=args.pscramble_size_pct,
            )))
        elif name == 'jpeg_corrupt':
            stack.append((name, PARAMETRIC['jpeg_corrupt'](
                count_pct=args.jpeg_count_pct,
                size_pct=args.jpeg_size_pct,
                quality=args.jpeg_quality,
            )))
        elif name == 'pixel_sort':
            stack.append((name, PARAMETRIC['pixel_sort'](
                sort_by=args.psort_sort_by,
                direction=args.psort_direction,
                lower_threshold=args.psort_lower,
                upper_threshold=args.psort_upper,
                reverse=args.psort_reverse,
                angle=args.psort_angle,
                audio_scale=args.psort_audio_scale,
            )))
        elif name in REGISTRY:
            stack.append((name, REGISTRY[name]()))
        else:
            raise SystemExit(f'unknown effect: {name}')
    return stack


def render(source_path: str, out_path: str, effects: list[str],
           stack_args, audio_path: str | None = None,
           seed: int | None = None, crf: int = 18,
           profile_path: str | None = None,
           accumulate: bool = False, accumulate_reset_sec: float = 2.0,
           accumulate_bleed: float = 0.0):
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

    stack = build_stack(effects, stack_args)
    print(f'[stack] {[n for n, _ in stack]}')

    reader, frame_nbytes = open_reader(source_path, meta)
    writer = open_writer(source_path, out_path, meta, crf=crf)

    print(f'[render] writing {out_path}')
    if accumulate:
        reset_every = max(1, int(round(accumulate_reset_sec * meta.fps)))
        bleed = float(np.clip(accumulate_bleed, 0.0, 1.0))
        print(f'[render] accumulate mode: reset every {reset_every} frames '
              f'({accumulate_reset_sec:.2f}s) or on section_change; '
              f'bleed={bleed:.2f}')
    persistent: np.ndarray | None = None
    frames_since_reset = 0
    t0 = time.time()
    i = 0
    try:
        while True:
            raw = reader.stdout.read(frame_nbytes)
            if not raw or len(raw) < frame_nbytes:
                break
            source_frame = np.frombuffer(raw, dtype=np.uint8).reshape(meta.height, meta.width, 3).copy()

            feats = af.features_at(i)
            feats['section_change'] = i in section_frames

            if accumulate:
                want_reset = (i == 0
                              or feats['section_change']
                              or frames_since_reset >= reset_every)
                if want_reset:
                    persistent = source_frame
                    frames_since_reset = 0
                if bleed > 0.0:
                    # Mix fresh source into accumulated state so animation flows
                    # through while old damage persists.
                    blended = (bleed * source_frame.astype(np.float32)
                               + (1.0 - bleed) * persistent.astype(np.float32))
                    frame = np.clip(blended, 0, 255).astype(np.uint8)
                else:
                    frame = persistent.copy()
            else:
                frame = source_frame

            for _, eff in stack:
                frame = eff.step(frame, feats)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)

            if accumulate:
                persistent = frame
                frames_since_reset += 1

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
        'settings': {'seed': seed, 'effects': list(effects)},
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
    effect_names = sorted(list(REGISTRY.keys()) + list(PARAMETRIC.keys()))
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
    p.add_argument('--accumulate', action='store_true',
                   help='effects compound frame-over-frame instead of starting fresh. '
                        'Between resets the source "freezes" and damage accumulates, '
                        'producing true destruction rather than per-frame overlay.')
    p.add_argument('--accumulate-reset-sec', type=float, default=2.0,
                   help='[accumulate mode] max seconds between resets to fresh source '
                        '(section_change also triggers reset). default 2.0')
    p.add_argument('--accumulate-bleed', type=float, default=0.0,
                   help='[accumulate mode] fraction 0..1 of fresh source mixed in per frame. '
                        '0.0 = pure freeze (default), 0.3 = animation flows through while '
                        'old damage lingers, 1.0 = no accumulation')

    cut_grp = p.add_argument_group('clip_cut options')
    cut_grp.add_argument('--clip-cut-with', default=None,
                         help='secondary video path; required when "clip_cut" is in --effects')

    tear_grp = p.add_argument_group('tear options')
    tear_grp.add_argument('--tear-source', choices=['displace', 'clip', 'color'],
                          default='displace',
                          help='what fills the tear blocks (default "displace" = shifted base; '
                               '"clip" = secondary video; "color" = solid color)')
    tear_grp.add_argument('--tear-clip', default=None,
                          help='secondary video path when --tear-source=clip')
    tear_grp.add_argument('--tear-color', default='255,0,0',
                          help='"R,G,B" triple when --tear-source=color (default 255,0,0)')
    tear_grp.add_argument('--tear-opacity', type=float, default=1.0,
                          help='blend amount 0.0..1.0 (default 1.0 = hard replace)')
    tear_grp.add_argument('--tear-orientation', choices=['horizontal', 'vertical'],
                          default='horizontal',
                          help='strip orientation (default horizontal)')
    tear_grp.add_argument('--tear-length-pct', type=float, default=80.0,
                          help='max tear length as %% of axis from the originating edge; '
                               'audio scales 0..this (default 80). Per-tear length is '
                               'biased toward the max so trails stay visibly long.')
    tear_grp.add_argument('--tear-thickness-pct', type=float, default=25.0,
                          help='max block thickness (perpendicular to strip direction) as %% '
                               'of the other axis; audio scales 0..this (default 25)')
    tear_grp.add_argument('--tear-count-pct', type=float, default=5.0,
                          help='max block count as %% of thickness axis; audio scales 0..this '
                               '(default 5)')
    tear_grp.add_argument('--tear-displace-max-pct', type=float, default=25.0,
                          help='[displace source] max random block offset as %% of axis; '
                               'audio scales 0..this (default 25)')
    tear_grp.add_argument('--tear-delay-frames', type=int, default=1,
                          help='only generate new tears every N frames (default 1 = every '
                               'frame). Higher values produce rhythmic tear bursts rather '
                               'than continuous damage.')
    tear_grp.add_argument('--tear-shape', choices=['band', 'square', 'circle'],
                          default='band',
                          help='tear region shape: band (default, edge-anchored strips), '
                               'square (axis-aligned squares at random positions), '
                               'circle (disks at random positions). For square/circle, '
                               'length_pct controls max side/diameter and orientation/'
                               'thickness_pct are ignored.')

    ps_grp = p.add_argument_group('pixel_scramble options')
    ps_grp.add_argument('--pscramble-count-pct', type=float, default=3.0,
                        help='max scramble block count as %% of height; audio scales 0..this '
                             '(default 3)')
    ps_grp.add_argument('--pscramble-size-pct', type=float, default=15.0,
                        help='max block size as %% of min(W,H); audio scales 0..this '
                             '(default 15)')

    jp_grp = p.add_argument_group('jpeg_corrupt options')
    jp_grp.add_argument('--jpeg-count-pct', type=float, default=3.0,
                        help='max corruption block count as %% of height; audio scales 0..this '
                             '(default 3)')
    jp_grp.add_argument('--jpeg-size-pct', type=float, default=20.0,
                        help='max block size as %% of min(W,H); audio scales 0..this '
                             '(default 20)')
    jp_grp.add_argument('--jpeg-quality', type=int, default=5,
                        help='JPEG quality level inside corrupted blocks (1=extreme artifacts, '
                             '100=clean). default 5')

    so_grp = p.add_argument_group('pixel_sort options')
    so_grp.add_argument('--psort-sort-by', choices=['brightness', 'hue', 'saturation'],
                        default='brightness',
                        help='property to sort pixels by (default brightness)')
    so_grp.add_argument('--psort-direction', choices=['horizontal', 'vertical', 'both'],
                        default='horizontal',
                        help='sort direction (default horizontal)')
    so_grp.add_argument('--psort-lower', type=float, default=0.25,
                        help='lower threshold; pixels below are not sorted (default 0.25)')
    so_grp.add_argument('--psort-upper', type=float, default=0.8,
                        help='upper threshold; pixels above are not sorted (default 0.8)')
    so_grp.add_argument('--psort-reverse', action='store_true',
                        help='sort descending instead of ascending')
    so_grp.add_argument('--psort-angle', type=int, default=0,
                        help='rotation in degrees before sort (e.g., 45 for diagonal streaks)')
    so_grp.add_argument('--psort-audio-scale', type=float, default=0.5,
                        help='how much audio activity widens the threshold window (0=no '
                             'reactivity, 1=full). default 0.5')

    args = p.parse_args()

    valid = set(REGISTRY) | set(PARAMETRIC)
    unknown = [e for e in args.effects if e not in valid]
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
           stack_args=args, audio_path=args.audio,
           seed=args.seed, crf=args.crf, profile_path=profile_path,
           accumulate=args.accumulate,
           accumulate_reset_sec=args.accumulate_reset_sec,
           accumulate_bleed=args.accumulate_bleed)


if __name__ == '__main__':
    main()
