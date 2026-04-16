"""Build a static HTML visual guide for every effect + shipped recipe.

Renders one short clip per registered effect (with default params) plus one
clip per recipe in glitch_maker/recipes/, then emits a dark-themed HTML page
with hover-to-play video cards grouped by section. Run from the glitch_maker/
directory.

    python3 build_effects_guide.py [--output-dir DIR] [--source VIDEO] [--audio WAV]
"""
from __future__ import annotations
import argparse
import html
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from effects import PARAMETRIC


DEFAULT_SOURCE = 'test/sunrise_5s.mp4'
DEFAULT_AUDIO = (
    '/home/mastergray/codex/seamless-disarray/songs/clips/test/archive/'
    'LANDR-10-11-25-2__00m00s_00m20s.wav'
)

EFFECT_DESCRIPTIONS = {
    'noise':           'Per-pixel random perturbation across the whole frame. Gaussian '
                       '(soft film grain) or uniform (harsher static); monochrome '
                       '(luminance-only) or per-channel (colored TV static).',
    'tear':            'Edge-anchored signal-tear bands; shape can be band, square, or circle. '
                       'Source fills the torn region with shifted base pixels, a secondary clip, '
                       'or a solid color.',
    'pixel_sort':      'Per-row or per-column sort by brightness/hue/saturation within a '
                       'threshold range. Classic "melt" streaks — especially strong with '
                       '--accumulate.',
    'pixel_scramble':  'Shuffle pixel positions within random blocks. Keeps each block\'s '
                       'average color but destroys structure — reads as localized data loss.',
    'jpeg_corrupt':    'Re-encode random blocks as heavily compressed JPEG for real DCT '
                       'macroblock artifacts. The "actual datamosh" look.',
    'chromatic':       'RGB channel offset for color-fringe aberration. Direction can be '
                       'horizontal, vertical, or circular orbit.',
    'databend':        'Raw byte-level corruption of frame data — shift, XOR, and/or swap '
                       'random chunks. Audio drives chunk count and size.',
    'clip_cut':        'Temporal jump-cut to a secondary clip on analyzer-detected section '
                       'changes. Cutup / breakbeat-video feel without manual editing.',
    'channel_swap':    'Swap or rotate RGB channels in random blocks for sudden neon color '
                       'pockets against the untouched frame.',
    'bit_crush':       'Quantize color depth in random blocks by masking low bits. '
                       'Posterized patches next to full-color regions.',
    'invert_blocks':   'Negate color (255 - value) in random rectangles. Stark tonal-flip '
                       'zones.',
    'slitscan':        'Each row (or column) shows content from a different past frame. '
                       'Time-warped motion smearing — stateful across frames.',
    'warp':            'Sine-wave displacement across rows/cols/both. Underwater, heat-haze, '
                       'or analog-tape wobble.',
}
# insert noise ordering is lexicographic via PARAMETRIC sort — guide auto-picks


# Demo-tuned flag overrides per effect so the guide reads clearly on 5s of
# real footage. Defaults tuned for general use are often too subtle for a
# 5-second loop without accumulation; these push each effect toward the
# "obviously this is the effect" end of its range.
DEMO_FLAGS = {
    'tear': [
        '--tear-count-pct', '10',
        '--tear-thickness-pct', '30',
        '--tear-length-pct', '90',
        '--tear-displace-max-pct', '35',
        '--accumulate', '--accumulate-bleed', '0.15', '--accumulate-reset-sec', '3',
    ],
    'pixel_sort': [
        '--psort-direction', 'vertical',
        '--psort-lower', '0.1',
        '--psort-upper', '0.9',
        '--psort-audio-scale', '0.3',
        '--accumulate', '--accumulate-bleed', '0.2', '--accumulate-reset-sec', '3',
    ],
    'pixel_scramble': [
        '--pscramble-count-pct', '8',
        '--pscramble-size-pct', '25',
    ],
    'jpeg_corrupt': [
        '--jpeg-count-pct', '10',
        '--jpeg-size-pct', '30',
        '--jpeg-quality', '3',
    ],
    'chromatic': [
        '--chroma-amount', '40',
        '--chroma-direction', 'circular',
        '--chroma-audio-scale', '0.5',
    ],
    'databend': [
        '--databend-count', '30',
        '--databend-chunk-size', '4096',
    ],
    'channel_swap': [
        '--cswap-count-pct', '12',
        '--cswap-size-pct', '30',
    ],
    'bit_crush': [
        '--bcrush-count-pct', '10',
        '--bcrush-size-pct', '30',
        '--bcrush-bits', '1',
    ],
    'invert_blocks': [
        '--invert-count-pct', '10',
        '--invert-size-pct', '30',
    ],
    'slitscan': [
        '--slitscan-max-delay', '60',
        '--slitscan-audio-scale', '0.2',
    ],
    'warp': [
        '--warp-amplitude', '50',
        '--warp-frequency', '0.025',
        '--warp-direction', 'both',
    ],
    'noise': [
        '--noise-amount', '30',
        '--noise-type', 'gaussian',
    ],
}


RECIPE_DESCRIPTIONS = {
    'subtle':       'Light chromatic fringing plus sparse, delayed tears. Barely-there '
                    'glitch accents — good starting point for music-video punches.',
    'signal_tear':  'Classic edge-anchored signal-tear bands with mild accumulation bleed. '
                    'Chromatic stacked on top for color fringing at tear boundaries.',
    'datamosh':     'Heavy tear + pixel_scramble with pure-freeze accumulation. Each '
                    'reset cycle the source freezes and destructs before snapping clean.',
    'melt':         'Vertical brightness pixel_sort with accumulation — the frame drips '
                    'downward into painted streaks that rebuild after each reset.',
    'destroy':      'Maximum stack: wide tear + chunky scramble + chromatic, long '
                    'accumulation window for near-total destruction before reset.',
    'vhs':          'Analog tape look — heavy horizontal chromatic bleed, full-width '
                    'scanline bands, monochrome gaussian noise for film grain, and a '
                    'subtle horizontal weave via warp. No discrete block patches.',
    'underwater':   'Warp-driven organic wobble, both axes, with a hint of circular '
                    'chromatic. No accumulation; the frame stays readable.',
    'scan_lines':   'Many thin horizontal tear-bands with tight spacing — reads as '
                    'CRT-style scanline glitching with horizontal chromatic tearing.',
    'time_smear':   'Pure slitscan with strong accumulation — time-warped motion drags '
                    'become ghost trails that persist across the full frame.',
    'neon_blocks':  'channel_swap (rotate) + harsh bit_crush + sparse invert_blocks, all '
                    'as horizontal bands now for full-width scanline corruption.',
    'signal_crush': 'tear bands + bit_crush bands + chromatic — horizontal strips get '
                    'both displaced AND harshly quantized. Signal tear where the tear '
                    'itself is data-crushed.',
    'rainbow_sort': 'pixel_sort by hue across the full threshold range — sunrise colors '
                    'rearrange into rainbow-banded streaks with accumulation carry.',
    'sync_loss':    'Heavy signal-tear — many thick displacement bands across the full '
                    'width with horizontal chromatic bleed. Analog TV losing sync.',
    'breakpoint':   'Rhythmic destruction via delay_frames — tear and scramble fire in '
                    'sync every 20 frames, accumulating damage between bursts.',
    'oblivion':     'Maximum stack with pure-freeze accumulation and long reset. Tear + '
                    'scramble + chromatic + databend all compounding. Total destruction.',
}


# Tear + bleed showcase uses a high-contrast source (centered subject on black)
# where tear displacements create visible contrast breaks. Sunrise gradient
# footage hides tear effects because displaced rows look like adjacent rows.
TEAR_SHOWCASE_SOURCE = 'test/atom_5s.mp4'
TEAR_SHOWCASE = [
    # (slug, label, description, extra CLI flags to pass render.py)
    ('bleed_subtle', 'tear · bleed 0.3 (fast decay)',
     'Medium tear with HIGH bleed (0.3). Damage fades in ~3 frames, so tears flash '
     'briefly and the source keeps flowing. Most subtle ghost feel.',
     ['--effects', 'tear',
      '--tear-count-pct', '8', '--tear-thickness-pct', '25', '--tear-length-pct', '80',
      '--tear-displace-max-pct', '30',
      '--accumulate', '--accumulate-bleed', '0.3', '--accumulate-reset-sec', '3']),
    ('bleed_medium', 'tear · bleed 0.15 (sweet spot)',
     'Moderate tear + medium bleed — the setting we iterated toward. Edge-anchored '
     'tears persist as ghost trails for ~10 frames before fading. Animation still '
     'visible underneath.',
     ['--effects', 'tear',
      '--tear-count-pct', '10', '--tear-thickness-pct', '30', '--tear-length-pct', '90',
      '--tear-displace-max-pct', '35',
      '--accumulate', '--accumulate-bleed', '0.15', '--accumulate-reset-sec', '3']),
    ('bleed_heavy', 'tear · bleed 0.05 (persistent)',
     'Heavy tear + very low bleed — old damage persists for ~50 frames. '
     'Ghost trails dominate and compound until the reset snaps the frame clean.',
     ['--effects', 'tear',
      '--tear-count-pct', '12', '--tear-thickness-pct', '35', '--tear-length-pct', '90',
      '--tear-displace-max-pct', '40',
      '--accumulate', '--accumulate-bleed', '0.05', '--accumulate-reset-sec', '4']),
    ('bursts', 'tear · rhythmic bursts',
     'delay_frames=30 + bleed 0.15 — tear only fires every 0.5s, producing '
     'punctuated damage waves that accumulate briefly then fade. Feels like glitch '
     'hits landing on the beat.',
     ['--effects', 'tear',
      '--tear-count-pct', '15', '--tear-thickness-pct', '35', '--tear-length-pct', '100',
      '--tear-displace-max-pct', '40', '--tear-delay-frames', '30',
      '--accumulate', '--accumulate-bleed', '0.15', '--accumulate-reset-sec', '4']),
    ('recipe_signal_tear', 'recipe · signal_tear',
     'The signal_tear recipe (tear + chromatic + mild bleed) on atom. Compare to '
     'the sunrise version in the Recipes section — much more dramatic here.',
     ['--config', 'recipes/signal_tear.json']),
    ('recipe_datamosh', 'recipe · datamosh',
     'datamosh recipe (tear + scramble, pure-freeze) on atom. Each 2s window the '
     'source freezes and progressively shreds.',
     ['--config', 'recipes/datamosh.json']),
    ('recipe_sync_loss', 'recipe · sync_loss',
     'sync_loss recipe on atom. Thick displacement bands with chromatic fringing; '
     'the atom visibly fragments along the torn rows.',
     ['--config', 'recipes/sync_loss.json']),
    ('recipe_oblivion', 'recipe · oblivion',
     'oblivion recipe on atom — maximum accumulation. By the end of each reset '
     'window the atom is reduced to fragments/noise.',
     ['--config', 'recipes/oblivion.json']),
]


SHAPE_SHOWCASE = [
    ('bcrush_rectangle', 'bit_crush · rectangle',
     'Block-based effects all support --shape rectangle (default). '
     'Random w/h blocks scattered across the frame.',
     ['--effects', 'bit_crush', '--bcrush-shape', 'rectangle',
      '--bcrush-count-pct', '10', '--bcrush-size-pct', '30', '--bcrush-bits', '1']),
    ('bcrush_square', 'bit_crush · square',
     '--shape square forces equal width and height. '
     'More uniform/structural than random rectangles.',
     ['--effects', 'bit_crush', '--bcrush-shape', 'square',
      '--bcrush-count-pct', '10', '--bcrush-size-pct', '30', '--bcrush-bits', '1']),
    ('bcrush_circle', 'bit_crush · circle',
     '--shape circle uses a circular mask so the effect only affects the disk '
     'interior. Soft/organic feel vs hard rectangle edges.',
     ['--effects', 'bit_crush', '--bcrush-shape', 'circle',
      '--bcrush-count-pct', '10', '--bcrush-size-pct', '30', '--bcrush-bits', '1']),
]


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>glitch-maker — effects guide</title>
<style>
  :root {{
    --bg: #0d0d11;
    --fg: #e8e8ea;
    --muted: #8a8a92;
    --accent: #ff5e8a;
    --accent2: #6ad1ff;
    --card-bg: #181820;
    --card-border: #2a2a35;
  }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--fg);
    font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; }}
  header {{ padding: 2rem 2rem 1.25rem; border-bottom: 1px solid var(--card-border); }}
  header h1 {{ margin: 0; font-size: 1.8rem; letter-spacing: -0.02em; }}
  header p {{ margin: 0.5rem 0 0; color: var(--muted); font-size: 0.95rem; max-width: 72ch;
    line-height: 1.5; }}
  main {{ padding: 1.5rem 2rem; }}
  section {{ margin-bottom: 2.5rem; }}
  section h2 {{ margin: 0 0 1rem; font-size: 1.2rem; color: var(--accent); letter-spacing: 0.02em;
    text-transform: uppercase; font-weight: 600; }}
  .grid {{
    display: grid; gap: 1.25rem;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  }}
  .card {{
    background: var(--card-bg); border: 1px solid var(--card-border);
    border-radius: 8px; overflow: hidden;
    display: flex; flex-direction: column;
    transition: transform 0.15s, border-color 0.15s;
  }}
  .card:hover {{ transform: translateY(-2px); border-color: var(--accent2); }}
  .card video {{ width: 100%; display: block; background: #000; cursor: pointer; }}
  .card-body {{ padding: 0.85rem 1rem 1rem; }}
  .card-body h3 {{ margin: 0 0 0.4rem; font-size: 1rem; color: var(--accent2);
    font-family: 'JetBrains Mono', Menlo, monospace; }}
  .card-body p {{ margin: 0; color: var(--muted); font-size: 0.88rem; line-height: 1.5; }}
  footer {{ padding: 1.5rem 2rem 2rem; color: var(--muted); font-size: 0.85rem;
    border-top: 1px solid var(--card-border); }}
  code {{ font-family: 'JetBrains Mono', Menlo, monospace; background: #222230;
    padding: 0.1rem 0.35rem; border-radius: 3px; font-size: 0.85em; }}
</style>
</head>
<body>
<header>
  <h1>glitch-maker — effects guide</h1>
  <p>{effect_count} effects + {recipe_count} curated recipes. Each clip below applies
    a single effect (or recipe) to the same {length}s source footage. Hover a tile to
    play; click to toggle controls. Invoke effects via
    <code>--effects NAME [NAME ...]</code> with per-effect flags, or load a recipe via
    <code>--config recipes/NAME.json</code>.</p>
</header>
<main>
  <section>
    <h2>Effects</h2>
    <div class="grid">
{effect_cards}
    </div>
  </section>
  <section>
    <h2>Shape variants</h2>
    <div class="grid">
{shape_cards}
    </div>
  </section>
  <section>
    <h2>Recipes</h2>
    <div class="grid">
{recipe_cards}
    </div>
  </section>
  <section>
    <h2>Tear showcase <span style="color:var(--muted);font-size:0.9rem;font-weight:400;text-transform:none;">on high-contrast source</span></h2>
    <p style="color:var(--muted);margin:-0.2rem 0 1rem;max-width:72ch;line-height:1.5;">
      Tear displacements need contrast to read — shifting rows of smooth sky
      looks the same as adjacent rows. These clips use an abstract source
      (centered subject on black) where tear + accumulate-bleed shows the
      signature edge-anchored ghost trails we iterated toward.
    </p>
    <div class="grid">
{tear_cards}
    </div>
  </section>
</main>
<footer>
  Generated {stamp} &middot; source: {source} &middot; audio: {audio}
</footer>
</body>
</html>
"""

CARD_TEMPLATE = """      <div class="card">
        <video src="clips/{clip}" loop muted playsinline preload="metadata"
               onmouseenter="this.play()"
               onmouseleave="this.pause(); this.currentTime = 0;"
               onclick="this.controls = !this.controls;"></video>
        <div class="card-body">
          <h3>{name}</h3>
          <p>{description}</p>
        </div>
      </div>"""


def render_effect_clip(source: str, audio: str, effect: str, out_path: Path, seed: int):
    cmd = [
        'python3', 'render.py', source, str(out_path),
        '--effects', effect,
        '--audio', audio,
        '--seed', str(seed),
        '--no-profile',
    ]
    cmd.extend(DEMO_FLAGS.get(effect, []))
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)


def render_recipe_clip(source: str, audio: str, recipe_path: str, out_path: Path, seed: int):
    subprocess.check_call([
        'python3', 'render.py', source, str(out_path),
        '--config', recipe_path,
        '--audio', audio,
        '--seed', str(seed),
        '--no-profile',
    ], stdout=subprocess.DEVNULL)


def render_shape_clip(source: str, audio: str, extra_flags: list, out_path: Path, seed: int):
    subprocess.check_call([
        'python3', 'render.py', source, str(out_path),
        '--audio', audio,
        '--seed', str(seed),
        '--no-profile',
        *extra_flags,
    ], stdout=subprocess.DEVNULL)


def build_index(output_dir: Path, effects: list[str], recipes: list[str],
                shapes: list[tuple], tears: list[tuple],
                length: float, source: str, audio: str):
    def card(name: str, clip: str, description: str) -> str:
        return CARD_TEMPLATE.format(
            name=html.escape(name),
            clip=html.escape(clip),
            description=html.escape(description),
        )

    effect_cards = '\n'.join(
        card(fx, f'fx_{fx}.mp4', EFFECT_DESCRIPTIONS.get(fx, 'No description.'))
        for fx in effects
    )
    recipe_cards = '\n'.join(
        card(r, f'recipe_{r}.mp4', RECIPE_DESCRIPTIONS.get(r, 'No description.'))
        for r in recipes
    )
    shape_cards = '\n'.join(
        card(label, f'shape_{slug}.mp4', desc)
        for slug, label, desc, _ in shapes
    )
    tear_cards = '\n'.join(
        card(label, f'tear_{slug}.mp4', desc)
        for slug, label, desc, _ in tears
    )
    page = HTML_TEMPLATE.format(
        effect_count=len(effects),
        recipe_count=len(recipes),
        length=round(length, 1),
        effect_cards=effect_cards,
        shape_cards=shape_cards,
        recipe_cards=recipe_cards,
        tear_cards=tear_cards,
        stamp=time.strftime('%Y-%m-%d'),
        source=html.escape(os.path.basename(source)),
        audio=html.escape(os.path.basename(audio)),
    )
    (output_dir / 'index.html').write_text(page)


def main() -> int:
    p = argparse.ArgumentParser(description='Build HTML visual guide of glitch-maker effects')
    p.add_argument('--output-dir', default='./effects-guide',
                   help='directory to write index.html + clips/ (default: ./effects-guide)')
    p.add_argument('--source', default=DEFAULT_SOURCE,
                   help=f'source video (default: {DEFAULT_SOURCE})')
    p.add_argument('--audio', default=DEFAULT_AUDIO,
                   help='audio override (default: LANDR test clip)')
    p.add_argument('--length', type=float, default=3.0,
                   help='per-card clip length in seconds (default 3.0)')
    p.add_argument('--seed', type=int, default=42,
                   help='seed for reproducibility (default 42)')
    p.add_argument('--skip-existing', action='store_true',
                   help='skip clips that already exist')
    p.add_argument('--only-fx', nargs='+', default=None,
                   help='render only these effects (for iteration)')
    p.add_argument('--only-recipes', nargs='+', default=None,
                   help='render only these recipes (for iteration)')
    p.add_argument('--skip-fx', action='store_true',
                   help='skip the effects section entirely')
    p.add_argument('--skip-recipes', action='store_true',
                   help='skip the recipes section entirely')
    args = p.parse_args()

    src = Path(args.source).resolve()
    if not src.exists():
        print(f'error: source video not found: {src}', file=sys.stderr)
        return 1
    if not Path(args.audio).exists():
        print(f'error: audio not found: {args.audio}', file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()
    clips_dir = output_dir / 'clips'
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Pre-slice source to --length so each per-effect render only processes the
    # exact duration we want to showcase. One slice, reused for every card.
    sliced_src = output_dir / f'_source_{int(args.length * 1000)}ms.mp4'
    if not sliced_src.exists() or args.skip_existing is False:
        print(f'[guide] slicing source to {args.length}s -> {sliced_src}')
        subprocess.check_call([
            'ffmpeg', '-y', '-v', 'error',
            '-i', str(src), '-t', str(args.length), '-c', 'copy',
            str(sliced_src),
        ])
    src = sliced_src

    # Discover effects (skip clip_cut — needs a secondary video; not suited to default guide)
    all_effects = sorted(fx for fx in PARAMETRIC if fx != 'clip_cut')
    effects = [fx for fx in all_effects if args.only_fx is None or fx in args.only_fx]
    if args.skip_fx:
        effects = []

    # Discover recipes
    recipes_dir = Path(__file__).parent / 'recipes'
    all_recipes = sorted(p.stem for p in recipes_dir.glob('*.json'))
    recipes = [r for r in all_recipes if args.only_recipes is None or r in args.only_recipes]
    if args.skip_recipes:
        recipes = []

    missing_desc_fx = [f for f in effects if f not in EFFECT_DESCRIPTIONS]
    missing_desc_recipe = [r for r in recipes if r not in RECIPE_DESCRIPTIONS]
    if missing_desc_fx:
        print(f'warning: missing fx descriptions: {missing_desc_fx}', file=sys.stderr)
    if missing_desc_recipe:
        print(f'warning: missing recipe descriptions: {missing_desc_recipe}', file=sys.stderr)

    print(f'[guide] source: {src}')
    print(f'[guide] audio:  {args.audio}')
    print(f'[guide] rendering {len(effects)} effects + {len(recipes)} recipes')
    t0 = time.time()

    for i, fx in enumerate(effects, 1):
        clip_path = clips_dir / f'fx_{fx}.mp4'
        if args.skip_existing and clip_path.exists():
            print(f'  [{i:2}/{len(effects)}] fx {fx}: skip (exists)')
            continue
        ts = time.time()
        try:
            render_effect_clip(str(src), args.audio, fx, clip_path, args.seed)
            print(f'  [{i:2}/{len(effects)}] fx {fx}: {time.time() - ts:.1f}s')
        except subprocess.CalledProcessError as e:
            print(f'  [{i:2}/{len(effects)}] fx {fx}: FAILED ({e})')

    for i, r in enumerate(recipes, 1):
        clip_path = clips_dir / f'recipe_{r}.mp4'
        recipe_path = recipes_dir / f'{r}.json'
        if args.skip_existing and clip_path.exists():
            print(f'  [{i:2}/{len(recipes)}] recipe {r}: skip (exists)')
            continue
        ts = time.time()
        try:
            render_recipe_clip(str(src), args.audio, str(recipe_path), clip_path, args.seed)
            print(f'  [{i:2}/{len(recipes)}] recipe {r}: {time.time() - ts:.1f}s')
        except subprocess.CalledProcessError as e:
            print(f'  [{i:2}/{len(recipes)}] recipe {r}: FAILED ({e})')

    for i, (slug, label, _desc, flags) in enumerate(SHAPE_SHOWCASE, 1):
        clip_path = clips_dir / f'shape_{slug}.mp4'
        if args.skip_existing and clip_path.exists():
            print(f'  [{i:2}/{len(SHAPE_SHOWCASE)}] shape {slug}: skip (exists)')
            continue
        ts = time.time()
        try:
            render_shape_clip(str(src), args.audio, flags, clip_path, args.seed)
            print(f'  [{i:2}/{len(SHAPE_SHOWCASE)}] shape {slug}: {time.time() - ts:.1f}s')
        except subprocess.CalledProcessError as e:
            print(f'  [{i:2}/{len(SHAPE_SHOWCASE)}] shape {slug}: FAILED ({e})')

    # Tear showcase uses a different source + its baked-in audio (atom clip).
    tear_src_path = Path(TEAR_SHOWCASE_SOURCE).resolve()
    if not tear_src_path.exists():
        print(f'warning: tear showcase source not found: {tear_src_path}', file=sys.stderr)
    else:
        for i, (slug, label, _desc, flags) in enumerate(TEAR_SHOWCASE, 1):
            clip_path = clips_dir / f'tear_{slug}.mp4'
            if args.skip_existing and clip_path.exists():
                print(f'  [{i:2}/{len(TEAR_SHOWCASE)}] tear {slug}: skip (exists)')
                continue
            ts = time.time()
            try:
                # Atom clip has baked-in audio — no --audio override needed.
                subprocess.check_call([
                    'python3', 'render.py', str(tear_src_path), str(clip_path),
                    '--seed', str(args.seed), '--no-profile', *flags,
                ], stdout=subprocess.DEVNULL)
                print(f'  [{i:2}/{len(TEAR_SHOWCASE)}] tear {slug}: {time.time() - ts:.1f}s')
            except subprocess.CalledProcessError as e:
                print(f'  [{i:2}/{len(TEAR_SHOWCASE)}] tear {slug}: FAILED ({e})')

    # Determine clip length from first rendered mp4
    probe_target = None
    for fx in effects:
        if (clips_dir / f'fx_{fx}.mp4').exists():
            probe_target = clips_dir / f'fx_{fx}.mp4'
            break
    if probe_target is None:
        for r in recipes:
            if (clips_dir / f'recipe_{r}.mp4').exists():
                probe_target = clips_dir / f'recipe_{r}.mp4'
                break
    length = 5.0
    if probe_target:
        try:
            out = subprocess.check_output(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'csv=p=0', str(probe_target)], text=True,
            ).strip()
            length = float(out)
        except Exception:
            pass

    build_index(output_dir, effects, recipes, SHAPE_SHOWCASE, TEAR_SHOWCASE,
                length, str(src), args.audio)
    print(f'[guide] wrote {output_dir / "index.html"}')
    print(f'[guide] total {time.time() - t0:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
