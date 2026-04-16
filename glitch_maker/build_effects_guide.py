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
}


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
    <h2>Recipes</h2>
    <div class="grid">
{recipe_cards}
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
    subprocess.check_call([
        'python3', 'render.py', source, str(out_path),
        '--effects', effect,
        '--audio', audio,
        '--seed', str(seed),
        '--no-profile',
    ], stdout=subprocess.DEVNULL)


def render_recipe_clip(source: str, audio: str, recipe_path: str, out_path: Path, seed: int):
    subprocess.check_call([
        'python3', 'render.py', source, str(out_path),
        '--config', recipe_path,
        '--audio', audio,
        '--seed', str(seed),
        '--no-profile',
    ], stdout=subprocess.DEVNULL)


def build_index(output_dir: Path, effects: list[str], recipes: list[str],
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
    page = HTML_TEMPLATE.format(
        effect_count=len(effects),
        recipe_count=len(recipes),
        length=round(length, 1),
        effect_cards=effect_cards,
        recipe_cards=recipe_cards,
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

    build_index(output_dir, effects, recipes, length, str(src), args.audio)
    print(f'[guide] wrote {output_dir / "index.html"}')
    print(f'[guide] total {time.time() - t0:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
