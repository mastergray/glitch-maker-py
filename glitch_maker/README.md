# glitch-maker — Audio-Reactive Video Glitch Effects

Python tool for applying destructive, audio-reactive glitch effects to video.
Reads a source MP4, routes frames through a stack of effects driven by audio
analysis (beats, onsets, sections, bass/mid/treble), and writes a new MP4
with the source audio preserved.

Built for the cases FFmpeg filter chains can't reach — real pixel sorting,
byte-level databending, actual JPEG corruption, temporal accumulation — while
keeping per-effect audio reactivity and frame-over-frame damage compounding.

Includes 13 effects, 15 shipped recipes (config-JSON presets), and an HTML
effects guide that showcases each effect and recipe on real footage.

---

## Requirements

- Python 3.8+
- `numpy`, `scipy`, `soundfile`, `Pillow` — `pip install numpy scipy soundfile Pillow`
- `ffmpeg` and `ffprobe` on PATH — both read and write pipelines use ffmpeg

The audio analyzer (`analyze_audio.py`, vendored from the dazzle project) is
called as a subprocess to generate beat/onset/section data for the source
audio. No additional install needed.

---

## Quick Start

Run everything from `glitch_maker/`. Simplest invocation:

```
python3 render.py input.mp4 output.mp4 --effects tear chromatic
```

This applies the `tear` and `chromatic` effects in sequence, using the
source video's own audio track for reactivity. The output includes the
original audio plus the visually-destroyed video.

For a config-based render (using a shipped preset):

```
python3 render.py input.mp4 output.mp4 --config recipes/signal_tear.json
```

---

## Usage

### Two invocation styles

**Direct CLI**: list effects and tune per-effect flags from the command line.

```
python3 render.py input.mp4 output.mp4 \
    --effects tear chromatic \
    --tear-count-pct 10 --tear-thickness-pct 30 \
    --chroma-amount 30
```

**Config file**: load a recipe JSON that declares the full effect stack.

```
python3 render.py input.mp4 output.mp4 --config recipes/datamosh.json
```

`--config` and `--effects` are mutually exclusive. Use `--config` for
reproducible/reusable setups, CLI for one-off exploration.

### Accumulation mode

Effects by default apply to each fresh source frame independently. With
`--accumulate`, the previous frame's output becomes the input for the next
frame's effects — so damage compounds over time. Periodic resets snap back
to fresh source to avoid terminal decay.

```
python3 render.py input.mp4 output.mp4 \
    --effects tear --accumulate \
    --accumulate-bleed 0.15 \
    --accumulate-reset-sec 3
```

- `--accumulate-bleed 0.0` = pure freeze mode. Source locks on reset frames;
  effects progressively destroy it until next reset. Very heavy destruction.
- `--accumulate-bleed 0.15` = ~10-frame damage half-life. Animation flows
  through while old damage lingers as ghost trails. Sweet spot for most cases.
- `--accumulate-bleed 0.3` = ~3-frame half-life. Damage flickers briefly then
  fades — subtle ghosting only.
- `--accumulate-bleed 1.0` = no accumulation (equivalent to `--accumulate` off).

Audio section-change events also trigger hard resets regardless of
`--accumulate-reset-sec`.

### Audio source

By default audio is extracted from the source video. Override with `--audio
PATH` if:

- Source has no audio track (render.py errors clearly if so)
- You want the glitch reactivity driven by a different audio than what's
  playing in the output

```
python3 render.py silent_footage.mp4 out.mp4 \
    --effects tear \
    --audio ~/songs/glitch_track.wav
```

The `--audio` file's audio track replaces the source's in the output MP4.

### Profile sidecar

Every render writes a JSON profile alongside the MP4 (unless
`--no-profile`) with the complete render configuration. Profile format
matches the `--config` schema, so **any written profile.json can be reused
as a recipe for a future render**.

```
python3 render.py v1.mp4 v1_out.mp4 --effects tear chromatic
# → v1_out.mp4 + v1_out.profile.json

python3 render.py v2.mp4 v2_out.mp4 --config v1_out.profile.json
# → identical effect stack applied to v2.mp4
```

---

## Examples

### Subtle vs aggressive

```
# Subtle — just a hint of color fringing
python3 render.py in.mp4 subtle.mp4 --config recipes/subtle.json

# Medium — clear signal tear + chromatic ghosts
python3 render.py in.mp4 medium.mp4 --config recipes/signal_tear.json

# Heavy — near-total destruction between resets
python3 render.py in.mp4 heavy.mp4 --config recipes/oblivion.json
```

### Stacking effects manually

Effects are applied in the order listed:

```
# Tear first, then chromatic on the torn result, then noise on top
python3 render.py in.mp4 stacked.mp4 \
    --effects tear chromatic noise \
    --tear-count-pct 10 --tear-thickness-pct 30 \
    --chroma-amount 40 \
    --noise-amount 20
```

Order matters: `tear chromatic` produces color fringes at tear boundaries;
`chromatic tear` tears an already-fringed frame. Different visual results.

### VHS look

```
python3 render.py in.mp4 vhs.mp4 --config recipes/vhs.json
```

Horizontal chromatic bleed + partial-width scan bands + film-grain noise +
subtle horizontal weave. No accumulation (real tape artifacts are per-frame,
not cumulative).

### Datamosh-style destruction

```
python3 render.py in.mp4 moshed.mp4 --config recipes/datamosh.json
```

Uses pure-freeze accumulation (`bleed=0`). Each 2-second window the source
freezes and gets progressively shredded by tear + pixel_scramble, then snaps
to a fresh frame and starts over.

### Iteration workflow

```
# 1. Try the default recipe
python3 render.py in.mp4 v1.mp4 --config recipes/destroy.json

# 2. Didn't hit hard enough — copy recipe, tune
cp recipes/destroy.json recipes/my_destroy.json
# edit my_destroy.json: bump tear-count, lower reset_sec
python3 render.py in.mp4 v2.mp4 --config recipes/my_destroy.json

# 3. Like v2 but want different seed / different audio
python3 render.py in.mp4 v3.mp4 --config v2.profile.json --seed 99
python3 render.py in.mp4 v4.mp4 --config v2.profile.json --audio other.wav
```

### Two clips, jump-cut between them

```
python3 render.py clip_a.mp4 out.mp4 \
    --effects clip_cut \
    --clip-cut-with clip_b.mp4
```

On each analyzer-detected section change, cuts to `clip_b.mp4` for ~30 frames,
then back to `clip_a.mp4`. Cutup/breakbeat-video editing without manual
timeline work.

---

## Effects reference

All 13 effects. Each section lists the effect's CLI flags and common params.

### tear

Spatial signal-tear: random strips/squares/circles of the frame get
displaced, filled with a secondary clip, or painted a solid color.
The original "tear-based destruction" effect we built most aggressively.

```
--tear-source {displace,clip,color}      what fills the torn regions
--tear-clip PATH                          [clip source] secondary video
--tear-color R,G,B                        [color source] fill color
--tear-opacity 0..1                       blend vs. source (1 = hard replace)
--tear-shape {band,square,circle}         tear region shape
--tear-orientation {horizontal,vertical}  [band] strip axis
--tear-edge {random,start,end}            [band] originating edge
--tear-length-pct N                       max tear length as % of axis
--tear-thickness-pct N                    max tear thickness (perpendicular)
--tear-count-pct N                        max tear count per frame
--tear-displace-max-pct N                 [displace source] max shift distance
--tear-displace-dir {any,positive,negative}  lock shift direction
--tear-delay-frames N                     fire only every N frames (rhythmic bursts)
```

`displace` is the most common source (pulls pixels from the same frame at a
random offset). Band + displace = classic signal tear. Square/circle spread
corruption patches around the frame.

### pixel_sort

Per-row (or column) Kim-Asendorf sort of contiguous pixel runs within a
brightness/hue/saturation threshold. Produces the characteristic melted/
streaky look.

```
--psort-sort-by {brightness,hue,saturation}    what to sort by
--psort-direction {horizontal,vertical,both}   axis of sort
--psort-lower 0..1                             lower threshold (pixels below skipped)
--psort-upper 0..1                             upper threshold (pixels above skipped)
--psort-reverse                                descending sort
--psort-angle N                                rotate frame before sort (45 = diagonal)
--psort-audio-scale 0..1                       how much audio widens the threshold window
```

Pairs extremely well with `--accumulate` — the sort result persists and
compounds into heavy drips/streaks.

### pixel_scramble

Shuffles pixel positions within random blocks. Each block keeps its average
color but loses all structure — reads as localized data loss.

```
--pscramble-count-pct N                        max block count as % of height
--pscramble-size-pct N                         max block size as % of min(W,H)
--pscramble-shape {rectangle,square,circle,band}
--pscramble-orientation {horizontal,vertical}  [band] strip axis
--pscramble-band-length-pct N                  [band] strip length %
--pscramble-delay-frames N                     rhythmic bursts
```

### jpeg_corrupt

Extracts random blocks, re-encodes them as heavily compressed JPEG, pastes
back. Produces authentic 8×8 DCT macroblock artifacts — the "real datamosh"
look. Slower than pixel_scramble due to PIL round-trip per block.

```
--jpeg-count-pct N                             max block count
--jpeg-size-pct N                              max block size
--jpeg-quality 1..100                          JPEG quality (lower = worse = more artifacts)
--jpeg-shape {rectangle,square,circle,band}
--jpeg-orientation {horizontal,vertical}
--jpeg-band-length-pct N
--jpeg-delay-frames N
```

### chromatic

RGB channel offset producing color-fringe aberration. Direction can be
horizontal, vertical, or a circular orbit.

```
--chroma-amount N                              max pixel offset at full audio
--chroma-direction {horizontal,vertical,circular}
--chroma-audio-scale 0..1                      how much audio drives amplitude
```

### databend

Raw byte-level corruption of the frame data. Random byte ranges get rolled,
XOR'd with a mask, or swapped with another range.

```
--databend-count N                             max chunks per frame at full audio
--databend-chunk-size N                        max bytes per chunk
--databend-modes                               comma-separated from: shift,xor,swap
--databend-audio-scale 0..1
```

Small chunks produce grain-like noise; large chunks produce chunky local
corruption. Small chunks + XOR mode approximates analog grain (but the
dedicated `noise` effect is better for that).

### channel_swap

Swap or rotate RGB channels within random blocks, producing neon color
pockets against the untouched frame.

```
--cswap-count-pct N                            max block count
--cswap-size-pct N                             max block size
--cswap-mode {random,rb,rg,gb,rotate}          channel permutation
--cswap-audio-scale 0..1
--cswap-shape {rectangle,square,circle,band}
--cswap-orientation {horizontal,vertical}
--cswap-band-length-pct N
--cswap-delay-frames N
```

### bit_crush

Quantizes color depth in random blocks by masking off low bits. At `bits=1`
each channel has only 2 values, giving 8 total colors — harsh posterized
patches against full-color regions.

```
--bcrush-count-pct N                           max block count
--bcrush-size-pct N                            max block size
--bcrush-bits 1..7                             bits per channel to keep
--bcrush-audio-scale 0..1
--bcrush-shape {rectangle,square,circle,band}
--bcrush-orientation {horizontal,vertical}
--bcrush-band-length-pct N
--bcrush-delay-frames N
```

### invert_blocks

Negates color (`255 - value`) inside random rectangles. Stark tonal flip
zones.

```
--invert-count-pct N
--invert-size-pct N
--invert-audio-scale 0..1
--invert-shape {rectangle,square,circle,band}
--invert-orientation {horizontal,vertical}
--invert-band-length-pct N
--invert-delay-frames N
```

### slitscan

Each row (or column) shows content from a different past frame, using a
rolling buffer of the last N frames. Produces time-warped motion smearing.

```
--slitscan-direction {horizontal,vertical}     rows-from-past vs cols-from-past
--slitscan-max-delay N                         max frames of history
--slitscan-audio-scale 0..1
```

Stateful across frames. Combines interestingly with `--accumulate` but can
get visually noisy.

### warp

Sine-wave displacement map. Rows shift left-right (or columns shift up-down)
by a wave of configurable amplitude/frequency/speed.

```
--warp-amplitude N                             max displacement in pixels
--warp-frequency N                             wave frequency (default 0.02)
--warp-speed N                                 wave phase advance per frame
--warp-direction {horizontal,vertical,both}
--warp-audio-scale 0..1
```

Default: subtle underwater wobble. Crank amplitude for cartoon-exaggerated
waves; `both` direction for full heat-haze distortion.

### noise

Per-pixel random value perturbation across the whole frame. Used for film
grain (mono) or analog TV static (per-channel).

```
--noise-amount N                               max amplitude in 0-255 units
--noise-type {gaussian,uniform}                distribution shape
--noise-mono / --no-noise-mono                 R=G=B (grain) vs per-channel (static)
--noise-audio-scale 0..1
```

### clip_cut

Temporal jump-cut to a second clip on analyzer-detected section changes.
Holds the secondary for ~30 frames then returns to primary. Unlike tear's
`clip` source, this is a full-frame temporal cut, not a spatial overlay.

```
--clip-cut-with PATH                           secondary video (required)
```

---

## Shipped recipes

15 curated configs in `glitch_maker/recipes/`. Use with
`--config recipes/NAME.json`.

| Recipe | Feel |
|---|---|
| `subtle` | Barely-there chromatic + delayed sparse tears |
| `underwater` | Warp-driven organic wobble, no accumulation |
| `signal_tear` | Classic edge-anchored signal-tear bands + chromatic |
| `scan_lines` | Dense thin horizontal tears, CRT-style |
| `vhs` | Horizontal chromatic bleed + bands + grain + weave |
| `time_smear` | Pure slitscan with strong accumulation |
| `rainbow_sort` | Full-threshold hue pixel_sort with accumulation |
| `melt` | Vertical brightness pixel_sort + accumulation (dripping melt) |
| `neon_blocks` | Partial-width channel-swap + bit_crush + invert bands |
| `datamosh` | Tear + scramble with pure-freeze accumulation |
| `sync_loss` | Thick signal-tear bands + chromatic (analog TV losing sync) |
| `breakpoint` | Rhythmic tear + scramble via delay_frames (damage pulses) |
| `signal_crush` | Tear bands + bit_crush bands + chromatic (tears that are data-crushed) |
| `destroy` | Maximum stack with long accumulation — near-total destruction |
| `oblivion` | Everything + pure-freeze accumulation — annihilation |

---

## Effects guide

Rendered HTML catalog showcasing each effect + recipe on a test source:

```
python3 build_effects_guide.py
```

Produces `effects-guide/index.html` with hover-to-play video cards grouped
by Effects / Shape variants / Recipes / Tear showcase (atom source).
Takes ~3-4 minutes at 1080p. Tunable flags:

```
--source PATH           change source video (default test/sunrise_5s.mp4)
--audio PATH            change audio (default LANDR test clip)
--length N              seconds per clip (default 3)
--skip-existing         skip clips that already exist
--only-fx NAME ...      render only specific effects
--only-recipes NAME ... render only specific recipes
```

The `test/` dir (and `effects-guide/` output) is gitignored — regenerate
locally as needed.

---

## Project structure

```
glitch_maker/
  render.py                  # main CLI, scheduler, ffmpeg pipeline
  audio.py                   # AudioFeatures loader
  analyze_audio.py           # vendored analyzer (beats, onsets, sections)
  video.py                   # ffprobe probe + ffmpeg read/write helpers
  effects/
    __init__.py              # PARAMETRIC registry of all 13 effects
    _frame.py                # shared helpers (block-mask generation, ClipReader)
    <effect>.py              # one file per effect
  recipes/
    <name>.json              # 15 curated configs
  build_effects_guide.py     # renders static HTML catalog
  README.md                  # you are here
```

---

## Future considerations

- **Noise grain presets** — film grain / digital sensor noise / tape grain
  variants would benefit from tuned presets or a `grain_type` switch.
- **More sort orderings** — pixel_sort could sort by a sum (e.g. R+G) or
  a perceptual distance from a reference color.
- **Generic delay-frames at the render level** — currently each effect
  has its own `delay_frames` knob. A global `--rhythmic-frames N` that
  gates the entire stack would simplify syncing multiple effects.
- **Beat-locked accumulation** — accumulate reset could sync to downbeats
  rather than wall-clock seconds for more rhythmic destruction cycles.
- **Better guide** — render-time-aware thumbnails, CLI snippet visibility
  per card, side-by-side before/after pairings.
- **Performance** — pixel_sort is Python-loop bound (~6-10 fps at 1080p).
  Numba or Cython inner loop could bring it to realtime.
