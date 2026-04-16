"""Video I/O — read frames from a source video, write them to a destination MP4.

Uses ffprobe to read metadata and ffmpeg pipes (rgb24 rawvideo) for both
directions. The source audio track is muxed into the output directly so we
never re-encode it.
"""
from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    duration_sec: float
    n_frames: int


def probe(path: str) -> VideoMeta:
    out = subprocess.check_output(
        ['ffprobe', '-v', 'error',
         '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
         '-show_entries', 'format=duration',
         '-of', 'json', path],
        text=True,
    )
    data = json.loads(out)
    stream = data['streams'][0]
    width = int(stream['width'])
    height = int(stream['height'])
    num, den = stream['r_frame_rate'].split('/')
    fps = float(num) / float(den)
    # duration can live on either stream or format; prefer format (container).
    fmt_dur = data.get('format', {}).get('duration')
    duration = float(fmt_dur if fmt_dur is not None else stream.get('duration', 0.0))
    # nb_frames is unreliable for some containers; compute from duration as fallback.
    try:
        n_frames = int(stream.get('nb_frames', 0)) or int(round(duration * fps))
    except (TypeError, ValueError):
        n_frames = int(round(duration * fps))
    return VideoMeta(width, height, fps, duration, n_frames)


def open_reader(path: str, meta: VideoMeta):
    """Return (proc, frame_nbytes). Read meta.width*meta.height*3 bytes per frame."""
    proc = subprocess.Popen(
        ['ffmpeg', '-v', 'error',
         '-i', path,
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE,
    )
    return proc, meta.width * meta.height * 3


def open_writer(audio_path: str, out_path: str, meta: VideoMeta, crf: int = 18):
    """Pipe rgb24 frames into ffmpeg which muxes them with the source audio."""
    proc = subprocess.Popen(
        ['ffmpeg', '-y', '-v', 'error',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{meta.width}x{meta.height}',
         '-r', str(meta.fps),
         '-i', '-',
         '-i', audio_path,
         '-map', '0:v', '-map', '1:a?',
         '-c:v', 'libx264', '-preset', 'medium',
         '-pix_fmt', 'yuv420p', '-crf', str(crf),
         '-c:a', 'aac', '-b:a', '192k',
         '-shortest',
         out_path],
        stdin=subprocess.PIPE,
    )
    return proc
