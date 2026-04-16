"""Effect registry — each entry is a zero-arg factory for an effect instance.

PARAMETRIC lists effects that take constructor args; render.py handles their
construction specially (CLI flags → kwargs).
"""
from .pixel_sort import PixelSort
from .databend import Databend
from .chromatic import Chromatic
from .clip_cut import ClipCut
from .tear import Tear
from .pixel_scramble import PixelScramble
from .jpeg_corrupt import JpegCorrupt
from .channel_swap import ChannelSwap
from .bit_crush import BitCrush
from .invert_blocks import InvertBlocks
from .slitscan import Slitscan
from .warp import Warp


REGISTRY = {}

PARAMETRIC = {
    'clip_cut': ClipCut,
    'tear': Tear,
    'pixel_scramble': PixelScramble,
    'jpeg_corrupt': JpegCorrupt,
    'pixel_sort': PixelSort,
    'chromatic': Chromatic,
    'databend': Databend,
    'channel_swap': ChannelSwap,
    'bit_crush': BitCrush,
    'invert_blocks': InvertBlocks,
    'slitscan': Slitscan,
    'warp': Warp,
}
