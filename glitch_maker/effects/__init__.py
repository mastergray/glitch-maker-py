"""Effect registry — each entry is a zero-arg factory for an effect instance.

SectionSwap takes a secondary video path so it isn't zero-arg; render.py
handles that specially when --swap-with is supplied.
"""
from .pixel_sort import PixelSort
from .databend import Databend
from .chromatic import Chromatic
from .section_swap import SectionSwap


REGISTRY = {
    'pixel_sort': PixelSort,
    'databend': Databend,
    'chromatic': Chromatic,
}

# Effects requiring constructor args live here for --help discoverability.
PARAMETRIC = {
    'section_swap': SectionSwap,
}
