"""LayerForge - decompose a flat image into meaningful, reusable design layers.

    from layerforge import LayerForge, Config
    result = LayerForge().decompose(rgb_array)

Every capability degrades gracefully: with optional neural backends installed
the pipeline uses them, and without any downloaded weights it still runs
end-to-end on classical computer vision.
"""

from .config import Config, DEFAULT_CONFIG
from .pipeline import LayerForge, decompose_file
from .types import Decomposition, Layer

__version__ = "1.0.0"

__all__ = [
    "LayerForge",
    "Config",
    "DEFAULT_CONFIG",
    "Decomposition",
    "Layer",
    "decompose_file",
    "__version__",
]
