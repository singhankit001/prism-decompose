"""Core data structures shared across pipeline stages.

A `Layer` is the unit of output: a full-canvas alpha mask plus semantic
metadata. Pixel extraction and cropping happen at export time so that stages
can freely combine, subtract and re-order masks without losing information.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Canonical layer kinds. Z-order is assigned in this nominal order, back to
# front, then refined by area and occlusion analysis.
KIND_BACKGROUND = "background"
KIND_DECOR = "decor"
KIND_SUBJECT = "subject"
KIND_GRAPHIC = "graphic"
KIND_TEXT = "text"
KIND_FOREGROUND = "foreground"

KIND_ORDER = [
    KIND_BACKGROUND,
    KIND_DECOR,
    KIND_SUBJECT,
    KIND_GRAPHIC,
    KIND_TEXT,
    KIND_FOREGROUND,
]


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Tight (x0, y0, x1, y1) bounding box of non-zero mask pixels."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@dataclass
class Layer:
    """One decomposed element of the source image."""

    kind: str
    label: str
    mask: np.ndarray                       # uint8, full canvas, 0..255
    z: int = 0
    confidence: float = 1.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    text: Optional[str] = None             # recognised string for text layers
    color: Optional[Tuple[int, int, int]] = None   # dominant RGB
    meta: Dict[str, Any] = field(default_factory=dict)

    # populated at export time
    bbox: Optional[Tuple[int, int, int, int]] = None
    filename: Optional[str] = None

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def coverage(self) -> float:
        h, w = self.mask.shape[:2]
        return self.area / float(max(h * w, 1))

    def compute_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        self.bbox = _bbox_from_mask(self.mask)
        return self.bbox

    def to_manifest(self) -> Dict[str, Any]:
        """JSON-serialisable description consumed by the 3D viewer."""
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "z": self.z,
            "confidence": round(float(self.confidence), 4),
            "area_px": self.area,
            "coverage": round(self.coverage, 6),
            "bbox": list(self.bbox) if self.bbox else None,
            "text": self.text,
            "color": list(self.color) if self.color else None,
            "file": self.filename,
            "meta": {k: v for k, v in self.meta.items() if _jsonable(v)},
        }


def _jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, list, dict, type(None)))


@dataclass
class Decomposition:
    """Complete result of running the pipeline on one image."""

    width: int
    height: int
    layers: List[Layer] = field(default_factory=list)
    source_name: str = "image"
    timings: Dict[str, float] = field(default_factory=dict)
    backends: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    # Reconstructed full-canvas background RGB, produced by the inpaint stage.
    # Stored separately because the background layer's "mask" is the whole
    # canvas while its pixels come from the clean plate, not the source image.
    plate: Optional[np.ndarray] = None

    def add(self, layer: Layer) -> Layer:
        self.layers.append(layer)
        return layer

    def sorted_layers(self) -> List[Layer]:
        return sorted(self.layers, key=lambda l: l.z)

    def by_kind(self, kind: str) -> List[Layer]:
        return [l for l in self.layers if l.kind == kind]

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "source": self.source_name,
            "width": self.width,
            "height": self.height,
            "layer_count": len(self.layers),
            "backends": self.backends,
            "timings_ms": {k: round(v * 1000, 1) for k, v in self.timings.items()},
            "warnings": self.warnings,
            "layers": [l.to_manifest() for l in self.sorted_layers()],
        }
