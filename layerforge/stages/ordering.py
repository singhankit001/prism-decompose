"""Stage 6 - stacking order.

A pile of cut-outs is not a decomposition; the value is in knowing what sits in
front of what. Z-order is resolved from three signals, in decreasing authority:

  1. Layer kind. Design files stack predictably: backdrop, then ornament, then
     imagery, then graphic furniture, then type on top.
  2. Occlusion evidence. Where two layers' dilated boundaries meet, the one
     whose edge is *interrupted* is behind. Sampling colour continuity across
     the shared boundary decides which mask owns the contested pixels.
  3. Area. Within a tier, larger elements sit further back - a reliable
     regularity in layout design.

The resulting integer `z` is what positions each plane along the depth axis in
the 3D viewer, so the exploded stack reflects real structure rather than an
arbitrary spacing.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from ..types import (KIND_BACKGROUND, KIND_DECOR, KIND_FOREGROUND, KIND_GRAPHIC,
                     KIND_ORDER, KIND_SUBJECT, KIND_TEXT, Layer)

# Base depth band per kind, leaving room to interleave within a band.
_TIER = {
    KIND_BACKGROUND: 0,
    KIND_DECOR: 100,
    KIND_SUBJECT: 200,
    KIND_GRAPHIC: 300,
    KIND_TEXT: 400,
    KIND_FOREGROUND: 500,
}


def _overlap(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero((a > 127) & (b > 127)))


def _occlusion_vote(image_rgb: np.ndarray, a: Layer, b: Layer) -> int:
    """Return +1 if `a` is likely in front of `b`, -1 if behind, 0 if unknown.

    Only meaningful when the two masks actually contest pixels. The layer whose
    interior colour better matches the contested band owns those pixels and is
    therefore in front.
    """
    contested = (a.mask > 127) & (b.mask > 127)
    n = int(contested.sum())
    if n < 40:
        return 0

    def interior_mean(layer: Layer) -> np.ndarray | None:
        solo = (layer.mask > 127) & ~contested
        if solo.sum() < 40:
            return None
        pixels = image_rgb[solo]
        if pixels.shape[0] > 8000:
            idx = np.random.default_rng(2).choice(pixels.shape[0], 8000, replace=False)
            pixels = pixels[idx]
        return pixels.mean(axis=0)

    mean_a = interior_mean(a)
    mean_b = interior_mean(b)
    if mean_a is None or mean_b is None:
        return 0

    band = image_rgb[contested]
    if band.shape[0] > 8000:
        idx = np.random.default_rng(3).choice(band.shape[0], 8000, replace=False)
        band = band[idx]
    band_mean = band.mean(axis=0)

    d_a = float(np.linalg.norm(band_mean - mean_a))
    d_b = float(np.linalg.norm(band_mean - mean_b))
    if abs(d_a - d_b) < 6.0:
        return 0
    return 1 if d_a < d_b else -1


def assign_z(image_rgb: np.ndarray, layers: List[Layer]) -> List[Layer]:
    """Assign each layer an integer depth, back (small) to front (large)."""
    if not layers:
        return layers

    total = float(image_rgb.shape[0] * image_rgb.shape[1])

    # Tier + area gives the baseline ordering.
    for layer in layers:
        tier = _TIER.get(layer.kind, 300)
        # Larger area -> further back within the tier (offset 0..80).
        area_rank = 1.0 - min(layer.area / total, 1.0)
        layer.z = tier + int(round(area_rank * 80))
        layer.meta.setdefault("z_tier", tier)

    # Occlusion refinement within overlapping same-tier pairs.
    movable = [l for l in layers if l.kind not in (KIND_BACKGROUND,)]
    for i in range(len(movable)):
        for j in range(i + 1, len(movable)):
            a, b = movable[i], movable[j]
            if _TIER.get(a.kind, 300) != _TIER.get(b.kind, 300):
                continue
            if _overlap(a.mask, b.mask) < 40:
                continue
            vote = _occlusion_vote(image_rgb, a, b)
            if vote > 0 and a.z <= b.z:
                a.z, b.z = b.z + 1, b.z
            elif vote < 0 and b.z <= a.z:
                b.z, a.z = a.z + 1, a.z

    # Normalise to a dense 0..n-1 sequence for a clean viewer stack.
    for rank, layer in enumerate(sorted(layers, key=lambda l: (l.z, -l.area))):
        layer.meta["z_raw"] = layer.z
        layer.z = rank

    return layers


def strip_occluded(layers: List[Layer]) -> List[Layer]:
    """Remove pixels a front layer already owns from the layers behind it.

    Without this the same pixels appear in several layers and the exploded 3D
    stack shows ghost duplicates. Front-to-back subtraction guarantees a clean
    partition of the canvas, which is also what makes the exported PSD
    recomposite to something close to the original.
    """
    ordered = sorted(layers, key=lambda l: -l.z)  # front first
    claimed = None
    for layer in ordered:
        if layer.kind == KIND_BACKGROUND:
            continue
        if claimed is not None:
            keep = cv2.bitwise_not(claimed)
            layer.mask = cv2.bitwise_and(layer.mask, keep)
        solid = (layer.mask > 127).astype(np.uint8) * 255
        claimed = solid if claimed is None else cv2.bitwise_or(claimed, solid)
    return [l for l in layers if l.kind == KIND_BACKGROUND or l.area > 0]
