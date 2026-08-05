"""Stage 4 - graphic and decorative element extraction.

Whatever remains in the foreground after text and subjects have been claimed is
design furniture: logos, icon rows, badges, ornamental frames, confetti, rule
lines, decorative shapes. These matter - they are exactly the reusable assets a
designer wants back - but no off-the-shelf model segments them.

The method is deliberately structural rather than semantic:

  1. Take the residual foreground (foreground minus text minus subjects).
  2. Split it into connected components, merging neighbours that are close
     enough to belong to one design element.
  3. Classify each component as flat vector artwork or photographic content
     using the texture metric, then label it with interpretable heuristics
     (aspect ratio, position, size, repetition).

Photographic residual components are promoted to secondary `subject` layers -
they are usually a second person or an inset photo that the salient-object
model did not rank first.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..config import Config
from ..imaging import (clean_mask, dominant_color, fill_holes, refine_alpha,
                       texture_score)
from ..types import KIND_DECOR, KIND_GRAPHIC, KIND_SUBJECT


class GraphicRegion:
    def __init__(self, mask: np.ndarray, kind: str, label: str, score: float):
        self.mask = mask
        self.kind = kind
        self.label = label
        self.score = score
        self.color = None
        self.meta: Dict[str, float] = {}


def _merge_components(residual: np.ndarray, cfg: Config) -> List[np.ndarray]:
    """Group residual pixels into design elements.

    Dilating before labelling merges the separate strokes of a logo mark or the
    glyph-and-rule pair of a lockup into one element, while the original pixels
    are preserved for the actual mask.
    """
    binary = (residual > 127).astype(np.uint8) * 255
    if np.count_nonzero(binary) == 0:
        return []

    k = cfg.graphic_merge_dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    grouped = cv2.dilate(binary, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (grouped > 0).astype(np.uint8), connectivity=8)

    h, w = binary.shape
    min_area = int(h * w * cfg.graphic_min_area_frac)

    canvas_area = float(h * w)
    split_limit = canvas_area * cfg.graphic_split_coverage

    regions: List[Tuple[int, np.ndarray]] = []
    for idx in range(1, count):
        envelope = (labels == idx)
        mask = np.where(envelope, binary, 0).astype(np.uint8)
        area = int(np.count_nonzero(mask))
        if area < min_area:
            continue

        # Proximity merging is what turns the separate strokes of a logo into
        # one element, but on dense collage artwork it can chain the entire
        # composition into a single blob. When a merged element grows past a
        # sane share of the canvas, discard the merge for that element and keep
        # its natural sub-components instead - one huge layer is never a useful
        # decomposition.
        if area > split_limit:
            regions.extend(_natural_subcomponents(mask, min_area))
            continue

        regions.append((area, mask))

    regions.sort(key=lambda item: -item[0])
    return [m for _, m in regions[: cfg.max_graphic_layers * 2]]


def _natural_subcomponents(mask: np.ndarray, min_area: int
                           ) -> List[Tuple[int, np.ndarray]]:
    """Connected components of a mask without any proximity merging."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    out: List[Tuple[int, np.ndarray]] = []
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        out.append((area, (labels == idx).astype(np.uint8) * 255))
    return out


def _describe(mask: np.ndarray, image_rgb: np.ndarray, texture: float
              ) -> Tuple[str, Dict[str, float]]:
    """Interpretable label for a graphic element.

    No classifier here on purpose: these heuristics are transparent, need no
    weights, and are easy for a reviewer to audit and extend. When CLIP or a
    similar model is available the same slot can be swapped for zero-shot
    labelling without changing the layer contract.
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return "element", {}

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    area = float(np.count_nonzero(mask))
    fill = area / float(max(bw * bh, 1))
    aspect = bw / float(max(bh, 1))
    coverage = area / float(h * w)

    cx = (x0 + x1) / 2.0 / w
    cy = (y0 + y1) / 2.0 / h

    meta = {
        "aspect": round(aspect, 3),
        "fill": round(fill, 3),
        "coverage": round(coverage, 5),
        "texture": round(texture, 3),
        "cx": round(cx, 3),
        "cy": round(cy, 3),
    }

    # Long thin runs are rules, dividers and underlines.
    if aspect > 8.0 and bh < h * 0.02:
        return "divider rule", meta
    if aspect < 0.125 and bw < w * 0.02:
        return "vertical rule", meta
    # A near-full-canvas thin shell is a frame or border treatment.
    if coverage < 0.10 and bw > w * 0.75 and bh > h * 0.75 and fill < 0.35:
        return "frame border", meta
    # Small, compact, near an edge or corner: icon or badge.
    if coverage < 0.012 and 0.35 < aspect < 2.8:
        if cy > 0.80:
            return "footer icon", meta
        if cy < 0.22:
            return "header mark", meta
        return "icon", meta
    # Wide band across the bottom is a footer/brand bar.
    if cy > 0.78 and aspect > 3.0:
        return "footer band", meta
    # Compact mark in the upper region: logo lockup.
    if cy < 0.30 and coverage < 0.06:
        return "logo mark", meta
    if coverage > 0.14:
        return "backdrop ornament", meta
    return "graphic element", meta


def extract_graphics(image_rgb: np.ndarray, residual: np.ndarray, cfg: Config
                     ) -> List[GraphicRegion]:
    """Classify residual foreground into graphic and secondary-subject regions."""
    if not cfg.enable_graphics:
        return []

    out: List[GraphicRegion] = []
    for mask in _merge_components(residual, cfg):
        mask = clean_mask(mask, open_k=3, close_k=5)
        if np.count_nonzero(mask) == 0:
            continue

        coverage = float(np.count_nonzero(mask)) / float(mask.size)
        texture = texture_score(image_rgb, mask)
        if texture >= cfg.photographic_texture_threshold:
            # Photographic residual: a second person, an inset photo, a product.
            solid = fill_holes(mask)
            alpha = refine_alpha(image_rgb, solid, cfg.alpha_guided_radius,
                                 cfg.alpha_guided_eps, cfg.alpha_feather)
            if coverage > cfg.decor_coverage_threshold:
                # Too large to be a discrete subject. On dense collage artwork
                # the residual is the composed backdrop itself, so calling it a
                # "subject" would be wrong; it is scene artwork that sits
                # behind the hero elements.
                region = GraphicRegion(alpha, KIND_DECOR, "backdrop artwork",
                                       min(0.45 + texture * 0.3, 0.85))
            else:
                region = GraphicRegion(alpha, KIND_SUBJECT, "photographic element",
                                       min(0.55 + texture * 0.4, 0.97))
        else:
            label, meta = _describe(mask, image_rgb, texture)
            alpha = refine_alpha(image_rgb, mask, max(4, cfg.alpha_guided_radius // 2),
                                 cfg.alpha_guided_eps, max(1, cfg.alpha_feather - 1))
            region = GraphicRegion(alpha, KIND_GRAPHIC, label,
                                   min(0.50 + (1.0 - texture) * 0.45, 0.95))
            region.meta = meta

        region.color = dominant_color(image_rgb, region.mask)
        out.append(region)

    # Keep the most substantial elements so output stays readable.
    out.sort(key=lambda r: -np.count_nonzero(r.mask))
    graphics = [r for r in out if r.kind == KIND_GRAPHIC][: cfg.max_graphic_layers]
    subjects = [r for r in out if r.kind == KIND_SUBJECT][:4]
    decor = [r for r in out if r.kind == KIND_DECOR][:3]
    return decor + subjects + graphics
