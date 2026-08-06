"""Stage 2 - text detection, line formation and block grouping.

Text is the layer type designers most want back as a separate asset, and the
one generic segmentation models handle worst: a headline is not a salient
object, and its glyphs are disconnected components that saliency models happily
merge into whatever they sit on.

Two detectors behind one interface:

  * `easyocr` (CRAFT) when installed - a learned character-region detector.
  * a classical detector otherwise, with no weights to download.

The classical path is a three-tier bottom-up analysis borrowed from document
layout research, because blind morphological dilation produces exactly the
fragmentation this stage exists to avoid:

  glyphs  -> candidate components (MSER + flat-colour regions, filtered by
             glyph geometry and stroke-width consistency)
  lines   -> components joined by vertical overlap, height similarity and
             horizontal proximity, via union-find
  blocks  -> lines joined by horizontal alignment and vertical proximity

Grouping to blocks matters: a headline set over four lines is *one* design
asset, not four, and that is how it appears in a real layered file.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..backends import easyocr_reader, has_tesseract
from ..config import Config
from ..imaging import dominant_color

log = logging.getLogger("prism.text")


class TextLine:
    """A detected block of text (one or more visual lines)."""

    def __init__(self, mask: np.ndarray, bbox: Tuple[int, int, int, int],
                 score: float, string: Optional[str] = None,
                 multiline: bool = False):
        self.mask = mask
        self.bbox = bbox
        self.score = score
        self.string = string
        self.color: Optional[Tuple[int, int, int]] = None
        # Carries the geometric single-vs-multi-line estimate through to OCR,
        # which is deferred until after merging/capping (see detect_text).
        self.multiline = multiline


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

class _Comp:
    """A candidate glyph: bounding box plus the pixels that produced it."""

    __slots__ = ("x0", "y0", "x1", "y1", "pixels")

    def __init__(self, x0: int, y0: int, x1: int, y1: int, pixels: np.ndarray):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.pixels = pixels  # (ys, xs) index arrays

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


def _stroke_width_cv(patch: np.ndarray) -> float:
    """Coefficient of variation of stroke width within a component patch.

    Glyphs are built from strokes of near-constant thickness, so the distance
    transform inside a glyph has low relative spread. Photographic blobs and
    decorative shapes do not share this property.
    """
    dist = cv2.distanceTransform((patch > 0).astype(np.uint8), cv2.DIST_L2, 3)
    values = dist[dist > 0.5]
    if values.size < 10:
        return 99.0
    mean = float(values.mean())
    if mean <= 1e-6:
        return 99.0
    return float(values.std() / mean)


# ---------------------------------------------------------------------------
# tier 1: glyph candidates
# ---------------------------------------------------------------------------

def _accept_component(patch: np.ndarray, w: int, h: int, min_h: int, max_h: int,
                      cfg: Config) -> bool:
    if h < min_h or h > max_h or w == 0 or h == 0:
        return False
    if w * h < cfg.text_min_area_px:
        return False
    if max(w / float(h), h / float(w)) > cfg.text_max_aspect:
        return False
    fill = float(patch.astype(bool).mean())
    # Glyphs occupy a middling share of their box: not a speck, not a solid bar.
    if fill < 0.10 or fill > 0.92:
        return False
    if _stroke_width_cv(patch) > cfg.text_max_swt_cv:
        return False
    return True


def _components_from_binary(binary: np.ndarray, min_h: int, max_h: int,
                            cfg: Config, budget: int) -> List[_Comp]:
    """Extract glyph-like connected components from a binary image."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8)
    out: List[_Comp] = []
    for idx in range(1, count):
        if len(out) >= budget:
            break
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < cfg.text_min_area_px:
            continue
        sub = (labels[y:y + h, x:x + w] == idx)
        if not _accept_component(sub.astype(np.uint8), w, h, min_h, max_h, cfg):
            continue
        ys, xs = np.nonzero(sub)
        out.append(_Comp(x, y, x + w, y + h, (ys + y, xs + x)))
    return out


def _glyph_candidates(image_rgb: np.ndarray, cfg: Config) -> List[_Comp]:
    """Propose glyph components from several complementary binarisations.

    Poster type is flat-coloured and high contrast, but not always separable in
    luminance alone - an orange word set inside a dark-green headline has very
    similar lightness. Running the same component analysis over adaptive
    luminance thresholds *and* over quantised colour planes catches both cases
    without needing a learned detector.
    """
    h, w = image_rgb.shape[:2]
    min_h = max(5, int(h * cfg.text_min_char_h_frac))
    max_h = int(h * cfg.text_max_char_h_frac)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)

    binaries: List[np.ndarray] = []

    # (a) adaptive threshold, both polarities - robust to uneven backdrops.
    block = max(15, (min(h, w) // 24) | 1)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 9)
    binaries.append(adaptive)
    binaries.append(cv2.bitwise_not(adaptive))

    # (b) flat-colour planes: quantise and take each dominant colour as a mask.
    small = cv2.resize(image_rgb, (max(1, w // 2), max(1, h // 2)),
                       interpolation=cv2.INTER_AREA)
    samples = small.reshape(-1, 3).astype(np.float32)
    if samples.shape[0] > 40000:
        idx = np.random.default_rng(5).choice(samples.shape[0], 40000, replace=False)
        samples = samples[idx]
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        cv2.setRNGSeed(cfg.random_seed)
        _c, _labels, centers = cv2.kmeans(samples, 6, None, criteria, 2,
                                          cv2.KMEANS_PP_CENTERS)
        flat = image_rgb.reshape(-1, 3).astype(np.float32)
        for center in centers:
            dist = np.linalg.norm(flat - center[None, :], axis=1).reshape(h, w)
            binaries.append((dist < 40).astype(np.uint8) * 255)
    except Exception:
        pass

    budget_per_pass = 900
    comps: List[_Comp] = []
    for binary in binaries:
        comps.extend(_components_from_binary(binary, min_h, max_h, cfg,
                                             budget_per_pass))
        if len(comps) > 4000:
            break

    return _deduplicate(comps)


def _deduplicate(comps: Sequence[_Comp]) -> List[_Comp]:
    """Drop near-duplicate components produced by overlapping binarisations."""
    kept: List[_Comp] = []
    seen: Dict[Tuple[int, int, int, int], bool] = {}
    for c in sorted(comps, key=lambda c: -(c.w * c.h)):
        key = (c.x0 // 4, c.y0 // 4, c.w // 4, c.h // 4)
        if key in seen:
            continue
        seen[key] = True
        kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# tier 2 + 3: lines and blocks
# ---------------------------------------------------------------------------

def _same_line(a: _Comp, b: _Comp, gap_ratio: float) -> bool:
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if overlap <= 0:
        return False
    if overlap < 0.42 * min(a.h, b.h):
        return False
    if max(a.h, b.h) / float(max(min(a.h, b.h), 1)) > 2.6:
        return False
    gap = max(a.x0, b.x0) - min(a.x1, b.x1)
    if gap < 0:  # overlapping horizontally
        return True
    return gap <= gap_ratio * max(a.h, b.h)


def _group_lines(comps: List[_Comp], cfg: Config) -> List[List[_Comp]]:
    if not comps:
        return []
    uf = _UnionFind(len(comps))
    # Sorting by x makes the neighbour scan local, keeping this near-linear in
    # practice despite the nominal quadratic bound.
    order = sorted(range(len(comps)), key=lambda i: comps[i].x0)
    for pos, i in enumerate(order):
        ci = comps[i]
        for j in order[pos + 1: pos + 40]:
            cj = comps[j]
            if cj.x0 - ci.x1 > 4 * max(ci.h, cj.h):
                break
            if _same_line(ci, cj, cfg.text_line_gap_ratio * 1.6):
                uf.union(i, j)
    return [[comps[i] for i in members] for members in uf.groups().values()]


def _line_bbox(line: Sequence[_Comp]) -> Tuple[int, int, int, int]:
    return (min(c.x0 for c in line), min(c.y0 for c in line),
            max(c.x1 for c in line), max(c.y1 for c in line))


def _group_blocks(lines: List[List[_Comp]]) -> List[List[_Comp]]:
    """Merge stacked lines that share horizontal alignment into one block."""
    if len(lines) <= 1:
        return lines
    boxes = [_line_bbox(l) for l in lines]
    uf = _UnionFind(len(lines))
    for i in range(len(lines)):
        xi0, yi0, xi1, yi1 = boxes[i]
        hi = yi1 - yi0
        for j in range(i + 1, len(lines)):
            xj0, yj0, xj1, yj1 = boxes[j]
            hj = yj1 - yj0
            # horizontal overlap relative to the narrower line
            overlap = min(xi1, xj1) - max(xi0, xj0)
            if overlap <= 0.32 * min(xi1 - xi0, xj1 - xj0):
                continue
            # comparable type size
            if max(hi, hj) / float(max(min(hi, hj), 1)) > 2.2:
                continue
            gap = max(yi0, yj0) - min(yi1, yj1)
            if gap < 0 or gap <= 0.85 * max(hi, hj):
                uf.union(i, j)
    merged: List[List[_Comp]] = []
    for members in uf.groups().values():
        block: List[_Comp] = []
        for idx in members:
            block.extend(lines[idx])
        merged.append(block)
    return merged


def _ink_colour_spread(image_rgb: np.ndarray, block: Sequence[_Comp]) -> float:
    """Median LAB distance of component ink colours from the block's median.

    The strongest non-OCR signal for "is this a word". Letters in a word share
    an ink colour — that is what makes them read as a unit. Decorative clutter
    that happens to group by size and spacing (confetti, bunting, scattered
    ornaments) is deliberately multi-coloured, so it separates cleanly here
    even when its geometry mimics type.

    Deliberately the *median* distance, not the mean. Candidates come from
    several binarisations including an inverted pass, so a block legitimately
    built from dark glyphs can also carry a few background-coloured
    components; those outliers drag a mean far enough to reject real type
    (measured 37.9 on a clean black-on-cream headline, against a median of
    3.0). Measured separation on the median is wide and unambiguous: real
    text lands at 1-7, decorative clutter at 17-80.
    """
    if len(block) < 3:
        return 0.0
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    means = []
    for comp in block:
        px = lab[comp.pixels]
        if px.size:
            means.append(px.reshape(-1, 3).mean(axis=0))
    if len(means) < 3:
        return 0.0
    arr = np.asarray(means, dtype=np.float32)
    median = np.median(arr, axis=0)
    return float(np.median(np.linalg.norm(arr - median[None, :], axis=1)))


def _baseline_deviation(block: Sequence[_Comp]) -> float:
    """Scatter of component bottom edges, normalised by glyph height.

    Type sits on a baseline; that shared bottom edge is the most reliable
    geometric property of a line of text. Scattered decoration has no such
    alignment. Normalising by height keeps this scale-free, and the taller
    half is used so descenders do not masquerade as misalignment.
    """
    if len(block) < 3:
        return 0.0
    heights = np.array([c.h for c in block], dtype=np.float32)
    tall = [c for c, hh in zip(block, heights) if hh >= np.median(heights)]
    if len(tall) < 3:
        return 0.0
    bottoms = np.array([c.y1 for c in tall], dtype=np.float32)
    ref = float(np.median([c.h for c in tall])) or 1.0
    return float(bottoms.std() / ref)


def _validate(block: Sequence[_Comp], shape: Tuple[int, int], cfg: Config,
              image_rgb: Optional[np.ndarray] = None) -> bool:
    """Reject groups that are not plausibly type.

    OCR cannot be used as the gate here: Tesseract is optional, stylised
    display faces defeat it even when installed, and on busy artwork it
    cheerfully returns noise ("VW WVYV", "vvv") for decoration — so a
    confidence score is not evidence of text either. Validation therefore
    stays structural, and behaves identically with or without the optional
    backends.
    """
    h, w = shape
    x0, y0, x1, y1 = _line_bbox(block)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return False

    ink = sum(c.pixels[0].size for c in block)
    if ink < cfg.text_min_area_px * 3:
        return False
    # A single isolated component is only type if it is a large display glyph.
    if len(block) < 2 and bh < h * 0.05:
        return False
    # Type is not a solid slab.
    if ink / float(bw * bh) > 0.80:
        return False
    # Reject groups that sprawl across most of the canvas in both axes.
    if bw > w * 0.97 and bh > h * 0.60:
        return False

    # Glyph-height consistency, measured only over the taller half of the
    # components.
    #
    # Raw height variance is the wrong statistic: a line like
    # "Radiate your essence." legitimately mixes caps, x-height letters,
    # descenders and a full stop, so its heights span a 4x range and any
    # sane threshold rejects real typography. Taking the components at or
    # above the median height isolates caps and ascenders, which genuinely
    # do share a common height — while foliage and texture fragments stay
    # irregular at every scale.
    if len(block) >= 3:
        heights = np.array([c.h for c in block], dtype=np.float32)
        tall = heights[heights >= np.median(heights)]
        mean_h = float(tall.mean())
        if mean_h > 1e-6 and float(tall.std()) / mean_h > cfg.text_max_height_cv:
            return False

    # A hard floor on size, applied regardless of component count. Previously
    # this only gated blocks of fewer than three components, so a cluster of
    # three specks passed while a single larger mark did not — which is how
    # 0.02%-coverage fragments ("B", "»") reached the output as their own
    # layers. A layer that small is never a useful asset to hand back.
    coverage = ink / float(h * w)
    if coverage < cfg.text_min_coverage:
        return False

    # Baseline alignment. Cheap, so it runs before the colour test.
    if _baseline_deviation(block) > cfg.text_max_baseline_dev:
        return False

    # Ink colour coherence — the decisive test on decorative artwork.
    if image_rgb is not None:
        if _ink_colour_spread(image_rgb, block) > cfg.text_max_colour_spread:
            return False

    return True


# ---------------------------------------------------------------------------
# recognition
# ---------------------------------------------------------------------------

def _recognise(image_rgb: np.ndarray, bbox: Tuple[int, int, int, int],
               multiline: bool) -> Tuple[Optional[str], float]:
    """OCR a cropped region with Tesseract; returns (text, mean confidence)."""
    if not has_tesseract():
        return None, 0.0
    try:
        import pytesseract  # type: ignore

        x0, y0, x1, y1 = bbox
        pad = 5
        crop = image_rgb[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        if crop.size == 0 or min(crop.shape[:2]) < 6:
            return None, 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        scale = max(1.0, 40.0 / max(min(crop.shape[0], crop.shape[1]), 1))
        scale = min(scale, 4.0)
        if scale > 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        psm = "--psm 6" if multiline else "--psm 7"
        best_text, best_conf = None, 0.0
        for variant in (gray, cv2.bitwise_not(gray)):
            try:
                data = pytesseract.image_to_data(
                    variant, config=psm, output_type=pytesseract.Output.DICT)
            except Exception:
                continue
            words, confs = [], []
            for token, conf in zip(data["text"], data["conf"]):
                try:
                    c = float(conf)
                except (TypeError, ValueError):
                    continue
                if token.strip() and c > 0:
                    words.append(token.strip())
                    confs.append(c)
            if confs:
                mean_conf = float(np.mean(confs))
                if mean_conf > best_conf:
                    best_conf, best_text = mean_conf, " ".join(words)
        return best_text, best_conf
    except Exception as exc:  # pragma: no cover
        log.debug("OCR failed: %s", exc)
        return None, 0.0


# ---------------------------------------------------------------------------
# detector entry points
# ---------------------------------------------------------------------------

def _tighten_to_glyphs(image_rgb: np.ndarray, quad_mask: np.ndarray) -> np.ndarray:
    """Inside a detected text quad, keep only the glyph pixels.

    A quad includes the ground between letters. Otsu within the quad separates
    ink from ground, giving a true glyph alpha so the exported text layer
    composites cleanly over the reconstructed backdrop.
    """
    ys, xs = np.nonzero(quad_mask)
    if ys.size == 0:
        return quad_mask
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = image_rgb[y0:y1, x0:x1]
    sub = quad_mask[y0:y1, x0:x1] > 0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    vals = gray[sub]
    if vals.size < 16:
        return quad_mask
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (gray <= thr) & sub
    light = (gray > thr) & sub
    glyphs = dark if dark.sum() <= light.sum() else light
    out = np.zeros_like(quad_mask)
    out[y0:y1, x0:x1] = glyphs.astype(np.uint8) * 255
    if np.count_nonzero(out) < 8:
        return quad_mask
    return out


def _detect_easyocr(image_rgb: np.ndarray, cfg: Config) -> List[TextLine]:
    reader = easyocr_reader()
    if reader is None:
        return []
    h, w = image_rgb.shape[:2]
    try:
        results = reader.readtext(image_rgb, detail=1, paragraph=False)
    except Exception as exc:  # pragma: no cover
        log.info("easyocr inference failed (%s)", exc)
        return []

    lines: List[TextLine] = []
    for box, string, conf in results:
        pts = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        mask = _tighten_to_glyphs(image_rgb, mask)
        if np.count_nonzero(mask) < cfg.text_min_area_px:
            continue
        x, y, bw, bh = cv2.boundingRect(pts)
        lines.append(TextLine(mask, (x, y, x + bw, y + bh), float(conf), string))
    return _merge_easyocr_blocks(lines, image_rgb.shape[:2])


def _merge_easyocr_blocks(lines: List[TextLine], shape: Tuple[int, int]) -> List[TextLine]:
    """Apply the same block grouping to CRAFT output for a consistent contract."""
    if len(lines) <= 1:
        return lines
    uf = _UnionFind(len(lines))
    for i in range(len(lines)):
        xi0, yi0, xi1, yi1 = lines[i].bbox
        hi = yi1 - yi0
        for j in range(i + 1, len(lines)):
            xj0, yj0, xj1, yj1 = lines[j].bbox
            hj = yj1 - yj0
            overlap = min(xi1, xj1) - max(xi0, xj0)
            if overlap <= 0.32 * min(xi1 - xi0, xj1 - xj0):
                continue
            if max(hi, hj) / float(max(min(hi, hj), 1)) > 2.2:
                continue
            gap = max(yi0, yj0) - min(yi1, yj1)
            if gap < 0 or gap <= 0.85 * max(hi, hj):
                uf.union(i, j)

    merged: List[TextLine] = []
    for members in uf.groups().values():
        group = [lines[i] for i in members]
        mask = np.zeros(shape, np.uint8)
        for item in group:
            mask = cv2.bitwise_or(mask, item.mask)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        group.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
        text = " ".join(t.string for t in group if t.string).strip() or None
        score = float(np.mean([t.score for t in group]))
        merged.append(TextLine(mask, bbox, score, text))
    return merged


def _detect_classical(image_rgb: np.ndarray, cfg: Config) -> List[TextLine]:
    """Geometric detection only - no OCR here.

    Several binarisations are run per image on purpose (see
    `_glyph_candidates`), which means the same headline is routinely proposed
    two or three times before `_merge_overlapping` collapses the duplicates.
    Recognising text means spawning a Tesseract subprocess per attempt, which
    is the single most expensive step in the pipeline - so OCR is deferred to
    `detect_text`, which runs it once per block that survives merging *and*
    the final layer cap, instead of once per raw candidate.
    """
    h, w = image_rgb.shape[:2]
    comps = _glyph_candidates(image_rgb, cfg)
    if not comps:
        return []

    lines = _group_lines(comps, cfg)
    blocks = _group_blocks(lines)

    results: List[TextLine] = []
    for block in blocks:
        # Blocks that fail validation are dropped rather than relabelled: the
        # text stage runs first, so anything it does not claim stays in the
        # residual and reaches the graphics stage, which classifies it on its
        # own terms (ornament, badge, decor). Dropping here is what lets
        # confetti come back as decoration instead of as a fake word.
        if not _validate(block, (h, w), cfg, image_rgb):
            continue
        mask = np.zeros((h, w), np.uint8)
        for comp in block:
            mask[comp.pixels] = 255
        bbox = _line_bbox(block)
        multiline = (bbox[3] - bbox[1]) > 1.8 * np.median([c.h for c in block])
        # Geometric evidence alone earns a moderate score; OCR agreement (once
        # it runs, after merge+cap) raises it - see the recognition pass in
        # detect_text.
        results.append(TextLine(mask, bbox, score=0.45, multiline=multiline))
    return results


def _recognise_survivors(image_rgb: np.ndarray, lines: List[TextLine],
                         cfg: Config) -> List[TextLine]:
    """Run OCR on the final block set only - see `_detect_classical`."""
    kept: List[TextLine] = []
    for line in lines:
        string, conf = _recognise(image_rgb, line.bbox, line.multiline)
        if conf < cfg.text_min_ocr_conf and not cfg.text_keep_unrecognised:
            continue
        line.string = string
        line.score = 0.45 + min(conf, 100.0) / 200.0
        kept.append(line)
    return kept


def _merge_overlapping(lines: List[TextLine], shape: Tuple[int, int]) -> List[TextLine]:
    """Collapse text blocks that describe the same region.

    Several binarisations are run per image on purpose - it is how coloured
    type on coloured grounds gets found at all - but that means one headline
    can be proposed more than once. Blocks are merged when they overlap
    substantially or when one is largely contained in another, so each piece of
    type leaves the stage exactly once.
    """
    if len(lines) <= 1:
        return lines

    order = sorted(range(len(lines)),
                   key=lambda i: -((lines[i].bbox[2] - lines[i].bbox[0]) *
                                   (lines[i].bbox[3] - lines[i].bbox[1])))
    uf = _UnionFind(len(lines))
    for pos, i in enumerate(order):
        ax0, ay0, ax1, ay1 = lines[i].bbox
        area_a = max((ax1 - ax0) * (ay1 - ay0), 1)
        for j in order[pos + 1:]:
            bx0, by0, bx1, by1 = lines[j].bbox
            area_b = max((bx1 - bx0) * (by1 - by0), 1)
            iw = min(ax1, bx1) - max(ax0, bx0)
            ih = min(ay1, by1) - max(ay0, by0)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            iou = inter / float(area_a + area_b - inter)
            containment = inter / float(min(area_a, area_b))
            if iou > 0.28 or containment > 0.68:
                uf.union(i, j)

    merged: List[TextLine] = []
    for members in uf.groups().values():
        group = [lines[i] for i in members]
        if len(group) == 1:
            merged.append(group[0])
            continue
        mask = np.zeros(shape, np.uint8)
        for item in group:
            mask = cv2.bitwise_or(mask, item.mask)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        # Keep the most confident transcription rather than concatenating
        # several partial reads of the same words. When OCR hasn't run yet
        # (classical path - see detect_text), every candidate's string is
        # still None and this just picks a representative geometry.
        best = max(group, key=lambda t: (t.score, len(t.string or "")))
        score = float(max(t.score for t in group))
        multiline = any(t.multiline for t in group)
        merged.append(TextLine(mask, bbox, score, best.string, multiline))
    return merged


def detect_text(image_rgb: np.ndarray, cfg: Config) -> List[TextLine]:
    """Detect text blocks using the best available backend."""
    if not cfg.enable_text:
        return []

    lines = _detect_easyocr(image_rgb, cfg)
    used_easyocr = bool(lines)
    if not lines:
        lines = _detect_classical(image_rgb, cfg)

    lines = _merge_overlapping(lines, image_rgb.shape[:2])

    # Largest first - headlines lead the layer list. Cap the count *before*
    # OCR runs for the classical path, since each attempt is a Tesseract
    # subprocess spawn and by far the most expensive step per block; there is
    # no reason to recognise text in a block that the cap would discard.
    lines.sort(key=lambda l: -int(np.count_nonzero(l.mask)))
    lines = lines[: cfg.max_text_layers]

    if not used_easyocr and cfg.text_recognise:
        lines = _recognise_survivors(image_rgb, lines, cfg)

    for line in lines:
        line.color = dominant_color(image_rgb, line.mask)

    return lines
