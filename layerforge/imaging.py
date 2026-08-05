"""Shared image-processing primitives used by multiple pipeline stages.

Kept dependency-light (OpenCV + NumPy only) so the whole pipeline runs without
any downloaded model weights.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# resizing helpers
# ---------------------------------------------------------------------------

def fit_within(image: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """Downscale so the longest side is <= max_dim. Returns (image, scale)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    resized = cv2.resize(
        image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resample a mask to (width, height) preserving soft edges."""
    w, h = size
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# edge-aware mask refinement
# ---------------------------------------------------------------------------

def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Fast guided filter (He et al.) for edge-aware alpha refinement.

    Snaps soft mask boundaries onto real image edges, which is what makes
    cut-outs look hand-made rather than blobby. Implemented directly so we do
    not depend on opencv-contrib's ximgproc being present.
    """
    guide_f = guide.astype(np.float32) / 255.0
    src_f = src.astype(np.float32) / 255.0
    if guide_f.ndim == 3:
        guide_f = cv2.cvtColor((guide_f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    d = 2 * radius + 1
    mean_i = cv2.boxFilter(guide_f, cv2.CV_32F, (d, d))
    mean_p = cv2.boxFilter(src_f, cv2.CV_32F, (d, d))
    corr_i = cv2.boxFilter(guide_f * guide_f, cv2.CV_32F, (d, d))
    corr_ip = cv2.boxFilter(guide_f * src_f, cv2.CV_32F, (d, d))

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (d, d))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (d, d))

    out = mean_a * guide_f + mean_b
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def refine_alpha(image_rgb: np.ndarray, mask: np.ndarray, radius: int = 8,
                 eps: float = 1e-3, feather: int = 2) -> np.ndarray:
    """Clean up a binary/soft mask into a presentable alpha channel."""
    if mask.max() == 0:
        return mask
    m = mask
    if feather > 0:
        k = 2 * feather + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    refined = guided_filter(image_rgb, m, radius, eps)
    # Re-expand contrast: the guided filter compresses the 0/255 endpoints.
    refined = np.clip((refined.astype(np.float32) - 24) * (255.0 / 200.0), 0, 255).astype(np.uint8)
    return refined


def clean_mask(mask: np.ndarray, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    """Morphological denoise: drop speckle, close pinholes."""
    binary = (mask > 127).astype(np.uint8) * 255
    if open_k > 0:
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    if close_k > 0:
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    return binary


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Flood-fill enclosed holes so subjects come out solid."""
    binary = (mask > 127).astype(np.uint8) * 255
    h, w = binary.shape
    flood = binary.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    return binary | cv2.bitwise_not(flood)


def largest_components(mask: np.ndarray, min_area: int, max_count: int = 32
                       ) -> List[np.ndarray]:
    """Split a mask into its connected components, largest first."""
    binary = (mask > 127).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    items = []
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        items.append((area, idx))
    items.sort(reverse=True)
    out = []
    for _, idx in items[:max_count]:
        out.append(((labels == idx).astype(np.uint8) * 255))
    return out


# ---------------------------------------------------------------------------
# region statistics: photographic vs flat vector artwork
# ---------------------------------------------------------------------------

def texture_score(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    """Score 0..1 estimating how *photographic* a masked region is.

    Photographs contain continuous tonal gradients, many distinct colours and
    high-frequency detail. Vector artwork (logos, icons, flat ornaments) is
    built from a small palette of flat fills with hard edges. Three cheap,
    complementary signals are combined:

      1. colour diversity  - unique quantised colours per unit area
      2. gradient activity - mean interior gradient magnitude (flat fills are
         near zero *inside* a shape even when edges are sharp)
      3. local variance    - texture energy from a box-filtered variance map

    This is the discriminator that keeps a logo out of the "person" layer and
    keeps skin texture out of the "graphic" layer.
    """
    ys, xs = np.nonzero(mask > 127)
    if ys.size < 64:
        return 0.0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    patch = image_rgb[y0:y1, x0:x1]
    patch_mask = (mask[y0:y1, x0:x1] > 127)
    if patch.size == 0 or patch_mask.sum() < 64:
        return 0.0

    pixels = patch[patch_mask]

    # 1) colour diversity on a coarse 5-bit-per-channel grid
    quant = (pixels // 32).astype(np.int32)
    codes = quant[:, 0] * 64 + quant[:, 1] * 8 + quant[:, 2]
    unique = np.unique(codes).size
    diversity = min(unique / 40.0, 1.0)

    # 2) interior gradient activity (erode so shape outlines do not count)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    interior = cv2.erode(patch_mask.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    if interior.sum() > 32:
        activity = float(grad[interior > 0].mean()) / 42.0
    else:
        activity = float(grad[patch_mask].mean()) / 42.0
    activity = min(activity, 1.0)

    # 3) local variance energy
    g32 = gray.astype(np.float32)
    mean = cv2.boxFilter(g32, cv2.CV_32F, (7, 7))
    sq = cv2.boxFilter(g32 * g32, cv2.CV_32F, (7, 7))
    var = np.clip(sq - mean * mean, 0, None)
    energy = min(float(np.sqrt(var[patch_mask]).mean()) / 22.0, 1.0)

    return float(0.40 * diversity + 0.35 * activity + 0.25 * energy)


def dominant_color(image_rgb: np.ndarray, mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Most representative colour of a masked region (median is robust here)."""
    sel = mask > 127
    if sel.sum() < 8:
        return None
    pixels = image_rgb[sel]
    if pixels.shape[0] > 20000:
        idx = np.random.default_rng(0).choice(pixels.shape[0], 20000, replace=False)
        pixels = pixels[idx]
    med = np.median(pixels, axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def subtract(base: np.ndarray, *others: np.ndarray) -> np.ndarray:
    """base minus the union of others, as a mask."""
    out = base.copy()
    for other in others:
        if other is None:
            continue
        out = cv2.bitwise_and(out, cv2.bitwise_not((other > 127).astype(np.uint8) * 255))
    return out


def union(*masks: np.ndarray) -> Optional[np.ndarray]:
    acc = None
    for m in masks:
        if m is None:
            continue
        b = (m > 127).astype(np.uint8) * 255
        acc = b if acc is None else cv2.bitwise_or(acc, b)
    return acc


def dilate(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1))
    return cv2.dilate(mask, k)


def cutout_rgba(image_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compose an RGBA image from source pixels and an alpha mask."""
    rgba = np.dstack([image_rgb, alpha])
    return rgba
