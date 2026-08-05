"""Stage 1 - background colour modelling.

Posters, adverts and social creatives are overwhelmingly built on a designed
backdrop: a flat fill, a soft gradient, or a lightly textured plane. Learning
that backdrop explicitly, *before* any segmentation, gives every later stage a
strong prior for what is "not content".

Approach:
  1. Sample the image border, where backdrop is most likely to be exposed.
  2. Cluster those samples in CIE-LAB (perceptually uniform, so a fixed
     distance threshold behaves consistently across hues).
  3. Discard sparse clusters - these are frames, bunting, ornaments and other
     decorative elements that happen to touch the edge.
  4. Optionally fit a smooth spatial model to the surviving background pixels
     so vignettes and gradients are followed rather than torn out.
  5. Emit a per-pixel background probability map.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..config import Config


def _border_samples(lab: np.ndarray, frac: float) -> np.ndarray:
    h, w = lab.shape[:2]
    band_y = max(2, int(round(h * frac)))
    band_x = max(2, int(round(w * frac)))
    parts = [
        lab[:band_y, :].reshape(-1, 3),
        lab[-band_y:, :].reshape(-1, 3),
        lab[:, :band_x].reshape(-1, 3),
        lab[:, -band_x:].reshape(-1, 3),
    ]
    samples = np.concatenate(parts, axis=0).astype(np.float32)
    # Subsample for k-means speed; the border is highly redundant.
    if samples.shape[0] > 60000:
        idx = np.random.default_rng(0).choice(samples.shape[0], 60000, replace=False)
        samples = samples[idx]
    return samples


def _palette(samples: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """k-means over LAB samples -> (centers, share_of_samples)."""
    k = max(1, min(k, max(1, samples.shape[0] // 50)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.6)
    cv2.setRNGSeed(seed)
    _compact, labels, centers = cv2.kmeans(
        samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    shares = np.bincount(labels, minlength=k).astype(np.float32) / float(labels.size)
    return centers, shares


def background_probability(image_rgb: np.ndarray, cfg: Config
                           ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Return (probability map float32 0..1, list of backdrop RGB colours).

    High probability means "this pixel looks like the designed backdrop".
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    samples = _border_samples(lab, cfg.bg_border_frac)
    centers, shares = _palette(samples, cfg.bg_palette_size, cfg.random_seed)

    keep = shares >= cfg.bg_cluster_min_share
    if not keep.any():
        keep[int(np.argmax(shares))] = True
    centers = centers[keep]

    # Distance from every pixel to the nearest surviving backdrop colour.
    flat = lab.reshape(-1, 3)
    best = None
    for c in centers:
        d = np.linalg.norm(flat - c[None, :], axis=1)
        best = d if best is None else np.minimum(best, d)
    dist = best.reshape(lab.shape[:2])

    sigma = max(cfg.bg_distance_sigma, 1.0)
    prob = np.exp(-(dist ** 2) / (2.0 * sigma * sigma)).astype(np.float32)

    if cfg.bg_fit_gradient:
        prob = _augment_with_gradient_fit(lab, prob, cfg)

    # A little smoothing removes salt-and-pepper flicker at soft edges.
    prob = cv2.GaussianBlur(prob, (5, 5), 0)

    palette_rgb = []
    for c in centers:
        patch = np.uint8([[[c[0], c[1], c[2]]]])
        rgb = cv2.cvtColor(patch, cv2.COLOR_LAB2RGB)[0, 0]
        palette_rgb.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))

    return np.clip(prob, 0.0, 1.0), palette_rgb


def _augment_with_gradient_fit(lab: np.ndarray, prob: np.ndarray, cfg: Config) -> np.ndarray:
    """Fit a smooth spatial colour surface to confident background pixels.

    Many creatives use vignettes or duotone gradients. A single colour cluster
    cannot represent those, so pixels far from the centre get wrongly flagged
    as foreground. Fitting a low-order polynomial in (x, y) per LAB channel
    over the *confident* background pixels captures the gradient, and pixels
    that agree with the fitted surface are folded back into the background.
    """
    h, w = prob.shape
    confident = prob > 0.75
    if confident.sum() < (h * w) * 0.04:
        return prob

    ys, xs = np.nonzero(confident)
    if ys.size > 40000:
        idx = np.random.default_rng(1).choice(ys.size, 40000, replace=False)
        ys, xs = ys[idx], xs[idx]

    xn = (xs / float(w)).astype(np.float32)
    yn = (ys / float(h)).astype(np.float32)
    # Quadratic basis: 1, x, y, x^2, y^2, xy
    A = np.stack([np.ones_like(xn), xn, yn, xn * xn, yn * yn, xn * yn], axis=1)

    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    gxn = gx / float(w)
    gyn = gy / float(h)
    B = np.stack([np.ones_like(gxn), gxn, gyn, gxn * gxn, gyn * gyn, gxn * gyn], axis=-1)

    residual = np.zeros((h, w), np.float32)
    for ch in range(3):
        target = lab[ys, xs, ch]
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        fitted = B @ coef
        residual += (lab[:, :, ch] - fitted) ** 2

    residual = np.sqrt(residual)
    sigma = max(cfg.bg_distance_sigma * 1.15, 1.0)
    grad_prob = np.exp(-(residual ** 2) / (2.0 * sigma * sigma)).astype(np.float32)

    # A pixel is background if EITHER model explains it.
    return np.maximum(prob, grad_prob)


def foreground_mask(prob: np.ndarray, cfg: Config) -> np.ndarray:
    """Binary foreground mask (uint8 0/255) from the background probability."""
    fg = (prob < cfg.bg_prob_threshold).astype(np.uint8) * 255
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return fg
