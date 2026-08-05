"""Stage 3 - salient subject extraction.

The "hero" of a creative: a person, a model, a product shot. Two paths:

  * rembg (U^2-Net / IS-Net family) when weights are available - a dedicated
    salient-object matting network, far ahead of anything classical on hair,
    fabric edges and jewellery.
  * a classical fallback that fuses three cheap saliency cues and refines the
    result with GrabCut, requiring no downloads at all.

The fallback deliberately combines the *backdrop model* from stage 1 with
spectral-residual and fine-grained saliency. On designed creatives the backdrop
prior is the strongest single cue, and saliency resolves which of the remaining
foreground blobs is the actual subject rather than an ornament.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

from ..backends import rembg_session
from ..config import Config
from ..imaging import (clean_mask, fill_holes, largest_components, refine_alpha,
                       texture_score)

log = logging.getLogger("layerforge.subject")


# ---------------------------------------------------------------------------
# neural path
# ---------------------------------------------------------------------------

def _rembg_alpha(image_rgb: np.ndarray, cfg: Config) -> Optional[np.ndarray]:
    session = rembg_session(cfg.rembg_model)
    if session is None:
        return None
    try:
        from PIL import Image
        from rembg import remove  # type: ignore

        pil = Image.fromarray(image_rgb)
        out = remove(pil, session=session, post_process_mask=True)
        alpha = np.array(out.convert("RGBA"))[:, :, 3]
        return alpha.astype(np.uint8)
    except Exception as exc:  # pragma: no cover
        log.info("rembg inference failed (%s); falling back to saliency", exc)
        return None


# ---------------------------------------------------------------------------
# classical path
# ---------------------------------------------------------------------------

def _saliency_map(image_rgb: np.ndarray) -> np.ndarray:
    """Fuse spectral-residual and fine-grained saliency into one 0..255 map."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    maps: List[np.ndarray] = []

    for factory in ("StaticSaliencySpectralResidual_create",
                    "StaticSaliencyFineGrained_create"):
        try:
            algo = getattr(cv2.saliency, factory)()
            ok, sal = algo.computeSaliency(bgr)
            if ok and sal is not None:
                sal = np.nan_to_num(sal.astype(np.float32))
                lo, hi = float(sal.min()), float(sal.max())
                if hi - lo > 1e-6:
                    maps.append((sal - lo) / (hi - lo))
        except Exception:
            continue

    if not maps:
        # Last-resort cue: distance from the image's dominant colour.
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        mean = lab.reshape(-1, 3).mean(axis=0)
        d = np.linalg.norm(lab - mean[None, None, :], axis=2)
        d = d / (d.max() + 1e-6)
        maps.append(d.astype(np.float32))

    fused = np.mean(maps, axis=0)
    fused = cv2.GaussianBlur(fused, (7, 7), 0)
    return np.clip(fused * 255.0, 0, 255).astype(np.uint8)


def _center_prior(shape) -> np.ndarray:
    """Gaussian weighting toward the frame centre.

    Subjects are composed near the optical centre far more often than not; this
    breaks ties between equally salient blobs without hard-coding positions.
    """
    h, w = shape
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    return np.exp(-(xs ** 2 / 0.85 + ys ** 2 / 1.25)).astype(np.float32)


def _grabcut_refine(image_rgb: np.ndarray, coarse: np.ndarray, cfg: Config) -> np.ndarray:
    """Refine a coarse mask with GrabCut seeded from confident regions."""
    if np.count_nonzero(coarse) < 400:
        return coarse
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gc = np.full(coarse.shape, cv2.GC_PR_BGD, np.uint8)

    sure_fg = cv2.erode(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    maybe_fg = cv2.dilate(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    sure_bg = cv2.bitwise_not(
        cv2.dilate(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))))

    gc[maybe_fg > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    gc[sure_bg > 0] = cv2.GC_BGD

    if not (gc == cv2.GC_FGD).any() or not (gc == cv2.GC_BGD).any():
        return coarse
    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, gc, None, bgd, fgd, cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
        out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        # Guard against GrabCut collapsing or exploding the mask.
        before, after = np.count_nonzero(coarse), np.count_nonzero(out)
        if after < before * 0.35 or after > before * 2.6:
            return coarse
        return out
    except Exception as exc:  # pragma: no cover
        log.debug("grabcut failed: %s", exc)
        return coarse


def _classical_alpha(image_rgb: np.ndarray, bg_prob: np.ndarray, cfg: Config) -> np.ndarray:
    fg_prior = 1.0 - bg_prob
    saliency = _saliency_map(image_rgb).astype(np.float32) / 255.0
    center = _center_prior(bg_prob.shape)

    # Backdrop evidence dominates; saliency and centre-bias resolve ambiguity.
    combined = (0.55 * fg_prior + 0.30 * saliency + 0.15 * saliency * center)
    combined = cv2.GaussianBlur(combined, (9, 9), 0)

    values = combined[combined > 0.05]
    if values.size == 0:
        return np.zeros(bg_prob.shape, np.uint8)

    total = float(combined.size)
    coarse = np.zeros(bg_prob.shape, np.uint8)

    # Busy compositions (dense collage, allover pattern, full-bleed artwork)
    # push the saliency distribution upward until a fixed threshold selects
    # nearly everything. Walking the percentile up until coverage respects the
    # cap keeps the subject layer meaningful instead of swallowing the canvas,
    # and leaves the remainder for the graphics stage to classify.
    for percentile in (62, 72, 80, 86, 91, 95):
        thr = float(np.percentile(values, percentile))
        thr = min(max(thr, 0.28), 0.86)
        candidate = (combined > thr).astype(np.uint8) * 255
        candidate = clean_mask(candidate, open_k=5, close_k=13)
        candidate = fill_holes(candidate)
        coverage = float(np.count_nonzero(candidate)) / total
        coarse = candidate
        if coverage <= cfg.subject_max_coverage:
            break

    if np.count_nonzero(coarse) == 0:
        return coarse
    return _grabcut_refine(image_rgb, coarse, cfg)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def extract_subjects(image_rgb: np.ndarray, bg_prob: np.ndarray, cfg: Config
                     ) -> List[np.ndarray]:
    """Return a list of subject alpha masks, largest/most confident first."""
    if not cfg.enable_subject:
        return []

    alpha = _rembg_alpha(image_rgb, cfg)
    used_neural = alpha is not None
    if alpha is None:
        alpha = _classical_alpha(image_rgb, bg_prob, cfg)

    binary = (alpha >= cfg.subject_alpha_threshold).astype(np.uint8) * 255
    binary = clean_mask(binary, open_k=3, close_k=9)
    binary = fill_holes(binary)

    h, w = binary.shape
    min_area = int(h * w * cfg.subject_min_area_frac)
    parts = largest_components(binary, min_area=min_area, max_count=6)
    if not parts and np.count_nonzero(binary) > min_area:
        parts = [binary]

    masks: List[np.ndarray] = []
    for part in parts:
        # Reject flat vector artwork that slipped through - it belongs in the
        # graphic layer group, not the subject group.
        if texture_score(image_rgb, part) < 0.22 and not used_neural:
            continue
        soft = part
        if used_neural:
            # Preserve the network's soft matte inside this component.
            soft = np.where(part > 0, alpha, 0).astype(np.uint8)
        soft = refine_alpha(image_rgb, soft, cfg.alpha_guided_radius,
                            cfg.alpha_guided_eps, cfg.alpha_feather)
        masks.append(soft)

    return masks
