"""Stage 5 - background plate reconstruction.

Removing every foreground element leaves holes. A usable background layer has
to fill them plausibly, because the background is the one layer that must cover
the entire canvas.

Three tiers, best available first:

  1. LaMa (`simple-lama-inpainting`) - a large-mask inpainting network, the
     right tool when the backdrop carries real texture or structure.
  2. Structural fill - fit the smooth colour surface learned in stage 1 and
     paste it into the holes. On flat and gradient backdrops, which is most
     designed creatives, this is effectively exact and costs nothing.
  3. Telea diffusion - OpenCV's fast marching inpaint as the final fallback.

Tiers 2 and 3 are blended: the structural fill supplies correct low-frequency
colour, Telea supplies local detail near hole boundaries, and a distance-based
weight favours Telea close to known pixels and the structural surface deep
inside large holes, where diffusion would otherwise smear.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..backends import lama_inpainter
from ..config import Config
from ..imaging import dilate

log = logging.getLogger("prism.reconstruct")


def _lama_fill(image_rgb: np.ndarray, hole: np.ndarray) -> np.ndarray | None:
    inpainter = lama_inpainter()
    if inpainter is None:
        return None
    try:
        from PIL import Image

        result = inpainter(Image.fromarray(image_rgb), Image.fromarray(hole))
        arr = np.array(result.convert("RGB"))
        if arr.shape[:2] != image_rgb.shape[:2]:
            arr = cv2.resize(arr, (image_rgb.shape[1], image_rgb.shape[0]),
                             interpolation=cv2.INTER_LANCZOS4)
        return arr
    except Exception as exc:  # pragma: no cover
        log.info("LaMa inpaint failed (%s); using structural fill", exc)
        return None


def _structural_fill(image_rgb: np.ndarray, hole: np.ndarray, cfg: Config) -> np.ndarray:
    """Reconstruct low-frequency backdrop colour by iterative coarse diffusion.

    Working at a small scale makes diffusion both fast and well-behaved: the
    hole occupies few pixels, so known colour propagates across it in a handful
    of blur-and-restore iterations, producing a smooth surface that follows any
    gradient present in the original backdrop.
    """
    h, w = image_rgb.shape[:2]
    scale = max(cfg.inpaint_coarse_scale, 0.05)
    sw, sh = max(8, int(w * scale)), max(8, int(h * scale))

    small = cv2.resize(image_rgb, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    small_hole = cv2.resize(hole, (sw, sh), interpolation=cv2.INTER_NEAREST) > 127

    known = ~small_hole
    if known.sum() < 16:
        return image_rgb.copy()

    # Seed holes with the mean of known pixels so diffusion starts unbiased.
    seed = small.copy()
    mean_color = small[known].mean(axis=0)
    seed[small_hole] = mean_color

    for _ in range(90):
        blurred = cv2.GaussianBlur(seed, (0, 0), sigmaX=2.0, sigmaY=2.0)
        seed[small_hole] = blurred[small_hole]
        seed[known] = small[known]

    coarse = cv2.resize(seed, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(coarse, 0, 255).astype(np.uint8)


def reconstruct_background(image_rgb: np.ndarray, foreground: np.ndarray,
                           cfg: Config) -> np.ndarray:
    """Return a full-canvas RGB background plate with foreground removed."""
    if not cfg.enable_inpaint or foreground is None or np.count_nonzero(foreground) == 0:
        return image_rgb.copy()

    # Widen the hole so anti-aliased fringes and soft shadows are not sampled.
    hole = dilate((foreground > 127).astype(np.uint8) * 255, cfg.inpaint_mask_dilate)

    if np.count_nonzero(hole) >= hole.size * 0.985:
        # Essentially nothing known: fall back to the median colour.
        flat = image_rgb.reshape(-1, 3)
        med = np.median(flat, axis=0).astype(np.uint8)
        return np.full_like(image_rgb, med)

    lama = _lama_fill(image_rgb, hole)
    if lama is not None:
        return lama

    structural = _structural_fill(image_rgb, hole, cfg)

    try:
        telea = cv2.inpaint(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), hole,
                            cfg.inpaint_radius, cv2.INPAINT_TELEA)
        telea = cv2.cvtColor(telea, cv2.COLOR_BGR2RGB)
    except Exception:
        telea = structural

    # Blend by depth into the hole: Telea near edges, structural deep inside.
    dist = cv2.distanceTransform((hole > 127).astype(np.uint8), cv2.DIST_L2, 5)
    falloff = max(float(cfg.inpaint_radius) * 2.5, 8.0)
    weight = np.clip(dist / falloff, 0.0, 1.0)[..., None].astype(np.float32)

    blended = telea.astype(np.float32) * (1.0 - weight) + structural.astype(np.float32) * weight
    out = image_rgb.copy()
    sel = hole > 127
    out[sel] = np.clip(blended, 0, 255).astype(np.uint8)[sel]

    # Gentle smoothing only inside the filled area hides seams.
    smoothed = cv2.bilateralFilter(out, 7, 45, 45)
    out[sel] = smoothed[sel]
    return out
