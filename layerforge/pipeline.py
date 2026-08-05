"""Pipeline orchestrator.

Runs the stages in dependency order and assembles a `Decomposition`.

    backdrop model -> text -> subjects -> graphics -> background plate -> z-order

Segmentation happens at a bounded working resolution for predictable latency;
every mask is resampled to full resolution before export, so output fidelity is
set by the source image, not by the working size.

Progress is reported through an optional callback so the web UI can stream
stage-by-stage updates while a job runs.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

import cv2
import numpy as np

from . import backends
from .config import Config, DEFAULT_CONFIG
from .imaging import (dominant_color, fit_within, resize_mask, subtract, union)
from .stages import backdrop, graphics as graphics_stage, ordering, reconstruct
from .stages import subject as subject_stage
from .stages import text as text_stage
from .types import (Decomposition, KIND_BACKGROUND, KIND_GRAPHIC, KIND_SUBJECT,
                    KIND_TEXT, Layer)

log = logging.getLogger("layerforge.pipeline")

ProgressFn = Callable[[str, float, str], None]

STAGES = [
    ("backdrop", "Modelling backdrop"),
    ("text", "Detecting type"),
    ("subject", "Matting subjects"),
    ("graphics", "Isolating graphics"),
    ("background", "Reconstructing plate"),
    ("ordering", "Resolving depth"),
]


class LayerForge:
    """Reusable, stateless-per-call image decomposition engine.

    One instance can serve many requests; heavy backends are cached at module
    level, so construction is cheap and thread-safe enough for a web worker.
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or DEFAULT_CONFIG

    # -- public API ---------------------------------------------------------

    def decompose(self, image_rgb: np.ndarray, source_name: str = "image",
                  progress: Optional[ProgressFn] = None) -> Decomposition:
        """Decompose an RGB image into an ordered stack of layers."""
        cfg = self.cfg
        if image_rgb.ndim == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        if image_rgb.shape[2] == 4:
            image_rgb = image_rgb[:, :, :3]
        image_rgb = np.ascontiguousarray(image_rgb.astype(np.uint8))

        full_h, full_w = image_rgb.shape[:2]
        work, scale = fit_within(image_rgb, cfg.max_working_dim)

        result = Decomposition(width=full_w, height=full_h, source_name=source_name)
        result.backends = backends.describe()

        def emit(key: str, frac: float, message: str) -> None:
            if progress:
                try:
                    progress(key, frac, message)
                except Exception:  # never let UI plumbing break a job
                    pass

        timer = _Timer(result.timings)

        # -- 1. backdrop model ---------------------------------------------
        emit("backdrop", 0.05, "Modelling backdrop")
        with timer("backdrop"):
            bg_prob, palette = backdrop.background_probability(work, cfg)
            fg_mask = backdrop.foreground_mask(bg_prob, cfg)

        # -- 2. text --------------------------------------------------------
        emit("text", 0.20, "Detecting type")
        with timer("text"):
            text_lines = text_stage.detect_text(work, cfg)
            text_union = union(*[l.mask for l in text_lines]) if text_lines else None

        # -- 3. subjects ----------------------------------------------------
        emit("subject", 0.40, "Matting subjects")
        with timer("subject"):
            subject_masks = subject_stage.extract_subjects(work, bg_prob, cfg)
            # Type sitting on top of a subject belongs to the text layer.
            if text_union is not None:
                subject_masks = [subtract(m, text_union) for m in subject_masks]
                subject_masks = [m for m in subject_masks if np.count_nonzero(m) > 0]
            subject_union = union(*subject_masks) if subject_masks else None

        # -- 4. graphics ----------------------------------------------------
        emit("graphics", 0.58, "Isolating graphics")
        with timer("graphics"):
            residual = subtract(fg_mask, text_union, subject_union)
            regions = graphics_stage.extract_graphics(work, residual, cfg)

        # -- 5. background plate -------------------------------------------
        emit("background", 0.74, "Reconstructing plate")
        with timer("background"):
            # NB: explicit `is not None` checks - a bare truth test on a numpy
            # mask raises "truth value of an array is ambiguous".
            has_foreground = (bool(regions) or text_union is not None
                              or subject_union is not None)
            all_fg = union(text_union, subject_union,
                           *[r.mask for r in regions]) if has_foreground else None
            # Reconstruct at full resolution for a clean, reusable plate.
            full_fg = resize_mask(all_fg, (full_w, full_h)) if all_fg is not None else None
            plate = reconstruct.reconstruct_background(image_rgb, full_fg, cfg)

        # -- assemble layers -------------------------------------------------
        bg_layer = Layer(
            kind=KIND_BACKGROUND,
            label="background plate",
            mask=np.full((full_h, full_w), 255, np.uint8),
            confidence=0.99,
        )
        bg_layer.color = palette[0] if palette else dominant_color(image_rgb, bg_layer.mask)
        bg_layer.meta["palette"] = [list(c) for c in palette]
        bg_layer.meta["pixels"] = "plate"
        result.add(bg_layer)
        result.plate = plate  # type: ignore[attr-defined]

        for idx, mask in enumerate(subject_masks):
            full_mask = resize_mask(mask, (full_w, full_h))
            layer = Layer(kind=KIND_SUBJECT,
                          label=f"subject {idx + 1}" if len(subject_masks) > 1 else "subject",
                          mask=full_mask, confidence=0.9)
            layer.color = dominant_color(image_rgb, full_mask)
            result.add(layer)

        graphic_index = 0
        for region in regions:
            full_mask = resize_mask(region.mask, (full_w, full_h))
            if np.count_nonzero(full_mask) == 0:
                continue
            if region.kind == KIND_SUBJECT:
                label = "secondary subject"
            else:
                graphic_index += 1
                label = region.label
            layer = Layer(kind=region.kind, label=label, mask=full_mask,
                          confidence=region.score)
            layer.color = region.color or dominant_color(image_rgb, full_mask)
            layer.meta.update(region.meta)
            result.add(layer)

        for line in text_lines:
            full_mask = resize_mask(line.mask, (full_w, full_h))
            if np.count_nonzero(full_mask) == 0:
                continue
            label = "text"
            if line.string:
                snippet = " ".join(line.string.split())[:34]
                if snippet:
                    label = f'text "{snippet}"'
            layer = Layer(kind=KIND_TEXT, label=label, mask=full_mask,
                          confidence=line.score, text=line.string)
            layer.color = line.color or dominant_color(image_rgb, full_mask)
            result.add(layer)

        # -- 6. ordering -----------------------------------------------------
        emit("ordering", 0.90, "Resolving depth")
        with timer("ordering"):
            content = [l for l in result.layers if l.kind != KIND_BACKGROUND]
            ordering.strip_occluded(content)
            result.layers = [l for l in result.layers
                             if l.kind == KIND_BACKGROUND or l.area > 0]
            ordering.assign_z(image_rgb, result.layers)
            for layer in result.layers:
                layer.compute_bbox()

        if len(result.layers) <= 1:
            result.warnings.append(
                "Only a background layer was produced - the image may be a flat "
                "graphic with no separable foreground.")

        emit("done", 1.0, "Complete")
        return result


class _Timer:
    """Context-manager stopwatch that writes into a timings dict."""

    def __init__(self, sink: dict):
        self.sink = sink
        self.key = ""

    def __call__(self, key: str) -> "_Timer":
        self.key = key
        return self

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.sink[self.key] = time.perf_counter() - self._t0
        return False


def decompose_file(path: str, config: Optional[Config] = None,
                   progress: Optional[ProgressFn] = None) -> Decomposition:
    """Convenience helper: load an image from disk and decompose it."""
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    import os
    name = os.path.splitext(os.path.basename(path))[0]
    return LayerForge(config).decompose(rgb, source_name=name, progress=progress)
