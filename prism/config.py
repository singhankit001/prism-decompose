"""Tunable configuration for the Prism decomposition pipeline.

Every threshold the pipeline depends on lives here so behaviour can be tuned
without touching algorithm code. Defaults were chosen to generalise across
poster/advert style imagery rather than to fit any particular sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class Config:
    # ---- working resolution -------------------------------------------------
    # Segmentation runs at a bounded resolution for speed; masks are resampled
    # back to full resolution before export so output quality is not reduced.
    max_working_dim: int = 1400

    # ---- background colour model -------------------------------------------
    # Fraction of the image border sampled to learn the background palette.
    bg_border_frac: float = 0.06
    # Number of colour clusters fitted to the border samples.
    bg_palette_size: int = 5
    # A border cluster must hold at least this share of samples to be treated
    # as real background (filters out bunting, ornaments, frames in the border).
    bg_cluster_min_share: float = 0.08
    # LAB distance at which a pixel is considered fully foreground.
    bg_distance_sigma: float = 14.0
    # Probability above which a pixel is committed to the background layer.
    bg_prob_threshold: float = 0.45
    # Fit a smooth spatial model so vignettes/gradients are treated as
    # background rather than being torn out as foreground.
    bg_fit_gradient: bool = True

    # ---- text detection -----------------------------------------------------
    enable_text: bool = True
    # Character candidate geometry filters (as a fraction of image height).
    text_min_char_h_frac: float = 0.008
    text_max_char_h_frac: float = 0.30
    text_min_area_px: int = 24
    text_max_aspect: float = 12.0
    # Stroke-width consistency: real glyphs have low stroke width variance.
    text_max_swt_cv: float = 0.65
    # Horizontal gap (in multiples of glyph height) still merged into one line.
    text_line_gap_ratio: float = 1.1
    # Run OCR to transcribe detected text. Detection, cut-out and layering all
    # work without it - only the recognised string is lost - and each block
    # costs a Tesseract subprocess, so this is the first thing to trade away
    # on CPU-constrained hosting.
    text_recognise: bool = True
    # Minimum OCR confidence for a grouped line to be kept when OCR is used.
    text_min_ocr_conf: float = 30.0
    # Keep geometrically-valid lines even when OCR fails (stylised display type
    # often defeats OCR but is still visually text).
    text_keep_unrecognised: bool = True
    # Cap on emitted text layers so a noisy image cannot flood the output.
    max_text_layers: int = 10
    # Max coefficient of variation of glyph heights within one block, measured
    # over the taller half of the components (caps and ascenders) so that
    # descenders and punctuation do not penalise ordinary mixed-case text.
    text_max_height_cv: float = 0.55
    # Emit every text block as one combined "text" layer instead of one layer
    # per block. Useful when the type is wanted as a single removable overlay.
    merge_text_layers: bool = False
    # Blocks below this canvas coverage need >=3 glyph components to qualify.
    text_min_coverage: float = 0.0012

    # ---- subject / photographic region --------------------------------------
    enable_subject: bool = True
    # Preferred rembg model when the optional dependency is installed.
    rembg_model: str = "isnet-general-use"
    # Alpha above which a rembg pixel counts as subject.
    subject_alpha_threshold: int = 128
    # Minimum blob area (fraction of canvas) to survive as its own subject.
    subject_min_area_frac: float = 0.0022
    # Upper bound on total subject coverage. On busy compositions a saliency
    # threshold can swallow the whole canvas; the classical path re-thresholds
    # upward until it respects this cap, leaving room for graphics and text.
    subject_max_coverage: float = 0.55
    # Watershed splitting of one oversized silhouette into separate subjects.
    # Lower peak_ratio finds more cores (more, smaller subjects).
    subject_split_peak_ratio: float = 0.34
    subject_max_parts: int = 8
    # GrabCut refinement iterations for the classical fallback path.
    grabcut_iters: int = 4

    # ---- graphic / vector element detection ---------------------------------
    enable_graphics: bool = True
    graphic_min_area_frac: float = 0.00035
    # Colour quantisation level used to find flat-fill graphic regions.
    graphic_quant_colors: int = 12
    # Regions scoring above this on the texture metric are photographic,
    # below it they are flat vector-style artwork.
    photographic_texture_threshold: float = 0.42
    # Merge graphic blobs whose bounding boxes are within this many pixels.
    graphic_merge_dilate: int = 6
    # If proximity merging produces an element covering more than this share of
    # the canvas, fall back to its unmerged connected components.
    graphic_split_coverage: float = 0.28
    # A photographic region larger than this is scene/backdrop artwork rather
    # than a discrete subject, and is labelled accordingly.
    decor_coverage_threshold: float = 0.35
    # Cap on emitted graphic layers (largest kept) to keep output manageable.
    max_graphic_layers: int = 20

    # ---- background reconstruction -----------------------------------------
    enable_inpaint: bool = True
    # Dilation applied to the foreground union before inpainting, so anti
    # aliased edges and drop shadows do not bleed into the clean plate.
    inpaint_mask_dilate: int = 5
    # Telea radius at full resolution.
    inpaint_radius: int = 7
    # Downscale factor for the coarse structural inpaint pass.
    inpaint_coarse_scale: float = 0.25

    # ---- alpha matting ------------------------------------------------------
    # Feather radius applied to cut-out edges for clean compositing.
    alpha_feather: int = 2
    # Guided-filter refinement of mask edges against the source image.
    alpha_guided_radius: int = 8
    alpha_guided_eps: float = 1e-3

    # ---- export -------------------------------------------------------------
    # Crop layers to their bounding box and record the offset (smaller files,
    # and the 3D viewer positions planes from the manifest).
    crop_layers: bool = True
    crop_padding: int = 2
    export_psd: bool = True
    png_optimize: bool = False
    # zlib level for layer PNGs. The default (6) spends most of the export
    # budget squeezing files that are about to be thrown away after one view;
    # level 1 encodes several times faster for a modest size increase, which
    # is the right trade for an interactive service. Raise it for archival
    # batch exports where wall-clock time doesn't matter.
    png_compress_level: int = 1

    # ---- misc ---------------------------------------------------------------
    random_seed: int = 17
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


DEFAULT_CONFIG = Config()
