"""Output writers: PNG assets, JSON manifest, layered PSD, ZIP bundle.

The deliverable is not a folder of crops - it is a package a designer can
actually open. Three artefacts are produced from one decomposition:

  * `layers/*.png`   RGBA cut-outs, cropped to their bounding box (offsets are
                     recorded in the manifest, so nothing is lost and files
                     stay small).
  * `manifest.json`  machine-readable description: kind, label, depth, bbox,
                     colour, recognised text, confidence, backend provenance.
  * `<name>.psd`     a real layered Photoshop document, correctly stacked.

The PSD writer is what turns this from an experiment into something usable in
a production design workflow.
"""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import Config, DEFAULT_CONFIG
from .types import Decomposition, KIND_BACKGROUND, Layer

log = logging.getLogger("prism.exporters")


# ---------------------------------------------------------------------------
# pixel extraction
# ---------------------------------------------------------------------------

class RgbaCache:
    """Memoises `layer_rgba` for one export run.

    Every consumer needs the same pixels: write_layers, the recomposite
    preview, the PSNR check (another composite) and the PSD writer. Computed
    fresh each time that was 82 un-premultiply passes over a 21-layer stack -
    four times more work than necessary, and the un-premultiply is a
    float32 round-trip over the full canvas per layer.

    Keyed by layer id, which is unique per decomposition.
    """

    def __init__(self, decomp: Decomposition, source_rgb: np.ndarray):
        self._decomp = decomp
        self._source = source_rgb
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, layer: Layer) -> np.ndarray:
        hit = self._cache.get(layer.id)
        if hit is None:
            hit = layer_rgba(self._decomp, layer, self._source)
            self._cache[layer.id] = hit
        return hit


def layer_rgba(decomp: Decomposition, layer: Layer, source_rgb: np.ndarray) -> np.ndarray:
    """Full-canvas RGBA pixels for one layer."""
    if layer.kind == KIND_BACKGROUND:
        plate = decomp.plate if decomp.plate is not None else source_rgb
        alpha = np.full(plate.shape[:2], 255, np.uint8)
        return np.dstack([plate, alpha])

    alpha = layer.mask
    rgb = source_rgb.copy()
    # Un-multiply edge colour slightly so soft edges do not carry a halo of the
    # backdrop colour into the cut-out.
    if decomp.plate is not None:
        soft = (alpha > 8) & (alpha < 248)
        if soft.any():
            a = (alpha[soft].astype(np.float32) / 255.0)[:, None]
            src = rgb[soft].astype(np.float32)
            bg = decomp.plate[soft].astype(np.float32)
            unpremul = np.where(a > 0.08, (src - bg * (1.0 - a)) / np.maximum(a, 0.08), src)
            rgb[soft] = np.clip(unpremul, 0, 255).astype(np.uint8)
    return np.dstack([rgb, alpha])


def crop_to_bbox(rgba: np.ndarray, bbox: Optional[Tuple[int, int, int, int]],
                 padding: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop RGBA to bbox with padding; returns (crop, (offset_x, offset_y))."""
    if bbox is None:
        return rgba, (0, 0)
    h, w = rgba.shape[:2]
    x0 = max(0, bbox[0] - padding)
    y0 = max(0, bbox[1] - padding)
    x1 = min(w, bbox[2] + padding)
    y1 = min(h, bbox[3] + padding)
    if x1 <= x0 or y1 <= y0:
        return rgba, (0, 0)
    return rgba[y0:y1, x0:x1], (x0, y0)


# ---------------------------------------------------------------------------
# PNG + manifest
# ---------------------------------------------------------------------------

def _safe_name(text: str, fallback: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in text.strip().lower()]
    name = "".join(keep).strip("-")
    while "--" in name:
        name = name.replace("--", "-")
    return (name or fallback)[:48]


def write_layers(decomp: Decomposition, source_rgb: np.ndarray, out_dir: str,
                 cfg: Config = DEFAULT_CONFIG,
                 rgba_cache: Optional[RgbaCache] = None,
                 preview: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Write per-layer PNGs plus manifest.json into `out_dir`.

    `preview` accepts an already-composited recomposite so callers that also
    need it for a quality metric don't pay for stacking the whole thing twice.
    """
    layers_dir = os.path.join(out_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)
    cache = rgba_cache or RgbaCache(decomp, source_rgb)

    for index, layer in enumerate(decomp.sorted_layers()):
        rgba = cache.get(layer)
        offset = (0, 0)
        if cfg.crop_layers and layer.kind != KIND_BACKGROUND:
            rgba, offset = crop_to_bbox(rgba, layer.bbox, cfg.crop_padding)

        stem = _safe_name(f"{index:02d}-{layer.kind}-{layer.label}", f"{index:02d}-layer")
        filename = f"{stem}.png"
        Image.fromarray(rgba, mode="RGBA").save(
            os.path.join(layers_dir, filename), optimize=cfg.png_optimize,
            compress_level=cfg.png_compress_level)

        layer.filename = f"layers/{filename}"
        layer.meta["offset"] = list(offset)
        layer.meta["size"] = [int(rgba.shape[1]), int(rgba.shape[0])]

    # A flattened preview makes it obvious at a glance whether the stack
    # recomposites to the original.
    if preview is None:
        preview = composite(decomp, source_rgb, cfg, cache)
    Image.fromarray(preview).save(os.path.join(out_dir, "recomposite.png"),
                                  compress_level=cfg.png_compress_level)
    Image.fromarray(source_rgb).save(os.path.join(out_dir, "source.png"),
                                     compress_level=cfg.png_compress_level)

    manifest = decomp.to_manifest()
    manifest["recomposite"] = "recomposite.png"
    manifest["source_image"] = "source.png"
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def composite(decomp: Decomposition, source_rgb: np.ndarray,
              cfg: Config = DEFAULT_CONFIG,
              rgba_cache: Optional[RgbaCache] = None) -> np.ndarray:
    """Stack all layers back together - a built-in correctness check.

    If decomposition is sound, this closely reproduces the input. The residual
    against the source is reported as a quality metric.
    """
    h, w = source_rgb.shape[:2]
    canvas = np.zeros((h, w, 3), np.float32)
    if decomp.plate is not None:
        canvas[:] = decomp.plate.astype(np.float32)

    cache = rgba_cache or RgbaCache(decomp, source_rgb)
    for layer in decomp.sorted_layers():
        if layer.kind == KIND_BACKGROUND:
            continue
        rgba = cache.get(layer)
        alpha = (rgba[:, :, 3:4].astype(np.float32) / 255.0)
        canvas = rgba[:, :, :3].astype(np.float32) * alpha + canvas * (1.0 - alpha)

    return np.clip(canvas, 0, 255).astype(np.uint8)


def reconstruction_error(decomp: Decomposition, source_rgb: np.ndarray,
                         recon: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Mean absolute error and PSNR between recomposite and source.

    Accepts an already-computed recomposite; export_all passes the preview it
    just wrote rather than compositing the entire stack a second time.
    """
    if recon is None:
        recon = composite(decomp, source_rgb)
    diff = np.abs(recon.astype(np.float32) - source_rgb.astype(np.float32))
    mae = float(diff.mean())
    mse = float((diff ** 2).mean())
    psnr = float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 1e-9 else 99.0
    return {"mae": round(mae, 3), "psnr_db": round(psnr, 2)}


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------

def write_psd(decomp: Decomposition, source_rgb: np.ndarray, path: str,
              rgba_cache: Optional[RgbaCache] = None) -> bool:
    """Write a layered PSD. Returns False when pytoshop is unavailable."""
    try:
        import pytoshop
        from pytoshop import enums
        from pytoshop.user import nested_layers
    except Exception as exc:  # pragma: no cover
        log.info("PSD export unavailable (%s)", exc)
        return False

    try:
        h, w = source_rgb.shape[:2]
        psd_layers = []
        cache = rgba_cache or RgbaCache(decomp, source_rgb)
        # pytoshop stacks the list bottom-last, so build front-to-back.
        for layer in sorted(decomp.sorted_layers(), key=lambda l: -l.z):
            rgba = cache.get(layer)

            # Real PSD layers are stored cropped to their own bounds with an
            # offset, not as full-canvas planes. Matching that keeps the file a
            # fraction of the size and is what Photoshop expects.
            if layer.kind == KIND_BACKGROUND or layer.bbox is None:
                top, left, bottom, right = 0, 0, h, w
            else:
                left, top, right, bottom = layer.bbox
                rgba = rgba[top:bottom, left:right]
            if rgba.size == 0:
                continue

            channels = {
                -1: np.ascontiguousarray(rgba[:, :, 3]),
                0: np.ascontiguousarray(rgba[:, :, 0]),
                1: np.ascontiguousarray(rgba[:, :, 1]),
                2: np.ascontiguousarray(rgba[:, :, 2]),
            }
            name = f"{layer.z:02d} {layer.kind} - {layer.label}"[:60]
            psd_layers.append(
                nested_layers.Image(
                    name=name, visible=True, opacity=255,
                    top=int(top), left=int(left),
                    bottom=int(bottom), right=int(right),
                    channels=channels,
                    color_mode=enums.ColorMode.rgb,
                )
            )

        # pytoshop's RLE path relies on a Cython extension that is not always
        # built from source. Try compressions in order of preference and fall
        # back rather than losing the PSD entirely.
        last_error: Optional[Exception] = None
        for compression in (enums.Compression.rle,
                            enums.Compression.zip,
                            enums.Compression.raw):
            try:
                psd = nested_layers.nested_layers_to_psd(
                    psd_layers, color_mode=enums.ColorMode.rgb,
                    compression=compression)
                with open(path, "wb") as fh:
                    psd.write(fh)
                log.debug("PSD written with %s compression", compression.name)
                return True
            except Exception as exc:
                last_error = exc
                continue

        log.warning("PSD export failed: %s", last_error)
        return False
    except Exception as exc:  # pragma: no cover
        log.warning("PSD export failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# ZIP bundle
# ---------------------------------------------------------------------------

def bundle_zip(out_dir: str, zip_path: str) -> str:
    """Zip an export directory for one-click download."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(out_dir):
            for name in files:
                full = os.path.join(root, name)
                if os.path.abspath(full) == os.path.abspath(zip_path):
                    continue
                zf.write(full, os.path.relpath(full, out_dir))
    return zip_path


def export_all(decomp: Decomposition, source_rgb: np.ndarray, out_dir: str,
               cfg: Config = DEFAULT_CONFIG, make_zip: bool = True,
               make_psd: Optional[bool] = None) -> Dict[str, object]:
    """Run every writer and return the manifest, augmented with quality metrics.

    `make_zip` / `make_psd` exist so a caller serving an interactive request
    can skip the two artefacts nobody is waiting on. The 3D viewer needs the
    layer PNGs and the manifest; the PSD and the ZIP are only fetched if the
    user actually clicks download, so building them inline just adds dead
    time to every job. See `build_downloads`.
    """
    os.makedirs(out_dir, exist_ok=True)
    cache = RgbaCache(decomp, source_rgb)

    # Composite once and use it for both the saved preview and the PSNR
    # check, instead of stacking the full layer set twice.
    preview = composite(decomp, source_rgb, cfg, cache)
    manifest = write_layers(decomp, source_rgb, out_dir, cfg, cache, preview)
    manifest["quality"] = reconstruction_error(decomp, source_rgb, preview)

    want_psd = cfg.export_psd if make_psd is None else (make_psd and cfg.export_psd)
    if want_psd:
        psd_path = os.path.join(out_dir, f"{_safe_name(decomp.source_name, 'layers')}.psd")
        if write_psd(decomp, source_rgb, psd_path, cache):
            manifest["psd"] = os.path.basename(psd_path)

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if make_zip:
        zip_path = os.path.join(out_dir, f"{_safe_name(decomp.source_name, 'layers')}-layers.zip")
        bundle_zip(out_dir, zip_path)
        manifest["zip"] = os.path.basename(zip_path)

    return manifest


def build_downloads(decomp: Decomposition, source_rgb: np.ndarray, out_dir: str,
                    cfg: Config = DEFAULT_CONFIG) -> Dict[str, object]:
    """Produce the deferred PSD + ZIP for an already-exported job, on demand.

    Idempotent: if the artefacts are already on disk it just reports them, so
    repeated download clicks cost nothing.
    """
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    stem = _safe_name(decomp.source_name, "layers")
    cache = RgbaCache(decomp, source_rgb)

    psd_path = os.path.join(out_dir, f"{stem}.psd")
    if cfg.export_psd and not os.path.isfile(psd_path):
        if write_psd(decomp, source_rgb, psd_path, cache):
            manifest["psd"] = os.path.basename(psd_path)
    elif os.path.isfile(psd_path):
        manifest["psd"] = os.path.basename(psd_path)

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # Built after the manifest is rewritten so the ZIP contains the final one.
    zip_path = os.path.join(out_dir, f"{stem}-layers.zip")
    if not os.path.isfile(zip_path):
        bundle_zip(out_dir, zip_path)
    manifest["zip"] = os.path.basename(zip_path)
    return manifest
