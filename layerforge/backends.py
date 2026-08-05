"""Optional-dependency detection with graceful degradation.

The pipeline is designed so that *nothing* here is required. Each capability
has a classical computer-vision fallback that needs no downloaded weights, so
the system runs in air-gapped or restricted environments. When the optional
neural backends are present the same pipeline transparently upgrades to them.

This keeps one code path, one output contract, and no runtime surprises.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional

log = logging.getLogger("layerforge.backends")

# Allow operators to force the classical path (useful for benchmarking and for
# reproducing results in restricted environments).
FORCE_CLASSICAL = os.environ.get("LAYERFORGE_CLASSICAL", "").lower() in {"1", "true", "yes"}


@functools.lru_cache(maxsize=1)
def rembg_session(model_name: str = "isnet-general-use"):
    """Return a warm rembg session, or None when unavailable.

    rembg downloads ONNX weights on first use. Any failure (missing package,
    no network, corrupted cache) degrades to the classical saliency path.
    """
    if FORCE_CLASSICAL:
        return None
    try:
        from rembg import new_session  # type: ignore

        return new_session(model_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        log.info("rembg unavailable (%s); using classical saliency fallback", exc)
        return None


@functools.lru_cache(maxsize=1)
def easyocr_reader():
    """Return an EasyOCR reader (CRAFT text detector), or None."""
    if FORCE_CLASSICAL:
        return None
    try:
        import easyocr  # type: ignore

        return easyocr.Reader(["en"], gpu=_torch_gpu(), verbose=False)
    except Exception as exc:  # pragma: no cover
        log.info("easyocr unavailable (%s); using MSER/SWT text fallback", exc)
        return None


@functools.lru_cache(maxsize=1)
def has_tesseract() -> bool:
    """Tesseract is used only to *recognise* text inside already-detected boxes."""
    try:
        import pytesseract  # type: ignore

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _torch_gpu() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def has_lama() -> bool:
    """Large-mask inpainting via simple-lama-inpainting, if installed."""
    if FORCE_CLASSICAL:
        return False
    try:
        import simple_lama_inpainting  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def lama_inpainter():
    if not has_lama():
        return None
    try:
        from simple_lama_inpainting import SimpleLama  # type: ignore

        return SimpleLama()
    except Exception as exc:  # pragma: no cover
        log.info("LaMa init failed (%s); using Telea/structural inpaint", exc)
        return None


@functools.lru_cache(maxsize=1)
def has_pytoshop() -> bool:
    try:
        import pytoshop  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def describe() -> dict:
    """Human-readable summary of which backend served each capability.

    Surfaced in the manifest and in the UI so evaluation is transparent about
    what actually ran.
    """
    return {
        "subject": "rembg" if rembg_session() is not None else "saliency+grabcut",
        "text_detection": "easyocr-craft" if easyocr_reader() is not None else "mser+swt",
        "text_recognition": "tesseract" if has_tesseract() else "none",
        "inpaint": "lama" if lama_inpainter() is not None else "structural+telea",
        "psd_export": "pytoshop" if has_pytoshop() else "unavailable",
        "mode": "classical (forced)" if FORCE_CLASSICAL else "auto",
    }
