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
    missing onnxruntime, no network, corrupted cache) degrades to the classical
    saliency path.

    NB: BaseException, not Exception. Installed without its `[cpu]`/`[gpu]`
    extra, rembg has no ONNX runtime and terminates the interpreter on import
    rather than raising a normal error. An optional dependency must never be
    able to take the process down.
    """
    if FORCE_CLASSICAL:
        return None
    try:
        from rembg import new_session  # type: ignore

        return new_session(model_name)
    except BaseException as exc:  # pragma: no cover - environment dependent
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
    except BaseException as exc:  # pragma: no cover
        log.info("easyocr unavailable (%s); using MSER/SWT text fallback", exc)
        return None


@functools.lru_cache(maxsize=1)
def has_rembg() -> bool:
    """Is rembg importable with a working ONNX runtime?

    Deliberately does NOT construct a session. `new_session` downloads ~170 MB
    of weights on first call, so anything that merely *reports* capability
    (health endpoint, launcher banner) must use this instead of
    `rembg_session()` or it will block startup on a silent download.
    """
    if FORCE_CLASSICAL:
        return False
    try:
        import rembg  # type: ignore  # noqa: F401

        return True
    except BaseException:
        return False


@functools.lru_cache(maxsize=1)
def has_easyocr() -> bool:
    """Is easyocr importable? Does not build a Reader (which downloads weights)."""
    if FORCE_CLASSICAL:
        return False
    try:
        import easyocr  # type: ignore  # noqa: F401

        return True
    except BaseException:
        return False


# Locations Tesseract commonly lands in. A process started outside a login
# shell — a launchd job, an IDE, a double-clicked app — often has a minimal
# PATH that omits Homebrew entirely, so relying on PATH alone reports the
# binary as missing on machines where it is plainly installed.
_TESSERACT_CANDIDATES = (
    "/opt/homebrew/bin/tesseract",      # Homebrew, Apple Silicon
    "/usr/local/bin/tesseract",         # Homebrew, Intel macOS
    "/usr/bin/tesseract",               # Linux distro packages
    "/opt/local/bin/tesseract",         # MacPorts
    "C:/Program Files/Tesseract-OCR/tesseract.exe",
)


@functools.lru_cache(maxsize=1)
def has_tesseract() -> bool:
    """Tesseract is used only to *recognise* text inside already-detected boxes.

    Resolves the binary explicitly rather than trusting PATH, and points
    pytesseract at whatever it finds.
    """
    try:
        import pytesseract  # type: ignore
    except BaseException:
        return False

    # 1) whatever pytesseract is already configured with, or PATH
    try:
        pytesseract.get_tesseract_version()
        return True
    except BaseException:
        pass

    # 2) shutil, which respects PATH but normalises platform differences
    import shutil

    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        try:
            pytesseract.get_tesseract_version()
            return True
        except BaseException:
            pass

    # 3) well-known install locations
    for candidate in _TESSERACT_CANDIDATES:
        if os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            try:
                pytesseract.get_tesseract_version()
                log.info("Tesseract resolved at %s (not on PATH)", candidate)
                return True
            except BaseException:
                continue

    return False


def tesseract_path() -> Optional[str]:
    """Path to the resolved Tesseract binary, or None. Diagnostics only."""
    if not has_tesseract():
        return None
    try:
        import pytesseract  # type: ignore

        return str(pytesseract.pytesseract.tesseract_cmd)
    except BaseException:
        return None


def _torch_gpu() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except BaseException:
        return False


@functools.lru_cache(maxsize=1)
def has_lama() -> bool:
    """Large-mask inpainting via simple-lama-inpainting, if installed."""
    if FORCE_CLASSICAL:
        return False
    try:
        import simple_lama_inpainting  # type: ignore  # noqa: F401

        return True
    except BaseException:
        return False


@functools.lru_cache(maxsize=1)
def lama_inpainter():
    if not has_lama():
        return None
    try:
        from simple_lama_inpainting import SimpleLama  # type: ignore

        return SimpleLama()
    except BaseException as exc:  # pragma: no cover
        log.info("LaMa init failed (%s); using Telea/structural inpaint", exc)
        return None


@functools.lru_cache(maxsize=1)
def has_pytoshop() -> bool:
    try:
        import pytoshop  # type: ignore  # noqa: F401

        return True
    except BaseException:
        return False


def describe() -> dict:
    """Human-readable summary of which backend serves each capability.

    Surfaced in the manifest and in the UI so evaluation is transparent about
    what actually ran.

    Uses import-level availability checks only. Constructing the neural
    backends here would trigger large weight downloads every time anything
    asked "what's available?" — including the health endpoint on every page
    load. Sessions are built lazily, on first real use.
    """
    return {
        "subject": "rembg" if has_rembg() else "saliency+grabcut",
        "text_detection": "easyocr-craft" if has_easyocr() else "mser+swt",
        "text_recognition": "tesseract" if has_tesseract() else "none",
        "tesseract_path": tesseract_path() or "",
        "inpaint": "lama" if has_lama() else "structural+telea",
        "psd_export": "pytoshop" if has_pytoshop() else "unavailable",
        "mode": "classical (forced)" if FORCE_CLASSICAL else "auto",
    }
