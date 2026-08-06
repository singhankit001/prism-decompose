"""Smoke and invariant tests for the decomposition pipeline.

These run without any downloaded model weights, so they pass in CI and in
air-gapped environments. Synthetic fixtures are used deliberately: they have a
known ground-truth structure, so the assertions test real behaviour rather than
memorised output for a particular sample image.

    python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism import Config, Prism                      # noqa: E402
from prism.exporters import (composite, export_all,        # noqa: E402
                                  reconstruction_error)
from prism.imaging import texture_score                    # noqa: E402
from prism.stages import backdrop                          # noqa: E402
from prism.types import KIND_BACKGROUND                    # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def make_poster(w: int = 600, h: int = 800) -> np.ndarray:
    """A synthetic creative: flat backdrop, photo-like subject, flat graphic, text."""
    rng = np.random.default_rng(11)
    img = np.full((h, w, 3), (232, 226, 210), np.uint8)          # flat backdrop

    # photographic-looking subject: smooth noise, so texture metrics see detail
    blob = rng.integers(60, 200, (240, 200, 3), dtype=np.uint8)
    blob = cv2.GaussianBlur(blob, (0, 0), 3)
    img[420:660, 200:400] = blob

    # flat vector graphic: solid circle, no interior texture
    cv2.circle(img, (int(w * 0.78), int(h * 0.16)), 54, (40, 90, 170), -1)

    # text: high-contrast glyphs
    cv2.putText(img, "HELLO", (52, 150), cv2.FONT_HERSHEY_SIMPLEX,
                2.1, (30, 30, 40), 6, cv2.LINE_AA)
    cv2.putText(img, "WORLD DESIGN", (52, 226), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (30, 30, 40), 3, cv2.LINE_AA)
    return img


def make_gradient_poster(w: int = 480, h: int = 640) -> np.ndarray:
    """Backdrop with a strong vertical gradient, to exercise the surface fit."""
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 40 + ys * 150
    img[..., 1] = 60 + ys * 120
    img[..., 2] = 120 + ys * 90
    img = img.astype(np.uint8)
    cv2.circle(img, (w // 2, h // 2), 90, (250, 240, 60), -1)
    return img


@pytest.fixture(scope="module")
def result():
    return Prism(Config()).decompose(make_poster(), source_name="synthetic")


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------

def test_produces_multiple_layers(result):
    assert len(result.layers) >= 2, "expected background plus content layers"


def test_exactly_one_background(result):
    backgrounds = result.by_kind(KIND_BACKGROUND)
    assert len(backgrounds) == 1
    assert backgrounds[0].z == 0, "background must sit at the back of the stack"


def test_background_covers_full_canvas(result):
    bg = result.by_kind(KIND_BACKGROUND)[0]
    assert bg.mask.shape == (result.height, result.width)
    assert bg.coverage == pytest.approx(1.0, abs=1e-6)


def test_plate_was_reconstructed(result):
    assert result.plate is not None
    assert result.plate.shape == (result.height, result.width, 3)


def test_z_order_is_dense_and_unique(result):
    zs = sorted(l.z for l in result.layers)
    assert zs == list(range(len(zs))), "z values must be a dense 0..n-1 sequence"


def test_every_layer_has_bbox_and_area(result):
    for layer in result.layers:
        assert layer.area > 0, f"{layer.label} is empty"
        assert layer.bbox is not None


def test_content_layers_do_not_overlap(result):
    """Front-to-back subtraction must yield a clean partition of the canvas."""
    acc = np.zeros((result.height, result.width), np.int32)
    for layer in result.layers:
        if layer.kind == KIND_BACKGROUND:
            continue
        acc += (layer.mask > 127).astype(np.int32)
    overlap = int((acc > 1).sum())
    total = max(int((acc > 0).sum()), 1)
    assert overlap / total < 0.02, f"{overlap} pixels claimed by multiple layers"


def test_manifest_is_json_serialisable(result):
    import json
    payload = json.dumps(result.to_manifest())
    assert len(payload) > 0
    assert '"layers"' in payload


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------

def test_recomposite_matches_source(result):
    """Stacking the layers back together must approximate the input."""
    metrics = reconstruction_error(result, make_poster())
    assert metrics["psnr_db"] > 22.0, f"weak reconstruction: {metrics}"


def test_composite_shape(result):
    out = composite(result, make_poster())
    assert out.shape == (result.height, result.width, 3)


def test_texture_metric_separates_photo_from_flat():
    """The photo/vector discriminator is the core of graphic classification."""
    img = make_poster()

    photo = np.zeros(img.shape[:2], np.uint8)
    photo[420:660, 200:400] = 255

    flat = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(flat, (int(img.shape[1] * 0.78), int(img.shape[0] * 0.16)), 54, 255, -1)

    assert texture_score(img, photo) > texture_score(img, flat)


def test_backdrop_model_finds_flat_background():
    img = make_poster()
    prob, palette = backdrop.background_probability(img, Config())
    assert len(palette) >= 1
    # A corner is unambiguously backdrop.
    assert prob[5, 5] > 0.8
    # The subject region is not.
    assert prob[540, 300] < 0.6


def test_backdrop_model_follows_gradient():
    """A vertical gradient must be modelled as backdrop, not torn out."""
    img = make_gradient_poster()
    prob, _ = backdrop.background_probability(img, Config())
    assert prob[10, 10] > 0.7, "top of gradient should read as background"
    assert prob[-10, -10] > 0.7, "bottom of gradient should read as background too"


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------

def test_handles_grayscale_input():
    gray = cv2.cvtColor(make_poster(), cv2.COLOR_RGB2GRAY)
    out = Prism(Config()).decompose(gray, source_name="gray")
    assert len(out.layers) >= 1


def test_handles_uniform_image():
    """A featureless image must not crash; it degrades to a background layer."""
    flat = np.full((240, 240, 3), 128, np.uint8)
    out = Prism(Config()).decompose(flat, source_name="flat")
    assert len(out.layers) >= 1
    assert out.by_kind(KIND_BACKGROUND)


def test_handles_tiny_image():
    tiny = np.full((32, 32, 3), 200, np.uint8)
    cv2.circle(tiny, (16, 16), 7, (20, 20, 20), -1)
    out = Prism(Config()).decompose(tiny, source_name="tiny")
    assert out.width == 32 and out.height == 32


def test_downscales_large_input_but_exports_full_resolution():
    """Segmentation runs bounded; masks must come back at source resolution."""
    big = cv2.resize(make_poster(), (2200, 2933), interpolation=cv2.INTER_CUBIC)
    cfg = Config()
    cfg.max_working_dim = 700
    out = Prism(cfg).decompose(big, source_name="big")
    assert out.width == 2200 and out.height == 2933
    for layer in out.layers:
        assert layer.mask.shape == (2933, 2200)


def test_disabled_stages_are_respected():
    cfg = Config()
    cfg.enable_text = False
    cfg.enable_graphics = False
    out = Prism(cfg).decompose(make_poster(), source_name="nostages")
    assert not out.by_kind("text")
    assert not out.by_kind("graphic")


def test_determinism():
    """Same input, same config -> same layer count. No unseeded randomness."""
    a = Prism(Config()).decompose(make_poster(), source_name="a")
    b = Prism(Config()).decompose(make_poster(), source_name="b")
    assert len(a.layers) == len(b.layers)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def test_export_writes_expected_artifacts(result):
    src = make_poster()
    with tempfile.TemporaryDirectory() as tmp:
        manifest = export_all(result, src, tmp, Config(), make_zip=True)

        assert os.path.isfile(os.path.join(tmp, "manifest.json"))
        assert os.path.isfile(os.path.join(tmp, "recomposite.png"))
        assert os.path.isfile(os.path.join(tmp, "source.png"))
        assert os.path.isdir(os.path.join(tmp, "layers"))

        pngs = os.listdir(os.path.join(tmp, "layers"))
        assert len(pngs) == len(result.layers)

        assert "quality" in manifest
        assert manifest["quality"]["psnr_db"] > 0

        if "zip" in manifest:
            assert os.path.isfile(os.path.join(tmp, manifest["zip"]))


def test_exported_layers_carry_offsets(result):
    src = make_poster()
    with tempfile.TemporaryDirectory() as tmp:
        manifest = export_all(result, src, tmp, Config(), make_zip=False)
        for entry in manifest["layers"]:
            assert "offset" in entry["meta"], "viewer needs offsets to place planes"
            assert "size" in entry["meta"]
