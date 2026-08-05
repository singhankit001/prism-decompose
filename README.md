# LayerForge

**Decompose any flat image into meaningful, reusable design layers — and explore the result in 3D.**

Give LayerForge a poster, advert or social creative. It returns the pieces a designer would have started with: a clean background plate, matted subject cut-outs, isolated graphic elements, and separated text blocks — each with an alpha channel, a bounding box, a depth index and a confidence score.

Output ships three ways: **PNG assets + JSON manifest**, a **layered `.psd`** that opens in Photoshop, and a **live 3D viewer** where the layer stack is exploded along the Z axis so you can orbit through the decomposition.

---

## Quick start

```bash
git clone <your-repo-url> && cd layerforge
pip install -r requirements.txt
python app.py
```

Open **http://localhost:7860** and drop in an image.

Tesseract powers text recognition and is the one non-pip dependency:

```bash
# macOS
brew install tesseract
# Debian / Ubuntu
sudo apt-get install tesseract-ocr tesseract-ocr-eng
# Windows
winget install UB-Mannheim.TesseractOCR
```

Or skip local setup entirely:

```bash
docker build -t layerforge .
docker run -p 7860:7860 layerforge
```

### Command line

```bash
python cli.py poster.png -o out/           # one image
python cli.py ./images -o out/ --workers 4 # a directory, in parallel
python cli.py poster.png --classical       # force the no-weights path
```

### As a library

```python
from layerforge import LayerForge, Config
from layerforge.exporters import export_all

result = LayerForge().decompose(rgb_array, source_name="poster")
export_all(result, rgb_array, "out/")

for layer in result.sorted_layers():
    print(layer.z, layer.kind, layer.label, layer.bbox)
```

---

## How it works

Six stages, each independently testable, each with a graceful fallback.

**1 · Backdrop modelling.** Designed creatives sit on a designed backdrop. That backdrop is learned *before* any segmentation and every later stage uses it as a prior. Border pixels are clustered in CIE-LAB (perceptually uniform, so one distance threshold behaves the same across hues); sparse clusters are discarded, which is what stops bunting, frames and ornaments that touch the edge from being mistaken for background. A quadratic colour surface is then fitted over the confident background pixels, so vignettes and duotone gradients are followed instead of being torn out as foreground.

**2 · Text detection.** Text is what designers most want back as a separate asset and what generic segmentation handles worst — a headline is not a salient object, and its glyphs are disconnected components. With `easyocr` installed, CRAFT does the detection. Without it, a three-tier bottom-up analysis runs: glyph candidates from adaptive thresholding *and* quantised colour planes (an orange word inside a dark-green headline has almost no luminance contrast, so luminance alone misses it), filtered by glyph geometry and stroke-width consistency; then lines formed by union-find over vertical overlap, height similarity and horizontal proximity; then blocks formed from aligned, stacked lines. Blocks matter — a headline set over four lines is one asset, not four.

**3 · Subject matting.** `rembg` (U²-Net / IS-Net) when weights are available. Otherwise spectral-residual and fine-grained saliency are fused with the backdrop prior and a centre bias, then refined with GrabCut. The classical path walks its threshold upward until total subject coverage respects a cap, so a busy composition cannot end up as one layer swallowing the canvas.

**4 · Graphic classification.** Whatever survives after text and subjects is design furniture: logos, icon rows, badges, frames, rules, ornaments. No off-the-shelf model segments these, so the method is structural. Residual pixels are grouped into elements by proximity, then each element is scored by a **photographic-vs-vector discriminator** built from three signals — colour diversity, *interior* gradient activity (flat fills are near-zero inside a shape even when their edges are hard), and local variance energy. Photographic residuals get promoted to secondary subject layers; flat ones are labelled by transparent, auditable geometry heuristics. This discriminator is what keeps a logo out of the "person" layer and skin texture out of the "graphic" layer.

**5 · Background reconstruction.** Removing the foreground leaves holes, and the background is the one layer that must cover the whole canvas. LaMa when installed; otherwise a structural fill (iterative coarse diffusion at 1/4 scale, which recovers gradients almost exactly on flat and gradient backdrops) blended with Telea by depth into the hole — Telea near known pixels for detail, structural deep inside where diffusion would smear.

**6 · Depth ordering.** Z-order comes from layer kind, then occlusion evidence where two layers contest pixels, then area. Finally, front-to-back subtraction removes pixels a nearer layer already owns, guaranteeing a clean partition — this is why the stack recomposites cleanly and why the 3D view shows no ghost duplicates.

---

## Design decisions worth calling out

**Every neural backend is optional.** The pipeline runs end to end with zero downloaded weights. `layerforge/backends.py` probes each capability at import and the pipeline transparently upgrades when one is present — same code path, same output contract, no runtime surprises. The manifest and the UI both report which backend actually served each stage, so results are never ambiguous about what ran.

This is a deliberate robustness property, not a workaround: the system works on an air-gapped machine, in a locked-down CI runner, and on a laptop with no GPU, and gets better — not different — when you install the extras.

**Correctness is measured, not asserted.** Every run recomposites its own layer stack and reports MAE and PSNR against the source. If a decomposition is wrong, the number says so. That metric is in the manifest, in the CLI output and in the UI header.

**Bounded working resolution.** Segmentation runs at a capped resolution for predictable latency; masks are resampled to full resolution before export, so output fidelity is set by the source image, not by the working size. A 2200×2933 input is processed as fast as a 1400px one and still exports at full size.

---

## The 3D viewer

Each layer becomes a textured plane positioned from its manifest bounding box, so the stack reproduces the original composition exactly — then separates along Z by its computed depth.

Orbit and zoom with the mouse. **Explode** sets the depth spacing; **Spread** fans the layers into an arc so dense stacks stay legible. Hovering isolates a layer and dims the rest; clicking focuses the camera on it. **Flatten** snaps back to a head-on view of the reassembled composition. Rendering is Three.js with unlit planes (source pixels shown faithfully rather than re-shaded), bloom post-processing, exponential fog and a drifting particle field.

Pipeline progress streams over server-sent events, so each stage lights up as it actually completes rather than animating a fake progress bar. If SSE is blocked by a proxy, the client falls back to polling automatically.

---

## Output

```
out/poster/
├── layers/
│   ├── 00-background-background-plate.png
│   ├── 01-subject-subject.png
│   ├── 06-graphic-logo-mark.png
│   └── 09-text-choose-your-energy.png
├── manifest.json        kind, label, depth, bbox, colour, text, confidence
├── poster.psd           layered, correctly stacked, opens in Photoshop
├── recomposite.png      the stack flattened — compare against source
├── source.png
└── poster-layers.zip
```

Layers are cropped to their bounding box with the offset recorded in the manifest, so files stay small and nothing is lost. The PSD stores layers cropped with offsets too, exactly as Photoshop expects — that alone took the sample PSD from 63 MB to 6.5 MB.

```json
{
  "kind": "text",
  "label": "text \"Choose your energy, every day.\"",
  "z": 13,
  "confidence": 0.83,
  "bbox": [104, 402, 618, 811],
  "color": [58, 74, 52],
  "text": "Choose your energy, every day.",
  "file": "layers/13-text-choose-your-energy.png",
  "meta": { "offset": [104, 402], "size": [514, 409] }
}
```

---

## Results

Recomposite PSNR against the source, measured by the pipeline itself. Both tables were produced on the **classical path with no model weights available** — the numbers are a floor, not a ceiling.

Provided samples:

| image | layers | PSNR |
|---|---|---|
| festival (dense collage) | 7 | 36.8 dB |
| jewels (dark, ornate) | 19 | 30.9 dB |
| wellness (light, editorial) | 20 | 32.6 dB |
| reference decomposition | 24 | 35.8 dB |

Held-out images the pipeline had never seen, chosen to be structurally unlike the samples, including deliberately pathological cases:

| image | size | layers | PSNR | result |
|---|---|---|---|---|
| dark tech poster | 1600×900 | 9 | 45.4 dB | text, subject and graphics all separated |
| light minimal | 1000×1000 | 7 | 55.0 dB | 3 text blocks, product isolated |
| tall story format | 1080×1920 | 6 | 59.3 dB | photo-heavy, no text present |
| wide strip banner | 2400×80 | 2 | 54.3 dB | 30:1 aspect, no crash |
| pure noise | 500×500 | 4 | 40.1 dB | degrades sensibly |
| pure white | 400×400 | 1 | 99.0 dB | correctly emits background only |

Test suite: **21 tests, passing in both auto and forced-classical mode.**

```bash
python -m pytest tests/ -v
LAYERFORGE_CLASSICAL=1 python -m pytest tests/ -v   # prove the no-weights path
```

Tests use synthetic fixtures with known ground-truth structure rather than the sample images, so they verify behaviour instead of memorised output. They cover the layer contract (dense unique z-order, exactly one full-canvas background, non-overlapping content layers), correctness (recomposite PSNR, the photo-vs-vector discriminator, gradient backdrop modelling), robustness (grayscale, uniform, 32×32, 2200×2933 inputs, determinism) and export integrity.

---

## Honest limitations

**Dense collage artwork is the hard case for the classical path.** In the festival poster the dancers, drums, speakers and mandala touch each other, forming one connected foreground region that classical segmentation cannot split — it is labelled `decor / backdrop artwork` rather than being carved into separate subjects. Installing `rembg` fixes this, which is exactly why the neural path exists. This is a limitation of the fallback, not of the design.

**Text recognition is weaker than text detection.** Stylised display and script faces are reliably *located* and cut out correctly, but Tesseract often misreads them (`"Radiate your essence"` → `"nce"`). The layer geometry is right even when the transcribed string is not; the recognised text is metadata, never a gate on whether a layer is emitted. Installing `easyocr` improves both.

**Layer semantics are heuristic, not learned.** Graphic labels come from transparent geometry rules (aspect, fill, position, size). They are auditable and need no weights, but they are not a substitute for a zero-shot classifier. `stages/graphics.py::_describe` is a drop-in slot for CLIP without changing the layer contract.

**The viewer needs a network on first load.** Three.js is loaded from unpkg via importmap. Vendor it locally for a fully offline deployment.

---

## Project layout

```
layerforge/
├── config.py           every threshold, in one place
├── backends.py         optional-dependency probing, graceful degradation
├── types.py            Layer / Decomposition contracts
├── imaging.py          guided filter, texture metric, mask algebra
├── pipeline.py         stage orchestration + progress streaming
├── exporters.py        PNG, manifest, PSD, ZIP, quality metrics
└── stages/
    ├── backdrop.py     background colour + gradient model
    ├── text.py         glyph → line → block analysis
    ├── subject.py      salient matting, neural or classical
    ├── graphics.py     photo-vs-vector discrimination
    ├── reconstruct.py  three-tier inpainting
    └── ordering.py     z-order + occlusion partitioning
app.py                  FastAPI service, SSE progress
cli.py                  batch CLI
static/index.html       3D viewer
tests/                  21 tests
```

## API

| method | route | purpose |
|---|---|---|
| `POST` | `/api/decompose` | upload an image, returns a job id immediately |
| `GET` | `/api/jobs/{id}/events` | SSE stream of live stage progress |
| `GET` | `/api/jobs/{id}` | status, and the manifest once complete |
| `GET` | `/api/jobs/{id}/file/{path}` | an individual layer PNG |
| `GET` | `/api/jobs/{id}/download` | the full ZIP bundle |
| `GET` | `/api/health` | version and active backend report |

Jobs run on a worker thread pool so uploads never block the event loop. Uploads are capped at 25 MB, job directories are TTL-purged, and layer paths are contained against traversal.

Configure with `PORT`, `LAYERFORGE_WORKERS`, `LAYERFORGE_MAX_UPLOAD`, `LAYERFORGE_JOB_TTL`, `LAYERFORGE_CLASSICAL`.

---

Built by **Ankit Singh**.
