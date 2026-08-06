#!/usr/bin/env python3
"""Command-line interface for batch decomposition.

    python cli.py samples/wellness.png -o out/
    python cli.py samples/ -o out/ --workers 4
    python cli.py image.png --classical     # force the no-weights path
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import cv2
import numpy as np

from prism import Config, Prism
from prism.exporters import export_all

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def collect_inputs(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    found = []
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in IMAGE_EXT:
                found.append(os.path.join(root, name))
    return found


def load_rgb(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"unreadable image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def process(path: str, out_root: str, cfg: Config, quiet: bool) -> dict:
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(out_root, name)
    rgb = load_rgb(path)

    started = time.perf_counter()
    engine = Prism(cfg)

    def progress(_key: str, frac: float, message: str) -> None:
        if not quiet:
            sys.stdout.write(f"\r  [{name}] {message:<26} {frac * 100:5.1f}%")
            sys.stdout.flush()

    result = engine.decompose(rgb, source_name=name, progress=progress)
    manifest = export_all(result, rgb, out_dir, cfg)
    elapsed = time.perf_counter() - started

    if not quiet:
        quality = manifest.get("quality", {})
        sys.stdout.write(
            f"\r  [{name}] {len(result.layers)} layers in {elapsed:.1f}s "
            f"(PSNR {quality.get('psnr_db', '?')} dB){' ' * 12}\n")
    return {
        "input": path,
        "output": out_dir,
        "layers": len(result.layers),
        "seconds": round(elapsed, 2),
        "quality": manifest.get("quality"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Decompose images into meaningful layers.")
    ap.add_argument("input", help="image file or directory of images")
    ap.add_argument("-o", "--output", default="output", help="output directory")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers for batch mode")
    ap.add_argument("--max-dim", type=int, default=None,
                    help="working resolution cap for segmentation")
    ap.add_argument("--no-psd", action="store_true", help="skip PSD export")
    ap.add_argument("--no-text", action="store_true", help="skip text detection")
    ap.add_argument("--merge-text", action="store_true",
                    help="emit all copy as one combined text layer")
    ap.add_argument("--classical", action="store_true",
                    help="force the no-weights classical path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.classical:
        os.environ["PRISM_CLASSICAL"] = "1"

    cfg = Config()
    if args.max_dim:
        cfg.max_working_dim = args.max_dim
    if args.no_psd:
        cfg.export_psd = False
    if args.no_text:
        cfg.enable_text = False
    if args.merge_text:
        cfg.merge_text_layers = True

    inputs = collect_inputs(args.input)
    if not inputs:
        print(f"No images found at {args.input}", file=sys.stderr)
        return 1

    os.makedirs(args.output, exist_ok=True)
    print(f"Decomposing {len(inputs)} image(s) -> {args.output}")

    reports = []
    if args.workers > 1 and len(inputs) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process, p, args.output, cfg, True) for p in inputs]
            for fut in futures:
                reports.append(fut.result())
                r = reports[-1]
                print(f"  {os.path.basename(r['input'])}: "
                      f"{r['layers']} layers, {r['seconds']}s")
    else:
        for path in inputs:
            try:
                reports.append(process(path, args.output, cfg, args.quiet))
            except Exception as exc:
                print(f"  FAILED {path}: {exc}", file=sys.stderr)

    summary_path = os.path.join(args.output, "batch_report.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)
    print(f"Done. Report: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
