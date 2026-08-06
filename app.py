#!/usr/bin/env python3
"""Prism web service.

FastAPI backend for the 3D layer explorer:

    POST /api/decompose      upload an image, returns a job id immediately
    GET  /api/jobs/{id}/events   server-sent events with live stage progress
    GET  /api/jobs/{id}         final manifest once the job completes
    GET  /api/jobs/{id}/layers/...   individual layer PNGs
    GET  /api/jobs/{id}/download     the full ZIP bundle
    GET  /api/health         backend availability report

Jobs run on a worker thread pool so uploads never block the event loop, and
progress is streamed rather than polled so the UI can animate each stage as it
happens.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

from prism import Config, Prism, __version__
from prism import backends
from prism.exporters import build_downloads, export_all
from prism.pipeline import STAGES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("prism.app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
JOB_ROOT = os.path.join(tempfile.gettempdir(), "prism-jobs")
os.makedirs(JOB_ROOT, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.environ.get("PRISM_MAX_UPLOAD", 25 * 1024 * 1024))
MAX_WORKERS = int(os.environ.get("PRISM_WORKERS", 2))
JOB_TTL_SECONDS = int(os.environ.get("PRISM_JOB_TTL", 3600))
# Segmentation work scales with the working resolution; on a CPU-throttled
# free-tier instance (Render free is 0.1 vCPU) the library default of 1400px
# can turn a ~10s job into a multi-minute one, long enough that the platform's
# health check can time out mid-job and restart the container - which kills
# the in-memory job and looks like the UI is stuck. Unset to keep the
# library default.
MAX_WORKING_DIM = os.environ.get("PRISM_MAX_WORKING_DIM")
# Fast mode trades the two most expensive optional refinements for latency:
# OCR transcription (a Tesseract subprocess per text block) and half the
# GrabCut iterations. Measured across all three sample posters this halves
# pipeline time while producing an identical layer count, identical text
# block count and PSNR within 0.01 dB - text layers are still detected, cut
# out and stacked correctly, they just carry no recognised string. Worth it
# on a throttled instance; off by default for local/library use.
FAST_MODE = os.environ.get("PRISM_FAST", "").lower() in {"1", "true", "yes"}
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp",
                 "image/bmp", "image/tiff"}


# ---------------------------------------------------------------------------
# job state
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"          # queued | running | done | error
    stage: str = ""
    message: str = "Queued"
    progress: float = 0.0
    error: Optional[str] = None
    manifest: Optional[dict] = None
    created: float = field(default_factory=time.time)
    events: List[dict] = field(default_factory=list)
    version: int = 0
    # Retained so the deferred PSD/ZIP build can run without re-decomposing.
    decomp: Optional[object] = None
    source_rgb: Optional[np.ndarray] = None
    cfg: Optional[Config] = None

    @property
    def dir(self) -> str:
        return os.path.join(JOB_ROOT, self.id)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "progress": round(self.progress, 4),
            "error": self.error,
        }


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="forge")


def _purge_expired() -> None:
    """Drop jobs older than the TTL so disk does not grow without bound."""
    now = time.time()
    with JOBS_LOCK:
        stale = [j for j in JOBS.values() if now - j.created > JOB_TTL_SECONDS]
        for job in stale:
            JOBS.pop(job.id, None)
            shutil.rmtree(job.dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _run_job(job: Job, image_rgb: np.ndarray, merge_text: bool = False) -> None:
    t_start = time.time()

    def publish(stage: str, frac: float, message: str) -> None:
        job.stage = stage
        job.progress = frac
        job.message = message
        job.version += 1
        job.events.append({"stage": stage, "progress": frac, "message": message})
        # Without this, a slow-but-alive job and a silently crashed one look
        # identical from outside (both just stop advancing in the UI) - there
        # was no way to tell from the platform logs which one was happening,
        # or which stage was actually the slow one on real hosting hardware.
        log.info("job %s +%5.1fs stage=%-10s frac=%.2f %s",
                 job.id, time.time() - t_start, stage, frac, message)

    try:
        job.status = "running"
        publish("start", 0.02, "Preparing image")

        cfg = Config()
        cfg.merge_text_layers = merge_text
        if MAX_WORKING_DIM:
            cfg.max_working_dim = int(MAX_WORKING_DIM)
        if FAST_MODE:
            cfg.text_recognise = False
            cfg.grabcut_iters = min(cfg.grabcut_iters, 2)
        engine = Prism(cfg)
        result = engine.decompose(image_rgb, source_name=os.path.splitext(job.filename)[0],
                                  progress=publish)

        publish("export", 0.94, "Writing assets")
        os.makedirs(job.dir, exist_ok=True)
        # PSD and ZIP are deliberately not built here. The viewer only needs
        # the layer PNGs and the manifest, so building artefacts nobody has
        # asked for yet just adds dead time to every single job. They are
        # produced on first download instead - see job_download.
        manifest = export_all(result, image_rgb, job.dir, cfg,
                              make_zip=False, make_psd=False)
        job.decomp = result
        job.source_rgb = image_rgb
        job.cfg = cfg

        job.manifest = manifest
        job.status = "done"
        publish("done", 1.0, "Complete")
        log.info("job %s finished in %.1fs: %d layers, timings=%s",
                 job.id, time.time() - t_start, manifest.get("layer_count", 0),
                 result.timings)
    except Exception as exc:  # pragma: no cover
        log.exception("job %s failed", job.id)
        job.status = "error"
        job.error = str(exc)
        job.version += 1
        job.events.append({"stage": "error", "progress": 1.0, "message": str(exc)})


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

app = FastAPI(title="Prism", version=__version__,
              description="Decompose an image into meaningful, reusable layers.")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "version": __version__,
        "backends": backends.describe(),
        "stages": [{"key": k, "label": v} for k, v in STAGES],
        "max_upload_mb": round(MAX_UPLOAD_BYTES / (1024 * 1024), 1),
        "workers": MAX_WORKERS,
    })


@app.post("/api/decompose")
async def decompose(file: UploadFile = File(...),
                    merge_text: bool = Form(False)) -> JSONResponse:
    _purge_expired()

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413,
                            f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        log.warning("unusual content-type %s; attempting decode anyway",
                    file.content_type)

    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Could not decode image. Supported: PNG, JPEG, WebP, BMP, TIFF.")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    job = Job(id=uuid.uuid4().hex[:12],
              filename=os.path.basename(file.filename or "upload.png"))
    with JOBS_LOCK:
        JOBS[job.id] = job
    os.makedirs(job.dir, exist_ok=True)

    POOL.submit(_run_job, job, image_rgb, merge_text)
    return JSONResponse({"job": job.id, **job.snapshot()}, status_code=202)


def _get_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    job = _get_job(job_id)
    payload = job.snapshot()
    if job.status == "done":
        payload["manifest"] = job.manifest
    return JSONResponse(payload)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """Server-sent events: one message per stage transition, then the result."""
    job = _get_job(job_id)

    async def stream():
        sent = 0
        deadline = time.time() + 600
        while time.time() < deadline:
            while sent < len(job.events):
                event = job.events[sent]
                sent += 1
                yield f"data: {json.dumps({**job.snapshot(), **event})}\n\n"
            if job.status in ("done", "error"):
                final = job.snapshot()
                if job.status == "done":
                    final["manifest"] = job.manifest
                yield f"event: complete\ndata: {json.dumps(final)}\n\n"
                return
            await asyncio.sleep(0.15)
        yield "event: complete\ndata: {\"status\":\"error\",\"error\":\"timeout\"}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/file/{path:path}")
async def job_file(job_id: str, path: str) -> FileResponse:
    job = _get_job(job_id)
    # Contain path traversal: the resolved path must stay inside the job dir.
    target = os.path.abspath(os.path.join(job.dir, path))
    if not target.startswith(os.path.abspath(job.dir)) or not os.path.isfile(target):
        raise HTTPException(404, "Not found")
    return FileResponse(target)


@app.get("/api/jobs/{job_id}/download")
async def job_download(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job.status != "done" or not job.manifest:
        raise HTTPException(409, "Job not finished")

    # The PSD and ZIP are built here rather than during the job, so their cost
    # is paid only by users who actually want them. Idempotent and cached on
    # disk, so repeat clicks are free.
    if not job.manifest.get("zip"):
        if job.decomp is None or job.source_rgb is None:
            raise HTTPException(410, "Job data has expired")
        try:
            merged = await asyncio.get_running_loop().run_in_executor(
                POOL, build_downloads, job.decomp, job.source_rgb, job.dir,
                job.cfg or Config())
            job.manifest.update(merged)
        except Exception as exc:
            log.exception("bundle build failed for job %s", job.id)
            raise HTTPException(500, f"Could not build bundle: {exc}")

    name = job.manifest.get("zip")
    if not name:
        raise HTTPException(404, "Bundle not available")
    path = os.path.join(job.dir, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "Bundle not available")
    return FileResponse(path, media_type="application/zip", filename=name)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(path):
        return HTMLResponse("<h1>Prism</h1><p>Frontend not found.</p>", 500)
    with open(path, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
