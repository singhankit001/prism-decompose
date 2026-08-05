#!/usr/bin/env python3
"""LayerForge web service.

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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

from layerforge import Config, LayerForge, __version__
from layerforge import backends
from layerforge.exporters import export_all
from layerforge.pipeline import STAGES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("layerforge.app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
JOB_ROOT = os.path.join(tempfile.gettempdir(), "layerforge-jobs")
os.makedirs(JOB_ROOT, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.environ.get("LAYERFORGE_MAX_UPLOAD", 25 * 1024 * 1024))
MAX_WORKERS = int(os.environ.get("LAYERFORGE_WORKERS", 2))
JOB_TTL_SECONDS = int(os.environ.get("LAYERFORGE_JOB_TTL", 3600))
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

def _run_job(job: Job, image_rgb: np.ndarray) -> None:
    def publish(stage: str, frac: float, message: str) -> None:
        job.stage = stage
        job.progress = frac
        job.message = message
        job.version += 1
        job.events.append({"stage": stage, "progress": frac, "message": message})

    try:
        job.status = "running"
        publish("start", 0.02, "Preparing image")

        cfg = Config()
        engine = LayerForge(cfg)
        result = engine.decompose(image_rgb, source_name=os.path.splitext(job.filename)[0],
                                  progress=publish)

        publish("export", 0.94, "Writing assets")
        os.makedirs(job.dir, exist_ok=True)
        manifest = export_all(result, image_rgb, job.dir, cfg, make_zip=True)

        job.manifest = manifest
        job.status = "done"
        publish("done", 1.0, "Complete")
        log.info("job %s finished: %d layers", job.id, manifest.get("layer_count", 0))
    except Exception as exc:  # pragma: no cover
        log.exception("job %s failed", job.id)
        job.status = "error"
        job.error = str(exc)
        job.version += 1
        job.events.append({"stage": "error", "progress": 1.0, "message": str(exc)})


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

app = FastAPI(title="LayerForge", version=__version__,
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
async def decompose(file: UploadFile = File(...)) -> JSONResponse:
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

    POOL.submit(_run_job, job, image_rgb)
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
        return HTMLResponse("<h1>LayerForge</h1><p>Frontend not found.</p>", 500)
    with open(path, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
