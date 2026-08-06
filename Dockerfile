# Prism - reproducible container for local runs or cloud deployment.
#
#   docker build -t prism .
#   docker run -p 7860:7860 prism
#
# Model weights for the optional neural backends download on first use and are
# cached in /root/.u2net. If the container has no network access the pipeline
# still runs end to end on the classical path.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    PRISM_WORKERS=1 \
    LAYERFORGE_CLASSICAL=1 \
    PRISM_MAX_WORKING_DIM=1000
# Safe-by-default: most free/small hosting tiers (Render free, HF free CPU,
# a bare `docker run`) have well under 1 GB of RAM and a fraction of a CPU
# core, so two things are tuned down here rather than left at the library's
# full-resources defaults:
#   - LAYERFORGE_CLASSICAL=1  rembg's ONNX runtime plus its ~170 MB of
#     weights can OOM-kill a 512 MB container the moment a real image is
#     uploaded, with no Python traceback since the kernel just SIGKILLs it.
#   - PRISM_MAX_WORKING_DIM=1000  segmentation cost scales with pixel count,
#     and on a CPU-throttled instance (Render free is 0.1 vCPU) the library
#     default of 1400px can turn a ~10s job into a multi-minute one - long
#     enough that the platform's health check can time out mid-job and
#     restart the container, silently killing the in-memory job.
# Both trade a little quality for reliability on tiny hosts. Anything with
# more headroom (docker-compose, a paid instance) opts back into the fuller
# defaults explicitly at the platform level.

# libgl/libglib: OpenCV runtime. tesseract: text recognition.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/api/health')" || exit 1

CMD ["python", "app.py"]
