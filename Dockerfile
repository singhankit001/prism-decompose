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
    LAYERFORGE_CLASSICAL=1
# Safe-by-default: most free/small hosting tiers (Render free, HF free CPU,
# a bare `docker run`) have well under 1 GB of RAM, and rembg's ONNX runtime
# plus its ~170 MB of weights can OOM-kill a 512 MB container the moment a
# real image is uploaded, with no Python traceback since the kernel just
# SIGKILLs the process. Defaulting to the weights-free classical path here
# means a plain `docker build && run` is safe out of the box. Anything with
# more RAM to spare (docker-compose, a paid instance) opts back into the
# neural path explicitly by overriding these two at the platform level.

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
