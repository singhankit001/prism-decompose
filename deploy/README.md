# Deploying Prism

Prism is a single FastAPI process serving a static frontend, so anything that runs a container or a Python web process will host it. Pick whichever is least friction for you.

Every target needs the same two things: **Python 3.9+ with the requirements installed**, and the **`tesseract-ocr` system package** if you want recognised text strings (layers are still detected and cut out without it).

---

## Hugging Face Spaces — free, recommended for a demo

Best fit: free GPU-less tier is enough, the Docker SDK runs our `Dockerfile` as-is, and model weights cache between restarts.

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space) → SDK: **Docker** → Blank.
2. Push this repository to the Space remote.
3. Prepend the YAML block below to the top of `README.md` **in the Space repo only** — Spaces requires it for configuration, and it would look like stray text on GitHub.

```yaml
---
title: Prism
emoji: 🔻
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
```

```bash
git remote add space https://huggingface.co/spaces/<user>/prism
git push space main
```

First build takes a few minutes. The first decomposition then downloads ~170 MB of `rembg` weights, once.

---

## Docker — anywhere

```bash
docker compose up --build          # http://localhost:7860
```

or without compose:

```bash
docker build -t prism .
docker run -p 7860:7860 -v prism-models:/root/.u2net prism
```

The named volume persists downloaded weights, so a rebuild doesn't re-download them.

---

## Render

`render.yaml` in the repo root is a complete blueprint. Point Render at the repository and it will pick it up — no dashboard configuration needed.

Free instances sleep when idle and have 512 MB RAM, which is tight for the neural path. Set `LAYERFORGE_CLASSICAL=1` on the free tier to stay on the weights-free path, or use a paid instance for `rembg`.

---

## Fly.io

```bash
fly launch --no-deploy      # accept the existing fly.toml
fly volumes create prism_models --size 1
fly deploy
```

---

## Railway

Railway auto-detects the `Dockerfile`. Add these variables:

| Variable | Value |
|---|---|
| `PORT` | `7860` |
| `PRISM_WORKERS` | `2` |

---

## Bare VPS with systemd

```bash
sudo apt-get update && sudo apt-get install -y python3-venv libgl1 libglib2.0-0 tesseract-ocr tesseract-ocr-eng
git clone <repo> /opt/prism && cd /opt/prism
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`/etc/systemd/system/prism.service`:

```ini
[Unit]
Description=Prism image decomposition
After=network.target

[Service]
WorkingDirectory=/opt/prism
Environment=PORT=7860
ExecStart=/opt/prism/.venv/bin/python app.py
Restart=always
User=www-data

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now prism
```

Put nginx or Caddy in front for TLS. If you proxy through nginx, disable buffering on the events endpoint or server-sent progress will arrive in one lump at the end:

```nginx
location /api/jobs/ {
    proxy_pass http://127.0.0.1:7860;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 600s;
}
```

The frontend falls back to polling if SSE never arrives, so it degrades rather than breaking.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `7860` | HTTP port |
| `PRISM_WORKERS` | `2` | Concurrent decomposition threads |
| `PRISM_MAX_UPLOAD` | `26214400` | Upload cap in bytes (25 MB) |
| `PRISM_JOB_TTL` | `3600` | Seconds before job output is purged |
| `LAYERFORGE_CLASSICAL` | unset | Set to `1` to force the weights-free path |

## Sizing

| Tier | RAM | Notes |
|---|---|---|
| Classical only | 512 MB | Set `LAYERFORGE_CLASSICAL=1`; no downloads, fastest cold start |
| With `rembg` | 2 GB | Recommended. ~170 MB weights on first run |
| With `easyocr` | 4 GB | Pulls in torch; best text detection |

CPU is the bottleneck, not memory — a typical 1086×1448 poster takes 5–10 s on two cores.
