#!/usr/bin/env bash
#
# Prism launcher.
#
#   ./run.sh              set up if needed, then start the server
#   ./run.sh --full       also install the optional neural backends
#   ./run.sh --port 8000  run on a different port
#   ./run.sh --no-open    do not open a browser
#
# Safe to re-run: setup steps are skipped once satisfied.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PORT=7860
OPEN_BROWSER=1
FULL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)     FULL=1; shift ;;
    --no-open)  OPEN_BROWSER=0; shift ;;
    --port)     PORT="${2:-7860}"; shift 2 ;;
    -h|--help)
      sed -n '3,11p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

c_reset=$'\033[0m'; c_dim=$'\033[2m'; c_cyan=$'\033[36m'
c_green=$'\033[32m'; c_yellow=$'\033[33m'; c_bold=$'\033[1m'

say()  { printf '%s\n' "${c_cyan}▸${c_reset} $*"; }
ok()   { printf '%s\n' "${c_green}✓${c_reset} $*"; }
warn() { printf '%s\n' "${c_yellow}!${c_reset} $*"; }

printf '\n%s\n' "${c_bold}Prism${c_reset} ${c_dim}— split any image into its parts${c_reset}"
printf '%s\n\n' "${c_dim}────────────────────────────────────────${c_reset}"

# ---------------------------------------------------------------- python ----
PY=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")
    major=${ver%%.*}; minor=${ver##*.}
    if [[ "$major" -eq 3 && "$minor" -ge 9 ]]; then PY="$candidate"; break; fi
  fi
done

if [[ -z "$PY" ]]; then
  warn "Python 3.9+ not found."
  echo "  Install it from https://www.python.org/downloads/ or: brew install python@3.11"
  exit 1
fi
ok "Python $("$PY" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"

# --------------------------------------------------------------- venv ------
if [[ ! -d .venv ]]; then
  say "Creating virtual environment (.venv)…"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "Virtual environment active"

# ------------------------------------------------------------ packages -----
STAMP=".venv/.deps-installed"
if [[ ! -f "$STAMP" || requirements.txt -nt "$STAMP" ]]; then
  say "Installing dependencies (first run takes a few minutes)…"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
  touch "$STAMP"
  ok "Dependencies installed"
else
  ok "Dependencies already installed"
fi

if [[ "$FULL" -eq 1 ]]; then
  FULL_STAMP=".venv/.deps-full"
  if [[ ! -f "$FULL_STAMP" ]]; then
    say "Installing optional neural backends (easyocr pulls in torch — this is large)…"
    python -m pip install --quiet easyocr simple-lama-inpainting || \
      warn "Optional backends failed to install; the pipeline still runs without them."
    touch "$FULL_STAMP"
  fi
  ok "Optional backends present"
fi

# ------------------------------------------------------------ tesseract ----
# Ask the library itself rather than checking PATH: Homebrew's bin directory is
# frequently absent from a non-login shell's PATH, which made this report
# "not found" on machines where Tesseract was installed and working.
python - <<'PY' 2>/dev/null || true
try:
    from prism import backends
    path = backends.tesseract_path()
    if path:
        print(f"\033[32m✓\033[0m Tesseract \033[2m{path}\033[0m")
    else:
        print("\033[33m!\033[0m Tesseract not found — text layers are still detected "
              "and cut out,\n  but the recognised string will be empty. To enable it:")
        import platform
        s = platform.system()
        print("    brew install tesseract" if s == "Darwin"
              else "    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng" if s == "Linux"
              else "    winget install UB-Mannheim.TesseractOCR")
except BaseException as exc:
    print(f"\033[33m!\033[0m Tesseract probe skipped ({type(exc).__name__})")
PY

# --------------------------------------------------------------- backends --
# Advisory only. Trailing `|| true` because probing an optional dependency must
# never abort startup under `set -e` — some packages terminate the interpreter
# on import when installed without their runtime extra.
python - <<'PY' 2>/dev/null || true
try:
    from prism import backends
    b = backends.describe()
    neural = b["subject"] != "saliency+grabcut"
    tag = "neural" if neural else "classical (no weights)"
    print(f"\033[32m✓\033[0m Backends: subject=\033[1m{b['subject']}\033[0m, "
          f"text=\033[1m{b['text_detection']}\033[0m  \033[2m[{tag}]\033[0m")
    if neural:
        print("\033[2m  First decomposition downloads ~170 MB of weights, once. "
              "Later runs are fast.\033[0m")
    else:
        print("\033[2m  Running the weights-free classical path.\033[0m")
except BaseException as exc:
    print(f"\033[33m!\033[0m Backend probe skipped ({type(exc).__name__}); "
          f"pipeline still runs on the classical path.")
PY

# ------------------------------------------------------------------ port ---
if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  warn "Port $PORT is already in use."
  for alt in 7861 7862 7863 8000 8080; do
    if ! lsof -iTCP:"$alt" -sTCP:LISTEN >/dev/null 2>&1; then
      PORT="$alt"; say "Using port $PORT instead."; break
    fi
  done
fi

URL="http://localhost:${PORT}"

printf '\n%s\n' "${c_dim}────────────────────────────────────────${c_reset}"
printf '  %s\n' "${c_bold}${c_cyan}${URL}${c_reset}"
printf '  %s\n' "${c_dim}drop an image in the browser · Ctrl+C to stop${c_reset}"
printf '%s\n\n' "${c_dim}────────────────────────────────────────${c_reset}"

# Open the browser once the server actually answers, so the first paint is the app.
if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  (
    for _ in $(seq 1 60); do
      if curl -sf -o /dev/null "${URL}/api/health" 2>/dev/null; then
        case "$(uname -s)" in
          Darwin) open "$URL" ;;
          Linux)  command -v xdg-open >/dev/null && xdg-open "$URL" >/dev/null 2>&1 ;;
          *)      command -v start >/dev/null && start "$URL" ;;
        esac
        break
      fi
      sleep 1
    done
  ) &
fi

export PORT
exec python app.py
