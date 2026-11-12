#!/usr/bin/env bash
# After cloning: ./start.sh
# An NVIDIA GPU installs the CUDA explanation stack and the app uses it.
# Otherwise the script installs the local explanation package.
# Override with NOTICE_BACKEND=cuda|mlx|openai.

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

log() {
  printf '[credit-auditor] %s\n' "$*"
}

if ! command -v python3 >/dev/null 2>&1; then
  log "python3 is not on PATH. Install Python 3.11, then run ./start.sh again."
  exit 1
fi

log "Python: $(python3 -V) at $(command -v python3)"
log "checkout: $(pwd)"

if [[ ! -x .venv/bin/python ]]; then
  log "creating .venv"
  python3 -m venv .venv
else
  log "reusing .venv ($(.venv/bin/python -V))"
fi

log "installing the shared packages from requirements.txt"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L; then
  log "NVIDIA GPU found. Installing the CUDA PyTorch wheel and the 4-bit explanation packages."
  .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
  .venv/bin/python -m pip install -r requirements-cuda.txt
elif [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  log "no NVIDIA GPU. Installing the local explanation package."
  log "installing mlx-lm from requirements-mlx.txt"
  .venv/bin/python -m pip install -r requirements-mlx.txt
else
  log "no NVIDIA GPU."
  log "the credit score and SHAP chart still run. Drafts need a CUDA GPU or NOTICE_BACKEND=openai."
fi

if ! .venv/bin/python -c "import xgboost" >/dev/null 2>&1; then
  log "XGBoost failed to import."
  if command -v brew >/dev/null 2>&1; then
    log "installing libomp"
    HOMEBREW_NO_AUTO_UPDATE=1 brew install libomp
  else
    log "install libomp, then run ./start.sh again."
    .venv/bin/python -c "import xgboost"
  fi
fi

log "preparing the saved credit model, the policy index, and the explanation weights"
.venv/bin/python - << 'PY'
from auditor.pipeline import build_context
from auditor.rag import open_store
from auditor.notice import preload

build_context()
open_store()
preload()
PY

log "opening the app at http://127.0.0.1:8501"
exec .venv/bin/streamlit run app.py --server.headless true --server.port 8501
