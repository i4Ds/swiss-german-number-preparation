#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
ENV_DIR="${REPO_DIR}/.venv"
DATA_ROOT="${DATA_ROOT:-${HOME}/.local/share/swiss-german-number-preparation}"
CACHE_DIR="${DATA_ROOT}/intermediate/collect_de_num_data/cache"

if [[ -z "${SLURM_JOB_ID:-}" && "${TTS_ALLOW_LOGIN_NODE:-0}" != "1" ]]; then
  echo "Refusing dependency installation outside a Slurm allocation." >&2
  exit 2
fi

mkdir -p "${CACHE_DIR}/uv" "${CACHE_DIR}/pip"
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"

python3 -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip uv
"${ENV_DIR}/bin/python" -m uv pip install \
  --python "${ENV_DIR}/bin/python" \
  --torch-backend=cu128 \
  -e "${REPO_DIR}[inference]"

echo ""
echo "Done. Activate with: source ${ENV_DIR}/bin/activate"
