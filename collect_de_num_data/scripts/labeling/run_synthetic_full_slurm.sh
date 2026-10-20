#!/bin/bash
#SBATCH --job-name=de-num-synthetic
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Submit from collect_de_num_data with:
# sbatch scripts/labeling/run_synthetic_full_slurm.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${HOME}/.local/share/swiss-german-number-preparation}"
export DATA_ROOT
LOG_DIR="${DATA_ROOT}/logs/collect_de_num_data"
mkdir -p "${LOG_DIR}"
exec >"${LOG_DIR}/synthetic_full_flash_lite_${SLURM_JOB_ID}.log" 2>&1
cd "$PROJECT_ROOT"

"${REPO_ROOT}/.venv/bin/python" -u scripts/labeling/generate_synthetic_labeled.py \
  --preset full \
  --run-name synthetic_full_flash_lite \
  --resume \
  --batch-size 20
