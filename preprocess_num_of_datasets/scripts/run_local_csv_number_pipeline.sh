#!/usr/bin/env bash
# Submit the local CSV number pipeline as one Slurm GPU job.

#SBATCH --job-name=local-csv-numbers
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00

set -Eeuo pipefail

# Slurm executes a staged copy of this file from /var/lib/slurm, so BASH_SOURCE
# no longer points into the repository. Locate the repository from the submit
# directory in that case; direct execution can still use the script location.
if [ -n "${PIPELINE_REPO_DIR:-}" ]; then
  REPO_DIR="$(cd "${PIPELINE_REPO_DIR}" && pwd)"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  SEARCH_DIR="$(cd "${SLURM_SUBMIT_DIR:-${PWD}}" && pwd)"
  while [ "${SEARCH_DIR}" != "/" ] && \
    [ ! -f "${SEARCH_DIR}/preprocess_num_of_datasets/scripts/run_local_csv_number_pipeline.py" ]; do
    SEARCH_DIR="$(dirname "${SEARCH_DIR}")"
  done
  if [ ! -f "${SEARCH_DIR}/preprocess_num_of_datasets/scripts/run_local_csv_number_pipeline.py" ]; then
    echo "Could not find the repository from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-${PWD}}." >&2
    echo "Submit from within the repository or export PIPELINE_REPO_DIR=/path/to/repository." >&2
    exit 1
  fi
  REPO_DIR="${SEARCH_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

SCRIPT_DIR="${REPO_DIR}/preprocess_num_of_datasets/scripts"
CONFIG_PATH="${1:-${REPO_DIR}/preprocess_num_of_datasets/configs/local_csv_number_pipeline.yaml}"
PYTHON_PATH="${PIPELINE_PYTHON:-${REPO_DIR}/.venv-audio-fhnw-cu128/bin/python}"

[ -f "${CONFIG_PATH}" ] || {
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
}
[ -x "${PYTHON_PATH}" ] || {
  echo "Python executable not found: ${PYTHON_PATH}" >&2
  exit 1
}

exec "${PYTHON_PATH}" "${SCRIPT_DIR}/run_local_csv_number_pipeline.py" \
  --config "${CONFIG_PATH}" --worker
