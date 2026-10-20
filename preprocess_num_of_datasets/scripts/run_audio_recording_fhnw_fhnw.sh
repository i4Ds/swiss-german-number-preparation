#!/usr/bin/env bash
# Submit and run the complete pipeline on FHNW Calculon.

#SBATCH --job-name=audio-num-fhnw
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00

set -Eeuo pipefail

PIPELINE_STAGE="initialization"
trap 'status=$?; echo "Pipeline failed: stage=${PIPELINE_STAGE}, line=${LINENO}, exit=${status}, job=${SLURM_JOB_ID:-local}" >&2; exit "$status"' ERR

if [ -n "${PIPELINE_REPO_DIR:-}" ]; then
  REPO_DIR="$(cd "${PIPELINE_REPO_DIR}" && pwd)"
  PROJECT_DIR="${REPO_DIR}/preprocess_num_of_datasets"
  SCRIPT_DIR="${PROJECT_DIR}/scripts"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
fi
LAUNCHER_PATH="${SCRIPT_DIR}/run_audio_recording_fhnw_fhnw.sh"

if [ -n "${PIPELINE_ENV_FILE:-}" ]; then
  ENV_FILE="${PIPELINE_ENV_FILE}"
elif [ -f "${HOME}/.env" ]; then
  ENV_FILE="${HOME}/.env"
elif [ -f "${PROJECT_DIR}/.env" ]; then
  ENV_FILE="${PROJECT_DIR}/.env"
else
  ENV_FILE="${REPO_DIR}/.env"
fi
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

RUN_NAME="${PIPELINE_RUN_NAME:-audio_recording_fhnw_numbers_v1}"
BATCH_SIZE="${PIPELINE_BATCH_SIZE:-100}"
REPO_ID="${PIPELINE_REPO_ID:-i4ds/audio_recording_fhnw}"
DATA_ROOT="${DATA_ROOT:-${HOME}/.local/share/swiss-german-number-preparation}"
export DATA_ROOT
PIPELINE_ROOT="${DATA_ROOT}/intermediate/preprocess_num_of_datasets/audio_recording_fhnw"
RUNS_DIR="${PIPELINE_RUNS_DIR:-${DATA_ROOT}/intermediate/preprocess_num_of_datasets/runs}"
LOG_DIR="${PIPELINE_LOG_DIR:-${DATA_ROOT}/logs/preprocess_num_of_datasets}"
HF_WORK_DIR="${PIPELINE_HF_HOME:-${PIPELINE_ROOT}/cache/huggingface}"
SOURCE_DIR="${PIPELINE_SOURCE_DIR:-${DATA_ROOT}/raw/preprocess_num_of_datasets/audio_recording_fhnw}"
ENV_DIR="${PIPELINE_ENV_PATH:-${REPO_DIR}/.venv-audio-fhnw-cu128}"
ENV_MARKER="${ENV_DIR}/.audio_numbers_fhnw_ready"
ENV_OVERRIDES="${PROJECT_DIR}/configs/fhnw-cu128-overrides.txt"
CUDA_MODULE="${PIPELINE_CUDA_MODULE:-cuda/12.8.1}"
DEFAULT_CONFIGS="BE_Julia_Locher,BE_Uriel_Brawer,BS_Luca_Zwicky"
IFS=',' read -r -a CONFIGS <<< "${PIPELINE_CONFIGS:-${DEFAULT_CONFIGS}}"

die() {
  echo "Error: $*" >&2
  exit 1
}

load_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then
    return
  fi
  local token_file="${HF_TOKEN_FILE:-${HOME}/.cache/huggingface/token}"
  if [ -s "${token_file}" ]; then
    IFS= read -r HF_TOKEN < "${token_file}"
    export HF_TOKEN
  fi
  [ -n "${HF_TOKEN:-}" ] || die \
    "HF_TOKEN is not set and no cached token was found at ${token_file}"
}

set_python_command() {
  if [ -n "${PIPELINE_CONDA_ENV:-}" ]; then
    command -v conda >/dev/null || die "conda is unavailable"
    PYTHON_CMD=(conda run --no-capture-output -n "${PIPELINE_CONDA_ENV}" python)
  else
    [ -x "${ENV_DIR}/bin/python" ] || die \
      "pipeline environment has no executable Python: ${ENV_DIR}"
    PYTHON_CMD=("${ENV_DIR}/bin/python")
  fi
}

environment_fingerprint() {
  {
    echo "schema=4 cuda=12.8"
    cat "${REPO_DIR}/collect_de_num_data/pyproject.toml" "${ENV_OVERRIDES}"
    echo "datasets>=4.8.4 pyarrow soundfile"
  } | sha256sum | awk '{print $1}'
}

validate_environment() {
  "${PYTHON_CMD[@]}" - "${FHNW_GPUS:-2}" <<'PYTHON'
import subprocess
import sys
from importlib.metadata import version

# FlashInfer launches ninja by name when compiling its CUDA kernels.
subprocess.run(["ninja", "--version"], check=True)

import torch

print(f"Environment: torch={torch.__version__}, CUDA={torch.version.cuda}, "
      f"vllm={version('vllm')}, visible GPUs={torch.cuda.device_count()}", flush=True)
if torch.version.cuda != "12.8":
    raise RuntimeError("This runner requires a CUDA 12.8 PyTorch build")
if version("vllm") != "0.15.0":
    raise RuntimeError("This runner requires vllm==0.15.0 (CUDA 12.8 wheel)")
if torch.cuda.device_count() < int(sys.argv[1]):
    raise RuntimeError(f"Expected at least {sys.argv[1]} visible GPUs")
import datasets, pyarrow, soundfile, vllm, yaml
# Exercise the driver/runtime as well as checking GPU visibility.
for device in range(torch.cuda.device_count()):
    torch.empty(1, device=f"cuda:{device}")
    torch.cuda.synchronize(device)
PYTHON
}

ensure_environment() {
  if [ -n "${PIPELINE_CONDA_ENV:-}" ]; then
    set_python_command
    validate_environment
    return
  fi

  local expected_fingerprint
  expected_fingerprint="$(environment_fingerprint)"
  if [ -x "${ENV_DIR}/bin/python" ] && [ -f "${ENV_MARKER}" ] && \
     [ "$(< "${ENV_MARKER}")" = "${expected_fingerprint}" ]; then
    echo "Reusing pipeline environment: ${ENV_DIR}"
    set_python_command
    validate_environment
    return
  fi

  command -v conda >/dev/null || die \
    "conda is required to create ${ENV_DIR}; use PIPELINE_CONDA_ENV if already installed"
  mkdir -p "${PIPELINE_ROOT}/conda_pkgs" "${PIPELINE_ROOT}/pip_cache"
  export CONDA_PKGS_DIRS="${PIPELINE_ROOT}/conda_pkgs"
  export PIP_CACHE_DIR="${PIPELINE_ROOT}/pip_cache"

  if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "Creating first-run Python environment at ${ENV_DIR} ..."
    conda create --yes --prefix "${ENV_DIR}" \
      --override-channels --channel conda-forge python=3.11 pip
  else
    echo "Refreshing incomplete or outdated environment at ${ENV_DIR} ..."
  fi

  "${ENV_DIR}/bin/python" -m pip install --upgrade pip uv
  "${ENV_DIR}/bin/python" -m uv pip install \
    --python "${ENV_DIR}/bin/python" \
    --torch-backend=cu128 \
    --override "${ENV_OVERRIDES}" \
    -e "${REPO_DIR}/collect_de_num_data" \
    "datasets>=4.8.4" pyarrow soundfile

  PYTHON_CMD=("${ENV_DIR}/bin/python")
  validate_environment
  printf '%s\n' "${expected_fingerprint}" > "${ENV_MARKER}"
  echo "Environment ready: ${ENV_DIR}"
}

submit_job() {
  mkdir -p "${LOG_DIR}"
  load_hf_token
  export PIPELINE_REPO_DIR="${REPO_DIR}"

  git -C "${REPO_DIR}" submodule update --init --recursive
  [ -f "${REPO_DIR}/num2words_CH/num2words/lang_CH_ZH.py" ] || die \
    "num2words_CH submodule is missing; run git submodule update --init --recursive"

  local partition="${FHNW_PARTITION:-h200}"
  local gpu_type="${FHNW_GPU_TYPE:-h200}"
  local gpu_count="${FHNW_GPUS:-2}"
  local cpus="${FHNW_CPUS:-16}"
  local memory="${FHNW_MEMORY:-96G}"
  local time_limit="${FHNW_TIME_LIMIT:-1-00:00:00}"
  local job_id
  job_id=$(sbatch --parsable \
    --export=ALL \
    --partition="${partition}" \
    --gres="gpu:${gpu_type}:${gpu_count}" \
    --cpus-per-task="${cpus}" \
    --mem="${memory}" \
    --time="${time_limit}" \
    --output="${LOG_DIR}/audio-num-fhnw-%j.log" \
    --error="${LOG_DIR}/audio-num-fhnw-%j.err" \
    "${LAUNCHER_PATH}" --worker)

  echo "Submitted FHNW end-to-end job: ${job_id}"
  echo "Log: ${LOG_DIR}/audio-num-fhnw-${job_id}.log"
  echo "Error log: ${LOG_DIR}/audio-num-fhnw-${job_id}.err"
  echo "Monitor: squeue -j ${job_id}"
}

download_source_data() {
  local download_args=(
    "${SCRIPT_DIR}/download_audio_recording_fhnw.py"
    --repo-id "${REPO_ID}"
    --output-dir "${SOURCE_DIR}"
  )
  local config
  for config in "${CONFIGS[@]}"; do
    [ -n "${config}" ] || die "PIPELINE_CONFIGS contains an empty config name"
    download_args+=(--config "${config}")
  done
  "${PYTHON_CMD[@]}" "${download_args[@]}"

  INPUTS=()
  for config in "${CONFIGS[@]}"; do
    local input="${SOURCE_DIR}/${config}/output.json"
    [ -f "${input}" ] || die "downloaded source manifest is missing: ${input}"
    INPUTS+=("${input}")
  done
}

run_pipeline() {
  load_hf_token
  mkdir -p "${RUNS_DIR}" "${LOG_DIR}" "${HF_WORK_DIR}"
  if [ -n "${CUDA_MODULE}" ]; then
    module load "${CUDA_MODULE}"
  fi
  export HF_HOME="${HF_WORK_DIR}"
  export HF_HUB_CACHE="${HF_WORK_DIR}/hub"
  export HF_DATASETS_CACHE="${HF_WORK_DIR}/datasets"
  export TOKENIZERS_PARALLELISM=false

  PIPELINE_STAGE="environment setup and validation"
  if [ -z "${PIPELINE_CONDA_ENV:-}" ]; then
    export PATH="${ENV_DIR}/bin:${PATH}"
  fi
  ensure_environment
  PIPELINE_STAGE="source download"
  download_source_data

  local run_dir="${RUNS_DIR}/${RUN_NAME}"
  local config_path="${run_dir}/config.yaml"
  local prompts_path="${run_dir}/prompts.json"
  local manifest_path="${run_dir}/manifest.json"
  local checkpoint_path="${run_dir}/results_checkpoint.jsonl"
  local results_path="${run_dir}/results.json"

  PIPELINE_STAGE="run preparation"
  if [ -f "${config_path}" ] && [ -f "${prompts_path}" ] && [ -f "${manifest_path}" ]; then
    echo "Resuming prepared run: ${RUN_NAME}"
  elif [ -e "${run_dir}" ]; then
    die "incomplete run setup exists at ${run_dir}; move it aside and resubmit"
  else
    echo "Preparing all audio_recording_fhnw configs..."
    "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/preprocess_dataset_numbers.py" prepare \
      "${INPUTS[@]}" \
      --run-name "${RUN_NAME}" \
      --runs-dir "${RUNS_DIR}" \
      --sentence-ch-column ch \
      --sentence-de-column de \
      --id-column id
  fi

  PIPELINE_STAGE="inference"
  if [ -f "${results_path}" ]; then
    echo "Inference already complete: ${results_path}"
  else
    echo "Running resumable LLM inference..."
    "${PYTHON_CMD[@]}" \
      "${REPO_DIR}/collect_de_num_data/scripts/labeling/run_llm_inference.py" \
      --run-name "${RUN_NAME}" \
      --config-path "${config_path}" \
      --prompts-path "${prompts_path}" \
      --checkpoint-path "${checkpoint_path}" \
      --results-path "${results_path}" \
      --batch_size "${BATCH_SIZE}"
  fi

  PIPELINE_STAGE="validation and publication"
  echo "Rebuilding, validating, and publishing every config..."
  "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/publish_audio_recording_fhnw.py" \
    --run-name "${RUN_NAME}" \
    --runs-dir "${RUNS_DIR}" \
    --repo-id "${REPO_ID}" \
    --push

  echo "End-to-end pipeline complete: https://huggingface.co/datasets/${REPO_ID}"
}

if [ "${1:-}" = "--worker" ]; then
  run_pipeline
else
  submit_job
fi
