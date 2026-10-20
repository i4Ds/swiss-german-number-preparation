#!/usr/bin/env bash
# Lightweight login-node bootstrap: clone/update code, then submit the GPU job.

set -euo pipefail

ENV_FILE="${1:-${PIPELINE_ENV_FILE:-${HOME}/.env}}"
[ -f "${ENV_FILE}" ] || {
  echo "Error: environment file not found: ${ENV_FILE}" >&2
  exit 1
}
ENV_FILE="$(cd "$(dirname "${ENV_FILE}")" && pwd)/$(basename "${ENV_FILE}")"

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

REPO_URL="${PIPELINE_GIT_REPO:?Set PIPELINE_GIT_REPO to this publication repository URL}"
REPO_BRANCH="${PIPELINE_GIT_BRANCH:-main}"
CHECKOUT_DIR="${PIPELINE_CHECKOUT_DIR:-${HOME}/projects/swiss-german-number-preparation}"

if [ -d "${CHECKOUT_DIR}/.git" ]; then
  if [ -n "$(git -C "${CHECKOUT_DIR}" status --porcelain)" ]; then
    echo "Error: checkout has local changes; refusing to overwrite: ${CHECKOUT_DIR}" >&2
    exit 1
  fi
  echo "Updating repository at ${CHECKOUT_DIR} ..."
  git -C "${CHECKOUT_DIR}" fetch --depth=1 origin "${REPO_BRANCH}"
  git -C "${CHECKOUT_DIR}" checkout "${REPO_BRANCH}"
  git -C "${CHECKOUT_DIR}" merge --ff-only "origin/${REPO_BRANCH}"
else
  [ ! -e "${CHECKOUT_DIR}" ] || {
    echo "Error: checkout path exists but is not a Git repository: ${CHECKOUT_DIR}" >&2
    exit 1
  }
  mkdir -p "$(dirname "${CHECKOUT_DIR}")"
  echo "Creating sparse repository checkout at ${CHECKOUT_DIR} ..."
  git clone \
    --branch "${REPO_BRANCH}" \
    --depth=1 \
    --filter=blob:none \
    --sparse \
    "${REPO_URL}" \
    "${CHECKOUT_DIR}"
fi

# Avoid fetching the historically tracked recording audio from Git. The GPU
# job downloads the authoritative dataset from Hugging Face into DATA_ROOT.
git -C "${CHECKOUT_DIR}" sparse-checkout set \
  collect_de_num_data \
  preprocess_num_of_datasets \
  num2words_CH
git -C "${CHECKOUT_DIR}" submodule sync --recursive
git -C "${CHECKOUT_DIR}" submodule update \
  --init \
  --recursive \
  --depth=1 \
  num2words_CH

export PIPELINE_ENV_FILE="${ENV_FILE}"
exec bash \
  "${CHECKOUT_DIR}/preprocess_num_of_datasets/scripts/run_audio_recording_fhnw_fhnw.sh"
