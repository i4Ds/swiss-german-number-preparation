#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 0 ]]; then
    export PIPELINE_ENV_FILE="$1"
fi
exec bash "$REPO_DIR/preprocess_num_of_datasets/scripts/run_audio_recording_fhnw_fhnw.sh"
