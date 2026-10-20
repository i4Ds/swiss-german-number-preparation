#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${HOME}/.local/share/swiss-german-number-preparation}"
export DATA_ROOT
COLLECTION_RUNNER="$PROJECT_DIR/../collect_de_num_data/scripts/labeling/run_on_scicore.sh"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run_on_scicore.sh RUN_NAME [BATCH_SIZE] [INPUT ...]" >&2
    echo "" >&2
    echo "Create and submit a run:" >&2
    echo "  bash scripts/run_on_scicore.sh test_numbers 10 data/test_samples.csv" >&2
    echo "" >&2
    echo "Resume an existing run:" >&2
    echo "  bash scripts/run_on_scicore.sh test_numbers 10" >&2
    exit 2
fi

RUN_NAME="$1"
BATCH_SIZE="${2:-100}"
RUN_DIR="$DATA_ROOT/intermediate/preprocess_num_of_datasets/runs/$RUN_NAME"
PROMPTS_PATH="$RUN_DIR/prompts.json"
MANIFEST_PATH="$RUN_DIR/manifest.json"
CONFIG_PATH="$RUN_DIR/config.yaml"

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: BATCH_SIZE must be a positive integer, got: $BATCH_SIZE" >&2
    exit 2
fi

if [ ! -x "$COLLECTION_RUNNER" ]; then
    echo "Error: existing SciCORE launcher is missing or not executable:" >&2
    echo "  $COLLECTION_RUNNER" >&2
    exit 1
fi

if [ -f "$PROMPTS_PATH" ] && [ -f "$MANIFEST_PATH" ] && [ -f "$CONFIG_PATH" ]; then
    if [ "$#" -gt 2 ]; then
        echo "Error: run '$RUN_NAME' already exists; do not provide new inputs when resuming." >&2
        echo "Use a new run name for different input data." >&2
        exit 1
    fi
    echo "Resuming prepared run: $RUN_NAME"
else
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        echo "Error: preparing a run is compute work; invoke this launcher inside srun." >&2
        exit 2
    fi
    if [ "$#" -lt 3 ]; then
        echo "Error: run '$RUN_NAME' is not prepared." >&2
        echo "Provide at least one input dataset as the third argument." >&2
        exit 1
    fi
    shift 2
    echo "Preparing run '$RUN_NAME' from $# input dataset(s)..."
    python "$SCRIPT_DIR/preprocess_dataset_numbers.py" prepare \
        "$@" \
        --run-name "$RUN_NAME"
fi

echo ""
echo "Submitting the large-LLM inference job through the existing launcher..."
echo "Run directory: $RUN_DIR"
echo "Config:        $CONFIG_PATH"
echo ""

bash "$COLLECTION_RUNNER" "$RUN_NAME" "$BATCH_SIZE" "$RUN_DIR"
