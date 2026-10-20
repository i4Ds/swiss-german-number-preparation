#!/usr/bin/env bash
# Submit the prepared German → Swiss German CSV translation to SciCORE.
#
# Usage:
#   bash scripts/translation/run_translation_on_scicore.sh [input_csv] [batch_size] [model_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${HOME}/.local/share/swiss-german-number-preparation}"
export DATA_ROOT

DEFAULT_INPUT="$DATA_ROOT/intermediate/collect_de_num_data/prepared/synthetic_full_flash_lite_prepared.csv"
DEFAULT_MODEL="${T5_MODEL:-}"

INPUT_CSV="${1:-$DEFAULT_INPUT}"
BATCH_SIZE="${2:-16}"
MODEL_DIR="${3:-$DEFAULT_MODEL}"
CONDA_ENV="${CONDA_ENV:-llm_env}"
PARTITION="${PARTITION:-h200}"
QOS="${QOS:-h200-6hours}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
LOG_DIR="${LOG_DIR:-$DATA_ROOT/logs/collect_de_num_data}"

if [[ ! -f "$INPUT_CSV" ]]; then
    echo "Error: input CSV not found: $INPUT_CSV" >&2
    exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: T5 checkpoint directory not found: $MODEL_DIR" >&2
    exit 1
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: batch_size must be a positive integer, got: $BATCH_SIZE" >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
    echo "Error: conda environment '$CONDA_ENV' not found." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "=== T5 CSV translation on SciCORE ==="
echo "Input CSV : $INPUT_CSV"
echo "Model     : $MODEL_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Environment: $CONDA_ENV"
echo "Partition : $PARTITION"
echo "QoS       : $QOS"
echo "Time limit: $TIME_LIMIT"

JOB_ID=$(sbatch --parsable <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=t5-csv-translate
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/t5-csv-translate-%j.log
#SBATCH --error=${LOG_DIR}/t5-csv-translate-%j.err

set -euo pipefail

module load CUDA/12.1 2>/dev/null || module load cuda 2>/dev/null || true

echo "SLURM_JOB_ID=\$SLURM_JOB_ID"
echo "HOSTNAME=\$HOSTNAME"
echo "GPU information:"
nvidia-smi


source "${CONDA_BASE}/etc/profile.d/conda.sh"
export JAVA_HOME="\${JAVA_HOME:-}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${T5_PROJECT_DIR}:\${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"
python -u scripts/translation/translate_prepared_csv.py "${INPUT_CSV}" --model "${MODEL_DIR}" --batch-size "${BATCH_SIZE}"

echo "T5 CSV translation completed successfully"
EOF
)

echo "Submitted job ID: $JOB_ID"
echo "Log: $LOG_DIR/t5-csv-translate-$JOB_ID.log"
echo "Error log: $LOG_DIR/t5-csv-translate-$JOB_ID.err"
echo "Monitor: squeue -j $JOB_ID"
