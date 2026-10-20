#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${HOME}/.local/share/swiss-german-number-preparation}"
export DATA_ROOT
RUN_NAME="${1:-qwen3vl_run_01}"
BATCH_SIZE="${2:-100}"
ARTIFACT_DIR="${3:-}"
PARTITION="rtx4090"
QOS="rtx4090-6hours"
NUM_GPUS=1

# Override partition/QOS based on gpu_type in config (read after config validation below)
TIME_LIMIT="04:00:00"
MEM_PER_CPU="8G"
CPUS_PER_TASK=8
HF_CACHE="${HF_HOME:-${DATA_ROOT}/intermediate/collect_de_num_data/cache/huggingface}"
LOG_DIR="${PIPELINE_LOG_DIR:-${DATA_ROOT}/logs/collect_de_num_data}"
PYTHON_BIN="${LLM_PYTHON:-${REPO_DIR}/.venv/bin/python}"
CUDA_MODULE="${CUDA_MODULE-CUDA/12.6.0}"


# Validate run inputs
if [ -n "$ARTIFACT_DIR" ]; then
    ARTIFACT_DIR="$(cd "$ARTIFACT_DIR" && pwd)"
    CONFIG_PATH="${ARTIFACT_DIR}/config.yaml"
    PROMPTS_PATH="${ARTIFACT_DIR}/prompts.json"
    CHECKPOINT_PATH="${ARTIFACT_DIR}/results_checkpoint.jsonl"
    RESULTS_PATH="${ARTIFACT_DIR}/results.json"
    INFERENCE_PATH_ARGS="--config-path ${CONFIG_PATH} --prompts-path ${PROMPTS_PATH} --checkpoint-path ${CHECKPOINT_PATH} --results-path ${RESULTS_PATH}"
else
    CONFIG_PATH="${PROJECT_DIR}/configs/llm_inference/${RUN_NAME}.yaml"
    PROMPTS_PATH="${DATA_ROOT}/intermediate/collect_de_num_data/prompts/${RUN_NAME}_prompts.json"
    INFERENCE_PATH_ARGS=""
fi

echo SCRIPT_DIR: $SCRIPT_DIR
if [ -f "$SCRIPT_DIR/set_proxy.sh" ]; then
    source "$SCRIPT_DIR/set_proxy.sh"
fi
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: config not found: ${CONFIG_PATH}" >&2
  echo "Available run configs:" >&2
  ls "${PROJECT_DIR}/configs/llm_inference/" >&2
  exit 1
fi

if [ ! -f "$PROMPTS_PATH" ]; then
  echo "Error: prompts.json not found: ${PROMPTS_PATH}" >&2
  exit 1
fi

# Read GPU count from config if set
CONFIG_GPUS=$(grep '^gpus:' "$CONFIG_PATH" | awk '{print $2}')
if [ -n "$CONFIG_GPUS" ]; then
  NUM_GPUS="$CONFIG_GPUS"
fi
NUM_GPUS="${SLURM_GPUS:-$NUM_GPUS}"

# Derive partition from gpu_type in config
CONFIG_GPU_TYPE=$(grep '^gpu_type:' "$CONFIG_PATH" | awk '{print $2}')
case "$CONFIG_GPU_TYPE" in
  h200)
    PARTITION="h200"
    ;;
  rtxA4500)
    PARTITION="rtx4090"
    ;;
  # rtx4090 and anything else: keep defaults
esac
PARTITION="${SLURM_PARTITION:-$PARTITION}"

# Read requested time limit from config (e.g. `time_limit: "24:00:00"`); default stays 4 hours
CONFIG_TIME_LIMIT=$(grep '^time_limit:' "$CONFIG_PATH" | awk '{print $2}' | tr -d '"')
if [ -n "$CONFIG_TIME_LIMIT" ]; then
  TIME_LIMIT="$CONFIG_TIME_LIMIT"
fi
TIME_LIMIT="${SLURM_TIME_LIMIT:-$TIME_LIMIT}"

# Pick the QOS tier (<partition>-<tier>) that matches the requested time limit
case "$TIME_LIMIT" in
  00:30:00) QOS_TIER="30min" ;;
  1-00:00:00|24:00:00) QOS_TIER="1day" ;;
  7-00:00:00|168:00:00) QOS_TIER="1week" ;;
  *) QOS_TIER="6hours" ;;
esac
QOS="${PARTITION}-${QOS_TIER}"
QOS="${SLURM_QOS:-$QOS}"

echo "=== LLM Labeling on scicore ==="
echo "Run name   : ${RUN_NAME}"
echo "Project dir: ${PROJECT_DIR}"
echo "Python     : ${PYTHON_BIN}"
echo "Partition  : ${PARTITION}"
echo "QOS        : ${QOS}"
echo "Time limit : ${TIME_LIMIT}"
echo "GPUs       : ${NUM_GPUS}"
echo "HF cache   : ${HF_CACHE}"
if [ -n "$ARTIFACT_DIR" ]; then
    echo "Artifacts  : ${ARTIFACT_DIR}"
fi
echo ""

# Step 1: Ensure the compute-installed environment exists
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python environment not found. Run install_dependencies.sh through srun:" >&2
  echo "  bash ${PROJECT_DIR}/scripts/install_dependencies.sh" >&2
  exit 1
fi

# Step 2: Prepare directories
mkdir -p "$HF_CACHE" "$LOG_DIR"

# Step 3: Submit Slurm job
echo "Submitting Slurm job..."
JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=llm-inf-${RUN_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --qos=${QOS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/llm-inf-${RUN_NAME}-%j.log
#SBATCH --error=${LOG_DIR}/llm-inf-${RUN_NAME}-%j.err

# Exit the Slurm job immediately if any command fails. Without this, a failed
# inference command could be masked by the successful completion echo below.
set -e

if [ -n "${CUDA_MODULE}" ]; then
  module load "${CUDA_MODULE}"
fi

export NCCL_P2P_DISABLE=1
export HF_HOME="${HF_CACHE}"

# Load credentials without word-splitting or evaluating command substitutions.
if [ -f "${PROJECT_DIR}/.env" ]; then
  set -a
  source "${PROJECT_DIR}/.env"
  set +a
fi

echo "SLURM_JOB_ID=\$SLURM_JOB_ID"
echo "HOSTNAME=\$HOSTNAME"
echo "Run name: ${RUN_NAME}"
echo ""

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" scripts/labeling/run_llm_inference.py --run-name "${RUN_NAME}" --batch_size "${BATCH_SIZE}" ${INFERENCE_PATH_ARGS}

echo "Job llm-inf-${RUN_NAME} completed"
EOF
)

echo "Submitted job ID: ${JOB_ID}"
echo "Log: ${LOG_DIR}/llm-inf-${RUN_NAME}-${JOB_ID}.log"
echo ""
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${LOG_DIR}/llm-inf-${RUN_NAME}-${JOB_ID}.log"
