#!/bin/bash
#SBATCH --job-name=download_numbers
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=performance
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Download + extract number sentences from:
#   1. coral-nlp/german-commons
#   2. storytracer/German-PD-Newspapers
#
# Submit from the project root:
#   sbatch scripts/collection/run_numbers_slurm.sh

run_job() {
    set -e

    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

    echo "========================================"
    echo "Number Data Download Job"
    echo "========================================"
    echo "Job ID:   $SLURM_JOB_ID"
    echo "Node:     $SLURM_NODELIST"
    echo "Workdir:  $PROJECT_ROOT"
    echo ""

    DATA_ROOT="${DATA_ROOT:-${HOME}/.local/share/swiss-german-number-preparation}"
    export DATA_ROOT

    source "${PROJECT_ROOT}/../.venv/bin/activate"

    cd "$PROJECT_ROOT"

    # Proxy for HuggingFace access on compute nodes
    if [ -f "$PROJECT_ROOT/set_proxy.sh" ]; then
        source "$PROJECT_ROOT/set_proxy.sh"
    fi

    # echo ""
    # echo "========================================"
    # echo "Step 1 / 2 – storytracer/German-PD-Newspapers"
    # echo "========================================"
    # python -u german_newspapers_numbers.py

    echo ""
    echo "========================================"
    echo "Step 2 / 2 – coral-nlp/german-commons"
    echo "========================================"
    python -u src/collection/german_commons_numbers.py



    echo ""
    echo "========================================"
    echo "All done!"
    echo "========================================"
}

# ── Output file setup (mirrors sb_t5.sh pattern) ─────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${HOME}/.local/share/swiss-german-number-preparation}"
LOG_DIR="$DATA_ROOT/logs/collect_de_num_data"
mkdir -p "$LOG_DIR"

current_date=$(date +%Y%m%d_%H%M)
job_id=${SLURM_JOB_ID:-local}
output_file="${LOG_DIR}/numbers__${current_date}__${job_id}.out"

run_job > "$output_file" 2>&1
echo "Log: $output_file"
