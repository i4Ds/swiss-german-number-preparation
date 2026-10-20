"""Storage and cluster-safety helpers for the number-data pipeline."""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

COMPONENT = "collect_de_num_data"
STAGES = ("raw", "intermediate", "processed", "results", "logs")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    configured = os.environ.get("DATA_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = (Path.home() / ".local/share/swiss-german-number-preparation").resolve()
    if root == REPOSITORY_ROOT or REPOSITORY_ROOT in root.parents:
        raise RuntimeError("DATA_ROOT must be outside the Git repository")
    return root


def storage_path(stage: str, *parts: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"invalid storage stage {stage!r}; expected one of {STAGES}")
    return (data_root() / stage / COMPONENT).joinpath(*parts)


def require_compute_node(operation: str) -> None:
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("TTS_ALLOW_LOGIN_NODE") == "1":
        return
    raise RuntimeError(
        f"Refusing to run {operation} without a Slurm allocation on "
        f"{socket.gethostname()!r}. Use srun or sbatch."
    )


def configure_job_environment() -> None:
    cache = storage_path("intermediate", "cache")
    if os.environ.get("SLURM_JOB_ID") and not os.environ.get("TMPDIR"):
        raise RuntimeError("TMPDIR is unset inside the Slurm job")
    tmp_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir())).resolve()
    job_tmp = tmp_root / "tts_data_collection_and_prep" / COMPONENT
    job_tmp.mkdir(parents=True, exist_ok=True)
    defaults = {
        "TMP": str(job_tmp),
        "TEMP": str(job_tmp),
        "HF_HOME": str(cache / "huggingface"),
        "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
        "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
