"""Storage and cluster-safety helpers for dataset number preprocessing."""

from __future__ import annotations

import os
import socket
from pathlib import Path

COMPONENT = "preprocess_num_of_datasets"
STAGES = ("raw", "intermediate", "processed", "results", "logs")
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


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
