#!/usr/bin/env python3
"""Run number annotation and conversion for separate local CSV files."""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESS = SCRIPT_DIR / "preprocess_dataset_numbers.py"
INFERENCE = (
    SCRIPT_DIR.parents[1]
    / "collect_de_num_data"
    / "scripts"
    / "labeling"
    / "run_llm_inference.py"
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML object: {path}")
    return config


def input_paths(config: dict) -> list[Path]:
    paths = sorted(Path(path) for path in glob.glob(config["input_glob"]))
    if not paths:
        raise FileNotFoundError(f"No input CSV files match: {config['input_glob']}")
    return paths


def command_args(config_path: Path, config: dict, paths: list[Path]) -> list[str]:
    return [
        str(PREPROCESS),
        "prepare",
        *(str(path) for path in paths),
        "--run-name",
        config["run_name"],
        "--runs-dir",
        config["runs_dir"],
        "--sentence-ch-column",
        config["sentence_ch_column"],
        "--sentence-de-column",
        config["sentence_de_column"],
        "--id-column",
        config["id_column"],
        "--config-template",
        config.get("llm_config_template", str(PREPROCESS.parent.parent / "configs" / "llm_inference.yaml")),
    ]


def run_worker(config_path: Path, config: dict) -> None:
    paths = input_paths(config)
    run_dir = Path(config["runs_dir"]) / config["run_name"]
    python = config.get("python", sys.executable)
    setup = run_dir / "manifest.json"
    if not setup.exists():
        subprocess.run([python, *command_args(config_path, config, paths)], check=True)

    results = run_dir / "results.json"
    if not results.exists():
        subprocess.run(
            [
                python,
                str(INFERENCE),
                "--run-name",
                config["run_name"],
                "--config-path",
                str(run_dir / "config.yaml"),
                "--prompts-path",
                str(run_dir / "prompts.json"),
                "--checkpoint-path",
                str(run_dir / "results_checkpoint.jsonl"),
                "--results-path",
                str(results),
                "--batch_size",
                str(config.get("batch_size", 100)),
            ],
            check=True,
        )

    subprocess.run(
        [
            python,
            str(PREPROCESS),
            "apply",
            "--run-name",
            config["run_name"],
            "--runs-dir",
            config["runs_dir"],
            "--output-dir",
            config["output_dir"],
            "--dialect",
            config.get("dialect", "ch_sg"),
            "--on-error",
            "keep",
            "--remaining-digits",
            "report",
            "--issue-flag-column",
            config.get("issue_flag_column", "number_processing_issue"),
            "--issue-reason-column",
            config.get("issue_reason_column", "number_processing_issue_reason"),
            "--overwrite",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    if args.worker or os.environ.get("SLURM_JOB_ID"):
        run_worker(config_path, config)
        return

    resources = config.get("slurm", {})
    job = [
        "sbatch",
        "--parsable",
        "--export=ALL",
        "--partition=" + resources.get("partition", "h200"),
        "--gres=gpu:" + resources.get("gpu_type", "h200") + ":" + str(resources.get("gpus", 2)),
        "--cpus-per-task=" + str(resources.get("cpus", 16)),
        "--mem=" + str(resources.get("memory", "96G")),
        "--time=" + str(resources.get("time", "1-00:00:00")),
        "--wrap=" + " ".join(
            [
                config.get("python", sys.executable),
                str(Path(__file__).resolve()),
                "--config",
                str(config_path),
                "--worker",
            ]
        ),
    ]
    print("Submitted job:", subprocess.check_output(job, text=True).strip())


if __name__ == "__main__":
    main()