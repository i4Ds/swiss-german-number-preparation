#!/usr/bin/env python3
"""Download Hugging Face speaker configs into preprocessing source folders."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_runtime import require_compute_node, storage_path


DEFAULT_REPO_ID = "i4ds/audio_recording_fhnw"
REQUIRED_COLUMNS = {"id", "de", "ch", "duration_s", "audio"}


def audio_suffix(audio_value) -> str:
    if isinstance(audio_value, dict):
        source_path = audio_value.get("path")
        if source_path:
            suffix = Path(source_path).suffix
            if suffix:
                return suffix
    return ".flac"


def materialize_audio(audio_value, destination: Path) -> None:
    if isinstance(audio_value, dict):
        audio_bytes = audio_value.get("bytes")
        source_path = audio_value.get("path")
    elif isinstance(audio_value, str):
        audio_bytes = None
        source_path = audio_value
    else:
        raise ValueError(f"Unsupported audio value: {type(audio_value).__name__}")

    if audio_bytes is not None:
        destination.write_bytes(audio_bytes)
    elif source_path and Path(source_path).is_file():
        shutil.copyfile(source_path, destination)
    else:
        raise ValueError("Audio entry has neither embedded bytes nor a readable path")


def download_config(repo_id: str, config: str, split: str, target: Path) -> dict:
    from datasets import Audio, load_dataset

    print(f"Downloading {repo_id}, config={config}, split={split} ...")
    dataset = load_dataset(repo_id, config, split=split)
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(f"Config {config!r} is missing columns: {sorted(missing)}")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    config_dir = target / config
    audio_dir = config_dir / "audio"
    audio_dir.mkdir(parents=True)
    records = []
    for row_index, source in enumerate(dataset):
        audio_value = source["audio"]
        filename = f"{row_index:08d}{audio_suffix(audio_value)}"
        materialize_audio(audio_value, audio_dir / filename)
        record = {key: value for key, value in source.items() if key != "audio"}
        record["audio"] = filename
        records.append(record)

    output_path = config_dir / "output.json"
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "config": config,
        "split": split,
        "rows": len(records),
        "columns": dataset.column_names,
        "input": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=storage_path("raw", "audio_recording_fhnw"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--config",
        action="append",
        help="Config to download; repeat as needed (default: every non-default config)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_compute_node("audio-recording dataset download")
    from datasets import get_dataset_config_names

    output_dir = args.output_dir.expanduser().resolve()
    completion_path = output_dir / "download_summary.json"
    if completion_path.is_file():
        print(f"Reusing completed source download: {output_dir}")
        print(completion_path.read_text(encoding="utf-8"), end="")
        return
    if output_dir.exists():
        raise FileExistsError(
            f"Incomplete source directory exists; move it aside before retrying: {output_dir}"
        )

    configs = args.config or [
        name for name in get_dataset_config_names(args.repo_id) if name != "default"
    ]
    if not configs:
        raise ValueError(f"No non-default configs found in {args.repo_id}")
    if len(set(configs)) != len(configs):
        raise ValueError("Each config may be specified only once")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.download-", dir=output_dir.parent)
    )
    try:
        summaries = [
            download_config(args.repo_id, config, args.split, temporary)
            for config in configs
        ]
        # Store paths relative to the final directory, not the temporary name.
        for summary in summaries:
            summary["input"] = str(
                output_dir / summary["config"] / "output.json"
            )
        (temporary / "download_summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        print(f"Incomplete temporary download retained for diagnosis: {temporary}")
        raise

    print(f"Downloaded {len(summaries)} configs to {output_dir}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
