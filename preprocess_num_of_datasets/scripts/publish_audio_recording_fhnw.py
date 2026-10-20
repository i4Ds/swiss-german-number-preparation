#!/usr/bin/env python3
"""Rebuild and publish audio_recording_fhnw after number preprocessing.

The LLM run may contain all speaker JSON files. This script preserves every
record, audio file, and source column. It appends the converted text as
``written_out_num_sentence_ch``, fully validates every config, and only then
starts uploading to Hugging Face.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_runtime import require_compute_node, storage_path

PREPROCESS_MODULE = PROJECT_ROOT / "scripts" / "preprocess_dataset_numbers.py"
DEFAULT_RUNS_DIR = storage_path("intermediate", "runs")
DEFAULT_REPO_ID = "i4ds/audio_recording_fhnw"
REQUIRED_KEYS = {"id", "de", "ch", "duration_s", "audio"}
OUTPUT_COLUMN = "written_out_num_sentence_ch"
DIGIT_RE = re.compile(r"\d")


def load_preprocessing_module():
    spec = importlib.util.spec_from_file_location(
        "preprocess_dataset_numbers", PREPROCESS_MODULE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load preprocessing module: {PREPROCESS_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_mapping(value: str) -> tuple[str, str]:
    key, separator, mapped_value = value.partition("=")
    if not separator or not key or not mapped_value:
        raise argparse.ArgumentTypeError("expected CONFIG=VALUE")
    return key, mapped_value


def dialect_for_config(config_name: str, overrides: dict[str, str]) -> str:
    if config_name in overrides:
        return overrides[config_name]
    # The converter currently has no Bernese implementation. Keep its documented
    # ch_sg fallback for BE, while using the matching implementation for Basel.
    return "ch_bs" if config_name.startswith("BS_") else "ch_sg"


def config_name_for_input(path: Path) -> str:
    if path.name != "output.json":
        raise ValueError(f"Expected a speaker output.json, got: {path}")
    return path.parent.name


def build_output_record(source: dict, audio_path: Path, written_text: str) -> dict:
    """Preserve source fields and append the generated column last."""
    missing = REQUIRED_KEYS - set(source)
    if missing:
        raise ValueError(f"Source record is missing columns: {sorted(missing)}")
    record = {
        key: value
        for key, value in source.items()
        if key not in {"audio", OUTPUT_COLUMN}
    }
    record["audio"] = str(audio_path)
    record[OUTPUT_COLUMN] = written_text
    return record


def build_configs(
    run_dir: Path,
    preprocessing,
    dialect_overrides: dict[str, str],
) -> tuple[dict[str, Dataset], list[dict]]:
    from datasets import Audio, Dataset

    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"Completed inference output not found: {results_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = preprocessing.load_result_entries(results_path)
    expected_ids = {row["prompt_id"] for row in manifest["prompt_rows"]}
    if set(results) != expected_ids:
        raise ValueError(
            "Inference is incomplete or mismatched: "
            f"{len(results)}/{len(expected_ids)} exact prompt results"
        )

    row_to_prompt = {
        (row["input_index"], row["row_index"]): row["prompt_id"]
        for row in manifest["prompt_rows"]
    }
    converter = preprocessing.load_number_converter()
    datasets = {}
    summaries = []

    for input_index, input_info in enumerate(manifest["inputs"]):
        input_path = Path(input_info["path"])
        if preprocessing.sha256_file(input_path) != input_info["sha256"]:
            raise ValueError(f"Source changed after run preparation: {input_path}")
        records = json.loads(input_path.read_text(encoding="utf-8"))
        config_name = config_name_for_input(input_path)
        if config_name in datasets:
            raise ValueError(f"Duplicate config in manifest: {config_name}")
        dialect = dialect_for_config(config_name, dialect_overrides)
        audio_dir = input_path.parent / "audio"
        upload_records = []
        changed_rows = 0
        labeled_rows = 0
        replacements = 0

        for row_index, source in enumerate(records):
            written_text = source["ch"]
            prompt_id = row_to_prompt.get((input_index, row_index))
            if prompt_id is not None:
                labeled_rows += 1
                written_text, row_replacements = preprocessing.apply_annotations(
                    source["ch"],
                    results[prompt_id],
                    converter=converter,
                    dialect=dialect,
                )
                if DIGIT_RE.search(written_text):
                    raise ValueError(
                        f"Digits remain in {config_name} row {row_index + 1}: "
                        f"{written_text!r}"
                    )
                changed_rows += written_text != source["ch"]
                replacements += len(row_replacements)

            audio_path = (audio_dir / source["audio"]).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            upload_records.append(
                build_output_record(source, audio_path, written_text)
            )

        dataset = Dataset.from_list(upload_records).cast_column("audio", Audio())
        datasets[config_name] = dataset
        summaries.append(
            {
                "config": config_name,
                "rows": len(records),
                "labeled_rows": labeled_rows,
                "changed_rows": changed_rows,
                "replacements": replacements,
                "dialect": dialect,
                "columns": dataset.column_names,
            }
        )

    prompt_counts = Counter(row["input_index"] for row in manifest["prompt_rows"])
    for input_index, summary in enumerate(summaries):
        if summary["labeled_rows"] != prompt_counts[input_index]:
            raise AssertionError(f"Prompt accounting failed for {summary['config']}")
    return datasets, summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--dialect",
        action="append",
        type=parse_mapping,
        metavar="CONFIG=DIALECT",
        help="Override a config's converter dialect (ch_sg, ch_bs, or ch_zh)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload every validated speaker config after the local rebuild",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_compute_node("audio-recording dataset publishing")
    run_dir = args.runs_dir.expanduser().resolve() / args.run_name
    preprocessing = load_preprocessing_module()
    dialect_overrides = dict(args.dialect or [])
    if len(dialect_overrides) != len(args.dialect or []):
        raise ValueError("Each config may have at most one --dialect override")
    invalid_dialects = set(dialect_overrides.values()) - {"ch_sg", "ch_bs", "ch_zh"}
    if invalid_dialects:
        raise ValueError(f"Unsupported dialects: {sorted(invalid_dialects)}")
    datasets, summaries = build_configs(
        run_dir,
        preprocessing,
        dialect_overrides,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))

    if not args.push:
        print("Validation complete; add --push to publish.")
        return

    print(f"Publishing {len(datasets)} configs to {args.repo_id} ...")
    for config_name, dataset in datasets.items():
        print(f"Uploading {config_name}: {dataset.num_rows} rows")
        dataset.push_to_hub(
            args.repo_id,
            config_name=config_name,
            split="train",
            token=args.token,
        )
    print(f"Published: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
