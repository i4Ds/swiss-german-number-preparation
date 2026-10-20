#!/usr/bin/env python3
"""Add written-out Swiss German numbers and optionally publish the dataset.

Each configured inference run corresponds to one split of
``i4ds/translator_data_wikipedia``. Source rows and columns are preserved. Export
number annotations, written text, and per-row conversion provenance. Select the
converter from each row's dialect; rows without digits retain their source text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_runtime import require_compute_node, storage_path

RUNS_DIR = storage_path("intermediate", "runs")
OUTPUT_DIR = storage_path("processed", "written_number_dataset")
PREPROCESS_MODULE = PROJECT_ROOT / "scripts" / "preprocess_dataset_numbers.py"
OUTPUT_COLUMN = "written_out_num_sentence_ch"
SUPPORTED_DIALECTS = ("ch_sg", "ch_bs", "ch_zh", "ch_lu", "ch_be")
DIALECT_NAMES = {
    "basel": "ch_bs", "st. gallen": "ch_sg", "st gallen": "ch_sg",
    "zürich": "ch_zh", "zurich": "ch_zh", "bern": "ch_be",
    "bernese": "ch_be", "luzern": "ch_lu", "lucerne": "ch_lu",
    "lucern": "ch_lu",
}
ADDED_COLUMNS = (
    OUTPUT_COLUMN, "numbers", "number_dialect", "number_dialect_source",
    "number_processing_status", "number_processing_error",
)
DEFAULT_REPO_ID = "i4ds/translator_data_wikipedia"
DEFAULT_RUNS = {
    "train": "wikipedia_train_numbers_v1",
    "validation": "wikipedia_validation_numbers_v1",
    "test": "wikipedia_test_v1",
    "swiss_german": "wikipedia_swiss_german_numbers_v1",
}


def resolve_dialect(row: dict, dialect: str, fallback: str) -> tuple[str, str]:
    if dialect != "auto":
        return dialect, "override"
    # The fine label retains the model's regional label even when the legacy
    # canton column contains the generic "ch" code.
    for column in ("dialect_fine", "dialect"):
        label = str(row.get(column) or "").strip().casefold()
        candidate = DIALECT_NAMES.get(label, label)
        if not candidate.startswith("ch_"):
            candidate = "ch_" + candidate
        if candidate in SUPPORTED_DIALECTS:
            return candidate, column
    return fallback, "fallback"


def annotation_type(pa):
    return pa.list_(pa.struct([
        pa.field("number", pa.string()), pa.field("type", pa.string()),
        pa.field("subtype", pa.string()), pa.field("value_json", pa.string()),
        pa.field("start", pa.int64()), pa.field("end", pa.int64()),
        pa.field("spoken_form", pa.string()),
        pa.field("annotation_json", pa.string()),
    ]))


def export_annotations(response: dict, sentence: str, preprocessing, replacements: list) -> list:
    """Keep heterogeneous values and all annotation fields losslessly as JSON.

    Arrow/Parquet cannot mix string, list and object values in one field. The
    containing numbers column remains a native list of structs, with searchable
    number/type/offset fields and reversible JSON for the variable payload.
    """
    try:
        spans = preprocessing.locate_number_spans(sentence, response["numbers"])
    except (TypeError, ValueError, KeyError):
        spans = [(None, None, info) for info in response["numbers"]]
    spoken = {(r["start"], r["end"]): r["spoken_form"] for r in replacements}
    rows = []
    for start, end, info in spans:
        rows.append({
            "number": str(info["number"]) if info.get("number") is not None else None,
            "type": str(info["type"]) if info.get("type") is not None else None,
            "subtype": str(info["subtype"]) if info.get("subtype") is not None else None,
            "value_json": json.dumps(info.get("value"), ensure_ascii=False),
            "start": start, "end": end,
            "spoken_form": spoken.get((start, end)),
            "annotation_json": json.dumps(info, ensure_ascii=False),
        })
    return rows


def load_preprocess_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "preprocess_dataset_numbers", PREPROCESS_MODULE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load preprocessing module: {PREPROCESS_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_run_mapping(value: str) -> tuple[str, str]:
    split, separator, run_name = value.partition("=")
    if not separator or not split or not run_name:
        raise argparse.ArgumentTypeError("expected SPLIT=RUN_NAME")
    return split, run_name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_parquet(table, output_path: Path) -> None:
    import pyarrow.parquet as parquet

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        parquet.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def load_results_and_failures(run_dir: Path, preprocessing) -> tuple[dict, dict]:
    """Load valid responses while retaining IDs of explicitly failed responses."""
    results_path = run_dir / "results.json"
    checkpoint_path = run_dir / "results_checkpoint.jsonl"
    inference_path = results_path if results_path.is_file() else checkpoint_path
    if not inference_path.is_file():
        raise FileNotFoundError(f"Inference output not found: {inference_path}")

    if inference_path.suffix == ".jsonl":
        entries = []
        with inference_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid checkpoint JSON at {inference_path}:{line_number}: {exc}"
                    ) from exc
    else:
        entries = json.loads(inference_path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"Results must be a JSON list: {inference_path}")

    results = {}
    failures = {}
    for entry_number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or "id" not in entry or "response" not in entry:
            raise ValueError(f"Invalid inference entry {entry_number} in {inference_path}")
        prompt_id = entry["id"]
        if prompt_id in results or prompt_id in failures:
            raise ValueError(f"Duplicate inference ID: {prompt_id}")
        try:
            results[prompt_id] = preprocessing.parse_llm_response(entry["response"])
        except (TypeError, ValueError) as exc:
            failures[prompt_id] = {
                "prompt_id": prompt_id,
                "stage": "inference_response",
                "error": f"{type(exc).__name__}: {exc}",
            }

    failure_path = run_dir / "results_failures.jsonl"
    if failure_path.is_file():
        with failure_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    failure = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid failure JSON at {failure_path}:{line_number}: {exc}"
                    ) from exc
                prompt_id = failure.get("id")
                if not prompt_id:
                    raise ValueError(f"Failure has no ID at {failure_path}:{line_number}")
                failures[prompt_id] = {
                    "prompt_id": prompt_id,
                    "stage": "inference_response",
                    "error": failure.get("error", "inference response was skipped"),
                }
    return results, failures


def build_split(
    *,
    split: str,
    run_name: str,
    runs_dir: Path,
    output_dir: Path,
    dialect: str,
    overwrite: bool,
    preprocessing,
    fallback_dialect: str = "ch_sg",
) -> tuple[Path, dict, list[dict]]:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    run_dir = runs_dir / run_name
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("inputs", [])) != 1:
        raise ValueError(f"Run {run_name!r} must contain exactly one input file")
    input_info = manifest["inputs"][0]
    input_path = Path(input_info["path"])
    if not input_path.is_file():
        raise FileNotFoundError(f"Source split not found: {input_path}")
    if input_path.suffix.lower() != ".parquet":
        raise ValueError(f"Source split must be Parquet: {input_path}")
    if sha256_file(input_path) != input_info["sha256"]:
        raise ValueError(f"Source split changed after run preparation: {input_path}")

    results, failed_results = load_results_and_failures(run_dir, preprocessing)
    prompt_rows = manifest["prompt_rows"]
    expected_ids = {row["prompt_id"] for row in prompt_rows}
    accounted_ids = set(results) | set(failed_results)
    missing_ids = expected_ids - accounted_ids
    unexpected_ids = accounted_ids - expected_ids
    if missing_ids or unexpected_ids:
        raise ValueError(
            f"Run {run_name!r} is incomplete or mismatched: "
            f"{len(accounted_ids)}/{len(expected_ids)} prompts accounted for "
            f"({len(missing_ids)} missing, {len(unexpected_ids)} unexpected)"
        )

    table = parquet.read_table(input_path)
    sentence_column = manifest["sentence_ch_column"]
    if sentence_column not in table.column_names:
        raise ValueError(f"Source split has no {sentence_column!r} column")

    row_to_prompt = {row["row_index"]: row["prompt_id"] for row in prompt_rows}
    sentences = table.column(sentence_column).to_pylist()
    dialect_columns = {
        column: table.column(column).to_pylist()
        for column in ("dialect", "dialect_fine") if column in table.column_names
    }
    converter = preprocessing.load_number_converter()
    written_sentences = []
    annotations = []
    number_dialects = []
    dialect_sources = []
    statuses = []
    errors = []
    changed_rows = 0
    conversion_failures = []
    for row_index, sentence_ch in enumerate(sentences):
        if not isinstance(sentence_ch, str):
            raise ValueError(
                f"Non-string {sentence_column!r} in split {split!r}, row {row_index + 1}"
            )
        prompt_id = row_to_prompt.get(row_index)
        row_dialect, dialect_source = resolve_dialect(
            {column: values[row_index] for column, values in dialect_columns.items()},
            dialect, fallback_dialect,
        )
        response = None
        replacements = []
        error = None
        if prompt_id is None:
            if preprocessing.DIGIT_RE.search(sentence_ch):
                raise ValueError(f"Unlabelled numeric source row: {split}:{row_index + 1}")
            written = sentence_ch
            status = "no_numbers"
        elif prompt_id in failed_results:
            written = None
            status = "inference_error"
            error = failed_results[prompt_id]["error"]
            conversion_failures.append(
                {
                    "split": split,
                    "run_name": run_name,
                    "row": row_index + 1,
                    **failed_results[prompt_id],
                }
            )
        else:
            response = deepcopy(results[prompt_id])
            try:
                written, replacements = preprocessing.apply_annotations(
                    sentence_ch,
                    response,
                    converter=converter,
                    dialect=row_dialect,
                    require_response_text_match=False,
                )
                if preprocessing.DIGIT_RE.search(written):
                    raise ValueError("Conversion left numeric digits in the written text")
                status = "converted"
            except Exception as exc:
                written = None
                replacements = []
                status = "conversion_error"
                error = f"{type(exc).__name__}: {exc}"
                conversion_failures.append(
                    {
                        "split": split,
                        "run_name": run_name,
                        "row": row_index + 1,
                        "prompt_id": prompt_id,
                        "stage": "number_conversion",
                        "error": error,
                    }
                )
        written_sentences.append(written)
        annotations.append(
            export_annotations(response, sentence_ch, preprocessing, replacements)
            if response is not None else (None if error else [])
        )
        number_dialects.append(row_dialect if prompt_id is not None else None)
        dialect_sources.append(dialect_source if prompt_id is not None else None)
        statuses.append(status)
        errors.append(error)
        changed_rows += written is not None and written != sentence_ch

    output_path = output_dir / "data" / f"{split}-00000-of-00001.parquet"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {output_path}")
    # Drop stale Hugging Face feature metadata from the source schema. The
    # datasets library recreates it, including the new column, during upload.
    table = table.replace_schema_metadata(None)
    table = table.drop([column for column in ADDED_COLUMNS if column in table.column_names])
    table = table.append_column(OUTPUT_COLUMN, pa.array(written_sentences, type=pa.string()))
    table = table.append_column("numbers", pa.array(annotations, type=annotation_type(pa)))
    for column, values in (
        ("number_dialect", number_dialects), ("number_dialect_source", dialect_sources),
        ("number_processing_status", statuses), ("number_processing_error", errors),
    ):
        table = table.append_column(column, pa.array(values, type=pa.string()))
    atomic_write_parquet(table, output_path)
    summary = {
        "split": split,
        "run_name": run_name,
        "source": str(input_path),
        "output": str(output_path),
        "rows": table.num_rows,
        "labeled_rows": len(prompt_rows),
        "changed_rows": changed_rows,
        "null_rows": len(conversion_failures),
        "dialect": dialect,
        "fallback_dialect": fallback_dialect,
        "number_dialects": dict(Counter(d for d in number_dialects if d)),
        "dialect_sources": dict(Counter(d for d in dialect_sources if d)),
        "statuses": dict(Counter(statuses)),
    }
    return output_path, summary, conversion_failures


def publish(paths: dict[str, Path], repo_id: str, token: str | None) -> None:
    from datasets import Dataset, DatasetDict

    dataset = DatasetDict(
        {split: Dataset.from_parquet(str(path)) for split, path in paths.items()}
    )
    dataset.push_to_hub(repo_id, token=token)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_mapping,
        metavar="SPLIT=RUN_NAME",
        help="Override split/run mappings; repeat for multiple splits",
    )
    parser.add_argument(
        "--dialect", choices=("auto", *SUPPORTED_DIALECTS), default="auto",
        help="Select each row's regional dialect automatically, or force one converter",
    )
    parser.add_argument(
        "--fallback-dialect", choices=SUPPORTED_DIALECTS, default="ch_sg",
        help="Explicit fallback for generic/unsupported regions (recorded per row)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Publish all four staged splits to Hugging Face after validation",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (normally use cached login or HF_TOKEN instead)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_compute_node("written-number dataset publishing")
    mappings = dict(args.run) if args.run else dict(DEFAULT_RUNS)
    if len(mappings) != len(args.run or mappings):
        raise ValueError("Each split may be specified only once")
    if args.push and set(mappings) != set(DEFAULT_RUNS):
        raise ValueError(
            "--push requires train, validation, test, and swiss_german together"
        )

    runs_dir = args.runs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    preprocessing = load_preprocess_module()
    paths = {}
    summaries = []
    all_failures = []
    for split, run_name in mappings.items():
        print(f"Building {split!r} from run {run_name!r} ...")
        path, summary, failures = build_split(
            split=split,
            run_name=run_name,
            runs_dir=runs_dir,
            output_dir=output_dir,
            dialect=args.dialect,
            overwrite=args.overwrite,
            preprocessing=preprocessing,
            fallback_dialect=args.fallback_dialect,
        )
        paths[split] = path
        summaries.append(summary)
        all_failures.extend(failures)
        print(
            f"  wrote {summary['rows']} rows; "
            f"changed {summary['changed_rows']} number-containing rows; "
            f"null {summary['null_rows']} failed rows"
        )

    summary_path = output_dir / "build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Build summary: {summary_path}")
    failures_path = output_dir / "conversion_failures.jsonl"
    with failures_path.open("w", encoding="utf-8") as handle:
        for failure in all_failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    print(f"Conversion failures: {failures_path} ({len(all_failures)})")

    if args.push:
        print(f"Publishing {len(paths)} splits to {args.repo_id} ...")
        publish(paths, args.repo_id, args.token)
        print(f"Published: https://huggingface.co/datasets/{args.repo_id}")
    else:
        print("Local build complete. Add --push to publish after reviewing it.")


if __name__ == "__main__":
    main()
