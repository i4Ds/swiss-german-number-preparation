#!/usr/bin/env python3
"""Prepare LLM number labels and apply Swiss German written number forms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_runtime import require_compute_node, storage_path

COLLECTION_ROOT = PROJECT_ROOT.parent / "collect_de_num_data"
LLM_SCRIPT = COLLECTION_ROOT / "scripts" / "labeling" / "run_llm_inference.py"
PROMPT_MODULE_DIR = COLLECTION_ROOT / "scripts" / "labeling"
CONVERTER_MODULE_DIR = PROMPT_MODULE_DIR
DEFAULT_CONFIG_TEMPLATE = PROJECT_ROOT / "configs" / "llm_inference.yaml"
DEFAULT_RUNS_DIR = storage_path("intermediate", "runs")
DEFAULT_PROCESSED_DIR = storage_path("processed")
SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet"}
DIGIT_RE = re.compile(r"\d")
PLAIN_SHORT_INTEGER_RE = re.compile(r"[1-9]\d{0,3}")
DATE_SINGLE_NUMBER_RE = re.compile(r"(\d{1,4})\.?")
IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
DECADE_YEAR_RE = re.compile(r"((?:18|19|20)\d{2})(?=er(?:\b|[-_]))", re.IGNORECASE)
COORDINATE_RE = re.compile(
    r"\s*(\d{1,3})\s*°\s*(?:(\d{1,2})\s*['′’”\"]\s*)?"
    r"(?:(\d{1,2})\s*(?:\"|″|”)\s*)?([NSEOWeow])?\s*"
)
RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
MONTH_WORDS = {
    "januar": 1,
    "februar": 2,
    "märz": 3,
    "april": 4,
    "mai": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "septämber": 9,
    "oktober": 10,
    "oktobor": 10,
    "november": 11,
    "novämber": 11,
    "novämbor": 11,
    "dezember": 12,
    "dezämber": 12,
    "dezämbor": 12,
}


@dataclass
class FileStats:
    rows: int = 0
    rows_sent_to_llm: int = 0
    changed_rows: int = 0
    unchanged_rows: int = 0
    rows_with_remaining_digits: int = 0
    errors: int = 0
    rows_without_numeric_digits_removed: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Create paired-sentence prompts and a resumable LLM run folder",
    )
    prepare.add_argument("inputs", nargs="+", type=Path)
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    prepare.add_argument("--sentence-ch-column", default="sentence_ch")
    prepare.add_argument("--sentence-de-column", default="sentence_de")
    prepare.add_argument("--id-column", default="id")
    prepare.add_argument(
        "--require-digits-in-both",
        action="store_true",
        help="Only select rows where both paired sentence columns contain a digit",
    )
    prepare.add_argument(
        "--filter-column",
        help="Only select rows whose value in this column matches --filter-value",
    )
    prepare.add_argument("--filter-value")
    prepare.add_argument(
        "--selected-rows-only",
        action="store_true",
        help="Write only selected rows during apply instead of retaining all input rows",
    )
    prepare.add_argument(
        "--config-template",
        type=Path,
        default=DEFAULT_CONFIG_TEMPLATE,
    )
    prepare.add_argument(
        "--overwrite-setup",
        action="store_true",
        help="Replace prompts/manifest/config only when no checkpoint exists",
    )

    apply = subparsers.add_parser(
        "apply",
        help="Apply completed LLM labels with convert_ch_numbers.py",
    )
    apply.add_argument("--run-name", required=True)
    apply.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    apply.add_argument("--results", type=Path, default=None)
    apply.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    apply.add_argument(
        "--dialect", choices=("ch_sg", "ch_bs", "ch_zh", "ch_lu", "ch_be"), default="ch_sg"
    )
    apply.add_argument(
        "--on-error",
        choices=("fail", "keep"),
        default="fail",
        help="Fail atomically or retain and audit a row on conversion errors",
    )
    apply.add_argument(
        "--remaining-digits",
        choices=("error", "report", "allow"),
        default="error",
        help="Policy for digits left after labeled conversion (default: error)",
    )
    apply.add_argument(
        "--issue-flag-column",
        default="number_processing_issue",
        help="Output column containing a boolean processing issue flag",
    )
    apply.add_argument(
        "--issue-reason-column",
        default="number_processing_issue_reason",
        help="Output column containing the processing issue reason",
    )
    apply.add_argument("--overwrite", action="store_true")

    postprocess = subparsers.add_parser(
        "postprocess-results",
        help="Normalize classifications inside an existing inference results JSON",
    )
    postprocess.add_argument("results", type=Path)
    postprocess.add_argument(
        "--coordinates-only",
        action="store_true",
        help="Only repair OTHER_NUMERIC geographic coordinates",
    )
    return parser


def validate_run_name(run_name: str) -> None:
    if RUN_NAME_RE.fullmatch(run_name) is None:
        raise ValueError(
            "Run name must contain only letters, digits, underscores, and hyphens"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: Path) -> tuple[list[dict], list[str] | None]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            return list(reader), list(reader.fieldnames)

    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            records = json.load(handle)
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ValueError(f"JSON must contain a list of objects: {path}")
        return records, None

    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL record must be an object at {path}:{line_number}"
                    )
                records.append(record)
        return records, None

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires pyarrow. Install it in the active environment."
            ) from exc
        return parquet.read_table(path).to_pylist(), None

    raise ValueError(f"Unsupported dataset extension: {path}")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class AtomicJsonArrayWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
        )
        self.path = path
        self.temporary_path = Path(temporary_name)
        self.handle = os.fdopen(descriptor, "w", encoding="utf-8")
        self.first = True

    def __enter__(self) -> "AtomicJsonArrayWriter":
        self.handle.write("[\n")
        return self

    def append(self, value) -> None:
        if not self.first:
            self.handle.write(",\n")
        json.dump(value, self.handle, ensure_ascii=False, indent=2)
        self.first = False

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc_type is None:
                self.handle.write("\n]\n")
                self.handle.close()
                os.replace(self.temporary_path, self.path)
            else:
                self.handle.close()
                self.temporary_path.unlink(missing_ok=True)
        except Exception:
            self.temporary_path.unlink(missing_ok=True)
            raise


def atomic_write_dataset(
    path: Path,
    records: list[dict],
    csv_fieldnames: list[str] | None,
    parquet_schema=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        if path.suffix.lower() == ".parquet":
            os.close(descriptor)
            try:
                import pyarrow as pa
                import pyarrow.parquet as parquet
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet output requires pyarrow. Install it in the active environment."
                ) from exc
            table = pa.Table.from_pylist(records, schema=parquet_schema)
            parquet.write_table(table, temporary_path, compression="zstd")
        else:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                if path.suffix.lower() == ".csv":
                    if csv_fieldnames is None:
                        raise ValueError("CSV fieldnames are missing")
                    writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
                    writer.writeheader()
                    writer.writerows(records)
                elif path.suffix.lower() == ".json":
                    json.dump(records, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                elif path.suffix.lower() == ".jsonl":
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    raise ValueError(f"Unsupported output extension: {path}")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def load_system_prompt() -> str:
    if str(PROMPT_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(PROMPT_MODULE_DIR))
    from generate_prompts import SYSTEM_PROMPT

    return SYSTEM_PROMPT + """

─── PAIRED SWISS GERMAN DATASET MODE ─────────────────────────────────────────
The user supplies a Swiss German target sentence and a paired Standard German
reference. Apply every rule above to the SWISS GERMAN TARGET. The Standard
German sentence is context only: use it to disambiguate meaning, number type,
structured value, grammatical case, and declension.

The output field "text" must reproduce the Swiss German sentence verbatim.
Every "number" must be copied verbatim from the Swiss German sentence, not from
the Standard German reference. If corresponding numeric expressions have
different surface forms, copy the Swiss German span but use the pair to infer
its meaning and structured value. Never annotate a number that occurs only in
the Standard German reference.

Annotate every non-overlapping numeric expression containing a digit in the
Swiss German sentence exactly once. Pre-detected hints are recall aids only;
they may include punctuation or combine multiple expressions and do not define
annotation boundaries or types. Do not translate, correct, or normalize either
sentence. Before responding, verify that every digit in the Swiss German target
belongs to exactly one annotation span and that the response is valid JSON.
"""


def coarse_number_hints(text: str) -> list[dict[str, str]]:
    """Provide whitespace-token hints; the LLM remains responsible for typing."""
    hints = []
    for token in text.split():
        if DIGIT_RE.search(token):
            hints.append({"number": token, "type": "UNKNOWN"})
    return hints


def paired_user_message(sentence_ch: str, sentence_de: str, hints: list[dict]) -> str:
    return (
        "Swiss German sentence to annotate: "
        f"{json.dumps(sentence_ch, ensure_ascii=False)}\n"
        "Paired Standard German reference: "
        f"{json.dumps(sentence_de, ensure_ascii=False)}\n"
        "Pre-detected number hints (may be incomplete and are not typed): "
        f"{json.dumps(hints, ensure_ascii=False)}"
    )


def prepare_run(args: argparse.Namespace) -> None:
    validate_run_name(args.run_name)
    if bool(args.filter_column) != bool(args.filter_value):
        raise ValueError("--filter-column and --filter-value must be supplied together")
    run_dir = (args.runs_dir / args.run_name).expanduser().resolve()
    prompts_path = run_dir / "prompts.json"
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "results_checkpoint.jsonl"
    results_path = run_dir / "results.json"

    if checkpoint_path.exists() or results_path.exists():
        raise FileExistsError(
            f"Run {args.run_name!r} already has inference output; use a new run name"
        )
    setup_paths = (prompts_path, manifest_path, config_path)
    existing_setup = [path for path in setup_paths if path.exists()]
    if existing_setup and not args.overwrite_setup:
        raise FileExistsError(
            "Setup already exists (use --overwrite-setup): "
            + ", ".join(str(path) for path in existing_setup)
        )

    config_template = args.config_template.expanduser().resolve()
    if not config_template.is_file():
        raise FileNotFoundError(f"LLM config template not found: {config_template}")
    system_prompt = load_system_prompt()
    manifest_inputs = []
    prompt_rows = []
    selected_rows = []
    prompt_count = 0

    with AtomicJsonArrayWriter(prompts_path) as prompts_writer:
        for input_index, raw_path in enumerate(args.inputs):
            input_path = raw_path.expanduser().resolve()
            if not input_path.is_file():
                raise FileNotFoundError(f"Input dataset not found: {input_path}")
            if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"Unsupported input dataset: {input_path}")
            records, csv_fieldnames = read_records(input_path)

            required_columns = [args.sentence_ch_column, args.sentence_de_column]
            if args.filter_column:
                required_columns.append(args.filter_column)
            for required_column in required_columns:
                if csv_fieldnames is not None and required_column not in csv_fieldnames:
                    raise ValueError(f"CSV is missing column {required_column!r}: {input_path}")

            manifest_inputs.append(
                {
                    "path": str(input_path),
                    "sha256": sha256_file(input_path),
                    "suffix": input_path.suffix.lower(),
                    "csv_fieldnames": csv_fieldnames,
                }
            )

            for row_index, record in enumerate(records):
                try:
                    sentence_ch = record[args.sentence_ch_column]
                    sentence_de = record[args.sentence_de_column]
                except KeyError as exc:
                    raise ValueError(
                        f"Row {row_index + 1} in {input_path} is missing {exc.args[0]!r}"
                    ) from exc
                if not isinstance(sentence_ch, str) or not isinstance(sentence_de, str):
                    raise ValueError(
                        f"Paired sentence fields must be strings in {input_path}, "
                        f"row {row_index + 1}"
                    )
                if args.filter_column and record.get(args.filter_column) != args.filter_value:
                    continue
                if not DIGIT_RE.search(sentence_ch):
                    continue
                if args.require_digits_in_both and not DIGIT_RE.search(sentence_de):
                    continue

                selected_rows.append(
                    {"input_index": input_index, "row_index": row_index}
                )

                prompt_id = (
                    f"{args.run_name}__file{input_index:03d}__row{row_index:09d}"
                )
                hints = coarse_number_hints(sentence_ch)
                prompts_writer.append(
                    {
                        "id": prompt_id,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": paired_user_message(
                                    sentence_ch, sentence_de, hints
                                ),
                            },
                        ],
                    }
                )
                prompt_count += 1
                prompt_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "input_index": input_index,
                        "row_index": row_index,
                        "source_id": record.get(args.id_column),
                    }
                )

    manifest = {
        "version": 1,
        "run_name": args.run_name,
        "sentence_ch_column": args.sentence_ch_column,
        "sentence_de_column": args.sentence_de_column,
        "id_column": args.id_column,
        "selection": {
            "require_digits_in_both": args.require_digits_in_both,
            "filter_column": args.filter_column,
            "filter_value": args.filter_value,
            "selected_rows_only": args.selected_rows_only,
        },
        "inputs": manifest_inputs,
        "prompt_rows": prompt_rows,
        "selected_rows": selected_rows,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)
    atomic_write_text(config_path, config_template.read_text(encoding="utf-8"))

    print(f"Run directory: {run_dir}")
    print(f"Input datasets: {len(manifest_inputs)}")
    print(f"Rows sent to the LLM: {prompt_count}")
    print("\nRun type detection with:")
    print(
        f"python {LLM_SCRIPT} --run-name {args.run_name} "
        f"--config-path {config_path} --prompts-path {prompts_path} "
        f"--checkpoint-path {checkpoint_path} --results-path {results_path}"
    )


def parse_llm_response(raw_response: str) -> dict:
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        response = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(response, dict):
        raise ValueError("LLM response must be a JSON object")
    if not isinstance(response.get("numbers"), list):
        raise ValueError("LLM response has no numbers list")
    normalize_number_annotations(response)
    return response


def _coordinate_value(number_text: str) -> dict[str, str] | None:
    match = COORDINATE_RE.fullmatch(number_text)
    if match is None:
        return None
    degrees, minutes, seconds, direction = match.groups()
    return {
        "degrees": degrees or "",
        "minutes": minutes or "",
        "seconds": seconds or "",
        "direction": (direction or "").upper(),
    }


def _normalization_start(text: str, annotation: dict) -> int | None:
    """Validate an existing span before changing it; retain offset-free inputs."""
    if "start" not in annotation and "end" not in annotation:
        return None
    start, end = annotation.get("start"), annotation.get("end")
    if (
        type(start) is not int
        or type(end) is not int
        or not isinstance(text, str)
        or not 0 <= start < end <= len(text)
    ):
        raise ValueError(f"Invalid offsets for annotation {annotation['number']!r}")
    if text[start:end] != annotation["number"]:
        raise ValueError(f"Offsets do not select annotation {annotation['number']!r}")
    return start


def _append_standalone_coordinate_direction(
    text: str, number_text: str, value: dict[str, str], *, start: int | None = None
) -> str:
    """Include an adjacent N/S/E/O/W letter without consuming direction words."""
    if value["direction"]:
        return number_text
    if start is None:
        start = text.find(number_text)
    if start < 0:
        return number_text
    suffix = text[start + len(number_text) :]
    match = re.match(r"(\s+)([NSEOWeow])(?![A-Za-zÄÖÜäöüß])", suffix)
    if match is None:
        return number_text
    value["direction"] = match.group(2).upper()
    return number_text + match.group(1) + match.group(2)


def normalize_number_annotations(
    response: dict, *, coordinates_only: bool = False
) -> dict[str, int]:
    """Apply deterministic repairs after LLM classification and before conversion."""
    numbers = response.get("numbers", [])
    text = response.get("text", "")
    normalized = []
    stats = {
        "coordinates": 0,
        "versions": 0,
        "dates": 0,
        "date_values_clipped": 0,
        "cardinals": 0,
        "product_ids": 0,
        "digitless_removed": 0,
    }

    for annotation in numbers:
        if not isinstance(annotation, dict):
            normalized.append(annotation)
            continue
        number_text = annotation.get("number")
        if not isinstance(number_text, str):
            normalized.append(annotation)
            continue

        if annotation.get("type") == "OTHER_NUMERIC" and "°" in number_text:
            coordinate_value = _coordinate_value(number_text)
            if coordinate_value is not None:
                start = _normalization_start(text, annotation)
                annotation["number"] = _append_standalone_coordinate_direction(
                    text, number_text, coordinate_value, start=start
                )
                if start is not None:
                    annotation["end"] = start + len(annotation["number"])
                annotation["value"] = coordinate_value
                annotation["type"] = "COORDINATE"
                annotation.pop("subtype", None)
                stats["coordinates"] += 1
                normalized.append(annotation)
                continue

        if not coordinates_only and annotation.get("type") in {
            "OTHER_NUMERIC",
            "NUMBER_WITH_SUFFIX",
        }:
            decade_match = DECADE_YEAR_RE.match(number_text)
        else:
            decade_match = None
        if decade_match is not None:
            start = _normalization_start(text, annotation)
            year = decade_match.group(1)
            annotation["number"] = year
            if start is not None:
                annotation["end"] = start + len(year)
            annotation["value"] = ["", "", year]
            annotation["type"] = "DATE"
            annotation.pop("subtype", None)
            stats["dates"] += 1
            normalized.append(annotation)
            continue

        # The LLM sometimes annotates only one numeric date component (for
        # example ``14.`` or ``1900`` in ``14. Juni 1900``) but copies the
        # adjacent month into the structured DATE value.  The written month
        # remains in the sentence, so verbalizing the inferred component here
        # would produce ``vierzähnte Juni Juni ...``.  A conversion may emit
        # only date components contained in its own surface span.
        if (
            not coordinates_only
            and annotation.get("type") == "DATE"
            and isinstance(annotation.get("value"), list)
            and len(annotation["value"]) == 3
        ):
            single_number = DATE_SINGLE_NUMBER_RE.fullmatch(number_text)
            if single_number is not None:
                surface_value = single_number.group(1)
                day, _, year = annotation["value"]
                if year not in (None, "") and str(year) == surface_value:
                    clipped_value = ["", "", str(year)]
                elif day not in (None, "") and str(day).lstrip("0") == surface_value.lstrip("0"):
                    clipped_value = [str(day), "", ""]
                else:
                    clipped_value = None
                if clipped_value is not None and annotation["value"] != clipped_value:
                    annotation["value"] = clipped_value
                    stats["date_values_clipped"] += 1

        if coordinates_only or annotation.get("type") != "OTHER_NUMERIC":
            normalized.append(annotation)
            continue

        if not DIGIT_RE.search(number_text):
            stats["digitless_removed"] += 1
            continue
        elif IPV4_RE.fullmatch(number_text):
            annotation["value"] = number_text
            annotation["type"] = "VERSION"
            annotation.pop("subtype", None)
            stats["versions"] += 1
        elif PLAIN_SHORT_INTEGER_RE.fullmatch(number_text):
            annotation["value"] = number_text
            annotation["type"] = "NUMBER_CARDINAL"
            annotation.pop("subtype", None)
            stats["cardinals"] += 1
        else:
            annotation["value"] = number_text
            annotation["type"] = "PRODUCT_ID"
            annotation.pop("subtype", None)
            stats["product_ids"] += 1
        normalized.append(annotation)

    response["numbers"] = normalized
    return stats


def postprocess_results_file(path: Path, *, coordinates_only: bool = False) -> dict[str, int]:
    """Normalize parseable responses in-place while preserving malformed responses."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Inference results not found: {path}")
    with path.open(encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise ValueError("Results JSON must contain a list")

    totals = {
        "coordinates": 0,
        "versions": 0,
        "dates": 0,
        "date_values_clipped": 0,
        "cardinals": 0,
        "product_ids": 0,
        "digitless_removed": 0,
        "malformed_responses_skipped": 0,
    }
    for entry in entries:
        try:
            response = json.loads(entry["response"])
        except (KeyError, TypeError, json.JSONDecodeError):
            totals["malformed_responses_skipped"] += 1
            continue
        stats = normalize_number_annotations(
            response, coordinates_only=coordinates_only
        )
        for key, count in stats.items():
            totals[key] += count
        entry["response"] = json.dumps(response, ensure_ascii=False, indent=2)

    atomic_write_json(path, entries)
    return totals


def load_result_entries(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Inference results not found: {path}")
    if path.suffix.lower() == ".jsonl":
        entries = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid result JSONL at line {line_number}: {exc}"
                    ) from exc
    else:
        with path.open(encoding="utf-8") as handle:
            entries = json.load(handle)
        if not isinstance(entries, list):
            raise ValueError("Results JSON must contain a list")

    results = {}
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or "id" not in entry or "response" not in entry:
            raise ValueError(f"Invalid result entry {index}")
        prompt_id = entry["id"]
        if prompt_id in results:
            raise ValueError(f"Duplicate result ID: {prompt_id}")
        results[prompt_id] = parse_llm_response(entry["response"])
    return results


def load_number_converter() -> Callable[[dict, str], str]:
    if str(CONVERTER_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(CONVERTER_MODULE_DIR))
    from convert_ch_numbers import convert_number

    return convert_number


def locate_number_spans(text: str, numbers: list[dict]) -> list[tuple[int, int, dict]]:
    """Locate each verbatim annotation once without allowing overlap."""
    located: list[tuple[int, int, dict]] = []
    for annotation_index, number_info in enumerate(numbers, start=1):
        if not isinstance(number_info, dict):
            raise ValueError(f"Number annotation {annotation_index} is not an object")
        number_text = number_info.get("number")
        if not isinstance(number_text, str) or not number_text:
            raise ValueError(f"Number annotation {annotation_index} has no number text")

        raw_start, raw_end = number_info.get("start"), number_info.get("end")
        # LLM-reported offsets are frequently miscounted; treat them as a hint and
        # fall back to a verbatim search instead of failing the whole run on them.
        if (
            isinstance(raw_start, int)
            and isinstance(raw_end, int)
            and 0 <= raw_start < raw_end <= len(text)
            and text[raw_start:raw_end] == number_text
        ):
            candidates = [(raw_start, raw_end)]
        else:
            candidates = [
                (match.start(), match.end())
                for match in re.finditer(re.escape(number_text), text)
            ]
        candidates = [
            span for span in candidates
            if all(span[1] <= old_start or span[0] >= old_end
                   for old_start, old_end, _ in located)
        ]
        if not candidates:
            raise ValueError(
                f"Annotated number {number_text!r} was not found as an unused span"
            )

        cursor = max((end for _, end, _ in located), default=0)
        def candidate_rank(span: tuple[int, int]) -> tuple[int, int, int]:
            start, end = span
            boundaries = (
                int(start == 0 or not text[start - 1].isalnum())
                + int(end == len(text) or not text[end].isalnum())
            )
            return boundaries, int(start >= cursor), -start

        start, end = max(candidates, key=candidate_rank)
        if number_info.get("type") == "TIME" and not number_text.casefold().endswith("uhr"):
            following_uhr = re.match(r"\s+Uhr\b", text[end:], re.IGNORECASE)
            if following_uhr is not None:
                end += following_uhr.end()
                number_info["number"] = text[start:end]
                number_info["start"] = start
                number_info["end"] = end
        located.append((start, end, number_info))
    return sorted(located, key=lambda item: item[0])


def apply_annotations(
    sentence_ch: str,
    response: dict,
    *,
    converter: Callable[[dict, str], str],
    dialect: str,
    require_response_text_match: bool = True,
) -> tuple[str, list[dict]]:
    response_text = response.get("text")
    if require_response_text_match and response_text != sentence_ch:
        raise ValueError("LLM response text does not exactly match sentence_ch")
    numbers = response["numbers"]
    if DIGIT_RE.search(sentence_ch) and not numbers:
        raise ValueError("LLM returned no annotations for a sentence containing digits")

    spans = locate_number_spans(sentence_ch, numbers)
    uncovered_digits = [
        position for position, char in enumerate(sentence_ch)
        if char.isdigit()
        and not any(start <= position < end for start, end, _ in spans)
    ]
    if uncovered_digits:
        raise ValueError(
            f"unannotated digit at character offset {uncovered_digits[0]}"
        )
    replacements = []
    for start, end, number_info in spans:
        spoken = converter(number_info, dialect=dialect)
        if not spoken or str(spoken).strip().lower() == "unknown":
            raise ValueError(
                f"Unsupported conversion for {number_info.get('number')!r} "
                f"of type {number_info.get('type')!r}"
            )
        replacements.append(
            {
                "start": start,
                "end": end,
                "number": number_info.get("number"),
                "type": number_info.get("type"),
                "value": number_info.get("value"),
                "spoken_form": spoken,
            }
        )

    converted = sentence_ch
    for replacement in reversed(replacements):
        converted = (
            converted[: replacement["start"]]
            + replacement["spoken_form"]
            + converted[replacement["end"] :]
        )
    words = re.findall(r"[A-Za-zÄÖÜäöü]+", converted.casefold())
    for first, second in zip(words, words[1:]):
        month = MONTH_WORDS.get(first)
        if month is not None and MONTH_WORDS.get(second) == month:
            raise ValueError(
                f"conversion produced a duplicated month: {first!r} {second!r}"
            )
    return converted, replacements


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_numbers_written{input_path.suffix.lower()}"


def audit_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_audit.jsonl")


def parquet_schema_for(path: Path):
    """Load the source Arrow schema so processed parquet keeps nested columns."""
    if path.suffix.lower() != ".parquet":
        return None
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow. Install it in the active environment."
        ) from exc
    return parquet.read_schema(path)


def apply_run(args: argparse.Namespace) -> None:
    validate_run_name(args.run_name)
    run_dir = (args.runs_dir / args.run_name).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    default_results = run_dir / "results.json"
    checkpoint_results = run_dir / "results_checkpoint.jsonl"
    results_path = args.results.expanduser().resolve() if args.results else (
        default_results if default_results.is_file() else checkpoint_results
    )
    results = load_result_entries(results_path)
    converter = load_number_converter()
    prompt_rows = {row["prompt_id"]: row for row in manifest["prompt_rows"]}
    missing_results = set(prompt_rows) - set(results)
    if missing_results:
        raise ValueError(
            f"Inference is incomplete: {len(missing_results)} prompt results are missing"
        )

    output_dir = args.output_dir.expanduser().resolve() / args.run_name
    sentence_ch_column = manifest["sentence_ch_column"]
    id_column = manifest["id_column"]
    row_to_prompt = {
        (row["input_index"], row["row_index"]): row["prompt_id"]
        for row in manifest["prompt_rows"]
    }
    selection = manifest.get("selection", {})
    selected_rows_only = selection.get("selected_rows_only", False)
    selected_rows = {
        (row["input_index"], row["row_index"])
        for row in manifest.get("selected_rows", manifest["prompt_rows"])
    }
    pending_outputs = []
    seen_output_paths: set[Path] = set()
    combined_stats = FileStats()

    for input_index, input_info in enumerate(manifest["inputs"]):
        input_path = Path(input_info["path"])
        if not input_path.is_file():
            raise FileNotFoundError(f"Original input no longer exists: {input_path}")
        if sha256_file(input_path) != input_info["sha256"]:
            raise ValueError(f"Original input changed after prompt preparation: {input_path}")
        records, csv_fieldnames = read_records(input_path)
        parquet_schema = parquet_schema_for(input_path)
        output_path = output_path_for(input_path, output_dir)
        audit_path = audit_path_for(output_path)
        collisions = seen_output_paths.intersection((output_path, audit_path))
        if collisions:
            raise ValueError(
                "Multiple inputs resolve to the same output path: "
                + ", ".join(str(path) for path in sorted(collisions))
            )
        seen_output_paths.update((output_path, audit_path))
        if not args.overwrite:
            existing = [path for path in (output_path, audit_path) if path.exists()]
            if existing:
                raise FileExistsError(
                    "Output already exists (use --overwrite): "
                    + ", ".join(str(path) for path in existing)
                )

        processed = []
        audit = []
        stats = FileStats(rows=len(records))
        output_fieldnames = list(csv_fieldnames) if csv_fieldnames is not None else None
        if output_fieldnames is not None:
            for column in (args.issue_flag_column, args.issue_reason_column):
                if column not in output_fieldnames:
                    output_fieldnames.append(column)
        for row_index, source_record in enumerate(records):
            if selected_rows_only and (input_index, row_index) not in selected_rows:
                continue
            record = dict(source_record)
            sentence_ch = record[sentence_ch_column]
            if not DIGIT_RE.search(str(sentence_ch)):
                stats.rows_without_numeric_digits_removed += 1
                record[args.issue_flag_column] = False
                record[args.issue_reason_column] = ""
                processed.append(record)
                continue
            prompt_id = row_to_prompt.get((input_index, row_index))
            if prompt_id is None:
                processed.append(record)
                stats.unchanged_rows += 1
                record[args.issue_flag_column] = True
                record[args.issue_reason_column] = "missing_inference_result"
                continue

            stats.rows_sent_to_llm += 1
            error_message = None
            replacements = []
            try:
                converted, replacements = apply_annotations(
                    sentence_ch,
                    results[prompt_id],
                    converter=converter,
                    dialect=args.dialect,
                )
                remaining = bool(DIGIT_RE.search(converted))
                if remaining and args.remaining_digits == "error":
                    raise ValueError("digits remain after all labeled conversions")
            except Exception as exc:
                if args.on_error == "fail":
                    raise RuntimeError(
                        f"Failed input {input_path}, row {row_index + 1}, "
                        f"id={record.get(id_column)!r}: {exc}"
                    ) from exc
                converted = sentence_ch
                remaining = bool(DIGIT_RE.search(converted))
                error_message = f"{type(exc).__name__}: {exc}"
                stats.errors += 1

            if converted == sentence_ch:
                stats.unchanged_rows += 1
            else:
                stats.changed_rows += 1
            if remaining:
                stats.rows_with_remaining_digits += 1
            record[sentence_ch_column] = converted
            issue_reason = error_message or (
                "digits_remain_after_conversion" if remaining else ""
            )
            if not issue_reason and converted == sentence_ch:
                issue_reason = "numeric_text_unchanged"
            record[args.issue_flag_column] = bool(issue_reason)
            record[args.issue_reason_column] = issue_reason
            processed.append(record)
            audit_entry = {
                "row": row_index + 1,
                "id": record.get(id_column),
                "prompt_id": prompt_id,
                "original_sentence_ch": sentence_ch,
                "processed_sentence_ch": converted,
                "text_quality": results[prompt_id].get("text_quality"),
                "replacements": replacements,
                "remaining_digits": remaining,
            }
            if error_message:
                audit_entry["error"] = error_message
            audit.append(audit_entry)

        pending_outputs.append(
            (
                output_path,
                audit_path,
                processed,
                audit,
                output_fieldnames,
                parquet_schema,
                stats,
            )
        )
        for key, value in asdict(stats).items():
            setattr(combined_stats, key, getattr(combined_stats, key) + value)

    for (
        output_path,
        audit_path,
        processed,
        audit,
        fieldnames,
        parquet_schema,
        stats,
    ) in pending_outputs:
        atomic_write_dataset(
            output_path,
            processed,
            fieldnames,
            parquet_schema=parquet_schema,
        )
        atomic_write_dataset(audit_path, audit, None)
        print(f"\nOutput: {output_path}")
        print(f"Audit:  {audit_path}")
        for key, value in asdict(stats).items():
            print(f"  {key}: {value}")

    if len(pending_outputs) > 1:
        print("\nCombined totals")
        for key, value in asdict(combined_stats).items():
            print(f"  {key}: {value}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    require_compute_node("dataset number preprocessing")
    try:
        if args.command == "prepare":
            prepare_run(args)
        elif args.command == "apply":
            apply_run(args)
        else:
            stats = postprocess_results_file(
                args.results, coordinates_only=args.coordinates_only
            )
            print(f"Post-processed: {args.results.expanduser().resolve()}")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
