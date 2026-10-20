#!/usr/bin/env python3
"""Translate a prepared number-labeling CSV from German to Swiss German.

The input CSV must contain ``id``, ``client_id``, ``sentence_de``, and
``sentence_ch`` columns. Existing ``sentence_ch`` values are replaced with T5
translations and the output adds a ``placeholders_preserved`` column. The
output CSV is resumable: if it already exists, its completed rows are validated
against the input and skipped.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline_runtime import require_compute_node, storage_path


REQUIRED_COLUMNS = {"id", "client_id", "sentence_de", "sentence_ch"}
PLACEHOLDER_FLAG_COLUMN = "placeholders_preserved"
PLACEHOLDER_RE = re.compile(r"x[a-z]{1,2}x")


def default_output_path(input_path: Path) -> Path:
    run_name = input_path.stem.removesuffix("_prepared")
    return storage_path("intermediate", "translated", f"{run_name}_translated.csv")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns {sorted(missing)}: {path}")
        return list(reader), fieldnames


def completed_row_count(output_path: Path, input_rows: list[dict[str, str]]) -> int:
    """Validate an existing output as an input prefix and return its row count."""
    if not output_path.exists():
        return 0

    output_rows, output_fieldnames = read_csv(output_path)
    if PLACEHOLDER_FLAG_COLUMN not in output_fieldnames:
        raise ValueError(
            f"Existing output is missing {PLACEHOLDER_FLAG_COLUMN!r}. Choose a new "
            "--output path or remove the old output before rerunning."
        )
    if len(output_rows) > len(input_rows):
        raise ValueError("Output has more rows than the input CSV")

    for index, (output_row, input_row) in enumerate(zip(output_rows, input_rows)):
        for column in ("id", "client_id", "sentence_de"):
            if output_row[column] != input_row[column]:
                raise ValueError(
                    f"Existing output differs from input at row {index + 1}, column "
                    f"{column!r}. Choose a new --output path or remove the invalid output."
                )
    return len(output_rows)


def placeholders_preserved(source: str, translation: str) -> bool:
    """Return whether all placeholders occur once, in their original order."""
    source_placeholders = PLACEHOLDER_RE.findall(source)
    translation_placeholders = PLACEHOLDER_RE.findall(translation)
    return source_placeholders == translation_placeholders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Prepared CSV to translate")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Translated CSV path (default: <input>_translated.csv)",
    )
    parser.add_argument("--model", type=Path, required=True, help="Local T5 checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=16, help="T5 inference batch size")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()
    require_compute_node("T5 number-data translation")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not args.input_csv.is_file():
        parser.error(f"Input CSV not found: {args.input_csv}")
    if not args.model.is_dir():
        parser.error(f"T5 checkpoint directory not found: {args.model}")

    input_rows, fieldnames = read_csv(args.input_csv)
    output_fieldnames = fieldnames + [PLACEHOLDER_FLAG_COLUMN]
    output_path = args.output or default_output_path(args.input_csv)
    done = completed_row_count(output_path, input_rows)
    print(f"Input rows: {len(input_rows)}; already translated: {done}")
    if done == len(input_rows):
        print(f"Output is already complete: {output_path}")
        return

    from t5_adapter import T5Translator

    print(f"Loading T5 checkpoint: {args.model}")
    translator = T5Translator(args.model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_fieldnames)
        if write_header:
            writer.writeheader()

        for batch_start in range(done, len(input_rows), args.batch_size):
            batch_rows = input_rows[batch_start:batch_start + args.batch_size]
            translations = translator.translate(
                [row["sentence_de"] for row in batch_rows],
                dialect=[row["client_id"] for row in batch_rows],
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            if not isinstance(translations, list) or len(translations) != len(batch_rows):
                raise RuntimeError("T5 translator returned an unexpected number of translations")

            output_rows = []
            for row, translation in zip(batch_rows, translations):
                preserved = placeholders_preserved(row["sentence_de"], translation)
                output_rows.append({
                    **row,
                    "sentence_ch": translation,
                    PLACEHOLDER_FLAG_COLUMN: "true" if preserved else "false",
                })

            writer.writerows(output_rows)
            fh.flush()
            completed = batch_start + len(batch_rows)
            print(f"Saved {completed}/{len(input_rows)} rows")

    print(f"Translated CSV written to: {output_path}")


if __name__ == "__main__":
    main()
