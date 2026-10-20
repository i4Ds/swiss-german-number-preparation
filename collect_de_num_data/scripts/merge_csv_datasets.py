#!/usr/bin/env python3
"""Concatenate compatible CSV datasets into one validated CSV file."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


UPPERCASE_UMLAUT_TRANSLATION = str.maketrans({"Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="CSV files to concatenate, in the desired output order",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to create the merged CSV",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Output filename; .csv is appended when omitted",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Column whose values must be non-empty and unique (default: id)",
    )
    parser.add_argument(
        "--text-column",
        default="sentence_ch",
        help="Column for the reported character count (default: sentence_ch)",
    )
    parser.add_argument(
        "--allow-duplicate-ids",
        action="store_true",
        help="Allow repeated values in the ID column",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    return parser.parse_args()


def output_path(output_dir: Path, output_name: str) -> Path:
    name = output_name if output_name.lower().endswith(".csv") else f"{output_name}.csv"
    if Path(name).name != name:
        raise ValueError("--output-name must be a filename, not a path")
    return output_dir.expanduser().resolve() / name


def merge_csvs(args: argparse.Namespace) -> Path:
    sources = [path.expanduser().resolve() for path in args.input_files]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files not found: {missing}")

    target = output_path(args.output_dir, args.output_name)
    if target in sources:
        raise ValueError("The output file cannot also be an input file")
    if target.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {target}. Pass --overwrite to replace it."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")

    fieldnames: list[str] | None = None
    seen_ids: set[str] = set()
    total_rows = 0
    character_count = 0
    uppercase_umlaut_count = 0
    per_file_counts: list[tuple[Path, int]] = []

    try:
        with temporary.open("x", encoding="utf-8", newline="") as output_file:
            writer: csv.DictWriter | None = None

            for source in sources:
                source_rows = 0
                with source.open(encoding="utf-8-sig", newline="") as input_file:
                    reader = csv.DictReader(input_file)
                    current_fields = reader.fieldnames
                    if not current_fields:
                        raise ValueError(f"Input has no CSV header: {source}")

                    if fieldnames is None:
                        fieldnames = current_fields
                        if args.id_column not in fieldnames:
                            raise ValueError(
                                f"ID column {args.id_column!r} is missing from {source}"
                            )
                        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                        writer.writeheader()
                    elif set(current_fields) != set(fieldnames):
                        raise ValueError(
                            f"Schema mismatch in {source}: expected {fieldnames}, "
                            f"found {current_fields}"
                        )

                    assert writer is not None
                    for line_number, row in enumerate(reader, start=2):
                        normalized_row = {}
                        for column, value in row.items():
                            if value is None:
                                raise ValueError(
                                    f"Missing value at {source}:{line_number} "
                                    f"in column {column!r}"
                                )
                            uppercase_umlaut_count += sum(
                                value.count(character) for character in "ÄÖÜ"
                            )
                            normalized_row[column] = value.translate(
                                UPPERCASE_UMLAUT_TRANSLATION
                            )
                        row = normalized_row

                        entry_id = row.get(args.id_column, "")
                        if not entry_id:
                            raise ValueError(
                                f"Empty {args.id_column!r} at {source}:{line_number}"
                            )
                        if not args.allow_duplicate_ids and entry_id in seen_ids:
                            raise ValueError(
                                f"Duplicate {args.id_column} {entry_id!r} "
                                f"at {source}:{line_number}"
                            )
                        seen_ids.add(entry_id)

                        writer.writerow(row)
                        source_rows += 1
                        total_rows += 1
                        if args.text_column in row:
                            character_count += len(row[args.text_column])

                per_file_counts.append((source, source_rows))

        with temporary.open(encoding="utf-8", newline="") as merged_file:
            verified_rows = sum(1 for _ in csv.DictReader(merged_file))
        if verified_rows != total_rows:
            raise RuntimeError(
                f"Output validation failed: wrote {total_rows} rows, "
                f"read back {verified_rows}"
            )

        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    for source, count in per_file_counts:
        print(f"{source}: {count} rows")
    print(f"Created: {target}")
    print(f"Rows: {total_rows}")
    print(f"Unique {args.id_column} values: {len(seen_ids)}")
    print(f"Uppercase umlauts expanded: {uppercase_umlaut_count}")
    if fieldnames and args.text_column in fieldnames:
        print(f"{args.text_column} characters: {character_count}")
    return target


def main() -> None:
    merge_csvs(parse_args())


if __name__ == "__main__":
    main()
