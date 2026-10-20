#!/usr/bin/env python3
"""Merge translated CSV rows with prepared number data into final JSON and CSV files.

Only rows whose placeholders were preserved are included. For each retained
record, ``replaced_ch_text`` contains the T5 translation with placeholders and
``sg_de_text`` contains the final text after each placeholder is replaced by
its Swiss German number form.

By default files are written below ``DATA_ROOT/results/collect_de_num_data``.
The CSV columns are
``client_id,id,sentence_ch,sentence_de``.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline_runtime import require_compute_node, storage_path


REQUIRED_CSV_COLUMNS = {
    "id",
    "sentence_ch",
    "placeholders_preserved",
}


def default_output_path(prepared_json: Path) -> Path:
    run_name = prepared_json.stem.removesuffix("_prepared")
    return storage_path("results", "final", f"{run_name}_final.json")


def default_csv_output_path(json_output: Path) -> Path:
    """Store the final CSV beside the final JSON with the same filename stem."""
    return json_output.with_suffix(".csv")


def load_translations(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_CSV_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"Translated CSV is missing columns: {sorted(missing)}")

        translations: dict[str, dict[str, str]] = {}
        for line_no, row in enumerate(reader, 2):
            entry_id = row["id"]
            if not entry_id:
                raise ValueError(f"Translated CSV line {line_no} has an empty id")
            if entry_id in translations:
                raise ValueError(f"Translated CSV has duplicate id {entry_id!r}")
            translations[entry_id] = row
    return translations


def insert_number_forms(translated_text: str, numbers: list[dict]) -> str:
    """Replace each unique placeholder once with its converted number form."""
    final_text = translated_text
    for number in numbers:
        placeholder = number.get("placeholder")
        spoken_form = number.get("sg_written_form")
        if not placeholder or not spoken_form or spoken_form == "unknown":
            raise ValueError(f"Invalid number conversion for placeholder {placeholder!r}")
        if final_text.count(placeholder) != 1:
            raise ValueError(
                f"Expected placeholder {placeholder!r} exactly once in translated text, "
                f"found {final_text.count(placeholder)} times"
            )
        final_text = final_text.replace(placeholder, spoken_form, 1)
    return final_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("translated_csv", type=Path, help="CSV produced by translate_prepared_csv.py")
    parser.add_argument("prepared_json", type=Path, help="JSON produced by prepare_jsonl.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final JSON path (default: DATA_ROOT results directory)",
    )
    parser.add_argument(
        "--client-id",
        default="sg",
        help="client_id to write to the final CSV and JSON (default: sg)",
    )
    args = parser.parse_args()
    require_compute_node("final number-dataset merge")

    if not args.translated_csv.is_file():
        parser.error(f"Translated CSV not found: {args.translated_csv}")
    if not args.prepared_json.is_file():
        parser.error(f"Prepared JSON not found: {args.prepared_json}")

    translations = load_translations(args.translated_csv)
    with args.prepared_json.open(encoding="utf-8") as fh:
        prepared_records = json.load(fh)
    if not isinstance(prepared_records, list):
        raise ValueError("Prepared JSON must contain a list of records")

    output_records = []
    csv_rows = []
    skipped_flagged = 0
    skipped_invalid = 0
    seen_ids: set[str] = set()

    for record in prepared_records:
        entry_id = record.get("id")
        if not entry_id:
            raise ValueError("Prepared JSON record has an empty id")
        if entry_id in seen_ids:
            raise ValueError(f"Prepared JSON has duplicate id {entry_id!r}")
        seen_ids.add(entry_id)

        translation_row = translations.get(entry_id)
        if translation_row is None:
            skipped_invalid += 1
            print(f"Skipping {entry_id}: no translated CSV row")
            continue
        if translation_row["placeholders_preserved"].strip().lower() != "true":
            skipped_flagged += 1
            continue

        translated_text = translation_row["sentence_ch"]
        try:
            final_text = insert_number_forms(translated_text, record.get("numbers", []))
        except ValueError as exc:
            skipped_invalid += 1
            print(f"Skipping {entry_id}: {exc}")
            continue

        output_records.append({
            **record,
            "client_id": args.client_id,
            "replaced_ch_text": translated_text,
            "sg_de_text": final_text,
            "placeholders_preserved": True,
        })
        csv_rows.append({
            "client_id": args.client_id,
            "id": entry_id,
            "sentence_ch": final_text,
            "sentence_de": record.get("full_de_text", ""),
        })

    unused_translations = set(translations) - seen_ids
    if unused_translations:
        print(f"Warning: {len(unused_translations)} translated CSV rows have no prepared JSON record")

    output_path = args.output or default_output_path(args.prepared_json)
    csv_output_path = default_csv_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output_records, fh, ensure_ascii=False, indent=2)

    csv_fieldnames = ["client_id", "id", "sentence_ch", "sentence_de"]
    with csv_output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Prepared records: {len(prepared_records)}")
    print(f"Final records: {len(output_records)}")
    print(f"Skipped (placeholder flag): {skipped_flagged}")
    print(f"Skipped (missing/invalid data): {skipped_invalid}")
    print(f"Final JSON written to: {output_path}")
    print(f"Final CSV written to: {csv_output_path}")


if __name__ == "__main__":
    main()
