#!/usr/bin/env python3
"""
Prepare a results checkpoint JSONL file or final results JSON array:
  1. Print stats for text_quality and number type distributions
  2. Reclassify colon-delimited RANGE annotations as RATIO
  3. Remove entries with text_quality != "Good", a slash-delimited RANGE,
     any excluded number type, or a number whose Swiss-German form cannot be
     generated
  4. Replace each number in the text with a placeholder (xax, xbx, xcx, ...)
  5. Write a CSV: id, client_id, sentence_de, sentence_ch
  6. Write a JSON: id, full_de_text, replaced_de_text, sg_de_text,
                   numbers (with placeholder + sg_written_form)
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from string import ascii_lowercase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline_runtime import require_compute_node, storage_path

DIGIT_RE = re.compile(r"\d")
SUPPORTED_DIALECTS = ("ch_sg", "ch_bs", "ch_zh", "ch_lu", "ch_be")


def convert_number(*args, **kwargs):
    """Load the pandas-backed converter only after the Slurm guard."""
    if __package__:
        from .convert_ch_numbers import convert_number as converter
    else:
        from convert_ch_numbers import convert_number as converter

    return converter(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def placeholder(index: int) -> str:
    """
    Map 0 -> xax, 1 -> xbx, 2 -> xcx, ...
    Supports up to 26 numbers per sentence (a-z).
    For index >= 26 uses double letters: xaax, xabx, ...
    """
    if index < 26:
        return f"x{ascii_lowercase[index]}x"
    # fallback: two-letter code
    first = ascii_lowercase[index // 26 - 1]
    second = ascii_lowercase[index % 26]
    return f"x{first}{second}x"


def replace_numbers_in_text(text: str, numbers: list, dialect: str = "ch_sg") -> tuple[str, list, str]:
    """
    Replace each number occurrence in text with xax, xbx, ... in order.
    Returns (replaced_text, enriched_numbers, sg_text) where:
      - enriched_numbers: each dict gains 'placeholder' and 'sg_written_form'
      - sg_text: full text with placeholders substituted by the selected forms
    """
    enriched = []
    spans = locate_number_spans(text, numbers)
    replaced = text

    for idx, (start, end, num_info) in enumerate(spans):
        ph = placeholder(idx)
        sg_form = convert_number(num_info, dialect=dialect)
        enriched.append({
            **num_info,
            "start": start,
            "end": end,
            "placeholder": ph,
            "sg_written_form": sg_form or "unknown",
        })

    for start, end, num_info in reversed(spans):
        enriched_number = next(
            item for item in enriched
            if item["start"] == start and item["end"] == end
        )
        replaced = replaced[:start] + enriched_number["placeholder"] + replaced[end:]

    # Build SG text by substituting each placeholder with its SG spoken form
    sg_text = replaced
    for num_info in enriched:
        sg_text = sg_text.replace(num_info["placeholder"], num_info.get("sg_written_form") or "unknown", 1)

    return replaced, enriched, sg_text


def locate_number_spans(text: str, numbers: list[dict]) -> list[tuple[int, int, dict]]:
    """Resolve exact non-overlapping spans, preferring supplied offsets and boundaries."""
    located: list[tuple[int, int, dict]] = []
    cursor = 0
    for index, annotation in enumerate(numbers, start=1):
        number = annotation.get("number")
        if not isinstance(number, str) or not number:
            raise ValueError(f"Number annotation {index} has no number text")

        raw_start, raw_end = annotation.get("start"), annotation.get("end")
        if isinstance(raw_start, int) and isinstance(raw_end, int):
            if not (0 <= raw_start < raw_end <= len(text)):
                raise ValueError(f"Invalid offsets for annotation {number!r}")
            if text[raw_start:raw_end] != number:
                raise ValueError(f"Offsets do not select annotation {number!r}")
            candidates = [(raw_start, raw_end)]
        else:
            candidates = [
                (match.start(), match.end())
                for match in re.finditer(re.escape(number), text)
            ]

        candidates = [
            span for span in candidates
            if all(span[1] <= old_start or span[0] >= old_end
                   for old_start, old_end, _ in located)
        ]
        if not candidates:
            raise ValueError(f"Annotated number {number!r} has no unused exact span")

        def candidate_rank(span: tuple[int, int]) -> tuple[int, int, int]:
            start, end = span
            left_boundary = start == 0 or not text[start - 1].isalnum()
            right_boundary = end == len(text) or not text[end].isalnum()
            follows_annotation_order = start >= cursor
            return (
                int(left_boundary) + int(right_boundary),
                int(follows_annotation_order),
                -start,
            )

        start, end = max(candidates, key=candidate_rank)
        if annotation.get("type") == "TIME" and not number.casefold().endswith("uhr"):
            following_uhr = re.match(r"\s+Uhr\b", text[end:], re.IGNORECASE)
            if following_uhr is not None:
                end += following_uhr.end()
                annotation["number"] = text[start:end]
                annotation["start"] = start
                annotation["end"] = end
        located.append((start, end, annotation))
        cursor = max(cursor, end)

    located.sort(key=lambda item: item[0])
    uncovered = [
        position for position, char in enumerate(text)
        if char.isdigit()
        and not any(start <= position < end for start, end, _ in located)
    ]
    if uncovered:
        raise ValueError(
            f"Unannotated digit at character offset {uncovered[0]}"
        )
    return located


def parse_response(raw_response: str):
    """Parse the JSON string stored in the 'response' field."""
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return None


def iter_result_records(input_path: Path):
    """Yield ``(location, record)`` from a JSONL checkpoint or JSON array."""
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"  [WARN] line {line_no}: JSON parse error – {exc}", file=sys.stderr)
                    continue
                if not isinstance(record, dict):
                    print(f"  [WARN] line {line_no}: expected a JSON object", file=sys.stderr)
                    continue
                yield f"line {line_no}", record
        return

    if input_path.suffix.lower() == ".json":
        try:
            with input_path.open(encoding="utf-8") as fh:
                records = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {input_path}: {exc}") from exc
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON array in {input_path}")
        for index, record in enumerate(records, 1):
            if not isinstance(record, dict):
                print(f"  [WARN] item {index}: expected a JSON object", file=sys.stderr)
                continue
            yield f"item {index}", record
        return

    raise ValueError(
        f"Unsupported input format for {input_path}; expected .jsonl or .json"
    )


def parse_number_types(raw_types: list[str] | None) -> set[str] | None:
    """Parse --number-types values; ``None`` means use the default filter."""
    if not raw_types:
        return None

    types = {
        item.strip().upper()
        for raw_value in raw_types
        for item in raw_value.split(",")
        if item.strip()
    }
    if not types:
        raise ValueError("--number-types must contain at least one type")
    if "ALL" in types:
        if len(types) != 1:
            raise ValueError("ALL cannot be combined with specific number types")
        return set()
    return types


def default_output_dir(input_path: Path) -> Path:
    """Write prepared artifacts to persistent intermediate storage."""
    return storage_path("intermediate", "prepared")


def run_name_from_labeled_path(input_path: Path) -> str:
    """Derive one run name from either supported results filename."""
    stem = input_path.stem
    if stem.endswith("_results_checkpoint"):
        return stem.removesuffix("_results_checkpoint")
    return stem.removesuffix("_results")


def has_spoken_forms(numbers: list, dialect: str = "ch_sg") -> bool:
    """Return whether every annotated number can be converted to a spoken form."""
    for num_info in numbers:
        spoken_form = convert_number(num_info, dialect=dialect)
        if not spoken_form or spoken_form.strip().lower() == "unknown":
            return False
    return True


def has_numeric_digit(response: dict) -> bool:
    """Return True when the original text or annotations contain a digit."""
    if DIGIT_RE.search(str(response.get("text", ""))):
        return True
    for num_info in response.get("numbers", []):
        if DIGIT_RE.search(str(num_info.get("number", ""))):
            return True
        if DIGIT_RE.search(json.dumps(num_info.get("value", ""), ensure_ascii=False)):
            return True
    return False


def reclassify_colon_ranges(numbers: list) -> int:
    """Turn legacy colon-delimited RANGE labels into RATIO labels in place."""
    changed = 0
    for num_info in numbers:
        if (
            str(num_info.get("type", "")).upper() == "RANGE"
            and ":" in str(num_info.get("number", ""))
        ):
            num_info["type"] = "RATIO"
            changed += 1
    return changed


def has_slash_range(numbers: list) -> bool:
    """Return True when an unsupported RANGE annotation contains a slash delimiter."""
    return any(
        str(num_info.get("type", "")).upper() == "RANGE"
        and "/" in str(num_info.get("number", ""))
        and str(num_info.get("subtype", "")).upper() not in {
            "DATE",
            "DATE_RANGE",
            "YEAR",
            "YEAR_RANGE",
        }
        for num_info in numbers
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare a results checkpoint .jsonl or final results .json file"
    )
    parser.add_argument("input_jsonl", help="Path to input .jsonl or .json file")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: DATA_ROOT/intermediate/collect_de_num_data/prepared)",
    )
    parser.add_argument(
        "--number-types",
        nargs="+",
        metavar="TYPE",
        help=(
            "Only keep entries whose number items all have one of these types. "
            "Accepts spaces or commas, e.g. --number-types TIME PHONE or "
            "--number-types TIME,PHONE. Use ALL to include every type. "
            "By default, only OTHER_NUMERIC is excluded."
        ),
    )
    parser.add_argument(
        "--dialect",
        choices=SUPPORTED_DIALECTS,
        default="ch_sg",
        help="Spoken number dialect (default: ch_sg)",
    )
    args = parser.parse_args()
    require_compute_node("number-label preparation")

    try:
        selected_types = parse_number_types(args.number_types)
    except ValueError as exc:
        parser.error(str(exc))

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        sys.exit(f"ERROR: file not found: {input_path}")

    out_dir = Path(args.out_dir) if args.out_dir else default_output_dir(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = run_name_from_labeled_path(input_path)
    csv_path  = out_dir / f"{run_name}_prepared.csv"
    json_path = out_dir / f"{run_name}_prepared.json"
    relabel_path = out_dir / f"{run_name}_relabel.jsonl"

    # ------------------------------------------------------------------
    # Pass 1: read all entries, collect stats
    # ------------------------------------------------------------------
    quality_counter: Counter = Counter()
    type_counter:    Counter = Counter()
    all_entries = []
    reclassified_ratios = 0

    seen_ids: dict[str, str] = {}
    duplicate_ids: list[tuple[str, str, str]] = []
    try:
        input_records = iter_result_records(input_path)
        for location, record in input_records:
            response = parse_response(record.get("response", ""))
            if response is None:
                print(f"  [WARN] {location}: could not parse 'response' field", file=sys.stderr)
                continue

            entry_id = str(record.get("id", location))
            if entry_id in seen_ids:
                duplicate_ids.append((entry_id, seen_ids[entry_id], location))
                continue
            seen_ids[entry_id] = location

            quality = response.get("text_quality", "UNKNOWN")
            quality_counter[quality] += 1

            numbers = response.get("numbers", [])
            reclassified_ratios += reclassify_colon_ranges(numbers)
            for num in numbers:
                type_counter[num.get("type", "UNKNOWN")] += 1

            all_entries.append((entry_id, response))
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    if duplicate_ids:
        examples = ", ".join(
            f"{entry_id!r} ({first}, {second})"
            for entry_id, first, second in duplicate_ids[:3]
        )
        sys.exit(
            "ERROR: duplicate IDs found in input: "
            f"{len(duplicate_ids)} duplicate record(s); examples: {examples}. "
            "Use the clean final results JSON instead of a concurrently written checkpoint."
        )

    total = len(all_entries)

    # ------------------------------------------------------------------
    # Print stats
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Total entries read: {total}")

    print(f"\n--- text_quality distribution ---")
    for quality, count in quality_counter.most_common():
        pct = 100 * count / total if total else 0
        print(f"  {quality:<12} {count:>8}  ({pct:.1f}%)")

    print(f"\n--- number type distribution ---")
    total_nums = sum(type_counter.values())
    for ntype, count in type_counter.most_common():
        pct = 100 * count / total_nums if total_nums else 0
        print(f"  {ntype:<25} {count:>8}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<25} {total_nums:>8}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Pass 2: filter + transform
    # ------------------------------------------------------------------
    good_entries = [(eid, resp) for eid, resp in all_entries if resp.get("text_quality") == "Good"]
    removed_quality = total - len(good_entries)

    numeric_entries = [(eid, resp) for eid, resp in good_entries if has_numeric_digit(resp)]
    removed_no_digits = len(good_entries) - len(numeric_entries)

    slash_range_eligible_entries = [
        (eid, resp)
        for eid, resp in numeric_entries
        if not has_slash_range(resp.get("numbers", []))
    ]
    removed_slash_range = len(numeric_entries) - len(slash_range_eligible_entries)

    if selected_types is None:
        # Preserve the original default: all types except OTHER_NUMERIC.
        def has_allowed_types(response: dict) -> bool:
            return not any(
                num.get("type") == "OTHER_NUMERIC"
                for num in response.get("numbers", [])
            )

        selection_description = "all types except OTHER_NUMERIC"
    elif not selected_types:  # --number-types ALL
        def has_allowed_types(response: dict) -> bool:
            return True

        selection_description = "all types"
    else:
        def has_allowed_types(response: dict) -> bool:
            return all(
                str(num.get("type", "")).upper() in selected_types
                for num in response.get("numbers", [])
            )

        selection_description = ", ".join(sorted(selected_types))

    type_eligible_entries = [
        (eid, resp)
        for eid, resp in slash_range_eligible_entries
        if has_allowed_types(resp)
    ]
    removed_type = len(slash_range_eligible_entries) - len(type_eligible_entries)
    spoken_entries = [
        (eid, resp)
        for eid, resp in type_eligible_entries
        if has_spoken_forms(resp.get("numbers", []), dialect=args.dialect)
    ]
    removed_unknown = len(type_eligible_entries) - len(spoken_entries)
    eligible_entries = []
    removed_incomplete_annotations = 0
    relabel_records = []
    for entry_id, response in spoken_entries:
        try:
            locate_number_spans(response.get("text", ""), response.get("numbers", []))
        except ValueError as exc:
            removed_incomplete_annotations += 1
            relabel_records.append(
                {
                    "id": entry_id,
                    "text": response.get("text", ""),
                    "numbers": response.get("numbers", []),
                    "reason": str(exc),
                }
            )
        else:
            eligible_entries.append((entry_id, response))
    print(f"Removed {removed_quality} entries with text_quality != 'Good'")
    print(f"Removed {removed_no_digits} Good entries without numeric digits")
    print(f"Reclassified {reclassified_ratios} colon-delimited RANGE items as RATIO")
    print(f"Removed {removed_slash_range} Good entries containing a slash-delimited RANGE")
    print(f"Number types kept: {selection_description}")
    print(f"Removed {removed_type} Good entries with excluded number types")
    print(f"Removed {removed_unknown} Good entries with an unknown written form")
    print(
        "Removed "
        f"{removed_incomplete_annotations} entries with invalid spans or unannotated digits"
    )
    print(f"Remaining: {len(eligible_entries)} entries\n")

    json_records = []
    csv_rows = []

    for entry_id, response in eligible_entries:
        full_text = response.get("text", "")
        numbers   = response.get("numbers", [])

        replaced_text, enriched_numbers, sg_text = replace_numbers_in_text(
            full_text, numbers, dialect=args.dialect
        )

        # CSV row
        csv_rows.append({
            "id":          entry_id,
            "client_id":   args.dialect.removeprefix("ch_"),
            "sentence_de": replaced_text,
            "sentence_ch": "",
        })

        # JSON record
        json_records.append({
            "id":               entry_id,
            "full_de_text":     full_text,
            "replaced_de_text": replaced_text,
            "sg_de_text":       sg_text,
            "numbers":          enriched_numbers,
        })

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    fieldnames = ["id", "client_id", "sentence_de", "sentence_ch"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV written  → {csv_path}  ({len(csv_rows)} rows)")

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(json_records, fh, ensure_ascii=False, indent=2)
    print(f"JSON written → {json_path}  ({len(json_records)} records)")

    with relabel_path.open("w", encoding="utf-8") as fh:
        for record in relabel_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Relabel queue → {relabel_path}  ({len(relabel_records)} records)")


if __name__ == "__main__":
    main()
