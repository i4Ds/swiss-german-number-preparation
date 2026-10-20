#!/usr/bin/env python3
"""Upload one or more CSV files as Hugging Face dataset repositories.

Each positional argument has the form ``CSV_PATH=REPO_ID``. Every CSV is
uploaded to the selected split (``train`` by default), and repositories are
private unless ``--public`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = {"client_id", "id", "sentence_ch", "sentence_de"}
DEFAULT_TRANSLATION_MODEL = "unspecified (pass --translation-model for provenance)"
CARD_SECTION_MARKER = "<!-- generated-by-upload_csv_datasets_to_hf.py -->"


def csv_repo_mapping(value: str) -> tuple[Path, str]:
    csv_path, separator, repo_id = value.rpartition("=")
    if not separator or not csv_path or not repo_id or "/" not in repo_id:
        raise argparse.ArgumentTypeError("expected CSV_PATH=OWNER/DATASET")
    return Path(csv_path), repo_id


def inspect_csv(path: Path) -> tuple[list[str], int]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(f"CSV is missing columns {sorted(missing)}: {path}")
        rows = sum(1 for _ in reader)
    if rows == 0:
        raise ValueError(f"CSV contains no data rows: {path}")
    return fieldnames, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        type=csv_repo_mapping,
        metavar="CSV_PATH=OWNER/DATASET",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--translation-model",
        default=DEFAULT_TRANSLATION_MODEL,
        help="Translation model recorded in each Hugging Face dataset card",
    )
    parser.add_argument(
        "--cards-only",
        action="store_true",
        help="Update dataset cards without uploading the CSV data again",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create/update public repositories (default: private)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (normally use cached login or HF_TOKEN)",
    )
    return parser


def update_dataset_card(
    *,
    repo_id: str,
    csv_path: Path,
    rows: int,
    split: str,
    translation_model: str,
    token: str | None,
) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    readme_path = hf_hub_download(
        repo_id,
        "README.md",
        repo_type="dataset",
        token=token,
        force_download=True,
    )
    existing = Path(readme_path).read_text(encoding="utf-8").rstrip()
    if CARD_SECTION_MARKER in existing:
        existing = existing.split(CARD_SECTION_MARKER, 1)[0].rstrip()
    title = repo_id.rsplit("/", 1)[-1].replace("_", " ").title()
    section = f"""

{CARD_SECTION_MARKER}
# {title}

Swiss German and Standard German sentence pairs prepared for number-aware
text-to-speech and translation training.

## Data

- Split: `{split}`
- Rows: {rows}
- Columns: `client_id`, `id`, `sentence_ch`, `sentence_de`
- Source CSV: `{csv_path.name}`

## Translation model

The Swiss German sentences were generated with the following fine-tuned T5
German-to-Swiss-German translation model:

```text
{translation_model}
```
"""
    HfApi(token=token).upload_file(
        path_or_fileobj=(existing + section + "\n").encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Document translation model and dataset provenance",
    )


def main() -> None:
    args = build_parser().parse_args()
    repo_ids = [repo_id for _, repo_id in args.datasets]
    if len(repo_ids) != len(set(repo_ids)):
        raise ValueError("Each repository may be targeted only once per invocation")

    inspected = []
    for raw_path, repo_id in args.datasets:
        path = raw_path.expanduser().resolve()
        fieldnames, rows = inspect_csv(path)
        inspected.append((path, repo_id, fieldnames, rows))
        print(f"Validated {path}: {rows} rows, columns={fieldnames}")

    from datasets import Dataset

    for path, repo_id, _, rows in inspected:
        if not args.cards_only:
            print(
                f"Uploading {rows} rows to {repo_id!r}, split={args.split!r}, "
                f"private={not args.public} ..."
            )
            dataset = Dataset.from_csv(str(path))
            if dataset.num_rows != rows:
                raise RuntimeError(
                    f"Loaded row count changed for {path}: {dataset.num_rows} != {rows}"
                )
            dataset.push_to_hub(
                repo_id,
                split=args.split,
                private=not args.public,
                token=args.token,
            )
            print(f"Uploaded data: https://huggingface.co/datasets/{repo_id}")
        print(f"Updating dataset card for {repo_id!r} ...")
        update_dataset_card(
            repo_id=repo_id,
            csv_path=path,
            rows=rows,
            split=args.split,
            translation_model=args.translation_model,
            token=args.token,
        )
        print(f"Updated card: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
