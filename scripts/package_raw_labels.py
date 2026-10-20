#!/usr/bin/env python3
"""Package exactly the two canonical raw label inputs, excluding duplicate logs."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Original 03_labeled directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    if output == root or root in output.parents:
        parser.error("Store annotation archives outside the code repository")
    manifest = json.loads((root / "datasets/raw_inputs.json").read_text())
    records = []
    for entry in manifest["canonical_inputs"]:
        path = args.input_dir / entry["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Input differs from the recorded canonical labels: {path}")
        output.mkdir(parents=True, exist_ok=True)
        target = output / (path.name + ".gz")
        if target.exists():
            raise FileExistsError(target)
        with path.open("rb") as source, target.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)
        records.append({**entry, "archive": target.name,
                        "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        print(target)
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
