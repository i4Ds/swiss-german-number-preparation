# Number annotation in existing parallel datasets

The [repository guide](../README.md) describes setup and published resources.
This component reuses the adjacent collection component's annotation prompts
and the pinned converter; neither requires the old combined TTS repository.

From the repository root, prepare a CSV, JSON, JSONL or Parquet run:

```bash
python preprocess_num_of_datasets/scripts/preprocess_dataset_numbers.py prepare /path/to/input.parquet --run-name corpus_v1
```

The command records source hashes and prints the annotation invocation. Run it
inside a GPU allocation, or use `scripts/run_on_scicore.sh`. Apply saved labels:

```bash
python preprocess_num_of_datasets/scripts/preprocess_dataset_numbers.py apply --run-name corpus_v1 --dialect ch_be
```

The generic application command uses an explicitly selected converter. The
Wikipedia-specific publisher instead defaults to per-row dialect selection,
with St. Gallen fallback for other regions. It exports `numbers`, written text,
converter provenance and failure status, preserving the original columns and
splits. See `scripts/publish_written_number_column.py --help`.

Optional FHNW audio publishing helpers download the source data from Hugging
Face; no local recording repository is required. Existing launchers are deployment
examples with configurable scheduler/environment settings. The bootstrap requires
`PIPELINE_GIT_REPO` to identify the checkout to deploy.
