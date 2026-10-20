# Swiss German number preparation

Standalone collection, semantic annotation and written-number preparation for
German / Swiss German translation and speech workflows. The two processing
components share a pinned `num2words_CH` submodule supporting Basel, St. Gallen,
Zürich, Luzern and Bern.

## Layout

- `collect_de_num_data/`: collect German numeric sentences, generate synthetic examples, label, prepare placeholders, translate and merge.
- `preprocess_num_of_datasets/`: annotate numeric spans in existing parallel datasets and publish written-number columns.
- `num2words_CH/`: converter submodule pinned to the Luzern/Bern revision.
- `datasets/`: published dataset cards, pinned release metadata, statistics and the manual duplicate-review audit.
- `scripts/`: download released data and package the two canonical raw label inputs.
- `examples/number_samples.csv`: small input fixture for local checks.

## Published datasets

[i4ds/german_numbers_labeled](https://huggingface.co/datasets/i4ds/german_numbers_labeled)
contains two prepared configurations:

| Configuration | Rows | Number forms |
|---|---:|---|
| `german_common_numbers_labeled` | 274,988 | St. Gallen |
| `gemini_3.5_flash_lite_gen_labeled` | 19,763 | St. Gallen |

These have German source text, semantic number annotations, placeholders and
German context with St. Gallen number words inserted. They are not full Swiss
German sentence translations. Five converters are available in the code;
five-dialect generated versions of these two configurations are future outputs.
See [dataset documentation](datasets/README.md), [release revisions/checksums](datasets/releases.json)
and the [duplicate review](datasets/curation/manual_duplicate_review.json).

The [Wikipedia dataset](https://huggingface.co/datasets/i4ds/translator_data_wikipedia)
already includes five-dialect written-number targets and per-row annotations.

## Install

Clone this repository with `--recurse-submodules`, or initialise the dependency:

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export DATA_ROOT=/path/to/number-data
```

`DATA_ROOT` defaults to `~/.local/share/swiss-german-number-preparation`.
Keep it outside the code checkout. Stages are `raw`, `intermediate`, `processed`,
`results` and `logs`, with separate component directories underneath.
Copy `.env.example` to `.env` for credentials; load storage settings into your shell.

For GPU annotation, install `python -m pip install -e '.[inference]'` in a suitable
GPU environment. For standalone T5 inference, install `'.[translation]'` instead.
The CPU number-conversion path does not require vLLM, spaCy or Java. The full
optional detection stack has its own notes in `num2words_CH/SETUP_NOTES.md`.

## Download the pinned prepared datasets

```bash
python scripts/download_release.py german-commons --output-dir "$DATA_ROOT/releases"
python scripts/download_release.py synthetic --output-dir "$DATA_ROOT/releases"
```

Private releases use `HF_TOKEN` or cached Hugging Face authentication. Downloads
are revision-pinned, checksum-verified and protected against accidental replacement.

## Canonical raw labels

[datasets/raw_inputs.json](datasets/raw_inputs.json) identifies exactly two raw
annotation inputs: the completed German Commons results JSON and the synthetic
checkpoint JSONL. The German Commons checkpoint is a duplicate response log and
must not be concatenated with the completed results. Package the canonical inputs:

```bash
python scripts/package_raw_labels.py --input-dir /path/to/03_labeled --output-dir "$DATA_ROOT/releases/raw-labels"
```

This creates two deterministic gzip archives and a checksum manifest outside Git.
The raw inputs remain distinct from the filtered, manually curated prepared
configurations hosted on Hugging Face.

## Pipeline entry points

Full processing runs require a Slurm allocation. `TTS_ALLOW_LOGIN_NODE=1` is only
for tiny fixtures and tests.

```bash
python collect_de_num_data/scripts/labeling/prepare_jsonl.py /path/to/results.json --dialect ch_lu --out_dir "$DATA_ROOT/prepared/lu"
python preprocess_num_of_datasets/scripts/preprocess_dataset_numbers.py prepare examples/number_samples.csv --run-name example_v1
# Run the printed annotation command, then:
python preprocess_num_of_datasets/scripts/preprocess_dataset_numbers.py apply --run-name example_v1 --dialect ch_be
```

For Wikipedia publishing, provide the four saved run manifests and inference outputs:

```bash
python preprocess_num_of_datasets/scripts/publish_written_number_column.py --runs-dir /path/to/wikipedia-runs --output-dir "$DATA_ROOT/wikipedia-release"
```

It exports native annotation lists, written targets, converter selection and
failure status. Supported regional labels select their own converter; every
other region uses St. Gallen. Source rows, labels and split assignments are preserved.
Uploads remain opt-in through `--push`.

## Translate prepared number sentences

```bash
python collect_de_num_data/scripts/translation/translate_prepared_csv.py /path/to/prepared.csv --model /path/to/t5/final_model
```

The included adapter uses the existing `de_ch_translator` dialect prompts and
loads a local T5 checkpoint directly through Transformers. It requires no sibling
TTS checkout. Long inputs are rejected rather than silently truncated; placeholder
preservation is recorded for every generated row. Model execution is not part of
the offline tests. Scheduler launchers remain optional deployment examples.

## Tests and provenance

```bash
TTS_ALLOW_LOGIN_NODE=1 python -m unittest discover -s collect_de_num_data/tests -v
python -m unittest discover -s preprocess_num_of_datasets/tests -v
```

Tests use small fixtures and no hosted inference. The extraction source commit
and original file checksums are in `provenance/extraction.json`; source and
third-party attribution are preserved in [NOTICE.md](NOTICE.md).
