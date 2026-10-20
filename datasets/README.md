---
language:
- de
- gsw
task_categories:
- text-generation
configs:
- config_name: german_common_numbers_labeled
  data_files:
  - split: train
    path: data/german_common_numbers_labeled/train-00000-of-00001.parquet
- config_name: gemini_3.5_flash_lite_gen_labeled
  data_files:
  - split: train
    path: data/gemini_3.5_flash_lite_gen_labeled/train-00000-of-00001.parquet
dataset_info:
- config_name: german_common_numbers_labeled
  features:
  - name: id
    dtype: string
  - name: full_de_text
    dtype: string
  - name: replaced_de_text
    dtype: string
  - name: sg_de_text
    dtype: string
  - name: numbers
    list:
    - name: number
      dtype: string
    - name: value
      dtype: string
    - name: type
      dtype: string
    - name: start
      dtype: int64
    - name: end
      dtype: int64
    - name: placeholder
      dtype: string
    - name: sg_written_form
      dtype: string
    - name: deklination
      dtype: string
    - name: genus
      dtype: string
    - name: kasus
      dtype: string
    - name: notation
      dtype: string
  splits:
  - name: train
    num_examples: 274988
    num_bytes: 174987601
  download_size: 54761615
  dataset_size: 174987601
- config_name: gemini_3.5_flash_lite_gen_labeled
  features:
  - name: id
    dtype: string
  - name: full_de_text
    dtype: string
  - name: replaced_de_text
    dtype: string
  - name: sg_de_text
    dtype: string
  - name: numbers
    list:
    - name: number
      dtype: string
    - name: value
      dtype: string
    - name: type
      dtype: string
    - name: start
      dtype: int64
    - name: end
      dtype: int64
    - name: placeholder
      dtype: string
    - name: sg_written_form
      dtype: string
    - name: subtype
      dtype: string
  splits:
  - name: train
    num_examples: 19763
    num_bytes: 8006286
  download_size: 1912455
  dataset_size: 8006286
pretty_name: German Number Annotations and St. Gallen Number Forms
---
# German number annotations and St. Gallen verbalisation data

Two German number resources for semantic numeric annotation, placeholder-based
translation workflows, and speech-oriented number preparation. Each record
contains source German text, typed number spans, placeholders and written-out
St. Gallen number forms. These prepared records do not yet provide full German
to Swiss German sentence translations.

| Configuration | Origin | Split | Rows | Description |
|---|---|---|---:|---|
| `german_common_numbers_labeled` | German Commons source sentences with model annotations | `train` | 274,988 | [Configuration details](docs/german_common_numbers_labeled.md) |
| `gemini_3.5_flash_lite_gen_labeled` | Synthetic German sentences and annotations | `train` | 19,763 | [Configuration details](docs/gemini_3.5_flash_lite_gen_labeled.md) |

## Fields

| Field | Meaning |
|---|---|
| `id` | Stable source record identifier |
| `full_de_text` | Original or generated German source sentence |
| `replaced_de_text` | German source with annotated spans replaced by placeholders |
| `sg_de_text` | German context with St. Gallen number forms substituted into the annotated spans |
| `numbers` | Native list of annotation objects: number text, value, type, source offsets, placeholder and written form |

`start` and `end` are zero-based character offsets into `full_de_text`, with end
exclusive. Optional annotation fields such as grammatical attributes, notation
or subtype depend on the configuration and are null when absent. `numbers[].value`
is JSON text because the original values can be strings, lists or objects.
Decode it with `json.loads` to recover the original value and its type.

```python
import json
from datasets import load_dataset

data = load_dataset("i4ds/german_numbers_labeled",
                    "german_common_numbers_labeled", split="train", token=True)
# Alternative configuration: "gemini_3.5_flash_lite_gen_labeled"
value = json.loads(data[0]["numbers"][0]["value"])
```

## October 2026 manual duplicate review

The published German Commons configuration contained 12 duplicate pairs under
case-insensitive whitespace normalization. All 15 pairs identified in the larger
raw label collection were inspected individually: 12 published redundant rows
were removed, two pairs were already excluded, and one had only one published
record. Six retained records received targeted annotation corrections. Source
text, IDs and ordering are preserved for retained records; only those six
records' derived annotations/placeholders/number forms changed. The synthetic
configuration is unchanged. The [audit](curation/manual_duplicate_review.json)
records every decision. This is targeted assistant review, not a claim that all
annotations are human-validated.

## What the checkpoint duplication count means

The internal raw annotation checkpoint contains 557,884 saved responses but only
293,942 distinct IDs: 263,942 entries repeat an earlier ID. Of those repeated IDs,
31,158 have different saved response strings. The final raw results JSON has one
record per ID; its response agrees with the last checkpoint entry for every ID.
These counts describe saved inference history, not duplicate sentences in this
published prepared configuration. The final raw collection and this filtered
prepared subset also have different row counts.

## Archives, provenance and limitations

Historical JSON exports under `originals/` remain unchanged pre-cleanup snapshots:
`qwen3vl_run_01_prepared_large.json` and
`synthetic_full_flash_lite_prepared_large.json`. Their values use the original
heterogeneous JSON types. Prefer the loadable Parquet configurations for the
current curated release; apply the audit to the historical German Commons JSON
if reproducing the cleanup from that snapshot.

Labels are primarily machine-generated. Numeric types, grammatical features and
number forms may still contain errors outside the reviewed examples. Only
St. Gallen number forms are included here; the five-dialect Wikipedia number
release is a separate resource. Only train splits are provided; create grouped
held-out splits before using these data as an evaluation benchmark. See the
configuration descriptions for annotation counts, data origins and intended use.
