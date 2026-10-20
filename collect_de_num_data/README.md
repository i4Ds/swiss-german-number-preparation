# German number collection and preparation

See the [repository guide](../README.md) for dependencies, release downloads,
canonical raw labels, dialect selection, translation and validation.

The processing sequence is collection → prompts → LLM labels → prepared
placeholders and spoken numbers → optional T5 translation → merged sentence pairs.
Run entry points from the repository root. `DATA_ROOT` controls external storage;
`NUM2WORDS_CH_ROOT` optionally overrides the pinned converter. The selected dialect
can be `ch_bs`, `ch_sg`, `ch_zh`, `ch_lu` or `ch_be`.

`configs/llm_inference/qwen3vl_run_01.yaml` records the existing Qwen annotation
configuration. `scripts/labeling/generate_synthetic_labeled.py --help` exposes the
synthetic generation presets. Both full annotation and translation require a
compute allocation. `scripts/labeling/prepare_jsonl.py --help` documents filtering;
`scripts/labeling/merge_translated_csv_and_json.py --help` documents final merging.

The Dockerfile provides the CPU tools when built from the repository root:
`docker build -f collect_de_num_data/Dockerfile .`. GPU annotation uses the
separate optional inference dependencies and a suitable CUDA environment.
