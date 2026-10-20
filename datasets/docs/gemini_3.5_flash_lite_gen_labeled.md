# Synthetic German number annotations

Configuration: `gemini_3.5_flash_lite_gen_labeled`; split: `train`;
**19,763 rows**. Data is unchanged by the October 2026 duplicate cleanup.

Synthetic German sentences generated with numeric annotations by the project's
Gemini 3.5 Flash Lite generation pipeline, then filtered and prepared with
placeholders and St. Gallen number forms. The generator targets varied numeric
categories and includes format and span validation. Generated examples and
labels remain machine-created and have not all been independently verified.

The schema is `id`, `full_de_text`, `replaced_de_text`, `sg_de_text`, and `numbers`.
`full_de_text` is generated German text; `replaced_de_text` contains placeholders;
`sg_de_text` inserts Swiss German number forms into otherwise German context.
It is not a fully translated Swiss German sentence. Nested annotation objects
include the original span, type, JSON-encoded `value`, offsets, placeholder,
`sg_written_form`, and optional subtype. Decode `value` using `json.loads`.

Examples were generated to exercise the pipeline and are not evidence that an
identifier, price, address, or event describes a real-world entity. This subset
has no official held-out test split and is not a manually adjudicated benchmark.
The historical source export is `originals/synthetic_full_flash_lite_prepared_large.json`.

Number annotation counts by type:

| Type | Annotations |
|---|---:|
| `CURRENCY` | 3,419 |
| `DATE` | 1,000 |
| `DECIMAL` | 140 |
| `IBAN` | 400 |
| `MATH_EXPRESSION` | 2,758 |
| `NUMBER_CARDINAL` | 140 |
| `NUMBER_WITH_SUFFIX` | 100 |
| `PHONE` | 2,000 |
| `PIN` | 500 |
| `PRODUCT_ID` | 5,971 |
| `TIME` | 2,135 |
| `VERSION` | 635 |
| `ZIP` | 1,500 |
