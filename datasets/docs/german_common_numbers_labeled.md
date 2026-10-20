# German Commons number annotations

Configuration: `german_common_numbers_labeled`; split: `train`; **274,988 rows**.

German sentences containing numeric expressions, collected by the project from
`coral-nlp/german-commons`, labelled with the saved Qwen3-VL number-annotation
pipeline, and prepared with placeholders and St. Gallen number forms.

`full_de_text` is the German source. `replaced_de_text` replaces annotated spans
with placeholders such as `xax`. `sg_de_text` substitutes Swiss German number
forms into the German source context; **it is not a translation of the complete
sentence into Swiss German**. `numbers` retains semantic labels, original-source
offsets, placeholder mappings, and `sg_written_form`. Its heterogeneous `value`
payloads are JSON-encoded strings; decode with `json.loads`.

The October 2026 cleanup individually inspected all 15 previously identified raw
duplicate pairs. Twelve pairs were present twice in this prepared configuration;
12 redundant records were removed. Two pairs were already absent and one pair
had only one published candidate. Six retained records received targeted label
corrections: weak ordinal inflection, year DATE labels, and a decimal RANGE.
Offsets, placeholders and spoken-number text were regenerated for those six
records. All other retained records are unchanged. No duplicates remain under
casefold-plus-whitespace normalization. This was an assistant review of these
specific pairs, not a full human or native-speaker validation of the corpus.

The [complete review](../curation/manual_duplicate_review.json) records every
selection, removal, reason and before/after correction. Historic files under
`originals/` remain the unchanged pre-cleanup source exports; load the Parquet
configuration for the curated release. There is no official held-out test split.

Number annotation counts by type:

| Type | Annotations |
|---|---:|
| `DATE` | 140,094 |
| `DECIMAL` | 17,527 |
| `FRACTION` | 871 |
| `NUMBER_CARDINAL` | 225,440 |
| `NUMBER_ORDINAL` | 28,033 |
| `PRODUCT_ID` | 1,851 |
| `RANGE` | 4,974 |
| `RATIO` | 1,747 |
| `TIME` | 3,708 |
