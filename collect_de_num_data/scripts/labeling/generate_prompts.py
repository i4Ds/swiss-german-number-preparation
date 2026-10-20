"""
Generate prompts for a number-labeling run.

Reads N samples from each JSONL file and produces a prompts.json
compatible with scripts/labeling/run_llm_inference.py.

Usage:
    python scripts/labeling/generate_prompts.py [--run-name qwen3vl_run_01] [--n 10]
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline_runtime import require_compute_node, storage_path

DEFAULT_INPUT = storage_path("raw", "german_commons_numbers.jsonl")
PROMPTS_DIR = storage_path("intermediate", "prompts")

# ── Prompt templates ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a German linguistics expert. Your task is to annotate numbers in German \
sentences for a Text-to-Speech (TTS) system.

Return a JSON object with exactly these fields:

{
  "text": "<original sentence verbatim>",
  "text_quality": "<Bad|Okay|Good>",
  "numbers": [
    {
      "number": "<original number verbatim>",
      "value": "<formatted value — see type rules below>",
      "type": "<type — see list below>",
      "start": 0,
      "end": 10,
      "subtype": "<optional subtype — see subtype rules below>"
    }
  ]
}

For NUMBER_ORDINAL items only, also add these fields: "deklination"
("weak" or "strong"), "genus" ("maskulin", "feminin", or "neutrum"),
and "kasus" ("Nominativ", "Genitiv", "Dativ", or "Akkusativ").

─── TEXT QUALITY ───────────────────────────────────────────────────────────────
  Bad  : Many OCR errors, garbled text, incoherent sentences.
  Okay : Minor OCR artefacts, some unusual characters, a few spelling mistakes.
  Good : Clean, readable German sentence.

─── NUMBER TYPES & VALUE FORMAT ────────────────────────────────────────────────
  NUMBER_CARDINAL  – value: full integer without any separators   e.g. "10010"
                     Only if the number is read out as a whole cardinal number.
                     IBAN, product codes, etc. do NOT belong here.
                     Numbers followed by ordinary units or symbols remain
                     NUMBER_CARDINAL; annotate only the numerical portion, e.g.
                     "12" in "12 %".

  NUMBER_ORDINAL   – value: integer without period                e.g. "3"
                     Also provide: deklination (weak|strong), genus, kasus.
                     Do not infer an ordinal merely because an item is ordered.
                     Numbers following labels such as "Artikel", "Kapitel",
                     "Abschnitt", "§", "Seite", or "Nummer" are
                     NUMBER_CARDINAL and are read as cardinal numbers, e.g.
                     "Artikel 5" → "Artikel fünf", unless the text explicitly
                     uses an ordinal form.

  DATE             – value: ["DD", "MM", "YYYY"]
                     Use "" for missing parts e.g. ["", "01", "2026"]
                     When a date contains a written-out month, include the full
                     date expression verbatim in "number", e.g.
                     "Januar 2016" or "01 September 2016".
                     Examples: "Januar 2016" → ["", "01", "2016"];
                     "01 September 2016" → ["01", "09", "2016"].
                     Numeric dates without an ordinal dot can still be DATE
                     when their structure and context clearly indicate a date.
                     Decade expressions such as "1990er-Jahre", "1990er Jahre",
                     or "1960er" are years: annotate only the four digits
                     ("1990" or "1960"), use value ["", "", "1990"] or
                     ["", "", "1960"], and classify them as DATE. Leave the
                     suffix "er" and any following word outside the annotation.

  TIME             – value: ["hh", "mm", "ss"]
                     Use "" for missing parts.
                     If a time is immediately followed by "Uhr", include
                     "Uhr" in the contiguous "number" span. This lets the
                     Swiss German normalizer replace the complete expression
                     without retaining "Uhr" in its output. Do not include a
                     preceding word such as "um". Examples: "um 10 Uhr" →
                     number "10 Uhr", value ["10", "", ""]; "10:30 Uhr" →
                     number "10:30 Uhr", value ["10", "30", ""].

  DECIMAL          – value: ["integer_part", "fractional_part"]
                     e.g. 3,45 → ["3", "45"]
                     Both "," and "." can be decimal separators depending on
                     source formatting; a period does not automatically make a
                     number an ordinal.

  FRACTION         – value: ["numerator", "denominator"]

  MIXED_FRACTION   – value: {"whole":"2","numerator":"1","denominator":"2"}
                     for expressions such as "2 1/2". Annotate the complete
                     mixed fraction as one item, never as a cardinal plus a
                     separate fraction.

  CURRENCY         – value: {"major": "...", "minor": "...", "currency": "ISO"}
                     e.g. CHF 10.45 →
                     {"major": "10", "minor": "45", "currency": "CHF"}
                     Include the complete amount and currency marker in "number".

  IBAN             – value: digits/letters grouped with a space after every 4
                     characters e.g. "CH93 0076 2011 6238 5295 7"

  PHONE            – value: digits only, no spaces or separators

  PIN              – short access/security code read digit by digit. Preserve
                     all digits including leading zeros; value is digits only.

  PRODUCT_ID       – identifiers and codes that are read letter-by-letter,
                     number-by-number, or in chunks, including product/model
                     IDs, ISBNs, ISO/Unicode codes, chemical-style compact
                     formulas such as "CO2", and forms such as
                     "ISBN 3-85124-698-5", "ISO 3166-2", "U+0339",
                     or "A12/3". Keep the complete identifier in "number" and
                     "value". Separators such as "/" and "-" are structural
                     and should be preserved in the spoken output, e.g.
                     "A12/3" -> "a zwölf / drü".

  NUMBER_WITH_SUFFIX – value: {"number": "digits", "suffix": "letter"}
                     Prefer NUMBER_CARDINAL for ordinary numbers when the
                     suffix can remain outside the annotated number span. Use
                     NUMBER_WITH_SUFFIX only when the compact expression needs
                     one replacement to avoid malformed text. Four-digit
                     decade/year forms ending in "er" are the DATE exception
                     described above, not NUMBER_WITH_SUFFIX.

  ZIP              – Swiss four-digit postal code; value is the same four digits

  VERSION          – software/firmware version, e.g. "v2.4.1"; value is the
                     complete version string. Also classify dotted IPv4
                     addresses as VERSION, e.g. "130.94.122.195"; keep the
                     complete address in both "number" and "value".

  MATH_EXPRESSION  – complete symbolic expression/equation, including all
                     numbers and operators; value is the same complete string

  Roman numerals   – do NOT use a separate ROMAN_NUMERAL type. Use
                     NUMBER_CARDINAL or NUMBER_ORDINAL according to context,
                     set value to the Arabic equivalent, and add
                     "notation": "ROMAN". Example: "Kapitel XIX" has
                     number "XIX", value "19", type NUMBER_CARDINAL, and
                     notation "ROMAN"; an explicitly ordinal use gets
                     type NUMBER_ORDINAL plus the normal declension fields.

  RANGE            – value: [first_mentioned_in_text, second_mentioned_in_text]
                     Preserve text order — do NOT sort.
                     Annotate the complete range as one RANGE item, including
                     its delimiter; do not annotate its endpoints separately.
                     A RANGE must express "from … to …" or "between … and …".
                     Expressions delimited with ":" are RATIO, not RANGE.
                     Expressions delimited with "/" are RANGE only when they are
                     compact date/year ranges such as "1998/99" or
                     "01.02.1998/03.02.1998"; set "subtype": "DATE". For this
                     subtype, prefer endpoint objects in value so each side can
                     reuse an existing number type, e.g. "1998/99" →
                     [{"type":"DATE","value":["","","1998"]},
                      {"type":"DATE","value":["","","99"],"subtype":"YEAR_SHORT"}].
                     Legacy endpoint values such as
                     [["", "", "1998"], ["", "", "99"]] are also accepted.
                     Slash date/year ranges are read compactly without "bis".
                     Other slash-delimited expressions are FRACTION,
                     PRODUCT_ID, VERSION, MATH_EXPRESSION, or OTHER_NUMERIC.

  RATIO            – value: [first_mentioned_in_text, second_mentioned_in_text]
                     Use for colon-delimited ratios such as "1:1000". Include
                     the complete expression in "number" and preserve text
                     order. Ratios are spoken with "zu", never "bis".
                     Set "subtype": "SCALE" for map scales or technical scales
                     such as "1:1000". Do not use RATIO for dates.

  COORDINATE       – geographic latitude/longitude in degree notation.
                     value: {"degrees":"70","minutes":"40","seconds":"11",
                             "direction":"O"}
                     Use "" for missing parts. Include the direction letter
                     when it is contiguous or directly follows the coordinate.
                     Example: 70°40'11" O -> "siebzig Grad, vierzig Minuten,
                     elf Sekunden Ost".
   OTHER_NUMERIC    – anything that does not fit the above categories

─── SUBTYPES ─────────────────────────────────────────────────────────────────
  Use "subtype" only when it changes how the item should be read:
    DATE     – RANGE endpoints are dates/years and should be read as dates.
    YEAR_SHORT – abbreviated year endpoint such as "99" or "00".
    SCALE    – RATIO is a map/technical scale and should be read with "zu".
    NEGATIVE – the leading minus sign denotes a negative value.

─── NEGATIVE NUMBERS ──────────────────────────────────────────────────────────
  Preserve the normal semantic type for negative values and additionally set
  "subtype": "NEGATIVE". Include the leading minus sign in both "number" and
  the signed component of "value". Examples:
    -24   → number "-24", value "-24", type NUMBER_CARDINAL,
            subtype NEGATIVE
    -3,5  → number "-3,5", value ["-3", "5"], type DECIMAL,
            subtype NEGATIVE
  Use NEGATIVE only when the minus sign denotes a negative value, not a range,
  hyphen, identifier, date, phone number, or mathematical subtraction.

─── IMPORTANT ──────────────────────────────────────────────────────────────────
  Respond ONLY with valid JSON. No markdown fences, no explanation.
  Every annotation must contain exact zero-based character offsets "start"
  (inclusive) and "end" (exclusive), measured in the returned text. Verify
  text[start:end] equals number exactly. Never identify an occurrence only by
  its repeated surface text.
  "number" is the complete contiguous numeric expression copied verbatim from
  the sentence. For dates, ranges, and ratios, include all their date parts or
  delimiters. For TIME, include a directly following "Uhr" as specified
  above. Otherwise, do not include surrounding words, units, or symbols. The
  decade/year rule is an explicit exception: copy only the four year digits.
  Do not create overlapping annotations.
  Annotate ALL numbers present in the sentence, including ones not in the hints.
"""


def make_user_message(text: str, hints: list[dict]) -> str:
    hints_str = json.dumps(hints, ensure_ascii=False)
    return (
        f"Sentence: {json.dumps(text, ensure_ascii=False)}\n"
        f"Pre-detected number hints (may be incomplete or coarsely typed): {hints_str}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="qwen3vl_run_01", help="Prefix used for the output file")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="Collected JSONL input (repeat for multiple files; default: german_commons_numbers.jsonl)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output prompts JSON path")
    parser.add_argument("--n", type=int, default=1000000, help="Samples per input file")
    args = parser.parse_args()
    require_compute_node("number-label prompt generation")

    input_files = args.input or [DEFAULT_INPUT]
    output_file = args.output or PROMPTS_DIR / f"{args.run_name}_prompts.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_file.name}.", suffix=".tmp", dir=output_file.parent
    )
    total = 0
    source_tags: set[str] = set()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write("[\n")
            first = True
            for jsonl_path in input_files:
                if not jsonl_path.exists():
                    print(f"WARNING: {jsonl_path} not found — skipping.")
                    continue

                source_tag = jsonl_path.stem
                if source_tag in source_tags:
                    raise ValueError(
                        f"Input filename stem {source_tag!r} occurs more than once; "
                        "prompt IDs would not be unique"
                    )
                source_tags.add(source_tag)
                count = 0
                with jsonl_path.open(encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        if count >= args.n:
                            break
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSONL at {jsonl_path}:{line_number}: {exc}"
                            ) from exc
                        text = record["text"]
                        hints = record.get("numbers", [])
                        prompt = {
                            "id": f"{source_tag}_{count:04d}",
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": make_user_message(text, hints),
                                },
                            ],
                        }
                        if not first:
                            output.write(",\n")
                        json.dump(prompt, output, ensure_ascii=False)
                        first = False
                        count += 1
                        total += 1
                print(f"  {source_tag}: {count} prompts added")
            output.write("\n]\n")
        os.replace(temporary_name, output_file)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    print(f"\nWrote {total} prompts → {output_file}")


if __name__ == "__main__":
    main()
