import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELING_DIR = PROJECT_ROOT / "scripts" / "labeling"
sys.path.insert(0, str(LABELING_DIR))

from convert_ch_numbers import convert_number  # noqa: E402
from generate_prompts import SYSTEM_PROMPT  # noqa: E402
from generate_synthetic_labeled import (  # noqa: E402
    PRESETS,
    copy_checkpoint_excluding_categories,
    user_prompt,
    validate,
)
from prepare_jsonl import has_numeric_digit, has_slash_range, reclassify_colon_ranges  # noqa: E402
from prepare_jsonl import (  # noqa: E402
    iter_result_records,
    locate_number_spans,
    replace_numbers_in_text,
    run_name_from_labeled_path,
)


class RatioAndRangeRulesTest(unittest.TestCase):
    def test_converter_supports_all_dialects_and_defaults_to_sg(self):
        annotation = {"number": "5", "value": "5", "type": "NUMBER_CARDINAL"}

        self.assertEqual(convert_number(annotation), "füf")
        self.assertEqual(convert_number(annotation, dialect="ch_sg"), "füf")
        self.assertEqual(convert_number(annotation, dialect="ch_bs"), "fünf")
        self.assertEqual(convert_number(annotation, dialect="ch_zh"), "foif")

    def test_ratio_uses_zu_and_range_uses_bis(self):
        ratio = convert_number(
            {"number": "1:1000", "value": ["1", "1000"], "type": "RATIO"}
        )
        numeric_range = convert_number(
            {"number": "1 bis 1000", "value": ["1", "1000"], "type": "RANGE"}
        )

        self.assertIn(" zu ", ratio)
        self.assertNotIn(" bis ", ratio)
        self.assertIn(" bis ", numeric_range)

    def test_legacy_colon_range_is_reclassified(self):
        numbers = [
            {"number": "1:1000", "value": ["1", "1000"], "type": "RANGE"}
        ]
        self.assertEqual(reclassify_colon_ranges(numbers), 1)
        self.assertEqual(numbers[0]["type"], "RATIO")

    def test_slash_range_detection_ignores_other_types(self):
        self.assertTrue(
            has_slash_range(
                [{"number": "1905/2006", "value": ["1905", "2006"], "type": "RANGE"}]
            )
        )
        self.assertFalse(
            has_slash_range(
                [{"number": "1/2", "value": ["1", "2"], "type": "FRACTION"}]
            )
        )
        self.assertFalse(
            has_slash_range(
                [
                    {
                        "number": "1998/99",
                        "value": [["", "", "1998"], ["", "", "99"]],
                        "type": "RANGE",
                        "subtype": "DATE",
                    }
                ]
            )
        )

    def test_preparation_keeps_date_slash_range_and_writes_ratio(self):
        records = [
            self._record(
                "drop_me",
                "Von 1905/2006 blieben 12 Exemplare.",
                [
                    {"number": "1905/2006", "value": ["1905", "2006"], "type": "RANGE"},
                    {"number": "12", "value": "12", "type": "NUMBER_CARDINAL"},
                ],
            ),
            self._record(
                "keep_date_range",
                "Die Saison 1998/99 war ausverkauft.",
                [
                    {
                        "number": "1998/99",
                        "value": [["", "", "1998"], ["", "", "99"]],
                        "type": "RANGE",
                        "subtype": "DATE",
                    }
                ],
            ),
            self._record(
                "keep_ratio",
                "Der Massstab ist 1:1000.",
                [{"number": "1:1000", "value": ["1", "1000"], "type": "RANGE"}],
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "test_results_checkpoint.jsonl"
            input_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            output_dir = temp_path / "04_prepared"
            result = subprocess.run(
                [
                    sys.executable,
                    str(LABELING_DIR / "prepare_jsonl.py"),
                    str(input_path),
                    "--out_dir",
                    str(output_dir),
                    "--number-types",
                    "NUMBER_CARDINAL",
                    "RANGE",
                    "RATIO",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            prepared = json.loads(
                (output_dir / "test_prepared.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [row["id"] for row in prepared], ["keep_date_range", "keep_ratio"]
            )
            self.assertEqual(prepared[0]["numbers"][0]["subtype"], "DATE")
            self.assertNotIn(" bis ", prepared[0]["numbers"][0]["sg_written_form"])
            self.assertEqual(prepared[1]["numbers"][0]["type"], "RATIO")
            self.assertIn(" zu ", prepared[1]["numbers"][0]["sg_written_form"])
            self.assertIn("Removed 1 Good entries containing a slash-delimited RANGE", result.stdout)

    def test_number_with_suffix_is_not_read_digit_by_digit(self):
        spoken = convert_number(
            {
                "number": "24c",
                "value": {"number": "24", "suffix": "c"},
                "type": "NUMBER_WITH_SUFFIX",
            }
        )
        self.assertTrue(spoken.endswith(" ce"))
        self.assertNotIn("zwei vier", spoken)

    def test_number_with_general_suffix_preserves_non_numeric_text(self):
        plain = convert_number(
            {
                "number": "25",
                "value": {"number": "25", "suffix": ""},
                "type": "NUMBER_WITH_SUFFIX",
            }
        )
        embedded = convert_number(
            {
                "number": "7ner-Seilschaft",
                "value": {"number": "7", "suffix": "ner"},
                "type": "NUMBER_WITH_SUFFIX",
            }
        )
        decimal = convert_number(
            {
                "number": "1.8i",
                "value": {"number": "1.8", "suffix": "i"},
                "type": "NUMBER_WITH_SUFFIX",
            }
        )

        self.assertNotEqual(plain, "unknown")
        self.assertTrue(embedded.endswith("ner-Seilschaft"))
        self.assertNotIn("7", embedded)
        self.assertTrue(decimal.endswith(" i"))

    def test_isbn_uses_identifier_conversion(self):
        spoken = convert_number(
            {
                "number": "978-3-033-01382-7",
                "value": "978-3-033-01382-7",
                "type": "ISBN",
            }
        )

        self.assertNotEqual(spoken, "unknown")
        self.assertNotRegex(spoken, r"\d")

    def test_decade_suffix_and_date_range_subtype_convert(self):
        decade = convert_number(
            {
                "number": "1970er",
                "value": {"number": "1970", "suffix": "er"},
                "type": "NUMBER_WITH_SUFFIX",
                "subtype": "DECADE",
            }
        )
        year_range = convert_number(
            {
                "number": "1998/99",
                "value": [
                    {"type": "DATE", "value": ["", "", "1998"]},
                    {"type": "DATE", "value": ["", "", "99"], "subtype": "YEAR_SHORT"},
                ],
                "type": "RANGE",
                "subtype": "DATE",
            }
        )

        self.assertNotEqual(decade, "unknown")

        self.assertTrue(decade.endswith("er"))
        self.assertNotEqual(year_range, "unknown")
        self.assertNotIn(" bis ", year_range)

    def test_version_conversion_allows_spaces_and_prefixes(self):
        for number in ("14. 2. 1", "REV-4.2.1", "2. 1."):
            with self.subTest(number=number):
                spoken = convert_number(
                    {"number": number, "value": number, "type": "VERSION"}
                )
                self.assertNotEqual(spoken, "unknown")
                self.assertNotRegex(spoken, r"\d")

    def test_year_range_uses_compact_reading_only_for_consecutive_years(self):
        consecutive = convert_number(
            {
                "number": "1998/99",
                "value": ["1998", "99"],
                "type": "RANGE",
            }
        )
        distant = convert_number(
            {
                "number": "1900/1999",
                "value": ["1900", "1999"],
                "type": "RANGE",
            }
        )
        self.assertNotIn(" bis ", consecutive)
        self.assertIn(" bis ", distant)
        self.assertNotIn("null", distant)

    def test_pin_and_fractions_have_semantic_readings(self):
        pin = convert_number({"number": "0417", "value": "0417", "type": "PIN"})
        half = convert_number(
            {"number": "2 1/2", "value": ["2", "1/2"], "type": "FRACTION"}
        )
        third = convert_number(
            {"number": "1/3", "value": ["1", "3"], "type": "FRACTION"}
        )
        quarter = convert_number(
            {"number": "1/4", "value": ["1", "4"], "type": "FRACTION"}
        )
        self.assertEqual(pin.split(), ["null", "vio", "eis", "siebe"])
        self.assertEqual(half, "zwei ähalb")
        self.assertEqual(third, "ei Drittel")
        self.assertEqual(quarter, "ei Viertel")

    def test_span_matching_prefers_standalone_number_over_identifier_digit(self):
        text = "CO2 stoht im Ahang 2 und 3."
        numbers = [
            {"number": "CO2", "value": "CO2", "type": "PRODUCT_ID"},
            {"number": "2", "value": "2", "type": "NUMBER_CARDINAL"},
            {"number": "3", "value": "3", "type": "NUMBER_CARDINAL"},
        ]
        spans = locate_number_spans(text, numbers)
        standalone_span = next(span for span in spans if span[2]["number"] == "2")
        self.assertEqual(standalone_span[0], text.rfind("2"))
        incomplete = numbers[1:]
        with self.assertRaisesRegex(ValueError, "Unannotated digit"):
            replace_numbers_in_text(text, incomplete)

    def test_synthetic_multi_time_and_pin_validation(self):
        multi_time = validate(
            {
                "text": "Das Geschäft ist von 08:30 bis 18:45 Uhr geöffnet.",
                "numbers": [
                    {"number": "08:30", "value": ["08", "30", ""]},
                    {"number": "18:45 Uhr", "value": ["18", "45", ""]},
                ],
            },
            "TIME",
            "MULTI",
        )
        pin = validate(
            {
                "text": "Der PIN lautet 0417.",
                "numbers": [{"number": "0417", "value": "0417"}],
            },
            "PIN",
            "DIGITS",
        )
        self.assertIsNotNone(multi_time)
        self.assertEqual(len(multi_time["numbers"]), 2)
        self.assertTrue(all("start" in item and "end" in item for item in multi_time["numbers"]))
        self.assertIsNotNone(pin)
        self.assertEqual(pin["numbers"][0]["type"], "PIN")
        self.assertEqual(PRESETS["tts_edge_cases"][("TIME", "MULTI")], 600)
        self.assertEqual(PRESETS["tts_edge_cases"][("PIN", "DIGITS")], 500)

    def test_identifier_and_coordinate_expansions(self):
        identifier = convert_number(
            {"number": "A12/3", "value": "A12/3", "type": "PRODUCT_ID"}
        )
        hyphenated = convert_number(
            {"number": "ISO 3166-2", "value": "ISO 3166-2", "type": "PRODUCT_ID"}
        )
        formula = convert_number(
            {"number": "CO2", "value": "CO2", "type": "PRODUCT_ID"}
        )
        unicode_codepoint = convert_number(
            {"number": "U+0339", "value": "U+0339", "type": "PRODUCT_ID"}
        )
        punctuated = convert_number(
            {
                "number": "AB.12,3?4!5",
                "value": "AB.12,3?4!5",
                "type": "MODEL_NUMBER",
            }
        )
        sentence_punctuation = convert_number(
            {"number": "A12?!", "value": "A12?!", "type": "PRODUCT_ID"}
        )
        coordinate = convert_number(
            {
                "number": "70°40'11\" O",
                "value": {
                    "degrees": "70",
                    "minutes": "40",
                    "seconds": "11",
                    "direction": "O",
                },
                "type": "COORDINATE",
            }
        )

        self.assertIn(" schreegstrich ", identifier)
        self.assertIn(" bindestrich ", hyphenated)
        self.assertIn("zwei", formula)
        self.assertIn(" plus ", unicode_codepoint)
        self.assertNotIn("tuusig", unicode_codepoint)
        self.assertIn(" punkt ", punctuated)
        self.assertIn(" komma ", punctuated)
        self.assertIn(" frogezeiche ", punctuated)
        self.assertIn(" usruefezeiche ", punctuated)
        self.assertTrue(sentence_punctuation.endswith("?!"))
        self.assertNotIn("frogezeiche", sentence_punctuation)
        self.assertNotIn("usruefezeiche", sentence_punctuation)
        self.assertIn("Grad", coordinate)
        self.assertIn("Minute", coordinate)
        self.assertIn("Sekunde", coordinate)
        self.assertTrue(coordinate.endswith("Ost"))

    def test_negative_subtype_prefixes_normal_conversion_with_minus(self):
        cases = [
            {"number": "-24", "value": "-24", "type": "NUMBER_CARDINAL"},
            {"number": "-3,5", "value": ["-3", "5"], "type": "DECIMAL"},
            {
                "number": "CHF -12.40",
                "value": {"major": "-12", "minor": "40", "currency": "CHF"},
                "type": "CURRENCY",
            },
        ]
        for annotation in cases:
            with self.subTest(annotation=annotation):
                annotation["subtype"] = "NEGATIVE"
                spoken = convert_number(annotation)
                self.assertTrue(spoken.startswith("minus "))
                self.assertEqual(spoken.count("minus"), 1)
                self.assertNotEqual(spoken, "unknown")

    def test_synthetic_topup_validation(self):
        currency = {
            "text": "Der Restbetrag beträgt nur CHF 0.40.",
            "numbers": [
                {
                    "number": "CHF 0.40",
                    "value": {"major": "0", "minor": "40", "currency": "CHF"},
                }
            ],
        }
        suffix = {
            "text": "Der Treffpunkt ist beim Gebäude 24c.",
            "numbers": [
                {"number": "24c", "value": {"number": "24", "suffix": "c"}}
            ],
        }
        negative = {
            "text": "Heute wurden -3,5 Grad gemessen.",
            "numbers": [
                {
                    "number": "-3,5",
                    "value": ["-3", "5"],
                    "type": "DECIMAL",
                    "subtype": "NEGATIVE",
                }
            ],
        }
        decade = {
            "text": "Die Ausstellung erinnert an die 1970er.",
            "numbers": [
                {
                    "number": "1970er",
                    "value": {"number": "1970", "suffix": "er"},
                    "type": "NUMBER_WITH_SUFFIX",
                    "subtype": "DECADE",
                }
            ],
        }
        date_range = {
            "text": "Die Saison 1998/99 bleibt im Archiv.",
            "numbers": [
                {
                    "number": "1998/99",
                    "value": [["", "", "1998"], ["", "", "99"]],
                    "type": "RANGE",
                    "subtype": "DATE",
                }
            ],
        }
        scale_ratio = {
            "text": "Der Plan nutzt den Massstab 1:1000.",
            "numbers": [
                {
                    "number": "1:1000",
                    "value": ["1", "1000"],
                    "type": "RATIO",
                    "subtype": "SCALE",
                }
            ],
        }
        coordinate = {
            "text": "Der Punkt liegt bei 70°40'11\" O.",
            "numbers": [
                {
                    "number": "70°40'11\" O",
                    "value": {
                        "degrees": "70",
                        "minutes": "40",
                        "seconds": "11",
                        "direction": "O",
                    },
                    "type": "COORDINATE",
                }
            ],
        }

        self.assertEqual(
            validate(currency, "CURRENCY_BELOW_ONE", "CHF")["numbers"][0]["type"],
            "CURRENCY",
        )
        currency["numbers"][0]["value"] = '{"major":"0","minor":"01-99","currency":"CHF"}'
        normalized_currency = validate(currency, "CURRENCY_BELOW_ONE", "CHF")
        self.assertEqual(normalized_currency["numbers"][0]["value"]["minor"], "40")
        self.assertEqual(
            validate(suffix, "NUMBER_WITH_SUFFIX", "MIXED")["numbers"][0]["type"],
            "NUMBER_WITH_SUFFIX",
        )
        suffix["numbers"][0]["value"] = '{"number":"24","suffix":"c"}'
        suffix["numbers"][0]["subtype"] = "address"
        normalized_suffix = validate(suffix, "NUMBER_WITH_SUFFIX", "MIXED")["numbers"][0]
        self.assertEqual(normalized_suffix["value"], {"number": "24", "suffix": "c"})
        self.assertNotIn("subtype", normalized_suffix)
        self.assertEqual(
            validate(negative, "NEGATIVE", "DECIMAL")["numbers"][0]["subtype"],
            "NEGATIVE",
        )
        negative["numbers"][0]["value"] = "-3,5"
        normalized_negative = validate(negative, "NEGATIVE", "DECIMAL")
        self.assertEqual(normalized_negative["numbers"][0]["value"], ["-3", "5"])
        negative["text"] = "Heute wurden minus -3,5 Grad gemessen."
        self.assertIsNone(validate(negative, "NEGATIVE", "DECIMAL"))
        normalized_decade = validate(decade, "NUMBER_WITH_SUFFIX", "DECADE")
        self.assertEqual(normalized_decade["numbers"][0]["subtype"], "DECADE")
        normalized_date_range = validate(date_range, "RANGE", "DATE")
        self.assertEqual(
            normalized_date_range["numbers"][0]["value"],
            [["", "", "1998"], ["", "", "99"]],
        )
        normalized_ratio = validate(scale_ratio, "RATIO", "SCALE")
        self.assertEqual(normalized_ratio["numbers"][0]["subtype"], "SCALE")
        normalized_coordinate = validate(coordinate, "COORDINATE", "MIXED")
        self.assertEqual(
            normalized_coordinate["numbers"][0]["value"],
            {"degrees": "70", "minutes": "40", "seconds": "11", "direction": "O"},
        )

    def test_zip_date_time_refresh_validation(self):
        self.assertEqual(
            sum(
                count
                for (category, _), count in PRESETS["zip_date_time_refresh"].items()
                if category == "DATE"
            ),
            1000,
        )
        self.assertEqual(
            sum(
                count
                for (category, _), count in PRESETS["zip_date_time_refresh"].items()
                if category == "TIME"
            ),
            600,
        )
        self.assertEqual(PRESETS["zip_date_time_refresh"][("ZIP", "CH_REAL")], 1500)

        zip_sample = {
            "text": "Das Paket geht an die Abholstelle in 8914 Aeugst am Albis.",
            "numbers": [{"number": "8914", "value": "8914"}],
        }
        wrong_zip_place = {
            "text": "Das Paket geht an die Abholstelle in 8914 Basel.",
            "numbers": [{"number": "8914", "value": "8914"}],
        }
        date_sample = {
            "text": "Der Beratungstermin ist am 03. Aug. 2026.",
            "numbers": [{"number": "03. Aug. 2026", "value": []}],
        }
        year_only = {
            "text": "Der Beratungstermin ist 2026.",
            "numbers": [{"number": "2026", "value": []}],
        }
        time_sample = {
            "text": "Die Abfahrt ist heute um 0630 geplant.",
            "numbers": [{"number": "0630", "value": []}],
        }
        ambiguous_hour = {
            "text": "Die Zahl lautet 8.",
            "numbers": [{"number": "8", "value": []}],
        }

        self.assertIsNotNone(validate(zip_sample, "ZIP", "CH_REAL"))
        self.assertIsNone(validate(wrong_zip_place, "ZIP", "CH_REAL"))
        normalized_date = validate(date_sample, "DATE", "WRITTEN_MONTH")
        self.assertEqual(normalized_date["numbers"][0]["value"], ["03", "08", "2026"])
        self.assertIsNone(validate(year_only, "DATE", "PARTIAL"))
        normalized_time = validate(time_sample, "TIME", "COMPACT")
        self.assertEqual(normalized_time["numbers"][0]["value"], ["06", "30", ""])
        self.assertIsNone(validate(ambiguous_hour, "TIME", "HOUR_CONTEXT"))
        self.assertIn("Never create a\nyear-only", user_prompt("DATE", "PARTIAL", "appointments", "medical", 2))

    def test_checkpoint_category_replacement_is_non_destructive(self):
        records = [
            {"synthetic_category": "ZIP", "id": "old_zip"},
            {"synthetic_category": "VERSION", "id": "keep_version"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            destination = root / "destination.tmp"
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            removed = copy_checkpoint_excluding_categories(
                source, destination, {"ZIP"}
            )
            retained = [json.loads(line) for line in destination.read_text().splitlines()]

        self.assertEqual(removed["ZIP"], 1)
        self.assertEqual(retained, [records[1]])

    def test_digitless_records_are_filtered(self):
        self.assertTrue(
            has_numeric_digit(
                {
                    "text": "Die Saison 1998/99 war lang.",
                    "numbers": [{"number": "1998/99", "value": "1998/99"}],
                }
            )
        )
        self.assertFalse(
            has_numeric_digit(
                {
                    "text": "Dieser Satz enthält nur ausgeschriebene Zahlen.",
                    "numbers": [{"number": "zwei", "value": "zwei"}],
                }
            )
        )

    def test_ordinary_prompt_documents_new_labels(self):
        self.assertIn("NUMBER_WITH_SUFFIX", SYSTEM_PROMPT)
        self.assertIn('"subtype": "NEGATIVE"', SYSTEM_PROMPT)
        self.assertIn('"subtype": "DATE"', SYSTEM_PROMPT)
        self.assertIn('dotted IPv4', SYSTEM_PROMPT)
        self.assertIn('annotate only the four digits', SYSTEM_PROMPT)

    def test_result_reader_accepts_json_array_and_jsonl(self):
        records = [
            {"id": "one", "response": "{}"},
            {"id": "two", "response": "{}"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "demo_results.json"
            jsonl_path = root / "demo_results_checkpoint.jsonl"
            json_path.write_text(json.dumps(records), encoding="utf-8")
            jsonl_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            self.assertEqual(
                [record for _, record in iter_result_records(json_path)], records
            )
            self.assertEqual(
                [record for _, record in iter_result_records(jsonl_path)], records
            )
            self.assertEqual(run_name_from_labeled_path(json_path), "demo")
            self.assertEqual(run_name_from_labeled_path(jsonl_path), "demo")

    def test_preparation_accepts_final_results_json_and_rejects_duplicate_ids(self):
        response = {
            "text": "Die Kiste enthält 24 Teile.",
            "text_quality": "Good",
            "numbers": [{"number": "24", "value": "24", "type": "NUMBER_CARDINAL"}],
        }
        record = {"id": "sample_one", "response": json.dumps(response)}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "demo_results.json"
            output_dir = root / "04_prepared"
            input_path.write_text(json.dumps([record]), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(LABELING_DIR / "prepare_jsonl.py"),
                    str(input_path),
                    "--out_dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            prepared = json.loads(
                (output_dir / "demo_prepared.json").read_text(encoding="utf-8")
            )
            self.assertEqual([item["id"] for item in prepared], ["sample_one"])

            input_path.write_text(json.dumps([record, record]), encoding="utf-8")
            duplicate_run = subprocess.run(
                [
                    sys.executable,
                    str(LABELING_DIR / "prepare_jsonl.py"),
                    str(input_path),
                    "--out_dir",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(duplicate_run.returncode, 0)
            self.assertIn("duplicate IDs", duplicate_run.stderr)

    def test_preparation_accepts_zh_dialect_but_defaults_to_sg(self):
        response = {
            "text": "Die Kiste enthält 5 Teile.",
            "text_quality": "Good",
            "numbers": [{"number": "5", "value": "5", "type": "NUMBER_CARDINAL"}],
        }
        record = {"id": "dialect", "response": json.dumps(response, ensure_ascii=False)}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "dialect_results.json"
            input_path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

            default_dir = root / "default"
            zh_dir = root / "zh"
            base_command = [
                sys.executable,
                str(LABELING_DIR / "prepare_jsonl.py"),
                str(input_path),
            ]
            subprocess.run(base_command + ["--out_dir", str(default_dir)], check=True)
            subprocess.run(
                base_command + ["--out_dir", str(zh_dir), "--dialect", "ch_zh"],
                check=True,
            )

            default_json = json.loads(
                (default_dir / "dialect_prepared.json").read_text(encoding="utf-8")
            )
            zh_json = json.loads(
                (zh_dir / "dialect_prepared.json").read_text(encoding="utf-8")
            )
            with (zh_dir / "dialect_prepared.csv").open(encoding="utf-8") as handle:
                import csv

                zh_csv = list(csv.DictReader(handle))

            self.assertEqual(default_json[0]["numbers"][0]["sg_written_form"], "füf")
            self.assertEqual(zh_json[0]["numbers"][0]["sg_written_form"], "foif")
            self.assertEqual(zh_csv[0]["client_id"], "zh")

    @staticmethod
    def _record(record_id, text, numbers):
        response = {"text": text, "text_quality": "Good", "numbers": numbers}
        return {"id": record_id, "response": json.dumps(response, ensure_ascii=False)}


if __name__ == "__main__":
    unittest.main()
