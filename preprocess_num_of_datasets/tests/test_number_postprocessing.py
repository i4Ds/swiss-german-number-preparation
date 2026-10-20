import importlib.util
import json
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_dataset_numbers.py"
SPEC = importlib.util.spec_from_file_location("preprocess_dataset_numbers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NumberPostprocessingTest(unittest.TestCase):
    def test_span_matching_uses_offsets_and_rejects_unannotated_digits(self):
        text = "CO2 stoht im Ahang 2."
        standalone = text.rfind("2")
        response = {
            "text": text,
            "numbers": [
                {
                    "number": "2",
                    "value": "2",
                    "type": "NUMBER_CARDINAL",
                    "start": standalone,
                    "end": standalone + 1,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "unannotated digit"):
            MODULE.apply_annotations(
                text,
                response,
                converter=lambda info, dialect: "zwei",
                dialect="ch_sg",
            )

        response["numbers"].insert(
            0,
            {
                "number": "CO2",
                "value": "CO2",
                "type": "PRODUCT_ID",
                "start": 0,
                "end": 3,
            },
        )
        converted, _ = MODULE.apply_annotations(
            text,
            response,
            converter=lambda info, dialect: "ce o zwei" if info["number"] == "CO2" else "zwei",
            dialect="ch_sg",
        )
        self.assertEqual(converted, "ce o zwei stoht im Ahang zwei.")

    def test_annotations_can_use_original_text_when_response_text_was_rewritten(self):
        response = {
            "text": "nach 12",
            "numbers": [
                {"number": "12", "value": "12", "type": "NUMBER_CARDINAL"}
            ],
        }

        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            MODULE.apply_annotations(
                "noch 12",
                response,
                converter=lambda info, dialect: "zwölf",
                dialect="ch_sg",
            )

        converted, _ = MODULE.apply_annotations(
            "noch 12",
            response,
            converter=lambda info, dialect: "zwölf",
            dialect="ch_sg",
            require_response_text_match=False,
        )
        self.assertEqual(converted, "noch zwölf")

    def test_other_numeric_fallbacks(self):
        response = {
            "text": "IP 130.94.122.195, Zahl 1234, Code A12-3 und Wort abc.",
            "numbers": [
                {"number": "130.94.122.195", "value": [], "type": "OTHER_NUMERIC"},
                {"number": "1234", "value": [], "type": "OTHER_NUMERIC"},
                {"number": "A12-3", "value": [], "type": "OTHER_NUMERIC"},
                {"number": "abc", "value": "abc", "type": "OTHER_NUMERIC"},
            ],
        }

        stats = MODULE.normalize_number_annotations(response)

        self.assertEqual(
            [number["type"] for number in response["numbers"]],
            ["VERSION", "NUMBER_CARDINAL", "PRODUCT_ID"],
        )
        self.assertEqual(stats["digitless_removed"], 1)

    def test_leading_zero_and_long_integer_become_product_ids(self):
        response = {
            "text": "Codes 0123 und 12345.",
            "numbers": [
                {"number": "0123", "value": "0123", "type": "OTHER_NUMERIC"},
                {"number": "12345", "value": "12345", "type": "OTHER_NUMERIC"},
            ],
        }
        MODULE.normalize_number_annotations(response)
        self.assertEqual(
            [number["type"] for number in response["numbers"]],
            ["PRODUCT_ID", "PRODUCT_ID"],
        )

    def test_decade_keeps_only_year_as_date(self):
        response = {
            "text": "In de 1990er-Johr.",
            "numbers": [
                {"number": "1990er-Johr", "value": "1990er", "type": "OTHER_NUMERIC"}
            ],
        }
        MODULE.normalize_number_annotations(response)
        self.assertEqual(
            response["numbers"][0],
            {"number": "1990", "value": ["", "", "1990"], "type": "DATE"},
        )

        suffix_response = {
            "text": "D 1960er.",
            "numbers": [
                {
                    "number": "1960er",
                    "value": {"number": "1960", "suffix": "er"},
                    "type": "NUMBER_WITH_SUFFIX",
                    "subtype": "DECADE",
                }
            ],
        }
        MODULE.normalize_number_annotations(suffix_response)
        self.assertEqual(
            suffix_response["numbers"][0],
            {"number": "1960", "value": ["", "", "1960"], "type": "DATE"},
        )

    def test_decade_offsets_preserve_suffix_during_replacement(self):
        for surface, suffix in [("1960er", "er"), ("1990er-Johr", "er-Johr")]:
            with self.subTest(surface=surface):
                text = f"Ära {surface}."
                start = text.index(surface)
                response = MODULE.parse_llm_response(json.dumps({
                    "text": text,
                    "numbers": [{
                        "number": surface, "value": surface,
                        "type": "NUMBER_WITH_SUFFIX", "subtype": "DECADE",
                        "start": start, "end": start + len(surface),
                    }],
                }))
                annotation = response["numbers"][0]
                self.assertEqual(annotation["number"], surface[:4])
                self.assertEqual((annotation["start"], annotation["end"]),
                                 (start, start + 4))
                converted, _ = MODULE.apply_annotations(
                    text, response, converter=lambda info, dialect: "YEAR",
                    dialect="ch_sg",
                )
                self.assertEqual(converted, f"Ära YEAR{suffix}.")
                before = json.dumps(response)
                MODULE.normalize_number_annotations(response)
                self.assertEqual(json.dumps(response), before)

    def test_coordinate_offsets_expand_at_the_correct_repeated_occurrence(self):
        surface = "22° 08'"
        text = f"Ä {surface} N; {surface} O."
        starts = [text.index(surface), text.rindex(surface)]
        response = {"text": text, "numbers": [
            {"number": surface, "value": [], "type": "OTHER_NUMERIC",
             "start": start, "end": start + len(surface)}
            for start in starts
        ]}
        MODULE.normalize_number_annotations(response, coordinates_only=True)
        for annotation, start, direction in zip(response["numbers"], starts, ["N", "O"]):
            self.assertEqual(annotation["start"], start)
            self.assertEqual(annotation["end"], start + len(surface) + 2)
            self.assertEqual(annotation["number"], surface + " " + direction)
            self.assertEqual(annotation["value"]["direction"], direction)
            self.assertEqual(text[annotation["start"]:annotation["end"]], annotation["number"])
        converted, _ = MODULE.apply_annotations(
            text, response, converter=lambda info, dialect: info["value"]["direction"],
            dialect="ch_sg",
        )
        self.assertEqual(converted, "Ä N; O.")

    def test_coordinate_rule_does_not_extend_into_direction_words(self):
        surface = "22° 08'"
        text = f"{surface} Ost."
        response = {"text": text, "numbers": [{
            "number": surface, "type": "OTHER_NUMERIC", "value": [],
            "start": 0, "end": len(surface),
        }]}
        MODULE.normalize_number_annotations(response)
        self.assertEqual(response["numbers"][0]["number"], surface)
        self.assertEqual(response["numbers"][0]["end"], len(surface))

    def test_span_normalization_rejects_invalid_original_offsets(self):
        for surface in ["1960er", "22° 08'"]:
            for offsets in [{"start": 0}, {"start": True, "end": len(surface)},
                            {"start": -1, "end": len(surface)},
                            {"start": 0, "end": 999},
                            {"start": 1, "end": len(surface)}]:
                with self.subTest(surface=surface, offsets=offsets):
                    response = {"text": surface, "numbers": [{
                        "number": surface, "type": "OTHER_NUMERIC", "value": [],
                        **offsets,
                    }]}
                    with self.assertRaisesRegex(ValueError, "[Oo]ffsets"):
                        MODULE.normalize_number_annotations(response)

    def test_value_only_normalization_preserves_offsets(self):
        text = "Im Juli 1900."
        start = text.index("1900")
        response = {"text": text, "numbers": [{
            "number": "1900", "type": "DATE", "value": ["", "07", "1900"],
            "start": start, "end": start + 4,
        }]}
        MODULE.normalize_number_annotations(response)
        annotation = response["numbers"][0]
        self.assertEqual(annotation["value"], ["", "", "1900"])
        self.assertEqual((annotation["start"], annotation["end"]), (start, start + 4))

    def test_year_only_date_drops_inferred_adjacent_month(self):
        response = {
            "text": "Am 1. Juli 1900 het er sis Gschäft uuftaa.",
            "numbers": [
                {
                    "number": "1.",
                    "value": "1",
                    "type": "NUMBER_ORDINAL",
                },
                {
                    "number": "1900",
                    "value": ["", "07", "1900"],
                    "type": "DATE",
                },
            ],
        }

        stats = MODULE.normalize_number_annotations(response)

        self.assertEqual(response["numbers"][1]["value"], ["", "", "1900"])
        self.assertEqual(stats["date_values_clipped"], 1)

    def test_complete_written_date_keeps_all_components(self):
        response = {
            "text": "Am 29. Februar 2024 het er Geburtstag.",
            "numbers": [
                {
                    "number": "29. Februar 2024",
                    "value": ["29", "02", "2024"],
                    "type": "DATE",
                }
            ],
        }

        stats = MODULE.normalize_number_annotations(response)

        self.assertEqual(
            response["numbers"][0]["value"], ["29", "02", "2024"]
        )
        self.assertEqual(stats["date_values_clipped"], 0)

    def test_day_only_date_drops_inferred_adjacent_month(self):
        response = {
            "text": "Am 14. Juni isch Paris bsetzt worde.",
            "numbers": [
                {
                    "number": "14.",
                    "value": ["14", "06", ""],
                    "type": "DATE",
                }
            ],
        }

        stats = MODULE.normalize_number_annotations(response)

        self.assertEqual(response["numbers"][0]["value"], ["14", "", ""])
        self.assertEqual(stats["date_values_clipped"], 1)

    def test_conversion_rejects_a_remaining_duplicated_month(self):
        response = {
            "text": "Endi März 07.",
            "numbers": [
                {
                    "number": "07",
                    "value": ["", "03", "2007"],
                    "type": "DATE",
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "duplicated month"):
            MODULE.apply_annotations(
                response["text"],
                response,
                converter=lambda info, dialect: "März zweituusigsiebe",
                dialect="ch_sg",
            )

    def test_coordinate_is_restructured_and_standalone_direction_is_included(self):
        response = {
            "text": "De Punkt ligt bi 22° 08' O.",
            "numbers": [
                {"number": "22° 08'", "value": ["22", "08"], "type": "OTHER_NUMERIC"}
            ],
        }
        MODULE.normalize_number_annotations(response, coordinates_only=True)
        self.assertEqual(response["numbers"][0]["type"], "COORDINATE")
        self.assertEqual(response["numbers"][0]["number"], "22° 08' O")
        self.assertEqual(
            response["numbers"][0]["value"],
            {"degrees": "22", "minutes": "08", "seconds": "", "direction": "O"},
        )


if __name__ == "__main__":
    unittest.main()
