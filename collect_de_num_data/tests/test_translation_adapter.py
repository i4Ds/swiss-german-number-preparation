import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/translation/t5_adapter.py"
SPEC = importlib.util.spec_from_file_location("publication_t5_adapter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TranslationPromptCompatibilityTest(unittest.TestCase):
    def test_five_dialect_prefixes_match_existing_translator_contract(self):
        for code, name in [("bs", "Basel"), ("sg", "St. Gallen"), ("zh", "Zurich"),
                           ("lu", "Lucerne"), ("be", "Bern")]:
            expected = f"translate German to {name} Swiss German: Zahl xax."
            self.assertEqual(MODULE.translation_prompt("Zahl xax.", code), expected)
            self.assertEqual(MODULE.translation_prompt("Zahl xax.", "ch_" + code), expected)

    def test_unknown_target_is_not_silently_translated_to_another_dialect(self):
        with self.assertRaises(ValueError):
            MODULE.translation_prompt("Zahl xax.", "unknown")


if __name__ == "__main__":
    unittest.main()
