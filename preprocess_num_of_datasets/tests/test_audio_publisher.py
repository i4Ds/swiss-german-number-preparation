import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_audio_recording_fhnw.py"
SPEC = importlib.util.spec_from_file_location("publish_audio_recording_fhnw", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AudioPublisherSchemaTest(unittest.TestCase):
    def test_preserves_source_columns_and_appends_written_text(self):
        source = {
            "id": "row-1",
            "de": "Im Jahr 2024.",
            "ch": "Im Johr 2024.",
            "duration_s": 1.5,
            "thumb": "up",
            "audio": "clip.flac",
        }

        record = MODULE.build_output_record(
            source,
            Path("/data/clip.flac"),
            "Im Johr zweituusigvieräzwänzg.",
        )

        self.assertEqual(record["de"], source["de"])
        self.assertEqual(record["ch"], source["ch"])
        self.assertEqual(record["thumb"], source["thumb"])
        self.assertEqual(record[MODULE.OUTPUT_COLUMN], "Im Johr zweituusigvieräzwänzg.")
        self.assertEqual(list(record)[-1], MODULE.OUTPUT_COLUMN)


if __name__ == "__main__":
    unittest.main()
