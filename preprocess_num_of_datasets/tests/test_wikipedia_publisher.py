import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_written_number_column.py"
SPEC = importlib.util.spec_from_file_location("wikipedia_publisher", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PREPROCESSING = MODULE.load_preprocess_module()


class WikipediaPublisherTest(unittest.TestCase):
    def test_automatic_dialect_uses_fine_region_and_marks_fallback(self):
        for label, code in [("Luzern", "ch_lu"), ("Bern", "ch_be"),
                            ("Basel", "ch_bs"), ("Zürich", "ch_zh"),
                            ("St. Gallen", "ch_sg")]:
            self.assertEqual(MODULE.resolve_dialect(
                {"dialect": "ch", "dialect_fine": label}, "auto", "ch_sg"
            ), (code, "dialect_fine"))
        self.assertEqual(MODULE.resolve_dialect(
            {"dialect_fine": "Swiss German"}, "auto", "ch_sg"
        ), ("ch_sg", "fallback"))

    def test_parquet_annotations_and_written_text_preserve_source_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = pa.table({
                "sentence_ch": ["Zahl 132.", "Zahl 132.", "Ohni Zahle.", "Zahl 2.", "Zahl 132."],
                "sentence_de": ["Zahl 132.", "Zahl 132.", "Ohne Zahlen.", "Zahl 2.", "Zahl 132."],
                "dialect": ["lu", "be", "ch", "lu", "vs"],
                "dialect_fine": ["Luzern", "Bern", "Swiss German", "Luzern", "Wallis"],
                "article_id": ["a", "b", "c", "d", "e"],
            })
            source_path = root / "source.parquet"
            pq.write_table(source, source_path)
            run = root / "runs" / "test_run"
            run.mkdir(parents=True)
            manifest = {
                "inputs": [{"path": str(source_path), "sha256": MODULE.sha256_file(source_path)}],
                "sentence_ch_column": "sentence_ch",
                "prompt_rows": [{"row_index": n, "prompt_id": str(n)} for n in (0, 1, 3, 4)],
            }
            (run / "manifest.json").write_text(json.dumps(manifest))
            entries = [{"id": str(n), "response": json.dumps({
                "text": "Zahl 132.", "numbers": [{"number": "132", "value": "132",
                "type": "NUMBER_CARDINAL", "custom": {"keep": [1, 2]}}],
            })} for n in (0, 1, 4)]
            entries.append({"id": "3", "response": "invalid response"})
            (run / "results.json").write_text(json.dumps(entries))
            path, summary, failures = MODULE.build_split(
                split="test", run_name="test_run", runs_dir=root / "runs",
                output_dir=root / "out", dialect="auto", overwrite=False,
                preprocessing=PREPROCESSING,
            )
            output = pq.read_table(path)
            self.assertTrue(output.select(source.column_names).equals(source))
            rows = output.to_pylist()
            self.assertEqual(rows[0][MODULE.OUTPUT_COLUMN], "Zahl eihondertzwoiedriisg.")
            self.assertEqual(rows[1][MODULE.OUTPUT_COLUMN], "Zahl hundertzwöiedrissg.")
            self.assertEqual(rows[0]["number_dialect"], "ch_lu")
            self.assertEqual(rows[1]["number_dialect"], "ch_be")
            self.assertEqual(rows[2]["numbers"], [])
            self.assertIsNone(rows[3]["numbers"])
            self.assertIsNone(rows[3][MODULE.OUTPUT_COLUMN])
            self.assertEqual(rows[3]["number_processing_status"], "inference_error")
            self.assertEqual(rows[4]["number_dialect"], "ch_sg")
            self.assertEqual(rows[4]["number_dialect_source"], "fallback")
            self.assertEqual(rows[4][MODULE.OUTPUT_COLUMN], "Zahl eihundertzweiädriisg.")
            annotation = rows[0]["numbers"][0]
            self.assertEqual(rows[0]["sentence_ch"][annotation["start"]:annotation["end"]], "132")
            self.assertEqual(json.loads(annotation["value_json"]), "132")
            self.assertEqual(json.loads(annotation["annotation_json"])["custom"], {"keep": [1, 2]})
            self.assertEqual(len(failures), 1)
            self.assertEqual(summary["rows"], 5)

    def test_heterogeneous_annotation_values_roundtrip(self):
        import pyarrow as pa

        values = ["132", ["", "", "2024"], {"hour": "14", "minute": "30"}]
        for value in values:
            response = {"numbers": [{"number": "132", "type": "EXAMPLE", "value": value}]}
            annotations = MODULE.export_annotations(response, "Zahl 132.", PREPROCESSING, [])
            restored = pa.array([annotations], type=MODULE.annotation_type(pa)).to_pylist()[0]
            self.assertEqual(json.loads(restored[0]["value_json"]), value)


if __name__ == "__main__":
    unittest.main()
