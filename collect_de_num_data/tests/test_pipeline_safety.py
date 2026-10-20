import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import pipeline_runtime  # noqa: E402


class PipelineSafetyTest(unittest.TestCase):
    def test_storage_stages_are_outside_git_and_ordered(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"DATA_ROOT": directory}
        ):
            self.assertEqual(
                pipeline_runtime.storage_path("raw", "input.jsonl"),
                Path(directory).resolve()
                / "raw"
                / "collect_de_num_data"
                / "input.jsonl",
            )
            self.assertEqual(
                pipeline_runtime.storage_path("results", "final", "output.json"),
                Path(directory).resolve()
                / "results"
                / "collect_de_num_data"
                / "final"
                / "output.json",
            )

    def test_repository_cannot_be_used_as_data_root(self):
        with patch.dict(os.environ, {"DATA_ROOT": str(pipeline_runtime.REPOSITORY_ROOT)}):
            with self.assertRaisesRegex(RuntimeError, "outside the Git repository"):
                pipeline_runtime.data_root()

    def test_compute_work_requires_slurm(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm allocation"):
                pipeline_runtime.require_compute_node("test work")

    def test_prompt_writer_preserves_jsonl_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ordered.jsonl"
            output = root / "prompts.json"
            source.write_text(
                "".join(
                    json.dumps({"text": text, "numbers": []}) + "\n"
                    for text in ("first", "second", "third")
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["DATA_ROOT"] = str(root / "data")
            environment["TTS_ALLOW_LOGIN_NODE"] = "1"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "labeling" / "generate_prompts.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                ],
                check=True,
                env=environment,
                capture_output=True,
                text=True,
            )
            prompts = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [prompt["id"] for prompt in prompts],
                ["ordered_0000", "ordered_0001", "ordered_0002"],
            )


if __name__ == "__main__":
    unittest.main()
