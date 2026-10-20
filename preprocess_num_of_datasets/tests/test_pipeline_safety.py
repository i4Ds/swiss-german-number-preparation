import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline_runtime  # noqa: E402


class PipelineSafetyTest(unittest.TestCase):
    def test_processed_data_uses_persistent_stage(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"DATA_ROOT": directory}
        ):
            self.assertEqual(
                pipeline_runtime.storage_path("processed", "run", "dataset.parquet"),
                Path(directory).resolve()
                / "processed"
                / "preprocess_num_of_datasets"
                / "run"
                / "dataset.parquet",
            )

    def test_repository_cannot_be_used_as_data_root(self):
        with patch.dict(os.environ, {"DATA_ROOT": str(pipeline_runtime.REPOSITORY_ROOT)}):
            with self.assertRaisesRegex(RuntimeError, "outside the Git repository"):
                pipeline_runtime.data_root()

    def test_compute_work_requires_slurm(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm allocation"):
                pipeline_runtime.require_compute_node("test work")


if __name__ == "__main__":
    unittest.main()
