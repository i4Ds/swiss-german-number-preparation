"""Exercise the runner's GPU preflight without importing GPU packages."""
import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


RUNNER = Path(__file__).resolve().parents[1] / 'scripts/run_audio_recording_fhnw_fhnw.sh'
PREFLIGHT = RUNNER.read_text().split("<<'PYTHON'\n", 1)[1].split('\nPYTHON', 1)[0]


class FhnwEnvironmentTest(unittest.TestCase):
    def run_preflight(self, cuda='12.8', vllm_version='0.15.0', devices=2,
                      allocation_error=None, ninja_error=None):
        torch = types.ModuleType('torch')
        torch.__version__ = '2.9.1+cu128'
        torch.version = types.SimpleNamespace(cuda=cuda)
        torch.cuda = types.SimpleNamespace(
            device_count=lambda: devices, synchronize=Mock())
        torch.empty = Mock(side_effect=allocation_error)
        modules = {name: types.ModuleType(name) for name in
                   ('datasets', 'pyarrow', 'soundfile', 'vllm', 'yaml')}
        modules['torch'] = torch
        with patch('subprocess.run', side_effect=ninja_error) as run_ninja, \
                patch.dict(sys.modules, modules), \
                patch('importlib.metadata.version', return_value=vllm_version), \
                patch.object(sys, 'argv', ['-', '2']), \
                contextlib.redirect_stdout(io.StringIO()):
            exec(compile(PREFLIGHT, str(RUNNER), 'exec'), {})
        run_ninja.assert_called_once_with(["ninja", "--version"], check=True)
        return torch

    def test_compatible_stack_exercises_each_gpu(self):
        torch = self.run_preflight()
        self.assertEqual(torch.empty.call_count, 2)
        torch.cuda.synchronize.assert_any_call(0)
        torch.cuda.synchronize.assert_any_call(1)

    def test_missing_ninja_fails_preflight(self):
        with self.assertRaisesRegex(FileNotFoundError, "ninja"):
            self.run_preflight(ninja_error=FileNotFoundError("ninja"))

    def test_rejects_cuda13_torch(self):
        with self.assertRaisesRegex(RuntimeError, 'CUDA 12.8'):
            self.run_preflight(cuda='13.0')

    def test_rejects_original_incompatible_vllm(self):
        with self.assertRaisesRegex(RuntimeError, 'vllm==0.15.0'):
            self.run_preflight(vllm_version='0.26.0')

    def test_rejects_missing_gpu(self):
        with self.assertRaisesRegex(RuntimeError, 'at least 2'):
            self.run_preflight(devices=1)

    def test_driver_failure_is_not_swallowed(self):
        with self.assertRaisesRegex(RuntimeError, 'driver failure'):
            self.run_preflight(allocation_error=RuntimeError('driver failure'))


if __name__ == '__main__':
    unittest.main()
