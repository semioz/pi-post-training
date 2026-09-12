import subprocess
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SftInfrastructureTest(unittest.TestCase):
    def test_training_script_exposes_trace_training_controls(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "train_sft.py"), "--help"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--dataset-id", "--prepare-only", "--trackio-space-id", "--push-to-hub"):
            self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
