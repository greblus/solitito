"""The Kaggle entry point must work without previous outputs or repo imports."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from build_onset_kaggle import build
from test_onset_preparation import fixture


class KaggleScriptTests(unittest.TestCase):
    def test_generated_script_matches_sources(self):
        directory = Path(__file__).resolve().parent
        self.assertEqual((directory / "prepare_onset_kaggle.py").read_text(), build(directory))

    def test_whole_pipeline_without_prior_json_cli_and_pasted_script(self):
        original = Path(__file__).with_name("prepare_onset_kaggle.py").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            manifest, _ = fixture(inputs)
            manifest.unlink()  # Only WAV/JAMS exist when each standalone run starts.
            for pasted in (False, True):
                output = root / ("pasted-output" if pasted else "cli-output")
                code = original.replace('INPUT_DIR = "/kaggle/input"', f"INPUT_DIR = {str(inputs)!r}")
                code = code.replace('OUTPUT_ROOT = "/kaggle/working"', f"OUTPUT_ROOT = {str(output)!r}")
                code = code.replace("SYNTHETIC_GROUPS = (60, 12, 12)", "SYNTHETIC_GROUPS = (1, 1, 1)")
                if pasted:
                    code = 'import sys\nsys.modules["ipykernel"] = object()\n' + code
                script = root / "standalone.py"
                script.write_text(code)
                command = [str(Path(sys.executable).absolute()), str(script)]
                if pasted:
                    command += ["-f", "kernel.json"]
                run = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                reports = list(output.glob("*/summary.json"))
                self.assertEqual(len(reports), 1)
                summary = json.loads(reports[0].read_text())
                self.assertTrue(summary["ok"])
                self.assertEqual(sum(s["clips"] for s in summary["synthetic"].values()), 24)
                self.assertEqual(sum(s["sources"] for s in summary["guitarset"].values()), 6)
                self.assertTrue(Path(summary["input_manifest"]).is_file())
                self.assertFalse(manifest.exists())

    def test_failed_audit_stops_before_synthetic_generation(self):
        script = Path(__file__).with_name("prepare_onset_kaggle.py").resolve()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "empty-input"
            inputs.mkdir()
            output = root / "output"
            run = subprocess.run([sys.executable, str(script), "--input-dir", str(inputs),
                                  "--output-root", str(output), "--groups", "1", "1", "1"],
                                 capture_output=True, text=True, timeout=60)
            self.assertNotEqual(run.returncode, 0)
            reports = list(output.glob("*/summary.json"))
            self.assertEqual(len(reports), 1)
            self.assertFalse(json.loads(reports[0].read_text())["ok"])
            self.assertFalse(list(output.rglob("*.wav")))


if __name__ == "__main__":
    unittest.main()
