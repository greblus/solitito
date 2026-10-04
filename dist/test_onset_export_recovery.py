"""Numerical boundary regression and recovery without repeating completed training."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from build_train_onset_kaggle import build_recovery
from resume_onset_training import find_recovery_run, recovery_sources, resume_training_run
from test_onset_preparation import fixture
from train_short_onset import (export_probability_check, export_event_comparison, onset_metrics,
                               run_training_pipeline, sha256)

HAS_TRAINING = all(importlib.util.find_spec(n) for n in ('torch', 'onnx', 'onnxruntime'))


class ExportRecoveryTests(unittest.TestCase):
    def source(self):
        return {'id': 'clip', 'frames': 125, 'domain': 'synthetic', 'case': 'root_hold', 'duration': 2.,
                'events': [{'id': 'attack', 't': .16, 'pc': 9, 'end': 2.}]}

    def test_tiny_probability_difference_can_change_threshold_event(self):
        source = self.source()
        a = np.zeros((125, 12), np.float32)
        a[12, 9] = .8 - 1e-7
        b = a.copy()
        b[12, 9] = .8 + 1e-7
        parity = export_probability_check([(source, a)], [(source, b)])
        self.assertTrue(parity['ok'])
        ma, da = onset_metrics([(source, a)], .8)
        mb, db = onset_metrics([(source, b)], .8)
        events = export_event_comparison(da, db)
        self.assertFalse(events['events_identical'])
        self.assertEqual(events['onnx_only_events'], 1)
        self.assertEqual((ma['groups']['all']['tp'], mb['groups']['all']['tp']), (0, 1))

    def test_tiny_rearm_difference_can_change_later_event(self):
        source = self.source()
        a = np.zeros((125, 12), np.float32)
        a[12:16, 9] = [.9, .27 - 1e-7, .85, 0.]
        b = a.copy()
        b[13, 9] = .27 + 1e-7
        self.assertTrue(export_probability_check([(source, a)], [(source, b)])['ok'])
        _, da = onset_metrics([(source, a)], .8)
        _, db = onset_metrics([(source, b)], .8)
        self.assertEqual(export_event_comparison(da, db)['pytorch_only_events'], 1)

    def test_large_or_invalid_export_errors_are_not_hidden(self):
        source = self.source()
        a = np.zeros((125, 12), np.float32)
        b = a.copy()
        b[12, 9] = .001
        self.assertFalse(export_probability_check([(source, a)], [(source, b)])['ok'])
        for other in (b[:100], np.full_like(b, np.nan)):
            with self.assertRaises(ValueError):
                export_probability_check([(source, a)], [(source, other)])
        with self.assertRaises(ValueError):
            export_probability_check([(source, a)], [])

    def test_missing_saved_output_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('resume_onset_training.train_onset_experiment') as train:
                with self.assertRaisesRegex(ValueError, 'No training was started'):
                    find_recovery_run('auto', [tmp])
                train.assert_not_called()

    def test_generated_recovery_script_has_no_repository_imports(self):
        directory = Path(__file__).parent
        generated = build_recovery(directory)
        self.assertNotIn('from train_short_onset import', generated)
        compile(generated, 'recovery.py', 'exec')

    @unittest.skipUnless(HAS_TRAINING, 'temporary Torch/ONNX environment required')
    def test_recovery_of_renamed_dataset_without_retraining_or_original_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / 'input'
            inputs.mkdir()
            fixture(inputs)
            initial = run_training_pipeline(inputs, root / 'original', groups=(1, 1, 1), epochs=1, device='cpu')
            original = Path(initial['summary_path']).parent
            # Simulate a Save Version output attached at an arbitrary new path.
            attached = root / 'attached' / 'dataset' / 'renamed'
            attached.parent.mkdir(parents=True)
            shutil.move(str(original), str(attached))
            shutil.rmtree(inputs)
            (attached / 'training_summary.json').write_text(json.dumps({'ok': False, 'error': 'export event mismatch'}))
            for arm in ('control', 'rise'):
                for name in ('validation_events.json', 'test_events.json', 'training_summary.json'):
                    (attached / arm / name).unlink()
                shutil.rmtree(attached / arm / 'probabilities')
            expected = {(arm, file): sha256(attached / arm / file) for arm in ('control', 'rise')
                        for file in ('short_onset_best.pt', 'history.json', 'contract.json')}
            found = find_recovery_run('auto', [root / 'working', root / 'attached'])
            self.assertEqual(found, attached)
            self.assertEqual(len(recovery_sources(found)), 30)
            with patch('resume_onset_training.train_onset_experiment', side_effect=AssertionError('Retraining forbidden')):
                report = resume_training_run(found, root / 'working', 'cpu')
            self.assertTrue(report['ok'])
            self.assertEqual(report['newly_trained_arms'], [])
            self.assertTrue(report['controlled_training_verified'])
            recovered = Path(report['summary_path']).parent
            for (arm, name), digest in expected.items():
                self.assertEqual(sha256(attached / arm / name), digest)
                self.assertEqual(sha256(recovered / arm / name), digest)
            self.assertFalse(json.loads((attached / 'training_summary.json').read_text())['ok'])
            # A corrupt cache must be rejected, never repaired by regenerating data.
            damaged = next((recovered / 'features').glob('*.npy'))
            damaged.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'changed cached'):
                recovery_sources(recovered)
            # Also run the pasted standalone, with control complete and rise NEVER started.
            shutil.rmtree(attached / 'rise')
            directory = Path(__file__).parent
            code = build_recovery(directory)
            code = code.replace('WORK_ROOT = "/kaggle/working"', f'WORK_ROOT = {str(root / "standalone-output")!r}')
            code = code.replace('INPUT_ROOT = "/kaggle/input"', f'INPUT_ROOT = {str(root / "attached")!r}')
            code = 'import sys\nsys.modules["ipykernel"] = object()\n' + code
            script = root / 'resume.py'
            script.write_text(code)
            completed = subprocess.run([sys.executable, str(script), '-f', 'kernel.json'], cwd=root,
                                       capture_output=True, text=True, timeout=180, env=dict(os.environ))
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            reports = list((root / 'standalone-output').glob('*/training_summary.json'))
            self.assertEqual(len(reports), 1)
            final = json.loads(reports[0].read_text())
            self.assertTrue(final['ok'])
            self.assertEqual(final['newly_trained_arms'], ['rise'])
            self.assertEqual(final['arms']['control']['checkpoint_sha256'], expected['control', 'short_onset_best.pt'])
            self.assertEqual(final['arms']['control']['evaluation_backend'], 'ONNX Runtime CPU')
            self.assertEqual(final['arms']['rise']['evaluation_backend'], 'ONNX Runtime CPU')
            # Do not silently restart an interrupted arm.
            (attached / 'control/history.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Incomplete training'):
                resume_training_run(attached, root / 'refused', 'cpu')


if __name__ == '__main__':
    unittest.main()
