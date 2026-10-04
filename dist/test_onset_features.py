import ast
import json
from pathlib import Path
import tempfile
import unittest

import librosa
import numpy as np
import soundfile as sf

from audit_onset_features import load_trainer_extractor, read_rust, response


class FeatureAuditTests(unittest.TestCase):
    def test_rust_timestamps_include_initial_fft_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.json"
            rows = [{"end_sample": end, "features": [0] * 168, "magnitude": [0] * 144, "rms": 0}
                    for end in (8192, 8448)]
            document = {"target_rate": 16000, "fft_samples": 8192, "hop_samples": 256, "frames": rows}
            path.write_text(json.dumps(document))
            np.testing.assert_allclose(read_rust(path)[0], [.512, .528], atol=1e-12)
            document["frames"][0]["end_sample"] = 0
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "shifted Rust export"):
                read_rust(path)

    def test_response_uses_physical_time_and_half_open_target(self):
        t = np.arange(100) * .016 + .512
        before = np.zeros((100, 1))
        after = np.clip((t - 1.536) / .256, 0, 1)[:, None]
        report = response(t, before, after, 1.536)
        self.assertAlmostEqual(report["first_50pct_ms"], 128)
        self.assertAlmostEqual(report["peak_fraction_inside_96ms_target"], .080 / .256)
        self.assertEqual(report["max_difference_before_onset"], 0)

    def test_instrumentation_preserves_actual_trainer_features(self):
        trainer = Path(__file__).with_name("model_trainer.py")
        extract = load_trainer_extractor(trainer)
        # Independently run the ORIGINAL function and literal configuration,
        # excluding the trainer's top-level installation/authentication code.
        tree = ast.parse(trainer.read_text())
        nodes = []
        factory, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "chord_runtime"]
        for node in factory.body:
            if isinstance(node, ast.Assign):
                try:
                    ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                nodes.append(node)
            if isinstance(node, ast.FunctionDef) and node.name == "process_audio_file":
                nodes.append(node)
        namespace = {"np": np, "librosa": librosa}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(trainer), "exec"), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            t = np.arange(32000) / 16000
            sf.write(path, np.sin(t * 2 * np.pi * 220).astype(np.float32) * .1, 16000, subtype="FLOAT")
            for boost in (0., 5.):
                namespace.update(BASS_BOOST_ENABLED=boost > 0, BASS_BOOST_GAIN=boost)
                expected, count = namespace["process_audio_file"](str(path))
                times, actual, magnitude = extract(path, boost)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(magnitude.shape, (count, 144))
                self.assertAlmostEqual(times[0], 0)
                self.assertAlmostEqual(times[-1], (count - 1) * .016)


if __name__ == "__main__":
    unittest.main()
