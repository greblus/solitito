"""Causality, context, initialization and gradient checks for spectral rise."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from onset_rise import RISE_PAST_FRAMES, positive_spectral_rise
from train_short_onset import FEATURE_DIM, HISTORY, OnsetBlocks, feature_block, make_onset_model


@unittest.skipUnless(importlib.util.find_spec("torch"), "temporary Torch environment required")
class OnsetRiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_rise_matches_independent_amplitude_reference_and_excludes_current(self):
        import torch
        # Different bands: constant, decay, reattack and delayed isolated note.
        amplitude = np.array([[3.] * 10, [8., 7., 6., 5., 4., 3., 2., 1., 0., 0.],
                              [0., 4., 4., 4., 4., 4., 8., 7., 6., 5.],
                              [0., 0., 0., 0., 0., 0., 0., 5., 0., 0.]])
        expected = np.zeros_like(amplitude)
        for band, values in enumerate(amplitude):
            for frame, value in enumerate(values):
                previous = sum(values[max(0, frame - 4):frame]) / 4
                expected[band, frame] = math.log1p(max(0, value - previous)) / math.log(1001)
        features = np.log1p(amplitude) / math.log(1001)
        actual = positive_spectral_rise(torch.from_numpy(features[None])).numpy()[0]
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        np.testing.assert_array_equal(actual[0, 4:], 0)
        np.testing.assert_array_equal(actual[1, 4:], 0)
        self.assertGreater(actual[2, 6], 0)
        self.assertAlmostEqual(actual[3, 7], features[3, 7])

    def test_zero_extension_preserves_backbone_rng_answer_and_receives_gradients(self):
        import torch
        torch.manual_seed(29)
        control = make_onset_model()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(29)
        rise = make_onset_model(True)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for key, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, rise.state_dict()[key]), key)
        self.assertEqual(torch.count_nonzero(rise.rise_project.weight).item(), 0)
        x = torch.rand(2, FEATURE_DIM, 90)
        torch.testing.assert_close(control(x), rise(x), rtol=0, atol=0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(rise(x), torch.ones(2, 12, 90))
        loss.backward()
        self.assertGreater(float(rise.rise_project.weight.grad.abs().sum()), 0)

    def test_learned_rise_is_causal_and_matches_full_file_at_chunk_boundaries(self):
        import torch
        torch.manual_seed(17)
        model = make_onset_model(True).eval()
        # A zero projection would hide missing rise context, so exercise nonzero weights.
        torch.nn.init.normal_(model.rise_project.weight, std=.025)
        features = np.random.default_rng(12).uniform(0, .8, (389, FEATURE_DIM)).astype(np.float32)
        history = HISTORY + RISE_PAST_FRAMES
        full = np.pad(features.T, ((0, 0), (history, 0)))[None]
        with torch.inference_mode():
            expected = model(torch.from_numpy(full)).numpy()[0, :, history:]
            changed = full.copy()
            changed[:, :, history + 200:] = 0
            early = model(torch.from_numpy(changed)).numpy()[0, :, history:history + 200]
            np.testing.assert_allclose(expected[:, :200], early, atol=1e-6)
            for start, length in ((0, 1), (1, 1), (127, 1), (128, 128), (256, 128), (384, 128)):
                x, count = feature_block(features, start, length, history)
                actual = model(torch.from_numpy(x[None])).numpy()[0, :, history:history + count]
                np.testing.assert_allclose(actual, expected[:, start:start + count], atol=1e-5)

    def test_common_training_frames_labels_masks_and_gain_are_identical(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npy"
            raw = np.random.default_rng(34).uniform(0, .8, (267, FEATURE_DIM)).astype(np.float16)
            np.save(path, raw)
            source = {"features": str(path), "frames": 267,
                      "events": [{"id": "note", "t": .032, "end": .3, "pc": 4}], "split": "train"}
            control = OnsetBlocks([source])
            rise = OnsetBlocks([source], history_frames=HISTORY + RISE_PAST_FRAMES)
            count = 0
            for i in range(len(control)):
                np.random.seed(i)
                a = control[i]
                np.random.seed(i)
                b = rise[i]
                np.testing.assert_array_equal(a[0], b[0][:, RISE_PAST_FRAMES:])
                for first, second in zip(a[1:], b[1:]):
                    np.testing.assert_array_equal(first, second)
                count += int(b[2].sum())
                # Independently verify gain is applied in amplitude before taking differences.
                unscaled, _ = feature_block(raw, i * 128, history_frames=HISTORY + RISE_PAST_FRAMES)
                np.random.seed(i)
                gain = 10 ** np.random.uniform(-.3, .3)
                amplitude = np.expm1(unscaled.astype(np.float64) * math.log(1001)) * gain
                expected = np.empty_like(amplitude)
                for t in range(amplitude.shape[1]):
                    delta = amplitude[:, t] - amplitude[:, max(0, t - 4):t].sum(axis=1) / 4
                    expected[:, t] = np.log1p(np.maximum(0, delta)) / math.log(1001)
                actual = positive_spectral_rise(torch.from_numpy(b[0][None])).numpy()[0]
                np.testing.assert_allclose(actual, expected, atol=2e-5)
            self.assertEqual(count, 267)


if __name__ == "__main__":
    unittest.main()
