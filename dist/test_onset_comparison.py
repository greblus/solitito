"""Regression checks for comparison timing, gates and duplicate attribution."""

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from compare_short_onset import (CachedOriginal, aggregate, compare_events, describe_extras,
                                 events_from_arrays, original_probabilities)
from onset_events import Event, score
from summarize_onset_comparison import checked_predictions


class RecordingSession:
    def __init__(self):
        self.inputs = []

    def run(self, names, feed):
        self.inputs.append(feed["features"].copy())
        return [np.zeros((len(feed["features"]), 12), dtype=np.float32)]


class ComparisonTests(unittest.TestCase):
    def test_saved_predictions_reject_truncated_shifted_or_nonfinite_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.npz"
            times = np.arange(1, 63) * .016
            probabilities = np.zeros((62, 12))
            np.savez(path, times=times, probabilities=probabilities)
            checked_predictions(path, 1., True)
            for t, p in ((times[:-1], probabilities[:-1]), (times + .016, probabilities),
                         (times, probabilities + np.nan)):
                np.savez(path, times=t, probabilities=p)
                with self.assertRaisesRegex(ValueError, "Incomplete/invalid"):
                    checked_predictions(path, 1., True)

    def test_cache_only_reuses_exact_inputs_and_deduplicates_within_batch(self):
        underlying = RecordingSession()
        cached = CachedOriginal(underlying)
        inputs = np.zeros((3, 48, 168), dtype=np.float32)
        inputs[2, 0, 0] = 1
        cached.run(["onset_logits"], {"features": inputs})
        self.assertEqual(len(underlying.inputs[0]), 2)
        cached.run(["onset_logits"], {"features": inputs})
        self.assertEqual(len(underlying.inputs), 1)
        inputs[0, 0, 0] = np.nextafter(np.float32(0), np.float32(1))
        cached.run(["onset_logits"], {"features": inputs})
        self.assertEqual(len(underlying.inputs[-1]), 1)

    def test_original_context_uses_physical_end_time_and_gates_before_windows(self):
        times = .512 + np.arange(51) * .016
        features = np.arange(51 * 168, dtype=np.float32).reshape(51, 168)
        rms = np.ones(51, dtype=np.float32)
        rms[3] = 0
        model = RecordingSession()
        t, fill, probabilities = original_probabilities(model, times, features, rms, batch=3)
        self.assertEqual(len(t), 4)
        self.assertAlmostEqual(t[0], 1.264)
        self.assertAlmostEqual(t[-1], 1.312)
        np.testing.assert_array_equal(probabilities, .5)
        inputs = np.concatenate(model.inputs)
        self.assertEqual(inputs.shape, (4, 48, 168))
        np.testing.assert_array_equal(inputs[0, 3], 0)
        np.testing.assert_array_equal(inputs[0, -1], features[47])
        np.testing.assert_array_equal(inputs[-1, -1], features[50])
        np.testing.assert_allclose(fill, 100 * 47 / 48)

    def test_late_real_attack_is_not_mistaken_for_a_proven_duplicate(self):
        refs = [Event("a", 1., 9)]
        predicted = [Event("p", 1.25, 9)]
        result = score(refs, predicted, 0, 3, early=.032, late=.128)
        self.assertEqual(describe_extras(refs, result)[0]["relation"], "unmatched_prior_pc")
        wide = score(refs, predicted, 0, 3, early=.05, late=.4)
        self.assertEqual((wide["tp"], wide["fp"]), (1, 0))

    def test_new_fifth_does_not_excuse_second_root_detection(self):
        refs = [Event("root", 1., 9), Event("fifth", 2., 4)]
        predicted = [Event("a", 1.04, 9), Event("e", 2.04, 4), Event("again", 2.04, 9)]
        result = score(refs, predicted, 0, 3, early=.032, late=.128)
        details = describe_extras(refs, result)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["relation"], "already_matched_pc")
        self.assertEqual(details[0]["nearby_attack_ids"], ["fifth"])
        self.assertAlmostEqual(details[0]["since_same_pc"], 1.04)

    def test_both_models_use_same_reference_denominator_including_boundaries(self):
        source = {"domain": "synthetic", "case": "root_repluck", "duration": 3,
                  "events": [{"id": "r", "t": .01, "pc": 9},
                             {"id": "c", "t": 2., "pc": 9, "role": "challenge"}]}
        scores = compare_events(source, {"candidate": [Event("c", 2.03, 9)], "original": []})
        self.assertEqual(scores["candidate/strict"]["reference"], 2)
        self.assertEqual(scores["original/strict"]["reference"], 2)
        self.assertEqual(scores["candidate/strict"]["challenge_tp"], 1)
        combined = aggregate([{**source, "scores": scores}])
        self.assertEqual(combined["synthetic:candidate/strict"]["fn"], 1)
        self.assertEqual(combined["synthetic:original/strict"]["fn"], 2)

    def test_frame_subsampling_preserves_actual_audio_times(self):
        times = 1.264 + np.arange(9) * .016
        probabilities = np.zeros((9, 12))
        probabilities[4:, 9] = .8
        events = events_from_arrays(times, probabilities, .6, stride=3, phase=1)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].t, times[4])
        # Numpy time scalars must not leak into the scorer's integer counters.
        result = score([Event("r", times[4].item(), 9)], events, 0, 2)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
