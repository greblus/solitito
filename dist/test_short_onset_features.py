import unittest

import numpy as np
from scipy.optimize import nnls

from short_onset_features import (
    harmonic_dictionary, positive_pitch_rise, rank_frame, short_features,
)


class ShortFeatureTests(unittest.TestCase):
    def test_future_audio_cannot_change_past_features_or_scores(self):
        rng = np.random.default_rng(23)
        audio = rng.normal(0, .05, 8192)
        changed = audio.copy()
        changed[4096:] += rng.normal(0, .1, 4096)
        for window in (1024, 2048):
            times, amplitudes, scores = short_features(audio, window)
            other_times, other_amplitudes, other_scores = short_features(changed, window)
            np.testing.assert_array_equal(times, other_times)
            past = times <= 4096 / 16000
            np.testing.assert_array_equal(amplitudes[past], other_amplitudes[past])
            np.testing.assert_array_equal(scores[past], other_scores[past])

    def test_new_class_repeat_and_octave_keep_their_identity(self):
        amplitudes = np.array([[10., 0., 0.], [10., 5., 0.], [12., 5., 0.], [12., 5., 3.]])
        scores = positive_pitch_rise(amplitudes, [9, 4, 9], lookback=1)
        self.assertEqual(np.flatnonzero(scores[1]).tolist(), [4])
        self.assertEqual(np.flatnonzero(scores[2]).tolist(), [9])
        self.assertEqual(np.flatnonzero(scores[3]).tolist(), [9])
        np.testing.assert_allclose(scores, positive_pitch_rise(amplitudes * .25, [9, 4, 9], 1))

    def test_fixed_dictionary_can_resolve_an_exact_polyphonic_mixture(self):
        for window in (1024, 2048):
            dictionary = harmonic_dictionary(window)
            expected = np.zeros(dictionary.shape[1])
            expected[[5, 9, 12]] = [1., .5, .7]
            actual, residual = nnls(dictionary, dictionary @ expected)
            np.testing.assert_allclose(actual, expected, atol=1e-8)
            self.assertLess(residual, 1e-8)

    def test_ranking_does_not_award_ties_silence_or_missing_chord_tones(self):
        self.assertFalse(rank_frame(np.zeros(12), [0])["correct_top_k"])
        self.assertEqual(rank_frame(np.zeros(12), [0])["top_k"], [])
        self.assertFalse(rank_frame(np.ones(12), [0])["correct_top_k"])
        row = np.zeros(12)
        row[[0, 4, 7]] = [.3, .2, .1]
        self.assertTrue(rank_frame(row, [0, 4, 7])["correct_top_k"])
        self.assertFalse(rank_frame(row, [0, 4, 11])["correct_top_k"])

    def test_decay_alone_and_invalid_features(self):
        scores = positive_pitch_rise(np.array([[10., 5.], [8., 4.], [5., 2.]]), [9, 4], 1)
        np.testing.assert_array_equal(scores[1:], 0)
        for invalid in (np.array([[float("nan")]]), np.array([[-1.]])):
            with self.assertRaises(ValueError):
                positive_pitch_rise(invalid, [0])


if __name__ == "__main__":
    unittest.main()
