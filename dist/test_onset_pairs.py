"""Training pairs must differ in the target attack, never in their background."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from onset_pairs import build_onset_pairs, OnsetPairBatches, onset_pair_loss, onset_pair_metrics
from prepare_onset_data import render_group
from train_short_onset import feature_block, onset_features, onset_targets, HISTORY


class OnsetPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.sources = []
        for index, clip in enumerate(render_group("train", 0, seed=20260923)):
            features = onset_features(clip.pop("audio"))
            path = Path(cls.directory.name) / f"{index}.npy"
            np.save(path, features)
            cls.sources.append(dict(clip, id=clip["name"], group=clip["source_group"],
                                    domain="synthetic", features=str(path), frames=len(features), pair_gain=clip["gain"]))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_pair_targets_protect_octaves_and_strummed_fifth(self):
        pairs, audit = build_onset_pairs(self.sources)
        self.assertEqual(audit["pairs"], 12)
        for p in pairs:
            a, b = (self.sources[p[k]] for k in ("positive", "negative"))
            interval = slice(p["start"], p["start"] + p["length"])
            self.assertTrue(np.all(onset_targets(a["events"], a["frames"])[interval, p["pc"]] == 1))
            self.assertTrue(np.all(onset_targets(b["events"], b["frames"])[interval, p["pc"]] == 0))
            self.assertEqual(a["group"], b["group"])
        kinds = audit["kinds"]
        self.assertEqual(kinds["triad_repluck/triad_fifth"], 2)
        self.assertEqual(kinds["triad_repluck/triad_hold"], 3)
        self.assertEqual(kinds["root_plus_octave/root_hold"], 1)
        self.assertNotIn("root_repluck/root_plus_octave", kinds)

    def test_no_cross_split_pairs_or_validation_training(self):
        sources = copy.deepcopy(self.sources)
        sources[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "crosses data splits"):
            build_onset_pairs(sources)
        with self.assertRaisesRegex(ValueError, "train sources only"):
            OnsetPairBatches(sources, feature_block)

    def test_staggered_strums_use_each_notes_own_time_and_shared_prefix(self):
        strums = set()
        # Fixed seed: group 15 supplies the 25ms strum absent from groups 0..11.
        for group in range(16):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as tmp:
                sources = []
                for index, clip in enumerate(render_group("train", group, seed=20260923)):
                    strums.add(clip["strum_seconds"])
                    features = onset_features(clip.pop("audio"))
                    path = Path(tmp) / f"{index}.npy"
                    np.save(path, features)
                    sources.append(dict(clip, id=clip["name"], group=clip["source_group"],
                                        domain="synthetic", features=str(path), frames=len(features),
                                        pair_gain=clip["gain"]))
                data = OnsetPairBatches(sources, feature_block)
                self.assertEqual(len(data.pairs), 12)
                for pair in data.pairs:
                    window = slice(pair["start"], pair["start"] + pair["length"])
                    for key, value in (("positive", 1), ("negative", 0)):
                        source = sources[pair[key]]
                        targets = onset_targets(source["events"], source["frames"])
                        np.testing.assert_array_equal(targets[window, pair["pc"]], value)
        self.assertEqual(strums, {0., .012, .025})

    def test_different_gain_and_unverified_attack_are_rejected(self):
        sources = copy.deepcopy(self.sources)
        sources[0]["pair_gain"] *= .5
        with self.assertRaisesRegex(ValueError, "different gain"):
            build_onset_pairs(sources)
        sources = copy.deepcopy(self.sources)
        sources[0]["events"][0]["pluck_verified"] = False
        with self.assertRaisesRegex(ValueError, "verified synthetic"):
            build_onset_pairs(sources)

    def test_corrupted_audio_prefix_rejected_before_training(self):
        sources = copy.deepcopy(self.sources)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "different.npy"
            features = np.load(sources[0]["features"])
            features[0, 0] = 1
            np.save(path, features)
            sources[0]["features"] = str(path)
            with self.assertRaisesRegex(ValueError, "differ before the challenge"):
                OnsetPairBatches(sources, feature_block)

    def test_each_pair_once_shared_gain_and_no_base_rng_interference(self):
        data = OnsetPairBatches(self.sources, feature_block)
        np.random.seed(42)
        expected = np.random.random(10)
        np.random.seed(42)
        batches = list(data.epoch(1, 7, 100))
        np.testing.assert_array_equal(np.random.random(10), expected)
        ids = [pair_id for batch in batches if batch is not None for pair_id in batch[3]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 12)
        for first, second in zip(batches, data.epoch(1, 7, 100)):
            if first is not None:
                for a, b in zip(first[:3], second[:3]):
                    np.testing.assert_array_equal(a, b)
                # All train group0 positive root windows see identical old context.
                for p, n, pair_id in zip(first[0], first[1], first[3]):
                    if "root_repluck:" in pair_id:
                        np.testing.assert_array_equal(p[:, :HISTORY], n[:, :HISTORY])
        sparse = list(data.epoch(1, 20, 100))
        self.assertEqual(sum(b is not None for b in sparse), 12)

    def test_pair_metrics_do_not_reward_equal_outputs(self):
        predictions = [(s, onset_targets(s["events"], s["frames"])) for s in self.sources]
        measured = onset_pair_metrics(predictions)
        self.assertEqual(measured["positive_above_negative"], 12)
        constant = [(s, np.zeros_like(p)) for s, p in predictions]
        measured = onset_pair_metrics(constant)
        self.assertEqual(measured["positive_above_negative"], 0)
        self.assertEqual(measured["ties"], 12)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "temporary Torch environment required")
    def test_ranking_gradient_raises_true_attack_and_lowers_wrong_pc_response(self):
        import torch
        positive = torch.full((2, 12, HISTORY + 6), -2., requires_grad=True)
        negative = torch.full((2, 12, HISTORY + 6), 2., requires_grad=True)
        pcs = torch.tensor([2, 9])
        loss = onset_pair_loss(positive, negative, pcs, HISTORY)
        loss.backward()
        for i, pc in enumerate(pcs):
            self.assertLess(float(positive.grad[i, pc, HISTORY:].sum()), 0)
            self.assertGreater(float(negative.grad[i, pc, HISTORY:].sum()), 0)
        self.assertEqual(int(torch.count_nonzero(positive.grad[:, :, :HISTORY])), 0)
        self.assertEqual(int(torch.count_nonzero(positive.grad[:, 0])), 0)
        self.assertLess(float(onset_pair_loss(negative, positive, pcs, HISTORY).detach()), float(loss.detach()))


if __name__ == "__main__":
    unittest.main()
