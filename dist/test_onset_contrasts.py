from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

from onset_contrasts import render_group, write_dataset
from onset_events import read_events, sha256
from measure_onset_contrasts import measure


class ContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clips = {clip["case"]: clip for clip in render_group(0, sr=16000)}

    def test_only_added_plucks_change_the_shared_background(self):
        for base, variants in (("root_hold", ["root_plus_fifth", "root_repluck"]),
                               ("triad_hold", ["triad_fifth", "triad_repluck"])):
            hold = self.clips[base]
            sample = round(hold["challenge_at"] * hold["sr"])
            for variant in variants:
                changed = self.clips[variant]
                np.testing.assert_array_equal(hold["audio"][:sample], changed["audio"][:sample])
                self.assertGreater(np.max(np.abs(hold["audio"][sample:] - changed["audio"][sample:])), .01)
                self.assertEqual(hold["gain"], changed["gain"])

    def test_selective_fifth_is_the_same_added_sound_in_both_contexts(self):
        single = self.clips["root_plus_fifth"]["audio"] - self.clips["root_hold"]["audio"]
        triad = self.clips["triad_fifth"]["audio"] - self.clips["triad_hold"]["audio"]
        np.testing.assert_allclose(single, triad, atol=1e-7)

    def test_labels_count_only_the_new_attacks(self):
        expected = {"root_hold": [], "root_plus_fifth": [4], "root_repluck": [9],
                    "triad_hold": [], "triad_fifth": [4], "triad_repluck": [1, 4, 9]}
        for case, pcs in expected.items():
            clip = self.clips[case]
            self.assertEqual(clip["expected_new_pcs"], pcs)
            self.assertEqual(sorted(e["pc"] for e in clip["events"] if e["role"] == "challenge"), pcs)
            for event in clip["events"]:
                self.assertEqual(event["t"], event["sample"] / clip["sr"])

    def test_group_variants_share_the_source_and_development_split(self):
        self.assertEqual(len({c["source_group"] for c in self.clips.values()}), 1)
        self.assertEqual({c["split"] for c in self.clips.values()}, {"development"})

    def test_repeatability_and_no_clipping(self):
        for clip in render_group(0, sr=16000):
            np.testing.assert_array_equal(clip["audio"], self.clips[clip["case"]]["audio"])
            self.assertTrue(np.isfinite(clip["audio"]).all())
            self.assertLessEqual(np.max(np.abs(clip["audio"])), .800001)

    def test_written_events_and_hashes_describe_the_actual_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "contrasts"
            manifest = write_dataset(output, 1, 42, 16000)
            self.assertEqual(len(manifest["clips"]), 6)
            for clip in manifest["clips"]:
                wav, csv = output / clip["wav"], output / clip["reference"]
                self.assertEqual(sha256(wav), clip["wav_sha256"])
                self.assertEqual(sha256(csv), clip["reference_sha256"])
                self.assertEqual(len(read_events(csv)), len(clip["events"]))
                info = sf.info(wav)
                self.assertEqual((info.channels, info.frames), (1, clip["frames"]))
            with self.assertRaises(FileExistsError):
                write_dataset(output, 1, 42, 16000)

    def test_changed_audio_is_rejected_before_running_a_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_dataset(root / "data", 1, 42, 16000)
            (root / "data" / manifest["clips"][0]["wav"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Dataset file changed"):
                measure(root / "data", root / "unused-model", root / "unused-binary", root / "result")
            self.assertFalse((root / "result").exists())

    def test_failed_probe_leaves_an_explicitly_incomplete_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root / "data", 1, 42, 16000)
            model = root / "dummy.onnx"
            model.write_bytes(b"not executed")
            # Python rejects --probe; this exercises a real subprocess failure
            # without requiring ONNX Runtime or an actual model in the tests.
            with self.assertRaisesRegex(RuntimeError, "capture_onset_probe.py failed"):
                measure(root / "data", model, Path(sys.executable), root / "result")
            report = json.loads((root / "result/summary.json").read_text())
            self.assertFalse(report["ok"])
            self.assertEqual(report["completed_clips"], 0)
            self.assertIn("error", report)


if __name__ == "__main__":
    unittest.main()
