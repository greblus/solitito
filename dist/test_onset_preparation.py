"""Dataset contract checks: leakage, attack identity, paired audio and failures."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from audit_onset_data import audit
from onset_events import read_events
from prepare_onset_data import find_manifest, pluck, prepare, render_group, sha256, split_sources
from test_onset_events import jams_fixture


def fixture(root):
    for player in ("00", "04", "05"):
        for style in ("solo", "comp"):
            take = f"{player}_Jazz1-100-C_{style}"
            (root / f"{take}.jams").write_text(json.dumps(jams_fixture()))
            audio = np.zeros(8000, dtype=np.float32)
            audio[800:1600] = .01 * (1 + int(player) + (style == "solo") * 10)
            sf.write(root / f"{take}_mix.wav", audio, 8000, subtype="FLOAT")
    document = audit(root, "auto")
    if not document["ok"]:
        raise AssertionError(document["errors"])
    path = root / "onset_manifest_mix.json"
    path.write_text(json.dumps(document))
    return path, document


class PreparationTests(unittest.TestCase):
    def test_manifest_found_in_nested_output_and_summary_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "input" / "prior-notebook-output"
            nested.mkdir(parents=True)
            manifest, _ = fixture(nested)
            (root / "onset_manifest_mix_summary.json").write_text('{"ok": true}')
            self.assertEqual(find_manifest("auto", [root]), manifest)
            # A missing explicit path must not silently select a different file.
            with self.assertRaisesRegex(FileNotFoundError, "Requested manifest"):
                find_manifest(root / "missing.json", [root])

    def test_identical_copies_allowed_but_different_audits_require_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, document = fixture(root)
            duplicate = root / "onset_manifest_copy.json"
            duplicate.write_bytes(manifest.read_bytes())
            self.assertIn(find_manifest("auto", [root]), (manifest, duplicate))
            document["root"] = "/different/audit"
            duplicate.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "Multiple different"):
                find_manifest("auto", [root])
            self.assertEqual(find_manifest(manifest, [root]), manifest)

    def test_missing_full_manifest_reports_visible_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "onset_manifest_mix_summary.json"
            summary.write_text('{"ok": true}')
            with self.assertRaisesRegex(FileNotFoundError, "summary or missing sources"):
                find_manifest("auto", [root])

    def test_whole_players_pairs_and_annotations_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, document = fixture(Path(tmp))
            sources = split_sources(document, "04", "05")
            for split, player in (("train", "00"), ("validation", "04"), ("test", "05")):
                selected = [s for s in sources if s["split"] == split]
                self.assertEqual({s["player"] for s in selected}, {player})
                self.assertEqual({s["style"] for s in selected}, {"solo", "comp"})
                self.assertEqual(len({s["paired_take_group"] for s in selected}), 1)
                self.assertTrue(all(s["encoder_exposure"] == "unknown" for s in selected))
            for before, after in zip(document["sources"], sources):
                self.assertEqual(before["events"], after["events"])
                self.assertEqual(before["split"], "unassigned")
                self.assertTrue(all(not e["pluck_verified"] for e in after["events"]))

    def test_rejects_cross_split_content_and_reassignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, document = fixture(Path(tmp))
            duplicate = copy.deepcopy(document)
            duplicate["sources"][-1]["audio"]["sha256"] = duplicate["sources"][0]["audio"]["sha256"]
            with self.assertRaisesRegex(ValueError, "leakage"):
                split_sources(duplicate, "04", "05")
            document["sources"][0]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "assigned split"):
                split_sources(document, "04", "05")

    def test_rejects_incomplete_audit_and_invalid_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, document = fixture(Path(tmp))
            for mutate in (lambda d: d.update(ok=False), lambda d: d.pop("sources"),
                           lambda d: d["summary"].update(notes=0),
                           lambda d: d["sources"][0]["events"][0].update(pc=99),
                           lambda d: d["sources"][0]["events"][0].update(t=float("nan")),
                           lambda d: d["sources"][0]["events"][0].update(pluck_verified=True)):
                bad = copy.deepcopy(document)
                mutate(bad)
                with self.assertRaises(ValueError):
                    split_sources(bad, "04", "05")
            with self.assertRaises(ValueError):
                split_sources(document, "04", "04")

    def test_changed_audio_leaves_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, document = fixture(root)
            audio = Path(document["sources"][0]["audio"]["path"])
            with audio.open("ab") as stream:
                stream.write(b"changed")
            output = root / "prepared"
            with self.assertRaisesRegex(ValueError, "Changed since audit"):
                prepare(path, output, groups=(1, 1, 1), sr=8000)
            report = json.loads((output / "summary.json").read_text())
            self.assertFalse(report["ok"])
            self.assertIn("error", report)
            self.assertFalse((output / "dataset.json").exists())

    def test_complete_preparation_can_be_read_by_existing_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root)
            output = root / "prepared"
            summary = prepare(path, output, groups=(1, 1, 1), sr=8000)
            self.assertTrue(summary["ok"])
            self.assertFalse(summary["training_ready"])
            dataset = json.loads((output / "dataset.json").read_text())
            for kind in ("guitarset", "synthetic"):
                self.assertEqual(sha256(output / dataset[kind]["path"]), dataset[kind]["sha256"])
            synthetic = json.loads((output / "synthetic/manifest.json").read_text())
            self.assertEqual(len(synthetic["clips"]), 24)
            for clip in synthetic["clips"]:
                wav = output / "synthetic" / clip["wav"]
                ref = output / "synthetic" / clip["reference"]
                self.assertEqual(sha256(wav), clip["wav_sha256"])
                self.assertEqual(sha256(ref), clip["reference_sha256"])
                self.assertEqual(len(read_events(ref)), len(clip["events"]))
                self.assertEqual(sf.info(wav).frames, clip["frames"])
            with self.assertRaises(FileExistsError):
                prepare(path, output, groups=(1, 1, 1), sr=8000)


class SyntheticPairsTests(unittest.TestCase):
    def test_rendered_pitch_matches_label_across_guitar_range(self):
        # Measure audio independently; integer-delay KS was >50 cents sharp
        # at MIDI76/16kHz despite carrying a perfectly consistent MIDI label.
        for sr in (16000, 44100):
            frequencies = np.fft.rfftfreq(262144, 1 / sr)
            for midi in range(40, 89):
                wave = pluck(midi, round(.5 * sr), sr, 23, .998, .003)
                segment = wave[round(.05 * sr):round(.45 * sr)]
                spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n=262144))
                expected = 440 * 2 ** ((midi - 69) / 12)
                band = np.flatnonzero((frequencies > expected * .95) & (frequencies < expected * 1.05))
                measured = frequencies[band[np.argmax(spectrum[band])]]
                cents = 1200 * np.log2(measured / expected)
                self.assertLess(abs(cents), 3., (sr, midi, measured, cents))

    @classmethod
    def setUpClass(cls):
        cls.clips = {c["case"]: c for c in render_group("train", 0, sr=8000)}

    def test_hold_and_challenge_share_identical_background(self):
        for hold, changed in (("root_hold", "root_plus_third"), ("root_hold", "root_plus_fifth"),
                              ("root_hold", "root_repluck"), ("root_hold", "root_plus_octave"),
                              ("triad_hold", "triad_fifth"), ("triad_hold", "triad_repluck")):
            a, b = self.clips[hold], self.clips[changed]
            start = round(b["challenge_at"] * b["sr"])
            np.testing.assert_array_equal(a["audio"][:start], b["audio"][:start])
            self.assertEqual(a["gain"], b["gain"])
            self.assertGreater(float(np.max(np.abs(a["audio"][start:] - b["audio"][start:]))), .001)
            # Adding a fifth must give exactly the same audio difference on a
            # root and on a triad; no per-case normalization may alter the tail.
        fifth = self.clips["root_plus_fifth"]["audio"] - self.clips["root_hold"]["audio"]
        triad_fifth = self.clips["triad_fifth"]["audio"] - self.clips["triad_hold"]["audio"]
        np.testing.assert_allclose(fifth, triad_fifth, atol=1e-7, rtol=0)

    def test_sample_times_polyphonic_labels_and_no_clipping(self):
        for case, clip in self.clips.items():
            self.assertTrue(np.isfinite(clip["audio"]).all())
            self.assertLessEqual(float(np.max(np.abs(clip["audio"]))), .800001)
            for event in clip["events"]:
                self.assertEqual(event["sample"], round(event["t"] * clip["sr"]))
                self.assertLess(event["sample"], clip["frames"])
                self.assertEqual(event["pc"], event["midi"] % 12)
            expected = 0 if case.endswith("hold") else 3 if case == "triad_repluck" else 1
            self.assertEqual(sum(e["role"] == "challenge" for e in clip["events"]), expected)
        octave = self.clips["root_plus_octave"]["events"]
        self.assertEqual(octave[0]["pc"], octave[1]["pc"])
        self.assertNotEqual(octave[0]["midi"], octave[1]["midi"])

    def test_determinism_and_no_shared_excitations_across_splits(self):
        again = render_group("train", 0, sr=8000)
        for clip in again:
            np.testing.assert_array_equal(clip["audio"], self.clips[clip["case"]]["audio"])
        used = set()
        for split in ("train", "validation", "test"):
            clips = render_group(split, 0, sr=8000)
            seeds = {e["excitation_seed"] for c in clips for e in c["events"]}
            self.assertFalse(seeds & used)
            used.update(seeds)
            self.assertEqual(len({c["source_group"] for c in clips}), 1)
            self.assertTrue(all(c["parent_groups"] == [c["source_group"]] for c in clips))


if __name__ == "__main__":
    unittest.main()
