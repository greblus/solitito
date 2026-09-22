import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest
import wave

from onset_events import Event, latch_events, match_events, read_events, read_probe, score
from audit_onset_data import audit, note_events, source_identity


def events(items):
    return [Event(str(i), t, pc) for i, (t, pc) in enumerate(items)]


class ScoringTests(unittest.TestCase):
    def test_polyphonic_strum_and_repeat(self):
        refs = events([(t + j * .015, pc) for t in (1, 2) for j, pc in enumerate((0, 4, 7, 11))])
        preds = [Event(e.id, e.t + .15, e.pc) for e in refs]
        result = score(refs, preds, 0, 3)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (8, 0, 0))
        self.assertAlmostEqual(result["matched_timing_seconds"]["p95"], .15)

    def test_new_fifth_does_not_justify_repeated_root(self):
        refs = events([(1, 0), (2, 7)])
        preds = events([(1.1, 0), (2.1, 0), (2.1, 7)])
        result = score(refs, preds, 0, 3)
        self.assertEqual((result["tp"], result["fp"]), (2, 1))
        self.assertEqual(result["extra"][0]["pc"], 0)

    def test_one_prediction_cannot_satisfy_two_attacks(self):
        result = score(events([(1, 0), (1.1, 0)]), events([(1.15, 0)]), 0, 2)
        self.assertEqual((result["tp"], result["fn"]), (1, 1))
        self.assertEqual(result["ambiguous_prediction_ids"], ["0"])

    def test_duplicate_and_wrong_pitch_count_once_each(self):
        result = score(events([(1, 0)]), events([(1.1, 0), (1.2, 0), (1.1, 7)]), 0, 2)
        self.assertEqual((result["tp"], result["fp"]), (1, 2))

    def test_maximum_cardinality_before_nearest_time(self):
        pairs = match_events(events([(1, 0), (1.3, 0)]), events([(1.25, 0), (1.6, 0)]), .05, .4)
        self.assertEqual(pairs, [(0, 0), (1, 1)])

    def test_minimum_timing_error_breaks_ties(self):
        self.assertEqual(match_events(events([(1, 0)]), events([(1.3, 0), (1.1, 0)]), .05, .4), [(0, 1)])

    def test_matching_against_exhaustive_small_cases(self):
        rng = random.Random(42)
        for _ in range(50):
            refs = events([(rng.randrange(20) / 10, 0) for _ in range(3)])
            preds = events([(rng.randrange(20) / 10, 0) for _ in range(3)])
            best = (0, 0)
            for count in range(1, 4):
                for ri in itertools.combinations(range(3), count):
                    for pj in itertools.permutations(range(3), count):
                        deltas = [preds[j].t - refs[i].t for i, j in zip(ri, pj)]
                        if all(-.05 - 1e-9 <= d <= .4 + 1e-9 for d in deltas):
                            best = max(best, (count, -sum(abs(d) for d in deltas)))
            pairs = match_events(refs, preds, .05, .4)
            self.assertEqual(len(pairs), best[0])
            self.assertAlmostEqual(-sum(abs(preds[j].t - refs[i].t) for i, j in pairs), best[1])

    def test_outside_time_tolerance_is_miss_and_extra(self):
        for t in (.94, 1.41):
            result = score(events([(1, 0)]), events([(t, 0)]), 0, 2)
            self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))

    def test_crop_both_sides_and_report_boundary(self):
        result = score(events([(1, 0), (1.9, 4), (3, 7)]), events([(1.1, 0), (3.1, 7)]), 0, 2)
        self.assertEqual((result["reference"], result["predicted"]), (2, 1))
        self.assertEqual(result["boundary_reference_ids"], ["1"])
        self.assertEqual((result["excluded_reference"], result["excluded_predicted"]), (1, 1))

    def test_same_class_different_octave_is_explicit(self):
        refs, preds = [Event("r", 1, 0, 48)], [Event("p", 1.1, 0, 60)]
        self.assertEqual(score(refs, preds, 0, 2)["tp"], 1)
        self.assertEqual(score(refs, preds, 0, 2, pitch="midi")["tp"], 0)
        with self.assertRaises(ValueError):
            score(refs, events([(1.1, 0)]), 0, 2, pitch="midi")

    def test_all_silence_has_no_fake_perfect_f1(self):
        self.assertIsNone(score([], [], 0, 2)["f1"])

    def test_latch_needs_rearm_for_same_pitch_but_not_new_pitch(self):
        rows = [(i * .016, 100, [a, b] + [0] * 10)
                for i, (a, b) in enumerate(((.7, 0), (.8, 0), (.8, .9), (.1, .9), (.7, .9)))]
        self.assertEqual([(e.t, e.pc) for e in latch_events(rows)], [(0, 0), (.032, 1), (.064, 0)])

    def test_latch_fill_and_sampling_are_explicit(self):
        rows = [(0, 40, [.8] * 12), (.016, 100, [.8] * 12), (.032, 100, [.8] * 12)]
        self.assertEqual(len(latch_events(rows, stride=2, phase=0)), 12)
        self.assertEqual(latch_events(rows, stride=2, phase=0)[0].t, .032)

    def test_probe_integrity_checks(self):
        def row(t):
            return f"{t:.2f} -20.0 100% " + " 0" * 12 + " |" + " 0" * 12 + " - model\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.txt"
            path.write_text(row(1.26) + row(1.28) + row(1.30))
            self.assertEqual(len(read_probe(path, 1.31)), 3)
            for text in (row(1.5), row(.5), row(1.26) + row(1.26), row(1.30), "❌ failed\n", "no rows"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_probe(path, 1.31)

    def test_csv_rejects_duplicate_ids_and_wrong_pc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            for text in ("t,pc,midi\n1,1,60\n", "id,t,pc\nx,1,0\nx,2,0\n", "t,pc\nnan,0\n"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_events(path)


def jams_fixture():
    # Shape from the GuitarSet loader contract, not a real annotated recording.
    return {"annotations": [{"namespace": "note_midi", "annotation_metadata": {"data_source": str(s)},
                              "data": [{"time": .1, "duration": .2, "value": m}]}
                             for s, m in enumerate((40, 45, 50, 55, 59, 64))]}


class AuditTests(unittest.TestCase):
    def test_string_metadata_and_polyphony_preserved(self):
        notes, _ = note_events(jams_fixture(), "take")
        self.assertEqual(len(notes), 6)
        self.assertEqual({n["string"] for n in notes}, set(range(6)))
        self.assertEqual(sum(n["pc"] == 4 for n in notes), 2)
        self.assertTrue(all(not n["pluck_verified"] for n in notes))

    def test_column_layout(self):
        document = jams_fixture()
        for annotation in document["annotations"]:
            obs = annotation["data"][0]
            annotation["data"] = {k: [v] for k, v in obs.items()}
        self.assertEqual(len(note_events(document, "take")[0]), 6)

    def test_fractional_midi_is_preserved_without_inventing_pluck_certainty(self):
        document = jams_fixture()
        document["annotations"][0]["data"][0]["value"] = 40.45
        note = note_events(document, "take")[0][0]
        self.assertEqual((note["midi_value"], note["midi"], note["pc"]), (40.45, 40, 4))
        self.assertTrue(note["near_semitone_boundary"])

    def test_no_guessing_unknown_or_missing_strings(self):
        for change in (lambda d: d["annotations"][0]["annotation_metadata"].clear(),
                       lambda d: d["annotations"].pop()):
            document = jams_fixture()
            change(document)
            with self.assertRaises(ValueError):
                note_events(document, "take")

    def test_grouping_keeps_solo_comp_distinct_but_same_split_group(self):
        solo, comp = [source_identity(f"00_Jazz1-200-C_{kind}") for kind in ("solo", "comp")]
        self.assertNotEqual(solo["take"], comp["take"])
        self.assertEqual(solo["split_group"], comp["split_group"])

    def test_manifest_fails_for_duplicate_audio_and_hex_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            take = "00_Jazz1-200-C_solo"
            (root / f"{take}.jams").write_text(json.dumps(jams_fixture()))
            def wav(path):
                with wave.open(str(path), "wb") as stream:
                    stream.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
                    stream.writeframes(b"\0" * 32000)
            wav(root / f"{take}_hex.wav")
            self.assertFalse(audit(root, "mic")["ok"])
            wav(root / f"{take}_mic.wav")
            result = audit(root, "mic")
            self.assertTrue(result["ok"])
            self.assertEqual(result["summary"]["notes"], 6)
            self.assertEqual(result["sources"][0]["split"], "unassigned")
            (root / "duplicate").mkdir()
            wav(root / "duplicate" / f"{take}_mic.wav")
            self.assertFalse(audit(root, "mic")["ok"])

    def test_auto_selects_only_the_available_mono_variant(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            take = "01_Funk3-112-Cs_comp"
            (root / f"{take}.jams").write_text(json.dumps(jams_fixture()))
            sf.write(root / f"{take}_mix.flac", [0.0] * 16000, 16000)
            result = audit(root, "auto")
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(result["selected_variant"], "mix")
            self.assertEqual(result["inventory"]["matching_takes"], {"mic": 0, "mix": 1})
            # An explicit choice must never silently become another variant.
            explicit = audit(root, "mic")
            self.assertFalse(explicit["ok"])
            self.assertEqual(len(explicit["errors"]), 1)
            sf.write(root / f"{take}_mic.wav", [0.0] * 16000, 16000)
            ambiguous = audit(root, "auto")
            self.assertFalse(ambiguous["ok"])
            self.assertIsNone(ambiguous["selected_variant"])
            self.assertIn("Both mic and mix", ambiguous["errors"][0]["error"])

    def test_missing_variant_is_reported_once_with_actual_file_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(200):
                (root / f"01_Funk{i}-112-Cs_comp.jams").write_text("{}")
            example = root / "01_Funk0-112-Cs_comp.wav"
            example.touch()  # Untagged audio must not be guessed to be mic.
            result = audit(root, "mic")
            self.assertEqual(len(result["errors"]), 1)
            self.assertEqual(result["errors"][0]["count"], 200)
            self.assertEqual(len(result["missing_audio_takes"]), 200)
            self.assertEqual(result["inventory"]["variants"]["untagged"]["examples"], [str(example)])
            automatic = audit(root, "auto")
            self.assertFalse(automatic["ok"])
            self.assertEqual(len(automatic["errors"]), 1)


if __name__ == "__main__":
    unittest.main()
