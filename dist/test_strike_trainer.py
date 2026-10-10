"""Tests of strike_trainer.py: data, features, scoring, training, the app's file.

Run from dist/:  python -m unittest test_strike_trainer
Tests that train or export need torch, onnx and onnxruntime and are skipped
without them. The feature fixture is shared with src/strike.rs:

    python test_strike_trainer.py --write-fixture   # only after a deliberate change
"""
import ast
import copy
import csv
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np
import soundfile as sf

from strike_trainer import (
    FEATURE_DIM, FEATURE_SPEC, HISTORY_FRAMES, INITIAL_ONSET, MODE, ONSET_GAIN_DB,
    ONSET_MASKING_PAIRS, ONSET_SR, RISE_PAST_FRAMES, RUN_TAG, Event, OnsetBlocks, SnapshotStore, audit,
    cache_onset_features, export_event_comparison, export_probability_check, feature_block, latch_events,
    local_event_matches, make_onset_model, match_events, note_events, onset_features, onset_metrics,
    onset_resample, onset_sources, onset_targets, pluck, positive_spectral_rise, prepare,
    prepare_features, render_group, render_masking_group, resolve_initial_onset, run, run_pipeline,
    sha256, source_identity, split_sources, strike_model_name, train_onset, validate_resume_sources,
    validation_choice, write_strike_model)

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "strike_trainer.py"
HAS_TRAINING = all(importlib.util.find_spec(n) for n in ("torch", "onnx", "onnxruntime"))
HAS_ONNX = all(importlib.util.find_spec(n) for n in ("onnx", "onnxruntime"))
FIXTURES = HERE / "fixtures"
FIXTURE_AUDIO = FIXTURES / "short_features.s16"  # 16 kHz mono, little-endian int16
FIXTURE_FRAMES = FIXTURES / "short_features.f16"  # frames x 770, little-endian binary16


def jams_fixture():
    # Shape from the GuitarSet loader contract, not a real annotated recording.
    return {"annotations": [{"namespace": "note_midi", "annotation_metadata": {"data_source": str(s)},
                              "data": [{"time": .1, "duration": .2, "value": m}]}
                             for s, m in enumerate((40, 45, 50, 55, 59, 64))]}


def fixture(root):
    """A tiny stand-in for GuitarSet: three players, solo and comp, 1 s each."""
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


def events(items):
    return [Event(str(i), t, pc) for i, (t, pc) in enumerate(items)]


def read_fixture_audio():
    return np.frombuffer(FIXTURE_AUDIO.read_bytes(), dtype="<i2").astype(np.float32) / 32768


def write_feature_fixture():
    """Silence, a low pluck, a quieter high one over it, faint noise."""
    t = np.arange(ONSET_SR // 2) / ONSET_SR
    audio = 1e-3 * np.random.default_rng(20261010).standard_normal(len(t))
    for start, pitch, level in ((.05, 110., .25), (.3, 329.63, .08)):
        after = np.clip(t - start, 0, None)
        note = sum(np.sin(2 * np.pi * pitch * k * after) / k for k in range(1, 8))
        audio += level * (t >= start) * np.exp(-3 * after) * note
    FIXTURES.mkdir(exist_ok=True)
    FIXTURE_AUDIO.write_bytes(np.round(np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    FIXTURE_FRAMES.write_bytes(onset_features(read_fixture_audio()).astype("<f2").tobytes())


class SettingsTests(unittest.TestCase):
    def test_defaults_are_the_released_recipe(self):
        self.assertEqual(RUN_TAG, "v2_take7_masking_v2_repro")
        self.assertEqual(MODE, "train")
        self.assertTrue(ONSET_MASKING_PAIRS)
        self.assertEqual(INITIAL_ONSET, "hf:checkpoint_v2_take7_onset_best.pth")
        self.assertEqual(ONSET_GAIN_DB, 6.)
        self.assertEqual(strike_model_name(RUN_TAG), "short_onset_masking_v2_repro.onnx")
        self.assertEqual(strike_model_name("v2_take7_masking_v2"), "short_onset_masking_v2.onnx")
        self.assertEqual(strike_model_name("v2_take7"), "short_onset_v2_take7.onnx")

    def test_one_file_without_repository_imports(self):
        tree = ast.parse(TRAINER.read_text())
        local = {p.stem for p in HERE.glob("*.py")}
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertFalse(imported & local)
        self.assertNotIn("exec(", TRAINER.read_text())


class ScoringTests(unittest.TestCase):
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

    def test_component_matching_equals_full_scorer_including_close_repeats(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            refs = [Event(f"r{i}", float(t), i % 3) for i, t in enumerate(rng.integers(0, 500, 45) / 100)]
            preds = [Event(f"p{i}", float(t), i % 3) for i, t in enumerate(rng.integers(0, 500, 60) / 100)]
            expected = {(refs[i].id, preds[j].id) for i, j in match_events(refs, preds, .032, .128)}
            actual = {(r.id, p.id) for r, p in local_event_matches(refs, preds)}
            self.assertEqual(expected, actual)

    def test_latch_needs_rearm_for_same_pitch_but_not_new_pitch(self):
        rows = [(i * .016, 100, [a, b] + [0] * 10)
                for i, (a, b) in enumerate(((.7, 0), (.8, 0), (.8, .9), (.1, .9), (.7, .9)))]
        self.assertEqual([(e.t, e.pc) for e in latch_events(rows)], [(0, 0), (.032, 1), (.064, 0)])

    def test_held_note_is_one_event_but_rearmed_tail_is_an_error(self):
        source = {"id": "hold", "domain": "synthetic", "case": "root_hold", "duration": 2.,
                  "events": [{"id": "initial", "t": .16, "pc": 9}]}
        values = np.zeros((125, 12), dtype=np.float32)
        values[12:60, 9] = .9
        metrics, _ = onset_metrics([(source, values)], .5)
        self.assertEqual((metrics["groups"]["all"]["tp"], metrics["groups"]["all"]["fp"]), (1, 0))
        values[100:110, 9] = .9
        metrics, _ = onset_metrics([(source, values)], .5)
        self.assertEqual(metrics["groups"]["all"]["fp"], 1)
        self.assertEqual(metrics["groups"]["all"]["tail_extra"], 1)
        source["events"].append({"id": "repluck", "t": 1.584, "pc": 9, "role": "challenge"})
        metrics, _ = onset_metrics([(source, values)], .5)
        counts = metrics["groups"]["all"]
        self.assertEqual((counts["tp"], counts["fp"], counts["repeated_tp"], counts["challenge_tp"]), (2, 0, 1, 1))
        self.assertEqual(counts["challenge_recall"], 1.0)
        values[100:110, 9] = 0
        metrics, _ = onset_metrics([(source, values)], .5)
        self.assertEqual(metrics["groups"]["all"]["challenge_recall"], 0.0)

    def test_same_pc_strings_and_unobservable_end_note_are_not_forgiven(self):
        source = {"id": "poly", "domain": "guitarset", "case": "comp", "duration": 1.,
                  "events": [{"id": "s0", "t": .16, "pc": 9},
                             {"id": "s3", "t": .16, "pc": 9},
                             {"id": "end", "t": .999, "pc": 4}]}
        values = np.zeros((62, 12), dtype=np.float32)
        values[12, 9] = .8
        metrics, _ = onset_metrics([(source, values)], .5)
        counts = metrics["groups"]["all"]
        self.assertEqual((counts["tp"], counts["fn"], counts["same_pc_frame_collisions"]), (1, 2, 1))
        self.assertEqual(counts["boundary_reference"], 1)
        self.assertEqual(counts["same_pc_target_overlaps"], 1)

    def test_threshold_ties_prefer_fewer_false_events_then_higher_threshold(self):
        table = [{"threshold": t, "macro_domain_f1": f1, "groups": {"all": {"fp": fp}}}
                 for t, f1, fp in ((.3, .7, 20), (.5, .7, 10), (.7, .7, 10), (.9, .6, 0))]
        self.assertEqual(validation_choice(table)["threshold"], .7)


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


class PreparationTests(unittest.TestCase):
    def test_masking_pairs_have_identical_background_and_measured_relative_levels(self):
        for group in (0, 1, 2, 3, 92, 95):
            hold, alone, added = render_masking_group("validation", group)
            self.assertEqual(hold["expected_new_pcs"], [])
            self.assertEqual(len(hold["events"]), 2)
            self.assertEqual(len(added["events"]), 3)
            self.assertEqual(added["events"][-1]["pc"], added["target_midi"] % 12)
            start = round(added["challenge_at"] * added["sr"])
            end = start + round(.096 * added["sr"])
            np.testing.assert_array_equal(hold["audio"][:start], added["audio"][:start])
            np.testing.assert_allclose(added["audio"], hold["audio"] + alone["audio"], atol=1e-7)
            bg = np.sqrt(np.mean(hold["audio"][start:end].astype(float) ** 2))
            target = np.sqrt(np.mean(alone["audio"][start:end].astype(float) ** 2))
            self.assertAlmostEqual(20 * np.log10(target / bg), added["target_background_db"], places=5)
            self.assertLess(np.max(np.abs(added["audio"])), 1)
            self.assertGreaterEqual(added["target_midi"], 62)
            self.assertLessEqual(added["target_midi"], 85)

    def test_masking_excitation_groups_do_not_cross_splits_and_are_deterministic(self):
        first = render_masking_group("train", 0)
        repeated = render_masking_group("train", 0)
        held_out = render_masking_group("test", 0)
        for a, b, c in zip(first, repeated, held_out):
            np.testing.assert_array_equal(a["audio"], b["audio"])
            self.assertNotEqual(a["source_group"], c["source_group"])
            self.assertNotEqual(a["events"][0]["excitation_seed"], c["events"][0]["excitation_seed"])
            self.assertFalse(np.array_equal(a["audio"], c["audio"]))

    def test_whole_players_pairs_and_annotations_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, document = fixture(Path(tmp))
            sources = split_sources(document, "04", "05")
            for split, player in (("train", "00"), ("validation", "04"), ("test", "05")):
                selected = [s for s in sources if s["split"] == split]
                self.assertEqual({s["player"] for s in selected}, {player})
                self.assertEqual({s["style"] for s in selected}, {"solo", "comp"})
                self.assertEqual(len({s["paired_take_group"] for s in selected}), 1)
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

    def test_complete_preparation_has_matching_files_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root)
            output = root / "prepared"
            summary = prepare(path, output, groups=(1, 1, 1), sr=8000)
            self.assertTrue(summary["ok"])
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
                with ref.open(newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), len(clip["events"]))
                self.assertEqual(sf.info(wav).frames, clip["frames"])
            with self.assertRaises(FileExistsError):
                prepare(path, output, groups=(1, 1, 1), sr=8000)

    def test_rendered_pitch_matches_label_across_guitar_range(self):
        # Measure audio independently; integer-delay KS was >50 cents sharp
        # at MIDI76/16kHz despite carrying a perfectly consistent MIDI label.
        for sr in (16000, 44100):
            frequencies = np.fft.rfftfreq(262144, 1 / sr)
            for midi in range(40, 89):
                wave_ = pluck(midi, round(.5 * sr), sr, 23, .998, .003)
                segment = wave_[round(.05 * sr):round(.45 * sr)]
                spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n=262144))
                expected = 440 * 2 ** ((midi - 69) / 12)
                band = np.flatnonzero((frequencies > expected * .95) & (frequencies < expected * 1.05))
                measured = frequencies[band[np.argmax(spectrum[band])]]
                cents = 1200 * np.log2(measured / expected)
                self.assertLess(abs(cents), 3., (sr, midi, measured, cents))

    def test_hold_and_challenge_share_identical_background(self):
        clips = {c["case"]: c for c in render_group("train", 0, sr=8000)}
        for hold, changed in (("root_hold", "root_plus_third"), ("root_hold", "root_plus_fifth"),
                              ("root_hold", "root_repluck"), ("root_hold", "root_plus_octave"),
                              ("triad_hold", "triad_fifth"), ("triad_hold", "triad_repluck")):
            a, b = clips[hold], clips[changed]
            start = round(b["challenge_at"] * b["sr"])
            np.testing.assert_array_equal(a["audio"][:start], b["audio"][:start])
            self.assertEqual(a["gain"], b["gain"])
            self.assertGreater(float(np.max(np.abs(a["audio"][start:] - b["audio"][start:]))), .001)
        # Adding a fifth gives exactly the same audio difference on a root and on a triad.
        fifth = clips["root_plus_fifth"]["audio"] - clips["root_hold"]["audio"]
        triad_fifth = clips["triad_fifth"]["audio"] - clips["triad_hold"]["audio"]
        np.testing.assert_allclose(fifth, triad_fifth, atol=1e-7, rtol=0)

    def test_sample_times_polyphonic_labels_and_no_clipping(self):
        clips = {c["case"]: c for c in render_group("train", 0, sr=8000)}
        for case, clip in clips.items():
            self.assertTrue(np.isfinite(clip["audio"]).all())
            self.assertLessEqual(float(np.max(np.abs(clip["audio"]))), .800001)
            for event in clip["events"]:
                self.assertEqual(event["sample"], round(event["t"] * clip["sr"]))
                self.assertLess(event["sample"], clip["frames"])
                self.assertEqual(event["pc"], event["midi"] % 12)
            expected = 0 if case.endswith("hold") else 3 if case == "triad_repluck" else 1
            self.assertEqual(sum(e["role"] == "challenge" for e in clip["events"]), expected)

    def test_determinism_and_no_shared_excitations_across_splits(self):
        first = {c["case"]: c for c in render_group("train", 0, sr=8000)}
        for clip in render_group("train", 0, sr=8000):
            np.testing.assert_array_equal(clip["audio"], first[clip["case"]]["audio"])
        used = set()
        for split in ("train", "validation", "test"):
            clips = render_group(split, 0, sr=8000)
            seeds = {e["excitation_seed"] for c in clips for e in c["events"]}
            self.assertFalse(seeds & used)
            used.update(seeds)
            self.assertEqual(len({c["source_group"] for c in clips}), 1)


class FeatureTests(unittest.TestCase):
    def test_trainer_features_are_the_shared_fixture_to_the_last_bit(self):
        # src/strike.rs holds the app to the same two files.
        expected = np.frombuffer(FIXTURE_FRAMES.read_bytes(), dtype="<f2").reshape(-1, FEATURE_DIM)
        actual = onset_features(read_fixture_audio())
        self.assertEqual(actual.shape, expected.shape)
        differing = int(np.sum(actual.view(np.uint16) != expected.view(np.uint16)))
        self.assertEqual(differing, 0, f"{differing} of {expected.size} values differ")
        # The first frame sees mostly zero padding; the plucks reach well above the noise.
        self.assertLess(float(expected[0].max()), .05)
        self.assertGreater(float(expected.max()), .5)

    def test_audio_future_cannot_change_features_at_any_input_rate(self):
        rng = np.random.default_rng(31)
        for sr in (8000, 16000, 44100, 48000):
            cut = round(.512 * sr)
            a = rng.normal(0, .1, sr).astype(np.float32)
            b = a.copy()
            b[cut:] = rng.normal(0, .8, len(b) - cut)
            first = onset_features(onset_resample(a, sr))
            changed = onset_features(onset_resample(b, sr))
            np.testing.assert_array_equal(first[:32], changed[:32])
            prefix = onset_features(onset_resample(a[:cut], sr))
            np.testing.assert_array_equal(first[:len(prefix)], prefix)

    def test_features_start_immediately_and_do_not_pad_the_tail(self):
        silence = onset_features(np.zeros(16000, dtype=np.float32))
        self.assertEqual(silence.shape, (62, FEATURE_DIM))
        self.assertEqual(np.count_nonzero(silence), 0)
        audio = np.zeros(257, dtype=np.float32)
        audio[256] = 1
        self.assertEqual(np.count_nonzero(onset_features(audio)), 0)
        audio[0] = 1
        self.assertGreater(np.count_nonzero(onset_features(audio)), 0)

    def test_targets_follow_exclusive_frame_end_and_preserve_polyphony(self):
        events_ = [{"t": 0., "pc": 9}, {"t": .032, "pc": 4}, {"t": .032, "pc": 1},
                   {"t": .320, "pc": 9}]
        labels = onset_targets(events_, 30)
        self.assertEqual(labels[:6, 9].tolist(), [1.] * 6)
        self.assertEqual(labels[6:20, 9].sum(), 0)
        self.assertEqual(labels[20:26, 9].sum(), 6)
        self.assertEqual(labels[1, 4], 0)
        self.assertEqual(labels[2, [1, 4]].tolist(), [1., 1.])

    def test_every_training_frame_is_used_once_with_history_and_padding_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npy"
            features = np.ones((267, FEATURE_DIM), dtype=np.float16)
            np.save(path, features)
            source = {"features": str(path), "frames": 267, "events": [], "split": "train"}
            dataset = OnsetBlocks([source])
            self.assertEqual(len(dataset), 3)
            self.assertEqual(sum(int(dataset[i][2].sum()) for i in range(3)), 267)
            x, count = feature_block(features, 256)
            self.assertEqual(count, 11)
            np.testing.assert_array_equal(x[:, :HISTORY_FRAMES + count], 1)
            self.assertEqual(np.count_nonzero(x[:, HISTORY_FRAMES + count:]), 0)
            with self.assertRaisesRegex(ValueError, "train sources only"):
                OnsetBlocks([dict(source, split="validation")])

    def test_gain_spread_is_the_released_one_by_default_and_zero_turns_it_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npy"
            raw = np.random.default_rng(34).uniform(0, .8, (200, FEATURE_DIM)).astype(np.float16)
            np.save(path, raw)
            source = {"features": str(path), "frames": 200, "events": [], "split": "train"}
            np.random.seed(7)
            x = OnsetBlocks([source])[0][0]
            # The masking_v2 run drew 10 ** uniform(-.3, .3); the same draw, bit for bit.
            np.random.seed(7)
            gain = 10 ** np.random.uniform(-.3, .3)
            unscaled, _ = feature_block(raw, 0)
            np.testing.assert_array_equal(
                x, (np.log1p(np.expm1(unscaled * math.log(1001)) * gain) / math.log(1001)).astype(np.float32))
            plain = OnsetBlocks([source], gain_db=0.)[0][0]
            np.testing.assert_allclose(plain, unscaled, atol=1e-6)

    def test_resume_accepts_wav_timestamp_change_only_with_identical_model_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "synthetic.wav"
            sf.write(wav, np.sin(np.arange(16000) * .1).astype(np.float32), 16000, subtype="FLOAT")
            source = dict(id="synthetic-note", domain="synthetic", split="train",
                          case="root_hold", duration=1., wav=str(wav), sha256=sha256(wav),
                          events=[dict(id="note", t=.2, pc=0, end=.8)])
            original = cache_onset_features([source], root / "first")[0]
            saved = [tuple(original[k] for k in ("id", "split", "sha256", "feature_sha256", "events"))]
            # Deterministically simulate a WAV written one second later.
            data = bytearray(wav.read_bytes())
            offset = 12
            while offset + 8 <= len(data):
                tag, size = struct.unpack_from("<4sI", data, offset)
                if tag == b"PEAK":
                    timestamp = struct.unpack_from("<I", data, offset + 12)[0]
                    struct.pack_into("<I", data, offset + 12, timestamp + 1)
                    break
                offset += 8 + size + size % 2
            else:
                self.fail("FLOAT WAV must have a PEAK timestamp for this regression")
            wav.write_bytes(data)
            regenerated = cache_onset_features([dict(source, sha256=sha256(wav))], root / "second")[0]
            self.assertNotEqual(original["sha256"], regenerated["sha256"])
            self.assertEqual(original["feature_sha256"], regenerated["feature_sha256"])
            report_path = root / "resume_data_check.json"
            result = validate_resume_sources(saved, [regenerated], report_path)
            self.assertTrue(result["ok"])
            self.assertEqual(result["synthetic_container_changes"], 1)
            for field, value in [("id", "another"), ("split", "test"),
                                 ("feature_sha256", "changed"), ("events", [])]:
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "changed data"):
                    validate_resume_sources(saved, [dict(regenerated, **{field: value})], report_path)
            with self.assertRaisesRegex(ValueError, "changed data"):
                validate_resume_sources(saved, [dict(regenerated, domain="guitarset")], report_path)
            # Even unchanged index metadata cannot conceal a corrupt array.
            Path(regenerated["features"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "feature_file"):
                validate_resume_sources(saved, [regenerated], report_path)

    def test_resume_keeps_source_order_count_and_annotations_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "check.json"
            sources = [dict(id=str(i), split="train", domain="guitarset", sha256="audio",
                            feature_sha256="features", events=[dict(t=.2, pc=i)]) for i in range(2)]
            saved = [tuple(s[k] for k in ("id", "split", "sha256", "feature_sha256", "events"))
                     for s in copy.deepcopy(sources)]
            self.assertTrue(validate_resume_sources(saved, sources, path)["ok"])
            for changed in (sources[::-1], sources[:1], sources + [sources[0]]):
                with self.assertRaisesRegex(ValueError, "changed data"):
                    validate_resume_sources(saved, changed, path)
            sources[0]["events"][0]["t"] += .016
            with self.assertRaisesRegex(ValueError, "events"):
                validate_resume_sources(saved, sources, path)

    def test_masking_preparation_reaches_training_sources_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "input"
            inputs.mkdir()
            fixture(inputs)
            work = root / "work"
            sources = prepare_features(inputs, work, (1, 1, 1), masking_pairs=True)
            masking = [s for s in sources if s["case"].startswith("masking_")]
            self.assertEqual(len(masking), 9)
            self.assertEqual({s["split"] for s in masking}, {"train", "validation", "test"})
            self.assertTrue(all(Path(s["features"]).is_file() for s in masking))
            cached = prepare_features(inputs, work, (1, 1, 1), masking_pairs=True)
            self.assertEqual(json.loads(json.dumps(sources)), cached)
            # A new Kaggle session regenerates WAV headers rather than reusing
            # the previous work directory. Check the complete recovery path.
            rebuilt = prepare_features(inputs, root / "rebuilt", (1, 1, 1), masking_pairs=True)
            identities = [tuple(s[k] for k in ("id", "split", "sha256", "feature_sha256", "events"))
                          for s in sources]
            checked = validate_resume_sources(identities, rebuilt, root / "resume_data_check.json")
            self.assertTrue(checked["ok"])
            with self.assertRaisesRegex(ValueError, "masking recipe changed"):
                prepare_features(inputs, work, (1, 1, 1), masking_pairs=False)

    def test_masking_recipe_cannot_reuse_an_old_feature_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "features").mkdir()
            index = root / "features/index.json"
            for present in (False, True):
                sources = [dict(case="masking_add_fifth_-12db" if present else "root_hold")]
                index.write_text(json.dumps(dict(feature_spec=FEATURE_SPEC, sources=sources)))
                with self.assertRaisesRegex(ValueError, "masking recipe changed"):
                    prepare_features(root, root, (1, 1, 1), masking_pairs=not present)


class ExportCheckTests(unittest.TestCase):
    def source(self):
        return {"id": "clip", "frames": 125, "domain": "synthetic", "case": "root_hold", "duration": 2.,
                "events": [{"id": "attack", "t": .16, "pc": 9, "end": 2.}]}

    def test_tiny_probability_difference_can_change_threshold_event(self):
        source = self.source()
        a = np.zeros((125, 12), np.float32)
        a[12, 9] = .8 - 1e-7
        b = a.copy()
        b[12, 9] = .8 + 1e-7
        self.assertTrue(export_probability_check([(source, a)], [(source, b)])["ok"])
        ma, da = onset_metrics([(source, a)], .8)
        mb, db = onset_metrics([(source, b)], .8)
        compared = export_event_comparison(da, db)
        self.assertFalse(compared["events_identical"])
        self.assertEqual(compared["onnx_only_events"], 1)
        self.assertEqual((ma["groups"]["all"]["tp"], mb["groups"]["all"]["tp"]), (0, 1))

    def test_large_or_invalid_export_errors_are_not_hidden(self):
        source = self.source()
        a = np.zeros((125, 12), np.float32)
        b = a.copy()
        b[12, 9] = .001
        self.assertFalse(export_probability_check([(source, a)], [(source, b)])["ok"])
        for other in (b[:100], np.full_like(b, np.nan)):
            with self.assertRaises(ValueError):
                export_probability_check([(source, a)], [(source, other)])
        with self.assertRaises(ValueError):
            export_probability_check([(source, a)], [])


class RunTests(unittest.TestCase):
    def test_explicit_parent_checkpoint_resolution_and_missing_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(tmp)
            path = Path(tmp) / "parent.pth"
            path.write_bytes(b"parent")
            self.assertEqual(resolve_initial_onset("hf:parent.pth", store), path)
            self.assertEqual(resolve_initial_onset(str(path), store), path)
            self.assertIsNone(resolve_initial_onset("", store))
            for value in ("hf:missing.pth", str(Path(tmp) / "missing.pt")):
                with self.assertRaises(FileNotFoundError):
                    resolve_initial_onset(value, store)
            for value in ("hf:", "hf:../parent.pth", "hf:model.onnx"):
                with self.assertRaises(ValueError):
                    resolve_initial_onset(value, store)
            with patch.object(store, "fetch", side_effect=ConnectionError("HF unavailable")):
                with self.assertRaises(ConnectionError):
                    resolve_initial_onset("hf:parent.pth", store)

    def test_new_run_uses_parent_but_resume_does_not_require_it_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            store = SnapshotStore(work)
            parent = work / "checkpoint_v2_take7_onset_best.pth"
            parent.write_bytes(b"parent")
            # The old run's last checkpoint must never be resumed as the new run.
            (work / "checkpoint_v2_take7_onset_last.pth").write_bytes(b"old run")
            config = dict(work_dir=tmp, input_dir=tmp, run_tag=RUN_TAG, mode=MODE,
                          initial_onset=INITIAL_ONSET, onset_masking_pairs=True)
            sources = [dict(split=s, case="masking_add_fifth_-12db")
                       for s in ("train", "validation", "test")]
            for resumed in (False, True):
                if resumed:
                    (work / f"checkpoint_{RUN_TAG}_onset_last.pth").write_bytes(b"new run")
                    parent.unlink()
                with patch("strike_trainer.prepare_features", return_value=sources) as prepared, \
                     patch("strike_trainer.train_onset", side_effect=RuntimeError("stop before optimizer")) as train:
                    with self.assertRaisesRegex(RuntimeError, "stop before optimizer"):
                        run(config, store)
                    prepared.assert_called_once_with(Path(tmp), work, (96, 96, 96), True)
                    self.assertEqual(train.call_args.kwargs["initial_checkpoint"], None if resumed else parent)
                    self.assertEqual(train.call_args.kwargs["gain_db"], 6.)
                record = json.loads((work / "run_configuration.json").read_text())
                self.assertTrue(record["onset_masking_pairs"])
                self.assertEqual(record["onset_gain_db"], 6.)
            self.assertEqual((work / "rise/short_onset_last.pt").read_bytes(), b"new run")

    def test_tag_and_mode_are_checked_before_anything_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(tmp)
            with patch("strike_trainer.prepare_features", side_effect=AssertionError("data forbidden")):
                with self.assertRaisesRegex(ValueError, "simple RUN_TAG"):
                    run(dict(work_dir=tmp, run_tag="../v2", mode="train"), store)
                # The old trainer's modes trained or exported the chord base too.
                for mode in ("onset_only", "auto", "full"):
                    with self.assertRaisesRegex(ValueError, "train or export_only"):
                        run(dict(work_dir=tmp, run_tag="v2_take7", mode=mode), store)

    def test_access_error_never_becomes_fresh_training(self):
        from types import SimpleNamespace
        store = SimpleNamespace(fetch=lambda name: (_ for _ in ()).throw(ConnectionError("HF down")))
        with tempfile.TemporaryDirectory() as tmp:
            with patch("strike_trainer.prepare_features", side_effect=AssertionError("data forbidden")):
                for mode in ("train", "export_only"):
                    with self.assertRaisesRegex(ConnectionError, "HF down"):
                        run(dict(work_dir=tmp, run_tag="v2_take7", mode=mode, initial_onset=INITIAL_ONSET), store)


def rise_stand_in(path):
    """A tiny exact graph with the Rise network's inputs and outputs."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    weights = [numpy_helper.from_array(np.array([v], np.int64), name)
               for name, v in [("start", 0), ("end", 12), ("axis", 1)]]
    graph = helper.make_graph(
        [helper.make_node("Slice", ["short_features", "start", "end", "axis"], ["onset_logits"])], "rise",
        [helper.make_tensor_value_info("short_features", TensorProto.FLOAT, ["batch", 770, "time"])],
        [helper.make_tensor_value_info("onset_logits", TensorProto.FLOAT, ["batch", 12, "time"])], weights)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), str(path))
    return path


def finished_run(work, tag="v2_take7", threshold=.8, seed=5):
    """A finished run's snapshots: the best checkpoint and, with a threshold, its summary."""
    import torch
    torch.manual_seed(seed)
    model = make_onset_model()
    torch.nn.init.normal_(model.rise_project.weight, std=.025)
    contract = dict(feature_spec=FEATURE_SPEC, history_frames=HISTORY_FRAMES, spectral_rise=True)
    torch.save(dict(state_dict=model.state_dict(), epoch=1, contract=contract),
               work / f"checkpoint_{tag}_onset_best.pth")
    if threshold is not None:
        (work / f"training_summary_{tag}.json").write_text(json.dumps(dict(
            training_complete=True, onset=dict(threshold=threshold, validation={"keep": "all metrics"}))))
    return model.eval()


@unittest.skipUnless(HAS_ONNX, "onnx and onnxruntime needed")
class StrikeFileTests(unittest.TestCase):
    def test_metadata_is_what_the_app_reads_and_answers_do_not_move(self):
        import onnx
        import onnxruntime as ort
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = rise_stand_in(root / "short_onset_rise.onnx")
            before = sha256(source)
            strike = write_strike_model(source, root / "app" / "short_onset_v2_take7.onnx", .9, "v2_take7")
            self.assertEqual(sha256(source), before)
            model = onnx.load(strike["path"])
            self.assertEqual([v.name for v in model.graph.input], ["short_features"])
            self.assertEqual([v.name for v in model.graph.output], ["onset_logits"])
            metadata = {p.key: p.value for p in model.metadata_props}
            # src/strike.rs parses onset_threshold as f32.
            self.assertEqual(metadata["onset_threshold"], "0.9")
            self.assertEqual(metadata["onset_history_frames"], "34")
            self.assertEqual(json.loads(metadata["onset_feature_spec"]), FEATURE_SPEC)
            self.assertEqual(metadata["run_tag"], "v2_take7")
            x = np.random.default_rng(3).random((2, 770, 40), dtype=np.float32)
            a, b = (ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                    .run(None, {"short_features": x})[0] for p in (source, strike["path"]))
            np.testing.assert_array_equal(a, b)

    def test_threshold_and_graph_are_checked_before_writing(self):
        import onnx
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = rise_stand_in(root / "rise.onnx")
            output = root / "short_onset_x.onnx"
            for threshold in (0, 1, -.5):
                with self.assertRaisesRegex(ValueError, "threshold"):
                    write_strike_model(source, output, threshold, "x")
            model = onnx.load(str(source))
            model.graph.input[0].name = "features"
            model.graph.node[0].input[0] = "features"
            onnx.save(model, str(source))
            with self.assertRaisesRegex(ValueError, "Rise network"):
                write_strike_model(source, output, .9, "x")
            self.assertFalse(output.exists())


@unittest.skipUnless(HAS_TRAINING, "torch, onnx and onnxruntime needed")
class ExportOnlyTests(unittest.TestCase):
    def test_export_only_matches_the_checkpoint_and_never_calls_training(self):
        import torch
        import onnxruntime as ort
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = finished_run(root)
            config = dict(work_dir=tmp, run_tag="v2_take7", mode="export_only")
            with patch("strike_trainer.prepare_features", side_effect=AssertionError("features forbidden")), \
                 patch("strike_trainer.train_onset", side_effect=AssertionError("training forbidden")):
                store = SnapshotStore(tmp)
                with patch.object(store, "publish", wraps=store.publish) as publish:
                    report = run(config, store)
                    self.assertEqual([c.args[1] for c in publish.call_args_list],
                                     ["short_onset_v2_take7.onnx", "training_summary_v2_take7.json"])
            self.assertEqual(report["onset"]["validation"], {"keep": "all metrics"})
            self.assertFalse(report["training_performed"])
            self.assertEqual(report["strike_model"]["metadata"]["onset_threshold"], "0.8")
            x = np.random.default_rng(4).uniform(0, .5, (2, FEATURE_DIM, 70)).astype(np.float32)
            session = ort.InferenceSession(str(root / "short_onset_v2_take7.onnx"), providers=["CPUExecutionProvider"])
            with torch.inference_mode():
                expected = model(torch.from_numpy(x)).numpy()
            np.testing.assert_allclose(session.run(None, {"short_features": x})[0], expected, rtol=1e-5, atol=1e-5)

    def test_missing_checkpoint_threshold_or_contract_never_starts_training(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(work_dir=tmp, run_tag="v2_take7", mode="export_only")
            with patch("strike_trainer.train_onset", side_effect=AssertionError("training forbidden")):
                with self.assertRaisesRegex(FileNotFoundError, "No training"):
                    run(config, SnapshotStore(tmp))
                finished_run(root, threshold=None)
                with self.assertRaisesRegex(ValueError, "Missing onset threshold"):
                    run(config, SnapshotStore(tmp))
                config["export_onset_threshold"] = .8
                self.assertTrue(run(config, SnapshotStore(tmp))["ok"])
                checkpoint = root / "checkpoint_v2_take7_onset_best.pth"
                saved = torch.load(checkpoint, weights_only=True)
                saved["contract"]["history_frames"] = 30
                torch.save(saved, checkpoint)
                with self.assertRaisesRegex(ValueError, "contract"):
                    run(config, SnapshotStore(tmp))

    def test_the_whole_file_exports_from_a_copy_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "v2_take7"
            work.mkdir()
            finished_run(work)
            script = root / "trainer.py"
            script.write_text(TRAINER.read_text())
            done = subprocess.run([sys.executable, str(script), "--mode", "export_only", "--run-tag", "v2_take7",
                                   "--no-hf", "--output-root", str(root)],
                                  cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            report = json.loads((work / "training_summary_v2_take7.json").read_text())
            self.assertEqual(Path(report["strike_model"]["path"]).name, "short_onset_v2_take7.onnx")
            self.assertFalse(report["training_performed"])


@unittest.skipUnless(HAS_TRAINING, "torch, onnx and onnxruntime needed")
class TrainingTests(unittest.TestCase):
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

    def test_rise_projection_starts_at_zero_and_learns(self):
        import torch
        torch.manual_seed(29)
        model = make_onset_model()
        self.assertEqual(torch.count_nonzero(model.rise_project.weight).item(), 0)
        x = torch.rand(2, FEATURE_DIM, 90)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), torch.ones(2, 12, 90))
        loss.backward()
        self.assertGreater(float(model.rise_project.weight.grad.abs().sum()), 0)

    def test_network_is_causal_and_matches_the_full_file_at_block_boundaries(self):
        import torch
        torch.manual_seed(17)
        model = make_onset_model().eval()
        # A zero projection would hide missing rise context, so exercise nonzero weights.
        torch.nn.init.normal_(model.rise_project.weight, std=.025)
        features = np.random.default_rng(12).uniform(0, .8, (389, FEATURE_DIM)).astype(np.float32)
        full = np.pad(features.T, ((0, 0), (HISTORY_FRAMES, 0)))[None]
        with torch.inference_mode():
            expected = model(torch.from_numpy(full)).numpy()[0, :, HISTORY_FRAMES:]
            changed = full.copy()
            changed[:, :, HISTORY_FRAMES + 200:] = 0
            early = model(torch.from_numpy(changed)).numpy()[0, :, HISTORY_FRAMES:HISTORY_FRAMES + 200]
            np.testing.assert_allclose(expected[:, :200], early, atol=1e-6)
            for start, length in ((0, 1), (1, 1), (127, 1), (128, 128), (256, 128), (384, 128)):
                x, count = feature_block(features, start, length)
                actual = model(torch.from_numpy(x[None])).numpy()[0, :, HISTORY_FRAMES:HISTORY_FRAMES + count]
                np.testing.assert_allclose(actual, expected[:, start:start + count], atol=1e-5)
        self.assertEqual(HISTORY_FRAMES, 30 + RISE_PAST_FRAMES)

    def test_interrupted_resume_equals_uninterrupted_and_finished_run_does_not_train(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "input"
            inputs.mkdir()
            fixture(inputs)
            prepared = run_pipeline(inputs, root / "prepared", "auto", groups=(1, 1, 1))
            features = root / "features"
            sources = cache_onset_features(onset_sources(Path(prepared["prepared_directory"])), features)
            full, resumed = root / "full", root / "resumed"
            full.mkdir()
            resumed.mkdir()
            kwargs = dict(epochs=2, batch_size=16, device_name="cpu", feature_directory=features, resume=True)
            reference = train_onset(sources, full, **kwargs)

            def interrupt(path):
                raise RuntimeError("simulated interruption after checkpoint")
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train_onset(sources, resumed, checkpoint_callback=interrupt, **kwargs)
            recovered = train_onset(sources, resumed, **kwargs)
            a = torch.load(full / "short_onset_last.pt", weights_only=False)
            b = torch.load(resumed / "short_onset_last.pt", weights_only=False)
            for key in a["state_dict"]:
                torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)
            self.assertEqual(reference["input_batches_sha256"], recovered["input_batches_sha256"])
            self.assertEqual(reference["test"], recovered["test"])
            with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("unexpected retraining")):
                train_onset(sources, resumed, **kwargs)
            with self.assertRaisesRegex(ValueError, "changed configuration"):
                train_onset(sources, resumed, **dict(kwargs, gain_db=12.))
            bad = dict(sources[0])
            bad["sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "changed data"):
                train_onset([bad] + sources[1:], resumed, **kwargs)

    def test_complete_run_from_a_copy_without_repository_or_hf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "input"
            inputs.mkdir()
            fixture(inputs)
            work = root / "v2_take7_check"
            work.mkdir()
            config = dict(run_tag="v2_take7_check", mode="train", initial_onset="",
                          input_dir=str(inputs), work_dir=str(work), device="cpu",
                          groups=[1, 1, 1], onset_epochs=1, onset_batch_size=16)
            (root / "standalone.py").write_text(TRAINER.read_text())
            runner = root / "run.py"
            runner.write_text("import torch; torch.set_num_threads(2)\nimport standalone as m\n"
                              f"c = {config!r}\nr = m.run(c, m.SnapshotStore(c['work_dir']))\nassert r['ok']\n")
            done = subprocess.run([sys.executable, str(runner)], cwd=root, capture_output=True,
                                  text=True, timeout=300)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            report = json.loads((work / "training_summary_v2_take7_check.json").read_text())
            self.assertTrue(report["ok"])
            self.assertEqual(Path(report["strike_model"]["path"]).name, "short_onset_check.onnx")
            self.assertEqual(float(report["strike_model"]["metadata"]["onset_threshold"]), report["onset"]["threshold"])
            self.assertEqual(report["onset"]["training_gain_db"], [-6., 6.])
            self.assertEqual(sorted(p.name for p in work.glob("*.onnx")), ["short_onset_check.onnx"])
            # The same run again: finished, so no optimizer step.
            runner.write_text("import torch; torch.set_num_threads(2)\nimport standalone as m\n"
                              "from unittest.mock import patch\n"
                              f"c = {config!r}\n"
                              "with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retraining')):\n"
                              "    r = m.run(c, m.SnapshotStore(c['work_dir']))\nassert r['ok']\n")
            again = subprocess.run([sys.executable, str(runner)], cwd=root, capture_output=True,
                                   text=True, timeout=300)
            self.assertEqual(again.returncode, 0, again.stdout + again.stderr)


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-fixture"]:
        write_feature_fixture()
    else:
        unittest.main()
