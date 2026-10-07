"""Physical-time, sustained-note, isolation and runnable-training contract checks."""

import importlib.util
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import struct
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from build_train_onset_kaggle import build
from onset_events import Event, match_events
from test_onset_preparation import fixture
from train_short_onset import (BLOCK_FRAMES, FEATURE_DIM, HISTORY, OnsetBlocks,
                              feature_block, local_event_matches, make_onset_model,
                              onset_features, onset_metrics, onset_predictions,
                              onset_resample, onset_targets, validation_choice, training_main)
from train_short_onset import cache_onset_features, sha256, validate_resume_sources


HAS_TRAINING = all(importlib.util.find_spec(n) for n in ("torch", "onnx", "onnxruntime"))


class OnsetTrainingTests(unittest.TestCase):
    def test_resume_accepts_wav_timestamp_change_only_with_identical_model_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / 'synthetic.wav'
            sf.write(wav, np.sin(np.arange(16000) * .1).astype(np.float32), 16000, subtype='FLOAT')
            source = dict(id='synthetic-note', domain='synthetic', split='train',
                          case='root_hold', duration=1., wav=str(wav), sha256=sha256(wav),
                          events=[dict(id='note', t=.2, pc=0, end=.8)])
            original = cache_onset_features([source], root / 'first')[0]
            saved = [tuple(original[k] for k in ('id', 'split', 'sha256', 'feature_sha256', 'events'))]
            # Deterministically simulate a WAV written one second later.
            data = bytearray(wav.read_bytes())
            offset = 12
            while offset + 8 <= len(data):
                tag, size = struct.unpack_from('<4sI', data, offset)
                if tag == b'PEAK':
                    timestamp = struct.unpack_from('<I', data, offset + 12)[0]
                    struct.pack_into('<I', data, offset + 12, timestamp + 1)
                    break
                offset += 8 + size + size % 2
            else:
                self.fail('FLOAT WAV must have a PEAK timestamp for this regression')
            wav.write_bytes(data)
            regenerated = cache_onset_features([dict(source, sha256=sha256(wav))], root / 'second')[0]
            self.assertNotEqual(original['sha256'], regenerated['sha256'])
            self.assertEqual(original['feature_sha256'], regenerated['feature_sha256'])
            report_path = root / 'resume_data_check.json'
            result = validate_resume_sources(saved, [regenerated], report_path)
            self.assertTrue(result['ok'])
            self.assertEqual(result['synthetic_container_changes'], 1)
            self.assertEqual(json.loads(report_path.read_text()), result)
            for field, value in [('id', 'another'), ('split', 'test'),
                                 ('feature_sha256', 'changed'), ('events', [])]:
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'changed data'):
                    validate_resume_sources(saved, [dict(regenerated, **{field: value})], report_path)
                self.assertIn(field, json.loads(report_path.read_text())['examples'][0]['fields'])
            with self.assertRaisesRegex(ValueError, 'changed data'):
                validate_resume_sources(saved, [dict(regenerated, domain='guitarset')], report_path)
            # Even unchanged index metadata cannot conceal a corrupt array.
            Path(regenerated['features']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'feature_file'):
                validate_resume_sources(saved, [regenerated], report_path)

    def test_resume_keeps_source_order_count_and_annotations_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'check.json'
            sources = [dict(id=str(i), split='train', domain='guitarset', sha256='audio',
                            feature_sha256='features', events=[dict(t=.2, pc=i)]) for i in range(2)]
            saved = [tuple(s[k] for k in ('id', 'split', 'sha256', 'feature_sha256', 'events'))
                     for s in copy.deepcopy(sources)]
            self.assertTrue(validate_resume_sources(saved, sources, path)['ok'])
            for changed in (sources[::-1], sources[:1], sources + [sources[0]]):
                with self.assertRaisesRegex(ValueError, 'changed data'):
                    validate_resume_sources(saved, changed, path)
            sources[0]['events'][0]['t'] += .016
            with self.assertRaisesRegex(ValueError, 'events'):
                validate_resume_sources(saved, sources, path)

    def test_entry_point_reports_export_failure_without_removing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "onset-prepared-example" / "control" / "short_onset_best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"completed checkpoint")
            error = ValueError("ONNX probability error exceeds tolerance")
            error.add_note(f"Partial run and any completed checkpoints: {checkpoint.parent.parent}")
            stdout = io.StringIO()
            with patch("train_short_onset.run_training_pipeline", side_effect=error), contextlib.redirect_stdout(stdout):
                report = training_main(["--output-root", tmp])
            saved = json.loads((root / "training_failure.json").read_text())
            self.assertEqual(report, saved)
            self.assertFalse(saved["ok"])
            self.assertFalse(saved["training_complete"])
            self.assertFalse(saved["app_ready"])
            self.assertIn(str(checkpoint.parent.parent), saved["traceback"])
            self.assertIn("TRAINING FAILED", stdout.getvalue())
            self.assertEqual(checkpoint.read_bytes(), b"completed checkpoint")

    def test_generated_script_has_no_repository_imports(self):
        directory = Path(__file__).resolve().parent
        generated = build(directory)
        self.assertNotIn("from prepare_onset_kaggle import", generated)
        self.assertNotIn("from onset_events import", generated)
        compile(generated, "standalone.py", "exec")

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
            # Running on just the prefix also produces exactly the same past.
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
        events = [{"t": 0., "pc": 9}, {"t": .032, "pc": 4}, {"t": .032, "pc": 1},
                  {"t": .320, "pc": 9}]
        labels = onset_targets(events, 30)
        self.assertEqual(labels[:6, 9].tolist(), [1.] * 6)
        self.assertEqual(labels[6:20, 9].sum(), 0)
        self.assertEqual(labels[20:26, 9].sum(), 6)
        self.assertEqual(labels[1, 4], 0)
        self.assertEqual(labels[2, [1, 4]].tolist(), [1., 1.])

    def test_component_matching_equals_full_scorer_including_close_repeats(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            refs = [Event(f"r{i}", float(t), i % 3) for i, t in enumerate(rng.integers(0, 500, 45) / 100)]
            preds = [Event(f"p{i}", float(t), i % 3) for i, t in enumerate(rng.integers(0, 500, 60) / 100)]
            expected = {(refs[i].id, preds[j].id) for i, j in match_events(refs, preds, .032, .128)}
            actual = {(r.id, p.id) for r, p in local_event_matches(refs, preds)}
            self.assertEqual(expected, actual)

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
            np.testing.assert_array_equal(x[:, :HISTORY + count], 1)
            self.assertEqual(np.count_nonzero(x[:, HISTORY + count:]), 0)
            with self.assertRaisesRegex(ValueError, "train sources only"):
                OnsetBlocks([dict(source, split="validation")])

    @unittest.skipUnless(HAS_TRAINING, "temporary Torch/ONNX test environment required")
    def test_causal_network_chunking_and_small_training_step(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(17)
        model = make_onset_model().eval()
        rng = np.random.default_rng(12)
        features = rng.uniform(0, .8, (389, FEATURE_DIM)).astype(np.float32)
        full = np.pad(features.T, ((0, 0), (HISTORY, 0)))[None]
        with torch.inference_mode():
            expected = model(torch.from_numpy(full)).numpy()[0, :, HISTORY:]
            changed = full.copy()
            changed[:, :, HISTORY + 200:] = 0
            early = model(torch.from_numpy(changed)).numpy()[0, :, HISTORY:HISTORY + 200]
            np.testing.assert_allclose(expected[:, :200], early, atol=1e-6)
            for start in (0, 128, 256, 384):
                x, count = feature_block(features, start)
                actual = model(torch.from_numpy(x[None])).numpy()[0, :, HISTORY:HISTORY + count]
                # Different convolution lengths can select different float32 kernels.
                np.testing.assert_allclose(actual, expected[:, start:start + count], atol=1e-5)
            # Ranking sees six target frames with the same causal history as a full file.
            for start in (0, 117, 233):
                x, count = feature_block(features, start, length=6)
                actual = model(torch.from_numpy(x[None])).numpy()[0, :, HISTORY:]
                np.testing.assert_allclose(actual, expected[:, start:start + count], atol=1e-5)
        # Exercise backward, not just inference; the model must learn a small labelled batch.
        model.train()
        x = torch.from_numpy(full[:, :, :HISTORY + BLOCK_FRAMES])
        target = torch.zeros((1, 12, BLOCK_FRAMES))
        target[:, 9, 30:36] = 1
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        losses = []
        for _ in range(15):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x)[:, :, HISTORY:], target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0] * .7)

    @unittest.skipUnless(HAS_TRAINING, "temporary Torch/ONNX test environment required")
    def test_standalone_training_export_and_full_test_without_prior_manifest(self):
        source = build(Path(__file__).resolve().parent)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "input"
            inputs.mkdir()
            manifest, _ = fixture(inputs)
            manifest.unlink()
            # Execute outside the repo exactly as a pasted notebook script, with kernel argv.
            code = source.replace('default=Path("/kaggle/input")', f"default=Path({str(inputs)!r})")
            code = code.replace('default=Path("/kaggle/working")', f"default=Path({str(root / 'output')!r})")
            code = code.replace("default=(60, 12, 12)", "default=(1, 1, 1)")
            code = code.replace("TRAIN_EPOCHS = 12", "TRAIN_EPOCHS = 1")
            code = 'import sys\nsys.modules["ipykernel"] = object()\n' + code
            script = root / "standalone.py"
            script.write_text(code)
            run = subprocess.run([sys.executable, str(script), "-f", "kernel.json"], cwd=root,
                                 capture_output=True, text=True, timeout=180, env=dict(os.environ))
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            reports = list((root / "output").glob("*/training_summary.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertTrue(report["ok"])
            self.assertFalse(report["app_ready"])
            self.assertTrue(report["controlled_training_verified"])
            self.assertEqual(report["experiment"], "positive-spectral-rise-v1")
            self.assertEqual(set(report["arms"]), {"control", "rise"})
            for arm in report["arms"].values():
                self.assertEqual(arm["checkpoint_epoch"], 1)
                self.assertLessEqual(arm["onnx_max_probability_error"], 2e-5)
                self.assertEqual(arm["evaluation_backend"], "ONNX Runtime CPU")
                self.assertEqual(arm["validation"], arm["onnx_validation_comparison"]["onnx_metrics"])
                self.assertEqual(arm["test"]["groups"]["all"]["recordings"], 10)
                self.assertTrue(Path(arm["model"]).is_file())
            self.assertEqual(report["arms"]["control"]["shared_initial_weights_sha256"],
                             report["arms"]["rise"]["shared_initial_weights_sha256"])
            self.assertEqual(report["arms"]["control"]["training_batches_sha256"],
                             report["arms"]["rise"]["training_batches_sha256"])
            self.assertNotEqual(report["arms"]["control"]["input_batches_sha256"],
                                report["arms"]["rise"]["input_batches_sha256"])
            self.assertNotEqual(report["arms"]["control"]["model_sha256"],
                                report["arms"]["rise"]["model_sha256"])
            run_dir = reports[0].parent
            contracts = [json.loads((run_dir / arm / "contract.json").read_text())
                         for arm in ("control", "rise")]
            self.assertEqual([c["ringing_negative_weight"] for c in contracts], [1., 1.])
            self.assertEqual([c["pair_weight"] for c in contracts], [0., 0.])
            self.assertEqual([c["history_frames"] for c in contracts], [30, 34])
            self.assertEqual([c["features"] for c in contracts], [770, 770])
            self.assertEqual([c["parameters"] for c in contracts], [186156, 260076])
            self.assertEqual(report["pair_audits"]["train"]["pairs"], 12)
            for arm in ("control", "rise"):
                history = json.loads((run_dir / arm / "history.json").read_text())
                self.assertEqual(history[0]["pair_count"], 0)
                self.assertIsNone(history[0]["pair_loss"])
                saved_table = json.loads((run_dir / arm / "validation_thresholds.json").read_text())["table"]
                compact_table = report["arms"][arm]["validation_thresholds"]
                self.assertEqual(len(compact_table), len(saved_table))
                for saved, compact in zip(saved_table, compact_table):
                    self.assertEqual(saved["threshold"], compact["threshold"])
                    for name, group in saved["groups"].items():
                        self.assertEqual(compact["groups"][name],
                                         {key: value for key, value in group.items() if not isinstance(value, list)})
            self.assertEqual(contracts[0]["data_index_sha256"], contracts[1]["data_index_sha256"])
            self.assertGreater(report["ringing_label_audit"]["train/synthetic"]["weighted_frames_classes"], 0)
            self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()
