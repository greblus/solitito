"""Compare the uploaded short detector with the original head on identical WAVs.

Original features come from the existing Rust CqtAnalyzer test exporter, not
an offline approximation. The original ONNX runs in batches, verified against
a complete archived Rust probe before comparison. No production files change.
Kaggle test synthesis is regenerated and candidate events must reproduce the
uploaded report. This is retrospective analysis, not new threshold selection.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import soundfile as sf

from onset_events import Event, latch_events, read_events, read_probe, score, sha256
from prepare_onset_data import render_group, write_json
from train_short_onset import (FEATURE_SPEC, HISTORY, BLOCK_FRAMES, onset_features,
                              onset_resample, onset_predictions, onset_metrics)


def session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class CachedOriginal:
    """Reuse only bit-identical 48-frame inputs, including across paired clips.

    Silence, shared prefixes and windows unaffected by the gate otherwise make
    the expensive original encoder do exactly the same work many times.
    No rounding of features, probability thresholding or event caching.
    """
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def run(self, names, feed):
        inputs = feed["features"]
        keys = [hashlib.sha256(x.tobytes()).digest() for x in inputs]
        missing = {}
        for key, row in zip(keys, inputs):
            if key not in self.cache:
                missing.setdefault(key, row)
        if missing:
            logits = self.model.run(names, {"features": np.stack(list(missing.values()))})[0]
            for key, row in zip(missing, logits):
                self.cache[key] = row.copy()
        return [np.stack([self.cache[key] for key in keys])]


def rust_features(wav, binary, output, repo):
    cached = output.with_suffix(".npz")
    if cached.exists():
        data = np.load(cached)
        return data["ends"] / 16000, data["features"], data["rms"]
    environment = dict(os.environ, SOLITITO_FEATURE_WAV=str(wav.resolve()),
                       SOLITITO_FEATURE_OUTPUT=str(output.resolve()), SOLITITO_FEATURE_BOOST="0")
    run = subprocess.run([str(binary.resolve()), "--exact", "audio::onset_feature_probe::export_causal_features",
                          "--ignored", "--nocapture"], env=environment, cwd=repo,
                         capture_output=True, text=True, timeout=120)
    if run.returncode or not output.is_file():
        raise RuntimeError(f"Rust feature export failed: {run.stdout}\n{run.stderr}")
    document = json.loads(output.read_text())
    frames = document["frames"]
    ends = np.array([f["end_sample"] for f in frames])
    features = np.array([f["features"] for f in frames], dtype=np.float32)
    rms = np.array([f["rms"] for f in frames], dtype=np.float32)
    duration = sf.info(wav).duration
    if (document["target_rate"], document["fft_samples"], document["hop_samples"],
            document["input_gain"], document["bass_boost"]) != (16000, 8192, 256, 2., 0.):
        raise ValueError("Unexpected Rust DSP settings")
    if not np.array_equal(ends, 8192 + np.arange(len(ends)) * 256):
        raise ValueError("Incomplete Rust feature grid")
    if features.shape != (len(ends), 168) or not np.isfinite(features).all():
        raise ValueError("Invalid Rust features")
    if not 0 <= duration - ends[-1] / 16000 < .017:
        raise ValueError("Rust export did not reach the end of the WAV")
    np.savez_compressed(output.with_suffix(".npz"), ends=ends, features=features, rms=rms)
    output.unlink()  # Our temporary verbose export; keep the lossless float32 NPZ.
    return ends / 16000, features, rms


def original_probabilities(model, times, features, rms, gate=-34., batch=32):
    live = rms > np.float32(10 ** (gate / 20))
    gated = features.copy()
    gated[~live] = 0
    windows = np.lib.stride_tricks.sliding_window_view(gated, 48, axis=0).transpose(0, 2, 1)
    fill = np.lib.stride_tricks.sliding_window_view(live, 48).mean(axis=1) * 100
    chunks = []
    for start in range(0, len(windows), batch):
        logits = model.run(["onset_logits"], {"features": np.ascontiguousarray(windows[start:start + batch])})[0]
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite original model output")
        chunks.append(1 / (1 + np.exp(-np.clip(logits, -80, 80))))
    return times[47:], fill, np.concatenate(chunks)


def verify_original(model, binary, repo, output):
    archive = repo / "dist/crediting_measurements/onset-stage1-old-fixed-20260920"
    manifest = json.loads((archive / "manifest.json").read_text())
    wav = Path(manifest["inputs"]["wav"]["path"])
    for name, path in (("wav", wav), ("dsp_weights", repo / "dsp_weights.json")):
        if sha256(path) != manifest["inputs"][name]["sha256"]:
            raise ValueError(f"Changed reference probe input: {name}")
    if sha256(repo / "best_model_v2_take6_onset.onnx") != manifest["inputs"]["model"]["sha256"]:
        raise ValueError("Original model differs from archived probe")
    if sha256(archive / "probe.txt") != manifest["probe_sha256"]:
        raise ValueError("Archived probe changed")
    saved = output / "baseline-verification-summary.json"
    if saved.exists():
        previous = json.loads(saved.read_text())
        if (previous["ok"] and previous["reference_probe_sha256"] == sha256(archive / "probe.txt")
                and previous["wav_sha256"] == sha256(wav)):
            return previous  # Full validation already completed in this fingerprint-checked run.
    times, features, rms = rust_features(wav, binary, output / "baseline-verification.json", repo)
    t, fill, probabilities = original_probabilities(model, times, features, rms)
    archived = read_probe(archive / "probe.txt", sf.info(wav).duration)
    old_t = np.array([row[0] for row in archived])
    old_fill = np.array([row[1] for row in archived])
    old_prob = np.array([row[2] for row in archived])
    if probabilities.shape != old_prob.shape:
        raise ValueError("Full-file baseline comparison has different frame counts")
    delta = float(np.max(np.abs(probabilities - old_prob)))
    if delta > .0051 or np.max(np.abs(old_t - t)) > .0051 or np.max(np.abs(old_fill - fill)) > .51:
        raise ValueError(f"Python ONNX batch disagrees with the actual Rust probe: max probability error {delta}")
    result = {"ok": True, "frames": len(t), "max_error_vs_rounded_rust_probability": delta,
              "reference_probe_sha256": sha256(archive / "probe.txt"), "wav_sha256": sha256(wav)}
    write_json(output / "baseline-verification-summary.json", result)
    return result


def events_from_arrays(times, probabilities, threshold, fill=None, fill_min=0, stride=1, phase=0):
    if fill is None:
        fill = np.full(len(times), 100.)
    rows = ((float(t), float(f), p) for t, f, p in zip(times, fill, probabilities))
    return latch_events(rows, threshold=threshold, fill_min=fill_min,
                        stride=stride, phase=phase)


def describe_extras(refs, result):
    """Factual timing/class relations; do not infer picking technique from a score."""
    matched = {m["reference_id"] for m in result["matches"]}
    descriptions = []
    for extra in result["extra"]:
        previous = sorted((r for r in refs if r.t <= extra["t"]), key=lambda r: r.t)
        same = [r for r in previous if r.pc == extra["pc"]]
        nearby = [r for r in refs if -.032 <= extra["t"] - r.t <= .128]
        prior = same[-1] if same else None
        if prior and prior.id in matched:
            kind = "already_matched_pc"
        elif prior:
            kind = "unmatched_prior_pc"
        else:
            kind = "pc_not_previously_played"
        descriptions.append({**extra, "relation": kind,
                             "last_same_pc": prior.id if prior else None,
                             "since_same_pc": extra["t"] - prior.t if prior else None,
                             "nearby_attack_ids": [r.id for r in nearby],
                             "since_any_attack": extra["t"] - previous[-1].t if previous else None})
    return descriptions


def compare_events(source, predictions):
    refs = [Event(e["id"], e["t"], e["pc"], e.get("midi"), e.get("case", source["case"]))
            for e in source["events"]]
    results = {}
    for name, events in predictions.items():
        for window, early, late in (("strict", .032, .128), ("wide", .05, .4)):
            result = score(refs, events, 0, source["duration"], early=early, late=late)
            challenges = {e["id"] for e in source["events"] if e.get("role") == "challenge"}
            challenge_hits = sum(m["reference_id"] in challenges for m in result["matches"])
            result["challenge_reference"] = len(challenges)
            result["challenge_tp"] = challenge_hits
            result["extra_relations"] = describe_extras(refs, result)
            results[f"{name}/{window}"] = result
    return results


def aggregate(records):
    result = {}
    for record in records:
        for mode, scores in record["scores"].items():
            for group in (record["domain"], f"{record['domain']}/{record['case']}"):
                key = f"{group}:{mode}"
                row = result.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "challenge_tp": 0,
                                              "challenge_reference": 0, "seconds": 0., "latencies": [],
                                              "extra_relations": Counter()})
                for field in ("tp", "fp", "fn", "challenge_tp", "challenge_reference"):
                    row[field] += scores[field]
                row["seconds"] += record["duration"]
                row["latencies"].extend(m["delta"] for m in scores["matches"])
                row["extra_relations"].update(e["relation"] for e in scores["extra_relations"])
    for row in result.values():
        tp, fp, fn = (row[k] for k in ("tp", "fp", "fn"))
        row.update(precision=tp / (tp + fp) if tp + fp else 0,
                   recall=tp / (tp + fn) if tp + fn else 0,
                   f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                   false_events_per_minute=60 * fp / row["seconds"])
        latencies = row.pop("latencies")
        row["latency_p50"] = float(np.median(latencies)) if latencies else None
        row["latency_p95"] = float(np.percentile(latencies, 95)) if latencies else None
    return result


def compare(repo, binary, output, resume=False):
    output.mkdir(parents=True, exist_ok=resume)
    artifact_paths = {name: repo / f"{name}.json.txt" for name in
                      ("contract", "training_summary", "test_events", "validation_thresholds")}
    artifacts = {name: json.loads(path.read_text()) for name, path in artifact_paths.items()}
    contract, kaggle = artifacts["contract"], artifacts["training_summary"]
    candidate_path = repo / "short_onset_experimental.onnx"
    original_path = repo / "best_model_v2_take6_onset.onnx"
    if sha256(candidate_path) != kaggle["model_sha256"] or contract["feature_spec"] != FEATURE_SPEC:
        raise ValueError("Model/feature contract does not match the Kaggle run")
    if (contract["history_frames"], contract["block_frames"], contract["seed"]) != (HISTORY, BLOCK_FRAMES, 20260923):
        raise ValueError("Unexpected training contract")
    if kaggle["threshold"] != artifacts["validation_thresholds"]["selected_threshold"]:
        raise ValueError("Threshold report mismatch")
    summary = {"ok": False, "models": {"candidate": sha256(candidate_path), "original": sha256(original_path)},
               "inputs": {name: sha256(path) for name, path in artifact_paths.items()},
               "binary_sha256": sha256(binary), "dsp_sha256": sha256(repo / "dsp_weights.json"),
               "generator_sha256": sha256(repo / "dist/prepare_onset_data.py"),
               "comparison_script_sha256": sha256(Path(__file__)),
               "settings": {"candidate_threshold": kaggle["threshold"], "original_threshold": .6,
                            "original_gate_db": -34, "original_bass_boost": False,
                            "original_control_gate_db": -120, "step_seconds": .016},
               "limitations": ["Detector events, not application credit counts or runtime latency.",
                               "Synthetic test already inspected; no threshold tuning here.",
                               "AtoA labels are approximate pitch-run starts with two user-confirmed open notes, not exact attack truth.",
                               "Ordinary old and new feature paths differ; ungated old is an additional diagnostic control.",
                               "No local GuitarSet audio; only its uploaded summary can be examined."],
               "records": []}
    summary_path = output / "summary.json"
    if resume:
        previous = json.loads(summary_path.read_text())
        for key in ("models", "inputs", "binary_sha256", "dsp_sha256", "generator_sha256", "settings"):
            if previous[key] != summary[key]:
                raise ValueError(f"Cannot resume with changed {key}")
        summary["resumed_from_script_sha256"] = previous["comparison_script_sha256"]
    write_json(summary_path, summary)
    try:
        original, candidate = session(original_path), session(candidate_path)
        print("Verifying old-model inference against the complete archived Rust probe...", flush=True)
        summary["baseline_verification"] = verify_original(original, binary, repo, output)
        original = CachedOriginal(original)
        write_json(summary_path, summary)
        print("Baseline verified. Regenerating 96 Kaggle test clips...", flush=True)
        sources = []
        for group in range(12):
            for clip in render_group("test", group, contract["seed"], 16000):
                audio = clip.pop("audio")
                wav = output / (clip["name"] + ".wav")
                sf.write(wav, audio, 16000, subtype="FLOAT")
                sources.append(dict(clip, id=clip["name"], wav=str(wav), domain="synthetic"))
        # Keep the reviewed labels, including the unverified possible extra A at45.186.
        atoa = Path.home() / "Documents/AtoA.wav"
        reference = repo / "dist/crediting_measurements/atoa-reference-reviewed.csv"
        refs = read_events(reference)
        sources.append({"id": "AtoA", "case": "approximate_reference", "domain": "atoa",
                        "wav": str(atoa), "duration": sf.info(atoa).duration,
                        "reference_sha256": sha256(reference),
                        "events": [{"id": e.id, "t": e.t, "pc": e.pc, "midi": e.midi, "case": e.case} for e in refs]})
        uploaded = {r["source"]: r for r in artifacts["test_events"]}
        # Candidate first: require reproduction before paying for the old-model sweep.
        short_results = {}
        for index, source in enumerate(sources):
            audio, sr = sf.read(source["wav"], dtype="float32")
            features = onset_features(onset_resample(audio, sr))
            path = output / (source["id"] + "-short-features.npy")
            np.save(path, features)
            source.update(features=str(path), frames=len(features), wav_sha256=sha256(Path(source["wav"])))
            _, probabilities = onset_predictions([source], lambda x: candidate.run(["onset_logits"], {"short_features": x})[0])[0]
            times = (np.arange(len(probabilities)) + 1) * .016
            short_results[source["id"]] = (times, probabilities)
            np.savez_compressed(output / (source["id"] + "-short-probabilities.npz"), times=times, probabilities=probabilities)
            if source["domain"] == "synthetic":
                _, details = onset_metrics([(source, probabilities)], kaggle["threshold"])
                expected = uploaded[source["id"]]
                for key in ("tp", "fp", "fn", "predicted", "missed_ids", "extra_ids"):
                    if details[0][key] != expected[key]:
                        raise ValueError(f"Local regeneration does not reproduce Kaggle {source['id']}:{key}")
            print(f"Candidate {index + 1}/{len(sources)} {source['id']}", flush=True)
        summary["reproduced_kaggle_test_clips"] = 96
        write_json(output / "sources.json", sources)
        write_json(summary_path, summary)
        for index, source in enumerate(sources):
            started = time.monotonic()
            times, features, rms = rust_features(Path(source["wav"]), binary, output / (source["id"] + "-rust.json"), repo)
            short_t, short_prob = short_results[source["id"]]
            predictions = {"candidate": events_from_arrays(short_t, short_prob, kaggle["threshold"])}
            for name, gate in (("original", -34), ("original_ungated", -120)):
                path = output / (source["id"] + "-" + name + ".npz")
                if resume and path.exists():
                    data = np.load(path)
                    old_t, fill, probabilities = data["times"], data["fill"], data["probabilities"]
                else:
                    old_t, fill, probabilities = original_probabilities(original, times, features, rms, gate)
                    np.savez_compressed(path, times=old_t, fill=fill, probabilities=probabilities)
                predictions[name] = events_from_arrays(old_t, probabilities, .6, fill)
                if name == "original":
                    predictions["original_fill50"] = events_from_arrays(old_t, probabilities, .6, fill, fill_min=50)
            scores = compare_events(source, predictions)
            write_json(output / (source["id"] + "-scores.json"), scores)
            summary["records"].append({"id": source["id"], "domain": source["domain"], "case": source["case"],
                                       "duration": source["duration"], "scores": scores})
            # Detailed event reports live in separate files; the summary remains compact.
            summary["aggregate"] = aggregate(summary["records"])
            compact = {k: v for k, v in summary.items() if k != "records"}
            compact["completed_recordings"] = len(summary["records"])
            write_json(summary_path, compact)
            print(f"Compared {index + 1}/{len(sources)} {source['id']} ({time.monotonic() - started:.1f}s)", flush=True)
        summary["ok"] = True
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        compact = {k: v for k, v in summary.items() if k != "records"}
        compact["completed_recordings"] = len(summary["records"])
        write_json(summary_path, compact)
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    compare(Path(__file__).resolve().parent.parent, arguments.test_binary, arguments.output_dir, arguments.resume)
