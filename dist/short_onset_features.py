"""Development probe: can short causal spectra identify NEW pitch classes?

Uses the saved WAV pairs and actual Rust CQT exports from audit_onset_features.
No neural training, ONNX replacement, or application judge is involved.

Candidate: Hann STFT of the last 64/128 ms, a fixed harmonic dictionary,
nonnegative least-squares pitch amplitudes, then positive changes over 96 ms.
The dictionary is analytical, not fitted to these examples. Each frame uses
only its own waveform and earlier frames, NEVER the counterfactual hold file.
The hold file is a separate negative control, used only for reporting.

Ranking at a known attack time with a known number of new classes is an
optimistic diagnostic, NOT end-to-end onset accuracy or a production rule.
Scores are neither calibrated probabilities nor comparable model thresholds.
Do not select a production threshold from these few development recordings.
"""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import nnls
import soundfile as sf

from audit_onset_features import read_rust
from onset_events import sha256

SR = 16000
HOP = 256
LOOKBACK = 6
MIDIS = np.arange(40, 89)
DELAYS_MS = (32, 64, 96, 128, 192, 256, 384)


@lru_cache(maxsize=2)
def harmonic_dictionary(window_samples):
    """Fixed 1/h partial amplitudes, normalized columns; no audio-derived tuning."""
    if window_samples not in (1024, 2048):
        raise ValueError("Expected a 64ms or 128ms window")
    window = np.hanning(window_samples)
    t = np.arange(window_samples) / SR
    columns = []
    for midi in MIDIS:
        fundamental = 440 * 2 ** ((int(midi) - 69) / 12)
        column = np.zeros(window_samples // 2 + 1)
        for harmonic in range(1, 9):
            frequency = fundamental * harmonic
            if frequency >= SR / 2:
                break
            column += np.abs(np.fft.rfft(window * np.sin(2 * np.pi * frequency * t))) / harmonic
        column /= np.linalg.norm(column)
        columns.append(column)
    return np.column_stack(columns)


def positive_pitch_rise(amplitudes, pcs, lookback=LOOKBACK):
    """Scale-invariant change per class; no global or future normalization."""
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if amplitudes.ndim != 2 or len(pcs) != amplitudes.shape[1] or lookback < 1:
        raise ValueError("Invalid pitch feature dimensions/lookback")
    if not np.isfinite(amplitudes).all() or np.any(amplitudes < 0):
        raise ValueError("Expected finite nonnegative amplitudes")
    if any(not 0 <= int(pc) < 12 for pc in pcs):
        raise ValueError("Invalid pitch class")
    previous = np.zeros_like(amplitudes)
    previous[lookback:] = amplitudes[:-lookback]
    change = np.maximum(amplitudes - previous, 0)
    denominator = amplitudes.sum(axis=1) + previous.sum(axis=1)
    result = np.zeros((len(amplitudes), 12))
    for column, pc in enumerate(pcs):
        result[:, int(pc)] += change[:, column]
    np.divide(result, denominator[:, None], out=result, where=denominator[:, None] > 1e-12)
    result[denominator <= 1e-12] = 0
    return result


def short_features(audio, window_samples):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono samples at 16kHz")
    dictionary = harmonic_dictionary(window_samples)
    window = np.hanning(window_samples)
    ends = np.arange(window_samples, len(audio) + 1, HOP)
    amplitudes = np.empty((len(ends), len(MIDIS)))
    for index, end in enumerate(ends):
        magnitude = np.abs(np.fft.rfft(audio[end - window_samples:end] * window))
        amplitudes[index], _ = nnls(dictionary, magnitude)
    return ends / SR, amplitudes, positive_pitch_rise(amplitudes, MIDIS % 12)


def read_short(path, window_samples):
    audio, rate = sf.read(path, dtype="float32")
    if rate != SR or audio.ndim != 1:
        raise ValueError("This controlled experiment requires mono 16kHz inputs")
    return short_features(audio, window_samples)


def rank_frame(row, target_pcs):
    targets = sorted(set(target_pcs))
    if not targets or any(pc not in range(12) for pc in targets):
        raise ValueError("Expected target pitch classes")
    other = [pc for pc in range(12) if pc not in targets]
    total = float(np.sum(row))
    minimum = float(min(row[pc] for pc in targets))
    maximum_other = float(max((row[pc] for pc in other), default=0))
    # A tie or all-zero row provides no pitch discrimination.
    correct = minimum > maximum_other + 1e-12
    return {"correct_top_k": correct,
            "top_k": [int(pc) for pc in np.argsort(-row, kind="stable")[:len(targets)]] if total > 1e-12 else [],
            "target_share": float(np.sum(row[targets]) / total) if total > 1e-12 else None,
            "weakest_target_score": minimum, "strongest_other_score": maximum_other,
            "total_rise": total}


def compare(times, hold, attack, at, targets):
    np.testing.assert_array_equal(hold[times <= at + 1e-9], attack[times <= at + 1e-9])
    points = {}
    for delay in DELAYS_MS:
        index = np.searchsorted(times, at + delay / 1000 - 1e-9)
        if index >= len(times):
            raise ValueError("Audio ends before requested comparison time")
        points[str(delay)] = {**rank_frame(attack[index], targets),
                              "actual_delay_ms": float((times[index] - at) * 1000)}
    # Exclude initial context attacks from negative controls. The previous
    # baseline files supply identical input up to the added physical excitation.
    region = (times >= at) & (times <= at + .8 + 1e-9)
    later = times >= at + 1.
    return {"at_delays": points,
            "hold_max_score_0_800ms": float(hold[region].max()),
            "hold_max_score_after_1s": float(hold[later].max()),
            "attack_max_score_after_1s": float(attack[later].max()),
            "new_attack_peak_score_0_800ms": float(attack[region][:, targets].max())}


def measure(dataset, output):
    dataset = dataset.resolve()
    manifest_path = dataset / "summary.json"
    previous = json.loads(manifest_path.read_text())
    if previous.get("ok") is not True:
        raise ValueError("Need a completed actual-DSP feature audit")
    cases = [c for c in previous["cases"] if c["boost"] == 0]
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Expected unique baseline cases without bass boost")
    output.mkdir(parents=True, exist_ok=False)
    report = {"ok": False, "dataset_manifest_sha256": sha256(manifest_path),
              "script_sha256": sha256(Path(__file__)), "cases": [],
              "numpy_version": np.__version__, "scipy_version": scipy.__version__,
              "protocol": {"windows_samples": [1024, 2048], "sr": SR, "hop": HOP,
                           "lookback_frames": LOOKBACK, "midi_range": [40, 88],
                           "harmonics": 8, "partial_amplitude": "1/h", "delays_ms": DELAYS_MS},
              "limitations": ["Known attack times and number of new classes: ranking is not onset detection accuracy.",
                              "The baseline is the RAW CQT-rise input, not the whole trained onset head.",
                              "Hold audio is scored independently, never subtracted from attacked audio.",
                              "Fixed harmonic templates may not represent a real guitar or pick scrapes.",
                              "Development corpus only; no thresholds or model parameters fitted.",
                              "Fast-repluck hold control includes the tail of the first attack 320ms earlier.",
                              "Magnitude/feature differences alone are not calibrated probabilities."]}
    started = time.monotonic()
    try:
        for case in cases:
            paths = {kind: dataset / f"{case['name']}-{kind}.wav" for kind in ("hold", "attack")}
            for kind, path in paths.items():
                if sha256(path) != case["wav_sha256"][kind]:
                    raise ValueError(f"WAV changed: {path}")
            target_pcs = sorted({midi % 12 for midi in case["new_midis"]})
            for method, window in (("rust_cqt_rise", None), ("short_64ms", 1024), ("short_128ms", 2048)):
                scores = {}
                for kind, path in paths.items():
                    if window is None:
                        dump = dataset / f"{case['name']}-{kind}-boost0.json"
                        times, features, _, _ = read_rust(dump)
                        scores[kind] = positive_pitch_rise(features[:, :144], (np.arange(144) // 2) % 12)
                    else:
                        times, _, scores[kind] = read_short(path, window)
                    if kind == "hold":
                        hold_times = times
                    else:
                        np.testing.assert_array_equal(times, hold_times)
                result = compare(times, scores["hold"], scores["attack"], case["onset"], target_pcs)
                np.savez_compressed(output / f"{case['name']}-{method}.npz", times=times, **scores)
                report["cases"].append({"name": case["name"], "method": method,
                                        "onset": case["onset"], "target_pcs": target_pcs, **result})
            print(f"Compared {case['name']}", flush=True)
        report["ranking_counts"] = {
            method: {str(delay): sum(c["at_delays"][str(delay)]["correct_top_k"] for c in report["cases"] if c["method"] == method)
                     for delay in DELAYS_MS}
            for method in ("rust_cqt_rise", "short_64ms", "short_128ms")}
        report["case_count_per_method"] = len(cases)
        report["elapsed_seconds"] = time.monotonic() - started
        report["ok"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def stress(dataset, output):
    """Candidate-only robustness check; the Rust baseline is not extrapolated."""
    source = json.loads((dataset / "summary.json").read_text())
    if source.get("ok") is not True:
        raise ValueError("Need a completed source feature audit")
    cases = [c for c in source["cases"] if c["boost"] == 0]
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Expected unique baseline cases without bass boost")
    output.mkdir(parents=True, exist_ok=False)
    report = {"ok": False, "dataset_manifest_sha256": sha256(dataset / "summary.json"),
              "script_sha256": sha256(Path(__file__)), "cases": [],
              "numpy_version": np.__version__, "scipy_version": scipy.__version__,
              "purpose": "Candidate features only: diagnostic gain/noise stress, not real-guitar validation",
              "conditions": [{"gain": .25, "noise_rms": 0.}, {"gain": 1., "noise_rms": .001},
                             {"gain": .25, "noise_rms": .001}], "noise_seed": 20260923,
              "limitations": ["Known attack times and class count; not end-to-end detection.",
                              "Shared noise within pairs; no hold subtraction in feature computation.",
                              "Noise is synthetic white noise, not pick scrapes or guitar handling."]}
    try:
        for case_index, case in enumerate(cases):
            waves = {}
            for kind in ("hold", "attack"):
                path = dataset / f"{case['name']}-{kind}.wav"
                if sha256(path) != case["wav_sha256"][kind]:
                    raise ValueError(f"WAV changed: {path}")
                waves[kind], sr = sf.read(path, dtype="float32")
                if sr != SR or waves[kind].ndim != 1:
                    raise ValueError("Expected mono 16kHz WAV")
            if waves["hold"].shape != waves["attack"].shape:
                raise ValueError("Paired audio lengths differ")
            noise = np.random.default_rng(report["noise_seed"] + case_index).normal(size=len(waves["hold"]))
            for condition in report["conditions"]:
                for window in (1024, 2048):
                    scores = {}
                    for kind, wave in waves.items():
                        times, _, scores[kind] = short_features(
                            wave * condition["gain"] + noise * condition["noise_rms"], window)
                    targets = sorted({m % 12 for m in case["new_midis"]})
                    result = compare(times, scores["hold"], scores["attack"], case["onset"], targets)
                    name = f"{case['name']}-{window}-gain{condition['gain']}-noise{condition['noise_rms']}"
                    np.savez_compressed(output / f"{name}.npz", times=times, **scores)
                    report["cases"].append({"name": case["name"], "window_samples": window,
                                            "target_pcs": targets, **condition, **result})
            print(f"Stress checked {case['name']}", flush=True)
        report["ok"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stress", action="store_true", help="candidate-only gain/noise check")
    args = parser.parse_args()
    result = (stress if args.stress else measure)(args.dataset, args.output_dir)
    print(json.dumps(result.get("ranking_counts", {"ok": result["ok"], "cases": len(result["cases"])}), indent=2))
