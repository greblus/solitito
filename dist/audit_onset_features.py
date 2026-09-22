"""Local timing audit of actual Rust DSP versus the trainer's feature function.

Build the TEST binary with cargo test --offline --release --bin solitito --no-run.
Pass it explicitly via --test-binary. The daily application is not replaced.
Uses development-only synthetic pairs, never the new training/test corpus.
No model predictions or credit counts: response percentages below refer to
the spectral difference between two signals differing only by a known pluck.

The trainer function is extracted with AST (no trainer import/auth/downloads).
Only its return is instrumented to retain pre-normalization magnitudes too.
The Rust export calls CqtAnalyzer, then exposes its actual sparse FFT result.
Physical frame times: Rust END sample, trainer CQT index * hop. They are not
interchangeable array indices; no target shift is inferred from peak timing.
"""

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess

import librosa
import numpy as np
import soundfile as sf

from onset_contrasts import pluck
from onset_events import sha256

SR, HOP = 16000, 256


def load_trainer_extractor(path):
    tree = ast.parse(path.read_text())
    required = {"SR", "HOP_LENGTH", "CTX_FRAMES", "MIN_NOTE", "N_BINS", "BINS_PER_OCTAVE",
                "BASS_BOOST_ENABLED", "BASS_BOOST_BINS", "BASS_BOOST_GAIN"}
    namespace = {"np": np, "librosa": librosa}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in required:
                namespace[name] = ast.literal_eval(node.value)
    if not required <= namespace.keys():
        raise ValueError("Trainer feature constants changed; review the audit")
    function, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_audio_file"]
    successful = [n for n in ast.walk(function) if isinstance(n, ast.Return) and
                  isinstance(n.value, ast.Tuple) and isinstance(n.value.elts[0], ast.Name) and
                  n.value.elts[0].id == "feat"]
    if len(successful) != 1:
        raise ValueError("Trainer feature return changed; review instrumentation")
    successful[0].value.elts.append(ast.Attribute(value=ast.Name(id="cqt_abs", ctx=ast.Load()),
                                                attr="T", ctx=ast.Load()))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    if (namespace["SR"], namespace["HOP_LENGTH"], namespace["N_BINS"]) != (SR, HOP, 144):
        raise ValueError("Unsupported feature grid")

    def extract(wav, boost):
        namespace["BASS_BOOST_ENABLED"] = boost > 0
        namespace["BASS_BOOST_GAIN"] = boost
        result = namespace["process_audio_file"](str(wav))
        if len(result) != 3 or result[0] is None:
            raise ValueError(f"Trainer feature extraction failed: {wav}")
        features, count, magnitude = result
        return np.arange(count) * HOP / SR, features, magnitude
    return extract


def write_pairs(output):
    """Known sample times; shared backgrounds AND gain within each pair."""
    cases = [("isolated_low", [], [40], 1.536, 1.),
             ("isolated_mid", [], [57], 1.536, 1.),
             ("isolated_high", [], [76], 1.536, 1.),
             ("root_plus_third", [45], [49], 2.736, 1.),
             ("root_plus_fifth", [45], [52], 2.736, 1.),
             ("quiet_fifth", [45], [52], 2.736, .25),
             ("fast_repluck", [45], [45], 1.856, 1.),
             ("triad_fifth", [45, 49, 52], [52], 2.736, 1.),
             ("triad_restrum", [45, 49, 52], [45, 49, 52], 2.736, 1.)]
    pairs = []
    for index, (name, background, new, at, level) in enumerate(cases):
        end = round((at + 2.) * SR)
        hold = np.zeros(end)
        attack = np.zeros(end)
        for role, midis, start, dest in (("context", background, 1.536, hold), ("attack", new, at, attack)):
            for j, midi in enumerate(midis):
                sample = round((start + j * .016) * SR)
                excitation = 2026092200 + index * 100 + j + (10 if role == "attack" else 0)
                dest[sample:] += pluck(midi, (end - sample) / SR, SR, excitation) * (level if role == "attack" else 1.)
        struck = hold + attack
        gain = .5 / max(np.max(np.abs(hold)), np.max(np.abs(struck)))
        paths = {}
        for kind, wave in (("hold", hold), ("attack", struck)):
            path = output / f"{name}-{kind}.wav"
            sf.write(path, (wave * gain).astype(np.float32), SR, subtype="FLOAT")
            paths[kind] = path
        pairs.append({"name": name, "at": at, "new_midis": new, "level": level, "gain": gain,
                      "paths": paths})
    return pairs


def read_rust(path):
    data = json.loads(path.read_text())
    rows = data["frames"]
    ends = np.array([r["end_sample"] for r in rows])
    if (data["target_rate"], data["fft_samples"], data["hop_samples"]) != (SR, 8192, HOP):
        raise ValueError("Unexpected Rust feature grid")
    if not np.array_equal(ends, 8192 + np.arange(len(rows)) * HOP):
        raise ValueError("Incomplete or shifted Rust export")
    features = np.array([r["features"] for r in rows])
    magnitude = np.array([r["magnitude"] for r in rows])
    if features.shape != (len(rows), 168) or magnitude.shape != (len(rows), 144):
        raise ValueError("Unexpected feature dimensions")
    if not np.isfinite(features).all() or not np.isfinite(magnitude).all():
        raise ValueError("Nonfinite features")
    return ends / SR, features, magnitude, np.array([r["rms"] for r in rows])


def response(times, before, after, onset):
    change = np.linalg.norm(after - before, axis=1)
    region = (times >= onset - .4 - 1e-9) & (times <= onset + .8 + 1e-9)
    t, values = times[region] - onset, change[region]
    peak = float(values.max())
    if peak <= 0:
        raise ValueError("Pair has no spectral response")
    def first(fraction):
        return float(t[np.flatnonzero(values >= peak * fraction)[0]] * 1000)
    target = (t >= -1e-9) & (t < .096 - 1e-9)
    prior = times < onset - 1e-9
    return {"first_10pct_ms": first(.1), "first_50pct_ms": first(.5),
            "peak_ms": float(t[np.argmax(values)] * 1000),
            "peak_fraction_inside_96ms_target": float(values[target].max() / peak),
            "max_difference_before_onset": float(change[prior].max(initial=0))}


def measure(binary, output):
    repo = Path(__file__).resolve().parent.parent
    trainer = repo / "dist/model_trainer.py"
    binary = binary.resolve()
    output.mkdir(parents=True, exist_ok=False)
    extract = load_trainer_extractor(trainer)
    report = {"ok": False, "purpose": "development DSP audit, not detection accuracy",
              "binary_sha256": sha256(binary), "trainer_sha256": sha256(trainer),
              "audio_rs_sha256": sha256(repo / "src/audio.rs"),
              "dsp_sha256": sha256(repo / "dsp_weights.json"),
              "librosa_version": librosa.__version__, "numpy_version": np.__version__,
              "gate_db": -34, "cases": [],
              "limitations": ["Spectral response timing is not model/credit latency.",
                              "96ms target is the existing trainer target, not a recommended shift.",
                              "These are synthetic development pairs at 16kHz; other resampling rates are not compared.",
                              "Rust un-gated features expose DSP; gate effects are reported separately."]}
    summary_path = output / "summary.json"
    try:
        for pair in write_pairs(output):
            for boost in (0., 5.):
                online, offline = {}, {}
                for kind, wav in pair["paths"].items():
                    dump = output / f"{pair['name']}-{kind}-boost{boost:g}.json"
                    env = dict(os.environ, SOLITITO_FEATURE_WAV=str(wav.resolve()),
                               SOLITITO_FEATURE_OUTPUT=str(dump.resolve()), SOLITITO_FEATURE_BOOST=str(boost))
                    run = subprocess.run([str(binary), "audio::onset_feature_probe::export_causal_features",
                                          "--exact", "--ignored"], cwd=repo, env=env,
                                         capture_output=True, text=True, check=False)
                    if run.returncode or not dump.is_file():
                        raise RuntimeError(run.stdout + run.stderr)
                    online[kind] = read_rust(dump)
                    offline[kind] = extract(wav, boost)
                t, features, mag, rms = online["attack"]
                old_t, old_features, old_mag, old_rms = online["hold"]
                np.testing.assert_array_equal(t, old_t)
                np.testing.assert_array_equal(features[t <= pair["at"] + 1e-9],
                                              old_features[t <= pair["at"] + 1e-9])
                ot, ofeat, omag = offline["attack"]
                np.testing.assert_array_equal(ot, offline["hold"][0])
                indices = np.rint(t * SR / HOP).astype(int)
                # Same PHYSICAL time, including 512ms preceding Rust frame zero.
                np.testing.assert_allclose(ot[indices], t, atol=1e-12)
                active = (t >= pair["at"] - .4) & (t <= pair["at"] + .8)
                comparison = np.abs(features[active] - ofeat[indices[active]])
                gate = 10 ** (-34 / 20)
                entry = {"name": pair["name"], "onset": pair["at"], "new_midis": pair["new_midis"],
                         "level": pair["level"], "boost": boost,
                         "wav_sha256": {k: sha256(p) for k, p in pair["paths"].items()},
                         "rust_magnitude": response(t, old_mag, mag, pair["at"]),
                         "trainer_magnitude": response(ot, offline["hold"][2], omag, pair["at"]),
                         "rust_features": response(t, old_features, features, pair["at"]),
                         "trainer_features": response(ot, offline["hold"][1], ofeat, pair["at"]),
                         "gated_rust_features": response(t, old_features * (old_rms > gate)[:, None],
                                                         features * (rms > gate)[:, None], pair["at"]),
                         "same_time_feature_mae": {"cqt": float(comparison[:, :144].mean()),
                                                   "chroma": float(comparison[:, 144:156].mean()),
                                                   "bass": float(comparison[:, 156:].mean())}}
                report["cases"].append(entry)
                summary_path.write_text(json.dumps(report, indent=2) + "\n")
                print(f"{pair['name']} boost={boost:g}: magnitude half-response "
                      f"Rust {entry['rust_magnitude']['first_50pct_ms']:.0f}ms, "
                      f"trainer {entry['trainer_magnitude']['first_50pct_ms']:.0f}ms", flush=True)
        report["ok"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        summary_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    measure(args.test_binary, args.output_dir)
