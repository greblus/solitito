"""Solitito strike trainer: short_onset_<name>.onnx, the model that says which
pitch class was just STRUCK.

Copy this WHOLE file into a Kaggle notebook (GPU, GuitarSet attached) and run
it, or run it locally: python strike_trainer.py --help. Settings are below.
The chord model, best_model_v2_take6_onset.onnx, has its own trainer,
model_trainer.py; the two models share nothing but the app.

One run writes, under OUTPUT_ROOT/RUN_TAG (and to HF_REPO_ID unless USE_HF=False):
  short_onset_<name>.onnx                 the file the app loads (src/strike.rs);
                                          <name> is RUN_TAG without "v2_take7_"
  checkpoint_<RUN_TAG>_onset_best.pth     the chosen weights, a parent for later runs
  checkpoint_<RUN_TAG>_onset_last.pth     the last finished epoch, to resume from
  training_summary_<RUN_TAG>.json         data, chosen threshold, validation and test

The network, "Rise", is a small causal one on short spectra. It is fine-tuned
from INITIAL_ONSET on GuitarSet and synthetic plucks, including masking pairs -
a quieter upper note struck over ringing ones.

The defaults are the recipe of the released short_onset_masking_v2.onnx,
under a new RUN_TAG, so a run trains it again and writes
short_onset_masking_v2_repro.onnx. Under an existing tag the run resumes from
its snapshots, or finds itself finished, and does not train again.
MODE="export_only" rebuilds the ONNX file from a finished run's best checkpoint
and summary, without data. For weights from scratch set INITIAL_ONSET="".
Errors reaching HF are errors, never a reason to start fresh.

Earlier controlled experiments (control vs Rise, pair loss, ringing weights)
are kept in the git branch `rise`; this file holds only the path the released
model took.
"""

RUN_TAG = "v2_take7_masking_v2_repro"  # the released run is v2_take7_masking_v2; a new tag trains again
MODE = "train"  # train, export_only
HF_REPO_ID = "greblus/chord-model-snapshots"  # set to your own repo for a new model
USE_HF = True  # False: entirely local, no token or HF account needed
INPUT_DIR = "/kaggle/input"
OUTPUT_ROOT = "/kaggle/working"
INITIAL_ONSET = "hf:checkpoint_v2_take7_onset_best.pth"  # in HF_REPO_ID; a local .pt/.pth works too; empty = from scratch
ONSET_EPOCHS = 12
ONSET_MASKING_PAIRS = True  # False is the recipe of the parent run, v2_take7
ONSET_GAIN_DB = 6.0  # training level spread, +-dB; 6 is the released model's
EXPORT_ONSET_THRESHOLD = None  # export_only: normally read from the saved summary

if __name__ == "__main__":
    import importlib.util
    import subprocess
    import sys
    packages = {"onnx": "onnx", "onnxruntime": "onnxruntime", "numpy": "numpy",
                "huggingface_hub": "huggingface_hub", "soundfile": "soundfile"}
    if not USE_HF or "--no-hf" in sys.argv:
        packages.pop("huggingface_hub")
    missing = [package for module, package in packages.items()
               if importlib.util.find_spec(module) is None]
    print("Solitito strike trainer: starting", flush=True)
    if missing:
        print("Installing " + " ".join(missing), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--progress-bar", "off", *missing])
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is required; select a Kaggle GPU image or install it first")

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time

import numpy as np
import soundfile as sf


# -----------------------------------------------------------------------------
# GuitarSet: whole takes, their note starts and strings.
# -----------------------------------------------------------------------------


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, document):
    # Readers never see a partially written success report.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def observations(annotation):
    data = annotation.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        columns = [data.get(k) for k in ("time", "duration", "value")]
        if not all(isinstance(c, list) for c in columns) or len({len(c) for c in columns}) != 1:
            raise ValueError("Malformed JAMS observation columns")
        return [dict(zip(("time", "duration", "value"), row)) for row in zip(*columns)]
    raise ValueError("Unknown JAMS observation layout")


def note_events(document, take):
    events, strings, namespaces = [], [], Counter()
    for ai, annotation in enumerate(document.get("annotations", [])):
        namespace = annotation.get("namespace", "unknown")
        namespaces[namespace] += 1
        if namespace != "note_midi":
            continue
        raw_string = annotation.get("annotation_metadata", {}).get("data_source")
        if str(raw_string) not in {str(n) for n in range(6)}:
            raise ValueError(f"Unknown note string data_source: {raw_string!r}")
        string = int(raw_string)
        strings.append(string)
        for oi, observation in enumerate(observations(annotation)):
            t, duration, midi = (float(observation[k]) for k in ("time", "duration", "value"))
            if not all(math.isfinite(v) for v in (t, duration, midi)) or t < 0 or duration <= 0 or not 0 <= midi <= 127:
                raise ValueError(f"Invalid note {ai}:{oi}")
            # MIDI annotations can be fractional. Keep the original pitch;
            # make the semitone rounding used by the 12-class head explicit.
            events.append({"id": f"{take}:{ai}:{oi}", "t": t, "end": t + duration,
                           "midi_value": midi, "midi": int(round(midi)), "pc": int(round(midi)) % 12,
                           "near_semitone_boundary": abs(midi - round(midi)) >= .4,
                           "string": string, "string_source": "annotation_metadata.data_source",
                           "pluck_verified": False})
    if sorted(strings) != list(range(6)):
        raise ValueError(f"Expected one note_midi annotation per string, got {strings}")
    if not events:
        raise ValueError("No annotated notes")
    return sorted(events, key=lambda e: (e["t"], e["id"])), dict(namespaces)


def source_identity(take):
    match = re.fullmatch(r"(0[0-5])_(.+)_(solo|comp)", take)
    if not match:
        raise ValueError(f"Unknown GuitarSet take name: {take}")
    player, performance, style = match.groups()
    return {"take": take, "player": player, "style": style,
            "paired_take_group": f"{player}_{performance}", "split_group": f"player:{player}"}


def scan_inputs(root):
    audio, annotations = defaultdict(list), defaultdict(list)
    variants = {v: [] for v in ("mic", "mix", "hex", "hex_cln", "untagged")}
    extensions, archives = Counter(), []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        extension = path.suffix.lower()
        if extension in (".wav", ".flac", ".ogg", ".mp3"):
            audio[path.stem].append(path)
            extensions[extension] += 1
            tag = next((v for v in ("hex_cln", "hex", "mic", "mix")
                        if path.stem.endswith("_" + v)), "untagged")
            variants[tag].append(path)
        elif extension == ".jams":
            annotations[path.stem].append(path)
        elif extension in (".zip", ".tar", ".tgz", ".gz"):
            archives.append(str(path))
    matching = {v: sum(bool(audio.get(f"{take}_{v}")) for take in annotations)
                for v in ("mic", "mix")}
    inventory = {"audio_files": sum(extensions.values()), "audio_extensions": dict(extensions),
                 "jams_files": sum(len(paths) for paths in annotations.values()),
                 "matching_takes": matching,
                 "variants": {v: {"files": len(paths), "examples": [str(p) for p in paths[:3]]}
                              for v, paths in variants.items() if paths},
                 "archives": {"count": len(archives), "examples": archives[:3]}}
    return audio, annotations, inventory


def audit(root, variant, validation_player=None, test_player=None):
    import soundfile as sf
    if (validation_player is None) != (test_player is None) or (validation_player is not None and validation_player == test_player):
        raise ValueError("Specify two different validation/test performers, or neither")
    if variant not in ("auto", "mic", "mix"):
        raise ValueError("Audio variant must be auto, mic or mix")
    audio_files, annotations, inventory = scan_inputs(root)
    records, errors = [], []
    requested_variant = variant
    if variant == "auto":
        available = [v for v, count in inventory["matching_takes"].items() if count]
        variant = available[0] if len(available) == 1 else None
        if annotations and not available:
            errors.append({"error": "No mic/mix audio filenames match the JAMS takes. See inventory for paths, formats and archives."})
        elif len(available) > 1:
            errors.append({"error": "Both mic and mix match JAMS takes. Set AUDIO_VARIANT explicitly to mic or mix."})
    missing_audio = [take for take in sorted(annotations)
                     if variant and not audio_files.get(f"{take}_{variant}")]
    if missing_audio:
        errors.append({"error": f"Missing {variant} audio for {len(missing_audio)} takes. See inventory and missing_audio_takes in the report.",
                       "count": len(missing_audio), "examples": missing_audio[:5]})
    for take, paths in sorted(annotations.items()) if variant else []:
        try:
            if len(paths) != 1:
                raise ValueError(f"Ambiguous JAMS: {[str(p) for p in paths]}")
            identity = source_identity(take)
            audio = audio_files.get(f"{take}_{variant}", [])
            if not audio:
                continue  # Already reported together, rather than once per take.
            if len(audio) != 1:
                raise ValueError(f"Ambiguous {variant} audio: {[str(p) for p in audio]}")
            info = sf.info(audio[0])
            duration = info.frames / info.samplerate
            if info.channels != 1:
                raise ValueError(f"Expected mono {variant}, got {info.channels} channels")
            document = json.loads(paths[0].read_text())
            events, namespaces = note_events(document, take)
            if any(e["end"] > duration + .02 for e in events):
                raise ValueError("Annotations extend beyond WAV")
            split = "unassigned"
            if validation_player is not None:
                split = ("validation" if identity["player"] == validation_player else
                         "test" if identity["player"] == test_player else "train")
            nearby = []
            # These can collide in a 96 ms, 12-class target; retain both events.
            for i, event in enumerate(events):
                for j in range(i - 1, -1, -1):
                    prior = events[j]
                    if event["t"] - prior["t"] >= .096:
                        break
                    if event["pc"] == prior["pc"]:
                        nearby.append([prior["id"], event["id"]])
            records.append({**identity, "split": split, "variant": variant,
                            "audio": {"path": str(audio[0].resolve()), "sha256": sha256(audio[0]),
                                      "channels": info.channels, "samplerate": info.samplerate,
                                      "frames": info.frames, "duration": duration},
                            "jams": {"path": str(paths[0].resolve()), "sha256": sha256(paths[0]),
                                     "version": document.get("file_metadata", {}).get("jams_version")},
                            "namespaces": namespaces, "events": events,
                            "same_pc_within_96ms": nearby})
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
            errors.append({"take": take, "error": str(error)})
    if not annotations:
        errors.append({"error": "No JAMS files found"})
    if validation_player is not None:
        for split in ("train", "validation", "test"):
            if not any(r["split"] == split for r in records):
                errors.append({"error": f"Empty {split} split"})
    return {"schema_version": 1, "root": str(root.resolve()), "ok": not errors,
            "requested_variant": requested_variant, "selected_variant": variant,
            "inventory": inventory, "missing_audio_takes": missing_audio,
            "summary": {"jams_takes": len(annotations), "usable_takes": len(records),
                        "styles": dict(Counter(r["style"] for r in records)),
                        "splits": dict(Counter(r["split"] for r in records)),
                        "notes": sum(len(r["events"]) for r in records)},
            "limitations": ["Note annotations do not certify pick/pluck technique.",
                            "Frozen encoder source exposure is unknown.",
                            "Synthetic chord CSV is not per-attack ground truth; audit it separately.",
                            "Dataset version is not inferred from directory names; file hashes identify inputs."],
            "errors": errors, "sources": records}


# -----------------------------------------------------------------------------
# Synthetic plucks with exact attack times, and the run's data splits.
# -----------------------------------------------------------------------------


VALIDATION_PLAYER = "04"
TEST_PLAYER = "05"
SYNTHETIC_GROUPS = (60, 12, 12)  # train, validation, test; the masking recipe uses 96 each
SEED = 20260922
SR = 16000
SPLITS = ("train", "validation", "test")
GENERATOR_VERSION = "onset-ks-v2"


def split_sources(document, validation_player, test_player):
    """Validate the audited sources and assign whole performers before rendering."""
    players = {f"{n:02}" for n in range(6)}
    if validation_player not in players or test_player not in players or validation_player == test_player:
        raise ValueError("Choose two different GuitarSet players (00..05)")
    if document.get("ok") is not True or document.get("schema_version") != 1 or document.get("errors"):
        raise ValueError("Expected a successful version-1 full GuitarSet audit")
    if not document.get("sources"):
        raise ValueError("Use the full audit manifest, not its small _summary.json")
    counts = document["summary"]
    if (counts["usable_takes"] != len(document["sources"]) or
            counts["notes"] != sum(len(s["events"]) for s in document["sources"])):
        raise ValueError("Source/event counts disagree with the audit summary")
    if document.get("selected_variant") not in ("mic", "mix"):
        raise ValueError("Expected audited mono mic/mix sources")
    sources, ids, takes, owners = [], set(), set(), {}
    for original in sorted(document["sources"], key=lambda s: s["take"]):
        take, player = original["take"], original["player"]
        if (player not in players or not take.startswith(player + "_") or
                original["style"] not in ("solo", "comp") or
                not take.endswith("_" + original["style"]) or take in takes):
            raise ValueError(f"Invalid or duplicate source identity: {take}")
        takes.add(take)
        split = "validation" if player == validation_player else "test" if player == test_player else "train"
        if original["split"] not in ("unassigned", split):
            raise ValueError(f"Refusing to change an assigned split: {take}")
        if original["variant"] != document["selected_variant"]:
            raise ValueError(f"Mixed audio variants: {take}")
        audio = original["audio"]
        duration = audio["duration"]
        if (audio["channels"] != 1 or not math.isfinite(duration) or duration <= 0 or
                audio["samplerate"] <= 0 or audio["frames"] <= 0 or
                abs(duration - audio["frames"] / audio["samplerate"]) > 1e-9):
            raise ValueError(f"Invalid mono audio metadata: {take}")
        # Audio content, source path and paired performances may not cross splits.
        for key in ("audio:" + audio["sha256"], "path:" + audio["path"],
                    "pair:" + original["paired_take_group"]):
            if key in owners and owners[key] != split:
                raise ValueError(f"Source leakage across splits: {take} ({key})")
            owners[key] = split
        if not original["events"]:
            raise ValueError(f"Missing note annotations: {take}")
        for event in original["events"]:
            if event["id"] in ids:
                raise ValueError(f"Duplicate event ID: {event['id']}")
            ids.add(event["id"])
            if (not all(math.isfinite(event[k]) for k in ("t", "end", "midi_value")) or
                    not 0 <= event["t"] < event["end"] <= duration + .02 or
                    not 0 <= event["midi_value"] <= 127 or
                    event["midi"] != round(event["midi_value"]) or
                    event["pc"] != event["midi"] % 12 or event["string"] not in range(6) or
                    event.get("pluck_verified") is not False):
                raise ValueError(f"Invalid or unjustifiably verified GuitarSet note: {event['id']}")
        sources.append({**original, "split": split, "split_group": f"player:{player}",
                        "source_id": f"guitarset:{take}", "label_kind": "annotated_note_start",
                        "encoder_exposure": "unknown", "base_onset_head_exposure": "unknown"})
    if {s["split"] for s in sources} != set(SPLITS):
        raise ValueError("All three source splits must be nonempty")
    return sources


def verify_source_files(sources):
    for index, source in enumerate(sources):
        for kind in ("audio", "jams"):
            item = source[kind]
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"Changed since audit: {item['path']}")
        info = sf.info(source["audio"]["path"])
        if (info.channels, info.samplerate, info.frames) != tuple(
                source["audio"][k] for k in ("channels", "samplerate", "frames")):
            raise ValueError(f"Audio metadata changed: {source['take']}")
        if (index + 1) % 60 == 0:
            print(f"Verified {index + 1}/{len(sources)} GuitarSet sources", flush=True)


def excitation_seed(seed, split, group, role):
    # Separate namespaces from onset_contrasts.py, independent of generation order.
    key = f"{GENERATOR_VERSION}:{seed}:{split}:{group}:{role}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def pluck(midi, frames, sr, seed, damping, attack_seconds):
    # The two-tap averaging filter adds half a sample of delay. The old
    # integer ring instead had effective period round(sr/f)-0.5: at 16kHz,
    # requested E5 (MIDI76) became 680.85Hz, closer to F5 than E5.
    # Add fractional delay BEFORE averaging, so total delay is sr/f. Keep
    # the recurrence explicit and causal, with zero history before sample0.
    period = sr / (440 * 2 ** ((midi - 69) / 12))
    delay = math.floor(period - .5)
    fraction = period - .5 - delay
    if frames < 0 or delay < 2 or not 0 < damping <= 1 or attack_seconds <= 0:
        raise ValueError("Invalid pluck parameters")
    excitation = np.random.default_rng(seed).uniform(-1, 1, delay)
    excitation = np.convolve(excitation, [.5, .5], mode="same")
    wave = np.zeros(frames, dtype=np.float64)
    count = min(frames, delay)
    wave[:count] = excitation[:count]
    weights = (.5 * (1 - fraction), .5, .5 * fraction)
    for i in range(delay, frames):
        past = i - delay
        wave[i] = damping * (weights[0] * wave[past] +
                             (weights[1] * wave[past - 1] if past >= 1 else 0.) +
                             (weights[2] * wave[past - 2] if past >= 2 else 0.))
    wave *= np.minimum(1, np.arange(frames) / (attack_seconds * sr))
    return wave


def render_group(split, group, seed=SEED, sr=SR):
    """One split owns all stems/variants; pairs have identical background and gain."""
    if split not in SPLITS or group < 0 or seed < 0 or sr < 8000:
        raise ValueError("Invalid synthetic split/group/seed/sample rate")
    source_id = f"{GENERATOR_VERSION}-{seed}-{split}-{group:04}"
    rng = np.random.default_rng(excitation_seed(seed, split, group, "parameters"))
    root = 40 + group % 12 + int(rng.choice([0, 12]))
    third = int(rng.choice([3, 4]))
    gap = float(rng.choice([.2, .32, .48, .64, 1.2, 2.0]))
    level = float(rng.choice([.25, .5, 1., 1.5]))
    strum = float(rng.choice([0., .012, .025]))
    damping = float(rng.choice([.995, .998, .9995]))
    attack = float(rng.choice([.001, .003, .006]))
    initial = round(1.5 * sr)
    challenge = initial + round(gap * sr)
    total = challenge + round(3.5 * sr)
    specs = {}
    for i, midi in enumerate([root, root + third, root + 7]):
        specs[f"context_{i}"] = (midi, initial + round(i * strum * sr), 1., "context")
        specs[f"restrum_{i}"] = (midi, challenge + round(i * strum * sr), level, "challenge")
    for name, midi in (("root", root), ("third", root + third), ("fifth", root + 7), ("octave", root + 12)):
        specs[name] = (midi, challenge, level, "challenge")
    stems, events = {}, {}
    for name, (midi, start, amplitude, role) in specs.items():
        exc_seed = excitation_seed(seed, split, group, name)
        wave = np.zeros(total, dtype=np.float64)
        wave[start:] = pluck(midi, total - start, sr, exc_seed, damping, attack) * amplitude
        stems[name] = wave
        events[name] = {"source_id": f"{source_id}:{name}", "t": start / sr,
                        "sample": start, "midi": midi, "pc": midi % 12,
                        "role": role, "excitation_seed": exc_seed, "level": amplitude,
                        "pluck_verified": True, "label_kind": "synthetic_excitation"}
    single = ["context_0"]
    triad = [f"context_{i}" for i in range(3)]
    cases = {"root_hold": single, "root_plus_third": single + ["third"],
             "root_plus_fifth": single + ["fifth"], "root_repluck": single + ["root"],
             "root_plus_octave": single + ["octave"], "triad_hold": triad,
             "triad_fifth": triad + ["fifth"],
             "triad_repluck": triad + [f"restrum_{i}" for i in range(3)]}
    tracks = {case: sum((stems[key] for key in keys), np.zeros(total)) for case, keys in cases.items()}
    peak = float(rng.choice([.2, .4, .8]))
    gain = peak / max(float(np.max(np.abs(track))) for track in tracks.values())
    clips = []
    for case, keys in cases.items():
        name = f"{source_id}-{case}"
        clip_events = [dict(events[key], id=f"{name}:{i}", case=f"{case}/{events[key]['role']}")
                       for i, key in enumerate(keys)]
        clips.append({"name": name, "source_group": source_id, "parent_groups": [source_id],
                      "split": split, "case": case, "seed": seed, "sr": sr,
                      "root_midi": root, "third_semitones": third, "gap_seconds": gap,
                      "challenge_at": challenge / sr, "challenge_level": level,
                      "strum_seconds": strum, "damping": damping, "attack_seconds": attack,
                      "gain": gain, "frames": total, "duration": total / sr,
                      "expected_new_pcs": sorted({e["pc"] for e in clip_events if e["role"] == "challenge"}),
                      "audio": (tracks[case] * gain).astype(np.float32), "events": clip_events})
    return clips


def render_masking_group(split, group, seed=SEED, sr=SR):
    """Paired upper-string attacks at measured levels relative to ringing notes.

    Independent synthetic excitations only. These are controlled training cases,
    not recordings of a guitar or a simulation of pick damping on a real string.
    Ninety-six groups cover 24 roots x four target/background RMS ratios.
    """
    if split not in SPLITS or group < 0 or sr < 8000 or seed < 0:
        raise ValueError("Invalid masking group")
    source_id = f"onset-masking-v1-{seed}-{split}-{group:04}"
    rng = np.random.default_rng(excitation_seed(seed, split, group, "masking-v1"))
    root = 55 + (group // 4) % 24
    third = int(rng.choice([3, 4]))
    relative_db = [-18, -12, -6, 0][group % 4]
    challenge = 1.5
    total = round(3.0 * sr)
    window = round(.096 * sr)
    start = round(challenge * sr)
    specs = [("root", root, .3), ("third", root + third, .65),
             ("fifth", root + 7, challenge)]
    stems, events, timbres = {}, {}, {}
    for key, midi, at in specs:
        exc_seed = excitation_seed(seed, split, group, "masking-v1-" + key)
        offset = round(at * sr)
        damping = float(rng.choice([.999, .9997, .9999]))
        attack = float(rng.choice([.001, .003, .006]))
        tone = pluck(midi, total - offset, sr, exc_seed, damping, attack)
        # A causal comb changes harmonic balance without changing the pitch.
        # Independent positions avoid giving every string the same spectrum.
        pick_position = float(rng.choice([.12, .23, .38]))
        delay = max(1, round(sr / (440 * 2 ** ((midi - 69) / 12)) * pick_position))
        shaped = tone.copy()
        shaped[delay:] -= .8 * tone[:-delay]
        tone = shaped / max(float(np.sqrt(np.mean(shaped[:window] ** 2))), 1e-12)
        wave = np.zeros(total)
        wave[offset:] = tone
        stems[key] = wave
        timbres[key] = dict(damping=damping, attack_seconds=attack, pick_position=pick_position)
        events[key] = dict(t=offset / sr, sample=offset, midi=midi, pc=midi % 12,
                           role="challenge" if key == "fifth" else "context",
                           source_id=f"{source_id}:{key}", excitation_seed=exc_seed,
                           pluck_verified=True, label_kind="synthetic_excitation")
    background = stems["root"] + stems["third"]
    background_rms = float(np.sqrt(np.mean(background[start:start + window] ** 2)))
    stems["fifth"] *= background_rms * 10 ** (relative_db / 20)
    target_rms = float(np.sqrt(np.mean(stems["fifth"][start:start + window] ** 2)))
    tracks = {"hold": background, "alone": stems["fifth"],
              "add_fifth": background + stems["fifth"]}
    gain = float(rng.choice([.15, .3, .6])) / max(np.max(np.abs(x)) for x in tracks.values())
    clips = []
    for variant, track in tracks.items():
        case = f"masking_{variant}_{relative_db:+d}db"
        name = f"{source_id}-{variant}"
        keys = {"hold": ["root", "third"], "alone": ["fifth"],
                "add_fifth": ["root", "third", "fifth"]}[variant]
        clip_events = [dict(events[key], id=f"{name}:{i}", case=f"{case}/{events[key]['role']}")
                       for i, key in enumerate(keys)]
        clips.append(dict(name=name, source_group=source_id, parent_groups=[source_id],
                          split=split, case=case, seed=seed, sr=sr, root_midi=root,
                          challenge_at=challenge, target_midi=root + 7,
                          target_background_db=relative_db, background_rms=background_rms * gain,
                          target_rms=target_rms * gain, timbres=timbres,
                          gain=gain, frames=total, duration=total / sr,
                          expected_new_pcs=[(root + 7) % 12] if variant != "hold" else [],
                          audio=(track * gain).astype(np.float32), events=clip_events))
    return clips


def source_summary(sources):
    result = {}
    for split in SPLITS:
        selected = [s for s in sources if s["split"] == split]
        result[split] = {"sources": len(selected),
                         "styles": dict(Counter(s["style"] for s in selected)),
                         "players": sorted({s["player"] for s in selected}),
                         "notes": sum(len(s["events"]) for s in selected),
                         "same_pc_pairs_within_96ms": sum(len(s["same_pc_within_96ms"]) for s in selected),
                         "seconds": sum(s["audio"]["duration"] for s in selected)}
    return result


def prepare(manifest_path, output, validation_player=VALIDATION_PLAYER, test_player=TEST_PLAYER,
            groups=SYNTHETIC_GROUPS, seed=SEED, sr=SR, masking_pairs=False):
    import csv
    if len(groups) != 3 or any(n < 1 for n in groups) or seed < 0 or sr < 8000:
        raise ValueError("Need three positive group counts, nonnegative seed and sample rate >=8000")
    # Read bytes once: the recorded digest identifies the document actually used.
    manifest_bytes = manifest_path.read_bytes()
    sources = split_sources(json.loads(manifest_bytes), validation_player, test_player)
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    summary = {"schema_version": 1, "ok": False, "training_ready": False,
               "generator": GENERATOR_VERSION, "masking_pairs": masking_pairs, "samplerate": sr,
               "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "input_manifest": str(manifest_path.resolve()), "seed": seed,
               "validation_player": validation_player, "test_player": test_player,
               "guitarset": source_summary(sources), "synthetic": {},
               "limitations": ["GuitarSet labels note starts, not verified picking technique.",
                               "Existing encoder and base onset head exposure to GuitarSet is unknown.",
                               "Synthetic test uses independent excitations of the same generator.",
                               "Synthetic re-plucks add a new excitation to the unchanged old tail; they do not model string damping by the pick.",
                               "Chord-only synthetic WAV/CSV datasets cannot supply exact pluck labels.",
                               "Feature extraction/alignment and the training adapter are not part of this preparation."]}
    write_json(summary_path, summary)
    try:
        verify_source_files(sources)
        guitarset_path = output / "guitarset.json"
        write_json(guitarset_path, {"schema_version": 1, "sources": sources})
        synthetic_dir = output / "synthetic"
        synthetic_dir.mkdir()
        synthetic = {"schema_version": 1, "generator": GENERATOR_VERSION,
                     "masking_generator": "onset-masking-v1" if masking_pairs else None,
                     "purpose": "onset_experiment_sources", "seed": seed, "samplerate": sr, "clips": []}
        for split, count in zip(SPLITS, groups):
            totals = {"groups": count, "clips": 0, "events": 0, "challenge_events": 0,
                      "seconds": 0., "cases": {}}
            for group in range(count):
                clips = render_group(split, group, seed, sr)
                if masking_pairs:
                    clips += render_masking_group(split, group, seed, sr)
                for clip in clips:
                    audio = clip.pop("audio")
                    wav = synthetic_dir / (clip["name"] + ".wav")
                    reference = synthetic_dir / (clip["name"] + ".csv")
                    sf.write(wav, audio, sr, subtype="FLOAT")
                    with reference.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(clip["events"][0]))
                        writer.writeheader()
                        writer.writerows(clip["events"])
                    clip.update(wav=wav.name, reference=reference.name,
                                wav_sha256=sha256(wav), reference_sha256=sha256(reference))
                    synthetic["clips"].append(clip)
                    totals["clips"] += 1
                    totals["events"] += len(clip["events"])
                    totals["challenge_events"] += sum(e["role"] == "challenge" for e in clip["events"])
                    totals["seconds"] += clip["duration"]
                    totals["cases"][clip["case"]] = totals["cases"].get(clip["case"], 0) + 1
                if (group + 1) % 12 == 0 or group + 1 == count:
                    print(f"Rendered {split}: {group + 1}/{count} source groups", flush=True)
            summary["synthetic"][split] = totals
            write_json(summary_path, summary)
        synthetic_path = synthetic_dir / "manifest.json"
        write_json(synthetic_path, synthetic)
        write_json(output / "dataset.json", {
            "schema_version": 1, "training_ready": False,
            "guitarset": {"path": "guitarset.json", "sha256": sha256(guitarset_path)},
            "synthetic": {"path": "synthetic/manifest.json", "sha256": sha256(synthetic_path)},
            "split_policy": "whole GuitarSet performers; whole synthetic excitation groups",
            "window_policy": "whole takes, including attacks at chord boundaries; no chord-segment crop",
            "target_time_axis": "physical audio seconds; no feature indices or label shift assigned"})
        summary["ok"] = True
        write_json(summary_path, summary)
    except Exception as error:
        summary["error"] = str(error)
        write_json(summary_path, summary)
        raise
    return summary


def run_pipeline(root, output_root, variant="auto", groups=SYNTHETIC_GROUPS,
                 validation_player=VALIDATION_PLAYER, test_player=TEST_PLAYER,
                 seed=SEED, sr=SR, masking_pairs=False):
    """Audit the attached GuitarSet, then render the synthetic sources beside it."""
    import tempfile
    if not root.is_dir():
        raise FileNotFoundError(f"GuitarSet input directory is unavailable: {root}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="onset-prepared-", dir=output_root)).resolve()
    print(f"Output directory: {run_dir}", flush=True)
    print("Stage 1/2: auditing the attached GuitarSet", flush=True)
    audited = audit(root, variant)
    manifest = run_dir / f"onset_manifest_{audited['selected_variant'] or 'auto'}.json"
    write_json(manifest, audited)
    if not audited["ok"]:
        failed = {"ok": False, "stage": "audit", "inventory": audited["inventory"],
                  "error_count": len(audited["errors"]), "errors": audited["errors"][:10],
                  "full_report": str(manifest)}
        write_json(run_dir / "summary.json", failed)
        raise ValueError(f"Audit failed. Small report: {run_dir / 'summary.json'}. "
                         f"First errors: {audited['errors'][:3]}")
    print(f"Audited {len(audited['sources'])} sources. Manifest: {manifest}", flush=True)
    print("Stage 2/2: preparing source splits and synthetic pairs", flush=True)
    result = prepare(manifest, run_dir / "prepared", validation_player, test_player,
                     groups, seed, sr, masking_pairs=masking_pairs)
    result["prepared_directory"] = str(run_dir / "prepared")
    result["summary_path"] = str(run_dir / "summary.json")
    write_json(run_dir / "summary.json", result)
    return result


# -----------------------------------------------------------------------------
# The strike detector's input: short spectra, one frame per 16 ms hop. The app
# computes the same frames (ShortFeatures in src/strike.rs); a shared fixture
# holds both sides to the same numbers.
# -----------------------------------------------------------------------------

TRAIN_SEED = 20260923
FEATURE_SPEC = {
    "version": "short-stft-v1", "samplerate": 16000, "hop": 256,
    "windows": [1024, 2048], "max_frequency": 4000,
    "window": "symmetric Hann", "amplitude": "2*abs(rfft)/sum(window)",
    "compression": "log1p(1000*amplitude)/log(1001); no file normalization",
    "resampling": "causal linear, one source-sample delay unless already 16kHz",
    "frame_time": "exclusive audio window end; first frame=0.016 seconds",
    "startup": "zero left audio padding; no right padding or future samples",
    "storage": "float16, converted back to float32 for both training and inference",
}
ONSET_HOP = 256
ONSET_SR = 16000
FEATURE_DIM = 770  # 0..4kHz for the 1024 and 2048 point transforms.
HISTORY = 30  # Four kernel-3 convolutions, dilations 1,2,4,8.
RISE_PAST_FRAMES = 4
HISTORY_FRAMES = HISTORY + RISE_PAST_FRAMES  # the app keeps these plus the current frame
BLOCK_FRAMES = 128
TARGET_SECONDS = .096
EVAL_EARLY = .032
EVAL_LATE = .128
THRESHOLDS = (.3, .4, .5, .6, .7, .8, .9)


def onset_resample(audio, source_sr):
    """Same time origin at every length; the interpolator never needs the future."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or source_sr < 8000:
        raise ValueError("Expected finite, nonempty mono audio with sample rate >=8000")
    if source_sr == ONSET_SR:
        return audio
    positions = np.arange(math.floor(len(audio) * ONSET_SR / source_sr)) * (source_sr / ONSET_SR) - 1
    lower = np.floor(positions).astype(np.int64)
    fraction = positions - lower
    # Zero history before the source starts. No centred antialiasing filter.
    a = np.where(lower >= 0, audio[np.clip(lower, 0, len(audio) - 1)], 0)
    b = np.where(lower + 1 >= 0, audio[np.clip(lower + 1, 0, len(audio) - 1)], 0)
    return (a + fraction * (b - a)).astype(np.float32)


def onset_features(audio):
    """Frame n sees only samples strictly before (n+1)*hop, even at startup."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono audio")
    frames = len(audio) // ONSET_HOP
    result = np.empty((frames, FEATURE_DIM), dtype=np.float16)
    offset = 0
    for size in FEATURE_SPEC["windows"]:
        window = np.hanning(size)
        bins = size * FEATURE_SPEC["max_frequency"] // ONSET_SR + 1
        padded = np.pad(audio, (size - ONSET_HOP, 0))
        if frames:
            windows = np.lib.stride_tricks.sliding_window_view(padded, size)[::ONSET_HOP][:frames]
            for start in range(0, frames, 1024):
                spectra = np.abs(np.fft.rfft(windows[start:start + 1024] * window, axis=1))[:, :bins]
                amplitude = spectra * (2 / window.sum())
                result[start:start + len(spectra), offset:offset + bins] = (
                    np.log1p(1000 * amplitude) / math.log(1001)).astype(np.float16)
        offset += bins
    return result


def onset_targets(events, frames):
    times = (np.arange(frames) + 1) * ONSET_HOP / ONSET_SR
    labels = np.zeros((frames, 12), dtype=np.float32)
    for event in events:
        # A sample at exactly the exclusive frame end is not visible yet.
        active = (times > event["t"] + 1e-9) & (times <= event["t"] + TARGET_SECONDS + 1e-9)
        labels[active, event["pc"]] = 1
    return labels


def onset_sources(prepared):
    """Use only this run's verified manifests; preserve raw events and groups."""
    dataset = json.loads((prepared / "dataset.json").read_text())
    documents = {}
    for name in ("guitarset", "synthetic"):
        path = prepared / dataset[name]["path"]
        if sha256(path) != dataset[name]["sha256"]:
            raise ValueError(f"Changed prepared manifest: {path}")
        documents[name] = json.loads(path.read_text())
    if documents["synthetic"]["generator"] != GENERATOR_VERSION:
        raise ValueError("Regenerate the tuned v2 synthetic sources")
    sources = []
    for source in documents["guitarset"]["sources"]:
        sources.append({"id": source["take"], "domain": "guitarset", "case": source["style"],
                        "split": source["split"], "wav": source["audio"]["path"],
                        "sha256": source["audio"]["sha256"], "duration": source["audio"]["duration"],
                        "events": source["events"], "group": source["split_group"]})
    for clip in documents["synthetic"]["clips"]:
        sources.append({"id": clip["name"], "domain": "synthetic", "case": clip["case"],
                        "split": clip["split"], "wav": str(prepared / "synthetic" / clip["wav"]),
                        "sha256": clip["wav_sha256"], "duration": clip["duration"],
                        "events": clip["events"], "group": clip["source_group"]})
    return sources


def cache_onset_features(sources, directory):
    directory.mkdir()
    cached = []
    for index, source in enumerate(sources):
        wav = Path(source["wav"])
        if sha256(wav) != source["sha256"]:
            raise ValueError(f"Audio changed after preparation: {wav}")
        audio, sr = sf.read(wav, dtype="float32")
        features = onset_features(onset_resample(audio, sr))
        if not len(features):
            raise ValueError(f"Recording shorter than one frame: {wav}")
        path = directory / f"{index:04d}.npy"
        np.save(path, features)
        cached.append(dict(source, features=str(path), feature_sha256=sha256(path),
                           frames=len(features), incomplete_final_hop_seconds=(
                               source["duration"] - len(features) * ONSET_HOP / ONSET_SR)))
        if (index + 1) % 30 == 0 or index + 1 == len(sources):
            print(f"Features: {index + 1}/{len(sources)} complete recordings", flush=True)
    write_json(directory / "index.json", {"feature_spec": FEATURE_SPEC, "sources": cached})
    return cached


def feature_block(features, start, length=BLOCK_FRAMES, history_frames=HISTORY_FRAMES):
    """Fixed left context; a training/inference boundary never resets the audio."""
    count = min(length, len(features) - start)
    x = np.zeros((history_frames + length, FEATURE_DIM), dtype=np.float32)
    low = max(0, start - history_frames)
    destination = history_frames + low - start
    x[destination:history_frames + count] = features[low:start + count]
    return x.T.copy(), count


class OnsetBlocks:
    """Every training frame once per epoch, in blocks with their left context."""
    def __init__(self, sources, gain_db=ONSET_GAIN_DB):
        if not sources or any(s["split"] != "train" for s in sources):
            raise ValueError("Training blocks must contain train sources only")
        self.features = [np.load(s["features"], mmap_mode="r") for s in sources]
        self.labels = [onset_targets(s["events"], s["frames"]) for s in sources]
        self.gain_db = gain_db
        self.blocks = [(i, start) for i, f in enumerate(self.features)
                       for start in range(0, len(f), BLOCK_FRAMES)]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        recording, start = self.blocks[index]
        x, count = feature_block(self.features[recording], start)
        # Exact gain transform of the fixed log spectrum, including left context.
        gain = 10 ** np.random.uniform(-self.gain_db / 20, self.gain_db / 20)
        x = np.log1p(np.expm1(x * math.log(1001)) * gain) / math.log(1001)
        y = np.zeros((12, BLOCK_FRAMES), dtype=np.float32)
        y[:, :count] = self.labels[recording][start:start + count].T
        mask = np.zeros(BLOCK_FRAMES, dtype=np.float32)
        mask[:count] = 1
        return x.astype(np.float32), y, mask


RISE_SPEC = {
    "version": "positive-spectral-rise-v1",
    "past_frames": RISE_PAST_FRAMES,
    "input": "existing 770 float32 log-magnitude features, after ordinary gain augmentation",
    "formula": "a=expm1(x*log(1001)); d=relu(a-mean(previous 4 a)); log1p(d)/log(1001)",
    "current_frame_excluded_from_baseline": True,
    "startup": "zero feature history",
    "future_frames": 0,
    "projection": "extra 770->96 linear projection, no bias, initialized to zero; added before first ReLU",
    "loss": "ordinary BCE, positive weight 4",
    "raw_feature_cache": "unchanged; rise is computed inside the exported ONNX graph",
}


def positive_spectral_rise(features):
    """How much each bin grew over the mean of the four frames before it."""
    import torch
    # Exp/Sub and Add/Log also work in the app's ONNX opset 17 exporter.
    amplitude = torch.exp(features * math.log(1001)) - 1
    # Exclude the current frame: a left pad of four produces T+1 averages,
    # and the final average belongs to the following (unavailable) frame.
    previous = torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(amplitude, (RISE_PAST_FRAMES, 0)),
        kernel_size=RISE_PAST_FRAMES, stride=1)[:, :, :-1]
    return torch.log(1 + torch.relu(amplitude - previous)) / math.log(1001)


def make_onset_model():
    """770 short-spectrum features in, 12 pitch-class strike logits out, causal."""
    import torch
    from torch import nn

    class CausalOnset(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Conv1d(FEATURE_DIM, 96, 1)
            self.temporal = nn.ModuleList([nn.Conv1d(96, 96, 3, dilation=d) for d in (1, 2, 4, 8)])
            self.output = nn.Conv1d(96, 12, 1)
            nn.init.constant_(self.output.bias, -3.)
            # Built last and outside the seeded stream, as when Rise was added
            # to the control network: the same seed gives the same backbone.
            with torch.random.fork_rng(devices=[]):
                self.rise_project = nn.Conv1d(FEATURE_DIM, 96, 1, bias=False)
                nn.init.zeros_(self.rise_project.weight)

        def forward(self, features):
            x = self.project(features) + self.rise_project(positive_spectral_rise(features))
            x = torch.relu(x)
            for dilation, layer in zip((1, 2, 4, 8), self.temporal):
                x = torch.relu(x + layer(nn.functional.pad(x, (2 * dilation, 0))))
            return self.output(x)

    return CausalOnset()


# -----------------------------------------------------------------------------
# Scoring: predicted strikes against annotated note starts, one to one, per
# pitch class. Used to pick the epoch and the threshold, and for the test split.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    id: str
    t: float
    pc: int
    midi: int | None = None
    case: str = ""


def match_events(reference, predicted, early, late, pitch="pc"):
    """Maximize matches, then minimize total absolute timing error.

    Per pitch, an ordered dynamic program suffices: with the same tolerance
    window for every event, uncrossing two feasible matches preserves
    feasibility and cannot increase absolute timing error. Keep indices so
    even simultaneous same-class events cannot consume a prediction twice.
    """
    pairs = []
    for key in sorted({getattr(e, pitch) for e in reference + predicted}):
        refs = sorted((i for i, e in enumerate(reference) if getattr(e, pitch) == key),
                      key=lambda i: reference[i].t)
        preds = sorted((i for i, e in enumerate(predicted) if getattr(e, pitch) == key),
                       key=lambda i: predicted[i].t)
        n, m = len(refs), len(preds)
        scores = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
        action = [[None] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                best, move = scores[i - 1][j], "ref"
                if scores[i][j - 1] > best:
                    best, move = scores[i][j - 1], "pred"
                delta = predicted[preds[j - 1]].t - reference[refs[i - 1]].t
                if -early - 1e-9 <= delta <= late + 1e-9:
                    count, cost = scores[i - 1][j - 1]
                    candidate = (count + 1, cost - abs(delta))
                    if candidate > best:
                        best, move = candidate, "match"
                scores[i][j], action[i][j] = best, move
        i, j = n, m
        while i and j:
            move = action[i][j]
            if move == "match":
                pairs.append((refs[i - 1], preds[j - 1]))
                i, j = i - 1, j - 1
            elif move == "ref":
                i -= 1
            else:
                j -= 1
    return sorted(pairs)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def latch_events(rows, threshold=.6, fill_min=50, stride=1, phase=0):
    """Isolated set_onsets latch. No app reset/credit semantics inferred."""
    if not 0 < threshold <= 1 or not 0 <= fill_min <= 100 or stride < 1 or not 0 <= phase < stride:
        raise ValueError("Invalid latch settings")
    armed, peaks, events = [True] * 12, [0.0] * 12, []
    for frame, (t, fill, values) in enumerate(rows):
        if frame % stride != phase or fill < fill_min:
            continue
        for pc, value in enumerate(values):
            if value < max(.3 * peaks[pc], .1):
                armed[pc] = True
            elif armed[pc] and value >= threshold:
                armed[pc], peaks[pc] = False, value
                events.append(Event(f"{frame}:{pc}", t, pc))
    return events


def local_event_matches(reference, predicted, early=EVAL_EARLY, late=EVAL_LATE):
    """Exact scorer on disjoint tolerance components, avoiding full-take DP tables."""
    pairs = []
    for pc in range(12):
        refs = sorted((e for e in reference if e.pc == pc), key=lambda e: e.t)
        preds = sorted((e for e in predicted if e.pc == pc), key=lambda e: e.t)
        pred_times = [e.t for e in preds]
        start = 0
        while start < len(refs):
            end = start + 1
            while end < len(refs) and refs[end].t - refs[end - 1].t <= early + late + 2e-9:
                end += 1
            selected = preds[bisect_left(pred_times, refs[start].t - early - 1e-9):
                             bisect_right(pred_times, refs[end - 1].t + late + 1e-9)]
            group = refs[start:end]
            pairs.extend((group[i], selected[j]) for i, j in match_events(group, selected, early, late))
            start = end
    return pairs


def onset_predictions(sources, infer, batch_size=16):
    """Full files, each frame once, with the same left context as training."""
    result = []
    for source in sources:
        features = np.load(source["features"], mmap_mode="r")
        probabilities = []
        starts = list(range(0, len(features), BLOCK_FRAMES))
        for offset in range(0, len(starts), batch_size):
            blocks = [feature_block(features, start) for start in starts[offset:offset + batch_size]]
            logits = infer(np.stack([x for x, _ in blocks]))
            if logits.shape != (len(blocks), 12, HISTORY_FRAMES + BLOCK_FRAMES) or not np.isfinite(logits).all():
                raise ValueError("Invalid model output")
            for row, (_, count) in zip(logits, blocks):
                probabilities.append((1 / (1 + np.exp(-np.clip(row[:, HISTORY_FRAMES:HISTORY_FRAMES + count], -80, 80)))).T)
        values = np.concatenate(probabilities)
        if len(values) != source["frames"]:
            raise ValueError("Refusing partial-file evaluation")
        result.append((source, values))
    return result


def onset_metrics(predictions, threshold):
    buckets = {}
    details = []
    for source, values in predictions:
        rows = (((i + 1) * ONSET_HOP / ONSET_SR, 100, row) for i, row in enumerate(values)
                if (i + 1) * ONSET_HOP / ONSET_SR < source["duration"])
        predicted = latch_events(rows, threshold=threshold, fill_min=0)
        refs = [Event(e["id"], e["t"], e["pc"], e.get("midi"), e.get("case", source["case"]))
                for e in source["events"] if 0 <= e["t"] < source["duration"]]
        pairs = local_event_matches(refs, predicted)
        found_r, found_p = {r.id for r, _ in pairs}, {p.id for _, p in pairs}
        extra = [p for p in predicted if p.id not in found_p]
        repeated = set()
        last = {}
        collisions = 0
        overlaps = 0
        target_frames = set()
        for ref in sorted(refs, key=lambda e: e.t):
            if ref.pc in last and 0 < ref.t - last[ref.pc] <= 2:
                repeated.add(ref.id)
            if ref.pc in last and ref.t - last[ref.pc] < TARGET_SECONDS:
                overlaps += 1
            last[ref.pc] = ref.t
            key = (math.floor(ref.t * ONSET_SR / ONSET_HOP + 1e-9), ref.pc)
            collisions += key in target_frames
            target_frames.add(key)
        challenges = {e["id"] for e in source["events"] if e.get("role") == "challenge"}
        deltas = [p.t - r.t for r, p in pairs]
        # Additional predictions >400ms after ALL annotated starts: conservative decay/silence diagnostic.
        tail_extra = sum(p.t > max((r.t for r in refs), default=0) + .4 for p in extra)
        counts = {"tp": len(pairs), "fp": len(extra), "fn": len(refs) - len(pairs),
                  "seconds": source["duration"], "recordings": 1, "tail_extra": tail_extra,
                  "repeated_reference": len(repeated), "repeated_tp": len(repeated & found_r),
                  "challenge_reference": len(challenges), "challenge_tp": len(challenges & found_r),
                  "same_pc_frame_collisions": collisions,
                  "same_pc_target_overlaps": overlaps,
                  "boundary_reference": sum(r.t < EVAL_EARLY or r.t + EVAL_LATE >= source["duration"] for r in refs)}
        for key in ("all", source["domain"], f"{source['domain']}/{source['case']}"):
            bucket = buckets.setdefault(key, dict.fromkeys(counts, 0) | {"deltas": []})
            for name, count in counts.items():
                bucket[name] += count
            bucket["deltas"].extend(deltas)
        details.append({"source": source["id"], **counts,
                        "predicted": [{"id": p.id, "t": p.t, "pc": p.pc} for p in predicted],
                        "missed_ids": [r.id for r in refs if r.id not in found_r],
                        "extra_ids": [p.id for p in extra]})
    for bucket in buckets.values():
        tp, fp, fn = (bucket[name] for name in ("tp", "fp", "fn"))
        bucket.update(precision=tp / (tp + fp) if tp + fp else 0.,
                      challenge_recall=(bucket["challenge_tp"] / bucket["challenge_reference"]
                                        if bucket["challenge_reference"] else None),
                      recall=tp / (tp + fn) if tp + fn else 0.,
                      f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                      false_events_per_minute=60 * fp / bucket["seconds"],
                      latency_p50=percentile(bucket["deltas"], .5),
                      latency_p95=percentile(bucket.pop("deltas"), .95))
    macro = float(np.mean([buckets[d]["f1"] for d in ("guitarset", "synthetic") if d in buckets]))
    return {"threshold": threshold, "macro_domain_f1": macro, "groups": buckets}, details


def validation_choice(table):
    # F1 first, fewer false events second, higher threshold as the final tie breaker.
    return max(table, key=lambda r: (r["macro_domain_f1"], -r["groups"]["all"]["fp"], r["threshold"]))


# -----------------------------------------------------------------------------
# Training the strike detector.
# -----------------------------------------------------------------------------


def validate_resume_sources(previous, sources, report_path):
    """Keep exact model inputs/labels; regenerated synthetic WAV headers may vary."""
    fields = ("id", "split", "sha256", "feature_sha256", "events")
    differences, container_changes = [], []
    for index, (saved, source) in enumerate(zip(previous, sources)):
        if len(saved) != len(fields):
            differences.append(dict(index=index, id=source["id"], fields=["snapshot_schema"]))
            continue
        changed = [key for key, value in zip(fields, saved) if value != source[key]]
        if changed == ["sha256"] and source["domain"] == "synthetic":
            # FLOAT WAVs contain a PEAK creation timestamp. A byte hash of
            # regenerated audio can change while the actual training tensors
            # remain bit-identical. Verify the array file too, not just its index.
            path = Path(source["features"])
            if path.is_file() and sha256(path) == source["feature_sha256"]:
                container_changes.append(source["id"])
                continue
            changed.append("feature_file")
        if changed:
            differences.append(dict(index=index, id=source["id"], fields=changed))
    ok = len(previous) == len(sources) and not differences
    report = dict(ok=ok, saved_sources=len(previous), current_sources=len(sources),
                  changed_sources=len(differences), examples=differences[:10],
                  synthetic_container_changes=len(container_changes),
                  synthetic_container_examples=container_changes[:10],
                  policy="Exact ordered IDs, splits, feature hashes and events; synthetic WAV hash may differ only with verified identical feature files")
    write_json(report_path, report)
    if not ok:
        details = json.dumps(dict(saved_sources=len(previous), current_sources=len(sources),
                                  examples=differences[:3]))
        raise ValueError(f"Cannot resume onset training with changed data: {details}. "
                         f"See {report_path}. Checkpoint was not changed.")
    if container_changes:
        print(f"Resume: {len(container_changes)} synthetic WAV hashes differ; "
              "feature files and labels match the checkpoint exactly.", flush=True)
    return report


def train_onset(sources, output, epochs=12, batch_size=16, seed=TRAIN_SEED, device_name="auto",
                gain_db=ONSET_GAIN_DB, feature_directory=None, initial_checkpoint=None,
                resume=False, checkpoint_callback=None):
    """Train, keep the best validation epoch, then export and score it (finish_onset)."""
    import torch
    if epochs < 1 or batch_size < 1:
        raise ValueError("Positive epochs and batch size required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    splits = {split: [s for s in sources if s["split"] == split] for split in ("train", "validation", "test")}
    if any(not split for split in splits.values()):
        raise ValueError("All three splits must be nonempty")
    feature_directory = feature_directory or output / "features"
    blocks = OnsetBlocks(splits["train"], gain_db)
    loader = torch.utils.data.DataLoader(blocks, batch_size=batch_size, shuffle=True, num_workers=0,
                                         generator=torch.Generator().manual_seed(seed))
    model = make_onset_model().to(device)
    if initial_checkpoint is not None:
        saved = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        prior = saved["contract"]
        if (prior["feature_spec"] != FEATURE_SPEC or
                prior["history_frames"] != HISTORY_FRAMES or
                not prior.get("spectral_rise", False)):
            raise ValueError("Initial onset checkpoint has a different model/DSP contract")
        model.load_state_dict(saved["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    checkpoint = output / "short_onset_best.pt"
    contract = {"feature_spec": FEATURE_SPEC, "features": FEATURE_DIM, "history_frames": HISTORY_FRAMES,
                "block_frames": BLOCK_FRAMES, "target_seconds": TARGET_SECONDS,
                "network_startup": f"{HISTORY_FRAMES} zero input frames before the first audio feature",
                "spectral_rise": True, "rise_spec": RISE_SPEC,
                "evaluation": {"early": EVAL_EARLY, "late": EVAL_LATE, "thresholds": list(THRESHOLDS),
                               "reference_policy": "all raw note starts; no same-PC deduplication",
                               "latch": "existing app peak hysteresis, fill gate disabled; no judge simulation"},
                "architecture": "770->96 raw projection + rise projection; four causal residual conv3 dilations1/2/4/8 ->12 logits",
                "seed": seed, "epochs": epochs, "batch_size": batch_size, "device": str(device),
                "initial_checkpoint_sha256": sha256(initial_checkpoint) if initial_checkpoint else None,
                "positive_weight": 4, "training_gain_db": [-gain_db, gain_db],
                "initial_weights_sha256": hashlib.sha256(b"".join(
                    p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "versions": {n: str(importlib.import_module(n).__version__) for n in
                             ("torch", "numpy", "soundfile", "onnx", "onnxruntime")},
                "data_index_sha256": sha256(feature_directory / "index.json")}
    last_path = output / "short_onset_last.pt"
    restored = None
    if resume and last_path.exists():
        restored = torch.load(last_path, map_location="cpu", weights_only=False)
        previous = restored["contract"]
        # Cache paths and package versions may differ after moving a Kaggle output.
        keys = ("feature_spec", "history_frames", "spectral_rise", "epochs",
                "batch_size", "seed", "training_gain_db")
        if any(previous[k] != contract[k] for k in keys):
            raise ValueError("Cannot resume onset training with changed configuration")
        validate_resume_sources(restored["sources"], sources, output / "resume_data_check.json")
        contract = previous
        model.load_state_dict(restored["state_dict"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        torch.save(restored["best_checkpoint"], checkpoint)
    write_json(output / "contract.json", contract)
    print(f"Training {contract['parameters']} parameters on {device}; {len(blocks)} blocks/epoch", flush=True)

    def infer(x):
        with torch.inference_mode():
            return model(torch.from_numpy(x).to(device)).cpu().numpy()

    history, best = [], None
    start_epoch = 1
    if restored is not None:
        history, best = restored["history"], restored["best"]
        start_epoch = restored["epoch"] + 1
        random.setstate(restored["python_rng"])
        np.random.set_state(restored["numpy_rng"])
        torch.set_rng_state(restored["torch_rng"])
        loader.generator.set_state(restored["loader_rng"])
        if device.type == "cuda" and restored["cuda_rng"]:
            torch.cuda.set_rng_state_all(restored["cuda_rng"])
        write_json(output / "history.json", history)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_total, examples = 0., 0
        started = time.monotonic()
        # Evidence of the exact batches: order and gain augmentation included.
        input_digest = hashlib.sha256()
        for batch_index, (x, y, mask) in enumerate(loader):
            input_digest.update(x.numpy().tobytes())
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[:, :, HISTORY_FRAMES:]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, pos_weight=torch.full((12, 1), 4., device=device), reduction="none")
            loss = (loss * mask[:, None, :]).sum() / (12 * mask.sum())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_total += float(loss.detach()) * float(mask.sum())
            examples += float(mask.sum())
            if (batch_index + 1) % max(1, len(loader) // 4) == 0:
                print(f"Epoch {epoch}: batch {batch_index + 1}/{len(loader)}", flush=True)
        model.eval()
        predictions = onset_predictions(splits["validation"], infer, batch_size)
        # Epoch selection fixed at .5. Only the best checkpoint gets the final threshold sweep.
        metrics, _ = onset_metrics(predictions, .5)
        choice = (metrics["macro_domain_f1"], -metrics["groups"]["all"]["fp"])
        if best is None or choice > best:
            best = choice
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "contract": contract}, checkpoint)
        history.append({"epoch": epoch, "loss": loss_total / examples,
                        "input_batches_sha256": input_digest.hexdigest(),
                        "seconds": time.monotonic() - started, "validation": metrics})
        write_json(output / "history.json", history)
        if resume:
            # A single atomic file includes BEST too: remote recovery never mixes epochs.
            state = dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                         epoch=epoch, contract=contract, history=history, best=best,
                         best_checkpoint=torch.load(checkpoint, map_location="cpu", weights_only=True),
                         sources=[(v["id"], v["split"], v["sha256"], v["feature_sha256"], v["events"])
                                  for v in sources],
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(), loader_rng=loader.generator.get_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
            temporary = last_path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(last_path)
            if checkpoint_callback:
                checkpoint_callback(last_path)
        total = metrics["groups"]["all"]
        print(f"Epoch {epoch}/{epochs}: loss={history[-1]['loss']:.5f}, "
              f"validation macro F1={metrics['macro_domain_f1']:.3f}, "
              f"FP/min={total['false_events_per_minute']:.2f}, {history[-1]['seconds']:.1f}s", flush=True)

    return finish_onset(sources, output, device_name)


def export_probability_check(reference, exported, tolerance=2e-5):
    """Keep the numerical guard strict; discrete events can differ at a boundary."""
    if not reference or len(reference) != len(exported):
        raise ValueError("Export comparison requires the same complete recordings")
    maximum, worst_source, frames = 0., None, 0
    for (source, a), (other, b) in zip(reference, exported):
        if (source["id"] != other["id"] or a.shape != b.shape or a.ndim != 2
                or a.shape != (source["frames"], 12) or not len(a)):
            raise ValueError("Export comparison source/frame mismatch")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Non-finite export comparison")
        error = float(np.max(np.abs(a - b)))
        if error > maximum:
            maximum, worst_source = error, source["id"]
        frames += len(a)
    return {"ok": maximum <= tolerance, "max_probability_error": maximum,
            "tolerance": tolerance, "worst_source": worst_source,
            "recordings": len(reference), "frames": frames}


def export_event_comparison(reference, exported):
    """Report actual event changes without rounding probabilities or forgiving notes."""
    if len(reference) != len(exported) or any(a["source"] != b["source"] for a, b in zip(reference, exported)):
        raise ValueError("Export event comparison source mismatch")
    changed, only_reference, only_exported, examples = 0, 0, 0, []
    for a, b in zip(reference, exported):
        before = {(p["t"], p["pc"]) for p in a["predicted"]}
        after = {(p["t"], p["pc"]) for p in b["predicted"]}
        if before == after:
            continue
        changed += 1
        only_reference += len(before - after)
        only_exported += len(after - before)
        if len(examples) < 20:
            examples.append({"source": a["source"],
                             "pytorch_only": sorted(before - after)[:20],
                             "onnx_only": sorted(after - before)[:20]})
    return {"events_identical": changed == 0, "changed_recordings": changed,
            "pytorch_only_events": only_reference, "onnx_only_events": only_exported,
            "examples": examples}


def finish_onset(sources, output, device_name="auto"):
    """Export the selected epoch, choose the threshold on ONNX output, score the test split.

    Runs without an optimizer step, so a finished run can be finished again.
    """
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    reference_device = str(device)
    checkpoint = output / "short_onset_best.pt"
    contract = json.loads((output / "contract.json").read_text())
    history = json.loads((output / "history.json").read_text())
    if [row["epoch"] for row in history] != list(range(1, contract["epochs"] + 1)):
        raise ValueError("Training is incomplete; refusing to silently restart it")
    batch_size = contract["batch_size"]
    splits = {split: [s for s in sources if s["split"] == split] for split in ("validation", "test")}
    model = make_onset_model().to(device)

    def infer(x):
        with torch.inference_mode():
            return model(torch.from_numpy(x).to(device)).cpu().numpy()

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved["contract"] != contract:
        raise ValueError("Checkpoint and contract disagree")
    best_row = max(history, key=lambda row: (row["validation"]["macro_domain_f1"],
                                            -row["validation"]["groups"]["all"]["fp"]))
    if saved["epoch"] != best_row["epoch"]:
        raise ValueError("Checkpoint is not the validation-selected epoch")
    model.load_state_dict(saved["state_dict"])
    model.eval()
    predictions = onset_predictions(splits["validation"], infer, batch_size)
    onnx_path = output / "short_onset_rise.onnx"
    export_rise_onnx(model, onnx_path)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])

    def infer_onnx(x):
        return session.run(["onset_logits"], {"short_features": x})[0]

    # Check every validation frame, not just one dummy tensor.
    exported = onset_predictions(splits["validation"], infer_onnx, batch_size)
    parity = export_probability_check(predictions, exported)
    write_json(output / "onnx_validation_comparison.json", parity)
    if not parity["ok"]:
        raise ValueError(f"ONNX differs from checkpoint beyond tolerance: {parity['max_probability_error']}; "
                         "see onnx_validation_comparison.json. Training checkpoint is preserved.")
    max_error = parity["max_probability_error"]
    # The threshold is chosen on what the app will run: the ONNX output.
    table = [onset_metrics(exported, threshold)[0] for threshold in THRESHOLDS]
    selected = validation_choice(table)
    write_json(output / "validation_thresholds.json", {"checkpoint_epoch": saved["epoch"], "table": table,
                                                       "selected_threshold": selected["threshold"]})
    exported_metrics, exported_details = onset_metrics(exported, selected["threshold"])
    reference_metrics, reference_details = onset_metrics(predictions, selected["threshold"])
    parity.update(export_event_comparison(reference_details, exported_details))
    parity.update(threshold=selected["threshold"], reference_device=reference_device,
                  threshold_selection_backend="ONNX Runtime CPU",
                  pytorch_metrics=reference_metrics, onnx_metrics=exported_metrics)
    write_json(output / "onnx_validation_comparison.json", parity)
    if not parity["events_identical"]:
        print(f"Export numerical boundary differences: {parity['changed_recordings']} recordings, "
              f"max probability error {max_error:.3g}; final scores/threshold use ONNX Runtime.", flush=True)
    write_json(output / "validation_events.json", exported_details)
    test_predictions = onset_predictions(splits["test"], infer_onnx, batch_size)
    tested, details = onset_metrics(test_predictions, selected["threshold"])
    write_json(output / "test_events.json", details)
    probability_directory = output / "probabilities"
    probability_directory.mkdir(exist_ok=True)
    for split, prediction_set in (("validation", exported), ("test", test_predictions)):
        for source, probabilities in prediction_set:
            np.save(probability_directory / (Path(source["features"]).stem + f"-{split}.npy"), probabilities)
    return {"ok": True, "training_complete": True,
            "checkpoint_epoch": saved["epoch"], "checkpoint_sha256": sha256(checkpoint),
            "model": str(onnx_path), "model_sha256": sha256(onnx_path),
            "onnx_max_probability_error": max_error, "threshold": selected["threshold"],
            "onnx_validation_comparison": parity, "evaluation_backend": "ONNX Runtime CPU",
            "initial_weights_sha256": contract["initial_weights_sha256"],
            "spectral_rise": True, "history_frames": contract["history_frames"],
            "training_gain_db": contract.get("training_gain_db", [-6, 6]),
            "input_batches_sha256": [row["input_batches_sha256"] for row in history],
            # The whole threshold curve, so same-threshold comparisons need no other file.
            "validation_thresholds": [{"threshold": row["threshold"], "macro_domain_f1": row["macro_domain_f1"],
                                       "groups": row["groups"]} for row in table],
            "validation": selected, "test": tested,
            "limitations": ["GuitarSet note starts do not verify picking technique; raw same-PC overlaps remain in recall.",
                            "A 12-class 96ms target cannot separate all closely spaced same-PC strings; overlap counts are reported.",
                            "Synthetic test uses the same simplified generator, not real guitar picking/noise.",
                            "Causal linear resampling has no antialiasing filter.",
                            "Audio-time latency excludes CPU scheduling and the app's judge.",
                            "All test frames were evaluated once at the validation-selected threshold."]}


# -----------------------------------------------------------------------------
# One run: snapshots, the strike detector, the file the app loads.
# -----------------------------------------------------------------------------


class SnapshotStore:
    """Local snapshots first; only a confirmed absent file means 'start fresh'."""
    def __init__(self, directory, repo=None, token=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.repo, self.token = repo, token
        self.api = None
        self.files = set()
        if repo:
            from huggingface_hub import HfApi
            self.api = HfApi(token=token)
            # Authentication/network/repository errors propagate. Never silently restart.
            self.files = set(self.api.list_repo_files(repo_id=repo, repo_type="model"))

    def fetch(self, name):
        target = self.directory / name
        if target.is_file():
            return target
        if not self.api or name not in self.files:
            return None
        from huggingface_hub import hf_hub_download
        print(f"Downloading {name} from {self.repo}", flush=True)
        cached = hf_hub_download(repo_id=self.repo, filename=name, token=self.token)
        shutil.copy2(cached, target)
        return target

    def publish(self, path, name):
        path = Path(path)
        target = self.directory / name
        if path.resolve() != target.resolve():
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(path, temporary)
            temporary.replace(target)
        if self.api:
            # Failure leaves the local snapshot intact and stops with a useful error.
            self.api.upload_file(path_or_fileobj=str(target), path_in_repo=name,
                                 repo_id=self.repo, repo_type="model")
            self.files.add(name)
        return target


def resolve_initial_onset(value, store):
    """Explicit parent weights, never a fallback to random initialization."""
    if not value:
        return None
    if value.startswith("hf:"):
        name = value[3:]
        if not name or Path(name).name != name or Path(name).suffix not in (".pt", ".pth"):
            raise ValueError("INITIAL_ONSET hf: reference must name a .pt/.pth checkpoint in HF_REPO_ID")
        path = store.fetch(name)
        if path is None:
            raise FileNotFoundError(f"Initial Rise checkpoint not found: {value}. "
                                    "Restore it or explicitly set INITIAL_ONSET empty to train from scratch.")
    else:
        path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Initial Rise checkpoint does not exist: {path}")
    return path


def prepare_features(root, work, groups, masking_pairs=False):
    index = work / "features" / "index.json"
    if index.exists():
        document = json.loads(index.read_text())
        if document["feature_spec"] != FEATURE_SPEC:
            raise ValueError("Cached onset features use a different DSP contract")
        sources = document["sources"]
        has_masking = any(s["case"].startswith("masking_") for s in sources)
        if has_masking != masking_pairs:
            raise ValueError("Onset masking recipe changed; use a new RUN_TAG, not this cache")
        for source in sources:
            path = work / "features" / Path(source["features"]).name
            if not path.is_file() or sha256(path) != source["feature_sha256"]:
                raise ValueError(f"Missing or corrupt onset cache: {path}")
            source["features"] = str(path)
        return sources
    prepared = run_pipeline(root, work, "auto", groups=groups, seed=20260923,
                            masking_pairs=masking_pairs)
    return cache_onset_features(onset_sources(Path(prepared["prepared_directory"])), work / "features")


def export_rise_onnx(model, path):
    """The trained network as ONNX: short_features [batch,770,time] -> onset_logits [batch,12,time]."""
    import torch
    example = torch.zeros(1, FEATURE_DIM, HISTORY_FRAMES + BLOCK_FRAMES)
    torch.onnx.export(model.cpu().eval(), example, str(path),
                      input_names=["short_features"], output_names=["onset_logits"],
                      dynamic_axes={"short_features": {0: "batch", 2: "time"},
                                    "onset_logits": {0: "batch", 2: "time"}},
                      opset_version=17, dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(path)))


def strike_model_name(tag):
    """The app's name for a run's strike model: v2_take7_masking_v2 -> short_onset_masking_v2.onnx."""
    return f"short_onset_{tag.removeprefix('v2_take7_')}.onnx"


def write_strike_model(onset_onnx, output, threshold, tag):
    """The file the app loads: the exported network plus what the app reads from it.

    src/strike.rs takes its threshold from `onset_threshold`; the rest says
    what the file expects, so a mismatched one can be told apart.
    """
    import onnx
    from onnx import helper
    import onnxruntime as ort
    if not 0 < float(threshold) < 1:
        raise ValueError("A validated onset threshold between zero and one is required")
    onset_onnx, output = Path(onset_onnx), Path(output)
    model = onnx.load(str(onset_onnx))
    if ([v.name for v in model.graph.input] != ["short_features"] or
            [v.name for v in model.graph.output] != ["onset_logits"]):
        raise ValueError("Expected the Rise network: short_features -> onset_logits")
    metadata = dict(onset_threshold=str(float(threshold)), onset_history_frames=str(HISTORY_FRAMES),
                    onset_feature_spec=json.dumps(FEATURE_SPEC, sort_keys=True), run_tag=tag)
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.onnx")
    onnx.save(model, str(temporary))
    # Metadata only: the answers must not move by a bit.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    sessions = [ort.InferenceSession(str(p), sess_options=options, providers=["CPUExecutionProvider"])
                for p in (onset_onnx, temporary)]
    short = np.random.default_rng(20261010).uniform(0, .5, (1, FEATURE_DIM, 65)).astype(np.float32)
    before, after = (s.run(["onset_logits"], {"short_features": short})[0] for s in sessions)
    if not np.array_equal(before, after):
        temporary.unlink()
        raise ValueError("Adding metadata changed the strike model's answers")
    temporary.replace(output)
    return dict(path=str(output), sha256=sha256(output), metadata=metadata)


def export_only(config, store):
    """Rebuild a finished run's strike model from its best checkpoint, without data."""
    tag = config["run_tag"]
    work = Path(config["work_dir"])
    report_path = store.fetch(f"training_summary_{tag}.json")
    report = json.loads(report_path.read_text()) if report_path else {}
    checkpoint = store.fetch(f"checkpoint_{tag}_onset_best.pth")
    if checkpoint is None:
        raise FileNotFoundError(f"export_only requires checkpoint_{tag}_onset_best.pth. No training was started.")
    threshold = config.get("export_onset_threshold")
    if threshold is None:
        threshold = report.get("onset", {}).get("threshold")
    if threshold is None:
        raise ValueError("Missing onset threshold: restore training_summary_<RUN_TAG>.json "
                         "or set EXPORT_ONSET_THRESHOLD to its selected threshold. No training was started.")
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = saved["contract"]
    if (contract["feature_spec"] != FEATURE_SPEC or contract["history_frames"] != HISTORY_FRAMES or
            not contract.get("spectral_rise", False)):
        raise ValueError("The checkpoint has a different model/DSP contract")
    model = make_onset_model()
    model.load_state_dict(saved["state_dict"], strict=True)
    onset_dir = work / "rise"
    onset_dir.mkdir(parents=True, exist_ok=True)
    export_rise_onnx(model, onset_dir / "short_onset_rise.onnx")
    strike = write_strike_model(onset_dir / "short_onset_rise.onnx", work / strike_model_name(tag), threshold, tag)
    store.publish(strike["path"], Path(strike["path"]).name)
    report.update(ok=True, run_tag=tag, strike_model=strike, export_only=True, training_performed=False,
                  checkpoint_sha256=sha256(checkpoint))
    destination = work / f"training_summary_{tag}.json"
    report["summary_path"] = str(destination)
    write_json(destination, report)
    store.publish(destination, destination.name)
    print(f"Export complete, no training: {strike['path']}", flush=True)
    return report


def run(config, store):
    work = Path(config["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    tag = config["run_tag"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Choose a simple RUN_TAG: letters, digits, _ and -")
    if config["mode"] == "export_only":
        return export_only(config, store)
    if config["mode"] != "train":
        raise ValueError("Mode must be train or export_only")
    masking_pairs = config.get("onset_masking_pairs", False)
    groups = config.get("groups", (96, 96, 96) if masking_pairs else (60, 12, 12))
    gain_db = config.get("onset_gain_db", ONSET_GAIN_DB)
    configuration = dict(run_tag=tag, mode=config["mode"],
                         onset_masking_pairs=masking_pairs, synthetic_groups=list(groups),
                         initial_onset=config.get("initial_onset") or "",
                         onset_epochs=config.get("onset_epochs", 12), onset_gain_db=gain_db)
    print("Configuration: " + json.dumps(configuration, sort_keys=True), flush=True)
    write_json(work / "run_configuration.json", configuration)
    last_name = f"checkpoint_{tag}_onset_last.pth"
    previous = store.fetch(last_name)
    # Resolve the parent before preparing data, and only for a new run.
    # A resumed run already carries both the best weights and their provenance.
    local_last = work / "rise" / "short_onset_last.pt"
    resuming = previous is not None or local_last.is_file()
    initial = None if resuming else resolve_initial_onset(config.get("initial_onset"), store)
    report_path = work / f"training_summary_{tag}.json"
    write_json(report_path, dict(ok=False, stage="data", run_tag=tag))
    sources = prepare_features(Path(config["input_dir"]), work, groups, masking_pairs)
    for split in ("train", "validation", "test"):
        selected = [s for s in sources if s["split"] == split]
        masking_count = sum(s["case"].startswith("masking_") for s in selected)
        print(f"Onset data {split}: {len(selected)} recordings, {masking_count} masking clips", flush=True)
        if masking_pairs and not masking_count:
            raise ValueError(f"Masking enabled but no masking clips in {split}; check the feature cache")
    write_json(report_path, dict(ok=False, stage="training", run_tag=tag))
    onset_dir = work / "rise"
    onset_dir.mkdir(exist_ok=True)
    if previous:
        shutil.copy2(previous, onset_dir / "short_onset_last.pt")
    print("Onsets: " + ("resuming this run" if resuming else f"initial Rise weights from {initial}" if initial else
                        "training Rise from scratch"), flush=True)
    result = train_onset(sources, onset_dir,
                         epochs=config.get("onset_epochs", 12),
                         batch_size=config.get("onset_batch_size", 16),
                         device_name=config.get("device", "auto"), gain_db=gain_db,
                         feature_directory=work / "features",
                         initial_checkpoint=initial, resume=True,
                         checkpoint_callback=lambda path: store.publish(path, last_name))
    strike = write_strike_model(result["model"], work / strike_model_name(tag), result["threshold"], tag)
    store.publish(strike["path"], Path(strike["path"]).name)
    store.publish(onset_dir / "short_onset_best.pt", f"checkpoint_{tag}_onset_best.pth")
    summary = dict(schema_version=4, ok=True, training_complete=True, run_tag=tag, mode=config["mode"],
                   configuration=configuration, strike_model=strike, onset=result,
                   summary_path=str(report_path))
    write_json(report_path, summary)
    store.publish(report_path, report_path.name)
    print(f"Done. Strike model for the app: {strike['path']}\nReport: {report_path}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("train", "export_only"), default=MODE)
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--hf-repo", default=HF_REPO_ID)
    parser.add_argument("--no-hf", action="store_true", default=not USE_HF)
    parser.add_argument("--initial-onset", default=INITIAL_ONSET)
    parser.add_argument("--onset-epochs", type=int, default=ONSET_EPOCHS)
    parser.add_argument("--onset-masking-pairs", action=argparse.BooleanOptionalAction, default=ONSET_MASKING_PAIRS)
    parser.add_argument("--onset-gain-db", type=float, default=ONSET_GAIN_DB)
    parser.add_argument("--export-onset-threshold", type=float, default=EXPORT_ONSET_THRESHOLD)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args([] if argv is None and "ipykernel" in sys.modules else argv)
    config = vars(args)
    config["work_dir"] = str(Path(args.output_root) / args.run_tag)
    if config["mode"] == "export_only":
        config["device"] = "cpu"
    elif config["device"] == "auto":
        print("Importing torch", flush=True)
        import torch
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {config['device']}", flush=True)
    token = os.environ.get("HF_TOKEN")
    if not args.no_hf and not token:
        print("Reading the HF_TOKEN Kaggle secret", flush=True)
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as error:
            raise RuntimeError("Set the Kaggle HF_TOKEN secret or use --no-hf / USE_HF=False") from error
    if not args.no_hf:
        print(f"Listing {args.hf_repo} on Hugging Face", flush=True)
    store = SnapshotStore(config["work_dir"], None if args.no_hf else args.hf_repo, token)
    try:
        return run(config, store)
    except Exception as error:
        write_json(Path(config["work_dir"]) / "training_failure.json",
                   dict(ok=False, training_complete=False, error=str(error), work_dir=config["work_dir"]))
        raise


if __name__ == "__main__":
    main()
