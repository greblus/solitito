"""Prepare onset experiment sources, without training or importing the trainer.

Run this entire script in the EXISTING Kaggle trainer notebook. It needs only
numpy and soundfile, no project imports, downloads or authentication changes.
First run audit_onset_data.py. The full manifest is found automatically in
/kaggle/working, attached /kaggle/input outputs, or the current directory.
MANIFEST can also be set to an exact path. Multiple different manifests
require an explicit choice; a small summary cannot substitute for the manifest.
Settings below also work when the script is pasted into that notebook.

The output contains whole GuitarSet takes (solo AND comp) with physical note
times, plus synthetic hold/attack pairs with exact excitation times. No chord
segment cropping, feature cache or speculative target time shift is applied.
This prepares sources, NOT a ready-to-train feature dataset. DSP alignment and
the training adapter remain separate checks before the GPU experiment.

Validation/test performers are withheld from the NEW onset experiment only;
the existing encoder and onset head may already have seen these recordings.
Synthetic test sources are independent excitations of the SAME synthesizer,
not evidence of generalization to real guitars. Previously inspected synthetic
development examples must remain outside these splits.

CLI: python prepare_onset_data.py --manifest onset_manifest_mix.json \
         --output-dir onset_prepared_v1
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

MANIFEST = "auto"  # Or the exact path printed as "Report saved to:" by the audit.
OUTPUT_DIR = "/kaggle/working/onset_prepared_v1"
VALIDATION_PLAYER = "04"
TEST_PLAYER = "05"
SYNTHETIC_GROUPS = (60, 12, 12)  # train, validation, test; eight variants/group
SEED = 20260922
SR = 16000
SPLITS = ("train", "validation", "test")
GENERATOR_VERSION = "onset-ks-v1"


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


def find_manifest(requested, roots=None):
    """Find the actual full audit, including mounted outputs of earlier runs."""
    explicit = str(requested) != "auto"
    if explicit and Path(requested).expanduser().is_file():
        return Path(requested).expanduser().resolve()
    if roots is None:
        roots = [Path("/kaggle/working"), Path("/kaggle/input"), Path.cwd()]
    candidates = set()
    searched = []
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir() or any(root == parent or parent in root.parents for parent in searched):
            continue
        searched.append(root)
        candidates.update(p.resolve() for p in root.rglob("onset_manifest*.json") if p.is_file())
    found, rejected = {}, []
    for path in sorted(candidates):
        try:
            data = path.read_bytes()
            document = json.loads(data)
            if not isinstance(document, dict) or not isinstance(document.get("sources"), list) or not document["sources"]:
                rejected.append(f"{path}: summary or missing sources")
            elif document.get("ok") is not True or document.get("schema_version") != 1:
                rejected.append(f"{path}: unsuccessful or unsupported audit")
            else:
                # Identical copies in working/input are interchangeable; distinct
                # audits must not be selected according to filesystem order.
                found.setdefault(hashlib.sha256(data).hexdigest(), []).append(path)
        except (ValueError, OSError) as error:
            rejected.append(f"{path}: {error}")
    paths = [str(path) for copies in found.values() for path in copies]
    details = "\nSearched: " + ", ".join(map(str, searched))
    details += "\nFull manifests: " + (", ".join(paths) or "none")
    if rejected:
        details += "\nOther reports: " + "; ".join(rejected[:10])
    if explicit:
        raise FileNotFoundError(f"Requested manifest does not exist: {requested}. "
                                "Set MANIFEST to an existing full manifest path or 'auto'." + details)
    if len(found) == 1:
        return next(iter(found.values()))[0]
    if found:
        raise ValueError("Multiple different full manifests found. Set MANIFEST to the chosen exact path." + details)
    raise FileNotFoundError("No full onset manifest is accessible in this runtime. "
                            "A file shown in a saved notebook's Output is not necessarily mounted here. "
                            "Run the audit in this active session, or attach its saved output. "
                            "Use the full report, not _summary.json." + details)


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
    period = round(sr / (440 * 2 ** ((midi - 69) / 12)))
    buffer = np.random.default_rng(seed).uniform(-1, 1, period)
    buffer = np.convolve(buffer, [.5, .5], mode="same")
    wave = np.empty(frames, dtype=np.float64)
    position = 0
    for i in range(frames):
        wave[i] = buffer[position]
        following = (position + 1) % period
        buffer[position] = damping * .5 * (buffer[position] + buffer[following])
        position = following
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
            groups=SYNTHETIC_GROUPS, seed=SEED, sr=SR):
    import csv
    if len(groups) != 3 or any(n < 1 for n in groups) or seed < 0 or sr < 8000:
        raise ValueError("Need three positive group counts, nonnegative seed and sample rate >=8000")
    # Read bytes once: the recorded digest identifies the document actually used.
    manifest_bytes = manifest_path.read_bytes()
    sources = split_sources(json.loads(manifest_bytes), validation_player, test_player)
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    summary = {"schema_version": 1, "ok": False, "training_ready": False,
               "generator": GENERATOR_VERSION, "samplerate": sr,
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
                     "purpose": "onset_experiment_sources", "seed": seed, "samplerate": sr, "clips": []}
        for split, count in zip(SPLITS, groups):
            totals = {"groups": count, "clips": 0, "events": 0, "challenge_events": 0,
                      "seconds": 0., "cases": {}}
            for group in range(count):
                for clip in render_group(split, group, seed, sr):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=MANIFEST, help="auto, or an exact full audit path")
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("--validation-player", default=VALIDATION_PLAYER)
    parser.add_argument("--test-player", default=TEST_PLAYER)
    parser.add_argument("--groups", type=int, nargs=3, default=SYNTHETIC_GROUPS,
                        metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sr", type=int, default=SR)
    args = parser.parse_args(argv)
    try:
        manifest = find_manifest(args.manifest)
        print(f"Using full onset manifest: {manifest}", flush=True)
        result = prepare(manifest, args.output_dir, args.validation_player,
                         args.test_player, args.groups, args.seed, args.sr)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    print(f"Sources prepared. Small report to share: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    status = main([] if "ipykernel" in sys.modules else None)
    if status:
        raise SystemExit(status)
