"""Read-only GuitarSet onset manifest audit; run on Kaggle before training.

Paste this entire file into one Kaggle cell and run it on CPU. No other
project files are needed. Edit INPUT_DIR, AUDIO_VARIANT and OUTPUT_DIR below.
The report is saved as /kaggle/working/onset_manifest_<variant>.json.
A small onset_manifest_<variant>_summary.json is saved alongside it for
sharing results without sending every note. Run this script in the existing
trainer notebook with its attached datasets; no extra notebook is needed.

Alternatively, save the file and run it from a terminal:
  python audit_onset_data.py /kaggle/input --variant mic \
      --output /kaggle/working/onset_manifest.json

Uses soundfile (already used by the trainer), never imports model_trainer.
Auto selects mic or mix only when exactly one of them matches the JAMS takes.
If both are present, choose explicitly. Hex is never substituted for mono.
No filename normalization that merges takes.
Optional --validation-player 04 --test-player 05 assigns whole performers;
without both flags all sources stay unassigned. This says nothing about which
sources the existing frozen encoder has already seen.

String schema follows mirdata's GuitarSet load_notes implementation:
https://mirdata.readthedocs.io/en/stable/_modules/mirdata/datasets/guitarset.html
annotation_metadata.data_source is 0=low E ... 5=high e. Unknown schemas fail
the audit rather than guessing strings from annotation order or MIDI.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

# Settings for pasting directly into a Kaggle cell.
INPUT_DIR = "/kaggle/input"
AUDIO_VARIANT = "auto"  # Or explicitly choose "mic" / "mix" if both are present.
OUTPUT_DIR = "/kaggle/working"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, nargs="?", default=Path(INPUT_DIR))
    parser.add_argument("--variant", choices=("auto", "mic", "mix"), default=AUDIO_VARIANT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-player", choices=[f"{n:02}" for n in range(6)])
    parser.add_argument("--test-player", choices=[f"{n:02}" for n in range(6)])
    args = parser.parse_args(argv)
    result = audit(args.root, args.variant, args.validation_player, args.test_player)
    output = args.output or Path(OUTPUT_DIR) / f"onset_manifest_{result['selected_variant'] or 'auto'}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    preview = {k: result[k] for k in ("ok", "requested_variant", "selected_variant", "inventory", "summary")}
    preview.update(error_count=len(result["errors"]), errors=result["errors"][:10])
    summary_output = output.with_name(f"{output.stem}_summary.json")
    summary_output.write_text(json.dumps(preview, indent=2) + "\n")
    print(json.dumps(preview, indent=2))
    if len(result["errors"]) > 10:
        print(f"Showing the first 10 errors; all {len(result['errors'])} are in the report.")
    print(f"Report saved to: {output}")
    print(f"Small summary to share: {summary_output}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    # A pasted cell must not parse the notebook kernel's own command-line flags.
    status = main([] if "ipykernel" in sys.modules else None)
    if status:
        raise SystemExit(status)
