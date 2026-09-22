"""Controlled synthetic onset pairs for development, not a training/holdout set.

  python dist/onset_contrasts.py --output-dir contrasts --groups 3

Each group shares the SAME rendered background and gain. Only the additional
pluck(s) differ: hold, new fifth, repeated root, held triad, repeated fifth in
the triad, repeated whole triad. Events come from rendering sample positions,
never a detector or chord boundaries. One mono WAV and event CSV per case.

All variants of a source group must stay together if used in a future dataset.
These development seeds must not later be presented as an unseen evaluation.
Uses the same Karplus-Strong construction as latency_material.py, with
independent, recorded excitation seeds for each physical pluck.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from onset_events import sha256


def pluck(midi, duration, sr, seed, damping=.998):
    period = int(round(sr / (440 * 2 ** ((midi - 69) / 12))))
    buffer = np.random.default_rng(seed).uniform(-1, 1, period)
    buffer = np.convolve(buffer, [.5, .5], mode="same")
    audio = np.empty(round(duration * sr), dtype=np.float64)
    position = 0
    for i in range(len(audio)):
        audio[i] = buffer[position]
        following = (position + 1) % period
        buffer[position] = damping * .5 * (buffer[position] + buffer[following])
        position = following
    audio *= np.minimum(1, np.arange(len(audio)) / (.003 * sr))
    return audio


def render_group(group, seed=20260921, sr=44100):
    if group < 0 or seed < 0 or sr < 8000:
        raise ValueError("Expected nonnegative group/seed and sample rate >=8000")
    source = f"ks-{seed}-{group}"
    root = [45, 47, 40, 50, 52, 55][group % 6]
    gap = [1.2, .32, .64][group % 3]
    level = [1.0, .35, 1.5][(group // 3) % 3]
    initial = 1.5
    challenge = round((initial + gap) * sr) / sr
    total = round((challenge + 3.5) * sr)
    starts = [round((initial + i * .015) * sr) for i in range(3)]
    midis = [root, root + 4, root + 7]
    samples, stems = {}, {}
    for role in ("context", "challenge"):
        for i, midi in enumerate(midis):
            excitation = seed + group * 100 + i + (10 if role == "challenge" else 0)
            start = starts[i] if role == "context" else round(challenge * sr) + round(i * .015 * sr)
            wave = pluck(midi, (total - start) / sr, sr, excitation)
            stem = np.zeros(total, dtype=np.float64)
            stem[start:start + len(wave)] = wave * (level if role == "challenge" else 1.0)
            key = (role, i)
            stems[key] = stem
            samples[key] = {"source_id": f"{source}:{role}:{i}", "t": start / sr,
                            "sample": start, "midi": midi, "pc": midi % 12,
                            "role": role, "excitation_seed": excitation,
                            "level": level if role == "challenge" else 1.0}
    # A selective fifth is struck at the challenge time, not at the offset of
    # the third string in a full strum. Render it once, share it across pairs.
    fifth_key = ("selective_fifth", 2)
    fifth_event = dict(samples[("challenge", 2)])
    start = round(challenge * sr)
    fifth_event.update(source_id=f"{source}:selective_fifth", t=challenge,
                       sample=round(challenge * sr))
    selective = np.zeros(total, dtype=np.float64)
    selective[start:] = pluck(root + 7, (total - start) / sr, sr,
                             fifth_event["excitation_seed"]) * level
    stems[fifth_key], samples[fifth_key] = selective, fifth_event
    root_context = [("context", 0)]
    triad_context = [("context", i) for i in range(3)]
    cases = {
        "root_hold": root_context,
        "root_plus_fifth": root_context + [fifth_key],
        "root_repluck": root_context + [("challenge", 0)],
        "triad_hold": triad_context,
        "triad_fifth": triad_context + [fifth_key],
        "triad_repluck": triad_context + [("challenge", i) for i in range(3)],
    }
    tracks = {case: sum((stems[key] for key in keys), np.zeros(total)) for case, keys in cases.items()}
    # One gain for the whole group: per-file normalization would change the
    # background too, confounding a contrast intended to change just the pluck.
    gain = .8 / max(float(np.max(np.abs(track))) for track in tracks.values())
    clips = []
    for case, track in tracks.items():
        events = [dict(samples[key], id=f"{source}:{case}:{i}", case=f"{case}/{samples[key]['role']}")
                  for i, key in enumerate(cases[case])]
        clips.append({"name": f"{source}-{case}", "case": case, "source_group": source,
                      "split": "development", "seed": seed, "root_midi": root,
                      "challenge_at": challenge, "gap_seconds": gap,
                      "challenge_level": level, "gain": gain, "sr": sr,
                      "expected_new_pcs": sorted({e["pc"] for e in events if e["role"] == "challenge"}),
                      "audio": (track * gain).astype(np.float32), "events": events})
    return clips


def write_dataset(output, groups, seed, sr):
    if groups < 1:
        raise ValueError("At least one group is required")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "purpose": "development", "generator": "Karplus-Strong",
                "seed": seed, "groups": groups, "samplerate": sr, "clips": []}
    for group in range(groups):
        for clip in render_group(group, seed, sr):
            wav_path = output / f"{clip['name']}.wav"
            csv_path = output / f"{clip['name']}.csv"
            audio = clip.pop("audio")
            sf.write(wav_path, audio, sr, subtype="FLOAT")
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(clip["events"][0]))
                writer.writeheader()
                writer.writerows(clip["events"])
            clip.update(wav=wav_path.name, reference=csv_path.name, frames=len(audio),
                        duration=len(audio) / sr, wav_sha256=sha256(wav_path),
                        reference_sha256=sha256(csv_path))
            manifest["clips"].append(clip)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--sr", type=int, default=44100)
    args = parser.parse_args()
    manifest = write_dataset(args.output_dir, args.groups, args.seed, args.sr)
    print(f"Created {len(manifest['clips'])} mono clips in {args.output_dir}; development only.")


if __name__ == "__main__":
    main()
