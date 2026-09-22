"""One-to-one, polyphonic onset scoring (not an app crediting simulation).

Reference/prediction CSV: t,pc plus optional id,midi,case. Times are seconds
from the beginning of the SAME audio. Each row is one event, not a frame.

Example (soundfile is only needed for the CLI's WAV validation):
  python dist/onset_events.py --reference onsets.csv --probe probe.txt \
      --wav probe.wav --output report.json

Probe mode measures the isolated onset latch from state.rs, using rounded
probabilities and audio-window end times. It cannot reproduce the live
scheduler, note judge, reset paths or wall-clock inference latency. Defaults
are fixed measurement settings, not proposed model/app thresholds.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class Event:
    id: str
    t: float
    pc: int
    midi: int | None = None
    case: str = ""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_events(path):
    events = []
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"t", "pc"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: expected CSV columns t,pc")
        for i, row in enumerate(reader):
            event = Event(row.get("id") or str(i), float(row["t"]), int(row["pc"]),
                          int(row["midi"]) if row.get("midi") else None,
                          row.get("case", ""))
            if not math.isfinite(event.t) or event.t < 0 or not 0 <= event.pc < 12:
                raise ValueError(f"{path}: invalid event {i}")
            if event.midi is not None and (not 0 <= event.midi <= 127 or event.midi % 12 != event.pc):
                raise ValueError(f"{path}: MIDI/PC mismatch at event {i}")
            events.append(event)
    if len({e.id for e in events}) != len(events):
        raise ValueError(f"{path}: duplicate event IDs")
    return events


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


def score(reference, predicted, start, end, early=0.05, late=0.4, pitch="pc"):
    if not all(math.isfinite(v) for v in (start, end, early, late)) or not 0 <= start < end or min(early, late) < 0:
        raise ValueError("Invalid scoring interval or tolerance")
    if pitch not in ("pc", "midi") or (pitch == "midi" and any(e.midi is None for e in reference + predicted)):
        raise ValueError("MIDI scoring requires MIDI in every event")
    refs = [e for e in reference if start <= e.t < end]
    preds = [e for e in predicted if start <= e.t < end]
    pairs = match_events(refs, preds, early, late, pitch)
    matched_r, matched_p = {i for i, _ in pairs}, {j for _, j in pairs}
    deltas = [preds[j].t - refs[i].t for i, j in pairs]
    tp, fp, fn = len(pairs), len(preds) - len(pairs), len(refs) - len(pairs)
    by_case = {}
    for case in sorted({e.case for e in refs}):
        indices = [i for i, e in enumerate(refs) if e.case == case]
        found = sum(i in matched_r for i in indices)
        by_case[case] = {"reference": len(indices), "tp": found, "fn": len(indices) - found,
                         "recall": found / len(indices)}
    return {
        "window": {"start": start, "end_exclusive": end, "early": early, "late": late, "pitch": pitch},
        "reference": len(refs), "predicted": len(preds), "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_events_per_audio_minute": fp * 60 / (end - start),
        "matched_timing_seconds": {"p50": percentile(deltas, .5), "p95": percentile(deltas, .95),
                                   "early_count": sum(d < 0 for d in deltas),
                                   "on_time_or_late_count": sum(d >= 0 for d in deltas)},
        "by_case": by_case,
        # Boundary notes remain in the denominator, never silently forgiven.
        "boundary_reference_ids": [e.id for e in refs if e.t - early < start or e.t + late >= end],
        "excluded_reference": len(reference) - len(refs),
        "excluded_predicted": len(predicted) - len(preds),
        "ambiguous_prediction_ids": [p.id for p in preds if sum(
            getattr(p, pitch) == getattr(r, pitch) and -early - 1e-9 <= p.t - r.t <= late + 1e-9
            for r in refs) > 1],
        "matches": [{"reference_id": refs[i].id, "predicted_id": preds[j].id,
                     "delta": preds[j].t - refs[i].t} for i, j in pairs],
        "missed": [asdict(e) for i, e in enumerate(refs) if i not in matched_r],
        "extra": [asdict(e) for j, e in enumerate(preds) if j not in matched_p],
    }


def read_probe(path, duration):
    rows = []
    for line in Path(path).read_text().splitlines():
        if "❌" in line:
            raise ValueError(f"Probe reported a failure: {line}")
        if not re.match(r"^\s*\d+\.\d+\s", line):
            continue
        try:
            left, right = line.split("|", 1)
            fields = left.split()
            t, fill = float(fields[0]), int(fields[2].rstrip("%"))
            onset = [int(v) / 100 for v in right.split()[:12]]
            if len(fields) != 15 or len(onset) != 12 or not 0 <= fill <= 100 or any(not 0 <= v <= 1 for v in onset):
                raise ValueError("invalid fields")
        except (ValueError, IndexError) as error:
            raise ValueError(f"Malformed probe row: {line}") from error
        if t > duration + .011 or (rows and t <= rows[-1][0]):
            raise ValueError("Probe timestamps exceed WAV duration or are not increasing")
        rows.append((t, fill, onset))
    if not rows:
        raise ValueError("Probe contains no frames")
    if abs(rows[0][0] - 1.264) > .006:
        raise ValueError("Probe is missing its initial frame at 1.264 seconds")
    # This parser expects --step 1 (16 ms, printed with 10 ms resolution).
    if any(abs(row[0] - (1.264 + i * .016)) > .0051 for i, row in enumerate(rows)):
        raise ValueError("Expected continuous --step 1 probe frames")
    if duration - rows[-1][0] > .04:
        raise ValueError("Probe ended before the WAV: refusing a partial full-file score")
    return rows


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predicted", type=Path)
    source.add_argument("--probe", type=Path)
    parser.add_argument("--manifest", type=Path, help="Optional capture_onset_probe provenance to verify")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--early", type=float, default=.05)
    parser.add_argument("--late", type=float, default=.4)
    parser.add_argument("--pitch", choices=("pc", "midi"), default="pc")
    parser.add_argument("--threshold", type=float, default=.6)
    parser.add_argument("--fill-min", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--phase", type=int, default=0)
    args = parser.parse_args()
    import soundfile as sf
    info = sf.info(args.wav)
    duration = info.frames / info.samplerate
    end = args.end if args.end is not None else duration
    if end > duration:
        parser.error("--end exceeds WAV duration")
    reference = read_events(args.reference)
    if any(e.t >= duration for e in reference):
        parser.error("Reference events extend beyond WAV")
    rows = read_probe(args.probe, duration) if args.probe else None
    predicted = (latch_events(rows, args.threshold, args.fill_min, args.stride, args.phase)
                 if rows is not None else read_events(args.predicted))
    if any(e.t > duration + .011 for e in predicted):
        parser.error("Predictions extend beyond WAV")
    result = score(reference, predicted, args.start, end, args.early, args.late, args.pitch)
    paths = {"reference": args.reference, "wav": args.wav, "predictions": args.probe or args.predicted}
    result["inputs"] = {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()}
    result["audio"] = {"duration": duration, "samplerate": info.samplerate, "channels": info.channels}
    result["measurement"] = {"kind": "isolated_rounded_probe_latch" if rows else "event_csv",
                             "model_and_binary_provenance": "not established by this scorer",
                             "settings": {k: v for k, v in vars(args).items() if not isinstance(v, Path)}}
    if rows:
        result["measurement"].update(frames=len(rows), first_frame=rows[0][0], last_frame=rows[-1][0])
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        if not args.probe or not manifest.get("ok") or manifest.get("probe_sha256") != sha256(args.probe) or manifest["inputs"]["wav"]["sha256"] != sha256(args.wav):
            parser.error("Capture manifest does not certify these probe/WAV inputs")
        result["measurement"]["model_and_binary_provenance"] = manifest
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("reference", "predicted", "tp", "fp", "fn", "precision", "recall", "f1")}, indent=2))


if __name__ == "__main__":
    main()
