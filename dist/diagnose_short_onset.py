"""Replay a short detector and distinguish low response from latch suppression.

Retrospective diagnosis at frozen validation thresholds, never app crediting.
Synthetic audio is reconstructed from the existing generator and seed; all
candidate events must exactly reproduce the supplied Kaggle validation report.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np

from compare_short_onset import session
from onset_events import latch_events
from prepare_onset_data import render_group, sha256, write_json
from train_short_onset import onset_features, onset_predictions, onset_metrics


def latch_trace(values, threshold):
    """Instrument the existing peak hysteresis; check against its actual output."""
    armed = np.ones(12, dtype=bool)
    # Keep scalar types identical to latch_events, including float32 peaks.
    peaks = [0.0] * 12
    before, after, levels, emitted = [], [], [], []
    for frame, row in enumerate(values):
        before.append(armed.copy())
        reset_level = [max(.3 * peak, .1) for peak in peaks]
        levels.append(reset_level.copy())
        events = []
        for pc, value in enumerate(row):
            if value < reset_level[pc]:
                armed[pc] = True
            elif armed[pc] and value >= threshold:
                armed[pc], peaks[pc] = False, value
                events.append(pc)
        after.append(armed.copy())
        emitted.extend((frame, pc) for pc in events)
    existing = latch_events((((i + 1) * .016, 100, row) for i, row in enumerate(values)),
                            threshold=threshold, fill_min=0)
    expected = [(int(e.id.split(":")[0]), e.pc) for e in existing]
    if emitted != expected:
        raise ValueError("Diagnostic trace diverges from the actual latch")
    return {"armed_before": np.asarray(before), "armed_after": np.asarray(after),
            "reset_level": np.asarray(levels), "events": existing}


def diagnose_event(event, values, trace, threshold):
    times = (np.arange(len(values)) + 1) * .016
    pc, t = event["pc"], event["t"]
    window = (times >= t - .032 - 1e-9) & (times <= t + .128 + 1e-9)
    indices = np.flatnonzero(window)
    if not len(indices):
        raise ValueError("No observable frames in this event's scoring window")
    peak_frame = int(indices[np.argmax(values[indices, pc])])
    peak = float(values[peak_frame, pc])
    events = [e for e in trace["events"] if e.pc == pc]
    in_window = [e for e in events if t - .032 - 1e-9 <= e.t <= t + .128 + 1e-9]
    if in_window:
        classification = "event_present_check_matching"
    elif peak < threshold:
        classification = "response_below_threshold"
    else:
        classification = "latch_blocked"
        high = indices[values[indices, pc] >= threshold]
        if np.any(trace["armed_before"][high, pc]):
            raise ValueError("An armed above-threshold response should have emitted an event")
    prior = [e for e in events if e.t < t - .032 - 1e-9]
    since_previous = (times > prior[-1].t if prior else np.ones(len(times), dtype=bool)) & (times <= t)
    rearmed = np.flatnonzero(since_previous & trace["armed_after"][:, pc])
    last_frame_before_attack = np.flatnonzero(times <= t)
    return {"reference": event, "classification": classification, "threshold": threshold,
            "max_probability": peak, "peak_time": float(times[peak_frame]),
            "armed_at_peak": bool(trace["armed_before"][peak_frame, pc]),
            "armed_before_attack": bool(trace["armed_after"][last_frame_before_attack[-1], pc]) if len(last_frame_before_attack) else True,
            "reset_level_at_peak": float(trace["reset_level"][peak_frame, pc]),
            "previous_event_time": prior[-1].t if prior else None,
            "first_armed_frame_since_previous_event": float(times[rearmed[0]]) if len(rearmed) else None,
            "same_pc_events": [{"t": e.t, "id": e.id} for e in events]}


def diagnose(model, summary_path, control_events, weighted_events, output):
    summary = json.loads(summary_path.read_text())
    arms = summary["arms"]
    if sha256(model) != arms["weighted"]["model_sha256"]:
        raise ValueError("Model does not match the weighted Kaggle report")
    reports = {arm: {r["source"]: r for r in json.loads(path.read_text())}
               for arm, path in (("control", control_events), ("weighted", weighted_events))}
    output.mkdir(exist_ok=True, parents=True)
    threshold = arms["weighted"]["threshold"]
    runtime = session(model)
    completed, diagnoses, missed, metrics = [], [], [], []
    write_json(output / "summary.json", {"ok": False, "stage": "replay"})
    with tempfile.TemporaryDirectory(prefix="onset-diagnostic-features-") as tmp:
        for group in range(12):
            for clip in render_group("validation", group, seed=20260923):
                audio = clip.pop("audio")
                features = onset_features(audio)
                feature_path = Path(tmp) / "current.npy"
                np.save(feature_path, features)
                source = dict(clip, id=clip["name"], domain="synthetic", group=clip["source_group"],
                              frames=len(features), features=str(feature_path))
                _, probabilities = onset_predictions([source], lambda x: runtime.run(
                    ["onset_logits"], {"short_features": x})[0])[0]
                measured, details = onset_metrics([(source, probabilities)], threshold)
                expected = reports["weighted"][source["id"]]
                if details[0] != expected:
                    raise ValueError(f"Events or counts differ from Kaggle: {source['id']}")
                times = (np.arange(len(probabilities)) + 1) * .016
                valid = times < source["duration"]
                trace = latch_trace(probabilities[valid], threshold)
                np.savez_compressed(output / (source["id"] + ".npz"), times=times[valid],
                                    probabilities=probabilities[valid], armed_before=trace["armed_before"],
                                    armed_after=trace["armed_after"], reset_level=trace["reset_level"])
                lost = set(expected["missed_ids"]) - set(reports["control"][source["id"]]["missed_ids"])
                for event in source["events"]:
                    if event["id"] in expected["missed_ids"]:
                        result = diagnose_event(event, probabilities[valid], trace, threshold)
                        result.update(source=source["id"], case=source["case"], group=source["group"],
                                      gap_seconds=source["gap_seconds"], challenge_level=source["challenge_level"],
                                      strum_seconds=source["strum_seconds"], lost_vs_control=event["id"] in lost)
                        missed.append(result)
                        if event["id"] in lost:
                            diagnoses.append(result)
                completed.append(source["id"])
                metrics.append(measured["groups"]["all"])
            print(f"Verified complete synthetic group {group + 1}/12", flush=True)
    expected_ids = {sid for sid in reports["weighted"] if sid.startswith("onset-ks-v2-")}
    if set(completed) != expected_ids:
        raise ValueError("Incomplete synthetic replay")
    result = {"ok": True, "model_sha256": sha256(model), "threshold": threshold,
              "provenance": {str(p): sha256(p) for p in (summary_path, control_events, weighted_events, Path(__file__))},
              "verified_complete_recordings": len(completed),
              "metrics": {key: sum(row[key] for row in metrics) for key in ("tp", "fp", "fn", "ringing_false_events")},
              "lost_classifications": dict(Counter(r["classification"] for r in diagnoses)),
              "missed_classifications": dict(Counter(r["classification"] for r in missed)),
              "lost_events": diagnoses, "all_missed_events": missed,
              "limitations": ["Retrospective validation diagnosis, no new threshold or model selection.",
                              "Control comparison uses saved events, not newly inferred control probabilities.",
                              "Only synthetic validation is reproduced; GuitarSet audio is not available locally.",
                              "A blocked latch diagnosis does not establish a safe replacement rule."]}
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--control-events", type=Path, required=True)
    parser.add_argument("--weighted-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.model, args.summary, args.control_events, args.weighted_events, args.output)
    print(json.dumps({k: result[k] for k in ("ok", "verified_complete_recordings", "metrics", "lost_classifications", "missed_classifications")}, indent=2))
