"""Audit saved full-file predictions and measure all three 48ms delivery phases.

No new inference, training, threshold selection or modification of references.
Requires a successfully finished compare_short_onset.py run.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from compare_short_onset import aggregate, compare_events, events_from_arrays
from prepare_onset_data import sha256, write_json


def checked_predictions(path, duration, candidate):
    data = np.load(path)
    t, p = data["times"], data["probabilities"]
    fill = np.full(len(t), 100.) if candidate else data["fill"]
    first = .016 if candidate else 1.264
    if (p.shape != (len(t), 12) or fill.shape != t.shape or not len(t)
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.allclose(t, first + np.arange(len(t)) * .016, atol=1e-9, rtol=0)
            or not 0 <= duration - t[-1] < .017):
        raise ValueError(f"Incomplete/invalid full recording: {path}")
    return t, p, fill


def summarize(directory, repo):
    summary = json.loads((directory / "summary.json").read_text())
    if not summary["ok"] or summary["completed_recordings"] != 97:
        raise ValueError("Wait for all 97 comparisons to complete successfully")
    for key, path in (("candidate", repo / "short_onset_experimental.onnx"),
                      ("original", repo / "best_model_v2_take6_onset.onnx")):
        if summary["models"][key] != sha256(path):
            raise ValueError(f"Changed model: {key}")
    sources = json.loads((directory / "sources.json").read_text())
    records, hashes, repeats, reviewed = [], {}, [], []
    for source in sources:
        if sha256(Path(source["wav"])) != source["wav_sha256"]:
            raise ValueError(f"Changed comparison WAV: {source['id']}")
        predictions = {}
        for name, suffix, threshold in (("candidate", "short-probabilities", summary["settings"]["candidate_threshold"]),
                                        ("original", "original", summary["settings"]["original_threshold"]),
                                        ("original_ungated", "original_ungated", summary["settings"]["original_threshold"])):
            path = directory / (source["id"] + "-" + suffix + ".npz")
            t, p, fill = checked_predictions(path, source["duration"], name == "candidate")
            hashes[path.name] = sha256(path)
            for stride, phase in ((1, 0), (3, 0), (3, 1), (3, 2)):
                key = f"{name}/step{stride}/phase{phase}"
                predictions[key] = events_from_arrays(t, p, threshold, fill, stride=stride, phase=phase)
                if name == "original":
                    predictions[key.replace("original/", "original_fill50/")] = events_from_arrays(
                        t, p, threshold, fill, fill_min=50, stride=stride, phase=phase)
        scores = compare_events(source, predictions)
        records.append({"domain": source["domain"], "case": source["case"], "duration": source["duration"], "scores": scores})
        if source["domain"] == "synthetic":
            result = scores["candidate/step1/phase0/strict"]
            for extra in result["extra_relations"]:
                if extra["relation"] != "already_matched_pc":
                    continue
                hold = source["source_group"] + ("-triad_hold" if source["case"] == "triad_fifth" else "-root_hold")
                with_attack = np.load(directory / (source["id"] + "-short-probabilities.npz"))
                without_attack = np.load(directory / (hold + "-short-probabilities.npz"))
                before = with_attack["times"] <= source["challenge_at"]
                if not np.array_equal(with_attack["probabilities"][before], without_attack["probabilities"][before]):
                    raise ValueError("Paired predictions differ before the added attack")
                frame = np.argmin(abs(with_attack["times"] - extra["t"]))
                repeats.append({**extra, "source": source["id"], "group": source["source_group"],
                                "case": source["case"], "hold_source": hold,
                                "challenge_at": source["challenge_at"], "new_pcs": source["expected_new_pcs"],
                                "with_new_attack": float(with_attack["probabilities"][frame, extra["pc"]]),
                                "without_new_attack": float(without_attack["probabilities"][frame, extra["pc"]])})
        else:
            checks = json.loads((repo / "dist/crediting_measurements/atoa-head-audit.json").read_text())["reviewed"]
            for check in checks:
                reviewed.append({"verdict": check["verdict"], "start": check["start"], "end": check["end"],
                                 "events": {name: [{"t": e.t, "pc": e.pc} for e in events
                                                    if check["start"] <= e.t < check["end"]]
                                            for name, events in predictions.items()}})
            write_json(directory / "AtoA-all-cadences.json", scores)
    result = {"ok": True, "complete_recordings": len(sources), "comparison_summary_sha256": sha256(directory / "summary.json"),
              "probability_files_sha256": hashes, "aggregate": aggregate(records),
              "candidate_repeated_matched_pc": len(repeats),
              "candidate_repeat_source_groups": len({r["group"] for r in repeats}),
              "candidate_repeat_cases": dict(Counter(r["case"] for r in repeats)),
              "counterfactual_repeats": repeats,
              "atoa_reviewed_windows": reviewed,
              "limitations": ["The repeated_matched_pc field refers to matched detector events, never app credits.",
                              "48ms sampling phases do not simulate asynchronous live scheduling or the judge.",
                              "AtoA reference times are approximate; retain the ambiguous A at45.186 in all denominators.",
                              "Reviewed scrape windows do not prove that a scrape caused each coincident event."]}
    write_json(directory / "delivery-and-error-analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory, Path(__file__).resolve().parent.parent)
    print(json.dumps({k: v for k, v in result.items() if k in (
        "ok", "complete_recordings", "candidate_repeated_matched_pc", "candidate_repeat_source_groups")}, indent=2))
