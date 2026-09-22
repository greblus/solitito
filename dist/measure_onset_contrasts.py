"""Measure all clips from onset_contrasts.py with an explicitly selected model.

  python dist/measure_onset_contrasts.py --dataset contrasts --model old.onnx \
      --binary /path/to/solitito-probe --output-dir measurements

Runs capture_onset_probe and onset_events without replacing the app/model.
Both fill>=50% and all-fill latch scores are reported. These are detector
diagnostics, NOT application credit counts. Keeps each complete log/report
and updates summary.json after every clip; a failed batch remains ok=false.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from onset_events import sha256


def measure(dataset, model, binary, output):
    scripts = Path(__file__).resolve().parent
    dataset, model, binary = dataset.resolve(), model.resolve(), binary.resolve()
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    clips = manifest["clips"]
    if not clips or len({c["name"] for c in clips}) != len(clips):
        raise ValueError("Expected nonempty clips with unique names")
    for clip in clips:
        for key in ("name", "wav", "reference"):
            if Path(clip[key]).name != clip[key]:
                raise ValueError("Clip names/paths must be plain filenames")
        for key in ("wav", "reference"):
            if sha256(dataset / clip[key]) != clip[f"{key}_sha256"]:
                raise ValueError(f"Dataset file changed: {clip[key]}")
    result = {"ok": False, "dataset_manifest_sha256": sha256(manifest_path),
              "model_sha256": sha256(model), "binary_sha256": sha256(binary),
              "expected_clips": len(clips), "completed_clips": 0, "scores": []}
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")

    def run(script, *args):
        process = subprocess.run([sys.executable, str(scripts / script), *map(str, args)],
                                 capture_output=True, text=True, check=False)
        if process.returncode:
            raise RuntimeError(f"{script} failed ({process.returncode}):\n{process.stdout}\n{process.stderr}")

    try:
        for clip in clips:
            dest = output / clip["name"]
            run("capture_onset_probe.py", "--wav", dataset / clip["wav"], "--model", model,
                "--binary", binary, "--output-dir", dest)
            capture = json.loads((dest / "manifest.json").read_text())
            if (capture["inputs"]["model"]["sha256"] != result["model_sha256"]
                    or capture["inputs"]["binary"]["sha256"] != result["binary_sha256"]):
                raise ValueError("Model or binary changed between clips")
            for fill in (50, 0):
                report = dest / f"score-fill{fill}.json"
                run("onset_events.py", "--reference", dataset / clip["reference"],
                    "--probe", dest / "probe.txt", "--wav", dataset / clip["wav"],
                    "--manifest", dest / "manifest.json", "--fill-min", fill, "--output", report)
                scores = json.loads(report.read_text())
                challenge_ids = {e["id"] for e in clip["events"] if e["role"] == "challenge"}
                hits = sum(m["reference_id"] in challenge_ids for m in scores["matches"])
                extras = [e for e in scores["extra"] if e["t"] >= clip["challenge_at"]]
                result["scores"].append({"name": clip["name"], "case": clip["case"],
                    "group": clip["source_group"], "gap_seconds": clip["gap_seconds"], "fill": fill,
                    "new_attacks": len(challenge_ids), "new_attacks_matched": hits,
                    "extra_events_after_challenge": len(extras), "extra_pcs": [e["pc"] for e in extras],
                    "ambiguous_matches": len(scores["ambiguous_prediction_ids"]),
                    "total_tp": scores["tp"], "total_fp": scores["fp"], "total_fn": scores["fn"]})
            result["completed_clips"] += 1
            summary_path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"{result['completed_clips']}/{len(clips)} {clip['name']}", flush=True)
        result["ok"] = True
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        result["error"] = str(error) or "Interrupted"
        raise
    finally:
        summary_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    measure(args.dataset, args.model, args.binary, args.output_dir)


if __name__ == "__main__":
    main()
