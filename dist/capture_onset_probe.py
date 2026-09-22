"""Capture a complete --step 1 probe with input hashes; never replace a model.

  python dist/capture_onset_probe.py --wav probe.wav --model model.onnx \
      --output-dir /tmp/probe-run

The directory must be new. Failed captures keep their logs and a manifest
with ok=false. Binary hash identifies what ran; repository HEAD does not
prove that a prebuilt binary was built from that revision.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from onset_events import read_probe, sha256


def main():
    import soundfile as sf
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=root / "target/release/solitito")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", type=float, default=-34)
    args = parser.parse_args()
    inputs = {"wav": args.wav.resolve(), "model": args.model.resolve(),
              "binary": args.binary.resolve(), "dsp_weights": root / "dsp_weights.json"}
    fingerprints = {key: {"path": str(path), "sha256": sha256(path)} for key, path in inputs.items()}
    info = sf.info(inputs["wav"])
    duration = info.frames / info.samplerate
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [str(inputs["binary"]), "--probe", str(inputs["wav"]), "--step", "1", "--gate", str(args.gate)]
    manifest = {"ok": False, "started_utc": datetime.now(timezone.utc).isoformat(),
                "command": command, "cwd": str(root), "inputs": fingerprints,
                "audio": {"duration": duration, "frames": info.frames, "samplerate": info.samplerate,
                          "channels": info.channels},
                "settings": {"gate_db": args.gate, "step": 1, "bass_boost": False},
                "binary_source_revision": "unknown; identified by binary hash",
                "probe_precision": {"timestamp_seconds": .01, "probability": .01}}
    started = time.monotonic()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        environment = dict(os.environ, SOLITITO_MODEL=str(inputs["model"]))
        with (output / "probe.txt").open("w") as stdout, (output / "stderr.txt").open("w") as stderr:
            run = subprocess.run(command, cwd=root, env=environment, stdout=stdout, stderr=stderr, check=False)
        manifest["returncode"] = run.returncode
        if run.returncode != 0 or "❌" in (output / "stderr.txt").read_text():
            raise ValueError("Probe process failed; see stderr.txt")
        log = (output / "probe.txt").read_text()
        if f"Model: {inputs['model']}" not in log or "onset head: yes" not in log:
            raise ValueError("Probe did not confirm the requested model with an onset head")
        rows = read_probe(output / "probe.txt", duration)
        for key, path in inputs.items():
            if sha256(path) != fingerprints[key]["sha256"]:
                raise ValueError(f"Input changed during capture: {key}")
        manifest.update(ok=True, frames=len(rows), first_frame=rows[0][0], last_frame=rows[-1][0],
                        probe_sha256=sha256(output / "probe.txt"))
    except (OSError, ValueError, KeyboardInterrupt) as error:
        manifest["error"] = str(error)
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
