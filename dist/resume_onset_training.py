"""Recover a CONTROL/RISE export failure from saved features and checkpoints."""
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from train_short_onset import (FEATURE_DIM, FEATURE_SPEC, RISE_SPEC, RINGING_SPEC,
                               finish_onset_experiment, train_onset_experiment, sha256, write_json)

RUN_DIR = "auto"  # Existing onset-prepared directory, or auto-search working/input.
WORK_ROOT = "/kaggle/working"
INPUT_ROOT = "/kaggle/input"


def recovery_contract(run, arm):
    contract = json.loads((run / arm / "contract.json").read_text())
    if (contract.get("spectral_rise") is not (arm == "rise")
            or contract.get("feature_spec") != FEATURE_SPEC
            or contract.get("history_frames") != (34 if arm == "rise" else 30)
            or contract.get("ringing_negative_weight") != 1.
            or contract.get("pair_weight") != 0.
            or arm == "rise" and contract.get("rise_spec") != RISE_SPEC):
        raise ValueError(f"Not a compatible control/rise contract: {run / arm}")
    history = json.loads((run / arm / "history.json").read_text())
    if [row["epoch"] for row in history] != list(range(1, contract["epochs"] + 1)):
        raise ValueError(f"Incomplete training history in {run / arm}; refusing to restart it")
    if not (run / arm / "short_onset_best.pt").is_file():
        raise FileNotFoundError(f"Missing checkpoint in {run / arm}")
    if sha256(run / "features/index.json") != contract["data_index_sha256"]:
        raise ValueError(f"Changed feature index in {run}")
    return contract


def find_recovery_run(requested, roots):
    if str(requested) != "auto":
        path = Path(requested)
        run = path.parent if path.is_file() else path
        recovery_contract(run, "control")
        return run
    candidates, rejected = [], []
    discovered = {p.parent.parent.resolve() for root in roots if Path(root).is_dir()
                  for p in Path(root).rglob("control/contract.json")}
    for run in sorted(discovered):
        try:
            recovery_contract(run, "control")
            candidates.append(run)
        except (ValueError, OSError, KeyError, TypeError) as error:
            rejected.append({"run": str(run), "error": str(error)})
    if len(candidates) != 1:
        raise ValueError(json.dumps({"error": "Set RUN_DIR to one saved control/rise run. No training was started.",
                                     "candidates": [str(p) for p in candidates], "rejected": rejected,
                                     "searched_roots": [str(p) for p in roots]}))
    return candidates[0]


def recovery_sources(run):
    index = json.loads((run / "features/index.json").read_text())
    if index["feature_spec"] != FEATURE_SPEC:
        raise ValueError("Unexpected cached feature format")
    sources, names = [], set()
    for source in index["sources"]:
        # Saved Kaggle datasets move. Resolve cached arrays by basename under
        # THIS run, never use an unrelated surviving absolute training path.
        path = run / "features" / Path(source["features"]).name
        if path.name in names or sha256(path) != source["feature_sha256"]:
            raise ValueError(f"Duplicate or changed cached features: {path}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (source["frames"], FEATURE_DIM) or not np.isfinite(values).all():
            raise ValueError(f"Invalid cached feature array: {path}")
        names.add(path.name)
        sources.append(dict(source, features=str(path)))
    if {s["split"] for s in sources} != {"train", "validation", "test"}:
        raise ValueError("All three cached splits are required")
    return sources


def resume_training_run(source_run, output_root, device="auto"):
    source_run, output_root = Path(source_run), Path(output_root)
    if output_root.resolve().is_relative_to(source_run.resolve()):
        raise ValueError("Recovery output must be outside the source run directory")
    control_contract = recovery_contract(source_run, "control")
    if (source_run / "rise").exists():
        candidate_contract = recovery_contract(source_run, "rise")
        for key in ("epochs", "seed", "batch_size", "data_index_sha256", "shared_initial_weights_sha256"):
            if candidate_contract[key] != control_contract[key]:
                raise ValueError(f"Control/rise contracts differ: {key}")
    output_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="onset-recovered-", dir=output_root))
    summary_path = run / "training_summary.json"
    print(f"Recovering {source_run} into {run}; completed training will NOT repeat.", flush=True)
    try:
        # A real copy survives Save Version and leaves attached read-only output intact.
        shutil.copytree(source_run, run, dirs_exist_ok=True)
        sources = recovery_sources(run)
        arms, trained = {}, []
        for name in ("control", "rise"):
            arm_dir = run / name
            baseline = arms["control"]["validation"] if name == "rise" else None
            if arm_dir.exists():
                print(f"{name}: using saved checkpoint; evaluation/export only", flush=True)
                arms[name] = finish_onset_experiment(sources, arm_dir, baseline, device)
            else:
                # Control is required above; only a never-started rise arm may train.
                arm_dir.mkdir()
                print("rise: this arm was not started; training it once", flush=True)
                arms[name] = train_onset_experiment(
                    sources, arm_dir, control_contract["epochs"], control_contract["batch_size"],
                    control_contract["seed"], device, 1., run / "features", baseline, 0., True)
                trained.append(name)
            write_json(arm_dir / "training_summary.json", arms[name])
        matched = all(arms["control"][key] == arms["rise"][key]
                      for key in ("shared_initial_weights_sha256", "training_batches_sha256"))
        if not matched:
            raise ValueError("Recovered arms have different shared initialization/order/augmentation")
        result = {"schema_version": 4, "ok": True, "training_complete": True, "candidate_only": True,
                  "app_ready": False, "experiment": RISE_SPEC["version"],
                  "controlled_training_verified": matched, "rise_spec": RISE_SPEC,
                  "ringing_spec": RINGING_SPEC, "evaluation_backend": "ONNX Runtime CPU",
                  "recovered_from": str(source_run), "newly_trained_arms": trained,
                  "validation_criteria_met": arms["rise"]["validation_acceptance"]["accepted"],
                  "arms": arms, "summary_path": str(summary_path),
                  "preparation_summary": str(run / "summary.json"),
                  "pair_audits": {split: json.loads((run / f"{split}_pairs.json").read_text())["audit"]
                                  for split in ("train", "validation", "test")},
                  "ringing_label_audit": json.loads((run / "ringing_label_audit.json").read_text())}
        write_json(summary_path, result)
        return result
    except Exception as error:
        write_json(summary_path, {"ok": False, "error": str(error), "recovered_from": str(source_run),
                                  "output_directory": str(run), "checkpoints_preserved": True})
        raise


def recovery_main():
    source = find_recovery_run(RUN_DIR, [WORK_ROOT, INPUT_ROOT])
    report = resume_training_run(source, WORK_ROOT)
    print(json.dumps(report, indent=2))
    print(f"Share the main summary: {report['summary_path']}")


if __name__ == "__main__":
    recovery_main()
