"""One take7 run: retain/train the chord base, train Rise, and export one four-output model.

Source for the generated, copyable model_trainer.py. No training on import.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from chord_training import chord_runtime
from train_short_onset import (run_pipeline, onset_sources, cache_onset_features,
                               train_onset_experiment, sha256, write_json, FEATURE_SPEC)

RUN_TAG = "v2_take7"
MODE = "auto"  # auto, onset_only, full, export_only; full resumes its own run
BASE_RUN = "v2_take6"
HF_REPO_ID = "greblus/chord-model-snapshots"  # set to your own repo for a new model
USE_HF = True  # False: entirely local training, no token or HF account needed
INPUT_DIR = "/kaggle/input"
OUTPUT_ROOT = "/kaggle/working"
INITIAL_ONSET = ""  # optional Rise .pt/.pth path, never an ONNX or take6 fc_onset
ONSET_EPOCHS = 12
EXPORT_ONSET_THRESHOLD = None  # export_only: normally read from the saved summary


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


def choose_chord_start(store, run_tag, base_run, mode):
    if mode not in ("auto", "onset_only", "full"):
        raise ValueError("Mode must be auto, onset_only or full")
    own = store.fetch(f"checkpoint_{run_tag}_best.pth")
    if own:
        return "resume", own
    if mode != "full":
        base = store.fetch(f"checkpoint_{base_run}_best.pth")
        if base:
            return "base", base
    if mode == "onset_only":
        raise ValueError("onset_only needs a chord checkpoint; use auto/full to train everything")
    return "fresh", None


def weights_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare_chords(config, store):
    import torch
    decision, path = choose_chord_start(store, config["run_tag"], config["base_run"], config["mode"])
    print(f"Chord base: {decision} ({path or 'random initialization'})", flush=True)
    runtime = chord_runtime(config, store)
    saved = torch.load(path, map_location="cpu", weights_only=False) if path else None
    if decision == "base" or saved and saved.get("phase1_done", False) and saved.get("phase2_done", False):
        model = runtime.model().to(runtime.device)
        runtime.load_weights(model, saved["model_state_dict"])
        own = Path(config["work_dir"]) / f"checkpoint_{config['run_tag']}_best.pth"
        if decision == "base":
            saved = dict(saved, model_state_dict=model.state_dict(),
                         phase1_done=True, phase2_done=True,
                         parent_checkpoint=path.name, parent_sha256=sha256(path))
            torch.save(saved, own)
            store.publish(own, own.name)
    else:
        if config["mode"] == "onset_only":
            raise ValueError("Own chord checkpoint is incomplete; use auto to resume the chord phases")
        runtime.train()
        own = store.fetch(f"checkpoint_{config['run_tag']}_best.pth")
        if own is None:
            raise RuntimeError("Full chord training produced no checkpoint")
        saved = torch.load(own, map_location="cpu", weights_only=False)
        model = runtime.model().to(runtime.device)
        runtime.load_weights(model, saved["model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    digest = weights_digest(model.state_dict())
    artifact = Path(config["work_dir"]) / f"best_model_{config['run_tag']}_chords.onnx"
    runtime.export(model, str(artifact), saved.get("best_threshold", .5))
    return dict(path=str(artifact), sha256=sha256(artifact), weights_sha256=digest,
                checkpoint=str(own), checkpoint_sha256=sha256(own),
                initialized_from=decision, pitch_threshold=saved.get("best_threshold", .5),
                outputs=["root_logits", "quality_logits", "pitch_logits"], legacy_onset=False)


def prepare_features(root, work, groups):
    index = work / "features" / "index.json"
    if index.exists():
        document = json.loads(index.read_text())
        if document["feature_spec"] != FEATURE_SPEC:
            raise ValueError("Cached onset features use a different DSP contract")
        sources = document["sources"]
        for source in sources:
            path = work / "features" / Path(source["features"]).name
            if not path.is_file() or sha256(path) != source["feature_sha256"]:
                raise ValueError(f"Missing or corrupt onset cache: {path}")
            source["features"] = str(path)
        return sources
    prepared = run_pipeline(root, work, "auto", groups=groups, seed=20260923)
    return cache_onset_features(onset_sources(Path(prepared["prepared_directory"])), work / "features")


def export_combined_model(chord_path, onset_path, output, onset_threshold):
    """Compose the trained branches into one graph; no Torch or training involved."""
    import onnx
    import onnxruntime as ort
    import numpy as np
    from onnx import compose, version_converter

    if not 0 < float(onset_threshold) < 1:
        raise ValueError("A validated onset threshold between zero and one is required")
    chord_path, onset_path, output = map(Path, (chord_path, onset_path, output))
    if output.resolve() in (chord_path.resolve(), onset_path.resolve()):
        raise ValueError("Combined export must not overwrite its source models")
    chords, onset = onnx.load(str(chord_path)), onnx.load(str(onset_path))
    chord_outputs = ["root_logits", "quality_logits", "pitch_logits"]
    if {v.name for v in chords.graph.input} != {"features"}:
        raise ValueError("Expected the CQT chord model with input 'features'")
    if {v.name for v in chords.graph.output} != set(chord_outputs):
        raise ValueError("Expected the three-output chord base without the retired onset head")
    if ({v.name for v in onset.graph.input} != {"short_features"} or
            {v.name for v in onset.graph.output} != {"onset_logits"}):
        raise ValueError("Expected the separate Rise model, not an old four-head chord model")
    if [d.dim_value for d in chords.graph.input[0].type.tensor_type.shape.dim][1:] != [48, 168]:
        raise ValueError("Chord model has an incompatible feature shape")
    if onset.graph.input[0].type.tensor_type.shape.dim[1].dim_value != 770:
        raise ValueError("Rise model must accept 770 short-spectrum features")
    # Both current exporters use opset17. Allow older chord exports through the
    # ONNX converter, guarded by numerical comparison to the original graph below.
    opsets = [{p.domain: p.version for p in m.opset_import} for m in (chords, onset)]
    if any(set(v) != {""} for v in opsets):
        raise ValueError("Combined export currently requires standard ONNX operators")
    version = max(v[""] for v in opsets)
    normalized = [version_converter.convert_version(m, version) if v[""] != version else m
                  for m, v in zip((chords, onset), opsets)]
    ir = max(m.ir_version for m in normalized)
    for model in normalized:
        model.ir_version = ir
    # Keep branch names disjoint, then restore the four public output names.
    merged = compose.merge_models(*normalized, io_map=[], prefix1="chords/", prefix2="rise/")
    names = {"chords/" + n: n for n in ["features", *chord_outputs]}
    names.update({"rise/" + n: n for n in ["short_features", "onset_logits"]})
    def rename_graph(graph):
        for item in [*graph.input, *graph.output, *graph.value_info, *graph.initializer]:
            item.name = names.get(item.name, item.name)
        for node in graph.node:
            for items in (node.input, node.output):
                for index, name in enumerate(items):
                    items[index] = names.get(name, name)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    rename_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph in attribute.graphs:
                        rename_graph(graph)
    rename_graph(merged.graph)
    metadata = {p.key: p.value for p in merged.metadata_props}
    metadata.update(model_kind="solitito-chord-rise-v1", onset_threshold=str(onset_threshold),
                    onset_history_frames="34", onset_feature_spec=json.dumps(FEATURE_SPEC, sort_keys=True),
                    chord_source_sha256=sha256(chord_path), onset_source_sha256=sha256(onset_path))
    onnx.helper.set_model_props(merged, metadata)
    onnx.checker.check_model(merged)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.onnx")
    onnx.save_model(merged, str(temporary), save_as_external_data=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    def session(path):
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    base_session, rise_session, combined = session(chord_path), session(onset_path), session(temporary)
    errors = dict.fromkeys([*chord_outputs, "onset_logits"], 0.)
    rng = np.random.default_rng(20260929)
    try:
        for batch, frames in ((1, 35), (2, 37), (1, 65)):
            cqt = rng.uniform(0, 1, (batch, 48, 168)).astype(np.float32)
            short = rng.uniform(0, .5, (batch, 770, frames)).astype(np.float32)
            reference = base_session.run(chord_outputs, {"features": cqt}) + rise_session.run(
                ["onset_logits"], {"short_features": short})
            actual = combined.run(list(errors), {"features": cqt, "short_features": short})
            for name, a, b in zip(errors, reference, actual):
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f"Invalid merged output: {name}")
                errors[name] = max(errors[name], float(np.max(np.abs(a - b))))
                if not np.allclose(a, b, rtol=2e-5, atol=2e-5):
                    raise ValueError(f"Merged output changed: {name}, max error {errors[name]}")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(path=str(output), sha256=sha256(output), inputs={"features": ["batch", 48, 168],
                "short_features": ["batch", 770, "time"]}, outputs=list(errors),
                onset_output_shape=["batch", 12, "time"], onset_threshold=float(onset_threshold),
                parity_max_absolute_error=errors, parity_cases=3,
                runtime_integration_required=True)


def export_take7_only(config, store):
    """Recover the completed two-file take7 run without datasets or optimizers."""
    tag = config["run_tag"]
    work = Path(config["work_dir"])
    report_path = store.fetch(f"training_summary_{tag}.json")
    report = json.loads(report_path.read_text()) if report_path else {}
    chord = store.fetch(f"best_model_{tag}_chords.onnx")
    onset = store.fetch(f"best_model_{tag}_onset.onnx")
    if onset is None and (work / "rise" / "short_onset_rise.onnx").is_file():
        onset = work / "rise" / "short_onset_rise.onnx"
    if chord is None or onset is None:
        raise FileNotFoundError("export_only requires the saved *_chords.onnx and *_onset.onnx "
                                "(or rise/short_onset_rise.onnx). No training was started.")
    threshold = config.get("export_onset_threshold")
    if threshold is None:
        threshold = report.get("onset", {}).get("threshold")
    if threshold is None:
        raise ValueError("Missing onset threshold: restore training_summary_<RUN_TAG>.json "
                         "or set EXPORT_ONSET_THRESHOLD to its selected threshold. No training was started.")
    model = export_combined_model(chord, onset, work / f"best_model_{tag}.onnx", threshold)
    store.publish(model["path"], Path(model["path"]).name)
    report.update(schema_version=2, ok=True, run_tag=tag, model=model, app_ready=False,
                  export_only=True, training_performed=False,
                  note="One ONNX, two inputs, four outputs. Requires the matching application input path.")
    destination = work / f"training_summary_{tag}.json"
    report["summary_path"] = str(destination)
    write_json(destination, report)
    store.publish(destination, destination.name)
    print(f"Export complete, no training: {model['path']}", flush=True)
    return report


def run_take7(config, store):
    work = Path(config["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    tag = config["run_tag"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag) or tag == config["base_run"]:
        raise ValueError("Choose a simple new RUN_TAG different from BASE_RUN; never overwrite take6")
    if config["mode"] == "export_only":
        return export_take7_only(config, store)
    report_path = work / f"training_summary_{tag}.json"
    write_json(report_path, dict(ok=False, stage="chords", run_tag=tag))
    chords = prepare_chords(config, store)
    write_json(report_path, dict(ok=False, stage="onsets", run_tag=tag, chords=chords))
    sources = prepare_features(Path(config["input_dir"]), work, config.get("groups", (60, 12, 12)))
    onset_dir = work / "rise"
    onset_dir.mkdir(exist_ok=True)
    last_name = f"checkpoint_{tag}_onset_last.pth"
    previous = store.fetch(last_name)
    if previous:
        shutil.copy2(previous, onset_dir / "short_onset_last.pt")
    initial = config.get("initial_onset") or None
    if initial and not Path(initial).is_file():
        raise ValueError(f"Initial Rise checkpoint does not exist: {initial}")
    print("Onsets: " + ("resuming take7" if previous else "initial Rise weights" if initial else
                        "training Rise from scratch; chord base is frozen"), flush=True)
    result = train_onset_experiment(sources, onset_dir,
                                   epochs=config.get("onset_epochs", 12),
                                   batch_size=config.get("onset_batch_size", 16),
                                   device_name=config.get("device", "auto"),
                                   feature_directory=work / "features", spectral_rise=True,
                                   initial_checkpoint=initial if not previous else None, resume=True,
                                   checkpoint_callback=lambda path: store.publish(path, last_name))
    # No reference to the chord model is passed into the onset optimizer.
    import torch
    saved = torch.load(chords["checkpoint"], map_location="cpu", weights_only=False)
    if weights_digest(saved["model_state_dict"]) != chords["weights_sha256"]:
        raise RuntimeError("Chord weights changed during onset training")
    model = export_combined_model(chords["path"], result["model"],
                                  work / f"best_model_{tag}.onnx", result["threshold"])
    store.publish(model["path"], Path(model["path"]).name)
    store.publish(onset_dir / "short_onset_best.pt", f"checkpoint_{tag}_onset_best.pth")
    summary = dict(schema_version=2, ok=True, training_complete=True,
                   candidate_only=True, app_ready=False, run_tag=tag, mode=config["mode"],
                   model=model, chords=chords, onset=result, summary_path=str(report_path),
                   note="One ONNX, two inputs, four outputs. Requires the matching application input path.")
    write_json(report_path, summary)
    store.publish(report_path, report_path.name)
    print(f"Take7 complete. Model: {model['path']}\nReport: {report_path}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "onset_only", "full", "export_only"), default=MODE)
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--base-run", default=BASE_RUN)
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--hf-repo", default=HF_REPO_ID)
    parser.add_argument("--no-hf", action="store_true", default=not USE_HF)
    parser.add_argument("--initial-onset", default=INITIAL_ONSET)
    parser.add_argument("--onset-epochs", type=int, default=ONSET_EPOCHS)
    parser.add_argument("--export-onset-threshold", type=float, default=EXPORT_ONSET_THRESHOLD)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    import sys
    args = parser.parse_args([] if argv is None and "ipykernel" in sys.modules else argv)
    config = vars(args)
    config["work_dir"] = str(Path(args.output_root) / args.run_tag)
    if config["mode"] == "export_only":
        config["device"] = "cpu"
    elif config["device"] == "auto":
        import torch
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    token = os.environ.get("HF_TOKEN")
    if not args.no_hf and not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as error:
            raise RuntimeError("Set the Kaggle HF_TOKEN secret or use --no-hf / USE_HF=False") from error
    store = SnapshotStore(config["work_dir"], None if args.no_hf else args.hf_repo, token)
    try:
        return run_take7(config, store)
    except Exception as error:
        write_json(Path(config["work_dir"]) / "training_failure.json",
                   dict(ok=False, training_complete=False, error=str(error), work_dir=config["work_dir"]))
        raise


if __name__ == "__main__":
    main()
