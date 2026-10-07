"""Causal onset training shared by take7 and earlier controlled experiments.

For take7 on Kaggle, copy model_trainer.py, not this source module.
The optional train_onset_kaggle.py bundles the earlier CONTROL/RISE comparison.
Inputs are mono audio only. GuitarSet note starts are not verified pick attacks.
All annotations (including simultaneous same-PC strings) remain in scoring.
This is a candidate-only experiment, not a comparison with the app's ONNX.
"""

import argparse
import hashlib
from bisect import bisect_left, bisect_right
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import soundfile as sf

from onset_events import Event, latch_events, match_events, percentile
from onset_ringing import (RINGING_NEGATIVE_WEIGHT, RINGING_SPEC, ringing_annotations, ringing_mask,
                           ringing_event_counts, ringing_acceptance)
from onset_rise import RISE_PAST_FRAMES, RISE_SPEC, positive_spectral_rise
from onset_pairs import (PAIR_WEIGHT, PAIR_SPEC, OnsetPairBatches, build_onset_pairs,
                         onset_pair_loss, onset_pair_metrics)
from prepare_onset_kaggle import run_pipeline, sha256, write_json


TRAIN_EPOCHS = 12
TRAIN_BATCH_SIZE = 16
TRAIN_SEED = 20260923
FEATURE_SPEC = {
    "version": "short-stft-v1", "samplerate": 16000, "hop": 256,
    "windows": [1024, 2048], "max_frequency": 4000,
    "window": "symmetric Hann", "amplitude": "2*abs(rfft)/sum(window)",
    "compression": "log1p(1000*amplitude)/log(1001); no file normalization",
    "resampling": "causal linear, one source-sample delay unless already 16kHz",
    "frame_time": "exclusive audio window end; first frame=0.016 seconds",
    "startup": "zero left audio padding; no right padding or future samples",
    "storage": "float16, converted back to float32 for both training and inference",
}
ONSET_HOP = 256
ONSET_SR = 16000
FEATURE_DIM = 770  # 0..4kHz for the 1024 and 2048 point transforms.
HISTORY = 30  # Four kernel-3 convolutions, dilations 1,2,4,8.
BLOCK_FRAMES = 128
TARGET_SECONDS = .096
EVAL_EARLY = .032
EVAL_LATE = .128
THRESHOLDS = (.3, .4, .5, .6, .7, .8, .9)


def onset_resample(audio, source_sr):
    """Same time origin at every length; the interpolator never needs the future."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or source_sr < 8000:
        raise ValueError("Expected finite, nonempty mono audio with sample rate >=8000")
    if source_sr == ONSET_SR:
        return audio
    positions = np.arange(math.floor(len(audio) * ONSET_SR / source_sr)) * (source_sr / ONSET_SR) - 1
    lower = np.floor(positions).astype(np.int64)
    fraction = positions - lower
    # Zero history before the source starts. No centred antialiasing filter.
    a = np.where(lower >= 0, audio[np.clip(lower, 0, len(audio) - 1)], 0)
    b = np.where(lower + 1 >= 0, audio[np.clip(lower + 1, 0, len(audio) - 1)], 0)
    return (a + fraction * (b - a)).astype(np.float32)


def onset_features(audio):
    """Frame n sees only samples strictly before (n+1)*hop, even at startup."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono audio")
    frames = len(audio) // ONSET_HOP
    result = np.empty((frames, FEATURE_DIM), dtype=np.float16)
    offset = 0
    for size in FEATURE_SPEC["windows"]:
        window = np.hanning(size)
        bins = size * FEATURE_SPEC["max_frequency"] // ONSET_SR + 1
        padded = np.pad(audio, (size - ONSET_HOP, 0))
        if frames:
            windows = np.lib.stride_tricks.sliding_window_view(padded, size)[::ONSET_HOP][:frames]
            for start in range(0, frames, 1024):
                spectra = np.abs(np.fft.rfft(windows[start:start + 1024] * window, axis=1))[:, :bins]
                amplitude = spectra * (2 / window.sum())
                result[start:start + len(spectra), offset:offset + bins] = (
                    np.log1p(1000 * amplitude) / math.log(1001)).astype(np.float16)
        offset += bins
    return result


def onset_targets(events, frames):
    times = (np.arange(frames) + 1) * ONSET_HOP / ONSET_SR
    labels = np.zeros((frames, 12), dtype=np.float32)
    for event in events:
        # A sample at exactly the exclusive frame end is not visible yet.
        active = (times > event["t"] + 1e-9) & (times <= event["t"] + TARGET_SECONDS + 1e-9)
        labels[active, event["pc"]] = 1
    return labels


def onset_sources(prepared):
    """Use only this run's verified manifests; preserve raw events and groups."""
    dataset = json.loads((prepared / "dataset.json").read_text())
    documents = {}
    for name in ("guitarset", "synthetic"):
        path = prepared / dataset[name]["path"]
        if sha256(path) != dataset[name]["sha256"]:
            raise ValueError(f"Changed prepared manifest: {path}")
        documents[name] = json.loads(path.read_text())
    if documents["synthetic"]["generator"] != "onset-ks-v2":
        raise ValueError("Regenerate the tuned v2 synthetic sources")
    sources = []
    for source in documents["guitarset"]["sources"]:
        sources.append({"id": source["take"], "domain": "guitarset", "case": source["style"],
                        "split": source["split"], "wav": source["audio"]["path"],
                        "sha256": source["audio"]["sha256"], "duration": source["audio"]["duration"],
                        "events": source["events"], "group": source["split_group"]})
    for clip in documents["synthetic"]["clips"]:
        sources.append({"id": clip["name"], "domain": "synthetic", "case": clip["case"],
                        "split": clip["split"], "wav": str(prepared / "synthetic" / clip["wav"]),
                        "sha256": clip["wav_sha256"], "duration": clip["duration"],
                        "events": clip["events"], "group": clip["source_group"], "pair_gain": clip["gain"]})
    return sources


def cache_onset_features(sources, directory):
    directory.mkdir()
    cached = []
    for index, source in enumerate(sources):
        wav = Path(source["wav"])
        if sha256(wav) != source["sha256"]:
            raise ValueError(f"Audio changed after preparation: {wav}")
        audio, sr = sf.read(wav, dtype="float32")
        features = onset_features(onset_resample(audio, sr))
        if not len(features):
            raise ValueError(f"Recording shorter than one frame: {wav}")
        path = directory / f"{index:04d}.npy"
        np.save(path, features)
        cached.append(dict(source, features=str(path), feature_sha256=sha256(path),
                           ringing_annotations=ringing_annotations(source),
                           frames=len(features), incomplete_final_hop_seconds=(
                               source["duration"] - len(features) * ONSET_HOP / ONSET_SR)))
        if (index + 1) % 30 == 0 or index + 1 == len(sources):
            print(f"Features: {index + 1}/{len(sources)} complete recordings", flush=True)
    write_json(directory / "index.json", {"feature_spec": FEATURE_SPEC, "sources": cached})
    return cached


def feature_block(features, start, length=BLOCK_FRAMES, history_frames=HISTORY):
    """Fixed left context; a training/inference boundary never resets the audio."""
    count = min(length, len(features) - start)
    x = np.zeros((history_frames + length, FEATURE_DIM), dtype=np.float32)
    low = max(0, start - history_frames)
    destination = history_frames + low - start
    x[destination:history_frames + count] = features[low:start + count]
    return x.T.copy(), count


class OnsetBlocks:
    def __init__(self, sources, ringing_weight=1., history_frames=HISTORY):
        if not sources or any(s["split"] != "train" for s in sources):
            raise ValueError("Training blocks must contain train sources only")
        self.features = [np.load(s["features"], mmap_mode="r") for s in sources]
        self.labels = [onset_targets(s["events"], s["frames"]) for s in sources]
        self.ringing = [ringing_mask(s, s["frames"], y)[0] for s, y in zip(sources, self.labels)]
        self.ringing_weight = ringing_weight
        self.history_frames = history_frames
        self.blocks = [(i, start) for i, f in enumerate(self.features)
                       for start in range(0, len(f), BLOCK_FRAMES)]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        recording, start = self.blocks[index]
        x, count = feature_block(self.features[recording], start, history_frames=self.history_frames)
        # Exact gain transform of the fixed log spectrum, including left context.
        gain = 10 ** np.random.uniform(-.3, .3)
        x = np.log1p(np.expm1(x * math.log(1001)) * gain) / math.log(1001)
        y = np.zeros((12, BLOCK_FRAMES), dtype=np.float32)
        y[:, :count] = self.labels[recording][start:start + count].T
        mask = np.zeros(BLOCK_FRAMES, dtype=np.float32)
        mask[:count] = 1
        weights = np.ones((12, BLOCK_FRAMES), dtype=np.float32)
        weights[:, :count] += (self.ringing_weight - 1) * self.ringing[recording][start:start + count].T
        return x.astype(np.float32), y, mask, weights


def make_onset_model(spectral_rise=False):
    import torch
    from torch import nn

    class CausalOnset(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Conv1d(FEATURE_DIM, 96, 1)
            self.temporal = nn.ModuleList([nn.Conv1d(96, 96, 3, dilation=d) for d in (1, 2, 4, 8)])
            self.output = nn.Conv1d(96, 12, 1)
            nn.init.constant_(self.output.bias, -3.)
            self.spectral_rise = spectral_rise
            if spectral_rise:
                # Build AFTER all common parameters so the same seed gives the
                # same initial backbone. Zero extension preserves its answer.
                with torch.random.fork_rng(devices=[]):
                    self.rise_project = nn.Conv1d(FEATURE_DIM, 96, 1, bias=False)
                    nn.init.zeros_(self.rise_project.weight)

        def forward(self, features):
            x = self.project(features)
            if self.spectral_rise:
                x = x + self.rise_project(positive_spectral_rise(features))
            x = torch.relu(x)
            for dilation, layer in zip((1, 2, 4, 8), self.temporal):
                x = torch.relu(x + layer(nn.functional.pad(x, (2 * dilation, 0))))
            return self.output(x)

    return CausalOnset()


def onset_predictions(sources, infer, batch_size=16, history_frames=HISTORY):
    """Full files, each frame once, with the same left context as training."""
    result = []
    for source in sources:
        features = np.load(source["features"], mmap_mode="r")
        probabilities = []
        starts = list(range(0, len(features), BLOCK_FRAMES))
        for offset in range(0, len(starts), batch_size):
            blocks = [feature_block(features, start, history_frames=history_frames) for start in starts[offset:offset + batch_size]]
            logits = infer(np.stack([x for x, _ in blocks]))
            if logits.shape != (len(blocks), 12, history_frames + BLOCK_FRAMES) or not np.isfinite(logits).all():
                raise ValueError("Invalid model output")
            for row, (_, count) in zip(logits, blocks):
                probabilities.append((1 / (1 + np.exp(-np.clip(row[:, history_frames:history_frames + count], -80, 80)))).T)
        values = np.concatenate(probabilities)
        if len(values) != source["frames"]:
            raise ValueError("Refusing partial-file evaluation")
        result.append((source, values))
    return result


def local_event_matches(reference, predicted, early=EVAL_EARLY, late=EVAL_LATE):
    """Exact scorer on disjoint tolerance components, avoiding full-take DP tables."""
    pairs = []
    for pc in range(12):
        refs = sorted((e for e in reference if e.pc == pc), key=lambda e: e.t)
        preds = sorted((e for e in predicted if e.pc == pc), key=lambda e: e.t)
        pred_times = [e.t for e in preds]
        start = 0
        while start < len(refs):
            end = start + 1
            while end < len(refs) and refs[end].t - refs[end - 1].t <= early + late + 2e-9:
                end += 1
            selected = preds[bisect_left(pred_times, refs[start].t - early - 1e-9):
                             bisect_right(pred_times, refs[end - 1].t + late + 1e-9)]
            group = refs[start:end]
            pairs.extend((group[i], selected[j]) for i, j in match_events(group, selected, early, late))
            start = end
    return pairs


def onset_metrics(predictions, threshold):
    buckets = {}
    group_sets = {}
    details = []
    for source, values in predictions:
        rows = (((i + 1) * ONSET_HOP / ONSET_SR, 100, row) for i, row in enumerate(values)
                if (i + 1) * ONSET_HOP / ONSET_SR < source["duration"])
        predicted = latch_events(rows, threshold=threshold, fill_min=0)
        refs = [Event(e["id"], e["t"], e["pc"], e.get("midi"), e.get("case", source["case"]))
                for e in source["events"] if 0 <= e["t"] < source["duration"]]
        pairs = local_event_matches(refs, predicted)
        found_r, found_p = {r.id for r, _ in pairs}, {p.id for _, p in pairs}
        extra = [p for p in predicted if p.id not in found_p]
        repeated = set()
        last = {}
        collisions = 0
        overlaps = 0
        target_frames = set()
        for ref in sorted(refs, key=lambda e: e.t):
            if ref.pc in last and 0 < ref.t - last[ref.pc] <= 2:
                repeated.add(ref.id)
            if ref.pc in last and ref.t - last[ref.pc] < TARGET_SECONDS:
                overlaps += 1
            last[ref.pc] = ref.t
            key = (math.floor(ref.t * ONSET_SR / ONSET_HOP + 1e-9), ref.pc)
            collisions += key in target_frames
            target_frames.add(key)
        challenges = {e["id"] for e in source["events"] if e.get("role") == "challenge"}
        deltas = [p.t - r.t for r, p in pairs]
        # Additional predictions >400ms after ALL annotated starts: conservative decay/silence diagnostic.
        tail_extra = sum(p.t > max((r.t for r in refs), default=0) + .4 for p in extra)
        counts = {"tp": len(pairs), "fp": len(extra), "fn": len(refs) - len(pairs),
                  "seconds": source["duration"], "recordings": 1, "tail_extra": tail_extra,
                  "repeated_reference": len(repeated), "repeated_tp": len(repeated & found_r),
                  "challenge_reference": len(challenges), "challenge_tp": len(challenges & found_r),
                  "same_pc_frame_collisions": collisions,
                  "same_pc_target_overlaps": overlaps,
                  "boundary_reference": sum(r.t < EVAL_EARLY or r.t + EVAL_LATE >= source["duration"] for r in refs)}
        ringing_counts, ringing_audit = ringing_event_counts(source, predicted, pairs)
        counts.update(ringing_counts)
        for key in ("all", source["domain"], f"{source['domain']}/{source['case']}"):
            bucket = buckets.setdefault(key, dict.fromkeys(counts, 0) | {"deltas": []})
            for name, count in counts.items():
                bucket[name] += count
            bucket["deltas"].extend(deltas)
            sets = group_sets.setdefault(key, {"opportunity_source_groups": set(), "error_source_groups": set()})
            group = f"{source['domain']}:{source.get('group', source.get('source_group', source['id']))}"
            if counts["ringing_opportunities"]:
                sets["opportunity_source_groups"].add(group)
            if counts["ringing_false_events"]:
                sets["error_source_groups"].add(group)
        details.append({"source": source["id"], **counts,
                        "ringing_annotation_audit": ringing_audit,
                        "predicted": [{"id": p.id, "t": p.t, "pc": p.pc} for p in predicted],
                        "missed_ids": [r.id for r in refs if r.id not in found_r],
                        "extra_ids": [p.id for p in extra]})
    for key, bucket in buckets.items():
        bucket.update({name: sorted(values) for name, values in group_sets[key].items()})
        tp, fp, fn = (bucket[name] for name in ("tp", "fp", "fn"))
        bucket.update(precision=tp / (tp + fp) if tp + fp else 0.,
                      challenge_recall=(bucket["challenge_tp"] / bucket["challenge_reference"]
                                        if bucket["challenge_reference"] else None),
                      recall=tp / (tp + fn) if tp + fn else 0.,
                      f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                      false_events_per_minute=60 * fp / bucket["seconds"],
                      latency_p50=percentile(bucket["deltas"], .5),
                      latency_p95=percentile(bucket.pop("deltas"), .95))
    macro = float(np.mean([buckets[d]["f1"] for d in ("guitarset", "synthetic") if d in buckets]))
    return {"threshold": threshold, "macro_domain_f1": macro, "groups": buckets}, details


def validation_choice(table):
    # F1 first, fewer false events second, higher threshold as the final tie breaker.
    return max(table, key=lambda r: (r["macro_domain_f1"], -r["groups"]["all"]["fp"], r["threshold"]))


def training_dependencies():
    """Kaggle already supplies Torch; install only missing export packages there."""
    import torch
    for package in ("onnx", "onnxruntime"):
        try:
            importlib.import_module(package)
        except ImportError:
            if not Path("/kaggle/working").is_dir():
                raise RuntimeError(f"Install {package} in the test environment before training")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", package], check=True)
            importlib.invalidate_caches()
            importlib.import_module(package)
    return torch


def validate_resume_sources(previous, sources, report_path):
    """Keep exact model inputs/labels; regenerated synthetic WAV headers may vary."""
    fields = ("id", "split", "sha256", "feature_sha256", "events")
    differences, container_changes = [], []
    for index, (saved, source) in enumerate(zip(previous, sources)):
        if len(saved) != len(fields):
            differences.append(dict(index=index, id=source["id"], fields=["snapshot_schema"]))
            continue
        changed = [key for key, value in zip(fields, saved) if value != source[key]]
        if changed == ["sha256"] and source["domain"] == "synthetic":
            # FLOAT WAVs contain a PEAK creation timestamp. A byte hash of
            # regenerated audio can change while the actual training tensors
            # remain bit-identical. Verify the array file too, not just its index.
            path = Path(source["features"])
            if path.is_file() and sha256(path) == source["feature_sha256"]:
                container_changes.append(source["id"])
                continue
            changed.append("feature_file")
        if changed:
            differences.append(dict(index=index, id=source["id"], fields=changed))
    ok = len(previous) == len(sources) and not differences
    report = dict(ok=ok, saved_sources=len(previous), current_sources=len(sources),
                  changed_sources=len(differences), examples=differences[:10],
                  synthetic_container_changes=len(container_changes),
                  synthetic_container_examples=container_changes[:10],
                  policy="Exact ordered IDs, splits, feature hashes and events; synthetic WAV hash may differ only with verified identical feature files")
    write_json(report_path, report)
    if not ok:
        details = json.dumps(dict(saved_sources=len(previous), current_sources=len(sources),
                                  examples=differences[:3]))
        raise ValueError(f"Cannot resume onset training with changed data: {details}. "
                         f"See {report_path}. Checkpoint was not changed.")
    if container_changes:
        print(f"Resume: {len(container_changes)} synthetic WAV hashes differ; "
              "feature files and labels match the checkpoint exactly.", flush=True)
    return report


def train_onset_experiment(sources, output, epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH_SIZE,
                           seed=TRAIN_SEED, device_name="auto", ringing_weight=1.,
                           feature_directory=None, control_validation=None, pair_weight=0., spectral_rise=False,
                           initial_checkpoint=None, resume=False, checkpoint_callback=None):
    torch = training_dependencies()
    if epochs < 1 or batch_size < 1:
        raise ValueError("Positive epochs and batch size required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    splits = {split: [s for s in sources if s["split"] == split] for split in ("train", "validation", "test")}
    if any(not split for split in splits.values()):
        raise ValueError("All three splits must be nonempty")
    if ringing_weight not in (1., RINGING_NEGATIVE_WEIGHT):
        raise ValueError("This controlled experiment supports only negative weights 1 and 4")
    feature_directory = feature_directory or output / "features"
    if spectral_rise and (pair_weight or ringing_weight != 1.):
        raise ValueError("The spectral-rise experiment must use ordinary BCE only")
    history_frames = HISTORY + (RISE_PAST_FRAMES if spectral_rise else 0)
    blocks = OnsetBlocks(splits["train"], ringing_weight, history_frames)
    pair_batches = (OnsetPairBatches(splits["train"], feature_block, ONSET_HOP / ONSET_SR, TARGET_SECONDS)
                    if pair_weight else None)
    if pair_weight not in (0., PAIR_WEIGHT):
        raise ValueError("The declared pair experiment uses weight 0 or 0.1")
    loader = torch.utils.data.DataLoader(blocks, batch_size=batch_size, shuffle=True, num_workers=0,
                                         generator=torch.Generator().manual_seed(seed))
    model = make_onset_model(spectral_rise).to(device)
    if initial_checkpoint is not None:
        saved = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        prior = saved["contract"]
        if (prior["feature_spec"] != FEATURE_SPEC or
                prior["history_frames"] != history_frames or
                prior.get("spectral_rise", False) != spectral_rise):
            raise ValueError("Initial onset checkpoint has a different model/DSP contract")
        model.load_state_dict(saved["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    checkpoint = output / "short_onset_best.pt"
    contract = {"feature_spec": FEATURE_SPEC, "features": FEATURE_DIM, "history_frames": history_frames,
                "block_frames": BLOCK_FRAMES, "target_seconds": TARGET_SECONDS,
                "network_startup": f"{history_frames} zero input frames before the first audio feature",
                "spectral_rise": spectral_rise, "rise_spec": RISE_SPEC if spectral_rise else None,
                "evaluation": {"early": EVAL_EARLY, "late": EVAL_LATE, "thresholds": list(THRESHOLDS),
                               "reference_policy": "all raw note starts; no same-PC deduplication",
                               "latch": "existing app peak hysteresis, fill gate disabled; no judge simulation"},
                "architecture": "770->96 raw projection + optional rise projection; four causal residual conv3 dilations1/2/4/8 ->12 logits",
                "seed": seed, "epochs": epochs, "batch_size": batch_size, "device": str(device),
                "initial_checkpoint_sha256": sha256(initial_checkpoint) if initial_checkpoint else None,
                "positive_weight": 4, "training_gain_db": [-6, 6],
                "ringing_negative_weight": ringing_weight,
                "ringing_spec": RINGING_SPEC,
                "pair_spec": PAIR_SPEC if pair_weight else None, "pair_weight": pair_weight,
                "training_pair_audit": pair_batches.audit if pair_batches else None,
                "shared_initial_weights_sha256": hashlib.sha256(b"".join(
                    p.detach().cpu().numpy().tobytes() for name, p in model.state_dict().items()
                    if not name.startswith("rise_project."))).hexdigest(),
                "initial_weights_sha256": hashlib.sha256(b"".join(
                    p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "versions": {n: str(importlib.import_module(n).__version__) for n in
                             ("torch", "numpy", "soundfile", "onnx", "onnxruntime")},
                "data_index_sha256": sha256(feature_directory / "index.json")}
    last_path = output / "short_onset_last.pt"
    restored = None
    if resume and last_path.exists():
        restored = torch.load(last_path, map_location="cpu", weights_only=False)
        previous = restored["contract"]
        # Cache paths and package versions may differ after moving a Kaggle output.
        keys = ("feature_spec", "history_frames", "spectral_rise", "epochs",
                "batch_size", "seed", "pair_weight", "ringing_negative_weight")
        if any(previous[k] != contract[k] for k in keys):
            raise ValueError("Cannot resume onset training with changed configuration")
        validate_resume_sources(restored["sources"], sources, output / "resume_data_check.json")
        contract = previous
        model.load_state_dict(restored["state_dict"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        torch.save(restored["best_checkpoint"], checkpoint)
    write_json(output / "contract.json", contract)
    print(f"Training {contract['parameters']} parameters on {device}; {len(blocks)} blocks/epoch", flush=True)

    def infer(x):
        with torch.inference_mode():
            return model(torch.from_numpy(x).to(device)).cpu().numpy()

    history, best = [], None
    start_epoch = 1
    if restored is not None:
        history, best = restored["history"], restored["best"]
        start_epoch = restored["epoch"] + 1
        random.setstate(restored["python_rng"])
        np.random.set_state(restored["numpy_rng"])
        torch.set_rng_state(restored["torch_rng"])
        loader.generator.set_state(restored["loader_rng"])
        if device.type == "cuda" and restored["cuda_rng"]:
            torch.cuda.set_rng_state_all(restored["cuda_rng"])
        write_json(output / "history.json", history)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_total, examples = 0., 0
        started = time.monotonic()
        batch_digest = hashlib.sha256()
        pair_digest = hashlib.sha256()
        paired = pair_batches.epoch(epoch, len(loader), seed) if pair_batches else None
        input_digest = hashlib.sha256()
        pair_total, pair_count = 0., 0
        for batch_index, (x, y, mask, weights) in enumerate(loader):
            # Evidence that order and gain augmentation are identical across arms.
            input_digest.update(x.numpy().tobytes())
            # Candidate needs four extra OLD frames, not future samples. Compare
            # common raw frames/labels/gain to the original control batches.
            common_x = x[:, :, history_frames - HISTORY:]
            for tensor in (common_x, y, mask):
                batch_digest.update(tensor.numpy().tobytes())
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[:, :, history_frames:]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, pos_weight=torch.full((12, 1), 4., device=device), reduction="none")
            loss = (loss * weights.to(device) * mask[:, None, :]).sum() / (12 * mask.sum())
            pair_batch = next(paired) if paired else None
            if pair_batch is not None:
                pos, neg, pcs, pair_ids = pair_batch
                for array in (pos, neg, pcs):
                    pair_digest.update(array.tobytes())
                pair_digest.update(json.dumps(pair_ids).encode())
                pair_count += len(pcs)
                if pair_weight:
                    pair_logits = model(torch.from_numpy(np.concatenate((pos, neg))).to(device))
                    rank_loss = onset_pair_loss(pair_logits[:len(pcs)], pair_logits[len(pcs):],
                                                torch.from_numpy(pcs).to(device), HISTORY)
                    loss = loss + pair_weight * rank_loss
                    pair_total += float(rank_loss.detach()) * len(pcs)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_total += float(loss.detach()) * float(mask.sum())
            examples += float(mask.sum())
            if (batch_index + 1) % max(1, len(loader) // 4) == 0:
                print(f"Epoch {epoch}: batch {batch_index + 1}/{len(loader)}", flush=True)
        if pair_batches and pair_count != len(pair_batches.pairs):
            raise ValueError("Not every training pair was used exactly once")
        model.eval()
        predictions = onset_predictions(splits["validation"], infer, batch_size, history_frames)
        # Epoch selection fixed at .5. Only the best checkpoint gets the final threshold sweep.
        metrics, _ = onset_metrics(predictions, .5)
        choice = (metrics["macro_domain_f1"], -metrics["groups"]["all"]["fp"])
        if best is None or choice > best:
            best = choice
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "contract": contract}, checkpoint)
        history.append({"epoch": epoch, "loss": loss_total / examples,
                        "training_batches_sha256": batch_digest.hexdigest(),
                        "input_batches_sha256": input_digest.hexdigest(),
                        "pair_batches_sha256": pair_digest.hexdigest(), "pair_count": pair_count,
                        "pair_loss": pair_total / pair_count if pair_weight else None,
                        "seconds": time.monotonic() - started, "validation": metrics})
        write_json(output / "history.json", history)
        if resume:
            # A single atomic file includes BEST too: remote recovery never mixes epochs.
            state = dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                         epoch=epoch, contract=contract, history=history, best=best,
                         best_checkpoint=torch.load(checkpoint, map_location="cpu", weights_only=True),
                         sources=[(v["id"], v["split"], v["sha256"], v["feature_sha256"], v["events"])
                                  for v in sources],
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(), loader_rng=loader.generator.get_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
            temporary = last_path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(last_path)
            if checkpoint_callback:
                checkpoint_callback(last_path)
        total = metrics["groups"]["all"]
        print(f"Epoch {epoch}/{epochs}: loss={history[-1]['loss']:.5f}, "
              f"validation macro F1={metrics['macro_domain_f1']:.3f}, "
              f"FP/min={total['false_events_per_minute']:.2f}, {history[-1]['seconds']:.1f}s", flush=True)

    return finish_onset_experiment(sources, output, control_validation, device_name)


def export_probability_check(reference, exported, tolerance=2e-5):
    """Keep the numerical guard strict; discrete events can differ at a boundary."""
    if not reference or len(reference) != len(exported):
        raise ValueError("Export comparison requires the same complete recordings")
    maximum, worst_source, frames = 0., None, 0
    for (source, a), (other, b) in zip(reference, exported):
        if (source["id"] != other["id"] or a.shape != b.shape or a.ndim != 2
                or a.shape != (source["frames"], 12) or not len(a)):
            raise ValueError("Export comparison source/frame mismatch")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Non-finite export comparison")
        error = float(np.max(np.abs(a - b)))
        if error > maximum:
            maximum, worst_source = error, source["id"]
        frames += len(a)
    return {"ok": maximum <= tolerance, "max_probability_error": maximum,
            "tolerance": tolerance, "worst_source": worst_source,
            "recordings": len(reference), "frames": frames}


def export_event_comparison(reference, exported):
    """Report actual event changes without rounding probabilities or forgiving notes."""
    if len(reference) != len(exported) or any(a["source"] != b["source"] for a, b in zip(reference, exported)):
        raise ValueError("Export event comparison source mismatch")
    changed, only_reference, only_exported, examples = 0, 0, 0, []
    for a, b in zip(reference, exported):
        before = {(p["t"], p["pc"]) for p in a["predicted"]}
        after = {(p["t"], p["pc"]) for p in b["predicted"]}
        if before == after:
            continue
        changed += 1
        only_reference += len(before - after)
        only_exported += len(after - before)
        if len(examples) < 20:
            examples.append({"source": a["source"],
                             "pytorch_only": sorted(before - after)[:20],
                             "onnx_only": sorted(after - before)[:20]})
    return {"events_identical": changed == 0, "changed_recordings": changed,
            "pytorch_only_events": only_reference, "onnx_only_events": only_exported,
            "examples": examples}


def finish_onset_experiment(sources, output, control_validation=None, device_name="auto"):
    """Evaluate/export a completed training run without another optimizer step."""
    torch = training_dependencies()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    reference_device = str(device)
    checkpoint = output / "short_onset_best.pt"
    contract = json.loads((output / "contract.json").read_text())
    history = json.loads((output / "history.json").read_text())
    if [row["epoch"] for row in history] != list(range(1, contract["epochs"] + 1)):
        raise ValueError("Training is incomplete; refusing to silently restart it")
    spectral_rise = contract.get("spectral_rise", False)
    history_frames = contract["history_frames"]
    batch_size = contract["batch_size"]
    pair_weight = contract["pair_weight"]
    ringing_weight = contract["ringing_negative_weight"]
    splits = {split: [s for s in sources if s["split"] == split] for split in ("validation", "test")}
    model = make_onset_model(spectral_rise).to(device)

    def infer(x):
        with torch.inference_mode():
            return model(torch.from_numpy(x).to(device)).cpu().numpy()

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved["contract"] != contract:
        raise ValueError("Checkpoint and contract disagree")
    best_row = max(history, key=lambda row: (row["validation"]["macro_domain_f1"],
                                            -row["validation"]["groups"]["all"]["fp"]))
    if saved["epoch"] != best_row["epoch"]:
        raise ValueError("Checkpoint is not the validation-selected epoch")
    model.load_state_dict(saved["state_dict"])
    model.eval()
    predictions = onset_predictions(splits["validation"], infer, batch_size, history_frames)
    # Export the selected epoch; choose the final threshold using the deployed backend.
    model.cpu()
    device = torch.device("cpu")
    example, _ = feature_block(np.load(splits["validation"][0]["features"], mmap_mode="r"), 0, history_frames=history_frames)
    model_label = "rise" if spectral_rise else ("paired" if pair_weight else ("weighted" if ringing_weight != 1. else "control"))
    onnx_path = output / f"short_onset_{model_label}.onnx"
    torch.onnx.export(model, torch.from_numpy(example[None]), str(onnx_path),
                      input_names=["short_features"], output_names=["onset_logits"],
                      dynamic_axes={"short_features": {0: "batch", 2: "time"},
                                    "onset_logits": {0: "batch", 2: "time"}},
                      opset_version=17, dynamo=False)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])

    def infer_onnx(x):
        return session.run(["onset_logits"], {"short_features": x})[0]

    # Check every validation frame, not just one dummy tensor.
    exported = onset_predictions(splits["validation"], infer_onnx, batch_size, history_frames)
    parity = export_probability_check(predictions, exported)
    write_json(output / "onnx_validation_comparison.json", parity)
    if not parity["ok"]:
        raise ValueError(f"ONNX differs from checkpoint beyond tolerance: {parity['max_probability_error']}; "
                         "see onnx_validation_comparison.json. Training checkpoint is preserved.")
    max_error = parity["max_probability_error"]
    table = [onset_metrics(exported, threshold)[0] for threshold in THRESHOLDS]
    pair_validation = onset_pair_metrics(exported)
    selected = validation_choice(table)
    acceptance = None
    if control_validation is not None:
        qualifying = [row for row in table if ringing_acceptance(row, control_validation)["accepted"]]
        if qualifying:
            selected = min(qualifying, key=lambda row: (row["groups"]["all"]["ringing_false_events"],
                                                       -row["macro_domain_f1"], -row["threshold"]))
        acceptance = {"accepted": bool(qualifying),
                      "selection": "constraints_then_held_pc_errors_then_f1" if qualifying else "diagnostic_f1_fallback",
                      "threshold_checks": [{"threshold": row["threshold"], **ringing_acceptance(row, control_validation)}
                                           for row in table]}
    write_json(output / "validation_thresholds.json", {"checkpoint_epoch": saved["epoch"], "table": table,
                                                       "acceptance": acceptance,
                                                       "selected_threshold": selected["threshold"]})
    exported_metrics, exported_details = onset_metrics(exported, selected["threshold"])
    reference_metrics, reference_details = onset_metrics(predictions, selected["threshold"])
    parity.update(export_event_comparison(reference_details, exported_details))
    parity.update(threshold=selected["threshold"], reference_device=reference_device,
                  threshold_selection_backend="ONNX Runtime CPU",
                  pytorch_metrics=reference_metrics, onnx_metrics=exported_metrics)
    write_json(output / "onnx_validation_comparison.json", parity)
    if not parity["events_identical"]:
        print(f"Export numerical boundary differences: {parity['changed_recordings']} recordings, "
              f"max probability error {max_error:.3g}; final scores/threshold use ONNX Runtime.", flush=True)
    write_json(output / "validation_events.json", exported_details)
    test_predictions = onset_predictions(splits["test"], infer_onnx, batch_size, history_frames)
    tested, details = onset_metrics(test_predictions, selected["threshold"])
    pair_test = onset_pair_metrics(test_predictions)
    write_json(output / "test_events.json", details)
    probability_directory = output / "probabilities"
    probability_directory.mkdir(exist_ok=True)
    for split, prediction_set in (("validation", exported), ("test", test_predictions)):
        for source, probabilities in prediction_set:
            np.save(probability_directory / (Path(source["features"]).stem + f"-{split}.npy"), probabilities)
    return {"ok": True, "training_complete": True, "candidate_only": True, "app_ready": False,
            "checkpoint_epoch": saved["epoch"], "checkpoint_sha256": sha256(checkpoint),
            "model": str(onnx_path), "model_sha256": sha256(onnx_path),
            "onnx_max_probability_error": max_error, "threshold": selected["threshold"],
            "onnx_validation_comparison": parity, "evaluation_backend": "ONNX Runtime CPU",
            "validation_acceptance": acceptance,
            "initial_weights_sha256": contract["initial_weights_sha256"],
            "shared_initial_weights_sha256": contract["shared_initial_weights_sha256"],
            "spectral_rise": spectral_rise, "history_frames": history_frames,
            "input_batches_sha256": [row["input_batches_sha256"] for row in history],
            "training_batches_sha256": [row["training_batches_sha256"] for row in history],
            "pair_batches_sha256": [row["pair_batches_sha256"] for row in history],
            "pair_weight": pair_weight, "pair_validation": pair_validation, "pair_test": pair_test,
            # Include the whole fixed threshold curve in the shareable summary,
            # so same-threshold comparisons do not require another file round trip.
            "validation_thresholds": [{"threshold": row["threshold"], "macro_domain_f1": row["macro_domain_f1"],
                "groups": {name: {key: value for key, value in group.items() if not isinstance(value, list)}
                           for name, group in row["groups"].items()}} for row in table],
            "validation": selected, "test": tested,
            "limitations": ["This is a separate detector with a new input contract, not a drop-in app model.",
                            "No comparison with the existing ONNX head or actual app credits was run.",
                            "GuitarSet note starts do not verify picking technique; raw same-PC overlaps remain in recall.",
                            "A 12-class 96ms target cannot separate all closely spaced same-PC strings; overlap counts are reported.",
                            "Synthetic test uses the same simplified generator, not real guitar picking/noise.",
                            "Causal linear resampling has no antialiasing filter; validate the future live input path.",
                            "Audio-time latency excludes CPU scheduling and the app judge.",
                            "All test frames were evaluated once at the validation-selected threshold.",
                            "Test and AtoA were previously inspected; they are diagnostic regressions, not a fresh holdout.",
                            "Ringing activity follows annotations/synthetic stem support, not measured audibility."]}


def run_training_pipeline(root, output_root, variant="auto", groups=(60, 12, 12),
                          epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH_SIZE,
                          seed=TRAIN_SEED, device="auto"):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Positive epochs and batch size required")
    training_dependencies()  # Fail before expensive data generation when setup is unavailable.
    prepared = run_pipeline(root, output_root, variant, groups, seed=seed)
    run_dir = Path(prepared["summary_path"]).parent
    summary_path = run_dir / "training_summary.json"
    write_json(summary_path, {"ok": False, "stage": "features", "preparation_summary": prepared["summary_path"]})
    try:
        sources = cache_onset_features(onset_sources(Path(prepared["prepared_directory"])), run_dir / "features")
        audits = {}
        for source in sources:
            _, audit = ringing_mask(source, source["frames"], onset_targets(source["events"], source["frames"]))
            key = source["split"] + "/" + source["domain"]
            total = audits.setdefault(key, dict.fromkeys(audit, 0))
            for name, value in audit.items():
                total[name] += value
        write_json(run_dir / "ringing_label_audit.json", audits)
        pair_audits = {}
        for split in ("train", "validation", "test"):
            pairs, audit = build_onset_pairs([s for s in sources if s["split"] == split])
            pair_audits[split] = audit
            write_json(run_dir / f"{split}_pairs.json", {"audit": audit, "pairs": pairs})
        write_json(summary_path, {"ok": False, "stage": "training", "sources": len(sources)})
        arms = {}
        for name, spectral_rise in (("control", False), ("rise", True)):
            arm_dir = run_dir / name
            arm_dir.mkdir()
            print(f"\nControlled experiment: {name}, positive spectral rise={spectral_rise}", flush=True)
            write_json(summary_path, {"ok": False, "stage": name, "completed_arms": list(arms)})
            arms[name] = train_onset_experiment(
                sources, arm_dir, epochs, batch_size, seed, device, 1., run_dir / "features",
                arms["control"]["validation"] if name == "rise" else None, 0., spectral_rise)
            write_json(arm_dir / "training_summary.json", arms[name])
        matched = all(arms["control"][key] == arms["rise"][key]
                      for key in ("shared_initial_weights_sha256", "training_batches_sha256"))
        if not matched:
            raise ValueError("Arms did not receive identical initialization/order/augmentation")
        result = {"schema_version": 4, "ok": True, "training_complete": True, "candidate_only": True, "app_ready": False,
                  "experiment": RISE_SPEC["version"], "controlled_training_verified": matched,
                  "ringing_spec": RINGING_SPEC,
                  "rise_spec": RISE_SPEC, "pair_audits": pair_audits,
                  "validation_criteria_met": arms["rise"]["validation_acceptance"]["accepted"],
                  "arms": arms, "ringing_label_audit": audits,
                  "summary_path": str(summary_path), "preparation_summary": prepared["summary_path"]}
        write_json(summary_path, result)
        return result
    except Exception as error:
        write_json(summary_path, {"ok": False, "error": str(error), "output_directory": str(run_dir)})
        error.add_note(f"Partial run and any completed checkpoints: {run_dir}")
        raise


def training_main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare and train the standalone onset experiment")
    parser.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--variant", choices=("auto", "mic", "mix"), default="auto")
    parser.add_argument("--groups", type=int, nargs=3, default=(60, 12, 12))
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    try:
        result = run_training_pipeline(args.input_dir, args.output_root, args.variant, args.groups,
                                       args.epochs, args.batch_size, args.seed, args.device)
    except Exception as error:
        # Keep a notebook run's diagnostic output readable even when export fails.
        # The library still raises; the notebook entry point reports explicit failure
        # and exits normally. This cannot guarantee retention by the hosting service.
        import traceback
        args.output_root.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_root / "training_failure.json"
        result = {"ok": False, "training_complete": False, "app_ready": False,
                  "error": str(error), "traceback": traceback.format_exc(),
                  "summary_path": str(summary_path), "output_root": str(args.output_root)}
        write_json(summary_path, result)
        print("TRAINING FAILED — results are incomplete. Saved checkpoints were not removed.", flush=True)
    print(json.dumps(result, indent=2))
    print(f"Small training summary to share: {result['summary_path']}")
    return result


if __name__ == "__main__":
    training_main([] if "ipykernel" in sys.modules else None)
