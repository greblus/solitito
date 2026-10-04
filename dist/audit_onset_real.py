"""Bounded, read-only audit of an existing CONTROL/PAIRED Kaggle run.

No training, inference, new thresholds or package installs. Saved probabilities
must first reproduce the existing validation report. Audio review can flag label
ambiguity; mono spectra cannot prove picking technique or an incorrect label.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import soundfile as sf

from onset_events import Event, sha256
from onset_ringing import ringing_annotations
from train_short_onset import FEATURE_SPEC, onset_metrics, local_event_matches

RUN_DIR = "auto"
WORK_ROOT = "/kaggle/working"
INPUT_ROOT = "/kaggle/input"
EXPECTED_PAIRED_MODEL = "0eaa5bc13c021155fa70b6c323c557d5ae18634bc69aa5455ac24c2c69d4c140"
AUDIT_THRESHOLDS = (.8, .9)  # Frozen comparisons, not a new threshold search.


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class AuditDiscoveryError(ValueError):
    def __init__(self, message, inventory):
        super().__init__(message)
        self.inventory = inventory


def training_summary_path(run):
    paths = [run / name for name in ("training_summary.json", "training_summary.json.txt")]
    existing = [p for p in paths if p.is_file()]
    if len(existing) == 2 and read_json(existing[0]) != read_json(existing[1]):
        raise ValueError(f"Conflicting training summaries in {run}")
    if not existing:
        raise FileNotFoundError(f"No training_summary.json or .json.txt in {run}")
    return existing[0]


def missing_audit_files(run):
    required = ["features/index.json"]
    for arm in ("control", "paired"):
        required.extend(f"{arm}/{name}" for name in
                        ("contract.json", "validation_events.json", "validation_thresholds.json"))
    missing = [name for name in required if not (run / name).is_file()]
    index = run / "features/index.json"
    if index.is_file():
        sources = read_json(index)["sources"]
        for source in sources:
            if source["split"] != "validation":
                continue
            feature = Path(source["features"])
            paths = [Path("features") / feature.name]
            paths.extend(Path(arm) / "probabilities" / (feature.stem + "-validation.npy")
                         for arm in ("control", "paired"))
            missing.extend(str(path) for path in paths if not (run / path).is_file())
    else:
        missing.extend(f"{arm}/probabilities/" for arm in ("control", "paired")
                       if not (run / arm / "probabilities").is_dir())
    return missing


def find_audit_run(requested, root, input_root=None):
    # Saved notebook output can be attached under /kaggle/input with arbitrary
    # nesting or a renamed run directory. Directory names are not provenance.
    roots = [Path(root)] + ([Path(input_root)] if input_root is not None else [])
    inventory = {"searched_roots": [], "candidate_runs": [], "rejected_summaries": [],
                 "surviving_models": [], "orphan_feature_indexes": []}
    summaries = set()
    for directory in roots:
        inventory["searched_roots"].append({"path": str(directory), "exists": directory.is_dir(),
            "entries": sorted(p.name for p in directory.iterdir())[:25] if directory.is_dir() else []})
        if str(requested) == "auto" and directory.is_dir():
            summaries.update(p.resolve().parent for p in directory.rglob("training_summary.json*")
                             if p.name in ("training_summary.json", "training_summary.json.txt"))
    if str(requested) != "auto":
        path = Path(requested)
        summaries = {path.parent if path.is_file() else path}
    complete = []
    for run in sorted(summaries):
        try:
            data = read_json(training_summary_path(run))
            if not isinstance(data, dict):
                raise ValueError("Training summary must be a JSON object")
            reason = None
            if not data.get("ok") or data.get("experiment") != "same-background-ranking-v1":
                reason = "Not a completed CONTROL/PAIRED main summary"
            elif data.get("arms", {}).get("paired", {}).get("model_sha256") != EXPECTED_PAIRED_MODEL:
                reason = "Different paired model; refusing to mix experiments"
            if reason:
                inventory["rejected_summaries"].append({"path":str(run), "reason":reason})
                continue
            missing = missing_audit_files(run)
            inventory["candidate_runs"].append({"path":str(run), "missing_file_count":len(missing),
                                                "missing_files_first_20":missing[:20]})
            if not missing:
                complete.append(run)
        except (OSError, ValueError, KeyError, TypeError) as error:
            inventory["rejected_summaries"].append({"path":str(run), "reason":str(error)})
    if len(complete) == 1:
        return complete[0]
    # Only inventory surviving artifacts on failure; never extract archives or
    # reconstruct/retrain a model behind the user's back.
    for directory in roots:
        if directory.is_dir():
            inventory["surviving_models"].extend(str(p) for p in directory.rglob("short_onset*")
                                                  if p.suffix in (".onnx", ".pt"))
            inventory["orphan_feature_indexes"].extend(str(p) for p in directory.rglob("index.json")
                if p.parent.name == "features" and p.parent.parent not in summaries)
    inventory["rejected_summaries"] = inventory["rejected_summaries"][:20]
    if complete:
        message = f"Multiple complete copies found: {complete}. Set RUN_DIR to one of these directories."
    elif inventory["candidate_runs"]:
        message = "Found the correct training summary, but its saved audit files are incomplete. See discovery.candidate_runs."
    else:
        message = "The completed paired training output is not visible in working or attached input. See discovery for actual files."
    raise AuditDiscoveryError(message + " Do NOT retrain yet: restore/attach the saved training output if available; "
                              "a summary JSON alone cannot reconstruct per-frame predictions.", inventory)


def load_audit_inputs(run):
    summary = read_json(training_summary_path(run))
    index_path = run / "features/index.json"
    index = read_json(index_path)
    if index["feature_spec"] != FEATURE_SPEC:
        raise ValueError("Unsupported feature contract")
    sources = [s for s in index["sources"] if s["split"] == "validation"]
    if not sources or len({s['id'] for s in sources}) != len(sources):
        raise ValueError("Missing or duplicate validation sources")
    predictions, hashes = {}, {str(index_path): sha256(index_path)}
    for arm in ("control", "paired"):
        contract = read_json(run / arm / "contract.json")
        if contract["data_index_sha256"] != sha256(index_path):
            raise ValueError(f"Changed feature index for {arm}")
        predictions[arm] = []
        for source in sources:
            path = run / arm / "probabilities" / (Path(source["features"]).stem + "-validation.npy")
            values = np.load(path, allow_pickle=False)
            if (values.shape != (source["frames"], 12) or not np.isfinite(values).all()
                    or np.any(values < 0) or np.any(values > 1)):
                raise ValueError(f"Invalid or incomplete probabilities: {path}")
            predictions[arm].append((source, values))
            hashes[str(path)] = sha256(path)
    return summary, sources, predictions, hashes


def replay_audit(run, summary, predictions):
    metrics, details, table_differences = {}, {}, []
    for arm in ("control", "paired"):
        table = read_json(run / arm / "validation_thresholds.json")
        frozen = summary["arms"][arm]["threshold"]
        for threshold in sorted(set(AUDIT_THRESHOLDS + (frozen,))):
            print(f"Replay {arm}, threshold {threshold}", flush=True)
            measured, rows = onset_metrics(predictions[arm], threshold)
            expected = next(r for r in table["table"] if r["threshold"] == threshold)
            if measured != expected:
                # Training tables use Torch, cached probabilities use ONNX. Export
                # guarantees identical events only at the selected threshold.
                # Keep other differences visible, never silently call them equal.
                table_differences.append({'arm':arm,'threshold':threshold,
                    'original_all':expected['groups']['all'],'replayed_all':measured['groups']['all']})
            if threshold == frozen:
                if measured != summary["arms"][arm]["validation"] or rows != read_json(run / arm / "validation_events.json"):
                    raise ValueError(f"Replay does not reproduce selected events: {arm}")
            metrics[arm, threshold] = measured
            details[arm, threshold] = {r["source"]: r for r in rows}
    return metrics, details, table_differences


def annotation_context(source, t, pc):
    """Report ambiguity indicators, never relabel an event or forgive an error."""
    events = source["events"]
    nearby = [e for e in events if t - .3 <= e["t"] <= t + .3]
    old = [e for e in events if e["pc"] == pc and e["t"] + .096 < t < e.get("end", e["t"])]
    conflict = any(a.get("string") is not None and a.get("string") == b.get("string")
                   and a["pc"] != b["pc"] and b["t"] <= t < b.get("end", b["t"])
                   for a in old for b in events)
    same = [e for e in events if e["pc"] == pc]
    nearest = min(same, key=lambda e: abs(e["t"] - t)) if same else None
    flags = {"same_pc_start_in_scoring_window": any(t - .128 - 1e-9 <= e["t"] <= t + .032 + 1e-9 for e in same),
             "same_pc_start_near_scoring_boundary": any(
                 -.096 <= t - e["t"] <= .192 and not -.032 - 1e-9 <= t - e["t"] <= .128 + 1e-9
                 for e in same),
             "old_note_end_within_32ms": any(e["end"] - t <= .032 for e in old),
             "conflicting_pitch_on_old_string": conflict}
    return {"nearby_starts": nearby, "old_notes": old, "flags": flags,
            "nearest_same_pc_start_delta": t - nearest["t"] if nearest else None}


def classify_miss(event, source, values, row, threshold):
    times = (np.arange(len(values)) + 1) * .016
    visible = (times >= event["t"] - .032 - 1e-9) & (times <= event["t"] + .128 + 1e-9) & (times < source["duration"])
    if not visible.any():
        return {"cause": "unobservable_boundary", "peak": None}
    peak = float(values[visible, event["pc"]].max())
    emitted = [p for p in row["predicted"] if p["pc"] == event["pc"] and event["t"] - .032 - 1e-9 <= p["t"] <= event["t"] + .128 + 1e-9]
    cause = "event_claimed_by_other_reference" if emitted else ("below_threshold" if peak < threshold else "latch_suppressed")
    return {"cause": cause, "peak": peak,
            "nearby_same_pc_reference_count": sum(e["id"] != event["id"] and e["pc"] == event["pc"] and abs(e["t"] - event["t"]) <= .16 for e in source["events"])}


def false_held_events(source, row):
    opportunities, _ = ringing_annotations(source)  # Recompute from raw labels.
    extras = set(row["extra_ids"])
    result = []
    for p in row["predicted"]:
        hits = [o for o in opportunities if o["pc"] == p["pc"] and o["t"] < p["t"] <= o["evaluation_end"] + 1e-9]
        if p["id"] in extras and hits:
            hit = max(hits, key=lambda o: o["t"])
            result.append({"prediction": p, "opportunity": hit, "context": annotation_context(source, p["t"], p["pc"])})
    if len(result) != row["ringing_false_events"]:
        raise ValueError(f"Cannot reproduce held-PC errors from raw annotations: {source['id']}")
    return result


def spectral_evidence(features, t, midi, other_midis):
    """Fixed descriptive spectral rise; NOT an attack classifier or separability test.

The two FFT resolutions are kept separate. Harmonics within one FFT bin of
another annotated note are marked shared, not independent evidence of a pluck.
"""
    times = (np.arange(len(features)) + 1) * .016
    before = (times > t - .064) & (times <= t)
    after = (times > t) & (times <= t + .096)
    if not before.any() or not after.any():
        return {"available": False}
    result, offset = {}, 0
    for size in FEATURE_SPEC["windows"]:
        bins = size * 4000 // 16000 + 1
        pre = np.expm1(features[before, offset:offset + bins].astype(float) * np.log(1001)) / 1000
        post = np.expm1(features[after, offset:offset + bins].astype(float) * np.log(1001)) / 1000
        rise = np.maximum(post - pre.mean(axis=0), 0).max(axis=0)
        def harmonics(note):
            f = 440 * 2 ** ((note - 69) / 12)
            return [round(f * h * size / 16000) for h in range(1, 7) if f * h <= 4000]
        own = harmonics(midi)
        other = [b for note in other_midis for b in harmonics(note)]
        mask = np.zeros(bins, dtype=bool)
        unshared = np.zeros(bins, dtype=bool)
        for b in own:
            mask[max(0, b - 1):min(bins, b + 2)] = True
            if not any(abs(b - c) <= 2 for c in other):
                unshared[max(0, b - 1):min(bins, b + 2)] = True
        energy = rise ** 2
        result[str(size)] = {"total_rise_energy": float(energy.sum()),
            "target_harmonics_rise_fraction": float(energy[mask].sum() / max(energy.sum(), 1e-20)),
            "unshared_harmonics_rise_fraction": float(energy[unshared].sum() / max(energy.sum(), 1e-20)),
            "unshared_harmonic_count": sum(not any(abs(b - c) <= 2 for c in other) for b in own)}
        offset += bins
    return {"available": True, "windows": result}


def locate_audio(source, input_root):
    original = Path(source["wav"])
    paths = [original] if original.is_file() else list(Path(input_root).rglob(original.name))
    matching = [p for p in paths if sha256(p) == source["sha256"]]
    if len(matching) != 1:
        raise ValueError(f"Expected one unchanged audio file for {source['id']}; found {len(matching)}")
    return matching[0]


def audit_real(run, input_root, output):
    summary, sources, predictions, hashes = load_audit_inputs(run)
    metrics, rows, table_differences = replay_audit(run, summary, predictions)
    probabilities = {arm: {s['id']: p for s,p in predictions[arm]} for arm in predictions}
    comp = [s for s in sources if s["domain"] == "guitarset" and s["case"] == "comp"]
    if not comp:
        raise ValueError("No real comp validation sources")
    errors, flags, lost, gained, changes = {}, Counter(), [], [], []
    for s in comp:
        sid = s['id']
        errors[sid] = false_held_events(s, rows['paired', .8][sid])
        for e in errors[sid]:
            flags.update(k for k,v in e['context']['flags'].items() if v)
        old_missed = set(rows['control', .8][sid]['missed_ids'])
        new_missed = set(rows['paired', .9][sid]['missed_ids'])
        for event in s['events']:
            if event['id'] in new_missed - old_missed:
                lost.append({'source':sid,'event':event, **classify_miss(event,s,probabilities['paired'][sid],rows['paired',.9][sid],.9)})
            elif event['id'] in old_missed - new_missed:
                gained.append({'source':sid,'event':event})
        changes.append({'source':sid,'ringing_control':rows['control',.8][sid]['ringing_false_events'],
                        'ringing_paired':len(errors[sid])})
    chosen = sorted(changes, key=lambda r: (-(r['ringing_paired']-r['ringing_control']),-r['ringing_paired'],r['source']))[:5]
    cases, audio_failures = [], []
    by_id = {s['id']:s for s in comp}
    for recording in chosen:
        sid = recording['source']; s = by_id[sid]
        feature_path = run / 'features' / Path(s['features']).name
        if sha256(feature_path) != s['feature_sha256']:
            raise ValueError(f"Changed features for {sid}")
        features = np.load(feature_path, mmap_mode='r', allow_pickle=False)
        if features.shape != (s['frames'],770):
            raise ValueError(f"Invalid feature shape for {sid}")
        try:
            audio_path = locate_audio(s,input_root)
        except (ValueError,OSError) as error:
            audio_path = None
            audio_failures.append(str(error))
        refs = [Event(e['id'],e['t'],e['pc']) for e in s['events']]
        pred = [Event(p['id'],p['t'],p['pc']) for p in rows['paired',.8][sid]['predicted']]
        matched = {r.id for r,_ in local_event_matches(refs,pred)}
        seen_true = set()
        selected = []
        for error in errors[sid][:2]:  # Chronological, not probability-cherry-picked.
            t,pc=error['opportunity']['t'],error['prediction']['pc']
            old = [e for e in s['events'] if e['id'] in error['opportunity']['old_ids']]
            # The detector predicts pitch class, not a string or octave. Keep
            # every annotated candidate rather than inventing a unique pitch.
            selected.append(('false_held',t,pc,sorted({e['midi'] for e in old}),error))
            true = [e for e in s['events'] if e['pc']==pc and e['id'] in matched and e['id'] not in seen_true and any(
                prev['pc']==pc and .096 < e['t']-prev['t'] <= 2 for prev in s['events'])]
            if true:
                e=min(true,key=lambda e:abs(e['t']-t));seen_true.add(e['id'])
                selected.append(('matched_repeat',e['t'],pc,[e['midi']],{'reference':e}))
        for kind,t,pc,midis,evidence in selected:
            number=len(cases)+1; stem=f'{number:02d}-{kind}-{sid}'
            start=max(0.,t-.8); stop=min(s['duration'],t+.4)
            first=max(0,int(np.floor((t-.24)/.016))); last=min(len(features),int(np.ceil((t+.24)/.016)))
            nearby=[e for e in s['events'] if t-.25<=e['t']<=t+.25]
            other=[e['midi'] for e in nearby if e['pc']!=pc and abs(e['t']-t)<=.064]
            case={'id':number,'kind':kind,'source':sid,'t':t,'pc':pc,'midi_candidates':midis,
                  'evidence':evidence,'annotation_context':annotation_context(s,t,pc),
                  'spectral_evidence':{str(midi):spectral_evidence(features,t,midi,other) for midi in midis},
                  'trace_times':((np.arange(first,last)+1)*.016).tolist(),
                  'probabilities':{arm:probabilities[arm][sid][first:last].tolist() for arm in probabilities},
                  'audio_start':start,'audio_stop':stop,'audio_file':None,
                  'feature_patch':stem+'.npz'}
            np.savez_compressed(output/(stem+'.npz'),features=features[first:last],times=(np.arange(first,last)+1)*.016)
            if audio_path:
                with sf.SoundFile(audio_path) as audio:
                    audio.seek(round(start*audio.samplerate))
                    wave=audio.read(round((stop-start)*audio.samplerate),dtype='float32')
                    sf.write(output/(stem+'.wav'),wave,audio.samplerate,subtype='PCM_16')
                case['audio_file']=stem+'.wav'
            cases.append(case)
        print(f"Reviewed {sid}: {len(errors[sid])} held-PC errors at .8",flush=True)
    save_json(output/'cases.json',cases)
    save_json(output/'missed_at_09.json',{'lost':lost,'gained':gained})
    result={'ok':True,'audit_only':True,'training_run':str(run),'models':{a:summary['arms'][a]['model_sha256'] for a in predictions},
            'selected_replay_verified_recordings':len(sources),'fixed_thresholds':list(AUDIT_THRESHOLDS),
            'unselected_threshold_table_differences':table_differences,
            'metrics':{f'{a}/{t}':metrics[a,t] for a in predictions for t in AUDIT_THRESHOLDS},
            'comp_held_errors_at_08':sum(len(v) for v in errors.values()),'annotation_flags_nonexclusive':dict(flags),
            'lost_comp_references_09_vs_control08':len(lost),'gained_comp_references_09_vs_control08':len(gained),
            'loss_causes':dict(Counter(e['cause'] for e in lost)),
            'review_recordings':chosen,'review_cases':len(cases),'audio_failures':audio_failures,
            'review_audio_complete':not audio_failures,
            'next_step':'Review the bounded false/true audio pairs before changing data, labels or training. Low response, latch suppression and reference competition are reported separately.',
            'limitations':['Only validation; no new threshold selection or training.',
                           'Annotations mark note starts, not verified picking technique. Flags do not prove label errors.',
                           'Spectral rise and harmonic overlap are descriptive; neither proves that the available features can or cannot separate attacks.',
                           'Lost/gained reference IDs may include different matches among closely spaced same-PC strings.',
                           'This does not measure app credits or change the live judge.']}
    # Keep the user-facing summary small: only relevant groups, not all 13 groups.
    for metric in result['metrics'].values():
        metric['groups']={k:{n:v for n,v in g.items() if not n.endswith('source_groups')}
                          for k,g in metric['groups'].items() if k in ('all','guitarset/comp','guitarset/solo','synthetic')}
    save_json(output/'provenance.json',{'inputs':hashes,'summary_sha256':sha256(training_summary_path(run))})
    save_json(output/'audit_summary.json',result)
    with zipfile.ZipFile(output/'review.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.name!='review.zip':
                archive.write(path,path.name)
    return result


def audit_main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',default=RUN_DIR)
    parser.add_argument('--work-root',type=Path,default=Path(WORK_ROOT))
    parser.add_argument('--input-root',type=Path,default=Path(INPUT_ROOT))
    args=parser.parse_args(argv)
    args.work_root.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='onset-real-audit-',dir=args.work_root))
    try:
        run=find_audit_run(args.run_dir,args.work_root,args.input_root)
        print(f'Using existing run: {run}; output: {output}',flush=True)
        result=audit_real(run,args.input_root,output)
        print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2))
    except (OSError,ValueError,KeyError,TypeError) as error:
        failure = {'ok':False,'error':str(error),'training_not_started':True}
        if isinstance(error, AuditDiscoveryError):
            failure['discovery'] = error.inventory
        save_json(output/'audit_summary.json',failure)
        print(json.dumps(failure,indent=2),flush=True)
        print(f'Audit stopped: {error}',flush=True)
    print(f'Share: {output / "audit_summary.json"}')
    print(f'Review files (if completed): {output / "review.zip"}')


if __name__=='__main__':
    audit_main([] if 'ipykernel' in sys.modules else None)
