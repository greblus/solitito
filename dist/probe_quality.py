# ==========================================
# PROBE QUALITY - where should chord quality come from?
# ==========================================
# The app used to derive quality from the pitch vector with hand-written
# thresholds while the model had a quality head all along. This compares three
# ways of deciding quality on ONE checkpoint, without retraining:
#
#   A) argmax of the quality head
#   B) template matching against the PREDICTED pitch vector
#   C) template matching against the TRUE pitch vector  <- ceiling of the method
#
# Reading:
#   B >> A  -> the architecture wastes information; derive quality from pitch
#   B ~= A  -> the pitch vector is uncertain per case despite good averages
#   C ~= A  -> even perfect pitch is not enough; the problem is in the labels
#
# It also measures temporal aggregation, i.e. what the app actually does when it
# votes over several consecutive windows.
#
# KAGGLE: put it next to model_trainer.py, paste into a cell and call main().
# ==========================================

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# The trainer installs dependencies and logs into HF at import time, and its main()
# sits behind __name__ == "__main__", so importing it is safe and gives us EXACTLY
# the same data-loading logic as training. No duplication = no divergence.
TRAINER_URL = "https://raw.githubusercontent.com/greblus/solitito/v2/dist/model_trainer.py"


def _locate_trainer():
    """Locates model_trainer.py (locally or from the repo) and adds it to sys.path."""
    import glob
    for pat in ("model_trainer.py", "/kaggle/working/model_trainer.py",
                "/kaggle/input/*/model_trainer.py", "/kaggle/input/*/*/model_trainer.py",
                "./dist/model_trainer.py", "../model_trainer.py"):
        for hit in glob.glob(pat):
            d = os.path.dirname(os.path.abspath(hit))
            if d not in sys.path:
                sys.path.insert(0, d)
            print(f"📎 model_trainer.py: {hit}")
            return True

    # Not local - fetch from the repo. Plain urllib: no shell, no wget, no
    # notebook magic (`!wget` is IPython syntax, not Python).
    print("📥 Not found locally - downloading from the repo...")
    try:
        import urllib.request
        dst = os.path.join(os.getcwd(), "model_trainer.py")
        with urllib.request.urlopen(TRAINER_URL, timeout=30) as r:
            body = r.read()
        if len(body) < 1000 or b"RUN_TAG" not in body:
            print("   ⚠️ The download does not look like the trainer - skipping.")
            return False
        with open(dst, "wb") as f:
            f.write(body)
        sys.path.insert(0, os.path.dirname(dst))
        print(f"   ✅ {dst}  ({len(body)//1024} kB)")
        return True
    except Exception as e:
        print(f"   ❌ {type(e).__name__}: {e}")
        return False


if not _locate_trainer():
    sys.exit(
        "\n❌ No model_trainer.py - the probe must import it to use EXACTLY the\n"
        "   same data-loading logic as training.\n\n"
        "   If the download failed, the notebook most likely has internet OFF\n"
        "   (Kaggle: right-hand panel -> Settings -> Internet: On).\n\n"
        "   Otherwise upload model_trainer.py to /kaggle/working or attach a\n"
        "   dataset that contains it.\n"
    )

import model_trainer as T

# A representative label per quality class. Intervals come from the trainer's OWN
# function so the template describes exactly what the model was taught (e.g. the
# trainer gives 'aug' {0,4,7,8}, a perfect and an augmented fifth at once -
# musically odd, but the probe measures the model, not the theory).
QUAL_PROBE_LABEL = {
    "maj": "", "min": "m", "maj7": "Maj7", "dom7": "7", "min7": "m7",
    "m7b5": "m7b5", "dim7": "dim7", "aug": "aug", "sus": "sus4",
    "note": "Note", "N": "N",
}


def build_templates(verbose=True):
    """[n_qual, 12, 12] -> mask per (quality, root) as a 12-pitch-class vector."""
    n = len(T.QUALITIES)
    tpl = np.zeros((n, 12, 12), dtype=np.float32)
    if verbose: print("   templates (intervals from the trainer's create_targets):")
    for qi, q in enumerate(T.QUALITIES):
        lbl = QUAL_PROBE_LABEL.get(q, "")
        ivs = set() if q == "N" else {s for s, _ in T.get_chord_intervals_with_types(lbl)}
        if verbose: print(f"     {q:<6} {sorted(ivs)}")
        for root in range(12):
            for iv in ivs:
                tpl[qi, root, (root + iv) % 12] = 1.0
    return tpl


def match(pitch_prob, roots, tpl, missing_weight=1.0):
    """Bernoulli log-likelihood per quality; returns the argmax.

    `missing_weight` scales the penalty for a template note that does NOT sound.
    At 1.0 (strict) `dom7`=[0,4,7,10] requires all four notes, and guitarists
    routinely drop the fifth. At 0.0 (lenient) only notes FOREIGN to the template
    are penalised, i.e. "does what was played fit the chord" rather than "was the
    whole chord played". The truth lies between the two readings.
    """
    p = np.clip(pitch_prob, 1e-6, 1 - 1e-6)               # [N,12]
    logp, log1p_ = np.log(p), np.log(1 - p)
    r = np.clip(roots, 0, 11)
    tsel = tpl[:, r, :]                                    # [n_qual, N, 12]
    score = (missing_weight * tsel * logp[None]
             + (1 - tsel) * log1p_[None]).sum(axis=2)      # [n_qual, N]
    return score.argmax(axis=0)


def main():
    ckpt_name = sys.argv[1] if len(sys.argv) > 1 else T.CKPT_BEST
    print(f"🧠 Checkpoint: {ckpt_name}\n")

    # --- data: exactly as in training, same seed and same split ---
    reg = T.FileRegistry(); reg.scan_all()
    data = T.load_data(reg)
    # EXACTLY the same split as training - by source, not by segment. The probe
    # used to shuffle segments, so its "validation" contained fragments of
    # recordings seen in training and inflated every method.
    import random; random.seed(42)
    groups = sorted({T.split_group_key(x) for x in data})
    random.shuffle(groups)
    n_val = max(1, int(round(len(groups) * (1.0 - T.TRAIN_FRAC))))
    val_groups = set(groups[:n_val])
    vl_items = [x for x in data if T.split_group_key(x) in val_groups]
    ds_vl = T.FrameBasedDataset(vl_items, training=False)
    loader = DataLoader(ds_vl, batch_size=T.BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"📊 Validation: {len(ds_vl)} windows from {n_val} groups (split by source)\n")

    # --- model z checkpointu ---
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo_id=T.HF_REPO_ID, filename=ckpt_name,
                            local_dir=T.WORK_DIR, token=T.hf_token)
    ck = torch.load(local, map_location=T.device, weights_only=False)
    model = T.ChordTransformer().to(T.device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    thr = ck.get("best_threshold", 0.5)
    print(f"   checkpoint epoch: {ck.get('epoch', '?')}   threshold: {thr}\n")

    # --- inferencja ---
    R, Qh, P, gR, gQ, gP, AUD, CH = [], [], [], [], [], [], [], []
    PR, PQ = [], []                       # full distributions, needed for aggregation
    with torch.no_grad():
        for x, root, qual, pitch, root_ok, qual_ok in loader:
            orr, oq, op = model(x.to(T.device))
            R.append(orr.argmax(1).cpu().numpy());  Qh.append(oq.argmax(1).cpu().numpy())
            PR.append(torch.softmax(orr, 1).cpu().numpy())
            PQ.append(torch.softmax(oq, 1).cpu().numpy())
            P.append(torch.sigmoid(op).cpu().numpy())
            gR.append(root.numpy()); gQ.append(qual.numpy()); gP.append(pitch.numpy())
            AUD.append(root_ok.numpy() > 0.5)
            CH.append(qual_ok.numpy() > 0.5)
    R, Qh, P = np.concatenate(R), np.concatenate(Qh), np.concatenate(P)
    PR, PQ = np.concatenate(PR), np.concatenate(PQ)
    gR, gQ, gP = np.concatenate(gR), np.concatenate(gQ), np.concatenate(gP)
    AUD, CH = np.concatenate(AUD), np.concatenate(CH)

    # Windows from SOLO recordings carry a chord label describing accompaniment
    # that is not in the signal. An earlier version counted them with the rest,
    # which produced the false conclusion "even perfect pitch gives only 41%".
    n_solo = int((~CH).sum())
    if n_solo:
        print(f"   skipped {n_solo} solo windows ({n_solo/len(CH):.0%}) - "
              f"the chord label does not describe the signal there\n")
    AUD = AUD & CH

    tpl = build_templates()
    A = Qh                                   # quality head
    B = match(P,  gR, tpl)                   # templates + predicted pitch (true root)
    C = match(gP, gR, tpl)                   # templates + TRUE pitch = ceiling
    Bp = match(P, R, tpl)                    # templates + everything predicted
    # D: the same ceiling as C but without penalising notes the guitarist omitted.
    # C and D bracket the range - C understates (it demands the full chord),
    # D overstates (it cannot tell a chord from a subset of it).
    D = match(gP, gR, tpl, missing_weight=0.0)

    def acc(pred, m=None):
        m = CH if m is None else m           # default: only windows with a real label
        return float((pred[m] == gQ[m]).mean()) if m.any() else 0.0
    print("=" * 76)
    print(f"{'method for deciding quality':<44} {'overall':>9} {'root audible':>14}")
    print("-" * 76)
    for lbl, pred in [("A) argmax of the quality head (current)", A),
                      ("B) templates + predicted pitch",          B),
                      ("   (+ predicted root)",                   Bp),
                      ("C) templates + TRUE pitch, strict",       C),
                      ("D) as above, no penalty for omitted notes", D)]:
        print(f"{lbl:<44} {acc(pred):>8.1%} {acc(pred, AUD):>13.1%}")
    print("=" * 76)
    print(f"   windows with an audible root: {AUD.sum()/max(CH.sum(),1):.0%}")
    print("   B >> A  -> derive quality from pitch rather than from CLS")
    print("   B ~= A  -> the pitch vector is uncertain per case despite good averages")
    print("   C ~= A  -> even perfect pitch is not enough; the labels are the problem")

    # --- per quality, to see WHICH classes gain ---
    print(f"\n{'quality':<8} {'n':>6} {'A head':>11} {'B templates':>12} {'C strict':>9} {'D lenient':>11}")
    print("-" * 52)
    for qi, q in enumerate(T.QUALITIES):
        m = (gQ == qi) & CH
        if m.sum() < 20: continue
        print(f"{q:<8} {m.sum():>6} {(A[m]==qi).mean():>10.0%} "
              f"{(B[m]==qi).mean():>11.0%} {(C[m]==qi).mean():>8.0%} "
              f"{(D[m]==qi).mean():>10.0%}")

    # --- BY SOURCE: synthetic (labels certain) vs GuitarSet (possible noise) ---
    # The loader has shuffle=False, so prediction order matches ds_vl.samples.
    rev = {cp: path for path, (cp, _) in ds_vl.cache_map.items()}
    is_synth = np.array([
        "synth_" in os.path.basename(rev.get(s["npy_path"], "")).lower()
        for s in ds_vl.samples], dtype=bool)
    if len(is_synth) == len(gQ) and is_synth.any() and (~is_synth).any():
        print(f"\n{'source':<12} {'n':>6} {'A':>7} {'B':>7} {'C':>7}   pitch recall b7 / 7")
        print("-" * 62)
        IVN = T.IV_NAMES
        for name, m in [("synthetic", is_synth & CH), ("GuitarSet", (~is_synth) & CH)]:
            if m.sum() < 20: continue
            # seventh recall, relative to the root
            rec = {}
            for iv in (10, 11):
                idx = (gR[m] + iv) % 12
                tgt = gP[m][np.arange(m.sum()), idx] > 0.5
                hit = (P[m][np.arange(m.sum()), idx] > thr) & tgt
                rec[IVN[iv]] = hit.sum() / max(tgt.sum(), 1)
            print(f"{name:<12} {m.sum():>6} {(A[m]==gQ[m]).mean():>6.0%} "
                  f"{(B[m]==gQ[m]).mean():>6.0%} {(C[m]==gQ[m]).mean():>6.0%}   "
                  f"b7={rec['b7']:.0%}  7={rec['7']:.0%}")
        print("\n  If synthetic scores MUCH higher on A/B and seventh recall than")
        print("  GuitarSet, the ceiling is GuitarSet's labels (played voicings !=")
        print("  written chord), not the model. Then: weight synthetic more, or")
        print("  loosen the targets for GS.")
    else:
        print("\n  (could not separate sources - check the synthetic file naming)")

    # --- werdykt ---
    print("\n" + "=" * 68)
    a, b, c = acc(A), acc(B), acc(C)
    if b > a + 0.05:
        print(f"→ TEMPLATES BEAT THE HEAD by {b-a:+.1%}. The architecture wastes")
        print("  information the pitch head already extracts. Make quality depend on")
        print("  pitch (or compute it from the pitch vector directly).")
    elif c < 0.85:
        print(f"→ Even with PERFECT pitch the templates only reach {c:.1%}.")
        print("  The problem is not the model - the quality labels are not determined")
        print("  by the set of sounding notes (label noise / voicings).")
    else:
        print(f"→ The head ({a:.1%}) matches the templates ({b:.1%}), ceiling {c:.1%}.")
        print("  The pitch vector is uncertain per case despite good averages - the")
        print("  bottleneck is detecting individual notes, not decoding them.")

    temporal_aggregation(ds_vl, PR, PQ, gR, gQ, AUD, CH)


# ==========================================
# TEMPORAL AGGREGATION - does voting over several windows rescue quality?
# ==========================================
# The app does not decide from one window: main.rs votes over several consecutive
# predictions and state.rs has a tolerance. A per-window metric therefore
# understates what the user actually sees. This measures what the app really does.
#
# Dwie metody:
#   voting    - majority over the argmaxes       (current main.rs behaviour)
#   averaging - argmax of the mean distribution  (usually better, same cost)
#
# Reading: if exact rises from 68% to ~85% at a ~1 s window, the model is ready and
# the work moves into the app. If only to ~72%, we go back to the data.

AGG_K = [1, 3, 5, 7, 9, 13]


def contiguous_runs(samples, max_gap, valid=None):
    """Runs of windows from one file, one chord, adjacent in time.

    Voting must not cross a chord change, so a run breaks on a file change, a
    label change, or a gap larger than max_gap frames (the energy gate drops
    single windows, hence a tolerance rather than strict adjacency).
    """
    runs, cur = [], []
    for i, s in enumerate(samples):
        if valid is not None and not valid[i]:
            # a window without a meaningful chord label (solo) breaks the run
            if cur: runs.append(cur); cur = []
            continue
        if cur:
            p = samples[cur[-1]]
            same = (s['npy_path'] == p['npy_path'] and s['root'] == p['root']
                    and s['qual'] == p['qual']
                    and 0 < s['frame_idx'] - p['frame_idx'] <= max_gap)
            if not same:
                runs.append(cur); cur = []
        cur.append(i)
    if cur: runs.append(cur)
    return runs


def temporal_aggregation(ds_vl, PR, PQ, gR, gQ, AUD, CH):
    import model_trainer as T

    stride_s = 16 * T.HOP_LENGTH / T.SR          # val: stride=16 klatek
    win_s    = T.CTX_FRAMES * T.HOP_LENGTH / T.SR
    runs = contiguous_runs(ds_vl.samples, max_gap=40, valid=CH)

    print("\n" + "=" * 76)
    print("TEMPORAL AGGREGATION - the way the app decides")
    print("=" * 76)
    print(f"   runs (one chord, adjacent windows): {len(runs)}   "
          f"median length: {int(np.median([len(r) for r in runs]))} windows")
    # CRUCIAL: for every K we also compute the SINGLE-window score on EXACTLY the
    # same centre windows. Without that we would compare K=1 over all 1971 windows
    # against K=13 over the 131 windows from long segments - and long segments are a
    # different, easier population (the chord rings longer, the root is audible more
    # often). The aggregation gain may only be read within a single row.
    print(f"{'windows':>7} {'time':>7} {'n':>6} │ {'exact: 1 win':>14} {'voting':>9} "
          f"{'averaging':>10} │ {'same, root audible':>26}")
    print("-" * 92)

    summary = []
    for K in AGG_K:
        # [single, voting, averaging] - exact hits
        tot = np.zeros(3, dtype=np.int64); n = 0
        tot_a = np.zeros(3, dtype=np.int64); n_a = 0
        for run in runs:
            if len(run) < K: continue
            for a in range(len(run) - K + 1):
                idx = run[a: a + K]
                c   = idx[K // 2]                        # label and flag from the centre
                preds = [
                    (int(PR[c].argmax()), int(PQ[c].argmax())),
                    (int(np.bincount(PR[idx].argmax(axis=1), minlength=PR.shape[1]).argmax()),
                     int(np.bincount(PQ[idx].argmax(axis=1), minlength=PQ.shape[1]).argmax())),
                    (int(PR[idx].mean(axis=0).argmax()),
                     int(PQ[idx].mean(axis=0).argmax())),
                ]
                hits = np.array([(rp == gR[c]) and (qp == gQ[c]) for rp, qp in preds])
                tot += hits; n += 1
                if AUD[c]:
                    tot_a += hits; n_a += 1
        if not n:
            print(f"{K:>7} {'-':>7} {0:>6}   (too few runs of this length)")
            continue
        span = win_s + (K - 1) * stride_s
        e  = tot / n
        ea = tot_a / max(n_a, 1)
        print(f"{K:>6} {span:>6.2f}s {n:>6} │ {e[0]:>13.1%} {e[1]:>9.1%} {e[2]:>9.1%} │ "
              f"{ea[0]:>8.1%} {ea[1]:>8.1%} {ea[2]:>8.1%}")
        summary.append((K, span, ea[0], max(ea[1], ea[2])))

    if summary:
        print("-" * 92)
        K, span, base, best = summary[-1]
        gain = best - base
        print(f"   On the same windows (root audible, {span:.2f} s): "
              f"{base:.1%} -> {best:.1%}  ({gain:+.1%})")
        print()
        if best >= 0.85:
            print(f"   -> Agregacja podnosi exact do {best:.1%}. Model jest gotowy;")
            print("      the remaining work is in the app (decision window length).")
        elif gain >= 0.08:
            print(f"   -> The {gain:+.1%} gain is real, but only to {best:.1%}, not 85%.")
            print("      Worth widening the decision window in the app AND fixing data.")
        else:
            print(f"   -> Only {gain:+.1%} gain. The errors are CORRELATED in time:")
            print("      the model makes the same mistake in consecutive windows, so")
            print("      voting fixes nothing. The bottleneck is the data, not the")
            print("      decision rule.")


if __name__ == "__main__":
    main()
