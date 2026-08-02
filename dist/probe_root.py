# ==========================================
# PROBE ROOT - can the root be heard in a 0.77 s window at all?
# ==========================================
# The model memorises quality on the training set (98%) but not the root (83%).
# When a model with full capacity cannot memorise a label, the same audio must be
# carrying different root labels - i.e. the label is partly not derivable from the
# signal, whatever the architecture.
#
# This computes a CEILING for the root head from the ANNOTATIONS ALONE (no audio,
# no model, no training - seconds of CPU):
#
#   1. How often the labelled root actually sounds in the window (note_midi).
#      That is the upper bound for any model that listens rather than guesses.
#   2. How often the root is the lowest sounding note (bass).
#   3. Whether the INTENDED and PLAYED annotations agree on the root.
#   4. How the ceiling changes with a longer window (48 / 96 / 144 frames).
#
# Reading:
#   ceiling ~= 100%        -> the root is audible; the model is at fault
#   ceiling ~= 85%         -> the model already extracts nearly everything audible
#   ceiling grows with the window -> widen CTX_FRAMES
#   ceiling flat           -> the root is simply not played; it would have to come
#                             from context rather than from the window
#
# KAGGLE: put it next to model_trainer.py (or let it download), paste, main().
# ==========================================

import glob
import os
import sys
from collections import Counter, defaultdict

TRAINER_URL = "https://raw.githubusercontent.com/greblus/solitito/v2/dist/model_trainer.py"

# Window sizes to measure. 48 frames = 0.768 s = the current CTX_FRAMES.
WINDOW_SIZES = [48, 96, 144]
STRIDE       = 16          # frames between successive windows when scanning


def _locate_trainer():
    """Locates model_trainer.py (locally or from the repo) and adds it to sys.path."""
    for pat in ("model_trainer.py", "/kaggle/working/model_trainer.py",
                "/kaggle/input/*/model_trainer.py", "/kaggle/input/*/*/model_trainer.py",
                "./dist/model_trainer.py", "../model_trainer.py"):
        for hit in glob.glob(pat):
            d = os.path.dirname(os.path.abspath(hit))
            if d not in sys.path:
                sys.path.insert(0, d)
            print(f"📎 model_trainer.py: {hit}")
            return True

    print("📥 Not found locally - downloading from the repo...")
    try:
        import urllib.request
        dst = os.path.join(os.getcwd(), "model_trainer.py")
        with urllib.request.urlopen(TRAINER_URL, timeout=30) as r:
            body = r.read()
        if len(body) < 10000:
            print("   ⚠️ The download is suspiciously small - check the URL/branch.")
            return False
        open(dst, "wb").write(body)
        sys.path.insert(0, os.getcwd())
        print(f"   ✓ {dst} ({len(body)//1024} kB)")
        return True
    except Exception as e:
        print(f"   ❌ {e}")
        return False


def find_jams():
    out = []
    for base in ("/kaggle/input", ".", "/kaggle/working"):
        if not os.path.isdir(base):
            continue
        for r, _, files in os.walk(base):
            for f in files:
                if f.endswith(".jams"):
                    out.append(os.path.join(r, f))
    return sorted(set(out))


def main():
    if not _locate_trainer():
        sys.exit("❌ No model_trainer.py - without it the label parser would diverge.")

    import json
    import numpy as np
    import model_trainer as T

    jams = find_jams()
    if not jams:
        sys.exit("❌ No .jams files found - is GuitarSet attached?")
    print(f"🔍 JAMS files: {len(jams)}")
    print(f"   base window: {T.CTX_FRAMES} frames = "
          f"{T.CTX_FRAMES * T.HOP_LENGTH / T.SR:.3f} s   "
          f"(note coverage >= {T.NOTE_MIN_COVER:.0%})\n")

    hop_s = T.HOP_LENGTH / T.SR
    maxw  = max(WINDOW_SIZES)

    # counter[window_size] -> [hits, total]
    audible   = {w: [0, 0] for w in WINDOW_SIZES}
    is_bass   = {w: [0, 0] for w in WINDOW_SIZES}
    too_short = {w: 0 for w in WINDOW_SIZES}         # segments shorter than the window
    # GuitarSet has a COMP (chords) and a SOLO (improvisation over the same
    # progression) version of every excerpt, with the SAME chord annotation.
    # Mixing them understates root audibility - a solo sounds one note at a time.
    by_kind = {"comp": [0, 0], "solo": [0, 0]}
    by_fam    = defaultdict(lambda: [0, 0])          # base-window ceiling by chord family
    root_disagree = [0, 0]                           # intended vs played
    disagree_ex   = Counter()
    files_used    = 0
    no_notes      = 0

    for p in jams:
        try:
            j = json.load(open(p))
        except Exception:
            continue

        # --- notes actually played: (start, end, midi) ---
        notes = []
        chords = {"instructed": [], "performed": []}
        for a in j.get("annotations", []):
            ns = a.get("namespace", "")
            if ns == "note_midi":
                for o in T.jams_observations(a):
                    t, dur, v = o.get("time"), o.get("duration"), o.get("value")
                    if t is None or v is None:
                        continue
                    notes.append((float(t), float(t) + float(dur or 0.0), float(v)))
            elif ns == "chord":
                src = str(a.get("annotation_metadata", {}).get("data_source", "")).lower()
                key = "performed" if "transcription" in src else "instructed"
                for o in T.jams_observations(a):
                    t, dur, v = o.get("time"), o.get("duration"), o.get("value")
                    if t is None or v is None:
                        continue
                    r, q = T.parse_raw(str(v))
                    if not r:
                        continue
                    r = T.NORM_MAP.get(r, r)
                    if r not in T.ROOTS:
                        continue
                    chords[key].append((float(t), float(t) + float(dur or 0.0), r, q))

        if not chords["instructed"]:
            continue
        files_used += 1
        if not notes:
            no_notes += 1
            continue

        # --- frame grid ---
        # pres[f, pc] = 1 when pitch class pc sounds in frame f
        # low[f]      = lowest sounding midi in frame f (for the "root = bass" test)
        t_end = max(max(e for _, e, _ in notes),
                    max(e for _, e, _, _ in chords["instructed"]))
        n_fr  = int(t_end / hop_s) + maxw + 2
        pres  = np.zeros((n_fr, 12), dtype=np.int32)
        low   = np.full(n_fr, np.inf, dtype=np.float64)
        for t0, t1, midi in notes:
            f0 = max(0, int(t0 / hop_s))
            f1 = min(n_fr, int(np.ceil(t1 / hop_s)))
            if f1 <= f0:
                continue
            pres[f0:f1, int(round(midi)) % 12] = 1
            np.minimum(low[f0:f1], midi, out=low[f0:f1])
        # cum[f] = frames < f in which pc sounded -> window coverage in O(1)
        cum = np.vstack([np.zeros((1, 12), np.int32), np.cumsum(pres, axis=0)])

        # --- 3. root agreement: intended vs played, frame by frame ---
        if chords["performed"]:
            def root_at(seq, t):
                for t0, t1, r, _ in seq:
                    if t0 <= t < t1:
                        return r
                return None
            for f in range(0, n_fr - maxw, STRIDE):
                t = f * hop_s
                ri, rp = root_at(chords["instructed"], t), root_at(chords["performed"], t)
                if ri is None or rp is None:
                    continue
                root_disagree[1] += 1
                if ri != rp:
                    root_disagree[0] += 1
                    disagree_ex[f"{ri}->{rp}"] += 1

        # --- 1./2./4. root audibility ceiling for the INTENDED label ---
        # The window must fit ENTIRELY inside the chord segment, otherwise we would
        # count notes from the neighbouring chord and longer windows would look
        # better by accident.
        for t0, t1, r, q in chords["instructed"]:
            root_pc = T.ROOTS.index(r)
            f0, f1 = int(t0 / hop_s), int(t1 / hop_s)
            for w in WINDOW_SIZES:
                if f1 - f0 < w:
                    too_short[w] += 1
                    continue
                for f in range(f0, f1 - w + 1, STRIDE):
                    cover   = cum[f + w] - cum[f]                 # (12,) frames per pitch class
                    present = np.nonzero(cover >= T.NOTE_MIN_COVER * w)[0]
                    if not len(present):
                        continue
                    audible[w][1] += 1
                    hit = root_pc in present
                    audible[w][0] += hit
                    lo = low[f:f + w].min()
                    if np.isfinite(lo):
                        is_bass[w][1] += 1
                        is_bass[w][0] += (int(round(lo)) % 12 == root_pc)
                    if w == T.CTX_FRAMES:
                        fam = T.get_family(q)
                        by_fam[fam][1] += 1
                        by_fam[fam][0] += hit
                        kind = "solo" if "_solo" in os.path.basename(p).lower() else "comp"
                        by_kind[kind][1] += 1
                        by_kind[kind][0] += hit

    def pct(a):
        return f"{100.0 * a[0] / a[1]:.1f}%" if a[1] else "  n/d"

    print("=" * 68)
    print("1./4. IS THE LABELLED ROOT AUDIBLE IN THE WINDOW AT ALL?")
    print("=" * 68)
    print(f"{'window':>6} {'time':>8} {'windows':>9} {'audible':>11} {'= bass':>8} {'segs too short':>18}")
    print("-" * 68)
    for w in WINDOW_SIZES:
        mark = "  <- CTX_FRAMES" if w == T.CTX_FRAMES else ""
        print(f"{w:>6} {w*hop_s:>7.2f}s {audible[w][1]:>9} "
              f"{pct(audible[w]):>11} {pct(is_bass[w]):>8} {too_short[w]:>18}{mark}")
    print("\n   'segs too short' = chords shorter than the window, skipped. When that")
    print("   number grows with the window, a longer frame would span a chord change.")

    print(f"\n{'='*68}\n1b. ACCOMPANIMENT vs IMPROVISATION (window {T.CTX_FRAMES})\n{'='*68}")
    for kind, a in (("comp (chords)", by_kind["comp"]), ("solo (improvisation)", by_kind["solo"])):
        print(f"   {kind:<22} root audible: {pct(a):>7}   ({a[1]} windows)")
    print("   A solo file carries a monophonic line while the label describes the")
    print("   accompaniment chord; counting them together lowers the ceiling.")

    print(f"\n{'='*68}\n2. CEILING BY CHORD FAMILY (window {T.CTX_FRAMES})\n{'='*68}")
    for fam, a in sorted(by_fam.items(), key=lambda kv: -kv[1][1]):
        print(f"   {str(fam):<16} {pct(a):>7}   ({a[1]} windows)")

    print(f"\n{'='*68}\n3. ROOT: INTENDED vs PLAYED\n{'='*68}")
    if root_disagree[1]:
        d = 100.0 * root_disagree[0] / root_disagree[1]
        print(f"   disagreements: {root_disagree[0]}/{root_disagree[1]} = {d:.1f}%")
        if disagree_ex:
            print("   most common: " +
                  "  ".join(f"{k}:{v}" for k, v in disagree_ex.most_common(8)))
    else:
        print("   no 'performed' annotations to compare against")

    print(f"\n{'='*68}\nVERDICT\n{'='*68}")
    base = audible[T.CTX_FRAMES]
    ceil = 100.0 * base[0] / base[1] if base[1] else 0.0
    print(f"   files with chords: {files_used}   without note_midi: {no_notes}")
    print(f"   CEILING for a {T.CTX_FRAMES}-frame window ({T.CTX_FRAMES*hop_s:.2f} s): {ceil:.1f}%")
    print(f"   TRAIN Root in v2_take2 (epoch 36):             83.0%")
    if ceil < 88:
        print("\n   -> The model already extracts nearly everything audible.")
        print("      The bottleneck is NOT capacity or regularisation but that the")
        print("      root is simply not played. Either a wider window, or a root")
        print("      derived from context rather than from one frame.")
    else:
        print("\n   -> The root is audible far more often than the model gets it.")
        print("      The ceiling is not the limit; the problem is model/training.")
    gain = (100.0 * audible[maxw][0] / audible[maxw][1] - ceil) if audible[maxw][1] else 0.0
    print(f"   gain from a {maxw}-frame window ({maxw*hop_s:.2f} s): {gain:+.1f} pp")


if __name__ == "__main__":
    main()
