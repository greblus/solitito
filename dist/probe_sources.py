# ==========================================
# PROBE SOURCES - which GuitarSet chord annotation to use?
# ==========================================
# Observed while playing (Autumn Leaves): the only chord the model struggles
# with is Gm7 voiced from the A string, read as Gm. It loses the minor seventh
# even though the pitch head reports b7 at 95-99%.
#
# Suspicion: the min7 class has NOT ONE example from a real guitar in training.
# The trainer uses `GUITARSET_CHORD_SOURCE = "instructed"`, the INTENDED chord,
# and GuitarSet simplifies the harmony there. If so, min7 and maj7 come only from
# the two synthetic renders - which would explain both the flattering 100% on
# validation (same instrument on both sides) and the collapse on a real
# instrument.
#
# The script reads ONLY THE ANNOTATIONS (no audio, no model, seconds) and
# compares the quality distribution of both sources:
#   "instructed" - chord from the chart, data_source ""
#   "performed"  - chord as played, data_source "...transcription..."
#
# Reading:
#   performed has min7/maj7, instructed does not -> switching gives the model
#                                                its first real sevenths
#   neither has them -> sevenths must come from somewhere else (more synthetic
#                       tones)
#
# KAGGLE: put it next to model_trainer.py (or let it download), paste, main().
# ==========================================

import glob
import json
import os
import sys
from collections import Counter, defaultdict

TRAINER_URL = "https://raw.githubusercontent.com/greblus/solitito/v2/dist/model_trainer.py"


def _locate_trainer():
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
        if len(body) < 1000 or b"RUN_TAG" not in body:
            print("   ⚠️ The downloaded file does not look like the trainer.")
            return False
        open(dst, "wb").write(body)
        sys.path.insert(0, os.path.dirname(dst))
        return True
    except Exception as e:
        print(f"   ❌ {type(e).__name__}: {e}")
        return False


def find_jams():
    out = []
    for base in ("/kaggle/input", ".", "/kaggle/working"):
        if not os.path.isdir(base):
            continue
        for r, _, files in os.walk(base):
            out += [os.path.join(r, f) for f in files if f.endswith(".jams")]
    return sorted(set(out))


def main():
    if not _locate_trainer():
        sys.exit("❌ Without model_trainer.py the label parser would diverge.")
    import model_trainer as T

    jams = find_jams()
    if not jams:
        sys.exit("❌ No .jams files found - is GuitarSet attached?")
    print(f"🔍 JAMS files: {len(jams)}\n")

    # counter[source][quality] -> segments; comp and solo counted separately
    qual = {"instructed": Counter(), "performed": Counter()}
    qual_comp = {"instructed": Counter(), "performed": Counter()}
    raw_examples = defaultdict(Counter)
    unparsed = defaultdict(Counter)

    for p in jams:
        is_solo = "_solo" in os.path.basename(p).lower()
        try:
            j = json.load(open(p))
        except Exception:
            continue
        for a in j.get("annotations", []):
            if a.get("namespace") != "chord":
                continue
            src = str(a.get("annotation_metadata", {}).get("data_source", "")).lower()
            key = "performed" if "transcription" in src else "instructed"
            for o in T.jams_observations(a):
                v = o.get("value")
                if v is None:
                    continue
                r, q = T.parse_raw(str(v))
                if not r:
                    unparsed[key][str(v)[:24]] += 1
                    continue
                qi = T.get_quality(q)
                qual[key][qi] += 1
                if not is_solo:
                    qual_comp[key][qi] += 1
                raw_examples[qi][str(v)[:24]] += 1

    hdr = f"{'quality':<8} {'instructed':>12} {'performed':>12} │ {'comp only:':>12} {'instr.':>8} {'perf.':>8}"
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for q in T.QUALITIES:
        i_all, p_all = qual["instructed"][q], qual["performed"][q]
        i_c, p_c = qual_comp["instructed"][q], qual_comp["performed"][q]
        if i_all == p_all == 0:
            continue
        mark = "  ⬅ ABSENT from instructed" if i_all == 0 and p_all > 0 else ""
        print(f"{q:<8} {i_all:>12} {p_all:>12} │ {'':>12} {i_c:>8} {p_c:>8}{mark}")
    print("=" * len(hdr))

    print("\nSAMPLE RAW LABELS (where each quality comes from):")
    for q in ("min7", "maj7", "dom7", "m7b5"):
        ex = raw_examples.get(q)
        if ex:
            print(f"   {q:<6} " + "  ".join(f"{k}×{v}" for k, v in ex.most_common(5)))
        else:
            print(f"   {q:<6} (none)")

    if unparsed:
        print("\nUNPARSED LABELS (rejected by the parser):")
        for key, c in unparsed.items():
            top = "  ".join(f"{k}×{v}" for k, v in c.most_common(5))
            print(f"   {key:<11} {sum(c.values()):>6} items:  {top}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    gain = []
    for q in T.QUALITIES:
        if qual["instructed"][q] == 0 and qual["performed"][q] > 0:
            gain.append((q, qual["performed"][q], qual_comp["performed"][q]))
    if gain:
        print("   Classes the model has NEVER SEEN on a real guitar, but which")
        print("   the 'performed' annotation provides:")
        for q, n, nc in gain:
            print(f"      {q:<6} +{n} segments ({nc} of them accompaniment)")
        print("\n   -> GUITARSET_CHORD_SOURCE = 'performed' gives them their first")
        print("      real examples. This is exactly the m7-read-as-m problem.")
    else:
        print("   Both annotations cover the same classes - switching the source")
        print("   adds no new quality. Sevenths have to come from extending the")
        print("   synthetic set: more tones, more fretboard positions.")


if __name__ == "__main__":
    main()
