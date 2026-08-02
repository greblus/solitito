# ==========================================
# INSPECT JAMS - what the GuitarSet annotations actually contain
# ==========================================
# probe_quality.py reported:
#   synthetic (labels certain):     quality 100%, seventh recall 100%
#   GuitarSet (chord from chart):   quality  61%, seventh recall 32%/20%
#
# Suspicion: the GuitarSet chord annotation describes the INTENDED chord, while
# a training window is 0.77 s, in which only part of the voicing may have sounded.
# The model then learns to detect notes that are not in the signal.
#
# GuitarSet was recorded with a hexaphonic pickup (one track per string), so the
# JAMS files may carry NOTE annotations - what actually sounded.
# This script checks which namespaces exist and whether exact pitch targets can
# be built from them.
#
# KAGGLE (CPU, seconds): paste into a cell and call main().
# ==========================================

import glob
import json
import os
import sys
from collections import Counter, defaultdict

INPUT_DIR = "/kaggle/input"


def find_jams(limit=None):
    out = []
    for base in [INPUT_DIR, ".", "/kaggle/working"]:
        if not os.path.isdir(base): continue
        for r, _, files in os.walk(base):
            for f in files:
                if f.endswith(".jams"):
                    out.append(os.path.join(r, f))
    out = sorted(set(out))
    return out[:limit] if limit else out


def observations(a):
    """JAMS stores 'data' either as a LIST of observations or as a DICT of columns
    (time/duration/value/confidence). Normalised here to a list of dicts."""
    d = a.get("data")
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        keys = ("time", "duration", "value", "confidence")
        cols = {k: d.get(k) or [] for k in keys}
        n = max((len(v) for v in cols.values() if isinstance(v, list)), default=0)
        return [{k: (cols[k][i] if i < len(cols[k]) else None) for k in keys}
                for i in range(n)]
    return []


def main():
    jams = find_jams()
    if not jams:
        sys.exit("❌ No .jams files found - is the GuitarSet dataset attached?")
    print(f"🔍 JAMS files: {len(jams)}\n")

    # --- which namespaces occur, and how often ---
    ns_count, ns_obs, ns_src = Counter(), defaultdict(int), defaultdict(Counter)
    for p in jams:
        try:
            j = json.load(open(p))
        except Exception:
            continue
        for a in j.get("annotations", []):
            ns = a.get("namespace", "?")
            ns_count[ns] += 1
            ns_obs[ns] += len(observations(a))
            ns_src[ns][str(a.get("annotation_metadata", {}).get("data_source", "?"))] += 1

    print(f"{'namespace':<20} {'adnotacji':>10} {'obserwacji':>12}   data_source")
    print("-" * 74)
    for ns, c in ns_count.most_common():
        srcs = ", ".join(f"{s}×{n}" for s, n in ns_src[ns].most_common(4))
        print(f"{ns:<20} {c:>10} {ns_obs[ns]:>12}   {srcs}")

    # --- a sample from each namespace ---
    print("\n" + "=" * 74)
    print("SAMPLE OBSERVATIONS")
    print("=" * 74)
    shown = set()
    for p in jams:
        try:
            j = json.load(open(p))
        except Exception:
            continue
        for a in j.get("annotations", []):
            ns = a.get("namespace", "?")
            obs = observations(a)
            if ns in shown or not obs:
                continue
            shown.add(ns)
            meta = a.get("annotation_metadata", {})
            print(f"\n── {ns}   (data_source: {meta.get('data_source','?')}, "
                  f"form: {'list' if isinstance(a.get('data'), list) else 'column dict'}, "
                  f"obserwacji: {len(obs)})")
            for o in obs[:3]:
                t, d, v = o.get("time"), o.get("duration"), o.get("value")
                ts = f"{t:.3f}" if isinstance(t, (int, float)) else str(t)
                ds = f"{d:.3f}" if isinstance(d, (int, float)) else str(d)
                print(f"     t={ts:<9} dur={ds:<8} value={str(v)[:60]}")
        if len(shown) == len(ns_count):
            break

    # --- how many DIFFERENT chord annotations a single file carries ---
    print("\n" + "=" * 74)
    print("CHORD ANNOTATIONS IN ONE FILE")
    print("=" * 74)
    try:
        j = json.load(open(jams[0]))
        print(f"  file: {os.path.basename(jams[0])}")
        for a in j.get("annotations", []):
            if "chord" not in a.get("namespace", ""):
                continue
            meta = a.get("annotation_metadata", {})
            obs = observations(a)
            print(f"    namespace={a['namespace']:<12} data_source="
                  f"{str(meta.get('data_source','?')):<12} obserwacji={len(obs):<5} "
                  f"annotator={str(meta.get('annotator', {}))[:40]}")
            for o in obs[:2]:
                print(f"        t={o.get('time')}  dur={o.get('duration')}  {o.get('value')}")
    except Exception as e:
        print(f"  ⚠️ {e}")

    # --- verdict: can targets be built from notes that ACTUALLY sounded? ---
    print("\n" + "=" * 70)
    note_ns = [n for n in ns_count if "note" in n or "pitch" in n or "midi" in n]
    chord_ns = [n for n in ns_count if "chord" in n]
    print(f"  NOTE namespaces:   {note_ns or 'NONE'}")
    print(f"  CHORD namespaces:  {chord_ns or 'NONE'}")
    if note_ns:
        print("\n  ✅ Note annotations present - pitch targets can be built from the")
        print("     notes that ACTUALLY sounded in each window, instead of from the")
        print("     chart. That removes the main source of label noise.")
        print("     Quality still comes from the chord annotation, which describes")
        print("     the harmony; pitch comes from the notes actually played.")
    else:
        print("\n  ⚠️ No note annotations. Pitch targets have to come from the chord,")
        print("     so the label noise stays. Weighting the synthetic set higher or")
        print("     extending it with more tones is then the realistic option.")
    if len(chord_ns) > 1:
        print(f"\n  ℹ️ More than one chord annotation: {chord_ns}")
        print("     If one of them describes the chord AS PLAYED, use that one.")
        print("     The trainer currently takes every namespace=='chord' alike.")


if __name__ == "__main__":
    main()
