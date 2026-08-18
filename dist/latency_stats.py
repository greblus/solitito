"""How late the app learns what was played, and how often it learns it wrong.

    ./target/release/solitito --probe probe.wav --step 1 > probe.txt
    python dist/latency_stats.py probe.txt onsets.csv

Reads the probe's per-frame table and the ground truth from
`latency_material.py` (plucks with onsets known to the sample) or
`latency_ground_truth.py` (a real recording, onsets from pyin). Reports, in the
order they decide anything:

  * LATENCY - from the strike to the first frame the app would act on, for each
    of the three answers the app can have: the model's pitch head, the
    single-frame estimate, and - where the model carries one - the onset head.
    Only a RISING edge counts: a class already lit when the note is struck says
    nothing about how fast anything noticed.
  * FALSE CREDITS - classes a rule would mark off that were not played in the
    last two seconds. This is the number the Formulas mode lives or dies by: a
    mark there never expires, so one frame is enough to light a function for the
    rest of the exercise. A sympathetic open string is SOUNDING, which is why
    "what sounds" cannot be the rule and "what was struck" can.
  * WHAT THE EAR NAMES - `mono_pitch` against the note really sounding, counted
    in semitones, which is how a partial gives itself away: +7 is the third
    harmonic, +10 the seventh, +4 the fifth.
  * A PARTLY FILLED WINDOW - what the model says before the context is full.
    The app asks nothing under 90%, which after silence is 688 ms of waiting.
"""
import csv, sys
from collections import defaultdict

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
N2P = {n: i for i, n in enumerate(NAMES)}
THR = 0.6          # the app's default note threshold
FILL_MIN = 90.0    # MIN_FILL in main.rs
STEADY = 4         # CQT_STEADY_TICKS in state.rs
ONSET_THRS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def read_probe(path):
    """(t, fill, pitches, struck, cqt) per frame. `struck` is empty for a probe
    run before the onset block existed, or with a model that has no head."""
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) < 16:
            continue
        try:
            t = float(p[0]); fill = float(p[2].rstrip("%"))
            pit = [int(x) / 100.0 for x in p[3:15]]
        except ValueError:
            continue
        rest = p[15:]
        struck = []
        if rest and rest[0].startswith("|"):
            # "|struck" columns: the first is glued to the bar by the format
            head = rest[0][1:]
            vals = ([head] if head else []) + rest[1:]
            try:
                struck = [int(x) / 100.0 for x in vals[:12]]
                rest = vals[12:]
            except ValueError:
                struck, rest = [], rest[1:]
        cqt = N2P[rest[0]] if rest and rest[0] in N2P else None
        rows.append((t, fill, pit, struck, cqt))
    return rows


def main(probe_path, onsets_path):
    rows = read_probe(probe_path)
    notes = [(float(r["t"]), int(r["pc"]), r["case"])
             for r in csv.DictReader(open(onsets_path))]
    notes.sort()
    has_struck = any(r[3] for r in rows)
    print(f"{len(rows)} frames, {len(notes)} notes  "
          f"(threshold {THR}, fill >= {FILL_MIN:.0f}%)"
          + ("" if has_struck else "  - no onset head in this run"))

    def live(a, b):
        return [r for r in rows if a <= r[0] <= b and r[1] >= FILL_MIN]

    def span(i):
        end = notes[i + 1][0] if i + 1 < len(notes) else notes[i][0] + 1.5
        return notes[i][0], end

    # --- claimed before played, and latency on the rising edge ---
    lit = 0
    lat = {"model": [], "ear": [], "struck": []}
    miss = defaultdict(int)
    for i, (t0, pc, case) in enumerate(notes):
        _, end = span(i)
        pre = live(t0 - 0.05, t0)
        if pre and pre[-1][2][pc] >= THR:
            lit += 1
            continue
        win = live(t0, min(t0 + 1.2, end + 0.35))
        for name, hit in (("model", lambda r: r[2][pc] >= THR),
                          ("ear",   lambda r: r[4] == pc),
                          ("struck", lambda r: r[3] and r[3][pc] >= THR)):
            if name == "struck" and not has_struck:
                continue
            d = next((r[0] - t0 for r in win if hit(r)), None)
            if d is None:
                miss[name] += 1
            else:
                lat[name].append(d)
    print(f"\nclaimed before being struck: {lit}/{len(notes)}")

    def q(a, f):
        return f"{1000 * sorted(a)[min(len(a) - 1, int(f * len(a)))]:.0f} ms" if a else "-"

    print("\nlatency, rising edge, frames the app would act on:")
    for name in ("model", "ear", "struck"):
        if name == "struck" and not has_struck:
            continue
        print(f"   {name:7} p50 {q(lat[name], 0.5):>8}   p90 {q(lat[name], 0.9):>8}   "
              f"never within 1.2 s: {miss[name]}")

    # --- false credits: the number the Formulas mode lives by ---
    runs, prev, run = [], object(), 0
    for r in rows:
        run = run + 1 if r[4] is not None and r[4] == prev else 1
        prev = r[4]
        runs.append(run)

    def played_near(t, within=2.0):
        return {pc for (t0, pc, _) in notes if t - within <= t0 <= t + 0.05}

    def false_credits(rule):
        """(false credits, notes affected, notes the rule never credited)."""
        seen, credited = set(), set()
        for i, (t0, pc, case) in enumerate(notes):
            _, end = span(i)
            ok = played_near(t0)
            for k, r in enumerate(rows):
                if r[0] < t0 or r[0] > min(end, t0 + 1.5) or r[1] < FILL_MIN:
                    continue
                for c in rule(r, runs[k]):
                    if c == pc:
                        credited.add(i)
                    elif c not in ok:
                        seen.add((i, c))
        return len(seen), len({i for (i, _) in seen}), len(notes) - len(credited)

    rules = [
        ("ear, every frame", lambda r, run: [r[4]] if r[4] is not None else []),
        (f"ear, held {STEADY} frames",
         lambda r, run: [r[4]] if r[4] is not None and run >= STEADY else []),
        ("model pitch >= 0.6",
         lambda r, run: [c for c in range(12) if r[2][c] >= THR]),
    ]
    if has_struck:
        rules.append(("onset head >= 0.6",
                      lambda r, run: [c for c in range(12) if r[3] and r[3][c] >= THR]))
    print(f"\nfalse credits - a class marked off that was not played in the last 2 s:")
    print(f"   {'rule':24} {'false':>6} {'notes hit':>10} {'notes never credited':>21}")
    for name, rule in rules:
        f, n, m = false_credits(rule)
        print(f"   {name:24} {f:6} {n:10} {m:21}")

    if has_struck:
        print("\n   the onset head, threshold by threshold:")
        print(f"   {'thr':>6} {'false':>6} {'notes hit':>10} {'never credited':>15}")
        for thr in ONSET_THRS:
            f, n, m = false_credits(
                lambda r, run, thr=thr: [c for c in range(12) if r[3] and r[3][c] >= thr])
            print(f"   {thr:6.2f} {f:6} {n:10} {m:15}")

    # --- what the ear names inside a sounding note ---
    lock, tot = defaultdict(int), 0
    for i, (t0, pc, case) in enumerate(notes):
        _, end = span(i)
        for r in live(t0 + 0.15, min(end, t0 + 1.2)):
            if r[4] is not None:
                tot += 1
                lock[(r[4] - pc) % 12] += 1
    if tot:
        says = {0: "the note itself", 7: "its fifth, the 3rd partial",
                10: "its minor seventh, the 7th partial",
                4: "its major third, the 5th partial", 9: "the 13th partial",
                5: "the string above, open", 1: "a semitone up", 11: "a semitone down"}
        print(f"\nwhat the ear names, over {tot} frames inside a sounding note:")
        for d in sorted(lock, key=lambda k: -lock[k])[:6]:
            print(f"   {d:2} semitones: {100 * lock[d] / tot:5.1f}%   {says.get(d, '')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
