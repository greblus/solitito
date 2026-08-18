"""What each crediting rule would cost, on a recording with known onsets.

    python dist/latency_rules.py probe.txt onsets.csv

A rule in the Formulas mode is judged by two numbers, and they pull apart:
FALSE CREDITS - classes marked off that nobody played, which never expire and so
end the exercise - and NOTES MISSED, which merely cost another strike. This
prints both for every rule worth arguing about, so the argument is settled by
the recording rather than by whoever speaks last.
"""
import csv, sys
sys.path.insert(0, "dist")
from latency_stats import read_probe, THR, FILL_MIN, STEADY
from collections import defaultdict

rows = read_probe(sys.argv[1])
notes = sorted((float(r["t"]), int(r["pc"])) for r in csv.DictReader(open(sys.argv[2])))

# rolling max of the onset head over the last N frames: "something was struck
# just now", which is the question the head answers best
def rolling(n):
    out, buf = [], []
    for r in rows:
        buf.append(r[3] if r[3] else [0.0] * 12)
        if len(buf) > n: buf.pop(0)
        out.append([max(b[c] for b in buf) for c in range(12)])
    return out
recent6 = rolling(6)

runs, prev, run = [], object(), 0
for r in rows:
    run = run + 1 if r[4] is not None and r[4] == prev else 1
    prev = r[4]; runs.append(run)

def played_near(t, within=2.0):
    return {pc for (t0, pc) in notes if t - within <= t0 <= t + 0.05}

def score(rule):
    seen, credited, lat = set(), set(), []
    for i, (t0, pc) in enumerate(notes):
        end = notes[i+1][0] if i+1 < len(notes) else t0 + 1.5
        ok = played_near(t0)
        first = None
        for k, r in enumerate(rows):
            if r[0] < t0 or r[0] > min(end, t0 + 1.5) or r[1] < FILL_MIN: continue
            for c in rule(r, runs[k], recent6[k]):
                if c == pc:
                    credited.add(i)
                    if first is None: first = r[0] - t0
                elif c not in ok:
                    seen.add((i, c))
        if first is not None: lat.append(first)
    lat.sort()
    p50 = f"{1000*lat[len(lat)//2]:.0f}" if lat else "-"
    return len(seen), len({i for (i, _) in seen}), len(notes) - len(credited), p50

RULES = [
    ("ear, held 4",              lambda r, run, rc: [r[4]] if r[4] is not None and run >= STEADY else []),
    ("onset >= 0.6",             lambda r, run, rc: [c for c in range(12) if r[3] and r[3][c] >= 0.6]),
    ("ear held 4 + any onset .5",lambda r, run, rc: [r[4]] if r[4] is not None and run >= STEADY and max(rc) >= 0.5 else []),
    ("ear held 4 + onset[c] .3", lambda r, run, rc: [r[4]] if r[4] is not None and run >= STEADY and rc[r[4]] >= 0.3 else []),
    ("ear held 2 + onset[c] .4", lambda r, run, rc: [r[4]] if r[4] is not None and run >= 2 and rc[r[4]] >= 0.4 else []),
    ("ear any + onset[c] .5",    lambda r, run, rc: [r[4]] if r[4] is not None and rc[r[4]] >= 0.5 else []),
    ("ear any + onset[c] .7",    lambda r, run, rc: [r[4]] if r[4] is not None and rc[r[4]] >= 0.7 else []),
]
print(f"{'rule':28} {'false':>6} {'notes hit':>10} {'missed':>7} {'p50 latency':>12}")
for name, rule in RULES:
    f, n, m, p = score(rule)
    print(f"{name:28} {f:6} {n:10} {m:7} {p:>9} ms")

# where the head's false ones sit
tally = defaultdict(int)
for i, (t0, pc) in enumerate(notes):
    end = notes[i+1][0] if i+1 < len(notes) else t0 + 1.5
    ok = played_near(t0)
    for k, r in enumerate(rows):
        if r[0] < t0 or r[0] > min(end, t0 + 1.5) or r[1] < FILL_MIN: continue
        for c in range(12):
            if r[3] and r[3][c] >= 0.6 and c != pc and c not in ok:
                tally[(c - pc) % 12] += 1
print("\nwhere the onset head's false credits sit, semitones from the note played:")
for d in sorted(tally, key=lambda k: -tally[k])[:6]:
    print(f"   {d:2}: {tally[d]}")

print("\nwhat the app would actually do (its three ways in, gated by the head):")
def app_rule(gate):
    def rule(r, run, rc):
        out = []
        for c in range(12):
            heard = (r[4] == c and run >= STEADY) or r[2][c] >= THR
            if heard and (gate is None or (rc[c] >= gate)):
                out.append(c)
        return out
    return rule
for gate in (None, 0.2, 0.3, 0.4, 0.5):
    f, n, m, p = score(app_rule(gate))
    label = "no gate" if gate is None else f"onset[c] >= {gate}"
    print(f"   {label:20} false {f:5}  notes hit {n:3}  missed {m:3}  p50 {p:>5} ms")
