"""Test material for the latency measurement: plucks with onsets known exactly.

    python dist/latency_material.py probe.wav onsets.csv
    ./target/release/solitito --probe probe.wav --step 1 > probe.txt
    python dist/latency_stats.py probe.txt onsets.csv

Karplus-Strong rather than a synthesiser: it is the physical model of a plucked
string, so the partials arise the way a real one's do - and partials are what
both the estimator and the model key on. Rendered at 44.1 kHz so the app's own
resampling is exercised as well.

The point of generating rather than recording is the ground truth: a latency
figure is the distance between an onset and an answer, and an onset guessed by a
detector would put its own error inside the number being measured. The cases
below are the ones the app actually meets - a line left ringing, a fast line,
semitone pairs, a quiet note under a loud one, and the same note struck again.
"""
import numpy as np, soundfile as sf, csv, sys

SR = 44100
OUT_WAV = sys.argv[1]
OUT_CSV = sys.argv[2]

def pluck(midi, dur, level=1.0, damp=0.996):
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    n = int(round(SR / f))
    rng = np.random.default_rng(midi * 7 + 13)
    buf = rng.uniform(-1, 1, n)
    # A pick excites the high end less than white noise does.
    buf = np.convolve(buf, [0.5, 0.5], mode="same")
    out = np.zeros(int(dur * SR), dtype=np.float64)
    idx = 0
    for i in range(len(out)):
        out[i] = buf[idx]
        nxt = (idx + 1) % n
        buf[idx] = damp * 0.5 * (buf[idx] + buf[nxt])
        idx = nxt
    # a soft attack envelope, and a body-ish tilt
    a = np.minimum(1.0, np.arange(len(out)) / (0.003 * SR))
    return out * a * level

events = []      # (t_sec, midi, case, level)
track = np.zeros(int(220 * SR))

def put(t, midi, dur=2.2, level=1.0, case=""):
    x = pluck(midi, dur, level)
    i = int(round(t * SR))
    track[i:i + len(x)] += x
    events.append((t, midi, case, level))

t = 1.0
# A. isolated notes, silence between: the easiest case there is
for m in [40, 45, 50, 55, 59, 64, 52, 61]:
    put(t, m, dur=1.6, case="isolated")
    t += 2.6

# B. a line at 500 ms, every note left ringing
t += 1.5
for m in [40, 43, 45, 47, 50, 52, 55, 57, 59, 62]:
    put(t, m, dur=2.4, case="line500")
    t += 0.5

# C. a fast line at 200 ms
t += 3.0
for m in [52, 54, 55, 57, 59, 60, 62, 64]:
    put(t, m, dur=2.4, case="line200")
    t += 0.2

# D. semitone pairs, the note before left ringing - the failure on the guitar
t += 3.0
for m in [42, 47, 52, 57, 62]:
    put(t, m, dur=2.4, case="semitone_a")
    put(t + 0.45, m + 1, dur=2.4, case="semitone_b")
    t += 2.4

# E. a quiet note under a loud one still ringing, and the other way round
t += 2.0
for m, (la, lb) in zip([45, 50, 55], [(1.0, 0.35), (0.35, 1.0), (1.0, 0.5)]):
    put(t, m, dur=2.6, level=la, case="level_a")
    put(t + 0.5, m + 5, dur=2.6, level=lb, case="level_b")
    t += 2.8

# F. the same note struck again
t += 2.0
for m in [43, 55]:
    for k in range(3):
        put(t, m, dur=1.4, case="repeat")
        t += 0.45
    t += 1.6

track = track[: int((t + 3.0) * SR)]
track *= 0.5 / np.max(np.abs(track))
sf.write(OUT_WAV, track.astype(np.float32), SR, subtype="FLOAT")

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "midi", "pc", "case", "level"])
    for (tt, m, case, lvl) in events:
        w.writerow([f"{tt:.4f}", m, m % 12, case, lvl])

print(f"{OUT_WAV}: {len(track)/SR:.1f} s, {len(events)} notes")
