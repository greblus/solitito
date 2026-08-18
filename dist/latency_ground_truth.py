"""Onsets and pitches of a REAL recording, for the same latency measurement.

    python dist/latency_ground_truth.py recording.wav onsets.csv

The ground truth comes from pyin - not from the app's model and not from its
estimator, so the measurement cannot flatter itself. Only monophonic material
makes sense here: one note at a time, which is what the note modes ask for.

Notes are cut out of the PITCH CONTOUR rather than off an onset detector. On a
plucked instrument the detector fires twice on one note as often as not, and a
phantom onset would put its error inside the very number being measured; a run
of frames agreeing on one note cannot be a phantom. The onset is then pulled
back to the start of that run.
"""
import sys, csv
import numpy as np, librosa

WAV, OUT = sys.argv[1], sys.argv[2]
SR = 22050
HOP = 256
FRAME_S = HOP / SR
MIN_RUN = 0.07          # a note has to hold this long to be one
MIN_PROB = 0.55         # how sure pyin has to be
MAX_CENTS = 40          # further off a semitone than this: not a settled note

y, _ = librosa.load(WAV, sr=SR, mono=True)
f0, voiced, prob = librosa.pyin(y, fmin=librosa.note_to_hz("E1"),
                                fmax=librosa.note_to_hz("E6"),
                                sr=SR, hop_length=HOP, fill_na=np.nan)
t = librosa.times_like(f0, sr=SR, hop_length=HOP)
midi = librosa.hz_to_midi(f0)

ok = voiced & (prob > MIN_PROB) & np.isfinite(midi)
note = np.where(ok, np.round(midi), np.nan)
cents = np.where(ok, np.abs(midi - np.round(midi)) * 100, 999)
ok &= cents <= MAX_CENTS

runs, i = [], 0
while i < len(note):
    if not ok[i]:
        i += 1; continue
    j = i + 1
    while j < len(note) and ok[j] and note[j] == note[i]:
        j += 1
    if (j - i) * FRAME_S >= MIN_RUN:
        runs.append((t[i], t[j - 1], int(note[i])))
    i = j

# Same note twice in a row with no gap: one note the tracker split, not two.
merged = []
for r in runs:
    if merged and r[2] == merged[-1][2] and r[0] - merged[-1][1] < 0.08:
        merged[-1] = (merged[-1][0], r[1], r[2])
    else:
        merged.append(list(r) if False else (r[0], r[1], r[2]))

print(f"{WAV}: {len(y)/SR:.1f} s -> {len(merged)} notes")
names = [librosa.midi_to_note(m, unicode=False) for (_, _, m) in merged]
for k in range(0, len(names), 16):
    print("   " + " ".join(f"{n:>4}" for n in names[k:k + 16]))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "midi", "pc", "case", "level"])
    for (t0, t1, m) in merged:
        w.writerow([f"{t0:.4f}", m, m % 12, "played", 1.0])
