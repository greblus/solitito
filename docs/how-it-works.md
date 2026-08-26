# How it works

The signal path, the model, and why single notes are not judged by the model alone.

[← back to the README](../README.md)

### Why single notes are not judged by the model alone

The model is asked about 48 frames, which is 0.77 s of audio, and it answers about all of it.
That is right for a chord you hold and wrong for a scale: measured on a scale at 0.6 s per
note, the pitch head named the note being played in 7% of windows and the one before it in
79%. Nothing is broken there — the model is reporting both notes it heard, because both were
in the window.

So the note modes ask a second question of a single CQT frame, which has no memory: a
harmonic sum over the log-magnitude bins gives the pitch class sounding right now. On the same
scale it named the current note 57% of the time and never named one that was not played. The
remaining lag is the 8192-sample FFT window, half a second wide, which is also why notes
shorter than about 0.4 s are still hard.

By default that estimate only ADDS a way to pass, because overruling the model would cost
something worth keeping: the pitch head is polyphonic, so strumming a whole chord walks its
intervals one by one, which no monophonic tuner can do. **Play the notes one at a time** turns
the estimate into the authority — then the model's window cannot credit the note before the
one under your fingers.

Whatever is asked for twice in a row — the same note twice in an arpeggio, a scale closing on
its root, the same chord written twice in a song — has to be played twice. What is still
ringing from the time before matches the moment it is asked for again, so passing needs a
fresh strike: the attack head's answer for the note in question has to cross 0.60. A chord
asks for two such strikes on its own notes — measured, a single one fires by itself under a
chord that is merely ringing, and two do not.

The envelope detector answers this only for a model that has no attack head at all. It
counts attacks on any string, so in a run of different notes it moves on every one of them:
in `1 2 3 4 5 6 7 1` the six notes between the two roots would have counted as the first
root being struck again. Every note is remembered separately for a related reason —
remembering only the note before would have forgotten the first root long before the last
one is due.

A note asked for a second time — the closing `1` of `1 2 3 4 5 6 7 1`, a degree the interval
box marks with `'`, an arpeggio that comes back to where it began — needs more than the
attack head alone, because the head spreads an attack over notes nobody played. While the
six degrees above the root are played, the root collects strikes of its own: two of them on
the test run, and in a fast run the head produced no strike for the closing root at all.

Two things can settle it. The estimate reads an absolute pitch and not just a note name, so
a note sounding six semitones or more from where it was read when it was credited is a
different string being played — the closing root against the opening one still ringing. That
is proof on its own and needs no attack; on the test run the two roots were read an octave
apart, 0.29 s after the string was hit. It cannot be a requirement, though: a run closed in
the octave it started in would never satisfy it, however many times it was played. So
otherwise the note's own strike counter has to have moved, and — where notes are played one
at a time — the estimate must not be reading some other note. That second half is what the
strike counter cannot supply on its own: the strays all land while the estimate is reading
whatever was actually played, so they no longer pass.

One more thing follows from the head's latency. Its answer arrives 0.2 to 0.5 s after the
string is hit, which is *after* the estimate has named the note and the step has been
credited on it — so that strike is still to come when the next step asks for the same note,
and it would answer for it. A credit therefore keeps up with its own note's counter for half
a second, for as long as the estimate is still reading that note. A pluck cannot pass its
own late strike on to the step after it.

Re-arming is relative. Under a strummed chord left ringing the head's answer for a note does
not fall back to nothing but hovers — 0.11 to 0.29 for a whole second on the measured
material — so a fixed floor would never re-arm and the next strum could not be seen at all.
A note is armed again once its answer drops below three tenths of the peak that counted the
strike before it.

---

## How it works

### Signal path

```
audio in → resample to 16 kHz → FFT (8192) → sparse pseudo-CQT → features → ONNX model
```

1. **Resampling.** Input is resampled to 16 kHz. The CQT spans 6 octaves from C1, so the
   highest bin sits around 2 kHz — far below the 8 kHz Nyquist limit.
2. **Pseudo-CQT.** Instead of a real constant-Q transform, the app multiplies the FFT
   spectrum by a precomputed kernel (144 bins, 24 per octave — quarter-tone resolution).
   The kernel comes from `librosa.filters.constant_q`, so the app and the trainer produce
   the same features.
3. **Features.** 168 values per frame: 144 CQT bins + 12 chroma + 12 bass-energy bins.
   The model sees 48 frames of history (0.77 s at a 256-sample hop).
4. **Inference.** One forward pass every 40 ms.

The CQT kernel is stored in a **sparse CSR format**. The full kernel has 4097×144 = 589,968
weights, but they concentrate around each bin's centre frequency. Dropping everything below
1e-4 of the peak keeps 6.9% of the weights and changes the output by 0.03% of peak
(measured on white noise, pink noise and a guitar-like harmonic series). The weights file
shrinks from 28 MB to 2 MB, and the audio thread does about 14× fewer multiplications per
frame.

### Model

A hybrid CNN + Transformer with four output heads:

| Stage | Detail |
|---|---|
| Input | `[48 frames, 168 features]` |
| CNN | Convolutional blocks with Squeeze-and-Excitation, InstanceNorm |
| Encoder | Transformer encoder with a CLS token, 384-dim |
| `root_logits` | 13 classes — 12 pitch classes + "Noise" |
| `quality_logits` | 11 classes — maj, min, maj7, dom7, min7, m7b5, dim7, aug, sus, note, N |
| `pitch_logits` | 12 sigmoid outputs — which pitch classes are sounding |
| `onset_logits` | 12 sigmoid outputs — which pitch classes were STRUCK in the last 6 frames |

The three heads answer different questions and are **not** interchangeable:

- `pitch_logits` is the strongest output (F1 0.909). It answers "which notes are sounding
  right now", which is exactly what the Intervals / Scales / Arpeggios modes need.
- `root_logits` names the tonal centre. 98.1%.
- `quality_logits` names the chord family. This is the hard one.
- `onset_logits` is the newest, and answers a question the other three cannot: not what is
  sounding but what was *struck*. Sounding is not enough — an open string ringing in
  sympathy is sounding, and so is the note before — which matters most in Formulas, where a
  mark never expires. It was trained on its own, with the rest of the network frozen, so the
  three heads above are bit-for-bit what they were. Measured against a real recording it is
  the fastest answer in the app (202 ms after the strike, against 676 ms) but it spreads an
  attack across neighbouring strings, so it does not decide *what* was played. What it does
  decide is whether something was struck **again**: a note or a chord asked for twice in a
  row needs its own strike, and the envelope detector cannot supply one — its level is the
  RMS of a 512 ms window, so a second pluck of a ringing string barely moves it. Measured on
  generated material the envelope caught 2 re-plucks of 6 and 2 re-strums of 6; the head
  caught all six of each, with nothing fired while a chord merely rang on. An older three-head model still runs: the names of the first three
  outputs did not change.

---
