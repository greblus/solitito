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
one under your fingers, and a repeated note needs a fresh attack.

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
  attack across neighbouring strings, so nothing is judged by it yet — it is read, logged
  and being measured. An older three-head model still runs: the names of the first three
  outputs did not change.

---
