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

### Crediting in Intervals

Intervals checks each new audio frame observed by the UI, even when the model has too
little signal to answer. Reading the same frame again does not extend a credit, and a gap
in incoming frames resets the pending confirmation. Model answers expire after 250 ms
without an update.

The text strip and fretboard show the same credited steps, including notes played out of
order. After the last note, the whole set stays green for 350 ms before the next chord.
Pause holds that transition; silence does not.

A ringing note cannot credit its own repeat. Muting the input below the gate for at least
200 ms allows that note to count again once the estimate hears it steadily, even if the
onset head missed the new pluck. A missing CQT estimate with the gate open is not muting.

Where the single-frame estimate names a class, no other class may be credited off the
model that frame. This holds in both orders; it used to run only in free order, and
playing in order - the default - had nothing holding a third credited off a ringing root.

More than one string sounding is the exception, and it is counted rather than guessed: see
*Which notes are sounding* below. Two voices is already something one plucked string
cannot be. The chord name used to answer this question and must not again - one plucked
root is enough for the model to recognise the shape, which is how a third nobody touched
was credited.

### Repeats: which string was struck

A class credited once counts again only when **that class** has been struck since - not when
some string was hit, and not when the class is merely still sounding. Everything else in the
app answers one of those two easier questions: flux says a string was hit and is blind to
which, the ear and `voices` say which classes sound and are blind to whether they were just
struck. The repeat rule needs the conjunction.

It comes from a small causal onset model, `short_onset_masking_v2.onnx` - the onset branch of
`best_model_v2_take7_masking_v2.onnx`, cut out of the combined file so it does not run the
chord trunk: 1 MB and 0.7 ms a hop. It reads two short windows of the newest audio (64 and
128 ms), 35 frames of history, and answers twelve per-class probabilities. Two things sit
between it and the judge:

- **Levelling.** Its features are not level-invariant, and a quiet player reads as a weak
  attack: on the user's own capture, 11 dB under the recording it was measured on, it found
  42 of 87 notes. A slow gain - about four seconds to settle - brings the playing to the level
  it knows: 81 of 87.
- **A refractory of 0.6 s per class.** The model fires on decaying notes too, and at those
  moments the signal is losing energy - a median 0.95 of what it had been, against 3.3 at real
  attacks. In the exercises a class comes back only after a credit, the 0.35 s the finished
  set is shown for, and the player's reply, so nothing real is lost to it.
- **An energy check on a class fired again within 2 s.** The refractory does not reach far
  enough: on the user's three recordings the model fired a class again within 2 s of itself 37
  times, in two groups with nothing between - 29 with the energy flat or falling (0.89 to 1.03
  of what it had been), the note dying away, and 8 with it jumping 3.7 to 49 times, a string
  struck again. One of the 29 landed just as that class was being asked for, at 0.67 s, and
  was the one false repeat the user saw in testing. So such a re-fire waits 32 ms and counts
  only if the energy rose by a quarter. Waiting matters: a gate that decided at the moment of
  firing refused real re-strikes, because the new note had barely entered the window then.

Measured on every reviewed note of AtoA spliced into new signals - the note alone, ringing out,
and the same note struck again 0.8 and 1.2 s later, with the pick stopping the old vibration:

| | 0.5.7 | now |
| --- | --- | --- |
| a repeat allowed while the note only rings | 24 / 51 | **0** / 51 |
| struck again after 0.8 s, counted in time | 36 / 51, 11 early | **48** / 51, none early |
| struck again after 1.2 s, counted in time | 35 / 51, 14 early | **50** / 51, none early |

Without the model file the app still starts and judges repeats on the older evidence, with two
of its leaks closed: an octave jump in the estimate counts as a new pluck only with an attack
behind it - that one branch gave 50 of the 58 false repeats above - and the same 0.6 s logic
applies to the pitch-blind attack. That fallback lets 13 ringing notes of 51 through.

### Carrying across a chord

What is still sounding when the exercise moves to the next chord counts as already used there,
and needs a strike of its own. The repeat rule alone guarded only classes credited before, so a
note that rang out of the last chord without being credited - a wrong note, an extra one - was
free to answer the next chord the moment the ear named it: on AtoA's notes, 47 times of 51.
Now none. A class struck while the finished set is still shown is exempt: that is a player
reaching the next chord early, not a note left over.

Requiring a strike for every first credit would close this too, but costs 7 to 11 of the 87
notes of the user's own session, which the detector does not catch; carrying costs nothing
where the note is played after it is asked for, which is every ordinary credit.

With single-note playing, the fixed order and "only what was struck" all switched off, none
of this applies: the model decides and nothing argues with it, carry-over included. That is
a choice, not an oversight - the rules exist to make a test honest, and the mode is also a
thing to play chords on for fun.

### Which notes are sounding

The spectrum is explained rather than ranked. The strongest candidate is taken, the partial
series it predicts is subtracted, and the residual is asked what is left: a class fully
explained as somebody else's harmonic leaves nothing behind and is not a voice. That is the
question ranking cannot answer, because the third harmonic of a root lands on the fifth and
the fifth harmonic on the major third.

A candidate also has to carry energy where its own fundamental would be, which stops one
below a sounding note scoring on borrowed partials.

Measured on a recording of 51 reviewed single notes, this reports exactly one voice 46
times and not one extra voice on a harmonic interval; on a recording where notes come every
half second and overlap, it reports two voices 38 times of 87. Two blind spots are known
and written into the tests: an octave cannot be resolved at all - its fundamental sits
exactly on the second partial of the note below, so no harmonic model can separate them -
and a fifth much quieter than the note under it is missed. This is why the exception counts
voices instead of looking for one particular class in the list.

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
