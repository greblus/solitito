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
scale it named the current note 57% of the time and never named one that was not played. What
lag it had came from the 8192-sample FFT window, half a second wide, with every filter in its
middle: a new note reached the estimate a quarter of a second late. Since 0.5.18 the estimate
reads, just after an attack, through filters slid to the end of the window - see *The ear at
the end of the window* below.

By default that estimate only ADDS a way to pass, because overruling the model would cost
something worth keeping: the pitch head is polyphonic, so strumming a whole chord walks its
intervals one by one, which no monophonic tuner can do. **Play the notes one at a time** turns
the estimate into the authority — then the model's window cannot credit the note before the
one under your fingers.

Whatever is asked for twice in a row — the same note twice in an arpeggio, a scale closing on
its root, the same chord written twice in a song — has to be played twice. What is still
ringing from the time before matches the moment it is asked for again, so passing needs a
fresh strike. For a note that is the strike detector's answer for its class, see *Repeats*
below. A chord asked for again needs any strike from that detector at all: it reports a strum
as one class, seldom two, and does not fire while a chord only rings. On 17 strums of a real
guitar it saw every one and took none of the 17 ringing chords for a strum; the model's attack
head, which judged this before, saw 9 and fired under all 17 while they rang.

Without the strike detector's file, flux - an attack on any string, whatever its pitch -
answers instead, and the envelope where flux never reports. Both count attacks on any
string, so in a run of different notes they move on every one of them: in `1 2 3 4 5 6 7 1` the six notes between the two roots would have counted as the first
root being struck again. Every note is remembered separately for a related reason —
remembering only the note before would have forgotten the first root long before the last
one is due.

A note asked for a second time — the closing `1` of `1 2 3 4 5 6 7 1`, a degree the interval
box marks with `'`, an arpeggio that comes back to where it began — needs more than a strike
counter alone. The model's old attack head, which counted strikes before the strike
detector, spread an attack over notes nobody played: while the six degrees above the root
were played, the root collected strikes of its own, two on the test run.

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

The strike detector is late in the same way, only less: it answers 22 to 32 ms after the
string is hit, and since 0.5.18 the ear can name a note in 25. The detector's report then
arrived after the credit and passed for a new strike: a note only left to ring was credited
again 14 times in 51. So for 0.2 s after a credit a strike of its class is the strike it was
credited for. Nothing real is lost to this - the detector does not report a class twice within
0.6 s anyway. Ringing notes credited again: 0 of 51; struck again after 0.8 and 1.2 s, credited
in time: 49 and 50 of 51, none early.

Re-arming is relative. Under a strummed chord left ringing the head's answer for a note does
not fall back to nothing but hovers — 0.11 to 0.29 for a whole second on the measured
material — so a fixed floor would never re-arm and the next strum could not be seen at all.
A note is armed again once its answer drops below three tenths of the peak that counted the
strike before it.

### Every hop is judged

Intervals, Scales, Arpeggios and the Fretboard trainer are judged on every hop of audio -
16 ms - and not when the model answers. Scales, Arpeggios and the Fretboard used to wait for
the model, and after a silence its first answer does not come for 0.38 s, until its window is
half full.

Nor are hops lost on the way. The sound card hands its samples over in portions - on the
user's laptop about every 23 ms - and the window's clock read only the newest hop of each: 36%
of the hops played live were never judged. Since 0.5.18 the audio thread keeps every hop's
reading in a queue and the judge works through all of them at each tick of the window. Unseen
hops: none. Counted with `SOLITITO_HOPS=1`.

Some limits in the judge are counted in hops - how long the ear has to hold a note, for one -
and some of them were set while trying the app live, that is on two hops of three. Live, the
app can therefore credit a little sooner than it did.

### Crediting in Intervals

Intervals checks each new audio frame, even when the model has too little signal to answer.
Reading the same frame again does not extend a credit, and a gap in incoming frames resets the
pending confirmation. Model answers expire after 250 ms without an update.

The text strip and fretboard show the same credited steps, including notes played out of
order. After the last note, the whole set stays green for 350 ms before the next chord.
Pause holds that transition; silence does not.

A ringing note cannot credit its own repeat. Muting the input below the gate for at least
200 ms allows that note to count again once the estimate hears it steadily, even if no
detector caught the new pluck. A missing CQT estimate with the gate open is not muting.

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
- **An energy check on a class fired again within 3 s.** The refractory does not reach far
  enough: on the user's three recordings the model fired a class again within 2 s of itself 37
  times, in two groups with nothing between - 29 with the energy flat or falling (0.89 to 1.03
  of what it had been), the note dying away, and 8 with it jumping 3.7 to 49 times, a string
  struck again. One of the 29 landed just as that class was being asked for, at 0.67 s, and
  was the one false repeat the user saw in testing. So such a re-fire waits 32 ms and counts
  only if the energy rose by a quarter. Waiting matters: a gate that decided at the moment of
  firing refused real re-strikes, because the new note had barely entered the window then.
  The check reached 2 s at first, until a ringing Bb fired again 2.43 s after its strike and
  was credited in the next chord. Of 113 of the user's notes left ringing into a chord that
  had them too, 3 s took that one out and changed nothing else; 4 s added nothing.

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
a choice, not an oversight - the rules exist to make a test honest, and with them off the
mode credits chords and single notes as the model hears them.

### Only what was struck

With "only what was struck" on, the model's pitch head may credit a class only while the
strike detector reports that class struck within the last 22 hops, about 0.35 s. The
single-frame estimate is not affected. Measured through the whole judge, with the chord model
every 40 ms and the detector every hop, against the model's own attack head that decided this
before:

| | attack head | strike detector |
| --- | --- | --- |
| AtoA, 51 single notes: classes credited that nobody played | 165 | **124** |
| ... of them through the model's branches | 57 | **0** |
| 17 chords struck with the whole hand: chord tones credited | **62** / 68 | 47 / 68 |
| ... classes outside the chord | 48 | **42** |
| 9 chords played as arpeggios: chord tones credited / classes outside | 34 / 36, 30 | 32 / 36, **24** |

No single note was missed and none credited later. The cost is a chord struck at once: the
detector reports one class per strum, so the model credits that one and the ear has to name
the rest. The 124 false credits that remain come through the estimate, which this option does
not govern. Without the detector's file only flux is left, which says that a string was hit
but not which; the model's old attack head, which decided this before, is gone since 0.5.10.

### The ear just after a strike

The estimate reads a frame of half a second, so just after an attack it hears the note before
and the new one together, and for a moment it can name neither: after a C, a D was read as F
for 13 hops - 0.2 s - before it settled. In Formulas one such reading is a mark for the rest
of the exercise; this is how playing the root of a formula on D credited its b3, F.

So for 16 hops, about 0.25 s, after the strike detector reports a strike, the ear may credit
only a class the detector reported struck. Afterwards it is believed again, so a note the
detector did not report is late, not lost.

| | before | now |
| --- | --- | --- |
| AtoA, chords allowed: classes credited that nobody played | 124 | **62** |
| AtoA, one note at a time | 82 | **36** |
| strummed chords, one note at a time: classes outside the chord | 26 | **21** |
| arpeggios, chords allowed: classes outside the chord | 24 | **17** |

No note of AtoA was missed or credited later. On four sessions of the user's own playing, 388
credits, nothing was lost that the app credited before. A strummed chord gave up one tone of
68. 22 and 30 hops did nothing that 16 did not. Without the detector's file the ear is
believed at once, as before.

The end of a note misleads the ear too. As it dies away the estimate can settle on another
class - a fourth above, the note whose third harmonic it is - and name it steadily from about
1.8 s on. In Formulas a mark never expires, so there the ear credits only a class the strike
detector reported struck in the last 3 s. Tried in every mode, it took AtoA's false credits
from 62 to 3, but lost 15 of 388 credits of the user's sessions - quiet notes, legato ones,
the low strings, which the detector does not see - and most of a strummed chord's tones, so
the other modes do without it. In Formulas the price is a quiet or legato note that has to be
picked again.

### The ear at the end of the window

The estimate reads the CQT of a window of 8192 samples, half a second, and each of its filters
was placed in the middle of that window. A filter is as long as its note needs - 414 ms for the
low E, 104 ms for the high e, 16 ms for the harmonics above - so the newest sample a filter
sees at all lies 49 to 248 ms before the end of the window. The estimate sums harmonics, which
sit in the short, high filters: exactly the ones that look furthest back. That was the quarter
of a second it took to name a new note.

Since 0.5.18 the ear has a second set of filters, slid to the end of the window. Sliding a
filter in time multiplies its transform by a phase ramp, so the set is computed from the same
weights when the app starts: no new file, and the pruning stays as it was. The chord model
still reads the filters in the middle - it was trained on them.

On a synthetic signal the end-aligned ear names a new note 24 to 96 ms after it is struck,
against 256 to 272 ms. Through the whole judge, on lines spliced from AtoA's notes:

| | 0.5.17 | 0.5.18 |
| --- | --- | --- |
| a line at 0.40 s a note, credited | 1 of 8 | **8** of 8 |
| Harris scales falling, 0.30 s a note | 1 of 9 | **9** of 9 |
| ordinary path, median from the strike to the credit | about 400 ms | **about 150 ms** |
| AtoA, median to the credit (chords allowed / one at a time) | 156 / 220 ms | **39 / 49 ms** |

Late in a note these filters mislead. They do not all reach back equally far: a low one still
holds what was played before, a high one only the newest 50 ms, and as a note dies away the two
halves add up to a note nobody played - most often a fourth above, the note whose third
harmonic was played. Used all the time, they took AtoA's classes credited that nobody played
from 62 and 37 to 77 and 52. So the ear reads through them only for 16 hops, 0.26 s, after an
attack - flux or the strike detector - and through the filters in the middle after that: 63 and
41. 12 and 20 hops named lines equally fast and left more false credits.

A note struck while the one before rings on - an arpeggio across strings - is louder in
neither set of filters: the ear goes on naming the ringing note. The model's pitch head used to
settle it, and where notes come one at a time the model is no longer asked. The voices settle it
now: where the ear names a note struck *before* the one asked for, or has only begun to name
the one asked for, that note passes if the strike detector reported it and the voices hear it.
On lines with the note before ringing 0.3 s, the ordinary path went from 8, 8, 1 at 0.5 s a note
and 1, 4, 1 at 0.3 s to 8, 8, 5 at every speed down to 0.25 s.

Still open: of Ted Greene's exercises played through the judge, two runs stop halfway, where
0.5.17 stopped none. The detector does not report a repeated F, and the rule for a note that
went quiet and came back does not see it go quiet.

### Quick crediting

Until 0.5.17 the ordinary path credited a single note about 0.4 s after it was struck, and the
time was not spent deciding: it was spent waiting. The ear read a window of half a second and
named a new note only once it outweighed the one before in that window, about 0.26 s after the
strike; the answer then had to hold for 0.12 s. On 388 credits of the user's own sessions the
ear had named the note 128 ms before the credit - and the strike detector, which answers some
25 ms after an attack, 350 to 370 ms before it. A line of single notes was credited whole at
half a second a note and stopped at 0.4 s: about two notes a second. With the ear at the end
of the window the ordinary path takes about 0.15 s and a line goes through at 0.25 s a note -
as long as each note stops the one before.

Two witnesses are quicker than the ear, and neither is good enough alone:

- **The strike detector** names the class struck 22 to 32 ms after the strike, in 18 to 21 of
  21 notes of a line at any speed, and almost never a class nobody played. But it misses
  some strikes, and what it does report is a claim about an attack, not about what sounds.
- **The quick ear** reads the newest samples only. Its first reading is the tuner's YIN on
  the last 64 ms: on a line where each note stops the one before, it names all 21 notes at
  every speed down to 0.15 s a note, 26 to 32 ms after the strike. It is no ear for two
  strings at once - with the note before left ringing 0.3 s under the next it named 14 of 21
  at 0.3 s a note - and on a note dying away it wanders. Its second reading is what rose in
  the spectrum of the last 128 ms against the 128 ms before: a string only ringing has not
  risen, so the note just struck is the strongest thing in the rise, ringing notes or not.
  That reading named all 21 notes at every speed with and without the ring - and on a held
  note it goes on naming things: on AtoA's 51 notes left to ring, another class for three
  hops or more 194 times.

With **Quick crediting** on, where notes are taken one at a time, a note is credited the
moment the two agree: the detector reported the class asked for within the last 8 hops, 0.13 s,
and the quick ear names the same class, by either reading, for two hops running. A strike that
has been credited does not count again, so a class asked for twice still has to be struck
twice. Everything else is left to the ordinary path, which credits as it always did - so a
note the detector missed is late, not lost.

Measured through the whole judge in 0.5.18, on lines spliced from AtoA's notes and judged as
Scales judges them - D dorian, G mixolydian, a D m7 arpeggio - credited notes of 8, 8 and 5, and
the median time from the strike to the credit:

| seconds a note | ordinary | ordinary, the note before ringing 0.3 s | quick | quick, the note before ringing 0.3 s |
| --- | --- | --- | --- | --- |
| 0.50 | 8, 8, 5 - 150 ms | 8, 8, 5 - 142 ms | 8, 8, 5 - 26 to 38 ms | 8, 8, 5 - 30 to 42 ms |
| 0.40 | 8, 8, 5 | 8, 8, 5 | 8, 8, 5 | 8, 8, 5 |
| 0.30 and 0.25 | 8, 8, 5 | 8, 8, 5 | 8, 8, 5 | 8, 8, 5 |
| 0.20 | 7, 5, 3 | 8, 5, 2 | 7, 7, 4 | 8, 8, 5 |
| 0.15 | 2, 2, 2 | 2, 4, 2 | 7, 7, 4 | 7, 7, 4 |

In 0.5.17 the ordinary path credited 1 note of each line below 0.5 s a note, and needed 400 ms
for each at 0.5; with the note before ringing, 1 of each below 0.5 s.

The notes still missed at 0.2 s and faster are strikes the detector did not report; a line
is judged in order, so it waits there for the ordinary path, which at that speed cannot
answer. On the user's sessions the detector and the quick ear agreed before the credit in
366 of 388 notes; the other 22 are credited as before.

The other side, on AtoA's 51 single notes each left to ring: the two agreed on the note
played 44 times, a median 33 ms after the strike, and on a class nobody played twice - a
semitone below the note, in the attack itself, both times. The ear proper, by its own rule
and alone, named another class for three hops or more 150 times on the same recording.

It is an option, on by default since version 0.5.17. All of this is measured on recordings, and
the live input adds what a recording does not have - the sound card's buffer above all; since
0.5.18 the judge sees every hop live as it does on a recording. Should it credit notes you do
not play, turn it off in the quick options: the ordinary path alone is left.

### The noise gate

The gate decides whether a window is playing or silence; what it calls silence is not judged,
and silence of 200 ms lets a note count again. It used to compare the level of the whole
half-second window with the threshold, so it opened late on a quiet attack, closed late after
one, and a note dying away round the threshold dropped out in the middle of its decay.

Since 0.5.18 the level is that of the newest 1024 samples, 64 ms, and the gate has two sides: it
opens on the threshold and closes only once the level has stayed 6 dB under it for 10 hops,
160 ms. A note fading round the threshold no longer flickers in and out. With AtoA played 6 dB
quieter against the app's own threshold, -34 dB:

| | old gate | new gate |
| --- | --- | --- |
| G mixolydian, 0.50 s a note | 5 of 8 | **8** of 8 |
| D dorian, 0.40 s a note | 1 of 8 | **8** of 8 |
| D m7 arpeggio, 0.40 s a note | 1 of 5 | **5** of 5 |

At the ordinary level the two gates credit the same.

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
audio in → resample to 16 kHz → FFT (8192) → sparse pseudo-CQT ─┬─ filters in the middle → features → chord model
                                                                 └─ filters at the end → the ear and the voices
            newest samples → strike detector, quick ear, noise gate
```

1. **Resampling.** Input is resampled to 16 kHz. The CQT spans 6 octaves from C1, so the
   highest bin sits around 2 kHz — far below the 8 kHz Nyquist limit.
2. **Pseudo-CQT.** Instead of a real constant-Q transform, the app multiplies the FFT
   spectrum by a precomputed kernel (144 bins, 24 per octave — quarter-tone resolution).
   The kernel comes from `librosa.filters.constant_q`, so the app and the trainer produce
   the same features.
3. **Features.** 168 values per frame: 144 CQT bins + 12 chroma + 12 bass-energy bins.
   The model sees 48 frames of history (0.77 s at a 256-sample hop).
4. **Inference.** One forward pass every 40 ms where the chord model decides, and none
   where it does not - see *When the chord model is asked* below.
5. **Every hop.** Each hop of 256 samples, 16 ms, is read once: the CQT, the ear, the strike
   detector and the gate. The readings wait in a queue for the judge, so none is skipped
   when the window redraws late.

The CQT kernel is stored in a **sparse CSR format**. The full kernel has 4097×144 = 589,968
weights, but they concentrate around each bin's centre frequency. Dropping everything below
1e-4 of the peak keeps 6.9% of the weights and changes the output by 0.03% of peak
(measured on white noise, pink noise and a guitar-like harmonic series). The weights file
shrinks from 28 MB to 2 MB, and the audio thread does about 14× fewer multiplications per
frame.

### Model

A hybrid CNN + Transformer with three output heads, `best_model_v2_take6_chords.onnx`:

| Stage | Detail |
|---|---|
| Input | `[48 frames, 168 features]` |
| CNN | Convolutional blocks with Squeeze-and-Excitation, InstanceNorm |
| Encoder | Transformer encoder with a CLS token, 384-dim |
| `root_logits` | 13 classes — 12 pitch classes + "Noise" |
| `quality_logits` | 11 classes — maj, min, maj7, dom7, min7, m7b5, dim7, aug, sus, note, N |
| `pitch_logits` | 12 sigmoid outputs — which pitch classes are sounding |

The three heads answer different questions and are **not** interchangeable:

- `pitch_logits` is the strongest output (F1 0.909). It answers "which notes are sounding
  right now", which is exactly what the Intervals / Scales / Arpeggios modes need.
- `root_logits` names the tonal centre. 98.1%.
- `quality_logits` names the chord family. This is the hard one.

What none of them answers is what was *struck*, as against what is sounding. That comes from
the separate strike detector, `short_onset_masking_v2.onnx`, described under
[Repeats](#repeats-which-string-was-struck). The chord model used to carry a fourth head for
it, `onset_logits`, trained in a phase of its own with the rest of the network frozen; the
detector does the job better, and since 0.5.10 the head is gone from the file the app loads.
The three outputs above are bit for bit what they were with it. An older three-head model,
`best_model_v2_take6.onnx`, still loads as a fallback, but it is an earlier export and names
some chords differently.

### A major or a dominant seventh

The quality head mixes up the two sevenths; the pitch head does not. On the user's recording
of CMaj7 strummed five times the model named it C7 nearly throughout, at 0.8 to 0.9, while the
pitch head heard C, E and B - the major seventh - and no B♭; not one of the five passed. So in
Chords a chord named with the right root and the other seventh passes when the pitch head hears
the root, the third and the seventh asked for above one half, and the other seventh below it.
On the user's 17 strummed chords: 17 pass, where 13 did, every failure a CMaj7. On 9 chords
played as arpeggios nothing changed but CMaj7, which passes after 1.5 s instead of 3.7.

### When the chord model is asked

The model costs nearly all of what the app costs: one pass takes 38.6 ms on the user's small
laptop (Ryzen 3 5300U) and 69 ms on their thin client, so asked every 40 ms it fills one core
and more. It is asked that often only where its answer decides: in Chords, and in Intervals and
Formulas while chords are allowed. Where notes are played one at a time - Scales, Arpeggios, the
Fretboard trainer, Intervals and Formulas with *one at a time* - it is not asked at all, and the
ear decides.

That is measured, not assumed: of 388 credits in four of the user's Intervals sessions, 388
were the ear's and none the model's; and the whole scorecard, with the model asked every 200 ms
or never in those modes, came out the same to the last digit. On the laptop, AtoA played through
`--file`:

| mode | 0.5.17 | 0.5.18 |
| --- | --- | --- |
| Scales | 113% of a core | **11%** |
| Arpeggios | 111% | **11%** |
| Chords | 111% | 108% |

---
