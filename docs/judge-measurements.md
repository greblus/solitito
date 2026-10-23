# Judge measurements

The numbers behind the judge's rules and its detectors' constants.

Each section is the comment that used to sit on that rule in the code, moved
here word for word in October 2026. The code keeps the rule, the reason and
one number; this keeps how the number was found - what was tried, on which
recording, and what it cost - so that whoever changes it knows what to
measure again.

The recordings named are measurement material: AtoA (51 reviewed single
notes), sesja-g1 and the other `-g1` captures of the user's playing in
`dist/crediting_measurements/`. "Hops" and "frames" are 16 ms
(256 samples at 16 kHz).

## The ear

### mono_pitch

*`src/audio.rs`*

The pitch class of a SINGLE sounding note, straight from one CQT frame.

A harmonic sum over the log-magnitude CQT - which, the axis being
logarithmic, is the harmonic PRODUCT spectrum, the standard monophonic
estimator. It exists because the model cannot answer this question: its
context window is 48 frames, so it reports everything that sounded in the
last 0.77 s, and in note practice that is two or three notes at once.

It has no memory of its own - but it is NOT the quick answer this comment
used to claim. One frame is 512 ms of audio (`FFT_SIZE` at `TARGET_SR`),
refreshed every 16 ms; the hop is what is short, not the window. Measured
against material with known onsets (`dist/latency_material.py`) it names a
new note at a median of 560 ms after the strike, against the model's 620 ms
- the two paths sit on the same floor, and that floor is the window.

Returns the pitch class (0 = C) and the score of the winning bin.

> Note (October 2026): The last line of this comment said the function returns a pitch class (0 = C). It returns the absolute semitone counted from the lowest CQT bin; the comment in the code now says so.

### true_fundamental

*`src/audio.rs`*

The note whose partial won the sum, which may lie below the winner.

Any partial can win the sum, because an impostor sitting on one of them
collects the note's upper harmonics as its own: a fifth above shares the
note's third and sixth, an octave shares every second one, and two octaves
and a major third - the fifth harmonic - is what actually won for a third
of a chromatic run of isolated plucks. The winner is therefore compared
against every candidate that would explain it, lowest first, and the first
one that stands up on its own bin takes it.

Measured on that run: 6 of 12 correct before, 12 of 12 after, with the
model naming 11 of the same 12 - so the information was in the frame all
along and the sum was the thing throwing it away.

### CREDIT_TICKS

*`src/state.rs`*

How long the single-frame estimate has to have been naming a note before it
is credited, in audio frames of 16 ms - and how long a DIFFERENT note has to
have been named before the one credited counts as having stopped sounding.

Measured on a 20 s recording of single notes, at its own level and at 6, 12
and 18 dB below it, against what the estimate reads at full level. Per frame
of a note actually sounding, credited correctly / named as some other note:

| rule          | full   | -6 dB  | -12 dB | -18 dB |
|---------------|--------|--------|--------|--------|
| three frames  | 88/0 % | 86/0 % | 84/1 % | 80/1 % |
| two frames    | 93/0 % | 90/1 % | 87/2 % | 83/3 % |
| 2 of the last 5 | 91/8 % | 90/9 % | 89/9 % | 87/12 % |

Two frames is the whole of the gain: it credits five points more of what is
played at any level and still names almost nothing wrong. A vote over a
window - which is what Formulas uses, see `EAR_VOTES` - buys no more than
that and misnames one frame in ten, which is the "it credits notes I never
played" this mode had before.

What is NOT the constraint: the estimate's own score gate. On that recording
the lowest score of any reading was 0.52 against a `MONO_MIN_SCORE` of 0.50,
so lowering it would have changed nothing. What does bind, before any of
this, is the noise gate: at the recording's own level the default -34 dBFS
passes 78 % of frames to the analyser, 6 dB softer 36 %, and 12 dB softer
none at all.

### EAR_SETTLE

*`src/state.rs`*

Hops after the strike detector reports a strike during which the ear may
credit only a class the detector reported struck: about 0.25 s.

The FFT window is half a second wide, so just after an attack it holds the
note before and the new one together, and the estimate wanders - after C,
a D read as F for 13 hops before it settled. In Formulas that one reading
is a mark for the rest of the exercise. Measured on AtoA, the ear's false
credits went from 124 to 62 (82 to 36 one note at a time) with no note
missed or later; on four sessions of the user's playing, 388 credits, it
cost nothing; 16 hops did all that 22 and 30 did.

### EAR_BACKED

*`src/state.rs`*

In Formulas, hops back the strike detector must have reported a class for
the ear to credit it at all: 3 s.

As a note dies away the estimate can settle on another class - a fourth
above, the note whose third harmonic it is - and name it steadily from
about 1.8 s on. A mark in Formulas never expires, so there it takes a
strike. Measured with it everywhere: AtoA's false credits 62 -> 3, ringing
notes credited in a next chord with no strike 20 -> 12 of 113 (the rest
played, quietly). The cost kept it out of the other modes: 15 more of 388
session credits lost - quiet notes, legato ones, the low strings, which
the detector does not see - and a strummed chord's tones 46 -> 20 of 68.

## Strikes and attacks

### Flux detector

*`src/flux.rs`*

A causal spectral-flux attack detector: pitch-blind, and not learned.

It answers WHEN a string was hit, nothing else. That is deliberate - the
question it refuses to answer is why it can be sensitive.

Measured on the first recording from the user's own rig
(`dist/crediting_measurements/rise-live/trace-20261007-213243-kKmraU`,
45.9 s of Intervals practice). He plucked about sixty times; the app
credited twelve notes, which is what "you have to pluck very hard" was:

| detector                                 | found |
|------------------------------------------|------:|
| this one                                 |    27 |
| the envelope in `audio.rs`               |    22 |
| Rise peaks >= 0.5, any class             |    19 |
| Rise peaks >= 0.8, the release threshold |    12 |

Every attack Rise reported there, this one also reports; it adds fifteen
Rise never saw. On AtoA, against 51 reviewed labels, it finds all 51 where
Rise finds 48.

The envelope compares the frame's whole level against a running baseline, so
a quietly plucked string is swallowed by the one still ringing beside it.
Summing the per-bin INCREASES does not care how loud the rest of the frame
is: a new string raises its own bins whatever else is sounding.

One pluck reporting once is measured on material, not asserted here: at the
working constants no two reports on AtoA fall within 160 ms of each other.
A synthetic ramp does report twice, and that is honest - a signal rising
steadily for seven frames is not the shape a plucked string makes, and
tuning the detector to call it one event would be fitting to a fiction.

### DEVIATIONS

*`src/flux.rs`*

How far above the recent median, in median absolute deviations.

Swept at 0.5.6 against AtoA's 51 reviewed labels, with both windows scaled:
25 finds 41, 40 finds 47, 60 finds 47 with a third fewer reports outside
them, 80 drops to 40. Sixty is the knee.

What this replaces is the take6 onset head, which on the same 51 notes hit
33, put 50 events in the wrong class and reported 18 duplicates. It is not
competing with a good detector.

Swept against the 51 reviewed labels of AtoA and the user's recording, with
the refractory below. Every value from 5 to 25 finds all 51 notes; what
changes is how much else it reports, so the highest was taken:

| deviations | AtoA notes | other sounds | attacks on the user's recording |
|-----------:|-----------:|-------------:|--------------------------------:|
|          5 |      51/51 |          148 |                              88 |
|         12 |      51/51 |           84 |                              48 |
|         18 |      51/51 |           58 |                              36 |
|         25 |      51/51 |           35 |                              27 |

"Other sounds" is not the same as error: this is pitch-blind, AtoA is a
performance, and its 51 labels are the notes of the piece rather than every
sound the strings made. Rise at its own threshold finds 48 of the 51.

The working value sits between: the first player to try it still had to hit
some strings harder than he wanted to. `SOLITITO_FLUX=<number>` overrides it
while that is being found out.

> Note (October 2026): This comment held two sweeps that disagree: the one at 0.5.6 calls sixty the knee, and the table after it finds every note from 5 to 25 and says the working value sits between. The constant is 60. The comment also named a `SOLITITO_FLUX=<number>` override; the app does not read that variable - only the diagnostic test reads `SOLITITO_FLUX_DEVIATIONS`.

### Flux REFRACTORY

*`src/flux.rs`*

The least gap between two reports, 48 ms. A floor only - what actually
separates two plucks is the latch below.

A fixed refractory was tried for both jobs and could do neither. Long enough
to cover one attack it silenced the next real one, costing a third of the
notes on AtoA; short enough to pass them it let a single pluck report twice.
A string hit once keeps rising for about seven frames, so no fixed number
separates "still the same attack" from "the next one".

> Note (October 2026): The comment said 48 ms, which was written for an earlier, shorter value; commit d69f9be (8 Oct 2026) set 14 frames together with `OF_RECENT`.

### Flux REARM

*`src/flux.rs`*

A class is ready to report again once the flux falls to this fraction of the
value that last fired. The same shape of latch the model's own events use.

This is what tells one attack from two: while a string is still being
excited the flux stays up, and between plucks it drops to the floor. The
fault it fixes was measured on the user's second recording - one pluck at
frame 2398 reported twice, 96 ms apart, and the first report named a note
nobody played because it landed on the transient.

### OF_RECENT

*`src/flux.rs`*

And an attack has to reach this fraction of what the playing has recently
been worth, which the absolute floor alone could not enforce.

Measured on the user's second recording: in a quiet passage the adaptive
median and deviation both collapse, so the bar falls to the noise and the
detector reports attacks in silence. One of those at frame 2389 disarmed the
latch and the real pluck at 2398 was not reported until 2401.

0.35 with a refractory of 14 is where sesja-g1 and AtoA both keep every note
they had - 84 of 87 credited notes and 47 of 51 labelled ones - while the
reports with nothing becoming audible fall from 96 to 81 and the total from
238 to 189. Tightening further costs notes on both.

A fourth mechanism was tried and dropped: requiring the rise to be worth a
fraction of what the bins it rose in were holding, which is the physical
difference between a struck string and a ringing one beating. It separates
nothing here. Meaningful on sesja-g1 only at settings that cost four of
AtoA's notes, and those four are quiet attacks over a ringing note - the one
thing this detector exists to hear. A small new thing and a small change in
a big thing look alike in this representation, which is also why counting
fundamentals failed.

### Strike REFRACTORY

*`src/strike.rs`*

Hops after a strike of a class during which that class is not reported
again: 0.6 s.

The model fires on decaying notes: on the user's own playing, levelled, it
reported a class again less than 0.6 s after itself 58 times, and at those
moments the signal was LOSING energy - a median 0.95 of what it had been,
against 3.3 at first firings. In the exercises a class comes back only after
a credit, the 0.35 s the finished set is shown for and the player's reply,
so nothing real is lost to this.

A gate on the class's own energy rising was tried instead, twice, and both
times refused real re-strikes. The model fires about 26 ms after the attack,
when the 128 ms window holds little of the new note and the old one has just
been stopped by the pick, so the energy at that moment is FALLING: on AtoA's
notes struck again 0.8 s later it let 3 of 51 through where this lets 48.

### RECHECK

*`src/strike.rs`*

A class reported again within this many hops of its last strike - 3 s -
has to show the energy RISING around the new one before it counts.

It was 2 s until a ringing Bb fired again 2.43 s after its strike, with no
rise at all, and was credited in the next chord. Of 113 notes of the
user's sessions left ringing into a chord that had them too, 3 s took that
one out and changed nothing else measured; 4 s added nothing.

The refractory does not reach far enough: the model also fires on decaying
notes later than 0.6 s. On the user's three recordings it fired a class
again within 2 s of itself 37 times, and they fall in two groups with
nothing between: 29 at an energy ratio of 0.89 to 1.03 - the note dying
away, the false repeat the user reported among them, at 0.67 s - and 8 at
3.7 to 48.8, a string struck again. Only one of the 29 happened to land when
that class was being asked for, which is why it was seen once.

### STRUCK_FRESH

*`src/state.rs`*

Hops after the strike detector reports a class within which that class
counts as struck for "credit only what was struck": about 0.35 s.

Measured through the whole judge (`onset_sources`): on AtoA's 51 notes it
took the 57 false credits the onset head let through the model's branches
down to none, with no note missed and no later credit, and anywhere from 6
to 50 hops gave the same. A strummed chord pays for it: the detector
reports one class per strum, so 47 of 68 chord tones passed against 62 -
the rest wait for the ear.

### CLASS_FRESH

*`src/state.rs`*

Hops within which a strike of a class still counts as just now: 0.6 s.

Longer than `FLUX_FRESH` because the strike is reported at the attack and the
credit waits for the ear to settle on the note, which takes up to 0.4 s.

Asking for a flux attack beside the strike as well was measured and dropped:
on AtoA's notes struck twice it took one false permission away and twelve
real re-strikes with it.

### struck_since

*`src/state/credits.rs`*

THIS class struck since its credit, and just now - the question the
rule always meant to ask, and the first evidence that can answer it.
A note played again an octave away is a strike of the same class and
counts through here; nothing else needs to look at octaves.

Measured on every note of AtoA spliced with itself: of 58 permissions
given while the note was only ringing, 50 came from the octave branch
below - the estimate flips an octave on a decaying note and the rule
read the flip as a new pluck - and 8 from the pitch-blind attack.

Pitch-blind, on purpose. The take6 onset head, asked WHICH class was
struck, answered 33 of AtoA's 51 notes right, 50 in the wrong class
and 18 twice.

Whether a string was hit at all is a far easier question, and the
flux detector answers it without a class to get wrong: 47 of the same
51 notes, 14 reports outside them, no pluck reported twice.

WHICH note it was is then left to the monophonic estimate above,
which names the loudest class - a fundamental, not its harmonics.
JUST NOW, not merely since. The counter is pitch-blind by design, so
"an attack since the credit" is satisfied by any other step of the
exercise being played - and by the time a chord hands over, several
have been. That is how a note credited in one chord was credited
again in the next while it merely rang on: shuffle put a shared class
at the boundary, the counter had moved, and nothing else objected.

A string struck again while it still rings is a different thing and
has to keep working, so this cannot ask for silence first - only for
the hit to be happening now.

### chord_struck_since

*`src/state/chords_mode.rs`*

Whether the strings have been hit since this chord became the target.

The strike detector answers it with any strike at all: it reports a
strum as one class, seldom two, and it does not fire on a chord that
only rings. Measured on 17 strums of a real guitar (`onset_sources`):
every strum seen and no ringing chord taken for one, where the take6
onset head's two strikes on the chord's notes saw 9 strums and fired on
all 17 chords while they rang. Without the detector, the envelope.

## Voices

### NEXT_VOICE

*`src/voices.rs`*

Least a residual peak has to score to count as another voice, as a fraction
of the first voice's score. A voice far quieter than the one found first is
what is left of its own subtraction, not a string somebody struck.

Measured on AtoA's single notes: the spurious extra voices score 0.53 to
0.67 of the first, so 0.4 let every one of them through and the count sat at
the cap for every note.

0.75 reads AtoA's isolated notes better still - one voice 48 times of 51
against 46 - but loses a fifth sounding as loudly as the note below it,
because the two share a partial and the subtraction takes it. Losing a tone
that was played is worse here than two frames of a ghost: the point of this
is to confirm that several strings are sounding.

### OWN_BIN

*`src/voices.rs`*

A new voice has to carry this fraction of the first voice's own fundamental
bin, in ITS own fundamental bin.

The weighted sum over partials is not enough, because a candidate BELOW a
sounding note collects that note as one of its own partials and scores well
on borrowed energy - which is what put spurious voices 14, 17 and 18
semitones below the note on AtoA. Subtraction cannot answer it either: it
removes a note's partials upward, and a sub-harmonic's evidence lies above
it, in the note itself. So the candidate is asked for something only a
struck string has - energy where its own fundamental would be.

Swept on two recordings. At 0.7, AtoA's isolated notes report exactly one
voice 48 times of 51 - against 41 at 0.2 - while sesja-g1, where notes come
every half second and overlap, still reports two voices 30 times of 87. So
it does not buy the single notes by refusing real second voices.

## Crediting branches

### note_is_sounding

*`src/state/judge.rs`*

Is the requested pitch class sounding right now?

Shared by the note modes and the fretboard trainer so the two cannot
drift apart. Four ways in, because no single one covers note practice:

Measured on a scale at 0.6 s per note, the model's pitch head named the
note actually being played in 7% of windows and the PREVIOUS one in 79% -
not because it is wrong, but because it is asked about 0.77 s of audio and
faithfully reports both notes it heard. The CQT estimate looks at one
frame and got the current note 57% of the time, the previous one the rest,
and never a note that was not played at all.

### One at a time

*`src/state/judge.rs`*

Measured over 49 notes of a real recording: the four ways below
together credited 110 things nobody played, the steady estimate alone
33, and it missed nothing. Almost all of those 110 were the model's
pitch head answering about a note still ringing inside its 0.77 s
window - which is exactly what this option exists to refuse. Letting
the model through "unless the estimate names something else" was not
enough: it says nothing when the estimate is silent, and the root
head was not held back at all.

### require_onset

*`src/state.rs`*

"Credit only what was struck": the model's branches may credit a class
only while it counts as struck just now - see `struck_just_now`.

The pitch head answers what SOUNDS: measured on a real recording it is
right about that in 94% of frames and still leaves 78% of notes with
some class lit that nobody played - an open string ringing in sympathy
is sounding, and so is the note before. Off by default.

Measured on a recording of single notes (`--probe`, 364 frames the app
would ask about): of 83 credits the model would hand out for a class
other than the one being played, two thirds carry no attack at all, and
almost all of those are the PREVIOUS note still inside the model's
0.77 s window - which is what this tells apart. The notes themselves keep passing on the CQT branch, which this
does not touch, so on that recording it costs nothing; an earlier
measurement of the whole crediting rule put the cost at 4 missed notes
in 49, so it is not free everywhere. Hence an option, not a default.

### collect_formula

*`src/state/formulas_mode.rs`*

Measured on a real recording, over 49 notes (dist/latency_stats.py):

  the three ways in together      110 false credits, 0 notes missed
  the steady estimate alone         33 false credits, 0 notes missed
  and with the take6 onset head     15 false credits, 4 notes missed

So the model's pitch head is struck out here. It answers "what is
sounding", and a string ringing on - or one resonating in sympathy,
which is why the fourth and the fifth kept lighting up - is sounding
without having been played. It was 99 of those 110.

The take6 onset head was tried as a gate here and dropped: live it
answered 0.09 for notes plainly being played, refusing far more than
it caught. The strike detector backs the ear instead - `EAR_BACKED`.

## Lines in order

### Semitone steps

*`src/state/line_harness.rs`*

A scale asks what an arpeggio does not: a step a semitone from the note
before. The analysis window still holds that note for half a second, so the
worry was a step credited before it is played. Measured October 2026 on
rising lines spliced from AtoA's notes, judged as Scales judges them - one at
a time, in order - each note cut to a length, and again with the note before
ringing 0.3 s under the next:

  seconds a note   D dorian   G mixolydian   D m7 arpeggio
  0.50             8/8        8/8            5/5
  0.45             8/8        1/8            5/5
  0.40 and less    1/8        1/8            1/5

Not one step in any run was credited before its note. Every credit came
0.37 to 0.45 s after the attack, semitone or not, and the arpeggio, with no
semitone in it, stops at the same speed. What limits a line is time: a note
shorter than the judge needs is gone before it is credited, and the line
waits for it. The scale exercises of Intervals carry the same limit Scales
has - about two notes a second.

### The ear holding a ring

*`src/state/judge.rs`, `ear_holds_only_a_ring`*

One at a time, the ear naming another note refuses the model. On a line
whose notes overlap - an arpeggio across strings, the note before left
ringing 0.3 s under the next - the ear went on naming that note for half a
second after the new one was struck, while the pitch head already answered
the new one at 0.88 against the old one's 0.94. By the time the ear named
it, the next note had been struck and the ear was settling again: the note
was never credited, and the line stopped there.

The ear is now overruled only where it is late: the target struck within
CLASS_FRESH and the class it names not. With STRUCK_FRESH instead it changed
nothing - the pitch head reached 90% of the strongest class a hop after the
22 were up. Measured on `line_harness` and the whole golden grid:

  D m7 arpeggio, 0.5 s a note, ringing   1/5 -> 5/5 credited, none early
  AtoA one at a time                     37 false / 33 notes, unchanged;
                                         median 260 -> 220 ms
  everything else in the grid            identical
