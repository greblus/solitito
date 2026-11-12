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

### diminished_heard_whole

*`src/state/chords_mode.rs`*

A diminished seventh asked for in Chords passes when the pitch head hears
all four of its notes above 0.5, whatever the chord name says.

Measured on two recordings of D♭°7 from the D string (`dim-g3`, `dim-g5`,
the model read through `chord_probe`, every 40 ms). The chord name was
`dim` in 39 readings of 514 and 6 of 547; nearly all the rest was a
dominant seventh on one of the chord's notes, G 7 most often - the weak D
and F the pitch head also hears are overtones of the B♭, and with the G
they spell it. The pitch head heard three of the four notes in 394 and 491
readings, all four in 162 and 165; the top E dies first.

Through the Chords bucket, the name alone first passed the chord 17.7 s
into `dim-g3`'s 23 s of it and never in `dim-g5`'s 21 s; with this rule,
0.75 s and 0.8 s after the first strum. Over the other chords of the song
in `dim-g3` (B♭maj7, D7, E♭maj7, G7, Cm7, F7, Dm7) it credits nothing.

Three of the four notes with nothing outside them passed more often
in the same simulation but is not taken: G B♭ D♭ is also E♭7 without its root.

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

### Falling lines and Harris scales

*`src/state/line_harness.rs`, `harris_scales_falling`*

The Harris lines of Arpeggios run mostly down, an eight-note scale with a
chromatic passing note in it: the major seventh between the flat seventh and
the root, or the sharp fifth between the fifth and the sixth. Measured
November 2026 on falling lines spliced the same way, judged in order. AtoA
holds only the white keys, so the black ones are the note a semitone below
read a semitone faster - an approximation, shorter and brighter than a
fretted note. Four runs at each speed:

  seconds a note   G bebop down   C sixth dim down   G mixolydian down
  0.50             9/9 (4 of 4)   9/9 (4 of 4)       8/8 (4 of 4)
  0.40 and less    1/9            1/9                1/8

No step was credited before its note, the passing notes included, and the
median credit came 0.36 to 0.41 s after the attack. The limit is the one the rising
lines have: about two notes a second.

AtoA holds some notes twice. The falling lines first took the later A3 and
stalled on it in all three, 6/8 to 7/9; the first take, the one the rising
lines use, goes through. The harness takes the first take of every note.

The rising lines were not repeatable to the note: four runs at 0.50 s gave
D dorian 4 to 8 of 8, G mixolydian 4 to 8 of 8, the D m7 arpeggio 3 to 5 of
5, against the single run of the table above. It was the harness, found
November 2026: Scales runs from E to e by default, in a position drawn for
every run, so the app asked for the scale from whatever note that position
starts on - from A or B as often as from D - while the audio was the line
from the root. The falling lines switched E to e off and came out whole
every time; the rising ones now do the same, and give the table's numbers
on every run.

### Enclosures and phrases

*`src/state/line_harness.rs`, `harris_enclosures`*

Three phrases of the Harris lines carry a note a half step below a chord
tone that the chord's scale may not hold: the enclosure (D B C E G B A over
Cmaj7), the drop to the third (C D# E B A G, a leap of most of an octave
down) and the third into the root (E B D C). Measured on the whole app in
Arpeggios, each line spliced at the pitches the app asks for - an octave up
where it drops below AtoA's lowest A - over C7 from the root, C6 diminished
and Dm7 from the third, three runs:

  seconds a note   enclosure   drop to the third   third into the root
  0.50             7/7         6/6                 4/4      every case, 3 of 3
  0.40             0           0                   0

No step credited before its note. The chromatic notes and the leap go
through like any other; the limit is the usual one.

### Ted Greene's exercises

*`src/state/line_harness.rs`, `greene_in_position`*

The exercise after Ted Greene walks every tone of a chord in one position:
right through it and back, or in groups - of three, four or five, each
group a tone on - so a note comes back after a few others, a dozen times
over. With the tensions the 13 lies a semitone under the flat seventh and
the 9 a tone above the root. Measured on the whole app in Arpeggios over C7,
Cmaj7, G7 and Am7 at the seventh fret, the whole position - 17 to 55 notes -
spliced at the pitches the app asks for, at 0.50 s a note:

  figure               without tensions        with tensions
  arpeggio in position 17-19 of 17-19, all     27-29 of 27-29, all
  groups of 4 up       24-28, all              44-48, all
  groups of 4 down     24-28, all              44-48, all
  groups of 3 up       21-24, all              36-39, all
  groups of 5 up       25-30, all              50-55, all

No note was credited before it was played. The first run of the groups
down stopped at the 25th note of 48, on every chord. It was the harness, not
the judge: the line starts on its highest note, at the very start of the
file, where the model has no history yet and answers late; the note was then
credited only when it came round again four notes on, and every step after it
waited the same. One note the line does not ask for, spliced in front, gives
the model its history - as it has in the app, running all along - and then
every figure goes through.

The groups of four in an order of their own (`harris::order`, numbered as
on Greene's sheet) were measured the same way, five of the 24 orders: 1, 5,
9, 16 and 20. Order 1 plays a group from its top tone down, and over a chord
of four tones the next group then began an octave above the note just
played - the same pitch class twice running:

  order 1, without tensions     C7, Cmaj7, G7    Am7
  as first built                13 of 28         12 of 24
  octave groups passed over     28 of 28         24 of 24

The line stopped at the third such octave. `ordered_groups` now passes over
a group that would begin on the class of the note just played, so in order 1
the groups climb two chord tones at a time. With that, every order measured
goes through whole, with the tensions (44-48 notes) and without (24-28), and
no note is credited before it is played. With the tensions a position holds
six classes to the octave and no group began on the last note's class to
begin with.

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

## Quick crediting

### struck_and_named

*`src/state/judge.rs`, `src/state/quick_credit.rs`; harness `src/state/ear_speed_harness.rs`*

Where the 0.4 s between a strike and its credit goes. On the user's sessions
(sesja-g1, gitara-g1 to g3, 388 credits in Intervals), how long before the
app's credit each witness had named the note's class - an estimate once it
had said so for two hops running, looked for from the credit before, at most
1.5 s back:

  witness                 named before the credit   sooner by (median)
  the ear, 8192 samples   388 of 388                128 ms
  the strike detector     366                       352 to 368 ms
  YIN, 2048 samples       355                       320 ms
  YIN, 1024 samples       359                       336 ms
  the rise, 2048 samples  387                       352 to 432 ms
  the detector and YIN 1024: 345; and the rise: 366; and either: 366

The ear's 128 ms is `NOTE_HOLD`: the credit comes eight hops after the ear
names the note, and the ear names it a quarter of a second after the strike.

The witnesses alone, on `line_harness`'s three rising lines (21 notes)
spliced from AtoA at several speeds - notes named before the next one
begins, the median time from the strike, and hops spent naming the note
before or some other class:

  each note stops the one before
  seconds a note   detector        ear 8192            YIN 2048       YIN 1024
  0.50             21, 22 ms       21, 262 ms          21, 54 ms      21, 26 ms
  0.30             18, 26 ms       21, 258 ms          21, 62 ms      21, 30 ms
  0.25             18, 22 ms       10, 232 ms          21, 64 ms      21, 32 ms
  0.20             18, 18 ms        0                  21, 58 ms      21, 26 ms
  0.15             18, 24 ms        0                  21, 66 ms      21, 30 ms
  hops astray      0 to 1          160 to 180 before,  0 to 6 other   5 to 8 other
                                   12 to 61 other

  the note before left ringing 0.3 s under the next
  0.50             21, 30 ms       20, 258 ms          21, 94 ms      21, 54 ms
  0.30             19, 30 ms       19, 270 ms          11, 70 ms      14, 50 ms
  0.20             20, 26 ms        1                   9             9
  hops astray      0 to 1          as above            98 to 165      107 to 149

YIN is the tuner's (`tuner::frequency`), on the end of the analysis window.
It reads one period, so two strings at once defeat it; the rise does not
care - see below. 1024 samples are taken: as many notes, half the wait.

The other side, on AtoA's 51 reviewed notes, each left to ring out
(`agreement_on_single_notes`): a strike report and an estimate naming the
same class for two hops within 8 hops of it.

  with the detector   the note played   a class nobody played   alone: runs of 3 hops on another class
  YIN, 1024           44 of 51, 32 ms   2                       64
  the rise            44 of 51, 34 ms   2                       194
  the ear, 8192        4 of 51          0                       150

The two false agreements are the same two for every estimate: a semitone
below the note at 56.24 s and at 60.78 s, in the attack itself. 8 hops for
`QUICK_AGREE` is the window of that measurement; `QUICK_TICKS` is two, as
`CREDIT_TICKS` is.

Through the whole judge (`lines_through_the_judge`, SOLITITO_QUICK=1 for the
option), the three lines judged as Scales judges them - credited of 8, 8
and 5, and the median time from the strike to the credit:

  seconds a note   ordinary           quick, YIN alone    quick, YIN and the rise
  each note stops the one before
  0.50             8 8 5, 400 ms      8 8 5, 26-46 ms     8 8 5, 26-46 ms
  0.40 to 0.25     1 1 1              8 8 5               8 8 5
  0.20             1 1 1              7 7 5               7 7 5
  0.15             1 1 1              7 7 4               7 7 4
  the note before ringing 0.3 s
  0.50             8 8 5, 400 ms      8 8 5, 46-94 ms     8 8 5, 30-62 ms
  0.40             1 1 1              8 5 1               8 8 5
  0.30             1 1 1              5 2 1               8 8 5
  0.20             1 1 1              1 2 1               8 8 5
  0.15             1 1 1              4 1 1               8 7 4

A line is judged in order, so one note missed stops it: the notes lost at
0.2 s and faster are strikes the detector did not report, left to the
ordinary path, which at that speed has no time. The harness reports a
first note credited 6 ms "before" it began: that is the label, which sits
after the attack - the splice starts 50 ms ahead of it, and on AtoA itself
a strike and an estimate agreed up to 30 ms before a label.

The quick ear costs 154 µs a hop of 16 ms on the measuring machine
(`what_the_quick_ear_costs`), and is computed only while the option is on
in a mode that uses it.

What is not measured: the live input. The harness feeds the judge every
hop; the app's judge reads the audio thread from the window's 16 ms timer,
after the sound card's buffer. The runs of the quick ear are counted on the
audio thread for that reason, hop by hop, and only the agreement is read
from the timer. Scales, Arpeggios and the fretboard trainer were judged
only when the chord model answered - at most every 40 ms and not before
half its window, 0.38 s, held sound - so the quick path has a clock of its
own, once per audio frame.

### The rise of the short spectrum

*`src/rise.rs`; harness `src/state/rise_harness.rs`*

The magnitude spectrum of the newest 2048 samples (`ShortSpectrum`, already
computed for flux) less the same spectrum 8 hops before, what is negative
dropped: the two windows share no audio, so a string only ringing leaves
nothing and a string struck leaves its partials, whatever rings beside it.
The rise is explained as `voices` explains a spectrum - the note with the
largest sum over five partials is taken, its partials struck out, the rest
asked again, up to four - and a note counts only where its own fundamental
rose at least as much as its mean partial, or the fifth above a struck
note collects that note's third and sixth partials and is named for it.
Nothing is read where the rise sums to less than a tenth of the spectrum.

On the lines above, the strongest risen note, two hops running:

  seconds a note   each note stops the one before   the note before ringing 0.3 s
  0.50             21 of 21, 38 ms                  21, 26 ms
  0.30             21, 34 ms                        21, 34 ms
  0.20             21, 34 ms                        21, 42 ms
  0.15             21, 50 ms                        21, 60 ms

It names every note at every speed, ringing or not - and on a held note it
goes on naming things: 38 to 56 hops on another class over the 21 notes at
half a second each, 194 runs of three hops on AtoA's notes left to ring.
So it is a witness for the moment after a strike and nothing else, which
is how `struck_and_named` uses it.

More than one note, measured and not used. Triads of AtoA's notes struck
15 ms apart, the fifth at full level, -6 and -12 dB and not played at all
(51 triads); classes named two hops running within 12 hops of the strum:

  fifth      roots   thirds   fifths   classes outside the triad
   0 dB      48      49       48       79
  -6 dB      50      49       45       77
  -12 dB     50      49       27       74
  absent     50      49        3       77

The judge with "struck only" credits the fifth 24, 16 and 10 times of 51 on
the same triads and 6 times where it is absent (see *A chord struck at
once* on the `experiments` branch), so the rise hears the quiet fifth far
more often and names it falsely less. But it names one and a half classes
outside each triad as well, and on the user's 17 strummed chords 60 of 68
chord tones with 74 classes outside them - four to a strum. Opening the
model's pitch head to every class it names would open it to those too; as
a gate for "struck only" it needs measuring through the judge, on the three
sides the detector's replacements were measured on, before anything is
built on it.

### The cells of the Harris lines through a key

*`src/harris.rs`; harness `cells_through_a_key` in `src/state/ear_speed_harness.rs`*

The six cells - thirds, triads, seventh chords, thirds led into, the bebop
arpeggio, the pivot arpeggio - taken through the chords of C major, the
whole pass spliced from AtoA at the pitches the app asks for (the black
keys a note a semitone below read faster) and run through the judge of
Arpeggios. The questions were the notes a half step under a chord tone,
and a class coming back within the cell or the next - the bebop arpeggio
begins and ends on the same class an octave apart. November 2026:

  seconds a note   ordinary                    quick crediting
  0.50             16 24 32 24 40 40 of the    the same, all of them;
                   same, all; 378-390 ms       30-42 ms
  0.30             0 to 3 notes                5 to 15 of 16 to 40
  0.20             0 to 3                      5 to 13

No note was credited before it was played. At half a second every figure
goes through whole either way.

Faster, the quick path stops where a class comes back: in the thirds, the
E of `E G` 0.9 s after the E of `C E`. The strike detector holds a class it
has reported within three seconds until the level has risen by a quarter
(`RECHECK`, `REFIRE_RISE`), and in a spliced line every note is as loud as
the last, so the second E is never reported and the note waits for the
ordinary path. The scales of *Quick crediting* above bring a class back
once, the closing root, and went through at 0.25 s all the same; these
cells bring one back every few notes. Whether a real pick - which stops
the string before it strikes it - raises the level enough has not been
measured; nor has letting the rise of the class itself answer for the
level's.

