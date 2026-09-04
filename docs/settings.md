# Settings

Every option in the four tabs, and the controls on the main window.

[← back to the README](../README.md)

---

Four tabs: **Audio** for the input and the noise gate, **General** for how strictly what you
play is judged, **Practice** for what to play, and **App** for what the window shows.

<div align="center">
<img width="220" alt="Settings, Audio tab" src="solitito_settings2.png" />
<img width="220" alt="Settings, General tab" src="solitito_settings1.png" />
<img width="220" alt="Settings, App tab" src="solitito_settings4.png" />
</div>

**Practice** holds only what belongs to the mode on screen — a song has nothing to say in
Formulas, a formula nothing in Chords — so it is a different tab in each of them, and the
fretboard trainer, which has no settings of its own, does not show it at all:

<div align="center">
<img width="175" alt="Practice, Chords" src="solitito_settings3.png" />
<img width="175" alt="Practice, Intervals" src="solitito_settings5.png" />
<img width="175" alt="Practice, Scales" src="solitito_settings6.png" />
<img width="175" alt="Practice, Arpeggios" src="solitito_settings7.png" />
<img width="175" alt="Practice, Formulas" src="solitito_settings8.png" />
</div>

| Setting | Description |
|---|---|
| **Song / Scale** | Chooses the progression or scale for the current mode. The standard you pick is remembered — between sessions too — and follows you through every mode that reads it: Chords, Intervals, Arpeggios over the changes and Formulas over a standard. The scale the same way: Scales comes back to the one you left off on, not to the top of the list |
| **Study** | Arpeggios, in the study exercise: which phrase to walk. The shuffle switch means the KEY here: a new one after every pass. An arpeggio is a phrase, so there is nothing in it to shuffle, and the study itself is left as chosen — the generator is a row in the list for anyone who wants a fresh phrase as well. The three chord-quality studies come first, then the broken thirds and triplets, then plain two-octave runs, and last a generator that builds a fresh phrase after every pass. The second combo beside it holds the key |
| **Exercise** (arpeggios) | *Study in a key* stands on one chord — its quality is set below, its key beside the study, and the shuffle draws a new key after every pass. *Over the changes* takes the chords from a tune and builds one arpeggio per chord |
| **Direction** | Over the changes: ascending, descending, or alternating from either side. Descending is the shape the studies use — from the root **downwards** through the chord's own tones, not the ascending phrase read backwards |
| **View** | Three ways to show the exercise: the line of degree names, tablature, or the neck itself. Kept per mode and remembered — a scale is an order of notes and reads as tablature, a grip is a hand shape and says nothing until it is on the neck. Tablature by default in Scales and Arpeggios, the neck in Intervals and the fretboard trainer. A phrase — a scale or an arpeggio — is drawn on the neck it is played on, with the note due ringed in white and every place already played green. An up-and-down phrase comes back to its own places, so by the turn the shape is green all through and the ring is what says where the player is on the way down — the tablature rings it too, which is what tells you where you are in a scale dealt in a drawn order; the picture is the size of a hand however long the phrase is, so a thirty-note study stays readable. A scale is drawn in a position drawn at random — the shape is the same everywhere, and the point is to know it wherever the hand lands. In the fretboard trainer the picture is the REGION: the strings named down the left, the ones out of play dimmed, the frets in play, and — once the note asked for is played — every place inside the region where it lies, in green. Nothing is drawn before that, and a wrong note is not drawn at all: where the note lies is the exercise. Each dot carries the degree as the exercise writes it, so a scale spelling its altered second `#2` reads `♯2` on the neck too. In Intervals the set is drawn as a GRIP: a chord box like the shape diagrams, strings across and frets down, with the position beside it — a strip has the ORDER of the notes on its horizontal axis, which says nothing about where the fingers go. With **Lead the voices** on — the default — each grip is taken where the fingers have least to move from the one before, so a progression walks itself across the neck. Off, every chord is taken wherever the neck is drawn to offer it, which is how one shape is learned all over the neck instead of in one corner of it; the shuffle switches the leading off as well. With the shuffle on there is no line to lead: the chords come in a drawn order and each grip is taken where the neck offers it. Which string a dot sits on is the octave, so nothing needs an apostrophe; what has been played lights green either way |
| **Fret numbers instead of degrees** | In that drawing only. The root keeps its colour |
| **Key** | Scales only: the tonic. With random order on, it is redrawn after each pass |
| **Intervals** | Which degrees to practise. `1 3 5` for triads, `1 3 5 7` for sevenths, `1 3` for shell voicings. `3` matches both major and minor thirds, `5` matches perfect and diminished fifths, according to the chord quality |
| **Show AI Debug in Main Window** | Shows the raw prediction on the main screen |
| **Input** | Which capture device to open. *System default* follows whatever the OS is set to. Saved, and falls back to the default if that device is gone |
| **Channel** | Which input of that device to listen on — a guitar in socket 2 of an interface is channel 2. Shown only when the device has more than one, and there is no mixing option: averaging the inputs pulls in whatever is on the other socket and costs 6 dB |
| **Noise gate** | Threshold in dBFS. The bar below shows the current input level on the same scale, with the threshold marked in red — set it just above the noise with the strings untouched |
| **Bass Boost** | Digital amplification of the lowest CQT bins. Useful for laptop microphones, which usually roll off the low strings |
| **Lock chord quality until new attack** | Holds the recognised quality until you strike the strings again. Without it, a held `m7` turns into `m` as the seventh dies away |
| **Judge short strums on the attack** | For chords struck and released rather than held. One clear reading of the target counts, and the decay that follows cannot undo it. A wrong chord still fails |
| **Credit only what was struck** | Note modes: a note may be credited only where the onset head also heard an attack — the chord-name reading included, which used to slip through and credit whatever note the model's root happened to be. Measured on a recording of single notes, two thirds of the credits it hands to a note other than the one being played carry no attack at all — almost always the previous note, still ringing inside its 0.77 s window. Off by default: a note whose attack the head misses then has only the CQT branch left |
| **Play the notes one at a time** | Offered in Intervals and the fretboard trainer. In Scales and Arpeggios the rule is in force whatever the setting says — nobody strums a scale — and Chords and Formulas have their own. Off, a strummed chord passes its intervals one after another — the pitch head is polyphonic and reports every tone at once. On, nothing but the single-frame estimate counts, and only once it has held for three audio frames: measured over 49 notes, the four ways in together credited 110 things nobody played, the steady estimate alone 33, and it missed none |
| **Random order** | The shuffle icon on the toolbar. In the note modes it shuffles the tones inside each chord; in Chords it reorders the progression, and in Scales it redraws the key after every pass |
| **Shuffle the chords as well** | Intervals and Arpeggios only, and only with the shuffle on. Off, the progression stays as written and just the tones move — shuffled intervals walking a real progression turn into tunes. On, the chords are drawn at random too, which is the more abstract exercise |
| **Exercise** (formulas) | *Formula in a key* is the mode as it was. *Over a chord* plants the same formula on one drawn chord, and *Over the changes* plants it on every chord of a tune in turn — the formula is the constant, the harmony moves under it |
| **Placement** | Over a chord: which kind to draw — one that spells the chord out, one that colours it, one outside it, or any. The screen shows the formula's functions read from the chord's root, with the chord's own tones in blue, and counts them |
| **Notes in a formula** | Formulas only: how many functions each drawn formula has, the root included |
| **Key** (formulas) | The root to read them against, or a fresh one drawn per formula |
| **Must contain** | Only draw formulas holding these functions, e.g. `b3 b7`. Empty draws from all 2048 |
| **Show note and chord names** | Formulas only: the letters under the functions and under the chords |
| **Show the nearest scale** | Formulas only: the closest scale you already know, spelled out with the formula's own degrees picked out |
| **Show the chords that fit** | Formulas only: chords playable without leaving the set, written as degrees. Only the fullest — over the major scale that leaves exactly the seven diatonic sevenths |
| **Favourites** | Formulas only: the star keeps the formula on screen under a name, and the list draws it again. A cross on a row throws it out |
| **Play the notes in order** | Formulas only: the set has to be walked lowest function first. Off, it is a set — any of them, in any order |
| **Play the notes in order** (Intervals) | The set has to be taken in the order it is dealt, and that note is ringed on the grip. Off, which is the default: the three notes in any order, shuffled or not — a grip is three notes under one hand, and which finger lands first is not the exercise. Not offered in Scales or Arpeggios, where the order IS the exercise |
| **Console debug** | Prints a line for every function credited, with what was heard. A window on the judging; on Windows a release build has no console to print it to |
| **End on the root again** | Scales only: the run reads 1 2 3 4 5 6 7 1, the last one an octave up. It is a step of its own and has to be played |
| **Show chord shapes** | The diagram thumbnails under the chord name in Chords mode |
| **Shapes** | Two boxes: the full grips and shell voicings — third and seventh over the root. Both ticked draws both, neither draws none. Past four shapes they are drawn in two rows. A `m7b5` has no shell of its own: its shell is the `m7` shell note for note, since the fifth is the only place the two differ, so that is what is drawn, captioned *substitute: the m7 shell*. With shells the only shapes on screen that grip is the one being asked for, so playing it passes green; with the full shapes drawn too, a `m7` reading means the flat fifth was missed and it stays yellow. A diminished seventh has nothing to leave out — its four notes are all a minor third apart, with no third-and-seventh pair to keep — so it keeps its full grips, captioned with what they also are: a `7b9` without its root, from a semitone below any of its notes |
| **Startup mode** | Which mode the app opens in |
| **Language** | Auto (from the system locale), Polski, English. Applied immediately, no restart |
| **Chord confidence** | How sure the model must be of the chord *name* before it counts (Chords mode) |
| **Note threshold** | How sure the model must be that a *single note* is sounding (Intervals / Scales / Arpeggios) |
| **Hold time** | How long a correct chord must be held before advancing. Chords only: a single note is credited as soon as it is recognised, which in the note modes and the fretboard trainer is a fixed 0.12 s |

The line under `Channel` says what actually opened — device, sample rate, channel count
and sample format. `./solitito --help` lists every option. `./solitito --devices` prints the same information for every device the backend can see, and `./solitito --bench` times one model inference — the app asks the model every 40 ms while a chord rings, so that figure is essentially the whole of its CPU load. A release build has the `windows` subsystem, so on Windows these modes write to the console that launched them; started with no console at all — from a shortcut carrying the flag — the program opens one of its own and waits for a key, so the report can be read.

---

## The main window

Three controls sit outside the settings panel, on the window itself.

**The strip along the bottom** shows the chord just played, the one being played, and the one
after it. The chord left behind keeps the colour it earned — green for an exact match, yellow
if a triad or a substitution got it through — so a pass stays readable once the app has moved
on.

**The shuffle** means something different in each mode, because the modes have different
things worth randomising:

| mode | what the shuffle draws |
|---|---|
| Chords | the order of the standard |
| Intervals | the tones inside each chord; the progression stays as written unless **Shuffle the chords as well** is on |
| Scales | the key, after every pass |
| Arpeggios | the key in the studies, the chord order over the changes |

Scales and Arpeggios never shuffle the order of the notes. A scale not walked in order is not
that scale, and a study dealt at random is not that study.

A scale moves after every pass whether the shuffle is on or not: it works through the three
strings worth starting from and turns round each time, up and down alternately. That is
counted, not drawn — three strings and two directions are coprime, so six passes cover all six
ways of taking it. Drawn instead, a fair coin gave five descents in a row, which reads as a
broken exercise.

**Pause** stops the progression while the colours go on reporting whether the chord is right,
so you can stay on one shape and work it out. While paused, the arrows either side of the
strip step back and forth through the progression.
