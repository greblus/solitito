<!-- # Solitito — Real-Time Polyphonic Guitar Trainer -->

*[Wersja polska](README_pl.md)*

**Solitito** is a real-time guitar trainer written in **Rust**. It listens to your guitar
through an audio interface (recommended) or a microphone, recognises what you are playing,
and walks you through jazz standards, intervals, scales, arpeggios and interval formulas.

Recognition runs on a small neural network (7.3M parameters) exported to ONNX. Everything —
DSP, inference, UI — happens locally on the CPU. 

<div align="center">
<img height="320" alt="Solitito main window" src="docs/solitito_main.png" />
<img height="320" alt="Shell voicings" src="docs/solitito_main_shell_voicings.png" />
<img height="320" alt="Chord shape, enlarged" src="docs/solitito_chord_diagrams.png" />
</div>

Chord shapes are labelled with **intervals, not fingerings**, so one diagram covers
all twelve keys: the red dot is the root, and the shape moves. Clicking shape of chord 
enlarges it. The middle shot shows the same chord as **shell voicings** — third and
seventh over the root, with the fifth left out — which is the other thing the shapes
can be drawn as.

Solitito is heavily inspired by [Solo](https://www.solotrainer.app/), an Android/iOS guitar
trainer which I still use. Solo is great — but it hears one note at a time — when playing
intervals through changes, you have to mute the strings for each note to register.
I kept wondering how Yousician managed to recognise notes of chords polyphonically and
that's how the idea of Solitito was born.

Solitito is a desktop application for Linux and Windows. There are no plans for mobile versions.

Development started in December 2025. The data pipeline was rewritten from scratch more than
once before it worked. Most of what follows is a record of what turned out to matter.

---

## What it does

Pick a song, and the app shows one chord at a time. Play it. When the model hears the right
chord, the app moves on to the next one.

- **Fretboard** — a region of the neck is selected at random (a set of strings, four frets) and
  held; you are asked for notes that live inside it. For learning where the notes are in one
  hand position.
- **Chords** — full jazz standards. Green confirms an exact match, yellow accepts a triad or
  a common substitution (a rootless voicing, an `m7` read as `m`), red means the chord was
  detected but the signal is too weak to lock.
- **Intervals** — play the chord tones one at a time. You choose which degrees to practise
  (`1 3 5` for triads, `1 3 5 7` for sevenths, `1 3` for shell voicings). '3' matches both
  3 and b3, '7'  = 7 and b7, etc - depending on current chord's quality.
- **Scales** — sequential note practice from a scale definition.
- **Arpeggios** — chord tones in sequence over a progression, written as degrees so one
  pattern fits every chord in a standard. Two-octave jazz phrases, plus a generator that
  builds a fresh one after every pass.

<div align="center">
<img height="360" alt="Fretboard trainer" src="docs/solitito_fretboard.png" />
<img height="360" alt="Intervals as a grip" src="docs/solitito_intervals.png" />
</div>

<div align="center">
<img height="360" alt="A scale on the neck" src="docs/solitito_scales.png" />
<img height="360" alt="An arpeggio study as tablature" src="docs/solitito_arpeggios_tab.png" />
</div>

- **Formulas** — random selection from all subsets of the 12 chromatic functions that contain "1",
  that is 2¹¹ = 2048. A set of intervals drawn over a root, played in any order. Each function
  lights up when played, and the set is finished when all of them have been played; underneath,
  the nearest scale, with the formula's own degrees picked out of it, and the chords that fit
  inside it — point at one and the row above shows what it is built from. Pause turns the set
  blue and stops judging altogether: your turn, improvise inside it. The same formula can also
  be played over a chord, or carried across a whole standard — see
  [practising with formulas](docs/formulas-practice.md) for how to work with all three.

  A formula worth coming back to is kept with the **star** under it: it asks for a name, and
  the name goes on the list in Practice settings, where picking one draws that formula again
  over whatever key is on screen — a formula is key-independent, so only the set is stored.
  The cross on a row throws it out, and the star, filled in, drops the one on screen.

The Formulas mode is inspired by **An Improviser's OS** by Wayne Krantz — possibly the most
interesting approach to creative improvisation ever put together.

The book is available from [Wayne Krantz](https://waynekrantz.bandcamp.com/merch/wayne-krantz-an-improvisers-os-2nd-edition) directly.

<div align="center">
<img height="479" alt="Formulas" src="docs/solitito_formulas.png" />
</div>


### The chord, scales and arpeggio visual shapes are for the beginning

Every note mode can be shown three ways: a line of degree names, tablature, or the neck
itself — and, if you want, with fret numbers in the dots instead of degrees. The shapes
help while the intervals are still new: they say where the fingers go, and the fret numbers
help further still, for when a shape has to be found on the neck before its degrees mean
anything.

The idea is to stop using them. What this app is for is hearing intervals, not memorising
shapes: reading `1 ♭3 5 ♭7` off a diagram is playing a box, hearing it is playing music. As
the degrees become familiar, switch back to the line of names, and later practise without
the pictures at all.

---

## Read on

The rest of the documentation is in `docs/`:
| | |
|---|---|
| [**Practising with formulas**](docs/formulas-practice.md) | Step by step through the three formula exercises: what to play, what to listen for, and what each row on the screen is telling you |
| [Choosing an input](docs/audio-input.md) | What the device names mean on Linux, where settings are stored, and the diagnostic modes |
| [Settings](docs/settings.md) | Every option in the four tabs, and the shuffle, the pause and the chord strip on the main window |
| [How it works](docs/how-it-works.md) | Signal path, the model, and why single notes are not judged by the model alone |
| [Training data and results](docs/training-data.md) | The synthetic set, GuitarSet, and what each fix was worth |
| [Custom file formats](docs/file-formats.md) | Your own songs and scales |
| [Running it](docs/running.md) | Packages, and building from source |

## Detailed Project summary

`docs/` holds a long-form write-up of the whole system: architecture, the four
GuitarSet defects and what fixing each one was worth, the training procedure,
the measurements behind every design decision, and the hypotheses that
measurement refuted. Same document in two languages, Markdown and PDF.

| file | |
|---|---|
| [`docs/Solitito_project_summary_en.md`](docs/Solitito_project_summary_en.md) | [PDF](docs/Solitito_project_summary_en.pdf) |
| [`docs/Solitito_project_summary_pl.md`](docs/Solitito_project_summary_pl.md) | [PDF](docs/Solitito_project_summary_pl.pdf) |

## Repository and dataset

- Model and DSP weights: <https://huggingface.co/greblus/solitito-ai>
- Dataset v2 (renders and annotations): <https://huggingface.co/datasets/greblus/solitito_dataset_v2>

Those two hold binary artifacts only. All code lives here.

The `dist/` directory contains everything used to build the dataset and train the model:

| file | role |
|---|---|
| `dataset_generator_v2.py` | generates the GP5 **and** the annotations; self-tests all shapes |
| `verify_annotations.py` | checks that labels describe the audio (numpy only, no librosa) |
| `model_trainer.py` | training; runs on Kaggle, checkpoints to Hugging Face |
| `gen_weights.py` | sparse pseudo-CQT weights for the Rust side |
| `probe_root.py` | how often the labelled root is actually audible |
| `probe_quality.py` | where chord quality should come from: the head or the pitch vector |
| `probe_sources.py` | which GuitarSet chord annotation to use |
| `inspect_jams.py` | what is actually inside the JAMS files |
| `latency_material.py` | plucks with onsets known by construction, as a yardstick |
| `latency_ground_truth.py` | onsets and pitches of a real recording, for the same measurement |
| `latency_stats.py` | how late the app learns what was played, and how often it learns it wrong |
| `latency_rules.py` | what each crediting rule would cost: credits nobody played, notes missed |
| `gp5_to_arpeggio.py` | turns a Guitar Pro file into the degree notation Arpeggios reads |
| `hf_cleanup.py` | clears the checkpoint repository before a run started from scratch |

---

[1] This project uses the GuitarSet dataset by Qingyang Xi, Rachel M. Bittner, Johan Pauwels,
Xuzhou Ye & Juan P. Bello, available at <https://guitarset.weebly.com/>, licensed under
Creative Commons Attribution 4.0 International (CC BY 4.0).
