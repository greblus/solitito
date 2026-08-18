# Preparing the dataset (v2)

The procedure after the pipeline was rewritten from scratch. The previous version
encoded sample IDs as an audible "barcode" and read them back from the render —
that turned out to be fragile and **destroyed the whole dataset**. The generator
now knows which measure each block occupies and **emits the annotations
directly**. No step is left that could lie.

---

## Files

| file | role |
|---|---|
| `dataset_generator_v2.py` | generates the GP5 **and** the annotations and reference |
| `verify_annotations.py` | checks that the labels really describe the audio |
| `model_trainer.py` | training (Kaggle + checkpoints on Hugging Face) |
| `gen_weights.py` | pseudo-CQT weights into `dsp_weights.json` for the Rust app |

---

## Step 1 — generate the GP5 and the annotations

```bash
python dataset_generator_v2.py
```

Produces:

```
synth_dataset.gp5          # 394 blocks, ~39 min, ONE file = one render
synth_annotations.csv      # file,start,end,label — ready, for both tones
synth_reference.csv        # id,label,kind,strings,frets — for inspection
```

Before generating, the script runs a **self-test**: for each of the 21 movable
shapes it checks at every fret whether it really produces the declared intervals.
A typo in the shape table stops generation instead of silently poisoning the
dataset.

Content: 12 roots × {maj, min, maj7, dom7, min7, m7b5, dim7, sus4, aug} in
several fretboard positions, plus all single notes (6 strings × 16 frets = 96).

**Block structure** (6 s at 120 BPM): measure 1 = attack, measure 2 = sustain
(tie), measure 3 = silence. The annotation covers `[start+0.05 s, +3.2 s]` —
attack and sustain, without the decay tail.

## Step 2 — render in a DAW

1. Open `synth_dataset.gp5` and export the guitar track as a **DI** signal (wav).
2. Set the DI level (see below), run it through NAM and save under **exactly
   these names**:
   - `synth_dataset_clean.wav`
   - `synth_dataset_eob.wav`

### Render format

**48 kHz**, not 44.1, for two independent reasons:

- **NAM runs natively at 48 kHz.** Its models are trained at that rate; at 44.1
  the plugin resamples internally, which can subtly change the tone. The point is
  to capture two specific amp sounds faithfully, so there is no reason to add a
  conversion.
- **Decimation to 16 kHz is exact.** The trainer and the app work at
  `SR = 16000`; `48000/16000 = 3` is a clean integer ratio, `44100/16000 =
  2.75625` is not.

Bandwidth is not the bottleneck: the CQT spans 6 octaves from C1, i.e. up to
~2 kHz, while Nyquist at 16 kHz is 8 kHz.

Bit depth and channel count are free — calibration handles PCM 16/24/32-bit and
32/64-bit float, mono and stereo. The trainer converts to mono 16 kHz anyway.

### Signal level before NAM

NAM models an amplifier, so it is **sensitive to input level** — that is the
entire difference between clean and edge of breakup.

- **No compression or limiting.** Attack and decay are information the model
  learns.
- **One gain for the whole file**, no automation, no per-segment normalisation.
  The same chord in different places has to hit the amp the same way.

> The `synth_` prefix instead of `01_` is deliberate. GuitarSet names its files
> `01_BN1-129-Eb_comp_mix.wav`, where `01` is the guitarist number — with numeric
> IDs the trainer pasted synthetic labels onto unrelated recordings.

## Step 3 — calibration (mandatory, 10 seconds)

```bash
python dataset_generator_v2.py --calibrate synth_dataset_clean.wav
```

Measures which second the first attack falls on. If the DAW added silence at the
start, the script reports the number to put into `RENDER_OFFSET_SEC` — set it and
run the generator again (the GP5 does not change, only the timings are
recomputed).

Do not skip this step. A one-measure shift is invisible to the eye and turns the
whole dataset into noise.

## Step 4 — verification

```bash
python verify_annotations.py
```

For each file it compares the label against the **actual audio content** (the
dominant pitch class in the window versus the labelled root). The script does not
need librosa — numpy is enough (its own WAV reader plus chroma from an FFT).

Reference values from the real v2 render:

| | clean | eob | chance |
|---|---|---|---|
| top1 | 87 % | 77 % | ~8 % |
| **top3** | **100 %** | **98 %** | ~25 % |

**Top3** is the meaningful figure — in a chord the root is often quieter than the
third or the fifth, so a low top1 means nothing on its own. Threshold:
`top3 > 75 %` means you can train, `top3 ≈ 25 %` means the labels do not describe
the audio, **stop**. (With librosa available the script uses the more accurate
`chroma_cqt` and then looks at top1 > 45 %.)

It also checks the block period from the energy envelope and the offset between
annotation start and attack. Those two tests are **complementary**: shifting the
annotations by a whole block still lands on *some* attack, so the timing test will
not see it — only the label comparison catches it (top3 drops to ~28 %). That is
exactly the class of error that destroyed the previous dataset.

## Step 5 — training

Upload the dataset (wavs + `synth_annotations.csv`) to Kaggle together with
GuitarSet. `model_trainer.py` reads GuitarSet from the JAMS files independently —
those annotations were always correct and they are the main source of real
playing.

### Run name

There is a single constant at the top of `model_trainer.py`:

```python
RUN_TAG = "v2_take1"        # next run: "v2_take2" and so on
```

Every name — checkpoints, ONNX exports, logs, cache, locally and on Hugging Face —
derives from it. **Changing this one line means a clean start**, and the previous
run stays untouched on HF as a backup.

Always change it when starting with a new dataset or after an architecture change,
otherwise the trainer resumes from weights trained on something else.

Watch for this line in the training log:

```
🎓 TRAIN (no augmentation): Root=..%  Qual=..%  Exact=..%
```

It answers one question: **can the model learn its own training data at all?** If
`Qual < 90 %`, the fault lies in the features or the labels rather than in
generalisation, and adding epochs is pointless.

### Which root figure to read

`probe_root.py` (360 GuitarSet files, 30653 windows) measured how often the
labelled root **is audible at all** in a training window, and reported **64.1 %**
for a 0.77 s window.

**That was a trap.** GuitarSet records each excerpt twice: accompaniment (`_comp`)
and improvisation (`_solo`) over the same progression — and the chord annotation
is **identical** in both. A solo file carries a monophonic line, so the chord root
is absent by definition. Half the dataset was dragging the figure down.

With the improvisations filtered out (`GUITARSET_SOLO_MODE = "mask_chord"`), the
root sounds in **97 %** of accompaniment windows. There never was a 64 % ceiling.

The wider lesson: before accepting a limit as a property of the phenomenon, check
whether it is a property of the dataset. Four training runs went into optimising a
model against data of which half described something other than what it sounded
like.

Read the split in the log:

```
root: audible=..%  (..% of windows)   silent=..%
```

- **`audible`** — the usable metric. A student practising a chord plays it with
  the root, so this column reflects the app's real behaviour. Ceiling: 100 %.
- **`silent`** — guessing from context. A low value is not a defect.

By default (`MASK_ROOT_WHEN_SILENT = True`) the trainer **does not compute the
root loss** on windows without an audible root — training on them taught
memorisation of GuitarSet progressions rather than hearing, and the shared encoder
received a gradient in conflict with the pitch objective. The pitch and quality
heads still learn from those windows.

> The probe also checked whether switching `GUITARSET_CHORD_SOURCE` to
> `"performed"` is worth it: the intended and the played root differ in **0 of
> 43056** comparisons. For the root that switch changes nothing — for quality it
> changes everything, see below.

### Which chord annotation to use (`GUITARSET_CHORD_SOURCE`)

GuitarSet has two: the chord **from the chart** (`instructed`) and the chord
**as played** (`performed`, transcribed from the hexaphonic pickup).
`probe_sources.py` counted both across 360 files — the segment totals are
identical, so this is not a choice between more or less data but a relabelling of
the same recordings:

| quality | instructed | performed | |
|---|---|---|---|
| maj | 2640 | 2106 | −534 |
| min | 960 | 460 | **−500** |
| min7 | **0** | **360** | **+360** |
| maj7 | **0** | **430** | **+430** |
| dom7 | 480 | 694 | +214 |
| m7b5 | 240 | 134 | −106 |
| sus | 0 | 132 | +132 |

Read those together: **five hundred segments the chart calls `m` were played as
`m7`.** With `instructed` the trainer taught the model to call a voicing
containing a minor seventh a plain minor chord — and that is exactly what showed
up in the app as `Gm7` recognised as `Gm`.

On top of that, `instructed` contains **not a single** `maj7` or `min7`, so both
classes came only from the two synthetic renders. Hence 100 % on validation (the
same instrument on both sides of the split) and fragility on a real guitar.

Use `"performed"`. Never `"both"` — that is the same fragment twice with
conflicting labels, and the duplicate leaked between train and validation.

---

## Phase 4 — the onset head

The app does not need to know what is SOUNDING. It needs to know what was
STRUCK, and those differ by more than they sound like they should: an open
string ringing in sympathy is sounding, the note before is still sounding, and
in the Formulas mode a mark never expires, so anything the model calls present
stays lit for the rest of the exercise.

Measured on a real recording (`dist/latency_ground_truth.py` for the truth,
`dist/latency_stats.py` for the figures), the pitch head is right in **94 % of
frames** and still leaves **78 % of notes** with some class lit that nobody
played. Per frame it is a good answer to the wrong question.

`fc_onset` answers the right one: which pitch classes were struck inside the
last `ONSET_FRAMES` (6 frames, 96 ms). Three things make it cheap:

* the time axis survives the encoder — the convolutions pool frequency only —
  so the transformer already holds one token per frame, and only CLS was ever
  read;
* the labels exist already: `note_midi` gives every onset to the frame, which is
  what `NOTES_BY_PATH` is built from;
* the trunk is frozen, so the chord path cannot regress by a decimal. The phase
  either produces a head worth using or leaves the model as it was.

The head is fed the newest frame **and** the newest minus what it looked like
`ONSET_LOOKBACK` frames ago. An attack is not a state but a change: every frame
covers 512 ms of audio, so the note before appears in both and cancels, while
the note just struck appears in only one.

`OnsetDataset` samples its own windows rather than reusing the phase-1 ones,
which stride four frames and drop quiet windows through the energy gate —
exactly the moment a new note arrives under one still ringing. Every offset from
an attack is sampled, plus as many windows with no attack in them, or the head
would learn that something is always being struck.

To run it: `RUN_PHASE4 = True` (the default) and leave `RUN_TAG` alone — the
phase reads the existing checkpoint, and bumping the tag would start the whole
model from scratch. It writes `best_model_<tag>_onset.onnx`, which still carries
the old three outputs under their old names, so the app keeps working with it
before anything is wired to the new one.

The F1 the phase prints is not the number that decides anything. That one is
false credits per note, from `dist/latency_stats.py` on a recording — the same
measurement that produced the figures above, so the two are comparable:

| rule | false credits | notes affected |
|---|---|---|
| ear, every frame | 122 | 38 / 49 |
| ear, held 4 frames | 33 | 27 / 49 |
| model pitch ≥ 0.6 | 91 | 37 / 49 |
