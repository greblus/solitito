# Training the models

[← README](../README.md) · [Po polsku](training_pl.md)

The app uses two models, and each has its own trainer in `dist/`. They share
no weights and no input, only the app:

| app file | what it does | trainer |
|---|---|---|
| `best_model_v2_take6_onset.onnx` | chord root, quality and the notes sounding, plus the older attack head (CQT input) | `model_trainer.py` |
| `short_onset_masking_v2.onnx` | which pitch class was just **struck** (short-spectrum input) | `strike_trainer.py` |

The app downloads both from <https://huggingface.co/greblus/solitito-ai>, and
the release packages carry them. How the app uses the strike model is in
[How it works](how-it-works.md). Both trainers keep their snapshots in
`greblus/chord-model-snapshots` on Hugging Face.

## The chord model: `model_trainer.py`

The trainer that produced `best_model_v2_take6_onset.onnx`, unchanged except
for two things: the ONNX export is pinned to the TorchScript exporter
(`dynamo=False`), and phase 4 does not train again when its checkpoint already
exists. On Kaggle: a GPU notebook with the dataset attached and an `HF_TOKEN`
secret, then paste the whole file and run it. `RUN_TAG` names the run.

It runs four phases, each resumed from its Hugging Face snapshot:

1. main training of the chord network;
2. the pitch threshold;
3. head fine-tuning, off (`RUN_PHASE3 = False`);
4. the attack head, `fc_onset`, trained with everything else frozen. It writes
   `checkpoint_<RUN_TAG>_onset.pth` and `best_model_<RUN_TAG>_onset.onnx`, the
   file the app loads.

With `RUN_TAG = "v2_take6"` and `checkpoint_v2_take6_onset.pth` on Hugging
Face, phase 4 only exports. The exported file answers bit for bit like the
released `best_model_v2_take6_onset.onnx` on all four outputs. Training it
again would overwrite the checkpoint the app's file came from, which happened
once, on 2026-09-20. A new attack head needs a new `RUN_TAG`.

The app reads root, quality and pitch, and also the fourth output,
`onset_logits`. The judge still uses it: "Credit only what was struck", and
whether a chord was struck. The strike model works beside it and does not
replace it.

## The strike model: `strike_trainer.py`

One file with nothing else from the repository. On Kaggle: a GPU notebook with
GuitarSet attached and an `HF_TOKEN` secret, then paste the **whole** file and
run it. Locally: `python dist/strike_trainer.py --help`; every setting at the
top of the file has a command-line flag.

```python
RUN_TAG = "v2_take7_masking_v2_repro"
MODE = "train"  # train, export_only
HF_REPO_ID = "greblus/chord-model-snapshots"
USE_HF = True
INITIAL_ONSET = "hf:checkpoint_v2_take7_onset_best.pth"
ONSET_EPOCHS = 12
ONSET_MASKING_PAIRS = True
ONSET_GAIN_DB = 6.0
```

- `train` trains the network, or resumes it, then writes the app's file.
- `export_only` rebuilds the app's file from a finished run's
  `checkpoint_<RUN_TAG>_onset_best.pth` and the threshold in its summary.
  It needs no dataset. Without the summary, set `EXPORT_ONSET_THRESHOLD` to
  the threshold that run chose; do not guess it from the epoch log. Missing
  inputs are an error, never a reason to start training.
- `USE_HF=False` runs entirely locally, without an account. With HF on, an
  authentication or network error is an error. The trainer never takes it to
  mean "empty repository, start fresh".

### What a run writes

Into `OUTPUT_ROOT/RUN_TAG/`, and to `HF_REPO_ID` when HF is on:

- `short_onset_<name>.onnx`: **the file the app loads.** `<name>` is `RUN_TAG`
  without `v2_take7_`: the released run `v2_take7_masking_v2` gives
  `short_onset_masking_v2.onnx`, the default run
  `short_onset_masking_v2_repro.onnx`. Input `short_features [batch,770,time]`,
  output `onset_logits [batch,12,time]`. In the metadata:
  - `onset_threshold`, which the app reads;
  - `onset_history_frames`;
  - the feature contract.

  Writing the metadata is checked not to change a single answer.
- `checkpoint_<RUN_TAG>_onset_best.pth`: the chosen weights, the parent for a
  later fine-tune and the source for `export_only`.
- `checkpoint_<RUN_TAG>_onset_last.pth`: the last finished epoch, with
  optimizer, random state and the best weights. A run with the same tag
  resumes from it.
- `training_summary_<RUN_TAG>.json`: configuration, data hashes, the chosen
  threshold, the whole validation threshold curve and the test results.

### Training the released strike model again

The defaults are the recipe of the released `short_onset_masking_v2.onnx`:
- fine-tuning from `checkpoint_v2_take7_onset_best.pth`, the `v2_take7` run;
- 12 epochs, masking pairs on;
- 96/96/96 synthetic groups, with GuitarSet players 04 and 05 as validation
  and test;
- levels spread by ±6 dB;
- seed 20260923.

The only difference is `RUN_TAG`, `v2_take7_masking_v2_repro`. A new tag is
what makes it train. Under an existing tag the trainer resumes from that tag's
snapshots, and for a finished run it only writes the files again.

How close a new run comes:
- **On CPU:** this file trains bit for bit like the scripts the released
  model was trained with. That was checked from scratch and from a parent
  checkpoint: the same data, batches, losses, weights, threshold and ONNX
  outputs.
- **On a Kaggle GPU:** a run with the default `RUN_TAG` reached the same
  weights as the released run. Its summary is kept in
  `dist/training_summary_v2_take7_masking_v2_repro.json`.
- **The exported file:** `export_only` on the released run's checkpoint gives
  a file that answers bit for bit like the released one.

To put a new strike model into the app, either:
- name it `short_onset_masking_v2.onnx` and put it beside the binary, or
- change `MODEL` in `src/strike.rs` and `STRIKE_MODEL` in
  `.github/workflows/release.yml`.

`./solitito --check` prints which strike model it found and its threshold.

### Where the parent comes from

The strike network was trained in three runs, each starting from the previous
one's best weights:

1. **Rise**: the network trained from scratch. The branch `rise` keeps this
   experiment: control vs Rise with identical starting weights and batches,
   then pair losses and ringing weights, which did not help.
2. **`v2_take7`**: fine-tuned.
3. **`v2_take7_masking_v2`**: masking pairs added.

Those runs trained the network as a branch of one combined graph, beside the
frozen chord network, and the app's file was cut out of it
(`dist/extract_onset_branch.py`). The branch never read the chord input, so
the trainer now writes the network on its own. `INITIAL_ONSET=""` starts from
scratch with this file's recipe. That is a new experiment, not a repeat of the
chain.

### The data

- **GuitarSet**, whole solo and comp takes, mono mix or mic (chosen
  automatically when only one is attached). Note starts come from the
  annotations, so they are not verified pick attacks. Players are split
  whole: 00–03 train, 04 validation, 05 test.
- **Synthetic plucks** (Karplus–Strong, `onset-ks-v2`), attack times exact by
  construction: holds, re-plucks of a ringing note, octaves, an added third or
  fifth, triads re-strummed.
- **Masking pairs**: a root and a third ringing, then an upper fifth struck at
  −18, −12, −6 or 0 dB relative to them, measured over its first 96 ms. Each
  group comes in three variants: background alone, the fifth alone, both
  together. They share the background and gain.

The input is 770 short-spectrum features per 16 ms hop. The app computes the
same frames (`ShortFeatures` in `src/strike.rs`). A shared fixture
(`dist/fixtures/`) holds the trainer and the app to the same numbers, bit for
bit, from both test suites.

Training spreads each block's level by ±`ONSET_GAIN_DB`. A player 11 dB
quieter than that spread missed half the strikes with the released model.
The app now levels the input before the model (`Leveller`), so the released
model needs no change for that. A wider spread is a separate experiment for
a new `RUN_TAG`.

### Resuming

The `_onset_last` snapshot continues exactly at an epoch boundary; a
half-done epoch is repeated. Resuming checks that sources, splits,
annotations, feature hashes and the training settings are unchanged. If any
differ, the trainer stops with `Cannot resume onset training with changed
data` (or `configuration`). Keep the checkpoint and restore the data. Do not
rename the run to force a restart.

Regenerated synthetic WAVs can change their file hash: the `PEAK` header
records the time of writing. That difference alone is accepted, but only when
the feature file computed from the WAV is identical. `rise/resume_data_check.json`
says what was compared.

A finished run leaves its data (`features/`, `onset-prepared-*`) in the run
directory. On Kaggle, saving that output takes a while after `Done.`.

### Tests

```bash
cd dist && python -m unittest test_strike_trainer
```

Tests that train or export need `torch`, `onnx` and `onnxruntime`, and are
skipped without them. On the Rust side, `cargo test` runs
`features_are_the_trainers_to_the_last_bit` on the same fixture. After a
deliberate change to the features, regenerate the fixture with
`python test_strike_trainer.py --write-fixture`.
