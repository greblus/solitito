# Training take7

Copy **the whole `dist/model_trainer.py`** into the existing Kaggle trainer
notebook and run it with GuitarSet attached. The script prepares the synthetic
onset examples itself. Full chord training also needs the original synthetic
chord WAV/CSV dataset. GPU training stays on Kaggle.

Configuration at the top:

```python
RUN_TAG = "v2_take7"
MODE = "auto"
BASE_RUN = "v2_take6"
HF_REPO_ID = "greblus/chord-model-snapshots"
USE_HF = True
INITIAL_ONSET = ""
ONSET_EPOCHS = 12
```

- `auto`: resume this run; otherwise reuse `checkpoint_v2_take6_best.pth` and
  train only the Rise branch. Without a base checkpoint, train both branches.
  The final export is one ONNX with four outputs.
- `onset_only`: require a ready chord base; never silently train the chord model.
- `full`: ignore take6 and train the whole model. It still resumes its own run.
  **Choose a new `RUN_TAG` for a completely fresh experiment.**
- `export_only`: combine the already exported chord and Rise branches without
  training, GPU, feature cache or datasets. See the recovery instructions below.
- Set `USE_HF=False` for local-only training without an account/token. With HF,
  configure the existing `HF_TOKEN` Kaggle secret. For your own model, use your
  own existing HF repository; an empty repository is supported. An authentication
  or network error does not count as an empty repository.

The script loads chord weights strictly. It removes only the retired `fc_onset`
keys if present; missing/incompatible chord layers are an error. The chord base
is not in the Rise optimizer. Root, quality and pitch outputs keep their take6
architecture. The old onset head is neither trained nor exported.

Rise keeps its current causal architecture and 770-feature input. By default it
starts with fresh onset weights: **a take6 chord snapshot does not contain Rise
weights**. To continue from an existing Rise model, set `INITIAL_ONSET` to its
PyTorch `short_onset_best.pt` (not its ONNX). This option must match the existing
Rise/DSP contract. A saved take7 onset checkpoint takes precedence.

Outputs in `/kaggle/working/<RUN_TAG>/` and, when enabled, on HF:

- **`best_model_<RUN_TAG>.onnx`**: the single final model with root, quality,
  pitch and onset outputs;
- `checkpoint_<RUN_TAG>_best.pth`: chord base;
- `checkpoint_<RUN_TAG>_onset_last.pth`: last complete epoch, optimizer, RNG,
  history and best onset checkpoint together;
- `checkpoint_<RUN_TAG>_onset_best.pth`: selected onset weights;
- `training_summary_<RUN_TAG>.json`: final model, source hashes, thresholds,
  export parity and training metrics.

Branch ONNX files remain local intermediate artifacts for validation and recovery;
new training runs publish only the combined ONNX to HF. Existing files from older
runs on HF are left intact.

The onset last snapshot permits exact continuation at completed epoch boundaries
under the same runtime. A partly executed epoch is repeated. The feature cache is
reused locally or regenerated from attached datasets; old absolute cache paths
are not required. An interrupted export resumes from saved weights without another
optimizer step. The chord phases retain the original best-checkpoint resume
policy; they do not promise exact mid-epoch replay.

The combined model has two independent inputs:

- `features`: `[batch, 48, 168]`, the existing CQT chord features;
- `short_features`: `[batch, 770, time]`, the existing causal Rise spectra.

Outputs are `root_logits`, `quality_logits`, `pitch_logits`, and `onset_logits`.
The onset output keeps its `[batch, 12, time]` shape. Both thresholds and the
onset feature contract are saved in model metadata. The exporter checks three
batch/context sizes against the original branch models in ONNX Runtime, and
writes a single self-contained ONNX only after parity passes.

Solitito 0.5.7 supports this contract. It loads each independent
branch into memory for its own worker: chords every 40 ms, Rise every 16 ms.
Copy `best_model_v2_take7.onnx` next to `dsp_weights.json`, then run
`./target/release/solitito --check`. Normal launch selects take7 automatically;
recording and weak-onset rescue are optional. See [running](running.md).
Binaries from 0.5.6 and earlier need updating before using this model. The trainer's
`app_ready=false` means training/export alone does not validate live crediting.

## Convert a completed take7 run without training again

Paste the updated **whole** `model_trainer.py`, keeping the original `RUN_TAG`
and HF repository, and set at the top:

```python
MODE = "export_only"
RUN_TAG = "v2_take7"
```

It retrieves `best_model_v2_take7_chords.onnx`, `best_model_v2_take7_onset.onnx`
and `training_summary_v2_take7.json` from the local run directory or HF, then
publishes **`best_model_v2_take7.onnx`**. It preserves the existing training
metrics and reads the selected onset threshold from that summary. No optimizer,
feature extraction, dataset or PyTorch is needed. The original two ONNX files
are not modified. Missing sources or threshold cause an explicit error, never
an automatic training restart.

For an offline copy, place the two ONNX files and summary under
`OUTPUT_ROOT/RUN_TAG/` and set `USE_HF=False`. The local intermediate
`rise/short_onset_rise.onnx` is also accepted. If the summary is missing, set
`EXPORT_ONSET_THRESHOLD` to the threshold selected by that run; do not guess it
from the epoch log. The report's `model.path` is the final model; `chords` and
`onset` retain provenance/metrics of the original branches.

Source files are `chord_training.py`, `take7_training.py` and the existing onset
modules. Regenerate the copyable script after source changes:

```bash
python3 dist/build_train_onset_kaggle.py
python3 dist/build_model_trainer.py
```

No pitch-shift augmentation, new loss, live rescue rule or application threshold
was added in this integration. It consolidates the existing Rise trainer with the
chord trainer. GuitarSet note starts remain note annotations, not verified pick
technique. Previously inspected test recordings are regression data.

## Resume errors

`Cannot resume onset training with changed data` means the ordered source IDs,
splits, audio hashes, feature hashes or event annotations differ from the saved
snapshot. Paths alone are not part of this comparison. Preserve the checkpoint;
do not remove this check or change the run name to force a restart. For an
already completed two-model run, use `export_only` above. For an interrupted
training run, restore the matching data/cache before resuming.

## Maintaining the script

Edit `dist/chord_training.py`, `dist/take7_training.py` and the onset source
modules, then run `python dist/build_model_trainer.py`. Commit those sources,
the relevant tests and the generated `dist/model_trainer.py` together.
The single generated script is the supported copy-and-run entry for Kaggle.
See [the tool index](../dist/README.md) for optional research scripts.
