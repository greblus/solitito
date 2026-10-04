# Training and onset tools

The supported Kaggle entry is **`model_trainer.py`**, a generated standalone
script. Copy the whole file. See [take7 training/export](../docs/training-take7.md)
([Polski](../docs/training-take7_pl.md)). Generated copies are not editable sources.

## Current training and runtime verification

| Files | Purpose |
|---|---|
| `chord_training.py`, `take7_training.py` | Chord base, resume routing and single-file take7 export |
| `train_short_onset.py`, `onset_rise.py`, `onset_events.py` | Causal features, Rise training and event evaluation |
| `audit_onset_data.py`, `prepare_onset_data.py` | GuitarSet inventory/annotations and tuned synthetic plucks |
| `prepare_onset_kaggle.py` | Standalone preparation script, also used by the trainer source |
| `build_model_trainer.py` | Regenerates the supported Kaggle trainer |
| `build_onset_kaggle.py` | Regenerates standalone preparation |
| `trace_onsets.sh` | Optional app trace; `--record` additionally captures audio |
| `short_onset_features.py`, `onset_rescue.py` | Python reference for live DSP and optional weak-attack confirmation |
| `test_take7_*.py`, `test_onset_*.py`, `test_short_onset_features.py` | Contract, data, export and regression checks |

## Research and diagnosis

Keep these sources to reproduce comparisons and investigate false or missed
attacks. They are not required to run the application.

| Files | Purpose |
|---|---|
| `compare_short_onset.py`, `summarize_onset_comparison.py` | Compare detectors on the same audio and annotations |
| `diagnose_short_onset.py` | Inspect onset errors and timing |
| `audit_onset_features.py` | Compare chord DSP features against the trainer |
| `audit_onset_real.py`, `build_audit_onset_real_kaggle.py` | Audit saved predictions and create review material |
| `onset_ringing.py`, `onset_pairs.py` | Earlier loss experiments and associated measurements; still imported by the training/evaluation code |
| `resume_onset_training.py` | Recover earlier paired CONTROL/RISE runs |
| `build_train_onset_kaggle.py` | Bundle paired experiments/recovery; also used internally by the take7 builder |

The ordinary take7 recipe uses spectral Rise and ordinary BCE; preserving earlier
experiment modules does not enable their alternative losses. Removing them needs
a separate dependency refactor, not deletion during file cleanup.

The large historical standalone copies are generated on demand and ignored:

```bash
python dist/build_train_onset_kaggle.py
# creates train_onset_kaggle.py and resume_onset_kaggle.py
python dist/build_audit_onset_real_kaggle.py
# creates audit_onset_real_kaggle.py
```

Tests compile/run freshly generated scripts instead of requiring saved duplicates.
Keep `model_trainer.py` and `prepare_onset_kaggle.py` versioned for the supported
copy-and-run workflows. Rebuild preparation before rebuilding the trainer when
changing data generation.

## Checks and artifacts

Run Python tests with `PYTHONPATH=dist python -m unittest <test_module> ...` in
an environment with their dependencies. Torch/ONNX integration tests need those
packages; a skipped test is not an export/training validation. `test_take7_export`
requires ONNX and ONNX Runtime even for collection.

Models, checkpoints, feature caches, WAVs and run reports are artifacts, not code.
Keep measurements under ignored `dist/crediting_measurements/` or outside the
repository. Keep working plans/session reports in the project memory directory
specified by `AGENTS.md`. Do not publish local recordings or checkpoint credentials.
