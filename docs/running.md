# Running it

[← back to the README](../README.md)

## Running it

Ready packages are attached to each [release](../../releases) — binary, ONNX
Runtime, the model and the DSP weights, nothing else needed:

Version **0.5.7** includes the combined take7 model with Rise onset detection.

```bash
tar xzf solitito_linux-*.tar.gz && cd solitito_linux-* && ./solitito.sh
```

On Windows, unpack the zip and run `solitito.exe`.

### From source

```bash
cargo build --release
```

The current source build needs `dsp_weights.json` (in this repository) and
`best_model_v2_take7.onnx` from [Hugging Face](https://huggingface.co/greblus/solitito-ai)
or [the take7 trainer/export](training-take7.md)
in its working directory. The ONNX weights are not committed to Git.
`./target/release/solitito --check` runs both branches and checks compatibility.

The app refuses to start on an old dense `dsp_weights.json` rather than accepting
it silently: the previous format also carried a different chroma mapping, which
would feed the model features it was not trained on. Regenerate with
`python dist/gen_weights.py` (needs librosa).

```bash
cargo build --release
./target/release/solitito
```

### Take7 onset detection (0.5.7)

Put the trainer's **`best_model_v2_take7.onnx`** in the working directory.
The app prefers it automatically. This single file contains the chord, pitch
and Rise outputs; `short_onset_rise.onnx` is no longer required with take7.
The two independent branches are loaded into memory for their existing workers,
so onset inference keeps its 16 ms cadence without running the chord encoder.
No extra model files are written. `dsp_weights.json` is still required.

```bash
./target/release/solitito --check
./target/release/solitito
```

`--check` runs both branches and reports the model paths and onset threshold.
Use `SOLITITO_MODEL=/path/model.onnx` for a different filename. Without take7,
the app still supports take6 plus the separate `short_onset_rise.onnx`.
An explicit `SOLITITO_ONSET_MODEL` overrides the onset source, including when
using take7; leave it unset to use take7's own onset branch.

Rise is enabled by default. With **Credit only what was struck** enabled, note
practice uses its per-pitch events, including simultaneous notes. Noise gate
still applies, using the short audio window. Take7 reads the validation-selected onset threshold from model metadata
(the older standalone Rise defaults to 0.8); the sounding-note threshold does not change it.
A new round needs new events. Rise can still misclassify attacks: this build is
for testing, not a claim that repeated credits are solved.

For a comparison with the original onset path, launch:

```bash
SOLITITO_MODEL=best_model_v2_take6_onset.onnx SOLITITO_ONSET=legacy ./target/release/solitito
```

`SOLITITO_ONSET_MODEL` overrides the Rise file path; `SOLITITO_ONSET_TRACE=1`
logs every frame, including subthreshold probabilities, the noise gate, and
event admission/credits. Run `./dist/trace_onsets.sh` to save this trace to a new
file under `dist/crediting_measurements/rise-live/` without flooding the terminal. `--probe` reports chord/pitch predictions and the legacy onset head, if present;
use the trace for take7 onset events. `--file` exercises the selected live
onset path using a WAV recording (first channel).

To record audio along with the diagnostic trace, run:

```bash
./dist/trace_onsets.sh --record
```

Close the application normally after playing. The trace and `-g*.wav` files
share a filename prefix. WAV files contain the exact mono float32 samples fed
to Rise after resampling to 16 kHz, without gain changes (about 4 MB/minute).
Each input restart gets a separate file; files are never overwritten. No audio
is recorded by the normal launcher. `SOLITITO_ONSET_RECORD` can also specify a
recording filename prefix directly. `RISE_CAPTURE end` reports completion and
whether the recording is continuous; captures with discontinuities must not be
used as uninterrupted replays. Recordings are for diagnosis, not training.

To try experimental confirmation of weak Rise responses:

```bash
SOLITITO_ONSET_RESCUE=1 ./dist/trace_onsets.sh --record
```

The ordinary detection threshold stays at the selected model value. A weaker
response (at least 0.6 and below that threshold) needs
pitch-specific spectral growth and pitch confirmation 16 ms later. The new note
need not be louder than an already ringing note: confirmation also checks the
positive residual against the background saved with the onset candidate. Both paths
share the same repetition latch. Only weak responses need this confirmation;
strong detections, including polyphony, do not wait. This variant is disabled
by default; launch normally to return to the original path.
