# Running it

[← back to the README](../README.md)

## Running it

Ready packages are attached to each [release](../../releases) — binary, ONNX
Runtime, the model and the DSP weights, nothing else needed:

```bash
tar xzf solitito_linux-*.tar.gz && cd solitito_linux-* && ./solitito.sh
```

On Windows, unpack the zip and run `solitito.exe`.

### From source

```bash
cargo build --release
```

The binary needs two files in its working directory. `dsp_weights.json` is in
this repository; the model is not, because it is 29 MB:

```bash
curl -LO https://huggingface.co/greblus/solitito-ai/resolve/main/best_model_v2_take6_onset.onnx
curl -LO https://huggingface.co/greblus/solitito-ai/resolve/main/short_onset_masking_v2.onnx
```

The second file (1 MB) says which string was struck, and the repeat rules lean on it: a note
asked for again counts only when that class has been struck again. It is optional - without it
the app starts, says so, and judges repeats on older evidence that lets more ringing notes
through. `short_onset_rise.onnx` is used in its place if it is the one present.

`./solitito --check` loads both and reports whether they are usable, which is
also what the release workflow runs against every package it builds.

The app refuses to start on an old dense `dsp_weights.json` rather than accepting
it silently: the previous format also carried a different chroma mapping, which
would feed the model features it was not trained on. Regenerate with
`python dist/gen_weights.py` (needs librosa).

```bash
cargo build --release
./target/release/solitito
```
