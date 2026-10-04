#!/usr/bin/env bash
# Run the experimental onset path and keep the trace out of the terminal.
set -euo pipefail
solitito_record=false
if [[ "${1:-}" == "--record" ]]; then
    solitito_record=true
    shift
fi
solitito_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$solitito_root"
solitito_binary="$solitito_root/target/release/solitito"
if [[ ! -x "$solitito_binary" ]]; then
    printf 'Build the application first: cargo build --release\n' >&2
    exit 1
fi
solitito_logs="$solitito_root/dist/crediting_measurements/rise-live"
mkdir -p -- "$solitito_logs"
solitito_trace=$(mktemp "$solitito_logs/trace-$(date +%Y%m%d-%H%M%S)-XXXXXX")
{
    printf 'Started: %s\n' "$(date --iso-8601=seconds)"
    printf 'Onset trace: pitch classes C=0, C#=1, ..., B=11; probabilities follow this order.\n'
    sha256sum -- "$solitito_binary"
    # Record available candidates; startup below reports the selected paths.
    for solitito_model in "${SOLITITO_MODEL:-}" "${SOLITITO_ONSET_MODEL:-}" \
        best_model_v2_take7.onnx best_model_v2_take6_onset.onnx \
        best_model_v2_take6.onnx short_onset_rise.onnx; do
        if [[ -f "$solitito_model" ]]; then sha256sum -- "$solitito_model"; fi
    done
} > "$solitito_trace"
if [[ "$solitito_record" == true ]]; then
    export SOLITITO_ONSET_RECORD="$solitito_trace"
    printf 'Audio: %s-g*.wav (16 kHz mono, about 4 MB/minute)\n' "$solitito_trace"
fi
printf 'Log: %s\nPlay a few missed notes, then close the application normally.\n' "$solitito_trace"
if SOLITITO_ONSET=rise SOLITITO_ONSET_TRACE=1 "$solitito_binary" "$@" >> "$solitito_trace" 2>&1; then
    printf 'Saved: %s\n' "$solitito_trace"
else
    solitito_status=$?
    tail -n 30 -- "$solitito_trace" >&2
    exit "$solitito_status"
fi
