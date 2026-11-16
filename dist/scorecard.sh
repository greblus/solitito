#!/usr/bin/env bash
# The scorecard: every measurement of the judge, from one frozen build.
#
#   dist/scorecard.sh OUT_DIR
#
# Builds the release test binary once and copies it into OUT_DIR, so the
# working tree can change while the measurements run. Each measurement goes
# to its own file; `diff -r BEFORE AFTER` shows what a change did.
#
# Material: AtoA from SOLITITO_MATERIAL (default ~/Documents), the rest from
# dist/crediting_measurements. A full run takes 30 to 40 minutes on a laptop;
# run it under `nice -n 19`.
set -uo pipefail

out="${1:?usage: dist/scorecard.sh OUT_DIR}"
mkdir -p "$out"
m=dist/crediting_measurements
material="${SOLITITO_MATERIAL:-$HOME/Documents}"

# Times change from run to run, the rest must not.
strip_times() { sed -E 's/finished in [0-9.]+s/finished/; s/[0-9]+ filtered out/N filtered out/'; }
strip_warnings() { awk '/^warning: /{skip=1} skip&&/^$/{skip=0; next} !skip'; }

bin=$(cargo test --release --no-run --message-format=json 2>/dev/null \
    | python3 -c 'import json,sys
for line in sys.stdin:
    m = json.loads(line)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["kind"] == ["bin"]:
        print(m["executable"])' | tail -1)
[ -x "$bin" ] || { echo "no test binary"; exit 1; }
cp "$bin" "$out/tests.bin"
git rev-parse --short HEAD > "$out/commit.txt"
git status --short >> "$out/commit.txt"

run() {
    local file="$1"; shift
    local name="$1"; shift
    env "$@" "$out/tests.bin" --ignored --exact --nocapture --test-threads=1 "$name" 2>&1 \
        | strip_warnings | strip_times > "$out/$file.txt"
    echo "$file: $(grep -E '^test result' "$out/$file.txt" | head -1)"
}

"$out/tests.bin" --test-threads=4 2>&1 | grep -E '^test result|FAILED|panicked' | strip_times > "$out/unit.txt"
echo "unit: $(tr '\n' ' ' < "$out/unit.txt")"

run repeats        state::repeat_harness::repeats_on_spliced_notes
run ringing        state::repeat_harness::ringing_into_the_next_chord
run long_rings     state::repeat_harness::long_rings_into_the_next_chord
run false_by_iv    state::onset_sources::false_credits_by_interval
run credit_gate    state::onset_sources::credit_gate_by_source
run chords_akordy  state::onset_sources::chord_gate_by_source \
    SOLITITO_WAV=$m/akordy-g1.wav SOLITITO_LABELS=$m/akordy-g1-labels.csv
run chords_arp     state::onset_sources::chord_gate_by_source \
    SOLITITO_WAV=$m/arpeggia-g1.wav SOLITITO_LABELS=$m/arpeggia-g1-labels.csv
run session        state::onset_sources::session_credits_by_source \
    SOLITITO_WAV=$m/sesja-g1.wav SOLITITO_NOTES=$m/sesja-g1-notes.csv
run grip_akordy    state::grip_harness::free_order_on_chords \
    SOLITITO_WAV=$m/akordy-g1.wav SOLITITO_LABELS=$m/akordy-g1-labels.csv
run quiet_fifth    state::grip_harness::quiet_fifth_on_spliced_triads
run formulas       state::formula_harness::strummed_chords_in_formulas
run lines          state::ear_speed_harness::lines_through_the_judge
run lines_quick    state::ear_speed_harness::lines_through_the_judge SOLITITO_QUICK=1
run sessions_speed state::ear_speed_harness::how_much_sooner_on_real_sessions
run semitones      state::line_harness::semitone_steps_in_rising_lines
run harris_down    state::line_harness::harris_scales_falling
run enclosures     state::line_harness::harris_enclosures
run greene         state::line_harness::greene_in_position

./target/release/solitito --probe "$material/AtoA.wav" > "$out/probe-atoa.txt" 2>&1 || true
./target/release/solitito --bench > "$out/bench.txt" 2>&1 || true
echo "written to $out"
