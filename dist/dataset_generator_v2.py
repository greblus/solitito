# ==========================================
# DATASET GENERATOR V2 - one file, no barcodes
# ==========================================
# What changed against v1 and why:
#
#  1. BARCODE REMOVED. The beeps encoded sample IDs and `decode_annotations.py`
#     had BPM=120 hard-coded while the generator ran at BPM=60 -> bits sampled
#     twice too densely -> all zeros -> ID 0 -> the label "C" for EVERYTHING.
#     85% of the dataset went into training with unrelated labels. Here the
#     generator KNOWS the measure of each block and writes `start,end,label`
#     directly. There is nothing to decode and no way to lie.
#
#  2. COLLISION-PROOF FILENAMES. `01_triads_clean.wav` used to collide with
#     GuitarSet's `01_BN1-129-Eb_comp_mix.wav` (01 = player number), so the
#     trainer spread synthetic labels over 60 unrelated recordings. The `synth_`
#     prefix ends that.
#
#  3. SHORTER BLOCKS. 16 s per sample (4 measures @ BPM 60) was mostly decay tail
#     that the energy gate discarded anyway. Now 6 s (3 measures @ BPM 120):
#     ~2.7x more samples from the same render length.
#
#  4. DECLARED CHORD SHAPES - movable shapes with DECLARED pitch classes, checked
#     by a self-test before generation (see `self_test()`). A typo in the shape
#     table stops the script rather than the training run.
#
# OUTPUT:
#   synth_dataset.gp5            -> export DI, run through NAM, save as:
#                                     synth_dataset_clean.wav
#                                     synth_dataset_eob.wav
#   synth_annotations.csv        -> annotations for BOTH wavs (file,start,end,label)
#   synth_reference.csv          -> id,label,strings,frets (for inspection)
#
# CALIBRATION: if the render starts with silence, set RENDER_OFFSET_SEC (one
# number for the whole file - see the instructions printed at the end).
#
# COMMIT THIS FILE TOGETHER WITH THE GENERATED CSV. Not having the reference in
# the repository once cost us a whole training cycle.
# ==========================================

import os
import subprocess
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
OUT_GP        = "synth_dataset.gp5"
OUT_ANN       = "synth_annotations.csv"
OUT_REF       = "synth_reference.csv"
WAV_NAMES     = ["synth_dataset_clean.wav", "synth_dataset_eob.wav"]

BPM           = 120          # a 4/4 measure = 2.0 s
MEASURES_PER_BLOCK = 3       # attack + sustain + silence  => 6.0 s block
ANN_START_OFF = 0.05         # annotation starts just after the attack [s]
ANN_LEN       = 3.20         # annotated window length [s] (attack + sustain)
RENDER_OFFSET_SEC = 0.0      # silence at the start of the render (see calibration)

MAX_FRET      = 15
VELOCITY      = 105
TICKS_PER_MEASURE = 3840     # 960 per quarter note * 4

NOTE_ORDER    = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# pitch class of each open string (1 = high e ... 6 = low E)
OPEN_PC       = {1: 4, 2: 11, 3: 7, 4: 2, 5: 9, 6: 4}
OPEN_MIDI     = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

SEC_PER_MEASURE = 4 * 60.0 / BPM
SEC_PER_BLOCK   = MEASURES_PER_BLOCK * SEC_PER_MEASURE


# ==========================================
# 2. SHAPE LIBRARY (movable shapes)
# ==========================================
# offsets: {string: fret relative to the root fret}.  root_str = string with the root.
# 'pcs' = pitch classes relative to the root that the shape MUST produce; self_test checks it.
SHAPES = [
    # --- major triad ---
    dict(q="",      lbl="maj",   root_str=6, offsets={6:0, 5:2, 4:2, 3:1, 2:0, 1:0}, pcs={0,4,7}),
    dict(q="",      lbl="maj",   root_str=5, offsets={5:0, 4:2, 3:2, 2:2},           pcs={0,4,7}),
    # --- minor triad ---
    dict(q="m",     lbl="min",   root_str=6, offsets={6:0, 5:2, 4:2, 3:0, 2:0, 1:0}, pcs={0,3,7}),
    dict(q="m",     lbl="min",   root_str=5, offsets={5:0, 4:2, 3:2, 2:1},           pcs={0,3,7}),
    # --- maj7 ---
    dict(q="Maj7",  lbl="maj7",  root_str=6, offsets={6:0, 4:1, 3:1, 2:0},           pcs={0,4,7,11}),
    dict(q="Maj7",  lbl="maj7",  root_str=5, offsets={5:0, 4:2, 3:1, 2:2},           pcs={0,4,7,11}),
    # --- dominant 7 ---
    dict(q="7",     lbl="dom7",  root_str=6, offsets={6:0, 4:0, 3:1, 2:0},           pcs={0,4,7,10}),
    dict(q="7",     lbl="dom7",  root_str=5, offsets={5:0, 4:2, 3:0, 2:2},           pcs={0,4,7,10}),
    # --- minor 7 ---
    dict(q="m7",    lbl="min7",  root_str=6, offsets={6:0, 4:0, 3:0, 2:0},           pcs={0,3,7,10}),
    dict(q="m7",    lbl="min7",  root_str=5, offsets={5:0, 4:2, 3:0, 2:1},           pcs={0,3,7,10}),
    # --- half-diminished ---
    dict(q="m7b5",  lbl="m7b5",  root_str=6, offsets={6:0, 4:0, 3:0, 2:-1},          pcs={0,3,6,10}),
    dict(q="m7b5",  lbl="m7b5",  root_str=5, offsets={5:0, 4:1, 3:0, 2:1},           pcs={0,3,6,10}),
    # --- diminished 7 ---
    dict(q="dim7",  lbl="dim7",  root_str=6, offsets={6:0, 4:-1, 3:0, 2:-1},         pcs={0,3,6,9}),
    dict(q="dim7",  lbl="dim7",  root_str=5, offsets={5:0, 4:1, 3:-1, 2:1},          pcs={0,3,6,9}),
    # --- family with the root on the D string (higher register, different tone) ---
    dict(q="",      lbl="maj",   root_str=4, offsets={4:0, 3:2, 2:3, 1:2},           pcs={0,4,7}),
    dict(q="m",     lbl="min",   root_str=4, offsets={4:0, 3:2, 2:3, 1:1},           pcs={0,3,7}),
    dict(q="Maj7",  lbl="maj7",  root_str=4, offsets={4:0, 3:2, 2:2, 1:2},           pcs={0,4,7,11}),
    dict(q="7",     lbl="dom7",  root_str=4, offsets={4:0, 3:2, 2:1, 1:2},           pcs={0,4,7,10}),
    dict(q="m7",    lbl="min7",  root_str=4, offsets={4:0, 3:2, 2:1, 1:1},           pcs={0,3,7,10}),
    # --- sus4 ---
    # lbl = the trainer quality class (QUALITIES); the audio label is "sus4"
    dict(q="sus4",  lbl="sus",   root_str=6, offsets={6:0, 5:2, 4:2, 3:2, 2:0, 1:0}, pcs={0,5,7}),
    # --- augmented ---
    dict(q="aug",   lbl="aug",   root_str=6, offsets={6:0, 5:3, 4:2, 3:1},           pcs={0,4,8}),
]


def shape_pcs(shape, base_fret):
    """Pitch classes relative to the root, computed from real frets and tuning."""
    root_pc = (OPEN_PC[shape["root_str"]] + base_fret) % 12
    out = set()
    for s, off in shape["offsets"].items():
        out.add(((OPEN_PC[s] + base_fret + off) - root_pc) % 12)
    return out


def self_test():
    """Checks that every shape REALLY produces the declared chord, at every fret.
    A typo in the shape table stops generation instead of poisoning the dataset."""
    print("🔍 Self-testing the shape library...")
    bad = 0
    for sh in SHAPES:
        for f in range(0, MAX_FRET + 1):
            got = shape_pcs(sh, f)
            if got != sh["pcs"]:
                print(f"   ❌ {sh['lbl']} (string {sh['root_str']}, fret {f}): "
                      f"expected {sorted(sh['pcs'])}, got {sorted(got)}")
                bad += 1
    if bad:
        sys.exit(f"❌ {bad} bad shapes - fix SHAPES; not generating.")
    print(f"   ✅ {len(SHAPES)} shapes give correct intervals at every fret\n")


# ==========================================
# 3. BUILDING THE BLOCK LIST
# ==========================================
def build_blocks():
    """Returns a list of dict(label, strings, frets). Order = order in the audio."""
    blocks = []

    # --- CHORDS: for every quality, every root, all matching positions ---
    for sh in SHAPES:
        for root_idx, root in enumerate(NOTE_ORDER):
            base = (root_idx - OPEN_PC[sh["root_str"]]) % 12
            for f_pos in (base, base + 12):
                frets = {s: f_pos + off for s, off in sh["offsets"].items()}
                if any(v < 0 or v > MAX_FRET for v in frets.values()):
                    continue
                label = f"{root}{sh['q']}" if sh["q"] in ("", "m") else f"{root} {sh['q']}"
                strings = sorted(frets.keys(), reverse=True)     # from the lowest string
                blocks.append(dict(label=label, strings=strings,
                                   frets=[frets[s] for s in strings], kind=sh["lbl"]))

    # --- SINGLE NOTES: every fret of every string ---
    for s in range(1, 7):
        for f in range(0, MAX_FRET + 1):
            name = NOTE_ORDER[(OPEN_PC[s] + f) % 12]
            blocks.append(dict(label=f"Note {name}", strings=[s], frets=[f], kind="note"))

    return blocks


# ==========================================
# 4. ZAPIS GP5
# ==========================================
def ensure_guitarpro():
    """Imports pyguitarpro, installing it if needed. On PEP 668 systems (Arch,
    Debian, Fedora) installing into the system Python is blocked, so we suggest a
    venv instead of dying with a stack trace."""
    try:
        import guitarpro
        return guitarpro
    except ImportError:
        pass
    print("📦 pyguitarpro missing - trying to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyguitarpro", "--quiet"])
        import guitarpro
        return guitarpro
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__)) or "."
        me   = os.path.basename(__file__)
        sys.exit(
            "\n❌ Cannot install pyguitarpro into this Python.\n"
            "   Most likely PEP 668 (Arch/Debian/Fedora block installs into the system\n"
            "   Python). Do NOT use --break-system-packages. Make a venv:\n\n"
            f"     cd {here}\n"
            f"     python -m venv .venv\n"
            f"     .venv/bin/pip install -q pyguitarpro\n"
            f"     .venv/bin/python {me}\n\n"
            "   (create the venv once; later runs need only the last line)\n"
        )


def write_gp(blocks):
    guitarpro = ensure_guitarpro()
    from guitarpro import models, write

    song = models.Song(title="Solitito Synth Dataset v2", tempo=BPM)
    tr = song.tracks[0]
    tr.name = "Guitar"
    tr.channel.instrument = 27                     # Electric Guitar (clean)
    tr.strings = [models.GuitarString(i, OPEN_MIDI[i]) for i in range(1, 7)]

    # CRITICAL: models.Song() creates one DEFAULT empty measure (start=960).
    # Without clearing it the blocks would be one measure (2 s) off from the
    # annotations - exactly the silent mismatch that destroyed the last dataset.
    song.measureHeaders.clear()
    tr.measures.clear()

    def add_measure(tick):
        h = models.MeasureHeader()
        h.number = len(song.measureHeaders) + 1
        h.start = tick
        h.timeSignature = models.TimeSignature(numerator=4,
                                               denominator=models.Duration(value=4))
        song.measureHeaders.append(h)
        m = models.Measure(tr, h)
        tr.measures.append(m)
        if len(m.voices) > 0:
            return m.voices[0]
        v = models.Voice(m); m.voices.append(v); return v

    def whole(voice, strings, frets, tie=False, rest=False):
        b = models.Beat(voice)
        b.duration = models.Duration(value=1)      # whole note
        if rest:
            b.status = models.BeatStatus.rest
        else:
            b.status = models.BeatStatus.normal
            for s, f in zip(strings, frets):
                n = models.Note(b)
                n.string = s; n.value = f; n.velocity = VELOCITY
                n.effect.letRing = True
                if tie: n.type = models.NoteType.tie
                b.notes.append(n)
        voice.beats.append(b)

    tick = 0
    for blk in blocks:
        # measure 1: attack | 2: sustain (tie) | 3: silence (block separator)
        whole(add_measure(tick), blk["strings"], blk["frets"]);            tick += TICKS_PER_MEASURE
        whole(add_measure(tick), blk["strings"], blk["frets"], tie=True);  tick += TICKS_PER_MEASURE
        whole(add_measure(tick), [], [], rest=True);                       tick += TICKS_PER_MEASURE

    if os.path.exists(OUT_GP):
        os.remove(OUT_GP)
    write(song, OUT_GP)


# ==========================================
# 5. ANNOTATIONS + REFERENCE (written directly, nothing to decode)
# ==========================================
def write_csvs(blocks):
    with open(OUT_REF, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "label", "kind", "strings", "frets"])
        for i, b in enumerate(blocks):
            w.writerow([i, b["label"], b["kind"],
                        " ".join(map(str, b["strings"])), " ".join(map(str, b["frets"]))])

    rows = []
    for i, b in enumerate(blocks):
        start = RENDER_OFFSET_SEC + i * SEC_PER_BLOCK + ANN_START_OFF
        end   = start + ANN_LEN
        for wav in WAV_NAMES:                       # same timings for both tones
            rows.append([wav, f"{start:.3f}", f"{end:.3f}", b["label"]])
    with open(OUT_ANN, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "start", "end", "label"]); w.writerows(rows)
    return len(rows)


# ==========================================
# 6. CALIBRATION FROM THE RENDER (stdlib only - works anywhere)
# ==========================================
def read_wav_head(path, seconds):
    """Minimal WAV reader (stdlib): PCM 16/24/32 bit and IEEE float 32/64.
    DAWs export in all sorts of formats - 24-bit and float are as likely as 16-bit.
    Returns (mono float samples, sample_rate, channels, format description)."""
    import array
    import struct
    with open(path, "rb") as f:
        hdr = f.read(12)
        if hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            sys.exit(f"❌ {os.path.basename(path)}: not a WAV file.")
        fmt = None
        while True:
            ch_hdr = f.read(8)
            if len(ch_hdr) < 8:
                sys.exit("❌ Nie znaleziono chunku 'data'.")
            cid, csz = struct.unpack("<4sI", ch_hdr)
            if cid == b"fmt ":
                d = f.read(csz)
                afmt, ch, sr, _, _, bits = struct.unpack("<HHIIHH", d[:16])
                if afmt == 0xFFFE and len(d) >= 26:          # WAVE_FORMAT_EXTENSIBLE
                    afmt = struct.unpack("<H", d[24:26])[0]
                fmt = (afmt, ch, sr, bits)
            elif cid == b"data":
                if fmt is None: sys.exit("❌ Chunk 'data' przed 'fmt '.")
                afmt, ch, sr, bits = fmt
                nbytes = min(csz, int(seconds * sr) * ch * (bits // 8))
                raw = f.read(nbytes)
                break
            else:
                f.seek(csz + (csz & 1), 1)

    if afmt == 3 and bits == 32:                              # IEEE float32
        a = array.array("f"); a.frombytes(raw[:len(raw) - len(raw) % 4]); scale = 1.0
        desc = "32-bit float"
    elif afmt == 3 and bits == 64:                            # IEEE float64
        a = array.array("d"); a.frombytes(raw[:len(raw) - len(raw) % 8]); scale = 1.0
        desc = "64-bit float"
    elif afmt == 1 and bits == 16:
        a = array.array("h"); a.frombytes(raw[:len(raw) - len(raw) % 2]); scale = 1 / 32768
        desc = "16-bit PCM"
    elif afmt == 1 and bits == 32:
        a = array.array("i"); a.frombytes(raw[:len(raw) - len(raw) % 4]); scale = 1 / 2147483648
        desc = "32-bit PCM"
    elif afmt == 1 and bits == 24:                            # 3 bajty, little-endian ze znakiem
        n = len(raw) // 3
        a = [int.from_bytes(raw[3*i:3*i+3], "little", signed=True) for i in range(n)]
        scale = 1 / 8388608
        desc = "24-bit PCM"
    else:
        sys.exit(f"❌ Unsupported WAV format (format={afmt}, {bits}-bit). "
                 f"Export as PCM 16/24/32-bit or 32-bit float.")

    mono = [a[i] * scale for i in range(0, len(a) - ch + 1, ch)] if ch > 1 \
           else [v * scale for v in a]
    return mono, sr, ch, desc


def calibrate(wav_path, probe_sec=40.0):
    """Finds the first attack in the render and reports what to put in RENDER_OFFSET_SEC.
    Note: pyguitarpro normalises ticks to the format convention (start=960).
    A DAW may add silence at the start, so the offset is MEASURED, not assumed."""
    mono, sr, ch, desc = read_wav_head(wav_path, probe_sec)

    win = max(1, sr // 100)                       # 10 ms windows
    energies = []
    for i in range(0, len(mono) - win, win):
        s = 0.0
        for j in range(i, i + win, 4):            # every 4th sample is enough
            v = mono[j]; s += v * v
        energies.append(s)
    if not energies: sys.exit("❌ File too short.")
    peak = max(energies)
    thr = 0.10 * peak
    first = next((i for i, e in enumerate(energies) if e > thr), None)
    if first is None: sys.exit("❌ Nie znaleziono ataku — czy to na pewno render gitary?")
    t = first * win / sr

    print(f"🎧 {os.path.basename(wav_path)}  ({sr} Hz, {ch} kan., {desc})")
    if sr != 48000:
        print(f"   ℹ️  render at {sr} Hz - 48000 Hz would be better (NAM runs natively at 48k,")
        print("      and 48000/16000=3 gives exact decimation to the trainer SR)")
    print(f"   Pierwszy atak: {t:.3f} s")
    print(f"   Annotations assume: {RENDER_OFFSET_SEC:.3f} s")
    d = t - RENDER_OFFSET_SEC
    if abs(d) < 0.03:
        print("   ✅ MATCH - the annotations are correct, change nothing.")
    else:
        print(f"   ⚠️  DRIFT {d:+.3f} s.  Set in this file:")
        print(f"        RENDER_OFFSET_SEC = {t:.3f}")
        print("      and run the script again (the GP5 will not change, only timings).")
    return t


# ==========================================
# 7. MAIN
# ==========================================
def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--calibrate":
        calibrate(sys.argv[2]); return
    self_test()
    blocks = build_blocks()

    from collections import Counter
    kinds = Counter(b["kind"] for b in blocks)
    total_sec = len(blocks) * SEC_PER_BLOCK
    print(f"📊 Blocks: {len(blocks)}   (block {SEC_PER_BLOCK:.1f}s @ BPM {BPM})")
    print("   " + "  ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print(f"   Render length: {total_sec/60:.1f} min\n")

    print(f"🎸 Zapis {OUT_GP}...")
    write_gp(blocks)
    n = write_csvs(blocks)
    print(f"💾 {OUT_GP}\n💾 {OUT_REF} ({len(blocks)} wierszy)\n💾 {OUT_ANN} ({n} wierszy)\n")

    print("=" * 74)
    print("DALEJ:")
    print(f"  1. Open {OUT_GP}, export the guitar track as DI (wav).")
    print(f"  2. Run through NAM -> save as {WAV_NAMES[0]} and {WAV_NAMES[1]}.")
    print("  3. CALIBRATION (mandatory, 10 seconds):")
    print(f"       python {os.path.basename(__file__)} --calibrate {WAV_NAMES[0]}")
    print("     Measures the first attack and reports whether annotations match.")
    print(f"  4. Verify: verify_annotations.py  (label<->audio agreement must be >60%)")
    print("     not ~8%). Only then train.")
    print(f"  5. ZACOMMITUJ {OUT_REF} i {OUT_ANN} razem z tym skryptem.")
    print("=" * 74)


if __name__ == "__main__":
    main()
