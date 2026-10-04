"""Solitito take7: one ONNX with chord, quality, pitch and Rise onset outputs.

Copy/run this WHOLE file in the existing Kaggle notebook. Configuration is at
the top: RUN_TAG, MODE, BASE_RUN, HF_REPO_ID, USE_HF, INITIAL_ONSET, ONSET_EPOCHS.
AUTO reuses take6 and trains only Rise; FULL ignores take6 and trains both models.
Both modes resume their own run. For a completely new run choose a new RUN_TAG.
USE_HF=False works locally without a Hugging Face account. Errors accessing HF
are errors, never evidence of an empty repository. The final graph has two inputs
(CQT features and short spectra) and four outputs. MODE="export_only" combines
an already completed two-file run without training or feature preparation.

Generated from chord_training.py, take7_training.py and the tested onset modules
by build_model_trainer.py. Edit sources and regenerate, not this copy.
"""

RUN_TAG = "v2_take7"
MODE = "auto"  # auto, onset_only, full, export_only; full resumes its own run
BASE_RUN = "v2_take6"
HF_REPO_ID = "greblus/chord-model-snapshots"  # set to your own repo for a new model
USE_HF = True  # False: entirely local training, no token or HF account needed
INPUT_DIR = "/kaggle/input"
OUTPUT_ROOT = "/kaggle/working"
INITIAL_ONSET = ""  # optional Rise .pt/.pth path, never an ONNX or take6 fc_onset
ONSET_EPOCHS = 12
EXPORT_ONSET_THRESHOLD = None  # export_only: normally read from the saved summary

if __name__ == "__main__":
    import importlib.util
    import subprocess
    import sys
    export_only = (MODE == "export_only" or "--mode=export_only" in sys.argv or
                   any(a == "--mode" and b == "export_only" for a, b in zip(sys.argv, sys.argv[1:])))
    packages = {"onnx": "onnx", "onnxruntime": "onnxruntime", "numpy": "numpy",
                "huggingface_hub": "huggingface_hub", "soundfile": "soundfile", "scipy": "scipy"}
    if not export_only:
        packages.update(joblib="joblib", librosa="librosa", pandas="pandas", tqdm="tqdm")
    if not USE_HF or "--no-hf" in sys.argv:
        packages.pop("huggingface_hub")
    missing = [package for module, package in packages.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    if not export_only and importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is required; select a Kaggle GPU image or install it first")

"""Take6 chord architecture and full training, without the retired onset head.

Used by take7_training.py; model_trainer.py is the generated standalone entry.
The factory isolates the original chord constants from the short-spectrum DSP.
"""


def chord_runtime(config, store):
    import sys
    import subprocess
    import os
    import random
    import shutil
    import warnings
    import json
    import re
    import math
    from collections import defaultdict

    import joblib
    import librosa
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from tqdm.auto import tqdm

    warnings.filterwarnings("ignore")

    # ==========================================
    # KONFIGURACJA
    # ==========================================
    SR             = 16000
    HOP_LENGTH     = 256
    MIN_NOTE       = 'C1'
    N_BINS         = 144
    BINS_PER_OCTAVE = 24
    INPUT_FEATURES = 168
    CTX_FRAMES     = 48

    # ---- Phase 1: main training ----
    BATCH_SIZE          = config.get("chord_batch_size", 48)
    EPOCHS              = config.get("chord_epochs", 120)
    WARMUP_EPOCHS       = 5
    MAX_LR              = 2e-4
    WEIGHT_DECAY        = 0.02
    DROPOUT_RATE        = 0.2
    SCHED_ETA_MIN       = 5e-6
    EARLY_STOP_PATIENCE = 15

    # ---- Phase 3: pitch head fine-tuning ----
    # DISABLED. Measured on three runs in a row, no effect every time:
    #   take2, 40 epochs: pitch_f1 0.9318 -> 0.9326 (+0.0008), exact 0.5455 -> 0.5445
    #   take3,  4 epochs: F1 0.933 -> 0.931,          exact 54.6% unchanged
    # The encoder is frozen and only the heads train at LR 1e-5, so the phase has
    # nothing to improve with - and it costs ~1.5 h of Kaggle time. Set True only if
    # its meaning changes (e.g. unfreezing the last encoder block).
    RUN_PHASE3          = False
    FINETUNE_EPOCHS     = 40
    FINETUNE_LR         = 1e-5
    FINETUNE_BATCH_SIZE = 64   # bigger batch - encoder frozen, less memory

    # ==========================================
    # RUN NAME - supplied by the take7 entry point. Every checkpoint,
    # ONNX and log name (local and on HF) derives from it, so a new take is one line
    # take7 startup/resume is chosen by the entry point (old snapshots stay intact).
    # ==========================================
    RUN_TAG = config["run_tag"]

    CKPT_BEST  = f"checkpoint_{RUN_TAG}_best.pth"
    CKPT_FT    = f"checkpoint_{RUN_TAG}_finetuned.pth"
    ONNX_BEST  = f"best_model_{RUN_TAG}_chords.onnx"
    ONNX_FT    = f"best_model_{RUN_TAG}_finetuned.onnx"
    HIST_CSV   = f"training_history_{RUN_TAG}.csv"
    HIST_FT    = f"training_history_{RUN_TAG}_finetune.csv"
    LOG_TXT    = f"training_log_{RUN_TAG}.txt"
    INPUT_DIR = str(config["input_dir"])
    WORK_DIR = str(config["work_dir"])
    # The cache holds CQT computed from audio - it depends ONLY on signal parameters,
    # not on the run. Tying it to RUN_TAG made every new run recompute exactly the
    # same features. The name carries the parameter signature, so changing
    # SR/HOP/N_BINS forces a recompute while changing RUN_TAG does not.
    CACHE_DIR  = os.path.join(WORK_DIR,
                              f"cache_feat_sr{SR}_h{HOP_LENGTH}_b{N_BINS}x{BINS_PER_OCTAVE}_{MIN_NOTE}")
    LOG_FILE   = os.path.join(WORK_DIR, HIST_CSV)
    LOG_FT     = os.path.join(WORK_DIR, HIST_FT)


    def cache_key(path):
        """Stable cache file name for a given audio file.

        This used to be `abs(hash(path))`. Python randomises the string hash seed per
        process (PYTHONHASHSEED), so the same audio got a different name every run -
        the cache NEVER hit across sessions and every run recomputed all the CQT.
        Within one process it worked, which is why the log never showed it.
        """
        import hashlib
        return hashlib.sha1(os.path.basename(path).encode("utf-8")).hexdigest()[:16]


    BASS_BOOST_ENABLED  = True
    BASS_BOOST_GAIN     = 5.0
    BASS_BOOST_BINS     = 36

    PITCH_SHIFT_ENABLED    = True
    PITCH_SHIFT_MAX        = 5
    TIME_MASK_ENABLED      = True
    TIME_MASK_MAX_FRAMES   = 8

    # Windows quieter than this fraction of the segment's loudest window are dropped
    # (in the decay the seventh fades first while the label stays -> label noise).
    ENERGY_KEEP_FRAC       = 0.55

    # Which GuitarSet chord annotation to use:
    #   "performed"  — the chord PLAYED, from the hexaphonic pickup transcription
    #   "instructed" — the chord as WRITTEN, i.e. what the player was told to play
    #   "both"       — BOTH (the same excerpt twice with contradictory labels, and the
    #                  duplicate leaked between train and val — do not use)
    #
    # probe_sources.py counted both over 360 files. The segment total is IDENTICAL
    # (4320 = 4320), so this is not "more or less data" but a relabelling of the
    # same recordings:
    #
    #   maj   2640 -> 2106   (-534)      maj7     0 ->  430   (+430)
    #   min    960 ->  460   (-500)      min7     0 ->  360   (+360)
    #   m7b5   240 ->  134   (-106)      dom7   480 ->  694   (+214)
    #                                    sus      0 ->  132   (+132)
    #
    # The key number is `min 960 -> 460` against `min7 0 -> 360`: five hundred
    # segments the score calls "m" were played as "m7". With "instructed" the trainer
    # taught the model to call a voicing with a minor seventh a minor chord — EXACTLY
    # the mistake visible in the app as Gm7 recognised as Gm.
    #
    # Also: "instructed" contains NOT ONE maj7 or min7, so up to take5 both classes
    # came only from the two synthetic renders. Hence 100% on validation (the same
    # instrument on both sides) and fragility on a real guitar.
    #
    # The root does not change with this — probe_root.py found 0 differences over
    # 43056 comparisons, so the switch only touches quality.
    GUITARSET_CHORD_SOURCE = "performed"

    # Pitch targets from the notes ACTUALLY played (GuitarSet hexaphonic pickup).
    # The chord annotation describes the INTENDED chord while a training window is
    # 0.77 s — a comping guitarist plays a fragment of the voicing in it.
    # probe_quality.py showed the effect: on synthetic data (certain labels) seventh
    # recall = 100%, on GuitarSet 32%/20%. The model was punished for not predicting
    # a note that is not in the signal.
    USE_NOTE_MIDI    = True
    NOTE_MIN_COVER   = 0.25    # a note must sound for >= this fraction of the window

    # Mask the root loss where the root is NOT in the window.
    # probe_root.py (360 GuitarSet files, 30653 windows), counting comp and solo
    # together:
    #   the labelled root actually sounds in  64.1% of windows
    #   the root is the lowest note           48.9%
    #   intended root != played root           0.0% (0/43056)
    # At a 2.30 s window that ceiling only rises to 72.1%, at the cost of 2640 chords
    # shorter than the window — a wider frame does not fix it.
    #
    # A guitarist playing a rootless voicing (normal in jazz) produces a signal from
    # which the root CANNOT be derived: G-Bb-D is as much Ebmaj7 without the root as
    # it is Gm. Training on such windows teaches memorising GuitarSet progressions
    # rather than listening, and the shared encoder gets a gradient that contradicts
    # the pitch target. Hence TRAIN Root=83% with TRAIN Qual=98% in v2_take2 — the
    # model could not even memorise its own data, because the same content carries
    # different roots.
    #
    # With True: windows without an audible root contribute nothing to the root loss
    # (pitch and quality still learn from them). Synthetic data always has the root,
    # so it is unaffected. The root metric is then reported split audible/silent.
    MASK_ROOT_WHEN_SILENT = True

    # Train/val split BY SOURCE, not by segment.
    # `random.shuffle(data)` used to shuffle a list of individual chord segments, so
    # neighbouring bars of THE SAME recording landed on both sides: same guitar, same
    # room, same microphone, same take, often the same chord a bar later. Worse in the
    # synthetic set — the `clean` and `eob` renders of one block are the same
    # performance through a different amp, and they went to train and val separately.
    # Hence maj7=100% and min7=100% (those qualities exist ONLY in the synthetic set).
    #
    # Group key: the whole file for GuitarSet, the block for synthetic (both renders
    # together). Consequence: validation metrics WILL DROP. That is not a regression
    # but the removal of an inflation that falsified every generalisation claim.
    SPLIT_BY_FILE = True
    TRAIN_FRAC    = 0.94

    # GuitarSet: SOLO vs COMP recordings.
    # The set has 360 files — accompaniment (`_comp`) and improvisation (`_solo`)
    # for every excerpt. The chord annotation is THE SAME in both cases: it describes
    # the progression the player played over. In a solo file, though, a MONOPHONIC
    # line sounds — the chord is simply not there.
    #
    # Training the root and quality heads on solo files teaches them that a single
    # note is a full chord. That fits everything we measured: quality that will not
    # train, errors correlated in time, min->maj as a systematic belief rather than
    # hesitation.
    #
    # PITCH targets from note_midi are fully correct in solo files (the notes really
    # were played) — and that is exactly the monophonic material we are short of (the
    # "note" class is only 1957 windows). So by default we mask the chord and keep
    # the pitch.
    #   "mask_chord" — pitch trains, root and quality do not   <- default
    #   "drop"       — solo files do not enter the data at all
    #   "keep"       — the previous behaviour (for an A/B comparison)
    GUITARSET_SOLO_MODE = "mask_chord"


    def is_solo_recording(path):
        """GuitarSet names files `05_Jazz2-110-Bb_solo_mix.wav` / `..._comp_mix.wav`."""
        return "_solo" in os.path.basename(path).lower()


    def split_group_key(item):
        """Everything sharing one performance must land on the same side."""
        base = os.path.splitext(os.path.basename(item['path']).lower())[0]
        if "synth" in base:
            # both renders (clean/eob) of the same block -> one group
            return f"synth@{item['start']:.2f}"
        # GuitarSet ships several tracks of the same take (mic/mix/hex)
        for suf in ("_mix", "_mic", "_hex_cln", "_hex", "_cln", "_debleeded"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        return base

    os.makedirs(CACHE_DIR, exist_ok=True)
    TRAIN_LOG = os.path.join(WORK_DIR, LOG_TXT)

    FAMILIES      = ["Major", "Minor", "Dominant", "Dim_HalfDim", "Sus_No3", "None"]
    FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILIES)}
    ROOTS         = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Noise"]
    NOTE_TO_IDX   = {n: i for i, n in enumerate(ROOTS[:-1])}
    NORM_MAP      = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

    # --- QUALITY HEAD (the model's main output) ---
    # Quality comes straight from the label (always correct), which sidesteps the
    # "theoretical vs actually played notes" problem. 'note' = single note, 'N' = noise.
    QUALITIES  = ["maj", "min", "maj7", "dom7", "min7", "m7b5",
                  "dim7", "aug", "sus", "note", "N"]
    QUAL_TO_IDX = {q: i for i, q in enumerate(QUALITIES)}
    QUAL_NOISE  = QUAL_TO_IDX["N"]
    IV_NAMES    = ["R", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🚀 Device: {device} | {RUN_TAG}")

    # ==========================================
    # LOGIKA MUZYCZNA
    # ==========================================
    def get_family(qual):
        q = qual.lower().strip()
        if q in ["n", "noise", "note"]: return "None"
        if q == "": return "Major"     # empty quality = MAJOR TRIAD (label "C"), not None/noise
        # This used to be `re.search(r"(^|[^#b])5", q)`, which caught GuitarSet slash
        # chords (C:maj/5, D:7/5) as sus - 39570 samples, 12% of the set. Now literal.
        if "sus" in q: return "Sus_No3"
        if "dim" in q or "o" in q or "hdim" in q or "m7b5" in q: return "Dim_HalfDim"
        if "maj" in q or re.search(r"6($|[^0-9])", q): return "Major"
        if re.search(r"(^|[^a-z])m(?!aj)", q) or "min" in q or "-" in q: return "Minor"
        if "7" in q or "9" in q or "13" in q or "alt" in q: return "Dominant"
        return "Major"

    def get_quality(qual_str):
        """Maps a raw quality onto one of QUALITIES (dominants 7/9/11/13 -> dom7)."""
        q = qual_str.lower().strip()
        if q == "note":            return "note"
        if q in ["n", "noise"]:    return "N"
        fam = get_family(qual_str)
        if fam == "None":          return "N"
        if fam == "Sus_No3":       return "sus"
        if fam == "Dim_HalfDim":
            if "m7b5" in q or "hdim" in q or "ø" in q: return "m7b5"
            return "dim7"                                   # dim, dim7, o
        if fam == "Major":
            if "aug" in q or "+" in q or "#5" in q:        return "aug"
            if "maj7" in q or "ma7" in q or "maj9" in q or "Δ" in qual_str: return "maj7"
            return "maj"                                    # triada, 6, add9
        if fam == "Minor":
            if "7" in q or "9" in q or "11" in q:          return "min7"
            return "min"                                    # triada, m6
        if fam == "Dominant":
            return "dom7"                                   # 7, 9, 11, 13, alt — lump (9/13 nieuczalne)
        return "maj"

    def get_chord_intervals_with_types(qual):
        q   = qual.lower()
        fam = get_family(q)
        intervals = [(0, 'core')]
        if fam == "Major":
            intervals.extend([(4, 'core'), (7, 'core')])
            if "7" in q: intervals.append((11, 'core'))
        elif fam == "Minor":
            intervals.extend([(3, 'core'), (7, 'core')])
            if "7" in q: intervals.append((10, 'core'))
        elif fam == "Dominant":
            intervals.extend([(4, 'core'), (7, 'core'), (10, 'core')])
        elif fam == "Dim_HalfDim":
            intervals.extend([(3, 'core'), (6, 'core')])
            # ORDER MATTERS: GuitarSet (Harte notation) writes half-diminished as
            # "hdim7", and that string CONTAINS "dim7". Checking dim7 first added a
            # diminished seventh (9) instead of a minor one (10), so the label said
            # "m7b5" while the pitch target described dim7. probe_quality.py showed it
            # as a 20% ceiling for m7b5 even with the TRUE pitch vector.
            if "hdim" in q or "m7b5" in q or "ø" in q:
                intervals.append((10, 'core'))          # half-diminished: minor seventh
            elif "dim7" in q or "o7" in q:
                intervals.append((9, 'core'))           # diminished: diminished seventh
            elif "7" in q:
                intervals.append((10, 'core'))
        elif fam == "Sus_No3":
            intervals.append((7, 'core'))
            if "2" in q: intervals.append((2, 'core'))
            if "4" in q or "sus" in q: intervals.append((5, 'core'))
            if "7" in q: intervals.append((10, 'core'))
        if "9" in q and "b9" not in q and "#9" not in q: intervals.append((2, 'tension'))
        if "b9" in q: intervals.append((1, 'tension'))
        if "#9" in q: intervals.append((3, 'tension'))
        if "11" in q and "#11" not in q: intervals.append((5, 'tension'))
        if "#11" in q or ("b5" in q and fam != "Dim_HalfDim"): intervals.append((6, 'tension'))
        if ("13" in q and "b13" not in q) or "6" in q: intervals.append((9, 'tension'))
        if "b13" in q or "#5" in q or "aug" in q: intervals.append((8, 'tension'))
        return intervals

    def create_targets(root_str, qual_str):
        root_norm = NORM_MAP.get(root_str, root_str)
        if root_norm == "Noise" or root_norm not in ROOTS:
            return 12, QUAL_NOISE, np.zeros(12, dtype=np.float32)
        root_idx  = NOTE_TO_IDX.get(root_norm, 0)
        qual_idx  = QUAL_TO_IDX.get(get_quality(qual_str), QUAL_NOISE)
        # dla 'note' get_chord_intervals_with_types zwraca tylko [(0,'core')] -> pitch = sam root
        pitch_vec = np.zeros(12, dtype=np.float32)
        for semitones, _ in get_chord_intervals_with_types(qual_str):
            pitch_vec[(root_idx + semitones) % 12] = 1.0
        return root_idx, qual_idx, pitch_vec

    def shift_targets(root_idx, pitch_vec, shift):
        if root_idx == 12:
            return root_idx, pitch_vec
        return (root_idx + shift) % 12, np.roll(pitch_vec, shift)

    # ==========================================
    # FILE REGISTRY
    # ==========================================
    class FileRegistry:
        def __init__(self):
            self.exact_map = {}
            self.norm_map  = {}
            self.id_map    = defaultdict(list)
            self.jams      = []
            self.csvs      = []

        def scan_all(self):
            print(f"🔍 Skanowanie {INPUT_DIR}...")
            self._scan(INPUT_DIR)
            print(f"   📂 Audio (ID Groups): {len(self.id_map)}")
            print(f"   📂 Audio (Exact):     {len(self.exact_map)}")

        def _normalize_aggressive(self, n):
            base = os.path.splitext(os.path.basename(n).lower())[0]
            for s in ["_mic", "_mix", "_clean", "_eob", "_raw", "_comp", "_hex"]:
                base = base.replace(s, "")
            return re.sub(r'[^a-z0-9]', '', base)

        def _extract_id(self, filename):
            match = re.match(r"^(\d+)[_.-]", filename)
            return int(match.group(1)) if match else None

        def _scan(self, root):
            if not os.path.exists(root): return
            for r, d, f in os.walk(root):
                for file in f:
                    path = os.path.join(r, file)
                    if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                        self.exact_map[file.lower()] = path
                        self.norm_map[self._normalize_aggressive(file)] = path
                        fid = self._extract_id(file)
                        if fid is not None:
                            self.id_map[fid].append(path)
                    elif file.endswith(".jams"):
                        self.jams.append(path)
                    elif "annotations.csv" in file:
                        self.csvs.append(path)

        def get_files_by_id(self, file_id): return self.id_map.get(int(file_id), [])
        def get_file_by_norm(self, name):   return self.norm_map.get(self._normalize_aggressive(name))
        def get_file_by_exact(self, name):  return self.exact_map.get(os.path.basename(name).lower())

    # ==========================================
    # DATA PARSING
    # ==========================================
    def parse_raw(txt):
        if ":" in txt: return txt.split(":", 1)
        t = txt.strip()
        if t == "N" or t.lower() == "noise": return "Noise", ""
        # single note "Note C" / "Note C#" (previously rejected -> notes were lost)
        m_note = re.match(r"^note\s+([A-G][#b]?)$", t, re.IGNORECASE)
        if m_note: return m_note.group(1), "Note"
        m = re.match(r"^([A-G][#b]?)\s*(.*)$", t)
        if m: return m.group(1), m.group(2)
        return None, None

    def jams_observations(a):
        """JAMS stores 'data' either as a list of observations or as a column dict."""
        dd = a.get("data")
        if isinstance(dd, list):
            return dd
        if isinstance(dd, dict):
            keys = ("time", "duration", "value")
            cols = {k: dd.get(k) or [] for k in keys}
            n = max((len(v) for v in cols.values() if isinstance(v, list)), default=0)
            return [{k: (cols[k][i] if i < len(cols[k]) else None) for k in keys}
                    for i in range(n)]
        return []


    # {audio_path: [(start_s, end_s, pitch_class), ...]} - filled in by load_data
    NOTES_BY_PATH = {}


    def load_data(reg):
        d = []
        jams_kept = defaultdict(int)
        NOTES_BY_PATH.clear()
        for p in reg.jams:
            w = reg.get_file_by_norm(os.path.basename(p))
            if not w: continue
            try:
                with open(p) as f:
                    for a in json.load(f)["annotations"]:
                        # --- notes actually played (6 annotations per file, one per string) ---
                        if USE_NOTE_MIDI and a["namespace"] == "note_midi":
                            ev = NOTES_BY_PATH.setdefault(w, [])
                            for o in jams_observations(a):
                                t, dur, v = o.get("time"), o.get("duration"), o.get("value")
                                if t is None or v is None: continue
                                ev.append((float(t), float(t) + float(dur or 0.0),
                                           int(round(float(v))) % 12))
                            continue
                        if a["namespace"] != "chord":
                            continue
                        # GuitarSet has TWO chord annotations per file:
                        #   data_source ""             -> INTENDED chord: "D#:maj"
                        #   data_source "Semi-auto..." -> PLAYED chord:   "D#:sus2(7)/1"
                        # Taking both produced THE SAME audio twice with contradictory
                        # labels (maj vs sus) - 8640 segments for 4320 excerpts.
                        src = str(a.get("annotation_metadata", {}).get("data_source", "")).lower()
                        performed = "transcription" in src
                        if GUITARSET_CHORD_SOURCE == "instructed" and performed:  continue
                        if GUITARSET_CHORD_SOURCE == "performed" and not performed: continue
                        jams_kept["zagrany" if performed else "zamierzony"] += 1
                        for o in a["data"]:
                            r, q = parse_raw(o["value"])
                            if not r: continue
                            r_norm = NORM_MAP.get(r, r)
                            if r_norm in ROOTS:
                                d.append({
                                    "path": w, "start": o["time"],
                                    "end": o["time"] + o["duration"],
                                    "root": r_norm, "qual": q,
                                    "fam_idx": FAMILY_TO_IDX[get_family(q)]
                                })
            except: pass
        if jams_kept:
            print("   🎸 GuitarSet, chord annotations: " +
                  "  ".join(f"{k}={v}" for k, v in sorted(jams_kept.items())) +
                  f"   (mode: {GUITARSET_CHORD_SOURCE})")

        # --- DIAGNOSTICS: how much of the material is monophonic improvisation? ---
        solo_seg = sum(1 for x in d if is_solo_recording(x['path']))
        solo_fil = len({x['path'] for x in d if is_solo_recording(x['path'])})
        comp_fil = len({x['path'] for x in d if not is_solo_recording(x['path'])})
        if solo_seg:
            print(f"   🎻 GuitarSet SOLO: {solo_fil} files / {solo_seg} segments "
                  f"({solo_seg/max(len(d),1):.0%} of chord annotations)  "
                  f"COMP: {comp_fil} files   -> mode: {GUITARSET_SOLO_MODE}")
            if GUITARSET_SOLO_MODE == "drop":
                d = [x for x in d if not is_solo_recording(x['path'])]
                print(f"      dropped the solo material, {len(d)} segments left")

        custom_cnt = 0
        unparsed   = defaultdict(int)     # diagnostyka: etykiety odrzucone przez parser
        unmatched  = defaultdict(int)     # diagnostics: rows with no matching wav
        for p in reg.csvs:
            try:
                df    = pd.read_csv(p, sep=None, engine='python')
                cols  = df.columns
                c_f   = next((c for c in cols if 'file'  in c or 'audio' in c), None)
                c_l   = next((c for c in cols if 'label' in c or 'chord' in c), None)
                c_s   = next((c for c in cols if 'start' in c), None)
                c_e   = next((c for c in cols if 'end'   in c), None)
                if not (c_f and c_l and c_s and c_e):
                    print(f"   ⚠️ CSV skipped (no file/label/start/end columns): {os.path.basename(p)} | columns: {list(cols)}")
                    continue
                rows_added = 0
                for _, row in df.iterrows():
                    val       = row[c_f]
                    wav_paths = []
                    sval      = str(val).strip()
                    # 1) EXACT file name - unambiguous, preferred
                    w_exact = reg.get_file_by_exact(sval)
                    if w_exact:
                        wav_paths = [w_exact]
                    elif isinstance(val, (int, float)) or sval.isdigit():
                        # 2) numeric ID - COLLIDES: GuitarSet has "01_BN1-129-Eb_comp.wav"
                        # (01 = player number), so ID=1 matched 01_triads_*.wav AND 60
                        # GuitarSet recordings -> synthetic labels pasted onto unrelated
                        # audio (764 CSV rows blown up into 47368 segments).
                        cand = reg.get_files_by_id(int(float(val)))
                        if len(cand) > 4:
                            print(f"   ⛔ ID '{sval}' matches {len(cand)} files - COLLISION, "
                                  f"skipping. Use exact file names in the file column.")
                            unmatched[sval] += 1
                            continue
                        wav_paths = cand
                    else:
                        w = reg.get_file_by_norm(sval)
                        if w: wav_paths = [w]
                    if not wav_paths:
                        unmatched[str(val)] += 1; continue
                    r, q = parse_raw(str(row[c_l]))
                    if not r:
                        unparsed[str(row[c_l])] += 1; continue
                    r_norm = NORM_MAP.get(r, r)
                    if r_norm in ROOTS:
                        for w in wav_paths:
                            d.append({
                                "path": w, "start": float(row[c_s]),
                                "end": float(row[c_e]),
                                "root": r_norm, "qual": q,
                                "fam_idx": FAMILY_TO_IDX[get_family(q)]
                            })
                            custom_cnt += 1
                            rows_added += 1
                print(f"   📄 {os.path.basename(p)}: +{rows_added} segments")
            except Exception as e:
                print(f"   ⚠️ CSV {os.path.basename(p)}: {e}")

        if unparsed:
            top = sorted(unparsed.items(), key=lambda kv: -kv[1])[:10]
            print("   ⚠️ Etykiety ODRZUCONE przez parser: " + "  ".join(f"'{k}'x{v}" for k, v in top))
        if unmatched:
            top = sorted(unmatched.items(), key=lambda kv: -kv[1])[:5]
            print("   ⚠️ Rows with no wav: " + "  ".join(f"'{k}'x{v}" for k, v in top))

        # SEGMENT quality distribution (before windowing) - shows at once if notes got in
        seg_q = defaultdict(int)
        for item in d: seg_q[get_quality(item['qual'])] += 1
        print("   📊 Segment qualities: " + "  ".join(f"{k}={v}" for k, v in sorted(seg_q.items())))
        print(f"   📊 {len(d)} segments ({custom_cnt} synthetic, {len(d)-custom_cnt} GuitarSet)")
        return d

    # ==========================================
    # PRZETWARZANIE AUDIO
    # ==========================================
    def process_audio_file(path):
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
            if len(y) < HOP_LENGTH * CTX_FRAMES: return None, 0
            cqt     = librosa.cqt(y, sr=SR, hop_length=HOP_LENGTH,
                                   fmin=librosa.note_to_hz(MIN_NOTE),
                                   n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
            cqt_abs = np.abs(cqt)
            if BASS_BOOST_ENABLED:
                cqt_abs[:BASS_BOOST_BINS, :] *= BASS_BOOST_GAIN
            norm   = np.clip((librosa.amplitude_to_db(cqt_abs, ref=np.max) + 80) / 80, 0, 1)
            chroma = librosa.feature.chroma_cqt(C=norm, sr=SR, hop_length=HOP_LENGTH,
                                                 n_chroma=12, bins_per_octave=BINS_PER_OCTAVE)
            bass_energy = np.zeros((12, norm.shape[1]), dtype=np.float32)
            for i in range(12):
                bass_energy[i, :] = np.mean(norm[i * 2: i * 2 + 2, :], axis=0)
            feat = np.vstack([norm, chroma, bass_energy]).T.astype(np.float32)
            return feat, feat.shape[0]
        except:
            return None, 0

    # ==========================================
    # DATASET
    # ==========================================
    # ==========================================
    # PITCH SHIFT
    # ==========================================
    def shift_feature_bins(feat, shift):
        if shift == 0: return feat
        feat = feat.copy()
        def roll_zero(a, sh):                 # shift with ZERO fill (not a wrap)
            out = np.zeros_like(a)
            if sh > 0:   out[:, sh:] = a[:, :-sh]
            elif sh < 0: out[:, :sh] = a[:, -sh:]
            else:        out[:] = a
            return out
        # CQT (0:144, 2 bins/semitone) and bass (156:168) are LINEAR in frequency
        # -> ZERO-FILL. np.roll (wrap) used to push upper harmonics into the low
        # bins and into the boosted bass, inventing PHANTOM notes -> the model did
        # not see sevenths (min7=0%, maj7=10%).
        feat[:, 0:144]   = roll_zero(feat[:, 0:144],   shift * 2)
        feat[:, 156:168] = roll_zero(feat[:, 156:168], shift)
        # chroma (144:156): pitch classes are CYCLIC -> wrap is correct here
        feat[:, 144:156] = np.roll(feat[:, 144:156], shift, axis=1)
        return feat


    class FrameBasedDataset(Dataset):
        def __init__(self, data_list, training=True):
            self.training = training
            self.epoch    = 0
            self.samples  = []

            unique_paths = list(set(d['path'] for d in data_list))
            self.cache_map = {}
            to_process   = []

            for p in unique_paths:
                h  = cache_key(p)
                cp = os.path.join(CACHE_DIR, f"{h}.npy")
                if os.path.exists(cp):
                    try:
                        shape = np.load(cp, mmap_mode='r').shape
                        self.cache_map[p] = (cp, shape[0])
                    except:
                        to_process.append(p)
                else:
                    to_process.append(p)

            if to_process:
                print(f"⚙️ Computing CQT for {len(to_process)} files...")
                results = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(process_audio_file)(p) for p in tqdm(to_process)
                )
                for p, (feat, n_frames) in zip(to_process, results):
                    if feat is not None:
                        h  = cache_key(p)
                        cp = os.path.join(CACHE_DIR, f"{h}.npy")
                        np.save(cp, feat)
                        self.cache_map[p] = (cp, n_frames)

            # --- map of notes actually sounding: {cache_path: [n_frames, 12] uint8} ---
            # Built once per file from the note_midi annotation. Lets us compute the
            # pitch target for a SPECIFIC window instead of inheriting it from the
            # whole chord segment.
            self.pitch_map = {}
            if USE_NOTE_MIDI and NOTES_BY_PATH:
                for path, (cp, n_frames) in self.cache_map.items():
                    ev = NOTES_BY_PATH.get(path)
                    if not ev: continue
                    pm = np.zeros((n_frames, 12), dtype=np.uint8)
                    for (t0, t1, pc) in ev:
                        f0 = max(0, int(t0 * SR / HOP_LENGTH))
                        f1 = min(n_frames, int(np.ceil(t1 * SR / HOP_LENGTH)))
                        if f1 > f0: pm[f0:f1, pc] = 1
                    self.pitch_map[cp] = pm
                if self.pitch_map:
                    print(f"   🎵 Pitch targets from note_midi for {len(self.pitch_map)} files "
                          f"(the rest: from the chord)")

            # Cache paths that come from SOLO recordings - there the chord annotation
            # describes accompaniment that is not in the signal.
            self.solo_cp = {cp for path, (cp, _) in self.cache_map.items()
                            if is_solo_recording(path)}

            stride = 4 if training else 16
            gated_windows = 0
            for item in data_list:
                if item['path'] not in self.cache_map: continue
                cp, n_frames = self.cache_map[item['path']]
                s_f = int(item['start'] * SR / HOP_LENGTH)
                e_f = min(int(item['end'] * SR / HOP_LENGTH), n_frames)
                if e_f - s_f <= CTX_FRAMES: continue
                fam_idx     = item['fam_idx']
                # root-aware: get_quality is blind to the root (for root=Noise, qual=""
                # would give "maj"), create_targets returns QUAL_NOISE correctly
                _, qual_idx, _ = create_targets(item['root'], item['qual'])
                curr_stride = max(1, stride // 2) if (training and fam_idx in [2, 3]) else stride

                # --- WINDOW ENERGY GATE (critical for sevenths) ---
                # Synthetic segments are attack + a long decay (letRing). In the decay
                # the seventh - the quietest note of the voicing - disappears first
                # while the label still says "m7", which is systematic label noise
                # teaching the collapse m7->m, Maj7->maj. We drop windows whose energy
                # is below ENERGY_KEEP_FRAC * the segment's peak window (the equivalent
                # of the app's noise gate). Class N (noise) is not gated.
                cand = list(range(s_f, e_f - CTX_FRAMES, curr_stride))
                if qual_idx != QUAL_NOISE and len(cand) > 1:
                    frame_e = np.load(cp, mmap_mode='r')[s_f:e_f, :144].mean(axis=1)
                    cum = np.concatenate([[0.0], np.cumsum(frame_e, dtype=np.float64)])
                    w_e = np.array([(cum[t - s_f + CTX_FRAMES] - cum[t - s_f]) / CTX_FRAMES
                                    for t in cand])
                    keep_thr = ENERGY_KEEP_FRAC * w_e.max()
                    kept = [t for t, e in zip(cand, w_e) if e >= keep_thr]
                    gated_windows += len(cand) - len(kept)
                    cand = kept if kept else [cand[int(np.argmax(w_e))]]

                for t in cand:
                    self.samples.append({
                        'npy_path': cp, 'frame_idx': t,
                        'root': item['root'], 'qual': item['qual'],
                        'fam_idx': fam_idx, 'qual_idx': qual_idx
                    })
            if gated_windows:
                print(f"   🔇 Energy gate: dropped {gated_windows} windows from decay/silence")

        def set_epoch(self, ep): self.epoch = ep
        def __len__(self):       return len(self.samples)

        def augment_features(self, feat):
            tilt  = np.linspace(random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), feat.shape[1])
            feat  = feat * tilt.astype(np.float32)
            feat += np.random.randn(*feat.shape).astype(np.float32) * random.uniform(0.005, 0.025)
            if TIME_MASK_ENABLED and random.random() < 0.4:
                mask_len   = random.randint(1, TIME_MASK_MAX_FRAMES)
                mask_start = random.randint(0, CTX_FRAMES - mask_len)
                feat[mask_start: mask_start + mask_len, :] = 0.0
            if random.random() < 0.3:
                f_start = random.randint(0, 130)
                f_len   = random.randint(4, 14)
                feat[:, f_start: min(f_start + f_len, 144)] = 0.0
            return np.clip(feat, 0.0, 1.0)

        def pitch_shift_features(self, feat, shift):
            return shift_feature_bins(feat, shift)

        def __getitem__(self, idx):
            s    = self.samples[idx]
            feat = np.load(s['npy_path'], mmap_mode='r')[s['frame_idx']: s['frame_idx'] + CTX_FRAMES].copy()
            root_idx, qual_idx, pitch_vec = create_targets(s['root'], s['qual'])

            # Pitch target from the notes ACTUALLY sounding in THIS window, when
            # note_midi is available. Root and quality stay from the chord annotation -
            # they describe the harmony of the passage while pitch must describe the
            # signal. Without this the model was punished for not predicting a seventh
            # that is not in the window (b7 recall on GuitarSet: 32%).
            pm = self.pitch_map.get(s['npy_path']) if self.pitch_map else None
            if pm is not None and qual_idx != QUAL_NOISE:
                win = pm[s['frame_idx']: s['frame_idx'] + CTX_FRAMES]
                if len(win) == CTX_FRAMES:
                    pitch_vec = (win.mean(axis=0) >= NOTE_MIN_COVER).astype(np.float32)

            # Does the root actually sound in this window? Computed AFTER pitch_vec is
            # settled but BEFORE the pitch shift - a shift moves root and pitch
            # together, so the flag is invariant to it. For synthetic data pitch_vec
            # comes from the chord and always contains the root, so the flag is 1.
            root_ok = 1.0 if root_idx == 12 else float(pitch_vec[root_idx] > 0.5)

            # Solo recording: a monophonic line labelled with the accompaniment chord.
            # Pitch (from note_midi) stays correct, the chord does not.
            qual_ok = 1.0
            if GUITARSET_SOLO_MODE == "mask_chord" and s['npy_path'] in self.solo_cp:
                root_ok = qual_ok = 0.0

            if self.training:
                # quality is INVARIANT to a pitch shift (a shifted C7 is still "7")
                if PITCH_SHIFT_ENABLED and root_idx != 12:
                    shift = random.randint(-PITCH_SHIFT_MAX, PITCH_SHIFT_MAX)
                    if shift != 0:
                        feat     = self.pitch_shift_features(feat, shift)
                        root_idx, pitch_vec = shift_targets(root_idx, pitch_vec, shift)
                feat = self.augment_features(feat)
            return (
                torch.tensor(feat,      dtype=torch.float32),
                torch.tensor(root_idx,  dtype=torch.long),
                torch.tensor(qual_idx,  dtype=torch.long),
                torch.tensor(pitch_vec, dtype=torch.float32),
                torch.tensor(root_ok,   dtype=torch.float32),
                torch.tensor(qual_ok,   dtype=torch.float32)
            )

    # ==========================================
    # FOCAL LOSS
    # ==========================================
    class FocalBCELoss(nn.Module):
        def __init__(self, gamma=2.0, pos_weight=2.5):
            super().__init__()
            self.gamma      = gamma
            self.pos_weight = pos_weight

        def forward(self, logits, targets):
            pw  = torch.tensor(self.pos_weight, device=logits.device)
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction='none')
            pt  = torch.exp(-bce)
            return ((1.0 - pt) ** self.gamma * bce).mean()

    # ==========================================
    # MODEL
    # ==========================================
    class SEBlock(nn.Module):
        def __init__(self, c, r=16):
            super().__init__()
            self.sq = nn.AdaptiveAvgPool2d(1)
            self.ex = nn.Sequential(
                nn.Linear(c, c // r, bias=False), nn.GELU(),
                nn.Linear(c // r, c, bias=False), nn.Sigmoid()
            )

        def forward(self, x):
            return x * self.ex(self.sq(x).view(x.size(0), x.size(1))).view(x.size(0), x.size(1), 1, 1)


    class ConvBlockSE(nn.Module):
        def __init__(self, i, o, dropout=0.1):
            super().__init__()
            self.c = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.GroupNorm(8, o), nn.GELU(),
                SEBlock(o), nn.MaxPool2d((1, 2)),
                nn.Dropout2d(dropout)
            )

        def forward(self, x): return self.c(x)


    class ChordTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inorm = nn.InstanceNorm2d(1, affine=True)
            self.enc   = nn.Sequential(
                ConvBlockSE(1,   48,  dropout=DROPOUT_RATE * 0.50),
                ConvBlockSE(48,  96,  dropout=DROPOUT_RATE * 0.50),
                ConvBlockSE(96,  192, dropout=DROPOUT_RATE * 0.75),
                ConvBlockSE(192, 384, dropout=DROPOUT_RATE * 0.75)
            )
            self.proj = nn.Linear(3840, 384)
            self.cls  = nn.Parameter(torch.randn(1, 1, 384))
            self.pos  = nn.Parameter(torch.randn(1, CTX_FRAMES + 1, 384))
            layer     = nn.TransformerEncoderLayer(
                d_model=384, nhead=8, dim_feedforward=768,
                dropout=DROPOUT_RATE, activation='gelu',
                batch_first=True, norm_first=True
            )
            self.tr       = nn.TransformerEncoder(layer, num_layers=4)
            self.fc_root  = nn.Sequential(
                nn.LayerNorm(384), nn.Dropout(DROPOUT_RATE * 0.5), nn.Linear(384, 13)
            )
            # MAIN quality output (root + quality = chord)
            self.fc_quality = nn.Sequential(
                nn.LayerNorm(384),
                nn.Linear(384, 192), nn.GELU(), nn.Dropout(DROPOUT_RATE),
                nn.Linear(192, 96),  nn.GELU(), nn.Dropout(DROPOUT_RATE * 0.5),
                nn.Linear(96, len(QUALITIES))
            )
            # AUXILIARY pitch output (multi-task; teaches the encoder harmony)
            self.fc_pitch = nn.Sequential(
                nn.LayerNorm(384),
                nn.Linear(384, 128), nn.GELU(), nn.Dropout(DROPOUT_RATE),
                nn.Linear(128, 64),  nn.GELU(), nn.Dropout(DROPOUT_RATE * 0.5),
                nn.Linear(64, 12)
            )
        def forward(self, x):
            x = self.enc(self.inorm(x.unsqueeze(1)))
            b, c, t, f = x.size()
            x = self.proj(x.permute(0, 2, 1, 3).reshape(b, t, c * f))
            x = torch.cat((self.cls.expand(b, -1, -1), x), 1) + self.pos
            emb = self.tr(x)[:, 0]
            return self.fc_root(emb), self.fc_quality(emb), self.fc_pitch(emb)

        def freeze_encoder(self):
            """Freezes everything but the quality+pitch heads - for phase 3."""
            for name, param in self.named_parameters():
                param.requires_grad = ("fc_pitch" in name) or ("fc_quality" in name)
            frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"   🔒 Frozen: {frozen:,} parameters | Trainable: {trainable:,}")

        def unfreeze_all(self):
            for param in self.parameters():
                param.requires_grad = True

    # ==========================================
    # METRYKI
    # ==========================================
    def compute_pitch_metrics(pred_logits, targets, threshold=0.5):
        pred = (torch.sigmoid(pred_logits) > threshold).float()
        tp   = (pred * targets).sum().item()
        fp   = (pred * (1.0 - targets)).sum().item()
        fn   = ((1.0 - pred) * targets).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2.0 * prec * rec / (prec + rec + 1e-8)
        return f1, prec, rec

    def compute_chord_exact_match(root_logits, pitch_logits, root_gt, pitch_gt, threshold=0.5):
        root_pred  = root_logits.argmax(dim=1)
        pitch_pred = (torch.sigmoid(pitch_logits) > threshold).float()
        return ((root_pred == root_gt) & (pitch_pred == pitch_gt).all(dim=1)).float().mean().item()

    def evaluate(model, loader, threshold=0.5):
        """Ewaluacja. exact = root ORAZ quality trafione (metryka istotna dla apki)."""
        model.eval()
        c_r = c_q = c_exact = tot = 0
        per_q_ok = defaultdict(int); per_q_tot = defaultdict(int)
        confus = defaultdict(int)                 # (true_qual, pred_qual) -> count, errors only
        iv_ok = np.zeros(12); iv_tot = np.zeros(12)   # pitch-head recall per INTERVAL from root
        all_f1, all_prec, all_rec = [], [], []
        r_aud_ok = r_aud_tot = r_sil_ok = r_sil_tot = 0   # root split by root audibility
        q_aud_ok = q_sil_ok = e_aud_ok = 0                # the same for quality and exact
        n_skip = 0                                        # windows whose chord label is unusable
        with torch.no_grad():
            for x, root, qual, pitch, root_ok, qual_ok in loader:
                x, root, qual, pitch = x.to(device), root.to(device), qual.to(device), pitch.to(device)
                root_ok, qual_ok = root_ok.to(device), qual_ok.to(device)
                out_root, out_qual, out_pitch = model(x)
                rp = out_root.argmax(1); qp = out_qual.argmax(1)
                # Chord metrics count ONLY where the chord label describes the signal.
                # In solo recordings it describes the accompaniment, so measuring
                # anything chord-related there is measuring noise.
                ch   = qual_ok > 0.5
                ok_r = (rp == root) & ch; ok_q = (qp == qual) & ch
                n_ch = ch.sum().item()
                c_r += ok_r.sum().item(); c_q += ok_q.sum().item()
                c_exact += (ok_r & ok_q).sum().item(); tot += n_ch
                n_skip += root.size(0) - n_ch
                aud = (root_ok > 0.5) & ch
                r_aud_tot += aud.sum().item();  r_aud_ok += (ok_r & aud).sum().item()
                sil = (~(root_ok > 0.5)) & ch
                r_sil_tot += sil.sum().item(); r_sil_ok += (ok_r & sil).sum().item()
                # Chord quality is defined RELATIVE TO THE ROOT: {G,Bb,D} is min if the
                # root is G and rootless maj7 if it is Eb. Where the root is inaudible,
                # quality is as undecidable as the root - so we measure it separately.
                q_aud_ok += (ok_q & aud).sum().item()
                q_sil_ok += (ok_q & sil).sum().item()
                e_aud_ok += (ok_r & ok_q & aud).sum().item()
                # per quality we count QUALITY-ONLY (root separately), to tell whether
                # min7=0% is the quality head's fault or the root head's
                for qi, pi, keep in zip(qual.tolist(), qp.tolist(), ch.tolist()):
                    if not keep: continue
                    per_q_tot[qi] += 1
                    if qi == pi: per_q_ok[qi] += 1
                    else:        confus[(qi, pi)] += 1
                f1, prec, rec = compute_pitch_metrics(out_pitch, pitch, threshold)
                all_f1.append(f1); all_prec.append(prec); all_rec.append(rec)
                # pitch-head recall per interval from the root: does it SEE b7 (pos. 10)?
                pp = (torch.sigmoid(out_pitch) > threshold).float().cpu().numpy()
                pg = pitch.cpu().numpy(); rg = root.cpu().numpy()
                for i in range(len(rg)):
                    if rg[i] >= 12: continue
                    idx = (rg[i] + np.arange(12)) % 12
                    tgt = pg[i][idx] > 0.5
                    iv_tot += tgt
                    iv_ok  += tgt & (pp[i][idx] > 0.5)
        acc_r = c_r / tot if tot else 0
        acc_q = c_q / tot if tot else 0
        exact = c_exact / tot if tot else 0
        # Best-checkpoint selection uses the root accuracy MEASURED ON WINDOWS WITH AN
        # AUDIBLE ROOT. The combined root_acc has a ~64% ceiling (probe_root.py) imposed
        # by the labels, so it would reward a model that guesses GuitarSet progressions
        # well rather than one that listens well. root_audible has a 100% ceiling.
        acc_r_aud = (r_aud_ok / r_aud_tot) if r_aud_tot else 0.0
        composite = (acc_r_aud + acc_q + exact) / 3.0
        iv_rec = {IV_NAMES[i]: (iv_ok[i] / iv_tot[i]) for i in range(12) if iv_tot[i] > 50}
        per_q = {QUALITIES[qi]: per_q_ok[qi] / per_q_tot[qi]
                 for qi in sorted(per_q_tot) if per_q_tot[qi] > 0}
        top_conf = sorted(confus.items(), key=lambda kv: -kv[1])[:8]
        conf_str = "  ".join(f"{QUALITIES[a]}->{QUALITIES[b]}:{n}" for (a, b), n in top_conf)
        return {
            'root_acc': acc_r, 'qual_acc': acc_q, 'exact': exact,
            'f1': float(np.mean(all_f1)), 'prec': float(np.mean(all_prec)),
            'rec': float(np.mean(all_rec)), 'composite': composite,
            'per_qual': per_q, 'confusions': conf_str, 'iv_recall': iv_rec,
            # root_audible matches how the app is really used: a student practising a
            # chord plays it with its root. root_silent measures guessing from context
            # and is inherently low - that is not a defect of the model.
            'root_audible': acc_r_aud,
            'root_silent':  (r_sil_ok / r_sil_tot) if r_sil_tot else 0.0,
            'root_aud_frac': (r_aud_tot / tot) if tot else 0.0,
            'qual_audible': (q_aud_ok / r_aud_tot) if r_aud_tot else 0.0,
            'qual_silent':  (q_sil_ok / r_sil_tot) if r_sil_tot else 0.0,
            'exact_audible': (e_aud_ok / r_aud_tot) if r_aud_tot else 0.0,
            'chord_skipped': n_skip,
        }

    # ==========================================
    # HF UTILS
    # ==========================================
    def upload_file_safe(file_path, name_in_repo):
        # Branch ONNX files are intermediates; only take7 publishes the final graph.
        if not name_in_repo.endswith(".onnx"):
            store.publish(file_path, name_in_repo)

    def export_onnx(model, save_path, threshold=0.5):
        """
        Exports the model to ONNX. Temporarily clears requires_grad on all parameters
        - the PyTorch ONNX exporter does not handle a mixed state (some frozen, some
        not).
        """
        model.eval()
        # remember each parameter's requires_grad
        grad_state = {name: p.requires_grad for name, p in model.named_parameters()}
        # clear it for the duration of the export
        for p in model.parameters():
            p.requires_grad_(False)
        # Disable the fused TransformerEncoderLayer fast path. Newer PyTorch (the Kaggle
        # image) fuses the layer into aten::_transformer_encoder_layer_fwd, which the ONNX
        # exporter supports on no opset -> UnsupportedOperatorError. This forces the slow,
        # exportable path for the export only; weights and architecture are unchanged.
        try:
            _fastpath_prev = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
        except Exception:
            _fastpath_prev = None
        try:
            dummy_x = torch.randn(1, CTX_FRAMES, INPUT_FEATURES).to(device)
            torch.onnx.export(
                model, (dummy_x,), save_path,
                input_names=["features"],
                output_names=["root_logits", "quality_logits", "pitch_logits"],
                dynamic_axes={"features": {0: "batch"},
                              "root_logits": {0: "batch"},
                              "quality_logits": {0: "batch"},
                              "pitch_logits": {0: "batch"}},
                opset_version=17, dynamo=False
            )
        finally:
            # restore the original requires_grad state
            for name, p in model.named_parameters():
                p.requires_grad_(grad_state.get(name, True))
            if _fastpath_prev is not None:
                try: torch.backends.mha.set_fastpath_enabled(_fastpath_prev)
                except Exception: pass
        # store the threshold and the quality taxonomy as ONNX custom metadata
        try:
            import onnx
            m = onnx.load(save_path)
            for k, v in [("pitch_threshold", str(threshold)),
                         ("qualities", ",".join(QUALITIES)),
                         ("roots", ",".join(ROOTS))]:
                meta = m.metadata_props.add(); meta.key = k; meta.value = v
            onnx.save(m, save_path)
        except: pass

    def load_weights(model, state):
        # Older four-head snapshots are accepted only by dropping the retired head.
        state = {k: v for k, v in state.items() if not k.startswith("fc_onset.")}
        model.load_state_dict(state, strict=True)


    def load_checkpoint_meta():
        path = store.fetch(CKPT_BEST)
        return torch.load(path, map_location="cpu", weights_only=False) if path else None


    def resume_from_checkpoint(model, opt, sched):
        ckpt = load_checkpoint_meta()
        if ckpt is None:
            return 0, 0., 999., 0., 0., False, .5
        load_weights(model, ckpt['model_state_dict'])
        if not ckpt.get('phase1_done', False):
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            sched.load_state_dict(ckpt['scheduler_state_dict'])
        return (ckpt['epoch'] + 1, ckpt.get('best_composite', 0.),
                ckpt.get('best_loss', 999.), ckpt.get('best_f1', 0.),
                ckpt.get('best_exact', 0.), ckpt.get('phase1_done', False),
                ckpt.get('best_threshold', .5))


    def save_checkpoint(model, opt, sched, ep, best_composite, best_loss, best_f1,
                        best_exact, acc_r, avg_f1, avg_prec, avg_rec, avg_exact,
                        phase1_done=False, best_threshold=0.5):
        ckpt_save = os.path.join(WORK_DIR, CKPT_BEST)
        torch.save({
            'epoch':                ep,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': opt.state_dict() if opt is not None else {},
            'scheduler_state_dict': sched.state_dict() if sched is not None else {},
            'best_composite':       best_composite,
            'best_loss':            best_loss,
            'best_f1':              best_f1,
            'best_exact':           best_exact,
            'phase1_done':          phase1_done,
            'best_threshold':       best_threshold,
            'metrics': {
                'root_acc':    acc_r,    'pitch_f1':  avg_f1,
                'pitch_prec':  avg_prec, 'pitch_rec': avg_rec,
                'chord_exact': avg_exact
            }
        }, ckpt_save)
        upload_file_safe(ckpt_save, CKPT_BEST)

    # ==========================================
    # PHASE 1 - MAIN TRAINING
    # ==========================================
    def phase1_train(model, tr_l, te_l, tr_eval_l=None):
        print("\n" + "="*60)
        print("PHASE 1 - MAIN TRAINING")
        print("="*60)

        opt  = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

        steps_per_epoch = len(tr_l)
        warmup_steps    = WARMUP_EPOCHS * steps_per_epoch
        total_steps     = EPOCHS * steps_per_epoch
        min_ratio       = SCHED_ETA_MIN / MAX_LR

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        loss_root_fn  = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05)
        loss_root_none = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.05)
        loss_qual_none = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.05)
        loss_qual_fn  = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05)  # the sampler balances classes
        loss_pitch_fn = FocalBCELoss(gamma=2.0, pos_weight=2.5)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))          # AMP: 2-3x faster
        QUAL_W = 1.5    # quality = main output; 2.0 choked the encoder (root 96->85%) on noisy labels

        start_epoch, best_composite, best_loss, best_f1, best_exact, phase1_done, _ = \
            resume_from_checkpoint(model, opt, sched)

        if phase1_done:
            print("✅ Phase 1 already done - skipping")
            return best_composite, best_loss, best_f1, best_exact

        if start_epoch == 0:
            with open(LOG_FILE, "w") as f:
                f.write("epoch,loss,loss_root,loss_qual,loss_pitch,root_acc,root_audible,"
                        "qual_acc,pitch_f1,chord_exact,composite,lr\n")

        print(f"   Epoki {start_epoch} → {EPOCHS} | "
              f"LR {MAX_LR:.0e}→{SCHED_ETA_MIN:.0e} | Batch {BATCH_SIZE}")

        no_improve_count = 0
        ds_tr = tr_l.dataset

        for ep in range(start_epoch, EPOCHS):
            ds_tr.set_epoch(ep)
            model.train()

            pitch_weight = 0.7                      # auxiliary head, constant light weight

            loop = tqdm(tr_l, desc=f"Ep {ep+1}/{EPOCHS}", leave=False)
            losses, root_losses, qual_losses, pitch_losses = [], [], [], []

            for x, root, qual, pitch, root_ok, qual_ok in loop:
                x, root, qual, pitch = x.to(device), root.to(device), qual.to(device), pitch.to(device)
                root_ok, qual_ok = root_ok.to(device), qual_ok.to(device)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    out_root, out_qual, out_pitch = model(x)
                    if MASK_ROOT_WHEN_SILENT:
                        # Windows without an audible root contribute nothing to the
                        # root loss. Averaging over the valid samples only keeps the
                        # loss scale - and so this head's effective LR - independent of
                        # the batch composition.
                        lr_all = loss_root_none(out_root, root)
                        loss_r = (lr_all * root_ok).sum() / root_ok.sum().clamp(min=1.0)
                    else:
                        loss_r = loss_root_fn(out_root, root)
                    # Solo: the chord does not sound, so the quality head gets no
                    # gradient from it. Pitch still learns from these windows.
                    lq_all = loss_qual_none(out_qual, qual)
                    loss_q = (lq_all * qual_ok).sum() / qual_ok.sum().clamp(min=1.0)
                    loss_p = loss_pitch_fn(out_pitch, pitch)
                    loss   = loss_r + QUAL_W * loss_q + pitch_weight * loss_p
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())
                root_losses.append(loss_r.item())
                qual_losses.append(loss_q.item())
                pitch_losses.append(loss_p.item())

            mean_loss   = float(np.mean(losses))
            mean_loss_r = float(np.mean(root_losses))
            mean_loss_q = float(np.mean(qual_losses))
            mean_loss_p = float(np.mean(pitch_losses))
            current_lr  = opt.param_groups[0]['lr']

            m = evaluate(model, te_l)
            acc_r, acc_q, avg_f1 = m['root_acc'], m['qual_acc'], m['f1']
            avg_prec, avg_rec, avg_exact, composite = m['prec'], m['rec'], m['exact'], m['composite']
            improved = ep == 0 or composite > best_composite + 1e-6

            print(
                f"📉 Ep {ep+1:3d} | Loss {mean_loss:.3f} (R:{mean_loss_r:.2f} Q:{mean_loss_q:.2f} P:{mean_loss_p:.2f}) | "
                f"Root: {acc_r:.1%} | Qual: {acc_q:.1%} | Exact: {avg_exact:.1%} | "
                f"pF1: {avg_f1:.3f} | Comp: {composite:.3f} | LR: {current_lr:.1e}"
                + (" ⭐" if improved else "")
            )
            # Root split: windows where the root SOUNDS (how a student practising a
            # chord plays it) vs windows where it does not (a rootless voicing -
            # guessing from context). probe_root.py: the root is missing in ~36% of
            # GuitarSet windows, so the combined root_acc is capped and says little.
            if m.get('chord_skipped'):
                print(f"   (chord metrics skip {m['chord_skipped']} solo windows - "
                      f"there the label describes the accompaniment, not the signal)")
            print(f"   root AUDIBLE ({m['root_aud_frac']:.0%} of windows): "
                  f"root={m['root_audible']:.1%} qual={m['qual_audible']:.1%} "
                  f"exact={m['exact_audible']:.1%}")
            print(f"   root INAUDIBLE:                "
                  f"root={m['root_silent']:.1%} qual={m['qual_silent']:.1%}")
            if m.get('per_qual'):     # EVERY epoch (not only on improvement)
                worst = sorted(m['per_qual'].items(), key=lambda kv: kv[1])[:6]
                print("   qualities (qual-only): " + "  ".join(f"{k}={v:.0%}" for k, v in worst))
                if m.get('confusions'):
                    print("   confusions: " + m['confusions'])
                if m.get('iv_recall'):
                    print("   pitch recall by interval: " +
                          "  ".join(f"{k}={v:.0%}" for k, v in m['iv_recall'].items()))
            # SANITY GATE: can the model even memorise its OWN training data?
            if tr_eval_l is not None and (ep % 5 == 0 or ep == start_epoch):
                mt = evaluate(model, tr_eval_l)
                print(f"   🎓 TRAIN (no augmentation): Root={mt['root_acc']:.1%} "
                      f"(audible={mt['root_audible']:.1%}) "
                      f"Qual={mt['qual_acc']:.1%} Exact={mt['exact']:.1%}"
                      f"   [val Exact={avg_exact:.1%}]")
                # Diagnose from the train-vs-val COMPARISON, not an absolute threshold.
                # A low train accuracy and a large train-val gap are opposite problems
                # and point in opposite directions.
                if ep >= 25:
                    gap = mt['qual_acc'] - acc_q
                    if gap > 0.15:
                        print(f"      ⚠️ Quality OVERFITTING: train-val gap {gap:.1%} "
                              f"({mt['qual_acc']:.1%} vs {acc_q:.1%}) -> regularisation / more "
                              f"data, not more epochs.")
                    elif mt['qual_acc'] < 0.85:
                        print(f"      ⚠️ UNDERFITTING: train qual {mt['qual_acc']:.1%} with a "
                              f"{gap:.1%} gap -> the model cannot memorise its own data; "
                              f"suspect the FEATURES/labels.")

            with open(LOG_FILE, "a") as f:
                f.write(f"{ep+1},{mean_loss:.4f},{mean_loss_r:.4f},{mean_loss_q:.4f},{mean_loss_p:.4f},"
                        f"{acc_r:.4f},{m['root_audible']:.4f},{acc_q:.4f},{avg_f1:.4f},"
                        f"{avg_exact:.4f},{composite:.4f},{current_lr:.2e}\n")
            upload_file_safe(LOG_FILE, HIST_CSV)

            if improved:
                best_composite = composite
                best_loss      = mean_loss
                best_f1        = avg_f1
                best_exact     = avg_exact
                no_improve_count = 0
                save_checkpoint(model, opt, sched, ep, best_composite, best_loss,
                                best_f1, best_exact, acc_r, avg_f1, avg_prec,
                                avg_rec, avg_exact, phase1_done=False)
                # ONNX is NOT exported every epoch. Early on every epoch improves, so
                # that meant 87 MB of checkpoint + 29 MB of ONNX to HF each time. The
                # checkpoint is needed (to resume after a Kaggle session dies), the
                # ONNX is not - it is produced from it at the end of the phase.
                print(f"   💾 New best: Root={acc_r:.1%} F1={avg_f1:.3f} "
                      f"Exact={avg_exact:.1%} Comp={composite:.3f}")
            else:
                no_improve_count += 1

            if (ep + 1) % 10 == 0:
                ckpt_name = f"checkpoint_{RUN_TAG}_ep{ep+1}.pth"
                ckpt_save = os.path.join(WORK_DIR, ckpt_name)
                torch.save({
                    'epoch': ep, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': sched.state_dict(),
                    'best_composite': best_composite, 'best_loss': best_loss,
                    'best_f1': best_f1, 'best_exact': best_exact,
                }, ckpt_save)
                upload_file_safe(ckpt_save, ckpt_name)
                # The notebook owns stdout; checkpoint and CSV are persisted above.

            if ep >= WARMUP_EPOCHS + 10 and no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"\n⏹️  Early stopping after epoch {ep+1} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

        # mark phase 1 as done in the checkpoint, keeping the last best metrics
        ckpt_meta = load_checkpoint_meta()
        if ckpt_meta:
            m_saved = ckpt_meta.get('metrics', {})
            load_weights(model, ckpt_meta['model_state_dict'])
            save_checkpoint(model, None, None,
                            ckpt_meta['epoch'], best_composite, best_loss,
                            best_f1, best_exact,
                            m_saved.get('root_acc', 0), m_saved.get('pitch_f1', 0),
                            m_saved.get('pitch_prec', 0), m_saved.get('pitch_rec', 0),
                            m_saved.get('chord_exact', 0), phase1_done=True)
            # ONNX once, from the best phase 1 weights
            load_weights(model, ckpt_meta['model_state_dict'])
            save_path = os.path.join(WORK_DIR, ONNX_BEST)
            export_onnx(model, save_path)
            upload_file_safe(save_path, ONNX_BEST)

        print(f"\n✅ Phase 1 done | Best Comp={best_composite:.3f} "
              f"F1={best_f1:.3f} Exact={best_exact:.1%}")
        return best_composite, best_loss, best_f1, best_exact

    # ==========================================
    # PHASE 2 - THRESHOLD TUNING
    # ==========================================
    def phase2_threshold_tuning(model, te_l):
        print("\n" + "="*60)
        print("PHASE 2 - THRESHOLD TUNING")
        print("="*60)

        # is the threshold already stored (phase 2 already done)?
        ckpt_meta = load_checkpoint_meta()
        if ckpt_meta and ckpt_meta.get('best_threshold', 0.5) != 0.5 \
                and ckpt_meta.get('phase2_done', False):
            best_thr = ckpt_meta['best_threshold']
            print(f"✅ Phase 2 already done - best threshold: {best_thr:.2f}")
            return best_thr

        print("   Skanowanie threshold 0.30 → 0.70 co 0.01...")
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.71, 0.01)]
        results    = []

        for thr in tqdm(thresholds, desc="Threshold scan"):
            m = evaluate(model, te_l, threshold=thr)
            results.append((thr, m['exact'], m['composite'], m['f1']))

        # Sort by the PITCH HEAD's F1 - the only metric the threshold affects. We used
        # to sort by 'exact', but exact = argmax(root) AND argmax(quality), so it was
        # identical for all 41 thresholds and the choice came out at random.
        results.sort(key=lambda x: (x[3], x[2]), reverse=True)

        print("\n   Top 5 threshold:")
        for thr, exact, comp, f1 in results[:5]:
            print(f"   threshold={thr:.2f} | pF1={f1:.4f} | Exact={exact:.1%} | Comp={comp:.3f}")

        best_thr = results[0][0]
        best_f1  = results[0][3]
        print(f"\n   🎯 Best threshold: {best_thr:.2f} (pF1={best_f1:.4f})")
        print(f"      It affects ONLY the pitch head (note detection);")
        print(f"      root and quality use argmax and are independent of it.")

        # write it into the checkpoint
        if ckpt_meta:
            m_saved = ckpt_meta.get('metrics', {})
            ckpt_save = os.path.join(WORK_DIR, CKPT_BEST)
            ckpt_meta['best_threshold'] = best_thr
            ckpt_meta['phase2_done']    = True
            torch.save(ckpt_meta, ckpt_save)
            upload_file_safe(ckpt_save, CKPT_BEST)

        # Zaktualizuj ONNX z nowym threshold w metadanych
        onnx_path = os.path.join(WORK_DIR, ONNX_BEST)
        if os.path.exists(onnx_path):
            try:
                import onnx
                m_onnx = onnx.load(onnx_path)
                # drop the old threshold if present
                for prop in list(m_onnx.metadata_props):
                    if prop.key == "pitch_threshold":
                        m_onnx.metadata_props.remove(prop)
                meta       = m_onnx.metadata_props.add()
                meta.key   = "pitch_threshold"
                meta.value = str(best_thr)
                onnx.save(m_onnx, onnx_path)
                upload_file_safe(onnx_path, ONNX_BEST)
                print(f"   💾 ONNX updated with threshold={best_thr:.2f}")
            except Exception as e:
                print(f"   ⚠️  Cannot update the ONNX metadata: {e}")

        return best_thr

    # ==========================================
    # PHASE 4 - THE ONSET HEAD
    # ==========================================
    def phase3_finetune_pitch(model, tr_l, te_l, best_threshold):
        print("\n" + "="*60)
        print("PHASE 3 - PITCH HEAD FINE-TUNING")
        print("="*60)

        # has phase 3 already run?
        ckpt_meta = load_checkpoint_meta()
        if ckpt_meta and ckpt_meta.get('phase3_done', False):
            print("✅ Phase 3 already done - skipping")
            return

        # best phase 1 weights are already in the model if we came through phase 2;
        # freeze the encoder - only fc_pitch trains
        model.freeze_encoder()

        opt_ft = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY
        )
        sched_ft = optim.lr_scheduler.CosineAnnealingLR(
            opt_ft, T_max=FINETUNE_EPOCHS, eta_min=FINETUNE_LR * 0.1
        )
        loss_pitch_fn = FocalBCELoss(gamma=2.0, pos_weight=2.5)
        loss_qual_fn  = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05)

        # Bigger batch - a frozen encoder needs less memory
        ft_loader = DataLoader(
            tr_l.dataset, batch_size=FINETUNE_BATCH_SIZE,
            sampler=tr_l.sampler, shuffle=False, num_workers=0, pin_memory=True
        )

        print(f"   Epoki: {FINETUNE_EPOCHS} | LR: {FINETUNE_LR:.0e} | "
              f"Batch: {FINETUNE_BATCH_SIZE} | Threshold: {best_threshold:.2f}")

        best_exact_ft  = 0.0
        best_composite_ft = 0.0

        with open(LOG_FT, "w") as f:
            f.write("epoch,loss_pitch,root_acc,pitch_f1,pitch_prec,pitch_rec,"
                    "chord_exact,composite,lr\n")

        ds_tr = ft_loader.dataset
        for ep in range(FINETUNE_EPOCHS):
            ds_tr.set_epoch(ep)
            model.train()

            loop = tqdm(ft_loader, desc=f"FT Ep {ep+1}/{FINETUNE_EPOCHS}", leave=False)
            pitch_losses = []

            for x, root, qual, pitch, _root_ok, _qual_ok in loop:
                x, root, qual, pitch = x.to(device), root.to(device), qual.to(device), pitch.to(device)
                opt_ft.zero_grad()
                out_root, out_qual, out_pitch = model(x)
                # phase 3 tunes the quality (main) + pitch (aux) heads; encoder frozen
                loss_q = loss_qual_fn(out_qual, qual)
                loss_p = loss_pitch_fn(out_pitch, pitch)
                loss   = loss_q + 0.5 * loss_p
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_ft.step()
                pitch_losses.append(loss_p.item())

            sched_ft.step()
            mean_loss_p = float(np.mean(pitch_losses))
            current_lr  = opt_ft.param_groups[0]['lr']

            m = evaluate(model, te_l, threshold=best_threshold)
            acc_r, avg_f1, avg_prec = m['root_acc'], m['f1'], m['prec']
            avg_rec, avg_exact, composite = m['rec'], m['exact'], m['composite']
            # A 1e-6 margin counted the fourth decimal as an improvement, so "new best"
            # fired EVERY epoch and pushed 29 MB .pth + 29 MB ONNX to HF each time.
            # 1e-3 is 0.1 pp - the smallest change worth reporting.
            improved = composite > best_composite_ft + 1e-3

            print(
                f"🎸 FT {ep+1:2d} | PitchLoss: {mean_loss_p:.4f} | "
                f"Root: {acc_r:.1%} | F1: {avg_f1:.3f} (P={avg_prec:.3f} R={avg_rec:.3f}) | "
                f"Exact: {avg_exact:.1%} | Comp: {composite:.3f} | LR: {current_lr:.2e}"
                + (" ⭐" if improved else "")
            )

            with open(LOG_FT, "a") as f:
                f.write(f"{ep+1},{mean_loss_p:.4f},{acc_r:.4f},{avg_f1:.4f},"
                        f"{avg_prec:.4f},{avg_rec:.4f},{avg_exact:.4f},"
                        f"{composite:.4f},{current_lr:.2e}\n")
            upload_file_safe(LOG_FT, HIST_FT)

            if improved:
                best_exact_ft     = avg_exact
                best_composite_ft = composite

                # save the fine-tuned checkpoint
                ckpt_save = os.path.join(WORK_DIR, CKPT_FT)
                torch.save({
                    'epoch':            ep,
                    'model_state_dict': model.state_dict(),
                    'best_threshold':   best_threshold,
                    'best_exact':       avg_exact,
                    'best_composite':   composite,
                    'phase3_done':      False,
                    'metrics': {
                        'root_acc': acc_r, 'pitch_f1': avg_f1,
                        'pitch_prec': avg_prec, 'pitch_rec': avg_rec,
                        'chord_exact': avg_exact
                    }
                }, ckpt_save)
                upload_file_safe(ckpt_save, CKPT_FT)
                print(f"   💾 New FT best: Exact={avg_exact:.1%} Comp={composite:.3f}")

        # mark phase 3 as done
        ckpt_save = os.path.join(WORK_DIR, CKPT_FT)
        if os.path.exists(ckpt_save):
            ckpt_ft = torch.load(ckpt_save, map_location='cpu', weights_only=False)
            ckpt_ft['phase3_done'] = True
            torch.save(ckpt_ft, ckpt_save)
            upload_file_safe(ckpt_save, CKPT_FT)
            # ONNX is exported ONCE, from the best weights - not every epoch. Exporting
            # and uploading 29 MB each epoch cost more than the fine-tuning itself.
            load_weights(model, ckpt_ft['model_state_dict'])
            save_path = os.path.join(WORK_DIR, ONNX_FT)
            export_onnx(model, save_path, threshold=best_threshold)
            upload_file_safe(save_path, ONNX_FT)

        # unfreeze the encoder in case anything else still uses the model
        model.unfreeze_all()

        print(f"\n✅ Phase 3 done | Best Exact={best_exact_ft:.1%} "
              f"(threshold={best_threshold:.2f})")

    # ==========================================
    # MAIN
    # ==========================================
    def main():
        # --- data (shared by all phases) ---
        reg = FileRegistry()
        reg.scan_all()
        data = load_data(reg)
        if len(data) < 100:
            print("❌ Not enough data!")
            return

        random.seed(42)
        if SPLIT_BY_FILE:
            groups = sorted({split_group_key(x) for x in data})
            random.shuffle(groups)
            n_val      = max(1, int(round(len(groups) * (1.0 - TRAIN_FRAC))))
            val_groups = set(groups[:n_val])
            tr_items = [x for x in data if split_group_key(x) not in val_groups]
            vl_items = [x for x in data if split_group_key(x) in val_groups]
            print(f"🔒 Split BY SOURCE: {len(groups)} groups -> "
                  f"{len(groups)-n_val} train / {n_val} val")
        else:
            random.shuffle(data)
            split = int(len(data) * TRAIN_FRAC)
            tr_items, vl_items = data[:split], data[split:]
            print("⚠️  RANDOM split by segment - validation metrics are inflated.")
        ds_tr  = FrameBasedDataset(tr_items, training=True)
        ds_vl  = FrameBasedDataset(vl_items, training=False)
        print(f"📊 Train: {len(ds_tr)} samples | Val: {len(ds_vl)} samples")

        # Sampler weighted by QUALITY, so the rare jazz classes (m7b5, dim7, maj7) get a
        # chance. SQRT-inverse rather than 1/count: at a 100:1 imbalance plain 1/count
        # repeated rare windows ~100x (overfitting); sqrt caps the repeat at ~10x.
        train_targets  = [s['qual_idx'] for s in ds_tr.samples]
        class_counts   = np.bincount(train_targets, minlength=len(QUALITIES))
        print("   Quality distribution (train): " +
              "  ".join(f"{QUALITIES[i]}={c}" for i, c in enumerate(class_counts) if c > 0))
        class_weights  = 1.0 / np.sqrt(class_counts + 1.0)
        sample_weights = [class_weights[t] for t in train_targets]
        sampler        = WeightedRandomSampler(weights=sample_weights,
                                               num_samples=len(sample_weights),
                                               replacement=True)

        tr_l = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler,
                          shuffle=False, num_workers=0, pin_memory=True)
        te_l = DataLoader(ds_vl, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

        # Loader for measuring accuracy ON THE TRAINING DATA (no augmentation, a subset
        # about the size of val). If the model cannot memorise its own data, the fault
        # is in the features or the labels.
        ds_tr_eval = FrameBasedDataset(tr_items, training=False)
        idx_eval   = random.sample(range(len(ds_tr_eval)), min(4000, len(ds_tr_eval)))
        tr_eval_l  = DataLoader(torch.utils.data.Subset(ds_tr_eval, idx_eval),
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = ChordTransformer().to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🧠 Model parameters: {total_params:,}")

        # --- Phase 1 ---
        phase1_train(model, tr_l, te_l, tr_eval_l)

        # --- Phase 2: threshold tuning on the best phase 1 model ---
        # reload the best weights (later phase 1 epochs may have changed the model)
        ckpt_meta = load_checkpoint_meta()
        if ckpt_meta:
            load_weights(model, ckpt_meta['model_state_dict'])
            print("🔄 Loaded the best weights for threshold tuning")

        best_threshold = phase2_threshold_tuning(model, te_l)

        # --- Phase 3: pitch head fine-tuning ---
        # reload the best phase 1 weights (phase 2 did not change the model)
        if RUN_PHASE3:
            if ckpt_meta:
                load_weights(model, ckpt_meta['model_state_dict'])
            phase3_finetune_pitch(model, tr_l, te_l, best_threshold)
        else:
            print("\n⏭️  Phase 3 skipped (RUN_PHASE3=False) - see the comment on the flag.")

        # Rise is trained separately by the take7 entry point.
        result = load_checkpoint_meta()
        if result is None:
            raise RuntimeError("Chord training did not produce a checkpoint")
        load_weights(model, result['model_state_dict'])
        export_onnx(model, os.path.join(WORK_DIR, ONNX_BEST), best_threshold)
        return result

    from types import SimpleNamespace
    return SimpleNamespace(model=ChordTransformer, train=main, export=export_onnx,
                           load_weights=load_weights, device=device,
                           phase1=phase1_train, phase2=phase2_threshold_tuning)


def _onset_runtime():
    INPUT_DIR = "/kaggle/input"

    AUDIO_VARIANT = "auto"

    OUTPUT_ROOT = "/kaggle/working"

    import argparse

    from collections import Counter, defaultdict

    import hashlib

    import json

    import math

    from pathlib import Path

    import re

    import sys

    def observations(annotation):
        data = annotation.get("data")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            columns = [data.get(k) for k in ("time", "duration", "value")]
            if not all(isinstance(c, list) for c in columns) or len({len(c) for c in columns}) != 1:
                raise ValueError("Malformed JAMS observation columns")
            return [dict(zip(("time", "duration", "value"), row)) for row in zip(*columns)]
        raise ValueError("Unknown JAMS observation layout")

    def note_events(document, take):
        events, strings, namespaces = [], [], Counter()
        for ai, annotation in enumerate(document.get("annotations", [])):
            namespace = annotation.get("namespace", "unknown")
            namespaces[namespace] += 1
            if namespace != "note_midi":
                continue
            raw_string = annotation.get("annotation_metadata", {}).get("data_source")
            if str(raw_string) not in {str(n) for n in range(6)}:
                raise ValueError(f"Unknown note string data_source: {raw_string!r}")
            string = int(raw_string)
            strings.append(string)
            for oi, observation in enumerate(observations(annotation)):
                t, duration, midi = (float(observation[k]) for k in ("time", "duration", "value"))
                if not all(math.isfinite(v) for v in (t, duration, midi)) or t < 0 or duration <= 0 or not 0 <= midi <= 127:
                    raise ValueError(f"Invalid note {ai}:{oi}")
                # MIDI annotations can be fractional. Keep the original pitch;
                # make the semitone rounding used by the 12-class head explicit.
                events.append({"id": f"{take}:{ai}:{oi}", "t": t, "end": t + duration,
                               "midi_value": midi, "midi": int(round(midi)), "pc": int(round(midi)) % 12,
                               "near_semitone_boundary": abs(midi - round(midi)) >= .4,
                               "string": string, "string_source": "annotation_metadata.data_source",
                               "pluck_verified": False})
        if sorted(strings) != list(range(6)):
            raise ValueError(f"Expected one note_midi annotation per string, got {strings}")
        if not events:
            raise ValueError("No annotated notes")
        return sorted(events, key=lambda e: (e["t"], e["id"])), dict(namespaces)

    def source_identity(take):
        match = re.fullmatch(r"(0[0-5])_(.+)_(solo|comp)", take)
        if not match:
            raise ValueError(f"Unknown GuitarSet take name: {take}")
        player, performance, style = match.groups()
        return {"take": take, "player": player, "style": style,
                "paired_take_group": f"{player}_{performance}", "split_group": f"player:{player}"}

    def scan_inputs(root):
        audio, annotations = defaultdict(list), defaultdict(list)
        variants = {v: [] for v in ("mic", "mix", "hex", "hex_cln", "untagged")}
        extensions, archives = Counter(), []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            extension = path.suffix.lower()
            if extension in (".wav", ".flac", ".ogg", ".mp3"):
                audio[path.stem].append(path)
                extensions[extension] += 1
                tag = next((v for v in ("hex_cln", "hex", "mic", "mix")
                            if path.stem.endswith("_" + v)), "untagged")
                variants[tag].append(path)
            elif extension == ".jams":
                annotations[path.stem].append(path)
            elif extension in (".zip", ".tar", ".tgz", ".gz"):
                archives.append(str(path))
        matching = {v: sum(bool(audio.get(f"{take}_{v}")) for take in annotations)
                    for v in ("mic", "mix")}
        inventory = {"audio_files": sum(extensions.values()), "audio_extensions": dict(extensions),
                     "jams_files": sum(len(paths) for paths in annotations.values()),
                     "matching_takes": matching,
                     "variants": {v: {"files": len(paths), "examples": [str(p) for p in paths[:3]]}
                                  for v, paths in variants.items() if paths},
                     "archives": {"count": len(archives), "examples": archives[:3]}}
        return audio, annotations, inventory

    def audit(root, variant, validation_player=None, test_player=None):
        import soundfile as sf
        if (validation_player is None) != (test_player is None) or (validation_player is not None and validation_player == test_player):
            raise ValueError("Specify two different validation/test performers, or neither")
        if variant not in ("auto", "mic", "mix"):
            raise ValueError("Audio variant must be auto, mic or mix")
        audio_files, annotations, inventory = scan_inputs(root)
        records, errors = [], []
        requested_variant = variant
        if variant == "auto":
            available = [v for v, count in inventory["matching_takes"].items() if count]
            variant = available[0] if len(available) == 1 else None
            if annotations and not available:
                errors.append({"error": "No mic/mix audio filenames match the JAMS takes. See inventory for paths, formats and archives."})
            elif len(available) > 1:
                errors.append({"error": "Both mic and mix match JAMS takes. Set AUDIO_VARIANT explicitly to mic or mix."})
        missing_audio = [take for take in sorted(annotations)
                         if variant and not audio_files.get(f"{take}_{variant}")]
        if missing_audio:
            errors.append({"error": f"Missing {variant} audio for {len(missing_audio)} takes. See inventory and missing_audio_takes in the report.",
                           "count": len(missing_audio), "examples": missing_audio[:5]})
        for take, paths in sorted(annotations.items()) if variant else []:
            try:
                if len(paths) != 1:
                    raise ValueError(f"Ambiguous JAMS: {[str(p) for p in paths]}")
                identity = source_identity(take)
                audio = audio_files.get(f"{take}_{variant}", [])
                if not audio:
                    continue  # Already reported together, rather than once per take.
                if len(audio) != 1:
                    raise ValueError(f"Ambiguous {variant} audio: {[str(p) for p in audio]}")
                info = sf.info(audio[0])
                duration = info.frames / info.samplerate
                if info.channels != 1:
                    raise ValueError(f"Expected mono {variant}, got {info.channels} channels")
                document = json.loads(paths[0].read_text())
                events, namespaces = note_events(document, take)
                if any(e["end"] > duration + .02 for e in events):
                    raise ValueError("Annotations extend beyond WAV")
                split = "unassigned"
                if validation_player is not None:
                    split = ("validation" if identity["player"] == validation_player else
                             "test" if identity["player"] == test_player else "train")
                nearby = []
                # These can collide in a 96 ms, 12-class target; retain both events.
                for i, event in enumerate(events):
                    for j in range(i - 1, -1, -1):
                        prior = events[j]
                        if event["t"] - prior["t"] >= .096:
                            break
                        if event["pc"] == prior["pc"]:
                            nearby.append([prior["id"], event["id"]])
                records.append({**identity, "split": split, "variant": variant,
                                "audio": {"path": str(audio[0].resolve()), "sha256": sha256(audio[0]),
                                          "channels": info.channels, "samplerate": info.samplerate,
                                          "frames": info.frames, "duration": duration},
                                "jams": {"path": str(paths[0].resolve()), "sha256": sha256(paths[0]),
                                         "version": document.get("file_metadata", {}).get("jams_version")},
                                "namespaces": namespaces, "events": events,
                                "same_pc_within_96ms": nearby})
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                errors.append({"take": take, "error": str(error)})
        if not annotations:
            errors.append({"error": "No JAMS files found"})
        if validation_player is not None:
            for split in ("train", "validation", "test"):
                if not any(r["split"] == split for r in records):
                    errors.append({"error": f"Empty {split} split"})
        return {"schema_version": 1, "root": str(root.resolve()), "ok": not errors,
                "requested_variant": requested_variant, "selected_variant": variant,
                "inventory": inventory, "missing_audio_takes": missing_audio,
                "summary": {"jams_takes": len(annotations), "usable_takes": len(records),
                            "styles": dict(Counter(r["style"] for r in records)),
                            "splits": dict(Counter(r["split"] for r in records)),
                            "notes": sum(len(r["events"]) for r in records)},
                "limitations": ["Note annotations do not certify pick/pluck technique.",
                                "Frozen encoder source exposure is unknown.",
                                "Synthetic chord CSV is not per-attack ground truth; audit it separately.",
                                "Dataset version is not inferred from directory names; file hashes identify inputs."],
                "errors": errors, "sources": records}

    import argparse

    from collections import Counter

    import hashlib

    import json

    import math

    from pathlib import Path

    import sys

    import numpy as np

    import soundfile as sf

    VALIDATION_PLAYER = "04"

    TEST_PLAYER = "05"

    SYNTHETIC_GROUPS = (60, 12, 12)

    SEED = 20260922

    SR = 16000

    SPLITS = ("train", "validation", "test")

    GENERATOR_VERSION = "onset-ks-v2"

    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def write_json(path, document):
        # Readers never see a partially written success report.
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def split_sources(document, validation_player, test_player):
        """Validate the audited sources and assign whole performers before rendering."""
        players = {f"{n:02}" for n in range(6)}
        if validation_player not in players or test_player not in players or validation_player == test_player:
            raise ValueError("Choose two different GuitarSet players (00..05)")
        if document.get("ok") is not True or document.get("schema_version") != 1 or document.get("errors"):
            raise ValueError("Expected a successful version-1 full GuitarSet audit")
        if not document.get("sources"):
            raise ValueError("Use the full audit manifest, not its small _summary.json")
        counts = document["summary"]
        if (counts["usable_takes"] != len(document["sources"]) or
                counts["notes"] != sum(len(s["events"]) for s in document["sources"])):
            raise ValueError("Source/event counts disagree with the audit summary")
        if document.get("selected_variant") not in ("mic", "mix"):
            raise ValueError("Expected audited mono mic/mix sources")
        sources, ids, takes, owners = [], set(), set(), {}
        for original in sorted(document["sources"], key=lambda s: s["take"]):
            take, player = original["take"], original["player"]
            if (player not in players or not take.startswith(player + "_") or
                    original["style"] not in ("solo", "comp") or
                    not take.endswith("_" + original["style"]) or take in takes):
                raise ValueError(f"Invalid or duplicate source identity: {take}")
            takes.add(take)
            split = "validation" if player == validation_player else "test" if player == test_player else "train"
            if original["split"] not in ("unassigned", split):
                raise ValueError(f"Refusing to change an assigned split: {take}")
            if original["variant"] != document["selected_variant"]:
                raise ValueError(f"Mixed audio variants: {take}")
            audio = original["audio"]
            duration = audio["duration"]
            if (audio["channels"] != 1 or not math.isfinite(duration) or duration <= 0 or
                    audio["samplerate"] <= 0 or audio["frames"] <= 0 or
                    abs(duration - audio["frames"] / audio["samplerate"]) > 1e-9):
                raise ValueError(f"Invalid mono audio metadata: {take}")
            # Audio content, source path and paired performances may not cross splits.
            for key in ("audio:" + audio["sha256"], "path:" + audio["path"],
                        "pair:" + original["paired_take_group"]):
                if key in owners and owners[key] != split:
                    raise ValueError(f"Source leakage across splits: {take} ({key})")
                owners[key] = split
            if not original["events"]:
                raise ValueError(f"Missing note annotations: {take}")
            for event in original["events"]:
                if event["id"] in ids:
                    raise ValueError(f"Duplicate event ID: {event['id']}")
                ids.add(event["id"])
                if (not all(math.isfinite(event[k]) for k in ("t", "end", "midi_value")) or
                        not 0 <= event["t"] < event["end"] <= duration + .02 or
                        not 0 <= event["midi_value"] <= 127 or
                        event["midi"] != round(event["midi_value"]) or
                        event["pc"] != event["midi"] % 12 or event["string"] not in range(6) or
                        event.get("pluck_verified") is not False):
                    raise ValueError(f"Invalid or unjustifiably verified GuitarSet note: {event['id']}")
            sources.append({**original, "split": split, "split_group": f"player:{player}",
                            "source_id": f"guitarset:{take}", "label_kind": "annotated_note_start",
                            "encoder_exposure": "unknown", "base_onset_head_exposure": "unknown"})
        if {s["split"] for s in sources} != set(SPLITS):
            raise ValueError("All three source splits must be nonempty")
        return sources

    def verify_source_files(sources):
        for index, source in enumerate(sources):
            for kind in ("audio", "jams"):
                item = source[kind]
                if sha256(item["path"]) != item["sha256"]:
                    raise ValueError(f"Changed since audit: {item['path']}")
            info = sf.info(source["audio"]["path"])
            if (info.channels, info.samplerate, info.frames) != tuple(
                    source["audio"][k] for k in ("channels", "samplerate", "frames")):
                raise ValueError(f"Audio metadata changed: {source['take']}")
            if (index + 1) % 60 == 0:
                print(f"Verified {index + 1}/{len(sources)} GuitarSet sources", flush=True)

    def excitation_seed(seed, split, group, role):
        # Separate namespaces from onset_contrasts.py, independent of generation order.
        key = f"{GENERATOR_VERSION}:{seed}:{split}:{group}:{role}"
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")

    def pluck(midi, frames, sr, seed, damping, attack_seconds):
        # The two-tap averaging filter adds half a sample of delay. The old
        # integer ring instead had effective period round(sr/f)-0.5: at 16kHz,
        # requested E5 (MIDI76) became 680.85Hz, closer to F5 than E5.
        # Add fractional delay BEFORE averaging, so total delay is sr/f. Keep
        # the recurrence explicit and causal, with zero history before sample0.
        period = sr / (440 * 2 ** ((midi - 69) / 12))
        delay = math.floor(period - .5)
        fraction = period - .5 - delay
        if frames < 0 or delay < 2 or not 0 < damping <= 1 or attack_seconds <= 0:
            raise ValueError("Invalid pluck parameters")
        excitation = np.random.default_rng(seed).uniform(-1, 1, delay)
        excitation = np.convolve(excitation, [.5, .5], mode="same")
        wave = np.zeros(frames, dtype=np.float64)
        count = min(frames, delay)
        wave[:count] = excitation[:count]
        weights = (.5 * (1 - fraction), .5, .5 * fraction)
        for i in range(delay, frames):
            past = i - delay
            wave[i] = damping * (weights[0] * wave[past] +
                                 (weights[1] * wave[past - 1] if past >= 1 else 0.) +
                                 (weights[2] * wave[past - 2] if past >= 2 else 0.))
        wave *= np.minimum(1, np.arange(frames) / (attack_seconds * sr))
        return wave

    def render_group(split, group, seed=SEED, sr=SR):
        """One split owns all stems/variants; pairs have identical background and gain."""
        if split not in SPLITS or group < 0 or seed < 0 or sr < 8000:
            raise ValueError("Invalid synthetic split/group/seed/sample rate")
        source_id = f"{GENERATOR_VERSION}-{seed}-{split}-{group:04}"
        rng = np.random.default_rng(excitation_seed(seed, split, group, "parameters"))
        root = 40 + group % 12 + int(rng.choice([0, 12]))
        third = int(rng.choice([3, 4]))
        gap = float(rng.choice([.2, .32, .48, .64, 1.2, 2.0]))
        level = float(rng.choice([.25, .5, 1., 1.5]))
        strum = float(rng.choice([0., .012, .025]))
        damping = float(rng.choice([.995, .998, .9995]))
        attack = float(rng.choice([.001, .003, .006]))
        initial = round(1.5 * sr)
        challenge = initial + round(gap * sr)
        total = challenge + round(3.5 * sr)
        specs = {}
        for i, midi in enumerate([root, root + third, root + 7]):
            specs[f"context_{i}"] = (midi, initial + round(i * strum * sr), 1., "context")
            specs[f"restrum_{i}"] = (midi, challenge + round(i * strum * sr), level, "challenge")
        for name, midi in (("root", root), ("third", root + third), ("fifth", root + 7), ("octave", root + 12)):
            specs[name] = (midi, challenge, level, "challenge")
        stems, events = {}, {}
        for name, (midi, start, amplitude, role) in specs.items():
            exc_seed = excitation_seed(seed, split, group, name)
            wave = np.zeros(total, dtype=np.float64)
            wave[start:] = pluck(midi, total - start, sr, exc_seed, damping, attack) * amplitude
            stems[name] = wave
            events[name] = {"source_id": f"{source_id}:{name}", "t": start / sr,
                            "sample": start, "midi": midi, "pc": midi % 12,
                            "role": role, "excitation_seed": exc_seed, "level": amplitude,
                            "pluck_verified": True, "label_kind": "synthetic_excitation"}
        single = ["context_0"]
        triad = [f"context_{i}" for i in range(3)]
        cases = {"root_hold": single, "root_plus_third": single + ["third"],
                 "root_plus_fifth": single + ["fifth"], "root_repluck": single + ["root"],
                 "root_plus_octave": single + ["octave"], "triad_hold": triad,
                 "triad_fifth": triad + ["fifth"],
                 "triad_repluck": triad + [f"restrum_{i}" for i in range(3)]}
        tracks = {case: sum((stems[key] for key in keys), np.zeros(total)) for case, keys in cases.items()}
        peak = float(rng.choice([.2, .4, .8]))
        gain = peak / max(float(np.max(np.abs(track))) for track in tracks.values())
        clips = []
        for case, keys in cases.items():
            name = f"{source_id}-{case}"
            clip_events = [dict(events[key], id=f"{name}:{i}", case=f"{case}/{events[key]['role']}")
                           for i, key in enumerate(keys)]
            clips.append({"name": name, "source_group": source_id, "parent_groups": [source_id],
                          "split": split, "case": case, "seed": seed, "sr": sr,
                          "root_midi": root, "third_semitones": third, "gap_seconds": gap,
                          "challenge_at": challenge / sr, "challenge_level": level,
                          "strum_seconds": strum, "damping": damping, "attack_seconds": attack,
                          "gain": gain, "frames": total, "duration": total / sr,
                          "expected_new_pcs": sorted({e["pc"] for e in clip_events if e["role"] == "challenge"}),
                          "audio": (tracks[case] * gain).astype(np.float32), "events": clip_events})
        return clips

    def source_summary(sources):
        result = {}
        for split in SPLITS:
            selected = [s for s in sources if s["split"] == split]
            result[split] = {"sources": len(selected),
                             "styles": dict(Counter(s["style"] for s in selected)),
                             "players": sorted({s["player"] for s in selected}),
                             "notes": sum(len(s["events"]) for s in selected),
                             "same_pc_pairs_within_96ms": sum(len(s["same_pc_within_96ms"]) for s in selected),
                             "seconds": sum(s["audio"]["duration"] for s in selected)}
        return result

    def prepare(manifest_path, output, validation_player=VALIDATION_PLAYER, test_player=TEST_PLAYER,
                groups=SYNTHETIC_GROUPS, seed=SEED, sr=SR):
        import csv
        if len(groups) != 3 or any(n < 1 for n in groups) or seed < 0 or sr < 8000:
            raise ValueError("Need three positive group counts, nonnegative seed and sample rate >=8000")
        # Read bytes once: the recorded digest identifies the document actually used.
        manifest_bytes = manifest_path.read_bytes()
        sources = split_sources(json.loads(manifest_bytes), validation_player, test_player)
        output.mkdir(parents=True, exist_ok=False)
        summary_path = output / "summary.json"
        summary = {"schema_version": 1, "ok": False, "training_ready": False,
                   "generator": GENERATOR_VERSION, "samplerate": sr,
                   "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                   "input_manifest": str(manifest_path.resolve()), "seed": seed,
                   "validation_player": validation_player, "test_player": test_player,
                   "guitarset": source_summary(sources), "synthetic": {},
                   "limitations": ["GuitarSet labels note starts, not verified picking technique.",
                                   "Existing encoder and base onset head exposure to GuitarSet is unknown.",
                                   "Synthetic test uses independent excitations of the same generator.",
                                   "Synthetic re-plucks add a new excitation to the unchanged old tail; they do not model string damping by the pick.",
                                   "Chord-only synthetic WAV/CSV datasets cannot supply exact pluck labels.",
                                   "Feature extraction/alignment and the training adapter are not part of this preparation."]}
        write_json(summary_path, summary)
        try:
            verify_source_files(sources)
            guitarset_path = output / "guitarset.json"
            write_json(guitarset_path, {"schema_version": 1, "sources": sources})
            synthetic_dir = output / "synthetic"
            synthetic_dir.mkdir()
            synthetic = {"schema_version": 1, "generator": GENERATOR_VERSION,
                         "purpose": "onset_experiment_sources", "seed": seed, "samplerate": sr, "clips": []}
            for split, count in zip(SPLITS, groups):
                totals = {"groups": count, "clips": 0, "events": 0, "challenge_events": 0,
                          "seconds": 0., "cases": {}}
                for group in range(count):
                    for clip in render_group(split, group, seed, sr):
                        audio = clip.pop("audio")
                        wav = synthetic_dir / (clip["name"] + ".wav")
                        reference = synthetic_dir / (clip["name"] + ".csv")
                        sf.write(wav, audio, sr, subtype="FLOAT")
                        with reference.open("w", newline="") as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(clip["events"][0]))
                            writer.writeheader()
                            writer.writerows(clip["events"])
                        clip.update(wav=wav.name, reference=reference.name,
                                    wav_sha256=sha256(wav), reference_sha256=sha256(reference))
                        synthetic["clips"].append(clip)
                        totals["clips"] += 1
                        totals["events"] += len(clip["events"])
                        totals["challenge_events"] += sum(e["role"] == "challenge" for e in clip["events"])
                        totals["seconds"] += clip["duration"]
                        totals["cases"][clip["case"]] = totals["cases"].get(clip["case"], 0) + 1
                    if (group + 1) % 12 == 0 or group + 1 == count:
                        print(f"Rendered {split}: {group + 1}/{count} source groups", flush=True)
                summary["synthetic"][split] = totals
                write_json(summary_path, summary)
            synthetic_path = synthetic_dir / "manifest.json"
            write_json(synthetic_path, synthetic)
            write_json(output / "dataset.json", {
                "schema_version": 1, "training_ready": False,
                "guitarset": {"path": "guitarset.json", "sha256": sha256(guitarset_path)},
                "synthetic": {"path": "synthetic/manifest.json", "sha256": sha256(synthetic_path)},
                "split_policy": "whole GuitarSet performers; whole synthetic excitation groups",
                "window_policy": "whole takes, including attacks at chord boundaries; no chord-segment crop",
                "target_time_axis": "physical audio seconds; no feature indices or label shift assigned"})
            summary["ok"] = True
            write_json(summary_path, summary)
        except Exception as error:
            summary["error"] = str(error)
            write_json(summary_path, summary)
            raise
        return summary

    def run_pipeline(root, output_root, variant="auto", groups=SYNTHETIC_GROUPS,
                     validation_player=VALIDATION_PLAYER, test_player=TEST_PLAYER,
                     seed=SEED, sr=SR):
        import tempfile
        if not root.is_dir():
            raise FileNotFoundError(f"GuitarSet input directory is unavailable: {root}")
        output_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="onset-prepared-", dir=output_root)).resolve()
        print(f"Output directory: {run_dir}", flush=True)
        print("Stage 1/2: auditing the attached GuitarSet", flush=True)
        audited = audit(root, variant)
        manifest = run_dir / f"onset_manifest_{audited['selected_variant'] or 'auto'}.json"
        write_json(manifest, audited)
        if not audited["ok"]:
            failed = {"ok": False, "stage": "audit", "inventory": audited["inventory"],
                      "error_count": len(audited["errors"]), "errors": audited["errors"][:10],
                      "full_report": str(manifest)}
            write_json(run_dir / "summary.json", failed)
            raise ValueError(f"Audit failed. Small report: {run_dir / 'summary.json'}. "
                             f"First errors: {audited['errors'][:3]}")
        print(f"Audited {len(audited['sources'])} sources. Manifest: {manifest}", flush=True)
        print("Stage 2/2: preparing source splits and synthetic pairs", flush=True)
        result = prepare(manifest, run_dir / "prepared", validation_player, test_player,
                         groups, seed, sr)
        result["prepared_directory"] = str(run_dir / "prepared")
        result["summary_path"] = str(run_dir / "summary.json")
        write_json(run_dir / "summary.json", result)
        return result

    import argparse

    import csv

    from dataclasses import asdict, dataclass

    import hashlib

    import json

    import math

    from pathlib import Path

    import re

    @dataclass(frozen=True)
    class Event:
        id: str
        t: float
        pc: int
        midi: int | None = None
        case: str = ""

    def match_events(reference, predicted, early, late, pitch="pc"):
        """Maximize matches, then minimize total absolute timing error.

        Per pitch, an ordered dynamic program suffices: with the same tolerance
        window for every event, uncrossing two feasible matches preserves
        feasibility and cannot increase absolute timing error. Keep indices so
        even simultaneous same-class events cannot consume a prediction twice.
        """
        pairs = []
        for key in sorted({getattr(e, pitch) for e in reference + predicted}):
            refs = sorted((i for i, e in enumerate(reference) if getattr(e, pitch) == key),
                          key=lambda i: reference[i].t)
            preds = sorted((i for i, e in enumerate(predicted) if getattr(e, pitch) == key),
                           key=lambda i: predicted[i].t)
            n, m = len(refs), len(preds)
            scores = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
            action = [[None] * (m + 1) for _ in range(n + 1)]
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    best, move = scores[i - 1][j], "ref"
                    if scores[i][j - 1] > best:
                        best, move = scores[i][j - 1], "pred"
                    delta = predicted[preds[j - 1]].t - reference[refs[i - 1]].t
                    if -early - 1e-9 <= delta <= late + 1e-9:
                        count, cost = scores[i - 1][j - 1]
                        candidate = (count + 1, cost - abs(delta))
                        if candidate > best:
                            best, move = candidate, "match"
                    scores[i][j], action[i][j] = best, move
            i, j = n, m
            while i and j:
                move = action[i][j]
                if move == "match":
                    pairs.append((refs[i - 1], preds[j - 1]))
                    i, j = i - 1, j - 1
                elif move == "ref":
                    i -= 1
                else:
                    j -= 1
        return sorted(pairs)

    def percentile(values, q):
        if not values:
            return None
        values = sorted(values)
        pos = (len(values) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    def latch_events(rows, threshold=.6, fill_min=50, stride=1, phase=0):
        """Isolated set_onsets latch. No app reset/credit semantics inferred."""
        if not 0 < threshold <= 1 or not 0 <= fill_min <= 100 or stride < 1 or not 0 <= phase < stride:
            raise ValueError("Invalid latch settings")
        armed, peaks, events = [True] * 12, [0.0] * 12, []
        for frame, (t, fill, values) in enumerate(rows):
            if frame % stride != phase or fill < fill_min:
                continue
            for pc, value in enumerate(values):
                if value < max(.3 * peaks[pc], .1):
                    armed[pc] = True
                elif armed[pc] and value >= threshold:
                    armed[pc], peaks[pc] = False, value
                    events.append(Event(f"{frame}:{pc}", t, pc))
        return events

    from bisect import bisect_left, bisect_right

    import numpy as np

    RINGING_WINDOW = .096

    RINGING_GUARD = .032

    RINGING_EVAL_WINDOW = .128

    RINGING_NEGATIVE_WEIGHT = 4.

    RINGING_SPEC = {
        "version": "held-pc-negative-v1", "negative_weight": RINGING_NEGATIVE_WEIGHT,
        "training_window_seconds": RINGING_WINDOW, "evaluation_window_seconds": RINGING_EVAL_WINDOW,
        "same_pc_exclusion": "onset in [attack-96ms, attack+128ms]; positives always protected",
        "activity": "GuitarSet note end; synthetic excitation support until file end, not audibility",
        "opportunity": "unique other-attack time and previously active pitch class, older than 96ms",
        "ambiguity": "exclude conflicting/missing GuitarSet string activity from extra loss only",
        "selection": "validation only: half held-PC errors, no increase at other times, preserved re-plucks, comp recall -2pp, P95 +16ms",
        "empty_control": "zero control held-PC errors is inconclusive, never an accepted improvement",
    }

    def ringing_annotations(source):
        events = sorted(source["events"], key=lambda e: (e["t"], e["id"]))
        event_times = [e["t"] for e in events]
        by_pc = [[e for e in events if e["pc"] == pc] for pc in range(12)]
        times = [[e["t"] for e in group] for group in by_pc]
        synthetic = source.get("domain") == "synthetic"
        active, opportunities = [], []
        audit = {"eligible": 0, "excluded_same_pc_attack": 0,
                 "excluded_string_conflict": 0, "missing_activity_end": 0}
        for event in events:
            if not synthetic and "end" not in event:
                audit["missing_activity_end"] += 1
        for attack in sorted({e["t"] for e in events}):
            active = [e for e in active if e.get("end", source["duration"] if synthetic else e["t"]) > attack]
            # Includes all strings at this instant. Short strums are protected below.
            new = events[bisect_left(event_times, attack):bisect_right(event_times, attack)]
            old_pcs = {e["pc"] for e in active if e["t"] + RINGING_WINDOW < attack - 1e-9}
            for pc in sorted(old_pcs):
                # Exclude the entire opportunity near a real same-PC start, including
                # an upcoming strum member. All ordinary targets/scoring remain intact.
                nearby = times[pc][bisect_left(times[pc], attack - RINGING_WINDOW - 1e-9):
                                   bisect_right(times[pc], attack + RINGING_WINDOW + RINGING_GUARD + 1e-9)]
                if nearby:
                    audit["excluded_same_pc_attack"] += 1
                    continue
                held = [e for e in active if e["pc"] == pc]
                if not synthetic:
                    # Conflicting pitches on one annotated string do not prove sustain.
                    held = [e for e in held if "string" in e and not any(
                        o.get("string") == e["string"] and o["pc"] != pc
                        for o in active + new)]
                if not held:
                    audit["excluded_string_conflict"] += 1
                    continue
                ends = []
                for old in held:
                    end = old.get("end", source["duration"])
                    if not synthetic:
                        # Another pitch later in this window can terminate this string
                        # before its overlapping annotation ends. Do not weight that gap.
                        end = min([end] + [e["t"] for e in events if attack < e["t"] < end
                                           and e.get("string") == old["string"] and e["pc"] != pc])
                    ends.append(end)
                support_end = max(ends)
                end = min(attack + RINGING_WINDOW, support_end)
                opportunities.append({"t": attack, "end": end, "pc": pc,
                                      "evaluation_end": min(attack + RINGING_EVAL_WINDOW, support_end),
                                      "old_ids": [e["id"] for e in held]})
                audit["eligible"] += 1
            active.extend(new)
        return opportunities, audit

    def ringing_mask(source, frames, targets, hop_seconds=.016):
        opportunities, audit = ringing_annotations(source)
        times = (np.arange(frames) + 1) * hop_seconds
        mask = np.zeros((frames, 12), dtype=bool)
        for opportunity in opportunities:
            mask[(times > opportunity["t"] + 1e-9) &
                 (times <= opportunity["end"] + 1e-9), opportunity["pc"]] = True
        # A positive on ANY string/octave always wins over the extra negative weight.
        mask &= targets == 0
        return mask, {**audit, "weighted_frames_classes": int(mask.sum())}

    def ringing_event_counts(source, predicted, pairs):
        opportunities, audit = source.get("ringing_annotations", (None, None))
        if opportunities is None:
            opportunities, audit = ringing_annotations(source)
        matched = {p.id for _, p in pairs}
        detection_by_ref = {r.id: p.t for r, p in pairs}
        counts = {"ringing_opportunities": len(opportunities), "ringing_false_events": 0,
                  "ringing_repeat_events": 0, "ringing_late_first_events": 0,
                  "ringing_affected_opportunities": 0, "foreign_pc_events": 0,
                  "held_pc_false_events_any_time": 0}
        affected = set()
        for prediction in predicted:
            if prediction.id in matched:
                continue
            held = [e for e in source["events"] if e["pc"] == prediction.pc
                    and e["t"] + RINGING_WINDOW < prediction.t
                    and prediction.t < e.get("end", source["duration"] if source.get("domain") == "synthetic" else e["t"])]
            counts["held_pc_false_events_any_time"] += bool(held)
            hits = [(i, o) for i, o in enumerate(opportunities)
                    if o["pc"] == prediction.pc and o["t"] < prediction.t <= o["evaluation_end"] + 1e-9]
            if hits:
                counts["ringing_false_events"] += 1
                repeated = any(detection_by_ref.get(eid, float("inf")) < prediction.t
                               for _, o in hits for eid in o["old_ids"])
                counts["ringing_repeat_events" if repeated else "ringing_late_first_events"] += 1
                affected.update(i for i, _ in hits)
            elif not any(e["pc"] == prediction.pc and e["t"] <= prediction.t for e in source["events"]):
                counts["foreign_pc_events"] += 1
        counts["ringing_affected_opportunities"] = len(affected)
        return counts, audit

    def ringing_acceptance(candidate, control):
        """Predeclared validation constraints; an inconclusive control cannot pass."""
        c, b = candidate["groups"], control["groups"]
        required = ("all", "synthetic", "synthetic/root_repluck", "synthetic/triad_repluck", "guitarset/comp")
        if any(key not in c or key not in b for key in required):
            return {"accepted": False, "checks": {"required_groups_present": False}}
        old = b["all"]["ringing_false_events"]
        checks = {
            "measurable_control_errors": old > 0,
            "half_as_many_held_pc_errors": c["all"]["ringing_false_events"] <= old / 2,
            "same_opportunity_count": c["all"]["ringing_opportunities"] == b["all"]["ringing_opportunities"],
            "no_increase_in_all_held_pc_errors": c["all"]["held_pc_false_events_any_time"] <= b["all"]["held_pc_false_events_any_time"],
            "root_repluck_preserved": c["synthetic/root_repluck"]["challenge_tp"] >= b["synthetic/root_repluck"]["challenge_tp"],
            "triad_repluck_preserved": c["synthetic/triad_repluck"]["challenge_tp"] >= b["synthetic/triad_repluck"]["challenge_tp"],
            "comp_recall_preserved": c["guitarset/comp"]["recall"] >= b["guitarset/comp"]["recall"] - .02 - 1e-9,
        }
        for domain in ("all", "synthetic", "guitarset/comp"):
            before, after = b[domain]["latency_p95"], c[domain]["latency_p95"]
            checks[f"{domain}_latency_preserved"] = (before is not None and after is not None
                                                    and after <= before + .016 + 1e-9)
        return {"accepted": all(checks.values()), "checks": checks}

    from collections import Counter

    import math

    import numpy as np

    PAIR_WEIGHT = .1

    PAIR_MARGIN = 1.

    PAIR_SPEC = {
        "version": "same-background-ranking-v1", "weight": PAIR_WEIGHT,
        "margin_logits": PAIR_MARGIN,
        "loss": "mean softplus(1 + max(negative logits) - max(positive logits))",
        "window": "existing 96ms positive target, same absolute frames in both clips",
        "positives": "verified new excitation of an already sounding pitch class, including octave",
        "negatives": "identical initial stems and gain, no onset of this PC near the target window",
        "augmentation": "same gain in both clips; independent RNG from ordinary training batches",
        "sampling": "each eligible train pair once per epoch, spread across existing optimizer steps",
        "control": "ordinary BCE, positive weight 4, no extra held-PC negative weight",
        "selection": "unchanged validation event criteria; pair ranking is diagnostic only",
    }

    def pair_context(source):
        fields = ("source_id", "t", "midi", "pc", "level", "excitation_seed")
        context = [e for e in source["events"] if e.get("role") == "context"]
        if not context or any(not e.get("pluck_verified") for e in source["events"]):
            raise ValueError("Pairs require verified synthetic excitations and an initial context")
        return tuple(sorted(tuple(e[k] for k in fields) for e in context))

    def build_onset_pairs(sources, hop_seconds=.016, target_seconds=.096):
        """Return references to clips, never copies assigned to a different split."""
        groups = {}
        for index, source in enumerate(sources):
            if source["domain"] == "synthetic":
                groups.setdefault(source["group"], []).append((index, source))
        pairs, exclusions = [], Counter()
        for group, members in sorted(groups.items()):
            if len({s["split"] for _, s in members}) != 1:
                raise ValueError(f"Pair group crosses data splits: {group}")
            if len({s["id"] for _, s in members}) != len(members):
                raise ValueError(f"Duplicate clip in pair group: {group}")
            contexts = {i: pair_context(s) for i, s in members}
            for positive_index, positive in sorted(members, key=lambda item: item[1]["id"]):
                old_pcs = {e["pc"] for e in positive["events"] if e["role"] == "context"}
                for event in positive["events"]:
                    if event["role"] != "challenge" or event["pc"] not in old_pcs:
                        continue
                    times = (np.arange(positive["frames"]) + 1) * hop_seconds
                    target = np.flatnonzero((times > event["t"] + 1e-9) &
                                           (times <= event["t"] + target_seconds + 1e-9))
                    if len(target) != round(target_seconds / hop_seconds):
                        raise ValueError("Truncated pair target window")
                    for negative_index, negative in sorted(members, key=lambda item: item[1]["id"]):
                        if contexts[positive_index] != contexts[negative_index]:
                            continue
                        if positive["pair_gain"] != negative["pair_gain"] or positive["frames"] != negative["frames"]:
                            raise ValueError("Same-context pair has different gain or length")
                        if any(e["pc"] == event["pc"] and
                               event["t"] - target_seconds - 1e-9 <= e["t"] <= event["t"] + target_seconds + .032 + 1e-9
                               for e in negative["events"]):
                            exclusions["negative_has_real_same_pc_attack"] += 1
                            continue
                        first_change = min(e["t"] for s in (positive, negative)
                                           for e in s["events"] if e["role"] == "challenge")
                        pairs.append({"id": f"{event['id']}|{negative['id']}", "group": group,
                                      "split": positive["split"], "positive": positive_index,
                                      "negative": negative_index, "pc": event["pc"],
                                      "start": int(target[0]), "length": len(target),
                                      "shared_prefix_frames": math.floor(first_change / hop_seconds + 1e-9),
                                      "positive_source": positive["id"], "negative_source": negative["id"],
                                      "positive_event": event["id"], "t": event["t"],
                                      "kind": positive["case"] + "/" + negative["case"]})
        audit = {"pairs": len(pairs), "groups": len({p["group"] for p in pairs}),
                 "kinds": dict(Counter(p["kind"] for p in pairs)), "exclusions": dict(exclusions)}
        return pairs, audit

    class OnsetPairBatches:
        def __init__(self, sources, block_function, hop_seconds=.016, target_seconds=.096):
            if not sources or any(s["split"] != "train" for s in sources):
                raise ValueError("Pair training must contain train sources only")
            self.pairs, self.audit = build_onset_pairs(sources, hop_seconds, target_seconds)
            if not self.pairs:
                raise ValueError("No valid synthetic training pairs")
            indices = {p[key] for p in self.pairs for key in ("positive", "negative")}
            self.features = {i: np.load(sources[i]["features"], mmap_mode="r") for i in indices}
            self.block_function = block_function
            checked = set()
            for pair in self.pairs:
                key = (pair["positive"], pair["negative"], pair["shared_prefix_frames"])
                if key in checked:
                    continue
                a, b = (self.features[pair[k]] for k in ("positive", "negative"))
                n = pair["shared_prefix_frames"]
                if a.shape != b.shape or not np.array_equal(a[:n], b[:n]):
                    raise ValueError(f"Pair audio features differ before the challenge: {pair['id']}")
                checked.add(key)
            self.audit["verified_shared_prefixes"] = len(checked)

        def epoch(self, epoch, steps, seed):
            # This generator does not consume np.random used by OnsetBlocks.
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20260925]))
            order = rng.permutation(len(self.pairs))
            for step in range(steps):
                chosen = order[len(order) * step // steps:len(order) * (step + 1) // steps]
                if not len(chosen):
                    yield None
                    continue
                positive, negative, pcs, ids = [], [], [], []
                for index in chosen:
                    pair = self.pairs[int(index)]
                    gain = 10 ** rng.uniform(-.3, .3)
                    for key, destination in (("positive", positive), ("negative", negative)):
                        x, count = self.block_function(self.features[pair[key]], pair["start"], pair["length"])
                        if count != pair["length"]:
                            raise ValueError("Pair crosses the recording end")
                        x = np.log1p(np.expm1(x * math.log(1001)) * gain) / math.log(1001)
                        destination.append(x.astype(np.float32))
                    pcs.append(pair["pc"])
                    ids.append(pair["id"])
                yield np.stack(positive), np.stack(negative), np.asarray(pcs, dtype=np.int64), ids

    def onset_pair_loss(positive_logits, negative_logits, pcs, history_frames):
        """Both sides receive gradients; ordinary BCE still anchors absolute targets."""
        import torch
        rows = torch.arange(len(pcs), device=positive_logits.device)
        positive = positive_logits[rows, pcs, history_frames:].amax(dim=-1)
        negative = negative_logits[rows, pcs, history_frames:].amax(dim=-1)
        return torch.nn.functional.softplus(PAIR_MARGIN + negative - positive).mean()

    def onset_pair_metrics(predictions, hop_seconds=.016, target_seconds=.096):
        sources = [s for s, _ in predictions]
        pairs, audit = build_onset_pairs(sources, hop_seconds, target_seconds)
        differences = []
        for pair in pairs:
            window = slice(pair["start"], pair["start"] + pair["length"])
            p = float(predictions[pair["positive"]][1][window, pair["pc"]].max())
            n = float(predictions[pair["negative"]][1][window, pair["pc"]].max())
            differences.append(p - n)
        return {**audit, "positive_above_negative": sum(d > 0 for d in differences),
                "ties": sum(d == 0 for d in differences),
                "mean_probability_difference": float(np.mean(differences)) if differences else None}

    import math

    RISE_PAST_FRAMES = 4

    RISE_SPEC = {
        "version": "positive-spectral-rise-v1",
        "past_frames": RISE_PAST_FRAMES,
        "input": "existing 770 float32 log-magnitude features, after ordinary gain augmentation",
        "formula": "a=expm1(x*log(1001)); d=relu(a-mean(previous 4 a)); log1p(d)/log(1001)",
        "current_frame_excluded_from_baseline": True,
        "startup": "zero feature history",
        "future_frames": 0,
        "projection": "extra 770->96 linear projection, no bias, initialized to zero; added before first ReLU",
        "shared_weights": "identical initialization of all original parameters",
        "loss": "ordinary BCE, positive weight 4, no paired loss or extra negative weights",
        "raw_feature_cache": "unchanged; rise is computed inside the exported ONNX graph",
    }

    def positive_spectral_rise(features):
        import torch
        # Exp/Sub and Add/Log also work in the app's ONNX opset 17 exporter.
        amplitude = torch.exp(features * math.log(1001)) - 1
        # Exclude the current frame: a left pad of four produces T+1 averages,
        # and the final average belongs to the following (unavailable) frame.
        previous = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(amplitude, (RISE_PAST_FRAMES, 0)),
            kernel_size=RISE_PAST_FRAMES, stride=1)[:, :, :-1]
        return torch.log(1 + torch.relu(amplitude - previous)) / math.log(1001)

    import argparse

    import hashlib

    from bisect import bisect_left, bisect_right

    import importlib

    import json

    import math

    from pathlib import Path

    import random

    import sys

    import time

    import numpy as np

    import soundfile as sf

    TRAIN_EPOCHS = 12

    TRAIN_BATCH_SIZE = 16

    TRAIN_SEED = 20260923

    FEATURE_SPEC = {
        "version": "short-stft-v1", "samplerate": 16000, "hop": 256,
        "windows": [1024, 2048], "max_frequency": 4000,
        "window": "symmetric Hann", "amplitude": "2*abs(rfft)/sum(window)",
        "compression": "log1p(1000*amplitude)/log(1001); no file normalization",
        "resampling": "causal linear, one source-sample delay unless already 16kHz",
        "frame_time": "exclusive audio window end; first frame=0.016 seconds",
        "startup": "zero left audio padding; no right padding or future samples",
        "storage": "float16, converted back to float32 for both training and inference",
    }

    ONSET_HOP = 256

    ONSET_SR = 16000

    FEATURE_DIM = 770  # 0..4kHz for the 1024 and 2048 point transforms.

    HISTORY = 30  # Four kernel-3 convolutions, dilations 1,2,4,8.

    BLOCK_FRAMES = 128

    TARGET_SECONDS = .096

    EVAL_EARLY = .032

    EVAL_LATE = .128

    THRESHOLDS = (.3, .4, .5, .6, .7, .8, .9)

    def onset_resample(audio, source_sr):
        """Same time origin at every length; the interpolator never needs the future."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or source_sr < 8000:
            raise ValueError("Expected finite, nonempty mono audio with sample rate >=8000")
        if source_sr == ONSET_SR:
            return audio
        positions = np.arange(math.floor(len(audio) * ONSET_SR / source_sr)) * (source_sr / ONSET_SR) - 1
        lower = np.floor(positions).astype(np.int64)
        fraction = positions - lower
        # Zero history before the source starts. No centred antialiasing filter.
        a = np.where(lower >= 0, audio[np.clip(lower, 0, len(audio) - 1)], 0)
        b = np.where(lower + 1 >= 0, audio[np.clip(lower + 1, 0, len(audio) - 1)], 0)
        return (a + fraction * (b - a)).astype(np.float32)

    def onset_features(audio):
        """Frame n sees only samples strictly before (n+1)*hop, even at startup."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Expected finite mono audio")
        frames = len(audio) // ONSET_HOP
        result = np.empty((frames, FEATURE_DIM), dtype=np.float16)
        offset = 0
        for size in FEATURE_SPEC["windows"]:
            window = np.hanning(size)
            bins = size * FEATURE_SPEC["max_frequency"] // ONSET_SR + 1
            padded = np.pad(audio, (size - ONSET_HOP, 0))
            if frames:
                windows = np.lib.stride_tricks.sliding_window_view(padded, size)[::ONSET_HOP][:frames]
                for start in range(0, frames, 1024):
                    spectra = np.abs(np.fft.rfft(windows[start:start + 1024] * window, axis=1))[:, :bins]
                    amplitude = spectra * (2 / window.sum())
                    result[start:start + len(spectra), offset:offset + bins] = (
                        np.log1p(1000 * amplitude) / math.log(1001)).astype(np.float16)
            offset += bins
        return result

    def onset_targets(events, frames):
        times = (np.arange(frames) + 1) * ONSET_HOP / ONSET_SR
        labels = np.zeros((frames, 12), dtype=np.float32)
        for event in events:
            # A sample at exactly the exclusive frame end is not visible yet.
            active = (times > event["t"] + 1e-9) & (times <= event["t"] + TARGET_SECONDS + 1e-9)
            labels[active, event["pc"]] = 1
        return labels

    def onset_sources(prepared):
        """Use only this run's verified manifests; preserve raw events and groups."""
        dataset = json.loads((prepared / "dataset.json").read_text())
        documents = {}
        for name in ("guitarset", "synthetic"):
            path = prepared / dataset[name]["path"]
            if sha256(path) != dataset[name]["sha256"]:
                raise ValueError(f"Changed prepared manifest: {path}")
            documents[name] = json.loads(path.read_text())
        if documents["synthetic"]["generator"] != "onset-ks-v2":
            raise ValueError("Regenerate the tuned v2 synthetic sources")
        sources = []
        for source in documents["guitarset"]["sources"]:
            sources.append({"id": source["take"], "domain": "guitarset", "case": source["style"],
                            "split": source["split"], "wav": source["audio"]["path"],
                            "sha256": source["audio"]["sha256"], "duration": source["audio"]["duration"],
                            "events": source["events"], "group": source["split_group"]})
        for clip in documents["synthetic"]["clips"]:
            sources.append({"id": clip["name"], "domain": "synthetic", "case": clip["case"],
                            "split": clip["split"], "wav": str(prepared / "synthetic" / clip["wav"]),
                            "sha256": clip["wav_sha256"], "duration": clip["duration"],
                            "events": clip["events"], "group": clip["source_group"], "pair_gain": clip["gain"]})
        return sources

    def cache_onset_features(sources, directory):
        directory.mkdir()
        cached = []
        for index, source in enumerate(sources):
            wav = Path(source["wav"])
            if sha256(wav) != source["sha256"]:
                raise ValueError(f"Audio changed after preparation: {wav}")
            audio, sr = sf.read(wav, dtype="float32")
            features = onset_features(onset_resample(audio, sr))
            if not len(features):
                raise ValueError(f"Recording shorter than one frame: {wav}")
            path = directory / f"{index:04d}.npy"
            np.save(path, features)
            cached.append(dict(source, features=str(path), feature_sha256=sha256(path),
                               ringing_annotations=ringing_annotations(source),
                               frames=len(features), incomplete_final_hop_seconds=(
                                   source["duration"] - len(features) * ONSET_HOP / ONSET_SR)))
            if (index + 1) % 30 == 0 or index + 1 == len(sources):
                print(f"Features: {index + 1}/{len(sources)} complete recordings", flush=True)
        write_json(directory / "index.json", {"feature_spec": FEATURE_SPEC, "sources": cached})
        return cached

    def feature_block(features, start, length=BLOCK_FRAMES, history_frames=HISTORY):
        """Fixed left context; a training/inference boundary never resets the audio."""
        count = min(length, len(features) - start)
        x = np.zeros((history_frames + length, FEATURE_DIM), dtype=np.float32)
        low = max(0, start - history_frames)
        destination = history_frames + low - start
        x[destination:history_frames + count] = features[low:start + count]
        return x.T.copy(), count

    class OnsetBlocks:
        def __init__(self, sources, ringing_weight=1., history_frames=HISTORY):
            if not sources or any(s["split"] != "train" for s in sources):
                raise ValueError("Training blocks must contain train sources only")
            self.features = [np.load(s["features"], mmap_mode="r") for s in sources]
            self.labels = [onset_targets(s["events"], s["frames"]) for s in sources]
            self.ringing = [ringing_mask(s, s["frames"], y)[0] for s, y in zip(sources, self.labels)]
            self.ringing_weight = ringing_weight
            self.history_frames = history_frames
            self.blocks = [(i, start) for i, f in enumerate(self.features)
                           for start in range(0, len(f), BLOCK_FRAMES)]

        def __len__(self):
            return len(self.blocks)

        def __getitem__(self, index):
            recording, start = self.blocks[index]
            x, count = feature_block(self.features[recording], start, history_frames=self.history_frames)
            # Exact gain transform of the fixed log spectrum, including left context.
            gain = 10 ** np.random.uniform(-.3, .3)
            x = np.log1p(np.expm1(x * math.log(1001)) * gain) / math.log(1001)
            y = np.zeros((12, BLOCK_FRAMES), dtype=np.float32)
            y[:, :count] = self.labels[recording][start:start + count].T
            mask = np.zeros(BLOCK_FRAMES, dtype=np.float32)
            mask[:count] = 1
            weights = np.ones((12, BLOCK_FRAMES), dtype=np.float32)
            weights[:, :count] += (self.ringing_weight - 1) * self.ringing[recording][start:start + count].T
            return x.astype(np.float32), y, mask, weights

    def make_onset_model(spectral_rise=False):
        import torch
        from torch import nn

        class CausalOnset(nn.Module):
            def __init__(self):
                super().__init__()
                self.project = nn.Conv1d(FEATURE_DIM, 96, 1)
                self.temporal = nn.ModuleList([nn.Conv1d(96, 96, 3, dilation=d) for d in (1, 2, 4, 8)])
                self.output = nn.Conv1d(96, 12, 1)
                nn.init.constant_(self.output.bias, -3.)
                self.spectral_rise = spectral_rise
                if spectral_rise:
                    # Build AFTER all common parameters so the same seed gives the
                    # same initial backbone. Zero extension preserves its answer.
                    with torch.random.fork_rng(devices=[]):
                        self.rise_project = nn.Conv1d(FEATURE_DIM, 96, 1, bias=False)
                        nn.init.zeros_(self.rise_project.weight)

            def forward(self, features):
                x = self.project(features)
                if self.spectral_rise:
                    x = x + self.rise_project(positive_spectral_rise(features))
                x = torch.relu(x)
                for dilation, layer in zip((1, 2, 4, 8), self.temporal):
                    x = torch.relu(x + layer(nn.functional.pad(x, (2 * dilation, 0))))
                return self.output(x)

        return CausalOnset()

    def onset_predictions(sources, infer, batch_size=16, history_frames=HISTORY):
        """Full files, each frame once, with the same left context as training."""
        result = []
        for source in sources:
            features = np.load(source["features"], mmap_mode="r")
            probabilities = []
            starts = list(range(0, len(features), BLOCK_FRAMES))
            for offset in range(0, len(starts), batch_size):
                blocks = [feature_block(features, start, history_frames=history_frames) for start in starts[offset:offset + batch_size]]
                logits = infer(np.stack([x for x, _ in blocks]))
                if logits.shape != (len(blocks), 12, history_frames + BLOCK_FRAMES) or not np.isfinite(logits).all():
                    raise ValueError("Invalid model output")
                for row, (_, count) in zip(logits, blocks):
                    probabilities.append((1 / (1 + np.exp(-np.clip(row[:, history_frames:history_frames + count], -80, 80)))).T)
            values = np.concatenate(probabilities)
            if len(values) != source["frames"]:
                raise ValueError("Refusing partial-file evaluation")
            result.append((source, values))
        return result

    def local_event_matches(reference, predicted, early=EVAL_EARLY, late=EVAL_LATE):
        """Exact scorer on disjoint tolerance components, avoiding full-take DP tables."""
        pairs = []
        for pc in range(12):
            refs = sorted((e for e in reference if e.pc == pc), key=lambda e: e.t)
            preds = sorted((e for e in predicted if e.pc == pc), key=lambda e: e.t)
            pred_times = [e.t for e in preds]
            start = 0
            while start < len(refs):
                end = start + 1
                while end < len(refs) and refs[end].t - refs[end - 1].t <= early + late + 2e-9:
                    end += 1
                selected = preds[bisect_left(pred_times, refs[start].t - early - 1e-9):
                                 bisect_right(pred_times, refs[end - 1].t + late + 1e-9)]
                group = refs[start:end]
                pairs.extend((group[i], selected[j]) for i, j in match_events(group, selected, early, late))
                start = end
        return pairs

    def onset_metrics(predictions, threshold):
        buckets = {}
        group_sets = {}
        details = []
        for source, values in predictions:
            rows = (((i + 1) * ONSET_HOP / ONSET_SR, 100, row) for i, row in enumerate(values)
                    if (i + 1) * ONSET_HOP / ONSET_SR < source["duration"])
            predicted = latch_events(rows, threshold=threshold, fill_min=0)
            refs = [Event(e["id"], e["t"], e["pc"], e.get("midi"), e.get("case", source["case"]))
                    for e in source["events"] if 0 <= e["t"] < source["duration"]]
            pairs = local_event_matches(refs, predicted)
            found_r, found_p = {r.id for r, _ in pairs}, {p.id for _, p in pairs}
            extra = [p for p in predicted if p.id not in found_p]
            repeated = set()
            last = {}
            collisions = 0
            overlaps = 0
            target_frames = set()
            for ref in sorted(refs, key=lambda e: e.t):
                if ref.pc in last and 0 < ref.t - last[ref.pc] <= 2:
                    repeated.add(ref.id)
                if ref.pc in last and ref.t - last[ref.pc] < TARGET_SECONDS:
                    overlaps += 1
                last[ref.pc] = ref.t
                key = (math.floor(ref.t * ONSET_SR / ONSET_HOP + 1e-9), ref.pc)
                collisions += key in target_frames
                target_frames.add(key)
            challenges = {e["id"] for e in source["events"] if e.get("role") == "challenge"}
            deltas = [p.t - r.t for r, p in pairs]
            # Additional predictions >400ms after ALL annotated starts: conservative decay/silence diagnostic.
            tail_extra = sum(p.t > max((r.t for r in refs), default=0) + .4 for p in extra)
            counts = {"tp": len(pairs), "fp": len(extra), "fn": len(refs) - len(pairs),
                      "seconds": source["duration"], "recordings": 1, "tail_extra": tail_extra,
                      "repeated_reference": len(repeated), "repeated_tp": len(repeated & found_r),
                      "challenge_reference": len(challenges), "challenge_tp": len(challenges & found_r),
                      "same_pc_frame_collisions": collisions,
                      "same_pc_target_overlaps": overlaps,
                      "boundary_reference": sum(r.t < EVAL_EARLY or r.t + EVAL_LATE >= source["duration"] for r in refs)}
            ringing_counts, ringing_audit = ringing_event_counts(source, predicted, pairs)
            counts.update(ringing_counts)
            for key in ("all", source["domain"], f"{source['domain']}/{source['case']}"):
                bucket = buckets.setdefault(key, dict.fromkeys(counts, 0) | {"deltas": []})
                for name, count in counts.items():
                    bucket[name] += count
                bucket["deltas"].extend(deltas)
                sets = group_sets.setdefault(key, {"opportunity_source_groups": set(), "error_source_groups": set()})
                group = f"{source['domain']}:{source.get('group', source.get('source_group', source['id']))}"
                if counts["ringing_opportunities"]:
                    sets["opportunity_source_groups"].add(group)
                if counts["ringing_false_events"]:
                    sets["error_source_groups"].add(group)
            details.append({"source": source["id"], **counts,
                            "ringing_annotation_audit": ringing_audit,
                            "predicted": [{"id": p.id, "t": p.t, "pc": p.pc} for p in predicted],
                            "missed_ids": [r.id for r in refs if r.id not in found_r],
                            "extra_ids": [p.id for p in extra]})
        for key, bucket in buckets.items():
            bucket.update({name: sorted(values) for name, values in group_sets[key].items()})
            tp, fp, fn = (bucket[name] for name in ("tp", "fp", "fn"))
            bucket.update(precision=tp / (tp + fp) if tp + fp else 0.,
                          recall=tp / (tp + fn) if tp + fn else 0.,
                          f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                          false_events_per_minute=60 * fp / bucket["seconds"],
                          latency_p50=percentile(bucket["deltas"], .5),
                          latency_p95=percentile(bucket.pop("deltas"), .95))
        macro = float(np.mean([buckets[d]["f1"] for d in ("guitarset", "synthetic") if d in buckets]))
        return {"threshold": threshold, "macro_domain_f1": macro, "groups": buckets}, details

    def validation_choice(table):
        # F1 first, fewer false events second, higher threshold as the final tie breaker.
        return max(table, key=lambda r: (r["macro_domain_f1"], -r["groups"]["all"]["fp"], r["threshold"]))

    def training_dependencies():
        """Kaggle already supplies Torch; install only missing export packages there."""
        import torch
        for package in ("onnx", "onnxruntime"):
            try:
                importlib.import_module(package)
            except ImportError:
                if not Path("/kaggle/working").is_dir():
                    raise RuntimeError(f"Install {package} in the test environment before training")
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", package], check=True)
                importlib.invalidate_caches()
                importlib.import_module(package)
        return torch

    def train_onset_experiment(sources, output, epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH_SIZE,
                               seed=TRAIN_SEED, device_name="auto", ringing_weight=1.,
                               feature_directory=None, control_validation=None, pair_weight=0., spectral_rise=False,
                               initial_checkpoint=None, resume=False, checkpoint_callback=None):
        torch = training_dependencies()
        if epochs < 1 or batch_size < 1:
            raise ValueError("Positive epochs and batch size required")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
        splits = {split: [s for s in sources if s["split"] == split] for split in ("train", "validation", "test")}
        if any(not split for split in splits.values()):
            raise ValueError("All three splits must be nonempty")
        if ringing_weight not in (1., RINGING_NEGATIVE_WEIGHT):
            raise ValueError("This controlled experiment supports only negative weights 1 and 4")
        feature_directory = feature_directory or output / "features"
        if spectral_rise and (pair_weight or ringing_weight != 1.):
            raise ValueError("The spectral-rise experiment must use ordinary BCE only")
        history_frames = HISTORY + (RISE_PAST_FRAMES if spectral_rise else 0)
        blocks = OnsetBlocks(splits["train"], ringing_weight, history_frames)
        pair_batches = (OnsetPairBatches(splits["train"], feature_block, ONSET_HOP / ONSET_SR, TARGET_SECONDS)
                        if pair_weight else None)
        if pair_weight not in (0., PAIR_WEIGHT):
            raise ValueError("The declared pair experiment uses weight 0 or 0.1")
        loader = torch.utils.data.DataLoader(blocks, batch_size=batch_size, shuffle=True, num_workers=0,
                                             generator=torch.Generator().manual_seed(seed))
        model = make_onset_model(spectral_rise).to(device)
        if initial_checkpoint is not None:
            saved = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
            prior = saved["contract"]
            if (prior["feature_spec"] != FEATURE_SPEC or
                    prior["history_frames"] != history_frames or
                    prior.get("spectral_rise", False) != spectral_rise):
                raise ValueError("Initial onset checkpoint has a different model/DSP contract")
            model.load_state_dict(saved["state_dict"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        checkpoint = output / "short_onset_best.pt"
        contract = {"feature_spec": FEATURE_SPEC, "features": FEATURE_DIM, "history_frames": history_frames,
                    "block_frames": BLOCK_FRAMES, "target_seconds": TARGET_SECONDS,
                    "network_startup": f"{history_frames} zero input frames before the first audio feature",
                    "spectral_rise": spectral_rise, "rise_spec": RISE_SPEC if spectral_rise else None,
                    "evaluation": {"early": EVAL_EARLY, "late": EVAL_LATE, "thresholds": list(THRESHOLDS),
                                   "reference_policy": "all raw note starts; no same-PC deduplication",
                                   "latch": "existing app peak hysteresis, fill gate disabled; no judge simulation"},
                    "architecture": "770->96 raw projection + optional rise projection; four causal residual conv3 dilations1/2/4/8 ->12 logits",
                    "seed": seed, "epochs": epochs, "batch_size": batch_size, "device": str(device),
                    "initial_checkpoint_sha256": sha256(initial_checkpoint) if initial_checkpoint else None,
                    "positive_weight": 4, "training_gain_db": [-6, 6],
                    "ringing_negative_weight": ringing_weight,
                    "ringing_spec": RINGING_SPEC,
                    "pair_spec": PAIR_SPEC if pair_weight else None, "pair_weight": pair_weight,
                    "training_pair_audit": pair_batches.audit if pair_batches else None,
                    "shared_initial_weights_sha256": hashlib.sha256(b"".join(
                        p.detach().cpu().numpy().tobytes() for name, p in model.state_dict().items()
                        if not name.startswith("rise_project."))).hexdigest(),
                    "initial_weights_sha256": hashlib.sha256(b"".join(
                        p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest(),
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "versions": {n: str(importlib.import_module(n).__version__) for n in
                                 ("torch", "numpy", "soundfile", "onnx", "onnxruntime")},
                    "data_index_sha256": sha256(feature_directory / "index.json")}
        last_path = output / "short_onset_last.pt"
        restored = None
        if resume and last_path.exists():
            restored = torch.load(last_path, map_location="cpu", weights_only=False)
            previous = restored["contract"]
            # Cache paths and package versions may differ after moving a Kaggle output.
            keys = ("feature_spec", "history_frames", "spectral_rise", "epochs",
                    "batch_size", "seed", "pair_weight", "ringing_negative_weight")
            if any(previous[k] != contract[k] for k in keys):
                raise ValueError("Cannot resume onset training with changed configuration")
            identities = [(v["id"], v["split"], v["sha256"], v["feature_sha256"], v["events"])
                          for v in sources]
            if restored["sources"] != identities:
                raise ValueError("Cannot resume onset training with changed data")
            contract = previous
            model.load_state_dict(restored["state_dict"], strict=True)
            optimizer.load_state_dict(restored["optimizer"])
            torch.save(restored["best_checkpoint"], checkpoint)
        write_json(output / "contract.json", contract)
        print(f"Training {contract['parameters']} parameters on {device}; {len(blocks)} blocks/epoch", flush=True)

        def infer(x):
            with torch.inference_mode():
                return model(torch.from_numpy(x).to(device)).cpu().numpy()

        history, best = [], None
        start_epoch = 1
        if restored is not None:
            history, best = restored["history"], restored["best"]
            start_epoch = restored["epoch"] + 1
            random.setstate(restored["python_rng"])
            np.random.set_state(restored["numpy_rng"])
            torch.set_rng_state(restored["torch_rng"])
            loader.generator.set_state(restored["loader_rng"])
            if device.type == "cuda" and restored["cuda_rng"]:
                torch.cuda.set_rng_state_all(restored["cuda_rng"])
            write_json(output / "history.json", history)
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            loss_total, examples = 0., 0
            started = time.monotonic()
            batch_digest = hashlib.sha256()
            pair_digest = hashlib.sha256()
            paired = pair_batches.epoch(epoch, len(loader), seed) if pair_batches else None
            input_digest = hashlib.sha256()
            pair_total, pair_count = 0., 0
            for batch_index, (x, y, mask, weights) in enumerate(loader):
                # Evidence that order and gain augmentation are identical across arms.
                input_digest.update(x.numpy().tobytes())
                # Candidate needs four extra OLD frames, not future samples. Compare
                # common raw frames/labels/gain to the original control batches.
                common_x = x[:, :, history_frames - HISTORY:]
                for tensor in (common_x, y, mask):
                    batch_digest.update(tensor.numpy().tobytes())
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)[:, :, history_frames:]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=torch.full((12, 1), 4., device=device), reduction="none")
                loss = (loss * weights.to(device) * mask[:, None, :]).sum() / (12 * mask.sum())
                pair_batch = next(paired) if paired else None
                if pair_batch is not None:
                    pos, neg, pcs, pair_ids = pair_batch
                    for array in (pos, neg, pcs):
                        pair_digest.update(array.tobytes())
                    pair_digest.update(json.dumps(pair_ids).encode())
                    pair_count += len(pcs)
                    if pair_weight:
                        pair_logits = model(torch.from_numpy(np.concatenate((pos, neg))).to(device))
                        rank_loss = onset_pair_loss(pair_logits[:len(pcs)], pair_logits[len(pcs):],
                                                    torch.from_numpy(pcs).to(device), HISTORY)
                        loss = loss + pair_weight * rank_loss
                        pair_total += float(rank_loss.detach()) * len(pcs)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                loss_total += float(loss.detach()) * float(mask.sum())
                examples += float(mask.sum())
                if (batch_index + 1) % max(1, len(loader) // 4) == 0:
                    print(f"Epoch {epoch}: batch {batch_index + 1}/{len(loader)}", flush=True)
            if pair_batches and pair_count != len(pair_batches.pairs):
                raise ValueError("Not every training pair was used exactly once")
            model.eval()
            predictions = onset_predictions(splits["validation"], infer, batch_size, history_frames)
            # Epoch selection fixed at .5. Only the best checkpoint gets the final threshold sweep.
            metrics, _ = onset_metrics(predictions, .5)
            choice = (metrics["macro_domain_f1"], -metrics["groups"]["all"]["fp"])
            if best is None or choice > best:
                best = choice
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "contract": contract}, checkpoint)
            history.append({"epoch": epoch, "loss": loss_total / examples,
                            "training_batches_sha256": batch_digest.hexdigest(),
                            "input_batches_sha256": input_digest.hexdigest(),
                            "pair_batches_sha256": pair_digest.hexdigest(), "pair_count": pair_count,
                            "pair_loss": pair_total / pair_count if pair_weight else None,
                            "seconds": time.monotonic() - started, "validation": metrics})
            write_json(output / "history.json", history)
            if resume:
                # A single atomic file includes BEST too: remote recovery never mixes epochs.
                state = dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                             epoch=epoch, contract=contract, history=history, best=best,
                             best_checkpoint=torch.load(checkpoint, map_location="cpu", weights_only=True),
                             sources=[(v["id"], v["split"], v["sha256"], v["feature_sha256"], v["events"])
                                      for v in sources],
                             python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                             torch_rng=torch.get_rng_state(), loader_rng=loader.generator.get_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
                temporary = last_path.with_suffix(".tmp")
                torch.save(state, temporary)
                temporary.replace(last_path)
                if checkpoint_callback:
                    checkpoint_callback(last_path)
            total = metrics["groups"]["all"]
            print(f"Epoch {epoch}/{epochs}: loss={history[-1]['loss']:.5f}, "
                  f"validation macro F1={metrics['macro_domain_f1']:.3f}, "
                  f"FP/min={total['false_events_per_minute']:.2f}, {history[-1]['seconds']:.1f}s", flush=True)

        return finish_onset_experiment(sources, output, control_validation, device_name)

    def export_probability_check(reference, exported, tolerance=2e-5):
        """Keep the numerical guard strict; discrete events can differ at a boundary."""
        if not reference or len(reference) != len(exported):
            raise ValueError("Export comparison requires the same complete recordings")
        maximum, worst_source, frames = 0., None, 0
        for (source, a), (other, b) in zip(reference, exported):
            if (source["id"] != other["id"] or a.shape != b.shape or a.ndim != 2
                    or a.shape != (source["frames"], 12) or not len(a)):
                raise ValueError("Export comparison source/frame mismatch")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Non-finite export comparison")
            error = float(np.max(np.abs(a - b)))
            if error > maximum:
                maximum, worst_source = error, source["id"]
            frames += len(a)
        return {"ok": maximum <= tolerance, "max_probability_error": maximum,
                "tolerance": tolerance, "worst_source": worst_source,
                "recordings": len(reference), "frames": frames}

    def export_event_comparison(reference, exported):
        """Report actual event changes without rounding probabilities or forgiving notes."""
        if len(reference) != len(exported) or any(a["source"] != b["source"] for a, b in zip(reference, exported)):
            raise ValueError("Export event comparison source mismatch")
        changed, only_reference, only_exported, examples = 0, 0, 0, []
        for a, b in zip(reference, exported):
            before = {(p["t"], p["pc"]) for p in a["predicted"]}
            after = {(p["t"], p["pc"]) for p in b["predicted"]}
            if before == after:
                continue
            changed += 1
            only_reference += len(before - after)
            only_exported += len(after - before)
            if len(examples) < 20:
                examples.append({"source": a["source"],
                                 "pytorch_only": sorted(before - after)[:20],
                                 "onnx_only": sorted(after - before)[:20]})
        return {"events_identical": changed == 0, "changed_recordings": changed,
                "pytorch_only_events": only_reference, "onnx_only_events": only_exported,
                "examples": examples}

    def finish_onset_experiment(sources, output, control_validation=None, device_name="auto"):
        """Evaluate/export a completed training run without another optimizer step."""
        torch = training_dependencies()
        torch.set_num_threads(min(4, torch.get_num_threads()))
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
        reference_device = str(device)
        checkpoint = output / "short_onset_best.pt"
        contract = json.loads((output / "contract.json").read_text())
        history = json.loads((output / "history.json").read_text())
        if [row["epoch"] for row in history] != list(range(1, contract["epochs"] + 1)):
            raise ValueError("Training is incomplete; refusing to silently restart it")
        spectral_rise = contract.get("spectral_rise", False)
        history_frames = contract["history_frames"]
        batch_size = contract["batch_size"]
        pair_weight = contract["pair_weight"]
        ringing_weight = contract["ringing_negative_weight"]
        splits = {split: [s for s in sources if s["split"] == split] for split in ("validation", "test")}
        model = make_onset_model(spectral_rise).to(device)

        def infer(x):
            with torch.inference_mode():
                return model(torch.from_numpy(x).to(device)).cpu().numpy()

        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved["contract"] != contract:
            raise ValueError("Checkpoint and contract disagree")
        best_row = max(history, key=lambda row: (row["validation"]["macro_domain_f1"],
                                                -row["validation"]["groups"]["all"]["fp"]))
        if saved["epoch"] != best_row["epoch"]:
            raise ValueError("Checkpoint is not the validation-selected epoch")
        model.load_state_dict(saved["state_dict"])
        model.eval()
        predictions = onset_predictions(splits["validation"], infer, batch_size, history_frames)
        # Export the selected epoch; choose the final threshold using the deployed backend.
        model.cpu()
        device = torch.device("cpu")
        example, _ = feature_block(np.load(splits["validation"][0]["features"], mmap_mode="r"), 0, history_frames=history_frames)
        model_label = "rise" if spectral_rise else ("paired" if pair_weight else ("weighted" if ringing_weight != 1. else "control"))
        onnx_path = output / f"short_onset_{model_label}.onnx"
        torch.onnx.export(model, torch.from_numpy(example[None]), str(onnx_path),
                          input_names=["short_features"], output_names=["onset_logits"],
                          dynamic_axes={"short_features": {0: "batch", 2: "time"},
                                        "onset_logits": {0: "batch", 2: "time"}},
                          opset_version=17, dynamo=False)
        import onnx
        import onnxruntime as ort
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])

        def infer_onnx(x):
            return session.run(["onset_logits"], {"short_features": x})[0]

        # Check every validation frame, not just one dummy tensor.
        exported = onset_predictions(splits["validation"], infer_onnx, batch_size, history_frames)
        parity = export_probability_check(predictions, exported)
        write_json(output / "onnx_validation_comparison.json", parity)
        if not parity["ok"]:
            raise ValueError(f"ONNX differs from checkpoint beyond tolerance: {parity['max_probability_error']}; "
                             "see onnx_validation_comparison.json. Training checkpoint is preserved.")
        max_error = parity["max_probability_error"]
        table = [onset_metrics(exported, threshold)[0] for threshold in THRESHOLDS]
        pair_validation = onset_pair_metrics(exported)
        selected = validation_choice(table)
        acceptance = None
        if control_validation is not None:
            qualifying = [row for row in table if ringing_acceptance(row, control_validation)["accepted"]]
            if qualifying:
                selected = min(qualifying, key=lambda row: (row["groups"]["all"]["ringing_false_events"],
                                                           -row["macro_domain_f1"], -row["threshold"]))
            acceptance = {"accepted": bool(qualifying),
                          "selection": "constraints_then_held_pc_errors_then_f1" if qualifying else "diagnostic_f1_fallback",
                          "threshold_checks": [{"threshold": row["threshold"], **ringing_acceptance(row, control_validation)}
                                               for row in table]}
        write_json(output / "validation_thresholds.json", {"checkpoint_epoch": saved["epoch"], "table": table,
                                                           "acceptance": acceptance,
                                                           "selected_threshold": selected["threshold"]})
        exported_metrics, exported_details = onset_metrics(exported, selected["threshold"])
        reference_metrics, reference_details = onset_metrics(predictions, selected["threshold"])
        parity.update(export_event_comparison(reference_details, exported_details))
        parity.update(threshold=selected["threshold"], reference_device=reference_device,
                      threshold_selection_backend="ONNX Runtime CPU",
                      pytorch_metrics=reference_metrics, onnx_metrics=exported_metrics)
        write_json(output / "onnx_validation_comparison.json", parity)
        if not parity["events_identical"]:
            print(f"Export numerical boundary differences: {parity['changed_recordings']} recordings, "
                  f"max probability error {max_error:.3g}; final scores/threshold use ONNX Runtime.", flush=True)
        write_json(output / "validation_events.json", exported_details)
        test_predictions = onset_predictions(splits["test"], infer_onnx, batch_size, history_frames)
        tested, details = onset_metrics(test_predictions, selected["threshold"])
        pair_test = onset_pair_metrics(test_predictions)
        write_json(output / "test_events.json", details)
        probability_directory = output / "probabilities"
        probability_directory.mkdir(exist_ok=True)
        for split, prediction_set in (("validation", exported), ("test", test_predictions)):
            for source, probabilities in prediction_set:
                np.save(probability_directory / (Path(source["features"]).stem + f"-{split}.npy"), probabilities)
        return {"ok": True, "training_complete": True, "candidate_only": True, "app_ready": False,
                "checkpoint_epoch": saved["epoch"], "checkpoint_sha256": sha256(checkpoint),
                "model": str(onnx_path), "model_sha256": sha256(onnx_path),
                "onnx_max_probability_error": max_error, "threshold": selected["threshold"],
                "onnx_validation_comparison": parity, "evaluation_backend": "ONNX Runtime CPU",
                "validation_acceptance": acceptance,
                "initial_weights_sha256": contract["initial_weights_sha256"],
                "shared_initial_weights_sha256": contract["shared_initial_weights_sha256"],
                "spectral_rise": spectral_rise, "history_frames": history_frames,
                "input_batches_sha256": [row["input_batches_sha256"] for row in history],
                "training_batches_sha256": [row["training_batches_sha256"] for row in history],
                "pair_batches_sha256": [row["pair_batches_sha256"] for row in history],
                "pair_weight": pair_weight, "pair_validation": pair_validation, "pair_test": pair_test,
                # Include the whole fixed threshold curve in the shareable summary,
                # so same-threshold comparisons do not require another file round trip.
                "validation_thresholds": [{"threshold": row["threshold"], "macro_domain_f1": row["macro_domain_f1"],
                    "groups": {name: {key: value for key, value in group.items() if not isinstance(value, list)}
                               for name, group in row["groups"].items()}} for row in table],
                "validation": selected, "test": tested,
                "limitations": ["This is a separate detector with a new input contract, not a drop-in app model.",
                                "No comparison with the existing ONNX head or actual app credits was run.",
                                "GuitarSet note starts do not verify picking technique; raw same-PC overlaps remain in recall.",
                                "A 12-class 96ms target cannot separate all closely spaced same-PC strings; overlap counts are reported.",
                                "Synthetic test uses the same simplified generator, not real guitar picking/noise.",
                                "Causal linear resampling has no antialiasing filter; validate the future live input path.",
                                "Audio-time latency excludes CPU scheduling and the app judge.",
                                "All test frames were evaluated once at the validation-selected threshold.",
                                "Test and AtoA were previously inspected; they are diagnostic regressions, not a fresh holdout.",
                                "Ringing activity follows annotations/synthetic stem support, not measured audibility."]}

    def run_training_pipeline(root, output_root, variant="auto", groups=(60, 12, 12),
                              epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH_SIZE,
                              seed=TRAIN_SEED, device="auto"):
        if epochs < 1 or batch_size < 1:
            raise ValueError("Positive epochs and batch size required")
        training_dependencies()  # Fail before expensive data generation when setup is unavailable.
        prepared = run_pipeline(root, output_root, variant, groups, seed=seed)
        run_dir = Path(prepared["summary_path"]).parent
        summary_path = run_dir / "training_summary.json"
        write_json(summary_path, {"ok": False, "stage": "features", "preparation_summary": prepared["summary_path"]})
        try:
            sources = cache_onset_features(onset_sources(Path(prepared["prepared_directory"])), run_dir / "features")
            audits = {}
            for source in sources:
                _, audit = ringing_mask(source, source["frames"], onset_targets(source["events"], source["frames"]))
                key = source["split"] + "/" + source["domain"]
                total = audits.setdefault(key, dict.fromkeys(audit, 0))
                for name, value in audit.items():
                    total[name] += value
            write_json(run_dir / "ringing_label_audit.json", audits)
            pair_audits = {}
            for split in ("train", "validation", "test"):
                pairs, audit = build_onset_pairs([s for s in sources if s["split"] == split])
                pair_audits[split] = audit
                write_json(run_dir / f"{split}_pairs.json", {"audit": audit, "pairs": pairs})
            write_json(summary_path, {"ok": False, "stage": "training", "sources": len(sources)})
            arms = {}
            for name, spectral_rise in (("control", False), ("rise", True)):
                arm_dir = run_dir / name
                arm_dir.mkdir()
                print(f"\nControlled experiment: {name}, positive spectral rise={spectral_rise}", flush=True)
                write_json(summary_path, {"ok": False, "stage": name, "completed_arms": list(arms)})
                arms[name] = train_onset_experiment(
                    sources, arm_dir, epochs, batch_size, seed, device, 1., run_dir / "features",
                    arms["control"]["validation"] if name == "rise" else None, 0., spectral_rise)
                write_json(arm_dir / "training_summary.json", arms[name])
            matched = all(arms["control"][key] == arms["rise"][key]
                          for key in ("shared_initial_weights_sha256", "training_batches_sha256"))
            if not matched:
                raise ValueError("Arms did not receive identical initialization/order/augmentation")
            result = {"schema_version": 4, "ok": True, "training_complete": True, "candidate_only": True, "app_ready": False,
                      "experiment": RISE_SPEC["version"], "controlled_training_verified": matched,
                      "ringing_spec": RINGING_SPEC,
                      "rise_spec": RISE_SPEC, "pair_audits": pair_audits,
                      "validation_criteria_met": arms["rise"]["validation_acceptance"]["accepted"],
                      "arms": arms, "ringing_label_audit": audits,
                      "summary_path": str(summary_path), "preparation_summary": prepared["summary_path"]}
            write_json(summary_path, result)
            return result
        except Exception as error:
            write_json(summary_path, {"ok": False, "error": str(error), "output_directory": str(run_dir)})
            error.add_note(f"Partial run and any completed checkpoints: {run_dir}")
            raise

    def training_main(argv=None):
        parser = argparse.ArgumentParser(description="Prepare and train the standalone onset experiment")
        parser.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"))
        parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working"))
        parser.add_argument("--variant", choices=("auto", "mic", "mix"), default="auto")
        parser.add_argument("--groups", type=int, nargs=3, default=(60, 12, 12))
        parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
        parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
        parser.add_argument("--seed", type=int, default=TRAIN_SEED)
        parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        args = parser.parse_args(argv)
        try:
            result = run_training_pipeline(args.input_dir, args.output_root, args.variant, args.groups,
                                           args.epochs, args.batch_size, args.seed, args.device)
        except Exception as error:
            # Keep a notebook run's diagnostic output readable even when export fails.
            # The library still raises; the notebook entry point reports explicit failure
            # and exits normally. This cannot guarantee retention by the hosting service.
            import traceback
            args.output_root.mkdir(parents=True, exist_ok=True)
            summary_path = args.output_root / "training_failure.json"
            result = {"ok": False, "training_complete": False, "app_ready": False,
                      "error": str(error), "traceback": traceback.format_exc(),
                      "summary_path": str(summary_path), "output_root": str(args.output_root)}
            write_json(summary_path, result)
            print("TRAINING FAILED — results are incomplete. Saved checkpoints were not removed.", flush=True)
        print(json.dumps(result, indent=2))
        print(f"Small training summary to share: {result['summary_path']}")
        return result
    from types import SimpleNamespace
    return SimpleNamespace(run_pipeline=run_pipeline, onset_sources=onset_sources, cache_onset_features=cache_onset_features, train_onset_experiment=train_onset_experiment, sha256=sha256, write_json=write_json, FEATURE_SPEC=FEATURE_SPEC)

import argparse

import hashlib

import json

import os

from pathlib import Path

import re

import shutil

_onset = _onset_runtime()
run_pipeline = _onset.run_pipeline
onset_sources = _onset.onset_sources
cache_onset_features = _onset.cache_onset_features
train_onset_experiment = _onset.train_onset_experiment
sha256 = _onset.sha256
write_json = _onset.write_json
FEATURE_SPEC = _onset.FEATURE_SPEC

class SnapshotStore:
    """Local snapshots first; only a confirmed absent file means 'start fresh'."""
    def __init__(self, directory, repo=None, token=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.repo, self.token = repo, token
        self.api = None
        self.files = set()
        if repo:
            from huggingface_hub import HfApi
            self.api = HfApi(token=token)
            # Authentication/network/repository errors propagate. Never silently restart.
            self.files = set(self.api.list_repo_files(repo_id=repo, repo_type="model"))

    def fetch(self, name):
        target = self.directory / name
        if target.is_file():
            return target
        if not self.api or name not in self.files:
            return None
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(repo_id=self.repo, filename=name, token=self.token)
        shutil.copy2(cached, target)
        return target

    def publish(self, path, name):
        path = Path(path)
        target = self.directory / name
        if path.resolve() != target.resolve():
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(path, temporary)
            temporary.replace(target)
        if self.api:
            # Failure leaves the local snapshot intact and stops with a useful error.
            self.api.upload_file(path_or_fileobj=str(target), path_in_repo=name,
                                 repo_id=self.repo, repo_type="model")
            self.files.add(name)
        return target

def choose_chord_start(store, run_tag, base_run, mode):
    if mode not in ("auto", "onset_only", "full"):
        raise ValueError("Mode must be auto, onset_only or full")
    own = store.fetch(f"checkpoint_{run_tag}_best.pth")
    if own:
        return "resume", own
    if mode != "full":
        base = store.fetch(f"checkpoint_{base_run}_best.pth")
        if base:
            return "base", base
    if mode == "onset_only":
        raise ValueError("onset_only needs a chord checkpoint; use auto/full to train everything")
    return "fresh", None

def weights_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def prepare_chords(config, store):
    import torch
    decision, path = choose_chord_start(store, config["run_tag"], config["base_run"], config["mode"])
    print(f"Chord base: {decision} ({path or 'random initialization'})", flush=True)
    runtime = chord_runtime(config, store)
    saved = torch.load(path, map_location="cpu", weights_only=False) if path else None
    if decision == "base" or saved and saved.get("phase1_done", False) and saved.get("phase2_done", False):
        model = runtime.model().to(runtime.device)
        runtime.load_weights(model, saved["model_state_dict"])
        own = Path(config["work_dir"]) / f"checkpoint_{config['run_tag']}_best.pth"
        if decision == "base":
            saved = dict(saved, model_state_dict=model.state_dict(),
                         phase1_done=True, phase2_done=True,
                         parent_checkpoint=path.name, parent_sha256=sha256(path))
            torch.save(saved, own)
            store.publish(own, own.name)
    else:
        if config["mode"] == "onset_only":
            raise ValueError("Own chord checkpoint is incomplete; use auto to resume the chord phases")
        runtime.train()
        own = store.fetch(f"checkpoint_{config['run_tag']}_best.pth")
        if own is None:
            raise RuntimeError("Full chord training produced no checkpoint")
        saved = torch.load(own, map_location="cpu", weights_only=False)
        model = runtime.model().to(runtime.device)
        runtime.load_weights(model, saved["model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    digest = weights_digest(model.state_dict())
    artifact = Path(config["work_dir"]) / f"best_model_{config['run_tag']}_chords.onnx"
    runtime.export(model, str(artifact), saved.get("best_threshold", .5))
    return dict(path=str(artifact), sha256=sha256(artifact), weights_sha256=digest,
                checkpoint=str(own), checkpoint_sha256=sha256(own),
                initialized_from=decision, pitch_threshold=saved.get("best_threshold", .5),
                outputs=["root_logits", "quality_logits", "pitch_logits"], legacy_onset=False)

def prepare_features(root, work, groups):
    index = work / "features" / "index.json"
    if index.exists():
        document = json.loads(index.read_text())
        if document["feature_spec"] != FEATURE_SPEC:
            raise ValueError("Cached onset features use a different DSP contract")
        sources = document["sources"]
        for source in sources:
            path = work / "features" / Path(source["features"]).name
            if not path.is_file() or sha256(path) != source["feature_sha256"]:
                raise ValueError(f"Missing or corrupt onset cache: {path}")
            source["features"] = str(path)
        return sources
    prepared = run_pipeline(root, work, "auto", groups=groups, seed=20260923)
    return cache_onset_features(onset_sources(Path(prepared["prepared_directory"])), work / "features")

def export_combined_model(chord_path, onset_path, output, onset_threshold):
    """Compose the trained branches into one graph; no Torch or training involved."""
    import onnx
    import onnxruntime as ort
    import numpy as np
    from onnx import compose, version_converter

    if not 0 < float(onset_threshold) < 1:
        raise ValueError("A validated onset threshold between zero and one is required")
    chord_path, onset_path, output = map(Path, (chord_path, onset_path, output))
    if output.resolve() in (chord_path.resolve(), onset_path.resolve()):
        raise ValueError("Combined export must not overwrite its source models")
    chords, onset = onnx.load(str(chord_path)), onnx.load(str(onset_path))
    chord_outputs = ["root_logits", "quality_logits", "pitch_logits"]
    if {v.name for v in chords.graph.input} != {"features"}:
        raise ValueError("Expected the CQT chord model with input 'features'")
    if {v.name for v in chords.graph.output} != set(chord_outputs):
        raise ValueError("Expected the three-output chord base without the retired onset head")
    if ({v.name for v in onset.graph.input} != {"short_features"} or
            {v.name for v in onset.graph.output} != {"onset_logits"}):
        raise ValueError("Expected the separate Rise model, not an old four-head chord model")
    if [d.dim_value for d in chords.graph.input[0].type.tensor_type.shape.dim][1:] != [48, 168]:
        raise ValueError("Chord model has an incompatible feature shape")
    if onset.graph.input[0].type.tensor_type.shape.dim[1].dim_value != 770:
        raise ValueError("Rise model must accept 770 short-spectrum features")
    # Both current exporters use opset17. Allow older chord exports through the
    # ONNX converter, guarded by numerical comparison to the original graph below.
    opsets = [{p.domain: p.version for p in m.opset_import} for m in (chords, onset)]
    if any(set(v) != {""} for v in opsets):
        raise ValueError("Combined export currently requires standard ONNX operators")
    version = max(v[""] for v in opsets)
    normalized = [version_converter.convert_version(m, version) if v[""] != version else m
                  for m, v in zip((chords, onset), opsets)]
    ir = max(m.ir_version for m in normalized)
    for model in normalized:
        model.ir_version = ir
    # Keep branch names disjoint, then restore the four public output names.
    merged = compose.merge_models(*normalized, io_map=[], prefix1="chords/", prefix2="rise/")
    names = {"chords/" + n: n for n in ["features", *chord_outputs]}
    names.update({"rise/" + n: n for n in ["short_features", "onset_logits"]})
    def rename_graph(graph):
        for item in [*graph.input, *graph.output, *graph.value_info, *graph.initializer]:
            item.name = names.get(item.name, item.name)
        for node in graph.node:
            for items in (node.input, node.output):
                for index, name in enumerate(items):
                    items[index] = names.get(name, name)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    rename_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph in attribute.graphs:
                        rename_graph(graph)
    rename_graph(merged.graph)
    metadata = {p.key: p.value for p in merged.metadata_props}
    metadata.update(model_kind="solitito-chord-rise-v1", onset_threshold=str(onset_threshold),
                    onset_history_frames="34", onset_feature_spec=json.dumps(FEATURE_SPEC, sort_keys=True),
                    chord_source_sha256=sha256(chord_path), onset_source_sha256=sha256(onset_path))
    onnx.helper.set_model_props(merged, metadata)
    onnx.checker.check_model(merged)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.onnx")
    onnx.save_model(merged, str(temporary), save_as_external_data=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    def session(path):
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    base_session, rise_session, combined = session(chord_path), session(onset_path), session(temporary)
    errors = dict.fromkeys([*chord_outputs, "onset_logits"], 0.)
    rng = np.random.default_rng(20260929)
    try:
        for batch, frames in ((1, 35), (2, 37), (1, 65)):
            cqt = rng.uniform(0, 1, (batch, 48, 168)).astype(np.float32)
            short = rng.uniform(0, .5, (batch, 770, frames)).astype(np.float32)
            reference = base_session.run(chord_outputs, {"features": cqt}) + rise_session.run(
                ["onset_logits"], {"short_features": short})
            actual = combined.run(list(errors), {"features": cqt, "short_features": short})
            for name, a, b in zip(errors, reference, actual):
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f"Invalid merged output: {name}")
                errors[name] = max(errors[name], float(np.max(np.abs(a - b))))
                if not np.allclose(a, b, rtol=2e-5, atol=2e-5):
                    raise ValueError(f"Merged output changed: {name}, max error {errors[name]}")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(path=str(output), sha256=sha256(output), inputs={"features": ["batch", 48, 168],
                "short_features": ["batch", 770, "time"]}, outputs=list(errors),
                onset_output_shape=["batch", 12, "time"], onset_threshold=float(onset_threshold),
                parity_max_absolute_error=errors, parity_cases=3,
                runtime_integration_required=True)

def export_take7_only(config, store):
    """Recover the completed two-file take7 run without datasets or optimizers."""
    tag = config["run_tag"]
    work = Path(config["work_dir"])
    report_path = store.fetch(f"training_summary_{tag}.json")
    report = json.loads(report_path.read_text()) if report_path else {}
    chord = store.fetch(f"best_model_{tag}_chords.onnx")
    onset = store.fetch(f"best_model_{tag}_onset.onnx")
    if onset is None and (work / "rise" / "short_onset_rise.onnx").is_file():
        onset = work / "rise" / "short_onset_rise.onnx"
    if chord is None or onset is None:
        raise FileNotFoundError("export_only requires the saved *_chords.onnx and *_onset.onnx "
                                "(or rise/short_onset_rise.onnx). No training was started.")
    threshold = config.get("export_onset_threshold")
    if threshold is None:
        threshold = report.get("onset", {}).get("threshold")
    if threshold is None:
        raise ValueError("Missing onset threshold: restore training_summary_<RUN_TAG>.json "
                         "or set EXPORT_ONSET_THRESHOLD to its selected threshold. No training was started.")
    model = export_combined_model(chord, onset, work / f"best_model_{tag}.onnx", threshold)
    store.publish(model["path"], Path(model["path"]).name)
    report.update(schema_version=2, ok=True, run_tag=tag, model=model, app_ready=False,
                  export_only=True, training_performed=False,
                  note="One ONNX, two inputs, four outputs. Requires the matching application input path.")
    destination = work / f"training_summary_{tag}.json"
    report["summary_path"] = str(destination)
    write_json(destination, report)
    store.publish(destination, destination.name)
    print(f"Export complete, no training: {model['path']}", flush=True)
    return report

def run_take7(config, store):
    work = Path(config["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    tag = config["run_tag"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag) or tag == config["base_run"]:
        raise ValueError("Choose a simple new RUN_TAG different from BASE_RUN; never overwrite take6")
    if config["mode"] == "export_only":
        return export_take7_only(config, store)
    report_path = work / f"training_summary_{tag}.json"
    write_json(report_path, dict(ok=False, stage="chords", run_tag=tag))
    chords = prepare_chords(config, store)
    write_json(report_path, dict(ok=False, stage="onsets", run_tag=tag, chords=chords))
    sources = prepare_features(Path(config["input_dir"]), work, config.get("groups", (60, 12, 12)))
    onset_dir = work / "rise"
    onset_dir.mkdir(exist_ok=True)
    last_name = f"checkpoint_{tag}_onset_last.pth"
    previous = store.fetch(last_name)
    if previous:
        shutil.copy2(previous, onset_dir / "short_onset_last.pt")
    initial = config.get("initial_onset") or None
    if initial and not Path(initial).is_file():
        raise ValueError(f"Initial Rise checkpoint does not exist: {initial}")
    print("Onsets: " + ("resuming take7" if previous else "initial Rise weights" if initial else
                        "training Rise from scratch; chord base is frozen"), flush=True)
    result = train_onset_experiment(sources, onset_dir,
                                   epochs=config.get("onset_epochs", 12),
                                   batch_size=config.get("onset_batch_size", 16),
                                   device_name=config.get("device", "auto"),
                                   feature_directory=work / "features", spectral_rise=True,
                                   initial_checkpoint=initial if not previous else None, resume=True,
                                   checkpoint_callback=lambda path: store.publish(path, last_name))
    # No reference to the chord model is passed into the onset optimizer.
    import torch
    saved = torch.load(chords["checkpoint"], map_location="cpu", weights_only=False)
    if weights_digest(saved["model_state_dict"]) != chords["weights_sha256"]:
        raise RuntimeError("Chord weights changed during onset training")
    model = export_combined_model(chords["path"], result["model"],
                                  work / f"best_model_{tag}.onnx", result["threshold"])
    store.publish(model["path"], Path(model["path"]).name)
    store.publish(onset_dir / "short_onset_best.pt", f"checkpoint_{tag}_onset_best.pth")
    summary = dict(schema_version=2, ok=True, training_complete=True,
                   candidate_only=True, app_ready=False, run_tag=tag, mode=config["mode"],
                   model=model, chords=chords, onset=result, summary_path=str(report_path),
                   note="One ONNX, two inputs, four outputs. Requires the matching application input path.")
    write_json(report_path, summary)
    store.publish(report_path, report_path.name)
    print(f"Take7 complete. Model: {model['path']}\nReport: {report_path}", flush=True)
    return summary

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "onset_only", "full", "export_only"), default=MODE)
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--base-run", default=BASE_RUN)
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--hf-repo", default=HF_REPO_ID)
    parser.add_argument("--no-hf", action="store_true", default=not USE_HF)
    parser.add_argument("--initial-onset", default=INITIAL_ONSET)
    parser.add_argument("--onset-epochs", type=int, default=ONSET_EPOCHS)
    parser.add_argument("--export-onset-threshold", type=float, default=EXPORT_ONSET_THRESHOLD)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    import sys
    args = parser.parse_args([] if argv is None and "ipykernel" in sys.modules else argv)
    config = vars(args)
    config["work_dir"] = str(Path(args.output_root) / args.run_tag)
    if config["mode"] == "export_only":
        config["device"] = "cpu"
    elif config["device"] == "auto":
        import torch
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    token = os.environ.get("HF_TOKEN")
    if not args.no_hf and not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as error:
            raise RuntimeError("Set the Kaggle HF_TOKEN secret or use --no-hf / USE_HF=False") from error
    store = SnapshotStore(config["work_dir"], None if args.no_hf else args.hf_repo, token)
    try:
        return run_take7(config, store)
    except Exception as error:
        write_json(Path(config["work_dir"]) / "training_failure.json",
                   dict(ok=False, training_complete=False, error=str(error), work_dir=config["work_dir"]))
        raise

if __name__ == "__main__":
    main()
