# ==========================================
# CHORD TRAINER — nazwa przebiegu w RUN_TAG
# ==========================================
# Three phases, run automatically in one session:
#
# PHASE 1 - MAIN TRAINING (epochs 0-120)
#   Fresh start when there is no checkpoint on HF, resume when one exists and
#   epoch < EPOCHS. Early stopping with patience=15 on the composite score.
#
# PHASE 2 - THRESHOLD TUNING
#   After phase 1 (or when the checkpoint carries the 'phase1_done' flag) the
#   script scans threshold 0.30-0.70 in steps of 0.01 on the validation set and
#   writes the best one into the checkpoint and to HF.
#
# PHASE 3 - PITCH HEAD FINE-TUNING
#   Encoder frozen, only fc_pitch trains for FINETUNE_EPOCHS epochs at
#   LR=FINETUNE_LR, using the threshold from phase 2. Exports the final ONNX
#   with the threshold in its metadata.
# ==========================================

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

print("🛠 Installing dependencies...")
def install_deps():
    packages = ["onnx", "onnxruntime", "jams", "mir_eval", "huggingface_hub", "joblib"]
    subprocess.call([sys.executable, "-m", "pip", "install"] + packages + ["--quiet"])

install_deps()

from huggingface_hub import HfApi, hf_hub_download
from kaggle_secrets import UserSecretsClient
import joblib
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")

# ==========================================
# KONFIGURACJA
# ==========================================
HF_REPO_ID     = "greblus/chord-model-snapshots"
HF_SECRET_NAME = "HF_TOKEN"

# Token: Kaggle secret first, environment variable second. The second path is a
# fallback - secrets must be attached to EVERY notebook separately (copying one
# does not carry them over), and the Kaggle UI moves around.
hf_token = None
_src = None
try:
    hf_token = UserSecretsClient().get_secret(HF_SECRET_NAME)
    _src = f"sekret Kaggle '{HF_SECRET_NAME}'"
except Exception:
    hf_token = os.environ.get(HF_SECRET_NAME)
    if hf_token:
        _src = f"environment variable {HF_SECRET_NAME}"

if not hf_token:
    sys.exit(
        f"\n❌ No HF token.\n\n"
        f"   A) Kaggle secret: Add-ons -> Secrets, label exactly "
        f"'{HF_SECRET_NAME}'.\n"
        f"      Secrets attach per notebook - a copy does not inherit them.\n\n"
        f"   B) Fallback, in a cell BEFORE running (private notebooks only!):\n"
        f"         import os; os.environ['{HF_SECRET_NAME}'] = 'hf_...'\n"
    )

api = HfApi(token=hf_token)
print(f"✅ Token HF z: {_src}   ->  {HF_REPO_ID}")

SR             = 16000
HOP_LENGTH     = 256
MIN_NOTE       = 'C1'
N_BINS         = 144
BINS_PER_OCTAVE = 24
INPUT_FEATURES = 168
CTX_FRAMES     = 48

# ---- Phase 1: main training ----
BATCH_SIZE          = 48
EPOCHS              = 120
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
# RUN NAME - the only thing to change when starting fresh. Every checkpoint,
# ONNX and log name (local and on HF) derives from it, so a new take is one line
# and training restarts from scratch (old files stay on HF as a backup).
# ==========================================
RUN_TAG    = "v2_take6"          # next run: "v2_take7" and so on

CKPT_BEST  = f"checkpoint_{RUN_TAG}_best.pth"
CKPT_FT    = f"checkpoint_{RUN_TAG}_finetuned.pth"
ONNX_BEST  = f"best_model_{RUN_TAG}.onnx"
ONNX_FT    = f"best_model_{RUN_TAG}_finetuned.onnx"
HIST_CSV   = f"training_history_{RUN_TAG}.csv"
HIST_FT    = f"training_history_{RUN_TAG}_finetune.csv"
LOG_TXT    = f"training_log_{RUN_TAG}.txt"

INPUT_DIR  = "/kaggle/input"
WORK_DIR   = "/kaggle/working"
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


def cleanup_legacy_cache():
    """Removes cache directories from older runs.

    Their files are named after a random hash, so after the move to `cache_key`
    they are UNREACHABLE - nothing will ever read them, and they can eat several
    GB of the /kaggle/working quota. The new directory is left alone.
    """
    import glob as _glob
    os.makedirs(CACHE_DIR, exist_ok=True)
    freed, dropped = 0, []
    for src in _glob.glob(os.path.join(WORK_DIR, "cache_*")):
        base = os.path.basename(src)
        if not os.path.isdir(src) or base.startswith("cache_feat_"):
            continue
        for r, _, fs in os.walk(src):
            for f in fs:
                try: freed += os.path.getsize(os.path.join(r, f))
                except OSError: pass
        shutil.rmtree(src, ignore_errors=True)
        dropped.append(base)
    if dropped:
        print(f"🧹 Removed dead cache ({', '.join(dropped)}): {freed/2**30:.1f} GB.")
        print("   Those files were named after a random hash - unreachable anyway.")
    n = len(os.listdir(CACHE_DIR)) if os.path.isdir(CACHE_DIR) else 0
    print(f"💾 Feature cache: {CACHE_DIR}  ({n} files)")

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

# The cache is shared between sessions - we never clear it automatically.
# Uncomment below only after changing audio parameters (SR, HOP_LENGTH, N_BINS...)
# if os.path.exists(CACHE_DIR):
#     try: shutil.rmtree(CACHE_DIR)
#     except: pass
cleanup_legacy_cache()

# --- CLEAN LOG: tee stdout to a file (tqdm goes to stderr, so no progress bars) ---
TRAIN_LOG = os.path.join(WORK_DIR, LOG_TXT)
class _Tee:
    def __init__(self, path):
        self.term = sys.stdout
        self.log  = open(path, "w", encoding="utf-8")
        self._ansi = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    def write(self, m):
        self.term.write(m); self.log.write(self._ansi.sub('', m)); self.log.flush()
    def flush(self):
        self.term.flush(); self.log.flush()
try:
    sys.stdout = _Tee(TRAIN_LOG)
    print(f"📝 Clean stdout log -> {TRAIN_LOG}")
except Exception as e:
    print(f"⚠️ Could not open the log file: {e}")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        x      = self.enc(self.inorm(x.unsqueeze(1)))
        b, c, t, f = x.size()
        x      = self.proj(x.permute(0, 2, 1, 3).reshape(b, t, c * f))
        x      = torch.cat((self.cls.expand(b, -1, -1), x), 1) + self.pos
        emb    = self.tr(x)[:, 0]
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
    try:
        api.upload_file(path_or_fileobj=file_path, path_in_repo=name_in_repo,
                        repo_id=HF_REPO_ID, repo_type="model")
    except: pass

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
            opset_version=14
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

def load_checkpoint_meta():
    """
    Downloads the checkpoint from HF and returns its dict (or None).
    Does not load the weights into a model - metadata only.
    """
    try:
        files = list(api.list_repo_files(repo_id=HF_REPO_ID))
        if CKPT_BEST not in files:
            return None
        local = hf_hub_download(repo_id=HF_REPO_ID, filename=CKPT_BEST,
                                 local_dir=WORK_DIR, token=hf_token)
        return torch.load(local, map_location='cpu', weights_only=False)
    except:
        return None

def resume_from_checkpoint(model, opt, sched):
    """
    Loads the best HF checkpoint into the model, optimizer and scheduler.
    Returns (start_epoch, best_composite, best_loss, best_f1, best_exact, phase1_done, best_threshold).
    """
    try:
        files = list(api.list_repo_files(repo_id=HF_REPO_ID))
        if CKPT_BEST not in files:
            print(f"🆕 No {RUN_TAG} checkpoint on HF - training from scratch")
            return 0, 0.0, 999.0, 0.0, 0.0, False, 0.5

        print("🔄 Pobieranie checkpointa z HF...")
        local = hf_hub_download(repo_id=HF_REPO_ID, filename=CKPT_BEST,
                                 local_dir=WORK_DIR, token=hf_token)
        ckpt  = torch.load(local, map_location=device, weights_only=False)

        model.load_state_dict(ckpt['model_state_dict'])
        # optimizer and scheduler only matter while phase 1 is unfinished
        if not ckpt.get('phase1_done', False):
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            sched.load_state_dict(ckpt['scheduler_state_dict'])

        ep            = ckpt['epoch']
        best_comp     = ckpt.get('best_composite',  0.0)
        best_loss     = ckpt.get('best_loss',       999.0)
        best_f1       = ckpt.get('best_f1',         0.0)
        best_exact    = ckpt.get('best_exact',       0.0)
        phase1_done   = ckpt.get('phase1_done',     False)
        best_threshold = ckpt.get('best_threshold', 0.5)

        if HIST_CSV in files:
            hf_hub_download(repo_id=HF_REPO_ID, filename=HIST_CSV,
                            local_dir=WORK_DIR, token=hf_token)

        status = "phase 1 done" if phase1_done else f"epoch {ep + 1}"
        print(f"   ✅ Loaded ({status}) | Loss={best_loss:.4f} F1={best_f1:.3f} "
              f"Exact={best_exact:.1%} Threshold={best_threshold:.2f}")
        return ep + 1, best_comp, best_loss, best_f1, best_exact, phase1_done, best_threshold

    except Exception as e:
        print(f"⚠️  Cannot load the checkpoint ({e}) - training from scratch")
        return 0, 0.0, 999.0, 0.0, 0.0, False, 0.5

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
        improved = composite > best_composite + 1e-6

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
            upload_file_safe(TRAIN_LOG, LOG_TXT)   # log to HF (insurance)

        if ep >= WARMUP_EPOCHS + 10 and no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"\n⏹️  Early stopping after epoch {ep+1} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # mark phase 1 as done in the checkpoint, keeping the last best metrics
    ckpt_meta = load_checkpoint_meta()
    if ckpt_meta:
        m_saved = ckpt_meta.get('metrics', {})
        save_checkpoint(model, None, None,
                        ckpt_meta['epoch'], best_composite, best_loss,
                        best_f1, best_exact,
                        m_saved.get('root_acc', 0), m_saved.get('pitch_f1', 0),
                        m_saved.get('pitch_prec', 0), m_saved.get('pitch_rec', 0),
                        m_saved.get('chord_exact', 0), phase1_done=True)
        # ONNX once, from the best phase 1 weights
        model.load_state_dict(ckpt_meta['model_state_dict'])
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
# PHASE 3 - PITCH HEAD FINE-TUNING
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
        sampler=tr_l.sampler, shuffle=False, num_workers=2, pin_memory=True
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
        model.load_state_dict(ckpt_ft['model_state_dict'])
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
                      shuffle=False, num_workers=2, pin_memory=True)
    te_l = DataLoader(ds_vl, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=2)

    # Loader for measuring accuracy ON THE TRAINING DATA (no augmentation, a subset
    # about the size of val). If the model cannot memorise its own data, the fault
    # is in the features or the labels.
    ds_tr_eval = FrameBasedDataset(tr_items, training=False)
    idx_eval   = random.sample(range(len(ds_tr_eval)), min(4000, len(ds_tr_eval)))
    tr_eval_l  = DataLoader(torch.utils.data.Subset(ds_tr_eval, idx_eval),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = ChordTransformer().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,}")

    # --- Phase 1 ---
    phase1_train(model, tr_l, te_l, tr_eval_l)

    # --- Phase 2: threshold tuning on the best phase 1 model ---
    # reload the best weights (later phase 1 epochs may have changed the model)
    ckpt_meta = load_checkpoint_meta()
    if ckpt_meta:
        model.load_state_dict(ckpt_meta['model_state_dict'])
        print("🔄 Loaded the best weights for threshold tuning")

    best_threshold = phase2_threshold_tuning(model, te_l)

    # --- Phase 3: pitch head fine-tuning ---
    # reload the best phase 1 weights (phase 2 did not change the model)
    if RUN_PHASE3:
        if ckpt_meta:
            model.load_state_dict(ckpt_meta['model_state_dict'])
        phase3_finetune_pitch(model, tr_l, te_l, best_threshold)
    else:
        print("\n⏭️  Phase 3 skipped (RUN_PHASE3=False) - see the comment on the flag.")

    # The cache is NOT deleted - it is shared by every run with the same signal
    # parameters and rebuilding it costs tens of CPU minutes. Delete it by hand, or
    # uncomment the cleanup at the top of the file, after changing SR/HOP/N_BINS.
    print(f"\n💾 Feature cache kept: {CACHE_DIR}")

    print("\n" + "="*60)
    print("✅ ALL PHASES COMPLETE")
    ckpt_ft = load_checkpoint_meta()
    if ckpt_ft:
        print(f"   Threshold:   {ckpt_ft.get('best_threshold', 0.5):.2f}")
        m = ckpt_ft.get('metrics', {})
        print(f"   Root Acc:    {m.get('root_acc', 0):.1%}")
        print(f"   Pitch F1:    {m.get('pitch_f1', 0):.3f}")
        print(f"   Exact Match: {m.get('chord_exact', 0):.1%}")
    # Without phase 3 the `_finetuned` file is never produced - calling it the
    # final model sent people after an artifact that does not exist.
    final_onnx = ONNX_FT if RUN_PHASE3 else ONNX_BEST
    print(f"   Final model: {final_onnx}")
    print("="*60)


if __name__ == "__main__":
    main()
