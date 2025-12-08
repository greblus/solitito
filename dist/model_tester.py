import sys
import subprocess
import importlib
import os
import re
import warnings
import glob

# ==========================================
# 0. DUAL LOGGER
# ==========================================
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(self.ansi_escape.sub('', message))

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger("model_benchmark.txt")

# ==========================================
# 1. SETUP
# ==========================================
def install_libs():
    pkgs = ["numpy", "pandas", "librosa", "soundfile", "onnxruntime", "seaborn", "matplotlib", "tqdm", "scikit-learn", "scipy"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    for p in pkgs:
        try: importlib.import_module(p if p != "scikit-learn" else "sklearn")
        except: subprocess.check_call([sys.executable, "-m", "pip", "install", p, "--quiet"])

print("🔍 Inicjalizacja środowiska...")
install_libs()

import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io.wavfile
import scipy.signal
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
from collections import Counter

warnings.filterwarnings("ignore")

# ==========================================
# 2. KONFIGURACJA V13 (HYBRID PRECISION)
# ==========================================
MODEL_FILENAME = "chord_model_v13_final.onnx" 
TEST_WAV = "dataset_eob.wav" 
TEST_CSV = "dataset_annotations.csv"

# Parametry DSP (Musi być 1:1 z Rust V13)
SR = 22050          
HOP_LENGTH = 512
MIN_NOTE = 'C1'
N_BINS = 192        
BINS_PER_OCTAVE = 24
FILTER_SCALE = 0.85 
RUST_FFT_SIZE = 8192
CTX_FRAMES = 32

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Noise"]
QUALS = ["", "m", "7", "Maj7", "m7", "dim7", "m7b5", "9", "13", "Note"] 

ROOT_TO_IDX = {r: i for i, r in enumerate(ROOTS)}
QUAL_TO_IDX = {q: i for i, q in enumerate(QUALS)}

JAZZ_QUALS = ["7", "Maj7", "m7", "dim7", "m7b5", "9", "13"]
BASIC_QUALS = ["", "m", "Note"]

# ==========================================
# 3. DSP SIMULATION (RUST TWIN V13)
# ==========================================
def rust_dsp_simulation(y):
    # 1. CQT
    cqt_complex = librosa.cqt(y, sr=SR, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz(MIN_NOTE), 
                              n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, 
                              filter_scale=FILTER_SCALE)
    cqt_mag = np.abs(cqt_complex)
    n_frames = cqt_mag.shape[1]
    
    # 2. BASS SURGERY: VECTORIZED GOERTZEL (FIXED TARGETS)
    # E2 (82.4Hz) jest na binie 32 przy 24 bins/oct (C1=0)
    target_freqs = [82.41, 87.31, 92.50, 98.00, 103.83]
    target_indices = [32, 34, 36, 38, 40]
    
    # Framing
    pad_len = RUST_FFT_SIZE // 2
    y_padded = np.pad(y, (pad_len, pad_len), mode='constant')
    y_frames = librosa.util.frame(y_padded, frame_length=RUST_FFT_SIZE, hop_length=HOP_LENGTH)
    
    min_frames = min(n_frames, y_frames.shape[1])
    cqt_mag = cqt_mag[:, :min_frames]
    y_frames = y_frames[:, :min_frames]
    
    # Basis Matrix
    N = RUST_FFT_SIZE
    basis_matrix = np.zeros((len(target_freqs), N), dtype=np.complex64)
    t_vec = np.arange(N)
    for i, freq in enumerate(target_freqs):
        k = np.round(freq * N / SR)
        basis_matrix[i, :] = np.exp(-2j * np.pi * k * t_vec / N)
        
    # Fast Goertzel Calculation
    goertzel_energies = np.abs(basis_matrix @ y_frames) * 0.5 
    
    # Injection & Smart Blend
    for i, idx in enumerate(target_indices):
        g_en_row = goertzel_energies[i, :]
        
        # Harmonic Check (+24 biny)
        harm_idx = idx + 24
        if harm_idx < N_BINS:
            harm_row = cqt_mag[harm_idx, :]
            boost_mask = harm_row > (g_en_row * 0.2)
            g_en_row[boost_mask] *= 1.5
        
        # AGC Blend
        fft_row = cqt_mag[idx, :]
        blend_mask = g_en_row > fft_row
        cqt_mag[idx, blend_mask] = fft_row[blend_mask] * 0.3 + g_en_row[blend_mask] * 0.7

    # 3. Cleanup (Sub-bass < E2)
    cqt_mag[0:32, :] *= 0.1
    
    # 4. Normalization
    ref_val = np.max(cqt_mag) if np.max(cqt_mag) > 0 else 1.0
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=ref_val)
    norm = np.clip((cqt_db + 80.0) / 80.0, 0, 1)

    # 5. SOFT SQUELCH (20% - LINEAR)
    mask = norm < 0.20
    norm[mask] = 0.0
    norm[~mask] = (norm[~mask] - 0.20) / 0.80 
    # Brak potęgowania!

    # 6. Chroma
    chroma = librosa.feature.chroma_cqt(C=norm, sr=SR, hop_length=HOP_LENGTH, 
                                        n_chroma=12, bins_per_octave=BINS_PER_OCTAVE)
    
    return np.vstack([norm, chroma]).T.astype(np.float32)

def robust_load(path, target_sr):
    try:
        y, _ = librosa.load(path, sr=target_sr, mono=True)
        return y
    except Exception:
        try:
            sr_native, y = scipy.io.wavfile.read(path)
            if y.dtype == np.int16: y = y.astype(np.float32) / 32768.0
            elif y.dtype == np.int32: y = y.astype(np.float32) / 2147483648.0
            elif y.dtype == np.uint8: y = (y.astype(np.float32) - 128.0) / 128.0
            else: y = y.astype(np.float32)
            if len(y.shape) > 1: y = y.mean(axis=1)
            if sr_native != target_sr:
                y = scipy.signal.resample(y, int(len(y) * target_sr / sr_native))
            return y
        except Exception as e2:
            print(f"❌ CRITICAL: Nie udało się otworzyć pliku audio: {e2}")
            return None

# ==========================================
# 4. PARSER & FINDER
# ==========================================
NOTE_MAP = {"Db":"C#", "Eb":"D#", "Gb":"F#", "Ab":"G#", "Bb":"A#"}
def split_chord_label_smart(chord_str):
    if not isinstance(chord_str, str): return None, None
    chord_str = chord_str.strip()
    if chord_str in ["N", "Noise"]: return "Noise", ""
    match = re.match(r"^([A-G][#b]?)\s*(.*)$", chord_str)
    if not match: 
        if ":" in chord_str:
            parts = chord_str.split(":")
            r = NOTE_MAP.get(parts[0], parts[0])
            q = parts[1].split("/")[0].split("(")[0]
            if "maj7" in q: q = "Maj7"
            elif "min7" in q: q = "m7"
            elif "7" in q: q = "7"
            elif "maj" in q: q = ""
            elif "min" in q: q = "m"
            return r, q
        return None, None

    root = NOTE_MAP.get(match.group(1), match.group(1))
    qual_raw = match.group(2).strip().lower()
    q = None
    if qual_raw in ["", "maj", "major"]: q = ""
    elif qual_raw in ["m", "min", "minor", "-"]: q = "m"
    elif qual_raw in ["7", "dom7"]: q = "7"
    elif qual_raw in ["maj7", "j7", "m7", "major7"]: q = "Maj7"
    elif qual_raw in ["m7", "min7", "-7"]: q = "m7"
    elif qual_raw in ["dim", "dim7", "o", "0"]: q = "dim7"
    elif qual_raw in ["m7b5", "hdim", "hdim7", "ø"]: q = "m7b5"
    elif qual_raw in ["9", "add9"]: q = "9"
    elif qual_raw in ["13"]: q = "13"
    elif qual_raw == "note": return root, "Note"
    return root, q

def find_file(filename):
    if os.path.exists(filename): return filename
    for root, dirs, files in os.walk('/kaggle/input'):
        if filename in files: return os.path.join(root, filename)
    for root, dirs, files in os.walk('./'):
        if filename in files: return os.path.join(root, filename)
    return None

# ==========================================
# 5. MAIN
# ==========================================
model_path = find_file(MODEL_FILENAME)
wav_path = find_file(TEST_WAV)
csv_path = find_file(TEST_CSV)

if not model_path: sys.exit(f"❌ Brak modelu: {MODEL_FILENAME}")
if not wav_path or not csv_path: sys.exit("❌ Brak plików datasetu.")

print(f"🧠 Model: {os.path.basename(model_path)}")
print(f"🎵 Audio: {os.path.basename(wav_path)}")

print("⏳ Przetwarzanie DSP (Symulacja Rust V13)...")
y = robust_load(wav_path, SR)
features = rust_dsp_simulation(y)

if features is None: sys.exit("❌ Błąd DSP.")
print(f"✅ DSP Gotowe. Kształt: {features.shape}")

# Inferencja
sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

try:
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = [c.strip().lower() for c in df.columns]
    col_lbl = next((c for c in df.columns if 'label' in c or 'chord' in c), None)
    col_start = next((c for c in df.columns if 'start' in c), None)
    col_end = next((c for c in df.columns if 'end' in c), None)
except: sys.exit("❌ Błąd CSV")

def get_truth_tuple(t_sec):
    row = df[(df[col_start] <= t_sec) & (df[col_end] > t_sec)]
    if not row.empty: 
        return split_chord_label_smart(str(row.iloc[0][col_lbl]))
    return None, None

def format_chord(r, q):
    if r == "Noise": return "Noise"
    if q == "Note": return f"Note {r}"
    if q == "": return r 
    return f"{r} {q}"

y_true_str, y_pred_str = [], []
y_true_q = []

STRIDE = 4
num_steps = features.shape[0] - CTX_FRAMES

print("🚀 Uruchamianie benchmarku...")
ignored = 0

for t in tqdm(range(0, num_steps, STRIDE)):
    center_time = (t + CTX_FRAMES//2) * HOP_LENGTH / SR
    t_root, t_qual = get_truth_tuple(center_time)
    
    if not t_root: continue
    if t_root not in ROOTS or t_qual not in QUALS:
        ignored += 1
        continue
    
    inp = features[t : t+CTX_FRAMES][np.newaxis, :, :]
    outs = sess.run(None, {input_name: inp})
    
    def sm(x): 
        e=np.exp(x-np.max(x)); return e/e.sum()
    
    pr = sm(outs[0][0])
    pq = sm(outs[1][0])
    
    p_root = ROOTS[np.argmax(pr)]
    p_qual = QUALS[np.argmax(pq)]
    
    t_full = format_chord(t_root, t_qual)
    p_full = format_chord(p_root, p_qual)
    
    y_true_str.append(t_full)
    y_pred_str.append(p_full)
    y_true_q.append(t_qual)

# ==========================================
# 6. RAPORT
# ==========================================
if not y_true_str: sys.exit("❌ Brak wyników.")

acc = 100 * sum([1 for t, p in zip(y_true_str, y_pred_str) if t == p]) / len(y_true_str)

jazz_ok = sum([1 for tq, t, p in zip(y_true_q, y_true_str, y_pred_str) if tq in JAZZ_QUALS and t==p])
jazz_tot = sum([1 for tq in y_true_q if tq in JAZZ_QUALS])
basic_ok = sum([1 for tq, t, p in zip(y_true_q, y_true_str, y_pred_str) if tq in BASIC_QUALS and t==p])
basic_tot = sum([1 for tq in y_true_q if tq in BASIC_QUALS])

print("\n" + "="*60)
print(f"📊 RAPORT SKUTECZNOŚCI MODELU: {MODEL_FILENAME}")
print("="*60)
print(f"🏆 GLOBAL ACCURACY:      {acc:.2f}%")
if basic_tot > 0:
    print(f"🔹 BASIC (Triady/Notes): {100*basic_ok/basic_tot:.2f}%")
if jazz_tot > 0:
    print(f"🎷 JAZZ (7/9/13/dim):    {100*jazz_ok/jazz_tot:.2f}%")
print("-" * 60)

stats = {}
for t, p in zip(y_true_str, y_pred_str):
    if t not in stats: stats[t] = {'ok': 0, 'tot': 0, 'errs': []}
    stats[t]['tot'] += 1
    if t == p: 
        stats[t]['ok'] += 1
    else:
        stats[t]['errs'].append(p)

results = sorted([(k, v) for k, v in stats.items()], key=lambda x: 100*x[1]['ok']/x[1]['tot'])

print(f"{'AKORD':<15} | {'ACC':<8} | {'SAMPLES'} | {'TYPOWE BŁĘDY'}")
print("-" * 75)

for label, data in results:
    acc_lbl = 100 * data['ok'] / data['tot']
    c = "\033[91m" if acc_lbl < 50 else "\033[93m" if acc_lbl < 80 else "\033[92m"
    
    err_str = ""
    if data['errs']:
        most_common = Counter(data['errs']).most_common(1)
        err_chord, err_count = most_common[0]
        err_pct = int(100 * err_count / len(data['errs']))
        err_str = f"-> {err_chord} ({err_pct}%)"

    print(f"{c}{label:<15} | {acc_lbl:6.2f}% | {data['tot']:<7} | {err_str}\033[0m")

print("\n✅ Wyniki zapisano do pliku: model_benchmark.txt")

plt.figure(figsize=(20, 18))
labels_sorted = sorted(list(set(y_true_str + y_pred_str)))
cm = confusion_matrix(y_true_str, y_pred_str, labels=labels_sorted, normalize='true')
sns.heatmap(cm, annot=False, xticklabels=labels_sorted, yticklabels=labels_sorted, cmap='viridis')
plt.title(f"Confusion Matrix (Accuracy: {acc:.2f}%)", fontsize=16)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("✅ Wykres zapisano jako: confusion_matrix.png")
