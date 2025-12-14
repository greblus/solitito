import sys
import subprocess
import importlib
import os
import re
import warnings
import json
import glob

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
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
from collections import Counter

warnings.filterwarnings("ignore")

# ==========================================
# 2. KONFIGURACJA V31 (16k)
# ==========================================
MODEL_FILENAME = "chord_model_v31_16k.onnx" 
TEST_WAV = "dataset_eob.wav" 
TEST_CSV = "dataset_annotations.csv"

SR = 16000          
HOP_LENGTH = 256    
CTX_FRAMES = 32
N_BINS = 144        
BINS_PER_OCTAVE = 24
MIN_NOTE = 'C1'

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Noise"]
QUALS = ["", "m", "7", "Maj7", "m7", "dim7", "m7b5", "9", "13", "Note"] 

ROOT_TO_IDX = {r: i for i, r in enumerate(ROOTS)}
QUAL_TO_IDX = {q: i for i, q in enumerate(QUALS)}

JAZZ_QUALS = ["7", "Maj7", "m7", "dim7", "m7b5", "9", "13"]
BASIC_QUALS = ["", "m", "Note"]

SILENCE_THRESHOLD = 0.02 

# ==========================================
# 3. DSP ENGINE (LIBROSA STANDARD V31)
# ==========================================
def process_audio(wav_path):
    try:
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
    except Exception as e:
        print(f"❌ Błąd ładowania audio: {e}")
        return None, None
    
    # CQT (16k)
    cqt = librosa.cqt(y, sr=SR, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz(MIN_NOTE), 
                      n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, filter_scale=1.0)
    cqt_mag = np.abs(cqt)
    
    energy_profile = np.mean(cqt_mag, axis=0)
    if np.max(energy_profile) > 0: energy_profile /= np.max(energy_profile)
    
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
    norm = (cqt_db + 80.0) / 80.0
    norm = np.clip(norm, 0, 1)
    
    chroma = librosa.feature.chroma_cqt(C=norm, sr=SR, hop_length=HOP_LENGTH, 
                                        n_chroma=12, bins_per_octave=BINS_PER_OCTAVE)
    
    return np.vstack([norm, chroma]).T.astype(np.float32), energy_profile

# ==========================================
# 4. PARSER & HELPERS
# ==========================================
def find_file(filename):
    if os.path.exists(filename): return filename
    for root, dirs, files in os.walk('/kaggle/input'):
        if filename in files: return os.path.join(root, filename)
    for root, dirs, files in os.walk('./'):
        if filename in files: return os.path.join(root, filename)
    return None

NOTE_MAP = {"Db":"C#", "Eb":"D#", "Gb":"F#", "Ab":"G#", "Bb":"A#"}
def split_chord_label_smart(chord_str):
    if not isinstance(chord_str, str): return None, None
    chord_str = chord_str.strip()
    if chord_str in ["N", "Noise"]: return "Noise", ""
    
    lower = chord_str.lower()
    if lower.startswith("note ") or lower.endswith(" note"):
        clean = lower.replace("note", "").strip().capitalize()
        root = NOTE_MAP.get(clean, clean)
        return root, "Note"

    match = re.match(r"^([A-G][#b]?)\s*(.*)$", chord_str)
    if not match: 
        if ":" in chord_str:
            p = chord_str.split(":")
            r = NOTE_MAP.get(p[0], p[0])
            q = p[1].split("/")[0].split("(")[0]
            if "maj7" in q: q = "Maj7"
            elif "min7" in q: q = "m7"
            elif "7" in q: q = "7"
            elif "maj" in q: q = ""
            elif "min" in q: q = "m"
            return r, q
        return None, None
    
    r_raw = match.group(1)
    root = NOTE_MAP.get(r_raw, r_raw)
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
    elif qual_raw == "note": q = "Note"
    
    return root, q

def format_chord(r, q):
    if r == "Noise": return "Noise"
    if q == "Note": return f"Note {r}"
    if q == "": return r 
    return f"{r} {q}"

# ==========================================
# 5. MAIN BENCHMARK
# ==========================================
model_path = find_file(MODEL_FILENAME)
wav_path = find_file(TEST_WAV)
csv_path = find_file(TEST_CSV)

if not model_path: sys.exit(f"❌ Brak modelu: {MODEL_FILENAME}")
if not wav_path or not csv_path: sys.exit("❌ Brak plików datasetu.")

print(f"🧠 Model: {os.path.basename(model_path)}")
print(f"🎵 Audio: {os.path.basename(wav_path)}")

print("⏳ Przetwarzanie DSP (Librosa 16k)...")
features, energy_profile = process_audio(wav_path)

if features is None: sys.exit("❌ Błąd DSP.")
print(f"✅ DSP Gotowe. Kształt: {features.shape}")

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
    if not row.empty: return split_chord_label_smart(str(row.iloc[0][col_lbl]))
    return None, None

y_true_str, y_pred_str = [], []
y_true_q = []

STRIDE = 4
num_steps = features.shape[0] - CTX_FRAMES

print("🚀 Uruchamianie benchmarku...")
ignored_silence = 0

for t in tqdm(range(0, num_steps, STRIDE)):
    window_energy = np.mean(energy_profile[t : t+CTX_FRAMES])
    if window_energy < SILENCE_THRESHOLD:
        ignored_silence += 1
        continue 

    center_time = (t + CTX_FRAMES//2) * HOP_LENGTH / SR
    t_root, t_qual = get_truth_tuple(center_time)
    
    if not t_root: continue
    if t_root not in ROOTS or t_qual not in QUALS: continue
    
    inp = features[t : t+CTX_FRAMES][np.newaxis, :, :]
    outs = sess.run(None, {input_name: inp})
    
    def sm(x): e=np.exp(x-np.max(x)); return e/e.sum()
    pr = sm(outs[0][0])
    pq = sm(outs[1][0])
    
    p_root = ROOTS[np.argmax(pr)]
    p_qual = QUALS[np.argmax(pq)]
    
    t_full = format_chord(t_root, t_qual)
    p_full = format_chord(p_root, p_qual)
    
    y_true_str.append(t_full)
    y_pred_str.append(p_full)
    y_true_q.append(t_qual)

print(f"ℹ️  Pominięto {ignored_silence} próbek ciszy.")

if not y_true_str: sys.exit("⚠️ Wszystkie próbki odrzucone.")

# RAPORT
acc = 100 * sum([1 for t, p in zip(y_true_str, y_pred_str) if t == p]) / len(y_true_str)
jazz_ok = sum([1 for tq, t, p in zip(y_true_q, y_true_str, y_pred_str) if tq in JAZZ_QUALS and t==p])
jazz_tot = sum([1 for tq in y_true_q if tq in JAZZ_QUALS])
basic_ok = sum([1 for tq, t, p in zip(y_true_q, y_true_str, y_pred_str) if tq in BASIC_QUALS and t==p])
basic_tot = sum([1 for tq in y_true_q if tq in BASIC_QUALS])
note_ok = sum([1 for tq, t, p in zip(y_true_q, y_true_str, y_pred_str) if tq == "Note" and t==p])
note_tot = sum([1 for tq in y_true_q if tq == "Note"])

print("\n" + "="*60)
print(f"📊 RAPORT SKUTECZNOŚCI MODELU: {MODEL_FILENAME}")
print("="*60)
print(f"🏆 GLOBAL ACCURACY:      {acc:.2f}%")
if basic_tot > 0: print(f"🔹 BASIC:       {100*basic_ok/basic_tot:.2f}%")
if jazz_tot > 0:  print(f"🎷 JAZZ:        {100*jazz_ok/jazz_tot:.2f}%")
if note_tot > 0:  print(f"🎵 NOTES:       {100*note_ok/note_tot:.2f}%")
print("-" * 60)

stats = {}
for t, p in zip(y_true_str, y_pred_str):
    if t not in stats: stats[t] = {'ok': 0, 'tot': 0, 'errs': []}
    stats[t]['tot'] += 1
    if t == p: stats[t]['ok'] += 1
    else: stats[t]['errs'].append(p)

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
