import sys
import subprocess
import importlib
import os
import re

# ==========================================
# 0. DUAL LOGGER (EKRAN + PLIK)
# ==========================================
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
        # Regex do usuwania kolorów z pliku tekstowego
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        self.terminal.write(message) # Na ekran z kolorami
        clean_msg = self.ansi_escape.sub('', message)
        self.log.write(clean_msg)    # Do pliku czysty tekst

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Przekierowanie wyjścia
sys.stdout = DualLogger("model_benchmark.txt")

# ==========================================
# 1. SETUP
# ==========================================
def install_libs():
    pkgs = ["numpy", "pandas", "librosa", "soundfile", "onnxruntime", "seaborn", "matplotlib", "tqdm", "scikit-learn"]
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

# ==========================================
# 2. KONFIGURACJA
# ==========================================
# Zmień nazwę, jeśli Twój plik nazywa się inaczej (np. v9)
MODEL_FILENAME = "chord_model_v10_final.onnx" 

TEST_WAV = "dataset_eob.wav"
TEST_CSV = "dataset_annotations.csv"

# Parametry DSP (Muszą być zgodne z treningiem)
SAMPLE_RATE = 22050
HOP_LENGTH = 512
MIN_NOTE = 'C1' 
N_BINS = 192         
BINS_PER_OCTAVE = 24 
CTX_FRAMES = 32

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Noise"]
QUALS = ["", "m", "7", "Maj7", "m7", "dim7", "m7b5", "9", "13", "Note"] 

ROOT_TO_IDX = {r: i for i, r in enumerate(ROOTS)}
QUAL_TO_IDX = {q: i for i, q in enumerate(QUALS)}

JAZZ_QUALS = ["7", "Maj7", "m7", "dim7", "m7b5", "9", "13"]
BASIC_QUALS = ["", "m", "Note"]

# ==========================================
# 3. PARSER (ROBUST REGEX)
# ==========================================
NOTE_MAP = {"Db":"C#", "Eb":"D#", "Gb":"F#", "Ab":"G#", "Bb":"A#"}
Q_MAP = { "maj": "", "min": "m", "maj7": "Maj7", "min7": "m7", "7": "7", "dim": "dim7", "dim7": "dim7", "hdim7": "m7b5", "maj9": "9", "min9": "m7", "9": "9", "maj13": "13", "min13": "m7", "13": "13" }

def split_chord_label_smart(chord_str):
    if not isinstance(chord_str, str): return None, None
    chord_str = chord_str.strip()
    
    if chord_str == "N" or chord_str == "Noise": return "Noise", "Note"
    if chord_str.startswith("Note"): 
        r = chord_str.split(" ")[1]
        return (NOTE_MAP.get(r, r), "Note")

    # JAMS style
    if ":" in chord_str:
        parts = chord_str.split(":")
        root = parts[0]
        root = NOTE_MAP.get(root, root)
        if len(parts) == 1: return root, "" 
        q_raw = parts[1].split("/")[0].split("(")[0]
        if "(9)" in parts[1] or "9" in q_raw: return root, ("m7" if "min" in q_raw else "9")
        if "(13)" in parts[1] or "13" in q_raw: return root, ("m7" if "min" in q_raw else "13")
        return root, Q_MAP.get(q_raw, None)

    # Custom CSV style (Regex)
    else:
        match = re.match(r"^([A-G][#b]?)(.*)$", chord_str)
        if not match: return None, None
        root = match.group(1)
        root = NOTE_MAP.get(root, root)
        qp = match.group(2).strip().lower()
        
        q = None
        if qp == "": q = ""
        elif qp in ["m", "min"]: q = "m"
        elif qp == "7": q = "7"
        elif qp in ["maj7", "maj"]: q = "Maj7"
        elif qp in ["m7", "min7"]: q = "m7"
        elif qp in ["dim7", "dim"]: q = "dim7"
        elif qp in ["m7b5", "hdim7"]: q = "m7b5"
        elif qp == "9": q = "9"
        elif qp == "13": q = "13"
        return root, q

# ==========================================
# 4. PLIKI & DSP
# ==========================================
def find_file(filename):
    if os.path.exists(filename): return filename
    for root, dirs, files in os.walk('/kaggle/input'):
        if filename in files: return os.path.join(root, filename)
    for root, dirs, files in os.walk('/kaggle/working'):
        if filename in files: return os.path.join(root, filename)
    return None

model_path = find_file(MODEL_FILENAME)
wav_path = find_file(TEST_WAV)
csv_path = find_file(TEST_CSV)

if not model_path: sys.exit(f"❌ Brak modelu: {MODEL_FILENAME}")
if not wav_path or not csv_path: 
    # Fallback
    wav_path = find_file("dataset_clean.wav")
    if not wav_path: sys.exit("❌ Brak plików audio.")

print(f"🧠 Model: {os.path.basename(model_path)}")
print(f"🎵 Audio: {os.path.basename(wav_path)}")

print("⏳ Generowanie CQT + Chroma...")
y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
cqt = librosa.cqt(y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz(MIN_NOTE), n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
chroma = librosa.feature.chroma_cqt(C=np.abs(cqt), sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_chroma=12, bins_per_octave=BINS_PER_OCTAVE)
cqt_norm = np.clip((cqt_db + 80.0) / 80.0, 0.0, 1.0)
features = np.vstack([cqt_norm, chroma]).T.astype(np.float32)
print(f"✅ DSP Gotowe. Kształt: {features.shape}")

# ==========================================
# 5. INFERENCJA
# ==========================================
sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

# Wczytanie CSV
try:
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = [c.strip().lower() for c in df.columns]
    col_lbl = next((c for c in df.columns if 'label' in c or 'chord' in c), None)
    col_start = next((c for c in df.columns if 'start' in c), None)
    col_end = next((c for c in df.columns if 'end' in c), None)
    
    # Fallback columns
    if not col_lbl and len(df.columns) >= 3:
        col_start, col_end, col_lbl = df.columns[0], df.columns[1], df.columns[2]
except: sys.exit("❌ Błąd CSV")

def get_truth_tuple(t_sec):
    row = df[(df[col_start] <= t_sec) & (df[col_end] > t_sec)]
    if not row.empty: 
        return split_chord_label_smart(str(row.iloc[0][col_lbl]))
    return None, None

def format_chord(r, q):
    if r == "Noise": return "Noise"
    if q == "Note": return f"Note {r}"
    if q == "": return r # Major
    return f"{r} {q}"

y_true_str, y_pred_str = [], []
y_true_q = []

STRIDE = 4
num_steps = features.shape[0] - CTX_FRAMES

print("🚀 Uruchamianie benchmarku...")
ignored = 0

for t in tqdm(range(0, num_steps, STRIDE)):
    center_time = (t + CTX_FRAMES//2) * HOP_LENGTH / SAMPLE_RATE
    t_root, t_qual = get_truth_tuple(center_time)
    
    if not t_root: continue
    if t_root not in ROOTS or t_qual not in QUALS:
        ignored += 1
        continue
    
    inp = features[t : t+CTX_FRAMES][np.newaxis, :, :]
    outs = sess.run(None, {input_name: inp})
    
    p_root = ROOTS[np.argmax(outs[0][0])]
    p_qual = QUALS[np.argmax(outs[1][0])]
    
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

# Szczegółowa tabela
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

# ==========================================
# 7. WYKRES (CONFUSION MATRIX)
# ==========================================
plt.figure(figsize=(20, 18)) # Duży rozmiar
labels_sorted = sorted(list(set(y_true_str + y_pred_str)))

# Normalizacja po wierszach (True label)
cm = confusion_matrix(y_true_str, y_pred_str, labels=labels_sorted, normalize='true')

sns.heatmap(cm, annot=False, xticklabels=labels_sorted, yticklabels=labels_sorted, cmap='viridis')
plt.title(f"Confusion Matrix (Accuracy: {acc:.2f}%)", fontsize=16)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Zapis wykresu
plt.savefig("confusion_matrix.png", dpi=150)
print("✅ Wykres zapisano jako: confusion_matrix.png")
plt.show()
