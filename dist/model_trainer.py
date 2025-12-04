import subprocess
import sys
import os

# --- INSTALACJA ---
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import librosa, torch, pandas, onnx, onnxscript, soundfile
except ImportError:
    install("librosa")
    install("torch")
    install("pandas")
    install("onnx")
    install("onnxscript")
    install("tqdm")
    install("soundfile")

import glob
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.onnx
import pandas as pd
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split

# ==========================================
# KONFIGURACJA
# ==========================================
TARGET_CSV = "dataset_annotations.csv"
TARGET_WAV_CLEAN = "dataset_clean.wav"
TARGET_WAV_EOB = "dataset_eob.wav"

# --- ZMIANY ---
# Zwiększamy kontekst (dłuższy fragment audio dla sieci)
CTX_FRAMES = 32  # Było 16. Teraz sieć widzi ~0.7 sekundy.
LOG_BINS = 216 
SAMPLE_RATE = 22050
FFT_SIZE = 2048
HOP_LENGTH = 512
BATCH_SIZE = 128
EPOCHS = 40

# Klasy
ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALS = ["", "m", "Maj7", "m7", "7", "dim7", "m7b5", "9", "13"] 
LABELS = [f"{r} {q}".strip() for r in ROOTS for q in QUALS] + ["Noise"]
for r in ROOTS: LABELS.append(f"Note {r}")
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# ==========================================
# WYSZUKIWANIE
# ==========================================
def find_file(filename):
    for root, dirs, files in os.walk("/kaggle/input"):
        if filename in files: return os.path.join(root, filename)
    for root, dirs, files in os.walk("."):
        if filename in files: return os.path.join(root, filename)
    return None

PATH_CSV = find_file(TARGET_CSV)
PATH_CLEAN = find_file(TARGET_WAV_CLEAN)
PATH_EOB = find_file(TARGET_WAV_EOB)

CUSTOM_AUDIO_PATHS = []
if PATH_CLEAN: CUSTOM_AUDIO_PATHS.append(PATH_CLEAN)
if PATH_EOB: CUSTOM_AUDIO_PATHS.append(PATH_EOB)

if not PATH_CSV or not CUSTOM_AUDIO_PATHS:
    print("❌ Brak plików wejściowych.")
    sys.exit(1)

# ==========================================
# DSP
# ==========================================
def compute_log_spectrogram(audio):
    if len(audio) < FFT_SIZE: audio = np.pad(audio, (0, FFT_SIZE - len(audio)))
    spec = np.abs(np.fft.rfft(audio, n=FFT_SIZE))
    log_spec = np.zeros(LOG_BINS, dtype=np.float32)
    hz_per_bin = SAMPLE_RATE / FFT_SIZE
    F_MIN = 65.4
    BINS_PER_OCTAVE = 36
    for i in range(LOG_BINS):
        center_freq = F_MIN * (2.0 ** (i / BINS_PER_OCTAVE))
        fft_idx = int(round(center_freq / hz_per_bin))
        if 0 <= fft_idx < len(spec):
            val = spec[fft_idx]
            if fft_idx > 0: val += spec[fft_idx-1] * 0.5
            if fft_idx < len(spec)-1: val += spec[fft_idx+1] * 0.5
            log_spec[i] = val
    log_spec = np.log1p(log_spec)
    if log_spec.max() > 0: log_spec /= log_spec.max()
    return log_spec

# ==========================================
# DATASET (FIXED)
# ==========================================
class CorrectedDataset(Dataset):
    def __init__(self, custom_csv, custom_wavs):
        self.samples = []
        print(f"\n>>> 🎸 Wczytywanie danych (BEZ BŁĘDNEJ AUGMENTACJI)...")
        
        df = pd.read_csv(custom_csv)
        
        for wav_path in custom_wavs:
            fname = os.path.basename(wav_path)
            print(f"-> {fname}")
            y_full, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            total_len = len(y_full)
            
            for _, row in tqdm(df.iterrows(), total=len(df)):
                label = row['label']
                if label not in LABEL_TO_IDX: continue
                lbl_idx = LABEL_TO_IDX[label]
                
                t_start = float(row['start'])
                t_end = float(row['end'])
                
                s_start = int(t_start * sr)
                s_end = int(t_end * sr)
                
                if s_end > total_len: s_end = total_len
                if s_end - s_start < FFT_SIZE: continue
                
                chunk = y_full[s_start:s_end]
                
                # --- POPRAWKA: TYLKO STRIDE, BEZ PITCH SHIFT ---
                # Żeby mieć więcej danych, robimy gęstszy stride (kroczymy oknem częściej)
                # Normalnie shiftowaliśmy pitch, teraz po prostu bierzemy więcej okienek z tego samego akordu.
                
                specs = []
                for k in range(0, len(chunk)-FFT_SIZE, HOP_LENGTH):
                    specs.append(compute_log_spectrogram(chunk[k:k+FFT_SIZE]))
                
                if len(specs) < CTX_FRAMES: continue
                specs = np.array(specs)
                
                # Stride = 4 (Bardzo gęsto, żeby nadrobić ilość danych)
                for t in range(0, len(specs)-CTX_FRAMES, 4):
                    self.samples.append((specs[t:t+CTX_FRAMES], lbl_idx))
        
            del y_full
            gc.collect()

        # Dodajemy SZUM (to jest bezpieczne i potrzebne)
        print(">>> 🌫️ Adding Noise samples...")
        num_noise = int(len(self.samples) * 0.1)
        noise_idx = LABEL_TO_IDX["Noise"]
        for _ in range(num_noise):
            n = np.random.normal(0, 0.02, (CTX_FRAMES, LOG_BINS)).astype(np.float32)
            self.samples.append((np.log1p(np.abs(n)), noise_idx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        return torch.tensor(self.samples[i][0], dtype=torch.float32), torch.tensor(self.samples[i][1], dtype=torch.long)

# ==========================================
# TRAINING LOOP
# ==========================================
ds = CorrectedDataset(PATH_CSV, CUSTOM_AUDIO_PATHS)
print(f"Liczba próbek: {len(ds)}")

train_len = int(0.9 * len(ds))
tr, te = random_split(ds, [train_len, len(ds)-train_len])
train_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(te, batch_size=BATCH_SIZE, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model nieco głębszy, bo mamy 32 ramki czasu
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1, 2)),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((1, 2)),
    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((1, 2)), 
    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((1, 2)), # Dodatkowa warstwa
).to(device)

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = model
        # Input size: LOG_BINS(216) / 2 / 2 / 2 / 2 = 13.5 -> 13
        # 256 kanałów * 13 cech = 3328
        self.gru = nn.GRU(3328, 256, 2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(256, NUM_CLASSES)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).reshape(b, t, c*f)
        _, hn = self.gru(x)
        return self.fc(hn[-1])

final_model = FullModel().to(device)
if torch.cuda.device_count() > 1: final_model = nn.DataParallel(final_model)

crit = nn.CrossEntropyLoss()
opt = optim.Adam(final_model.parameters(), lr=0.001)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, verbose=True)

best_acc = 0.0
OUTPUT_DIR = "/kaggle/working/"

print("\n--- START TRENINGU (BEZ BŁĘDÓW) ---")
for ep in range(EPOCHS):
    final_model.train()
    loss_sum = 0
    loop = tqdm(train_loader, desc=f"Ep {ep+1}", leave=False)
    for X, y in loop:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        out = final_model(X)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        loop.set_postfix(loss=loss.item())
    
    final_model.eval()
    corr = 0; tot = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = final_model(X)
            _, p = torch.max(out, 1)
            tot += y.size(0)
            corr += (p==y).sum().item()
    
    avg_loss = loss_sum/len(train_loader)
    acc = 100*corr/tot
    print(f"Ep {ep+1}: Loss {avg_loss:.4f} | Acc {acc:.2f}% | LR {opt.param_groups[0]['lr']:.6f}")
    sched.step(avg_loss) # ReduceLR based on Loss
    
    if acc > best_acc:
        best_acc = acc
        m = final_model.module if isinstance(final_model, nn.DataParallel) else final_model
        torch.onnx.export(m, torch.randn(1, CTX_FRAMES, LOG_BINS).to(device), "chord_model_fixed.onnx", 
                          input_names=['in'], output_names=['out'], dynamic_axes={'in':{0:'b'}, 'out':{0:'b'}})
        print("💾 Model Zapisany")

print("Done.")
