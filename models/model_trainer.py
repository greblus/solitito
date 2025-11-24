# -*- coding: utf-8 -*-

!pip install onnx onnxscript jams librosa tqdm torch

import os
import glob
import zipfile
import requests
import jams
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.onnx
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.signal import sawtooth

# --- KONFIGURACJA ---
SAMPLE_RATE = 44100 # Standard
FFT_SIZE = 4096     # Duże okno dla basu
HOP_LENGTH = 1024
# Log Mapping: 128 binów (ok. 10.5 oktawy od ~32Hz)
LOG_BINS = 128      
BINS_PER_OCTAVE = 12 # Jeden bin na półton (CNN to kocha)
F_MIN = 32.7         # C1 (niski bas)

# Etykiety (Akordy + Nuty + Noise)
ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALS = ["", "m", "Maj7", "m7", "7"]
LABELS = []
for r in ROOTS:
    for q in QUALS: LABELS.append(f"{r} {q}".strip())
for r in ROOTS: LABELS.append(f"Note {r}")
LABELS.append("Noise")
NUM_CLASSES = len(LABELS)
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

# --- 1. KLUCZOWA FUNKCJA DSP (To samo będzie w Rust) ---
def compute_log_spectrogram(audio, sr):
    # 1. FFT
    # Padujemy, żeby mieć stały rozmiar
    if len(audio) < FFT_SIZE:
        audio = np.pad(audio, (0, FFT_SIZE - len(audio)))
    
    spec = np.abs(np.fft.rfft(audio, n=FFT_SIZE))
    
    # 2. Mapowanie Liniowe -> Logarytmiczne (Ręczne, żeby w Rust było to samo)
    log_spec = np.zeros(LOG_BINS, dtype=np.float32)
    
    # Pre-calc częstotliwości binów FFT
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1/sr)
    
    for i in range(LOG_BINS):
        # Środek kubełka logarytmicznego
        # freq = f_min * 2^(bin / bins_per_octave)
        center_freq = F_MIN * (2.0 ** (i / BINS_PER_OCTAVE))
        
        # Znajdujemy odpowiedni indeks w FFT
        # idx = freq / (sr / fft_size)
        fft_idx = int(round(center_freq / (sr / FFT_SIZE)))
        
        if 0 <= fft_idx < len(spec):
            # Bierzemy energię z tego punktu (i sąsiadów dla wygładzenia)
            # Proste sumowanie 3 prążków
            val = spec[fft_idx]
            if fft_idx > 0: val += spec[fft_idx-1] * 0.5
            if fft_idx < len(spec)-1: val += spec[fft_idx+1] * 0.5
            log_spec[i] = val

    # 3. Logarytm amplitudy i Normalizacja
    log_spec = np.log1p(log_spec)
    if log_spec.max() > 0:
        log_spec /= log_spec.max()
        
    return log_spec

# --- 2. DANE (Hybrid: Real + Synth) ---
def prepare_data():
    if not os.path.exists("guitarset_audio"):
        r = requests.get("https://zenodo.org/records/3371780/files/audio_mono-pickup_mix.zip?download=1")
        with open("audio.zip", "wb") as f: f.write(r.content)
        with zipfile.ZipFile("audio.zip", "r") as z: z.extractall("guitarset_audio")
    if not os.path.exists("guitarset_annotations"):
        r = requests.get("https://zenodo.org/records/3371780/files/annotation.zip?download=1")
        with open("annot.zip", "wb") as f: f.write(r.content)
        with zipfile.ZipFile("annot.zip", "r") as z: z.extractall("guitarset_annotations")
prepare_data()

def map_jams(chord_str):
    if chord_str == "N": return "Noise"
    try:
        r, q = chord_str.split(':')
        r = r.replace('b','#').replace('Eb','D#').replace('Bb','A#').replace('Ab','G#').replace('Db','C#').replace('Gb','F#')
        if "maj7" in q: t="Maj7"
        elif "min7" in q or "m7" in q: t="m7"
        elif "7" in q and "maj" not in q and "min" not in q: t="7"
        elif "maj" in q: t=""
        elif "min" in q or "m" in q: t="m"
        else: return None
        l = f"{r} {t}".strip()
        return l if l in LABEL_TO_IDX else None
    except: return None

def gen_note(root, dur=0.3):
    try: idx = ROOTS.index(root)
    except: return np.zeros(int(SAMPLE_RATE*dur))
    t = np.linspace(0, dur, int(SAMPLE_RATE*dur), endpoint=False)
    midi = 12 * np.random.choice([2,3,4]) + idx + 12
    if midi < 40: midi += 12
    f = 440.0 * (2**((midi-69)/12))
    w = (sawtooth(2*np.pi*f*t)*0.6 + np.sin(2*np.pi*f*t)*0.4) * np.exp(-4.0*t)
    return w.astype(np.float32)

class Dataset40(Dataset):
    def __init__(self, files, jams_dir):
        self.data = []
        print("Building Dataset 4.0...")
        # Real
        for af in tqdm(files):
            y, sr = librosa.load(af, sr=SAMPLE_RATE, mono=True)
            fname = os.path.basename(af).replace("_mix.wav", ".jams")
            jpath = glob.glob(f"{jams_dir}/**/{fname}", recursive=True)[0]
            ann = jams.load(jpath).annotations.search(namespace='chord')[0]
            
            # Kroimy na kawałki FFT_SIZE (ok 0.1s)
            num_samples = len(y)
            step = FFT_SIZE // 2 # Overlap 50%
            
            for i in range(0, num_samples - FFT_SIZE, step * 4): # Co 4 krok
                chunk = y[i : i + FFT_SIZE]
                time_sec = (i + FFT_SIZE/2) / sr
                
                # Znajdź etykietę
                lbl_idx = LABEL_TO_IDX["Noise"]
                for obs in ann.data:
                    if obs.time <= time_sec <= obs.time + obs.duration:
                        l = map_jams(obs.value)
                        if l: lbl_idx = LABEL_TO_IDX[l]
                        break
                
                spec = compute_log_spectrogram(chunk, sr)
                self.data.append((spec, lbl_idx))
                
        # Synth Notes
        for _ in range(len(self.data)//3):
            r = np.random.choice(ROOTS)
            w = gen_note(r)
            # Wklejamy w środek ramki
            if len(w) > FFT_SIZE: chunk = w[:FFT_SIZE]
            else: chunk = np.pad(w, (0, FFT_SIZE-len(w)))
            spec = compute_log_spectrogram(chunk, SAMPLE_RATE)
            self.data.append((spec, LABEL_TO_IDX[f"Note {r}"]))
            
        # Noise
        for _ in range(len(self.data)//5):
            noise = np.random.normal(0, 0.05, FFT_SIZE).astype(np.float32)
            spec = compute_log_spectrogram(noise, SAMPLE_RATE)
            self.data.append((spec, LABEL_TO_IDX["Noise"]))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return torch.tensor(self.data[i][0]), torch.tensor(self.data[i][1])

fs = sorted(glob.glob("guitarset_audio/**/*.wav", recursive=True))
ds = Dataset40(fs, "guitarset_annotations")
train_loader = DataLoader(ds, batch_size=64, shuffle=True)

# --- 3. MODEL CNN (Log Input) ---
class ChordCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Input: [Batch, 1, 128]
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), # -> 64
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), # -> 32
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), # -> 16
            nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x): return self.net(x.unsqueeze(1))

model = ChordCNN()
if torch.cuda.is_available(): model = model.cuda()
opt = optim.Adam(model.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()

print("Training...")
for ep in range(15):
    model.train()
    tl = 0
    for x, y in train_loader:
        if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        tl += loss.item()
    print(f"Ep {ep}: {tl/len(train_loader)}")

model.eval().cpu()
torch.onnx.export(model, torch.randn(1, LOG_BINS), "chord_model.onnx", input_names=['input'], output_names=['output'], dynamic_axes={'input':{0:'b'},'output':{0:'b'}})
print("Done.")
