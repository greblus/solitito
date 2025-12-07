import os
import sys
import subprocess
import importlib
import time
import requests
import zipfile
import glob
import json
import shutil
import warnings
import random
import re
from collections import Counter
from tqdm import tqdm

# ==========================================
# 1. INSTALACJA
# ==========================================
def install_libs():
    print("⬇️  Środowisko...")
    pkgs = ["soundfile", "librosa", "pandas", "tqdm", "onnx", "onnxscript", "onnxruntime", "numpy", "torch", "requests", "scipy"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--quiet"])
    for p in pkgs:
        try: importlib.import_module(p)
        except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", p, "--only-binary=:all:", "--quiet"])

install_libs()

import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import onnxruntime as ort
import scipy.signal
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset

# ==========================================
# 2. KONFIGURACJA
# ==========================================
WORK_DIR = "./workspace"
GUITARSET_DIR = os.path.join(WORK_DIR, "guitarset")
os.makedirs(GUITARSET_DIR, exist_ok=True)

SAMPLE_RATE = 22050
HOP_LENGTH = 512
MIN_NOTE = 'C1' 
N_BINS = 192         
BINS_PER_OCTAVE = 24 
CTX_FRAMES = 32      
AUDIO_WINDOW_SIZE = (CTX_FRAMES * HOP_LENGTH) + 4096 

BATCH_SIZE = 256
EPOCHS = 60

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Noise"]
QUALS = ["", "m", "7", "Maj7", "m7", "dim7", "m7b5", "9", "13", "Note"] 

ROOT_TO_IDX = {r: i for i, r in enumerate(ROOTS)}
QUAL_TO_IDX = {q: i for i, q in enumerate(QUALS)}

# ==========================================
# 3. DANE
# ==========================================
def download_file(url, destination):
    if os.path.exists(destination): return
    print(f"⬇️  Pobieranie {os.path.basename(destination)}...")
    response = requests.get(url, stream=True)
    if response.status_code != 200: sys.exit(1)
    with open(destination, 'wb') as file:
        for data in response.iter_content(1024*1024):
            file.write(data)

def setup_guitarset():
    URL_AUDIO = "https://zenodo.org/records/3371780/files/audio_mono-pickup_mix.zip?download=1"
    URL_JAMS = "https://zenodo.org/records/3371780/files/annotation.zip?download=1"
    if not os.path.exists(os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix")):
        download_file(URL_AUDIO, "audio.zip")
        try:
            with zipfile.ZipFile("audio.zip", 'r') as z: z.extractall(GUITARSET_DIR)
            os.remove("audio.zip")
        except: pass
    if not os.path.exists(os.path.join(GUITARSET_DIR, "annotation")):
        download_file(URL_JAMS, "jams.zip")
        try:
            with zipfile.ZipFile("jams.zip", 'r') as z: z.extractall(GUITARSET_DIR)
            os.remove("jams.zip")
        except: pass

setup_guitarset()

OUTPUT_CSV = "hybrid_dataset.csv"
NOTE_MAP = {"Db":"C#", "Eb":"D#", "Gb":"F#", "Ab":"G#", "Bb":"A#"}
Q_MAP = { "maj": "", "min": "m", "maj7": "Maj7", "min7": "m7", "7": "7", "dim": "dim7", "dim7": "dim7", "hdim7": "m7b5", "maj9": "9", "min9": "m7", "9": "9", "maj13": "13", "min13": "m7", "13": "13" }

# --- FIX: ROBUST PARSER V10 ---
def split_chord_label(chord_str):
    if not isinstance(chord_str, str): return None, None
    chord_str = chord_str.strip()
    
    # 1. Noise / Note
    if chord_str == "N" or chord_str == "Noise": return "Noise", "Note"
    if chord_str.startswith("Note"): 
        parts = chord_str.split()
        if len(parts) > 1: return (NOTE_MAP.get(parts[1], parts[1]), "Note")
        return None, None

    # 2. JAMS format (dwukropek)
    if ":" in chord_str:
        parts = chord_str.split(":")
        root = parts[0]
        root = NOTE_MAP.get(root, root)
        if len(parts) == 1: return root, "" 
        q_raw = parts[1].split("/")[0].split("(")[0]
        if "(9)" in parts[1] or "9" in q_raw: return root, ("m7" if "min" in q_raw else "9")
        if "(13)" in parts[1] or "13" in q_raw: return root, ("m7" if "min" in q_raw else "13")
        return root, Q_MAP.get(q_raw, None)

    # 3. Custom Format (Regex + Strip)
    else:
        match = re.match(r"^([A-G][#b]?)(.*)$", chord_str)
        if not match: return None, None
        
        root = match.group(1)
        root = NOTE_MAP.get(root, root)
        
        # FIX: strip() usuwa spację przed ' Maj7'
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

def generate_csv():
    if os.path.exists(OUTPUT_CSV): os.remove(OUTPUT_CSV)
    data = []
    
    # 1. GuitarSet
    print("🔄 GuitarSet...")
    audio_map = {}
    for f in glob.glob(os.path.join(GUITARSET_DIR, "**", "*.wav"), recursive=True):
        stem = os.path.basename(f).replace(".wav", "").replace("_mic", "").replace("_mix", "")
        audio_map[stem] = f

    gs_cnt = 0
    for j_path in tqdm(glob.glob(os.path.join(GUITARSET_DIR, "**", "*.jams"), recursive=True)):
        j_stem = os.path.basename(j_path).replace(".jams", "")
        if j_stem not in audio_map: continue
        try:
            with open(j_path, "r") as f: content = json.load(f)
            for ann in content["annotations"]:
                if ann["namespace"] == "chord":
                    for obs in ann["data"]:
                        r, q = split_chord_label(obs["value"])
                        if r in ROOT_TO_IDX and q in QUAL_TO_IDX:
                            data.append({"filename": audio_map[j_stem], "start": obs["time"], "end": obs["time"] + obs["duration"], "root": r, "quality": q, "source": "GuitarSet"})
                            gs_cnt += 1
        except: pass

    # 2. Custom Data
    print("🔄 Custom Data...")
    cu_cnt = 0
    cu_rejected = 0
    for root, dirs, files in os.walk("/kaggle/input"):
        if "dataset_annotations.csv" in files:
            csv_path = os.path.join(root, "dataset_annotations.csv")
            try:
                # Robust CSV read
                df_u = pd.read_csv(csv_path, encoding='utf-8-sig', sep=None, engine='python')
                df_u.columns = [c.strip().lower() for c in df_u.columns]
                
                col_lbl = next((c for c in df_u.columns if 'label' in c or 'chord' in c), None)
                col_start = next((c for c in df_u.columns if 'start' in c), None)
                col_end = next((c for c in df_u.columns if 'end' in c), None)
                
                # Fallback
                if not col_lbl and len(df_u.columns) >= 3:
                    col_start, col_end, col_lbl = df_u.columns[0], df_u.columns[1], df_u.columns[2]

                for wav in ["dataset_clean.wav", "dataset_eob.wav"]:
                    if wav in files:
                        w_path = os.path.join(root, wav)
                        for _, row in df_u.iterrows():
                            try:
                                raw_lbl = str(row[col_lbl])
                                r, q = split_chord_label(raw_lbl)
                                if r in ROOT_TO_IDX and q in QUAL_TO_IDX:
                                    data.append({"filename": w_path, "start": float(row[col_start]), "end": float(row[col_end]), "root": r, "quality": q, "source": "Custom"})
                                    cu_cnt += 1
                                else:
                                    cu_rejected += 1
                            except: pass
            except: pass

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV Ready: {len(df)} rows. (GuitarSet: {gs_cnt}, Custom: {cu_cnt}, Rejected: {cu_rejected})")
    
    if cu_cnt == 0:
        print("⚠️ OSTRZEŻENIE: Nie załadowano żadnych Twoich próbek! Sprawdź parser.")

generate_csv()

# ==========================================
# 4. DATASET & CACHE
# ==========================================
print("\n🔥 CACHING CQT...")
CQT_CACHE = {}

def precompute_file(filepath):
    if filepath in CQT_CACHE: return
    try:
        y, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        cqt = librosa.cqt(y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz(MIN_NOTE), n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        chroma = librosa.feature.chroma_cqt(C=np.abs(cqt), sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_chroma=12, bins_per_octave=BINS_PER_OCTAVE)
        combined = np.vstack([np.clip((cqt_db + 80.0)/80.0, 0, 1), chroma]).T.astype(np.float32)
        CQT_CACHE[filepath] = combined
    except: pass

full_df = pd.read_csv(OUTPUT_CSV).fillna("")
if len(full_df) == 0: sys.exit("❌ Pusty CSV")

for f in tqdm(full_df['filename'].unique()): precompute_file(f)

class RamDataset(Dataset):
    def __init__(self, df, source, augmentor=None):
        self.indices = []
        self.augmentor = augmentor
        subset = df[df['source'] == source]
        stride = 2 if source == "Custom" else 8
        for _, row in subset.iterrows():
            if row['filename'] not in CQT_CACHE: continue
            f_start = int(row['start'] * SAMPLE_RATE / HOP_LENGTH)
            f_end = int(row['end'] * SAMPLE_RATE / HOP_LENGTH)
            total = CQT_CACHE[row['filename']].shape[0]
            if f_end > total: f_end = total
            if f_end - f_start < CTX_FRAMES: continue
            for s in range(f_start, f_end - CTX_FRAMES, stride):
                self.indices.append((row['filename'], s, ROOT_TO_IDX[row['root']], QUAL_TO_IDX[row['quality']]))
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        fname, start, r, q = self.indices[i]
        spec = CQT_CACHE[fname][start:start+CTX_FRAMES, :].copy()
        if self.augmentor and random.random() < 0.5:
            spec[:, random.randint(0, 180):random.randint(0, 180)+10] = 0.0
            t0 = random.randint(0, 28)
            spec[t0:t0+4, :] = 0.0
        return torch.tensor(spec), torch.tensor(r), torch.tensor(q)

class AudioAugmentor:
    def __init__(self, p_heavy=0.5): self.p = p_heavy
    def apply(self, audio, sr):
        if random.random() < 0.5: audio *= random.uniform(0.5, 1.5)
        if random.random() < 0.5: audio += np.random.normal(0, 0.005, audio.shape).astype(np.float32)
        if random.random() < 0.4: # Varispeed
            try: audio = scipy.signal.resample(audio, int(len(audio)/random.uniform(0.95, 1.05)))
            except: pass
        if random.random() < 0.4: audio = np.tanh(audio * random.uniform(1.2, 3.0))
        return audio

ds_gs = RamDataset(full_df, "GuitarSet", AudioAugmentor(0.8))
ds_cu = RamDataset(full_df, "Custom", AudioAugmentor(0.2))
full_ds = ConcatDataset([ds_gs, ds_cu])

train_len = int(0.9 * len(full_ds))
tr, te = random_split(full_ds, [train_len, len(full_ds)-train_len])
train_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(te, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

# ==========================================
# 5. MODEL V10 (SE + FOCAL)
# ==========================================
device = torch.device("cuda")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        return (self.alpha * (1 - pt) ** self.gamma * self.ce(inputs, targets)).mean()

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel//reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel//reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1).expand_as(x)

class ConvBlockSE(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.gn = nn.GroupNorm(8, out_c)
        self.relu = nn.ReLU()
        self.se = SEBlock(out_c)
        self.pool = nn.MaxPool2d((1, 2)) 
        self.drop = nn.Dropout2d(0.1)
    def forward(self, x): return self.drop(self.pool(self.se(self.relu(self.gn(self.conv(x))))))

class TransformerV10(nn.Module):
    def __init__(self):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(1, affine=True)
        self.layer1 = ConvBlockSE(1, 32)
        self.layer2 = ConvBlockSE(32, 64)
        self.layer3 = ConvBlockSE(64, 128)
        self.layer4 = ConvBlockSE(128, 256)
        self.proj = nn.Linear(256 * 12, 256)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        self.pos_encoder = nn.Parameter(torch.randn(1, CTX_FRAMES + 1, 256))
        enc = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True, dropout=0.2, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=3)
        self.fc_root = nn.Linear(256, len(ROOTS))
        self.fc_qual = nn.Linear(256, len(QUALS))
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.inorm(x) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).reshape(b, t, c*f)
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_encoder
        x = self.transformer(x)
        return self.fc_root(x[:, 0]), self.fc_qual(x[:, 0])

model = TransformerV10().to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

crit_root = nn.CrossEntropyLoss(label_smoothing=0.1)
crit_qual = FocalLoss(gamma=3.0) 
opt = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, verbose=True, factor=0.5)

print("\n🚀 START TRENINGU V10 (FIXED PARSER)...")
best_loss = 999.0

for ep in range(EPOCHS):
    model.train()
    loss_sum = 0
    loop = tqdm(train_loader, desc=f"Ep {ep+1}", leave=False)
    for specs, roots, quals in loop:
        specs, roots, quals = specs.to(device), roots.to(device), quals.to(device)
        opt.zero_grad()
        out_r, out_q = model(specs)
        loss = 0.5 * crit_root(out_r, roots) + 3.0 * crit_qual(out_q, quals)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item()
        loop.set_postfix(loss=loss.item())
    
    model.eval()
    val_loss = 0
    corr_r, corr_q, tot = 0, 0, 0
    with torch.no_grad():
        for specs, roots, quals in test_loader:
            specs, roots, quals = specs.to(device), roots.to(device), quals.to(device)
            out_r, out_q = model(specs)
            loss = 0.5 * crit_root(out_r, roots) + 3.0 * crit_qual(out_q, quals)
            val_loss += loss.item()
            _, pr = torch.max(out_r, 1)
            _, pq = torch.max(out_q, 1)
            corr_r += (pr == roots).sum().item()
            corr_q += (pq == quals).sum().item()
            tot += roots.size(0)
    avg_loss = val_loss / len(test_loader)
    print(f"Ep {ep+1}: Loss {avg_loss:.4f} | Root: {100*corr_r/tot:.2f}% | Qual: {100*corr_q/tot:.2f}%")
    sched.step(avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dummy = torch.randn(1, CTX_FRAMES, 204).to(device)
                m = model.module if isinstance(model, nn.DataParallel) else model
                torch.onnx.export(m, dummy, "chord_model_v10_final.onnx", input_names=['in'], output_names=['out_root', 'out_qual'], opset_version=14, dynamic_axes={'in':{0:'b'}, 'out_root':{0:'b'}, 'out_qual':{0:'b'}})
            print("💾 Saved")
        except: pass
