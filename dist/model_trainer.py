import os
import sys
import subprocess
import random
import json
import glob
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import soundfile as sf
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import warnings
import shutil
import gc

# ==========================================
# 1. KONFIGURACJA (V31 - MATCHING RUST)
# ==========================================
SR = 16000          # 16k (zgodnie z Rust)
HOP_LENGTH = 256    # 256 (zgodnie z Rust)
MIN_NOTE = 'C1'

# STRUKTURA WEJŚCIA:
N_BINS = 144        # 6 oktaw (144 biny)
CHROMA_BINS = 12
INPUT_FEATURES = 156 # 144 + 12

BINS_PER_OCTAVE = 24
CTX_FRAMES = 32     
BATCH_SIZE = 64     
EPOCHS = 60         

WORK_DIR = "./workspace"
GUITARSET_DIR = os.path.join(WORK_DIR, "guitarset")
CUSTOM_DATA_DIR = "./custom_data" 
MODELS_DIR = "./models"
CACHE_DIR = "./temp_cache_v31"

for d in [WORK_DIR, GUITARSET_DIR, CUSTOM_DATA_DIR, MODELS_DIR]: os.makedirs(d, exist_ok=True)
if os.path.exists(CACHE_DIR): shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Noise"]
QUALS = ["", "m", "7", "Maj7", "m7", "dim7", "m7b5", "9", "13", "Note"] 
ROOT_TO_IDX = {r: i for i, r in enumerate(ROOTS)}
QUAL_TO_IDX = {q: i for i, q in enumerate(QUALS)}
NOTE_MAP = {"Db":"C#", "Eb":"D#", "Gb":"F#", "Ab":"G#", "Bb":"A#"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")
print(f"🎯 Config: SR={SR}, Bins={N_BINS}, Input={INPUT_FEATURES}")

# ==========================================
# 2. INSTALACJA
# ==========================================
def install_libs():
    subprocess.call([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime", "librosa", "soundfile", "tqdm", "requests"], stdout=subprocess.DEVNULL)
try: import librosa
except: install_libs(); import librosa

def download_file(url, destination):
    if os.path.exists(destination) and os.path.getsize(destination) > 10240: return
    try:
        import requests
        response = requests.get(url, stream=True)
        with open(destination, 'wb') as file:
            for data in response.iter_content(1024*1024): file.write(data)
    except: pass

def setup_guitarset():
    if not os.path.exists(os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix")):
        download_file("https://zenodo.org/records/3371780/files/audio_mono-pickup_mix.zip", os.path.join(GUITARSET_DIR, "audio.zip"))
        try:
            import zipfile
            with zipfile.ZipFile(os.path.join(GUITARSET_DIR, "audio.zip"), 'r') as z: z.extractall(GUITARSET_DIR)
        except: pass
    if not os.path.exists(os.path.join(GUITARSET_DIR, "annotation")):
        download_file("https://zenodo.org/records/3371780/files/annotation.zip", os.path.join(GUITARSET_DIR, "jams.zip"))
        try:
            import zipfile
            with zipfile.ZipFile(os.path.join(GUITARSET_DIR, "jams.zip"), 'r') as z: z.extractall(GUITARSET_DIR)
        except: pass

# ==========================================
# 3. DSP (STANDARD LIBROSA - 16k)
# ==========================================
def process_audio(audio_path, augment=False):
    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        if len(y) < HOP_LENGTH * CTX_FRAMES: return None

        # Augmentacja (tylko szum)
        if augment:
            if random.random() < 0.4:
                y = y + 0.002 * np.random.randn(len(y))
        
        # CQT (16k / 144 Bins)
        cqt = librosa.cqt(y, sr=SR, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz(MIN_NOTE), 
                          n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, filter_scale=1.0)
        cqt_mag = np.abs(cqt)
        
        # Log & Norm
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
        norm = (cqt_db + 80.0) / 80.0
        norm = np.clip(norm, 0, 1)
        
        # Chroma
        chroma = librosa.feature.chroma_cqt(C=norm, sr=SR, hop_length=HOP_LENGTH, 
                                            n_chroma=12, bins_per_octave=BINS_PER_OCTAVE)
        
        # Stack [Time, 156]
        return np.vstack([norm, chroma]).T.astype(np.float32)
    except: return None

# ==========================================
# 4. DATASET & PARSER
# ==========================================
def split_chord_label(chord_str):
    if not isinstance(chord_str, str): return None, None
    chord_str = chord_str.strip()
    
    if chord_str in ["N", "Noise"]: return "Noise", ""
    
    # Obsługa "Note A#"
    lower = chord_str.lower()
    if lower.startswith("note ") or lower.endswith(" note"):
        clean = lower.replace("note", "").strip().capitalize()
        root = NOTE_MAP.get(clean, clean)
        return root, "Note"

    match = re.match(r"^([A-G][#b]?)\s*(.*)$", chord_str)
    if not match: 
        # JAMS Fallback
        if ":" in chord_str:
            p = chord_str.split(":")
            r = NOTE_MAP.get(p[0], p[0])
            q_raw = p[1].split("/")[0].split("(")[0]
            if "maj7" in q_raw: return r, "Maj7"
            elif "min7" in q_raw: return r, "m7"
            elif "7" in q_raw: return r, "7"
            elif "maj" in q_raw: return r, ""
            elif "min" in q_raw: return r, "m"
            return r, ""
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

def load_guitarset_data():
    data = []
    print("🔍 GS...")
    audio_files = glob.glob(os.path.join(GUITARSET_DIR, "**", "*.wav"), recursive=True)
    audio_map = {os.path.basename(f).replace("_mic.wav", "").replace("_mix.wav", ""): f for f in audio_files}
    jams_files = glob.glob(os.path.join(GUITARSET_DIR, "**", "*.jams"), recursive=True)
    for j_path in tqdm(jams_files):
        stem = os.path.basename(j_path).replace(".jams", "")
        audio_path = None
        for k, v in audio_map.items():
            if stem in k: audio_path = v; break
        if not audio_path: continue
        try:
            with open(j_path, 'r') as f: content = json.load(f)
            for ann in content["annotations"]:
                if ann["namespace"] == "chord":
                    for obs in ann["data"]:
                        r, q = split_chord_label(obs["value"])
                        if r in ROOTS and q in QUALS:
                            data.append({"path": audio_path, "start": obs["time"], "end": obs["time"]+obs["duration"], 
                                         "root": ROOT_TO_IDX[r], "qual": QUAL_TO_IDX[q]})
                        elif obs["value"] == "N":
                             data.append({"path": audio_path, "start": obs["time"], "end": obs["time"]+obs["duration"], 
                                          "root": ROOT_TO_IDX["Noise"], "qual": QUAL_TO_IDX[""]})
        except: pass
    return pd.DataFrame(data)

def load_custom_data(root_dir):
    data = []
    print("🔍 Custom...")
    target_wavs = ["dataset_clean.wav", "dataset_eob.wav"]
    for root, dirs, files in os.walk(root_dir):
        if "dataset_annotations.csv" in files:
            csv_path = os.path.join(root, "dataset_annotations.csv")
            try:
                df_raw = pd.read_csv(csv_path, sep=None, engine='python')
                cols = [c.lower() for c in df_raw.columns]; df_raw.columns = cols
                col_lbl = next((c for c in cols if 'label' in c or 'chord' in c), None)
                col_start = next((c for c in cols if 'start' in c), None)
                col_end = next((c for c in cols if 'end' in c), None)
                col_file = next((c for c in cols if 'file' in c or 'audio' in c), None)
                if not col_lbl: continue
                local_wavs = [f for f in files if f.endswith(".wav")]
                found_targets = [t for t in target_wavs if t in local_wavs]
                for _, row in df_raw.iterrows():
                    r, q = split_chord_label(str(row[col_lbl]))
                    if r not in ROOTS or q not in QUALS: continue
                    start = float(row[col_start]) if col_start else 0.0
                    end = float(row[col_end]) if col_end else 10.0
                    files_proc = []
                    if col_file and str(row[col_file]) in local_wavs: files_proc.append(str(row[col_file]))
                    elif found_targets: files_proc = found_targets
                    else: files_proc = local_wavs
                    for fname in files_proc:
                        data.append({"path": os.path.join(root, fname), "start": start, "end": end,
                                     "root": ROOT_TO_IDX[r], "qual": QUAL_TO_IDX[q]})
            except: pass
    return pd.DataFrame(data)

class DiskCacheDataset(Dataset):
    def __init__(self, df):
        self.indices = []
        print(f"📦 Budowanie datasetu...")
        grouped = df.groupby("path")
        for path, group in tqdm(grouped):
            feats = process_audio(path, augment=True)
            if feats is None: continue
            
            cache_name = f"track_{abs(hash(path))}.npy"
            cache_path = os.path.join(CACHE_DIR, cache_name)
            np.save(cache_path, feats)
            n_total = feats.shape[0]
            del feats
            
            for _, row in group.iterrows():
                s = int(row['start'] * SR / HOP_LENGTH)
                e = int(row['end'] * SR / HOP_LENGTH)
                if e - s > CTX_FRAMES:
                    for i in range(s, e - CTX_FRAMES, 6):
                        if i + CTX_FRAMES <= n_total:
                            self.indices.append((cache_path, i, row['root'], row['qual']))
        
        # Silence Injection (Input = 156)
        print("🤫 Generowanie ciszy...")
        silence_feats = np.random.rand(5000, INPUT_FEATURES).astype(np.float32) * 0.02
        silence_path = os.path.join(CACHE_DIR, "silence.npy")
        np.save(silence_path, silence_feats)
        for i in range(0, 5000 - CTX_FRAMES, 32):
            self.indices.append((silence_path, i, ROOT_TO_IDX["Noise"], QUAL_TO_IDX[""]))
            
        gc.collect()

    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        path, start, r, q = self.indices[idx]
        data_mmap = np.load(path, mmap_mode='r')
        x = data_mmap[start : start+CTX_FRAMES].copy()
        return torch.tensor(x, dtype=torch.float32), torch.tensor(r, dtype=torch.long), torch.tensor(q, dtype=torch.long)

# ==========================================
# 5. MODEL (156 Input)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets); pt = torch.exp(logpt)
        return (self.alpha * (1-pt)**self.gamma * self.ce(inputs, targets)).mean()

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel//16, bias=False), nn.ReLU(True), nn.Linear(channel//16, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size(); y = self.avg_pool(x).view(b, c); y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlockSE(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.gn = nn.GroupNorm(8, out_c); self.relu = nn.ReLU(); self.se = SEBlock(out_c)
        self.pool = nn.MaxPool2d((1, 2)); self.drop = nn.Dropout2d(0.1)
    def forward(self, x): return self.drop(self.pool(self.se(self.relu(self.gn(self.conv(x))))))

class TransformerV10(nn.Module):
    def __init__(self):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(1, affine=True)
        self.enc = nn.Sequential(ConvBlockSE(1, 32), ConvBlockSE(32, 64), ConvBlockSE(64, 128), ConvBlockSE(128, 256))
        
        # OBLICZENIE SPŁASZCZENIA DLA 156 CECH
        # 156 -> pool(2) -> 78 -> pool(2) -> 39 -> pool(2) -> 19 -> pool(2) -> 9
        # 9 * 256 = 2304
        self.proj = nn.Linear(2304, 256) 
        
        self.cls = nn.Parameter(torch.randn(1, 1, 256))
        self.pos = nn.Parameter(torch.randn(1, CTX_FRAMES + 1, 256))
        self.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 512, 0.2, batch_first=True, norm_first=True), 3)
        self.fc_r = nn.Linear(256, len(ROOTS)); self.fc_q = nn.Linear(256, len(QUALS))

    def forward(self, x):
        x = self.enc(self.inorm(x.unsqueeze(1)))
        b, c, t, f = x.size()
        x = self.proj(x.permute(0, 2, 1, 3).reshape(b, t, c*f))
        x = torch.cat((self.cls.expand(b, -1, -1), x), 1) + self.pos
        x = self.tr(x)[:, 0]
        return self.fc_r(x), self.fc_q(x)

# ==========================================
# 6. MAIN
# ==========================================
if __name__ == "__main__":
    install_libs()
    setup_guitarset()
    
    df_gs = load_guitarset_data()
    df_custom = load_custom_data(CUSTOM_DATA_DIR)
    df_kaggle = load_custom_data("/kaggle/input")
    
    df_final = pd.concat([df_gs, df_custom, df_kaggle], ignore_index=True).drop_duplicates(subset=["path", "start"])
    print(f"📊 Dataset: {len(df_final)} regionów.")
    if len(df_final)==0: sys.exit("Brak danych.")

    ds = DiskCacheDataset(df_final)
    tr_len = int(0.9 * len(ds))
    tr, te = random_split(ds, [tr_len, len(ds)-tr_len])
    
    tr_l = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    te_l = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False)
    
    model = TransformerV10().to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit_r, crit_q = FocalLoss(gamma=1.5), FocalLoss(gamma=2.0)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, verbose=True)

    best_v = float('inf')
    print("\n🔥 START TRENINGU (V31 - 16k Clean)...")
    
    for ep in range(EPOCHS):
        model.train(); l_sum = 0
        loop = tqdm(tr_l, desc=f"Ep {ep+1}")
        for x, r, q in loop:
            try:
                x,r,q = x.to(device), r.to(device), q.to(device)
                opt.zero_grad(); or_, oq_ = model(x)
                loss = crit_r(or_, r) + crit_q(oq_, q)
                loss.backward(); opt.step()
                l_sum += loss.item(); loop.set_postfix(loss=loss.item())
            except RuntimeError as e: print(e); continue
        
        model.eval(); v_loss, cr, cq, tot = 0, 0, 0, 0
        with torch.no_grad():
            for x, r, q in te_l:
                try:
                    x,r,q = x.to(device), r.to(device), q.to(device)
                    or_, oq_ = model(x)
                    v_loss += (crit_r(or_, r) + crit_q(oq_, q)).item()
                    cr += (or_.argmax(1)==r).sum().item(); cq += (oq_.argmax(1)==q).sum().item()
                    tot += r.size(0)
                except: pass
        
        if tot > 0:
            vl = v_loss/len(te_l)
            print(f"📉 Val: {vl:.4f} | R: {cr/tot:.2%} | Q: {cq/tot:.2%}")
            sched.step(vl)
            if vl < best_v:
                best_v = vl
                m_save = model.module if hasattr(model, 'module') else model
                torch.onnx.export(m_save, torch.randn(1, CTX_FRAMES, 156).to(device), 
                                "chord_model_v31_16k.onnx", input_names=["in"], output_names=["out_root", "out_qual"],
                                dynamic_axes={"in":{0:"b"}, "out_root":{0:"b"}, "out_qual":{0:"b"}}, opset_version=14)
                print("💾 Saved.")
    
    shutil.rmtree(CACHE_DIR)
