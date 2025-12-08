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
import zipfile 
import gc

# ==========================================
# 1. KONFIGURACJA V13
# ==========================================
SR = 22050
HOP_LENGTH = 512
MIN_NOTE = 'C1'
N_BINS = 192        
BINS_PER_OCTAVE = 24
FILTER_SCALE = 0.85 
RUST_FFT_SIZE = 8192

CTX_FRAMES = 32     
BATCH_SIZE = 64     
EPOCHS = 60         

WORK_DIR = "./workspace"
GUITARSET_DIR = os.path.join(WORK_DIR, "guitarset")
CUSTOM_DATA_DIR = "./custom_data" 
MODELS_DIR = "./models"

for d in [WORK_DIR, GUITARSET_DIR, CUSTOM_DATA_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Noise"]
QUALS = ["", "m", "7", "Maj7", "m7", "dim7", "m7b5", "9", "13", "Note"] 
ROOT_TO_IDX = {r: i for i, r in enumerate(ROOTS)}
QUAL_TO_IDX = {q: i for i, q in enumerate(QUALS)}
NOTE_MAP = {"Db":"C#", "Eb":"D#", "Gb":"F#", "Ab":"G#", "Bb":"A#"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# ==========================================
# 2. INSTALACJA
# ==========================================
def install_libs():
    subprocess.call([sys.executable, "-m", "pip", "install", "requests", "librosa", "pandas", "tqdm", "onnx", "onnxruntime", "numpy", "torch", "soundfile"], stdout=subprocess.DEVNULL)

try: import requests
except ImportError: install_libs(); import requests

def download_file(url, destination):
    if os.path.exists(destination) and os.path.getsize(destination) > 10240: return
    try:
        response = requests.get(url, stream=True)
        with open(destination, 'wb') as file:
            for data in response.iter_content(1024*1024): file.write(data)
    except: pass

def setup_guitarset():
    URL_AUDIO = "https://zenodo.org/records/3371780/files/audio_mono-pickup_mix.zip?download=1"
    URL_JAMS = "https://zenodo.org/records/3371780/files/annotation.zip?download=1"
    
    if not os.path.exists(os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix")):
        zip_path = os.path.join(GUITARSET_DIR, "audio.zip")
        download_file(URL_AUDIO, zip_path)
        try:
            with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(GUITARSET_DIR)
        except: pass

    if not os.path.exists(os.path.join(GUITARSET_DIR, "annotation")):
        zip_path = os.path.join(GUITARSET_DIR, "jams.zip")
        download_file(URL_JAMS, zip_path)
        try:
            with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(GUITARSET_DIR)
        except: pass

# ==========================================
# 3. DSP SIMULATION V13 (FIXED GOERTZEL + SOFT SQUELCH)
# ==========================================
def rust_dsp_simulation(audio_path, augment=False):
    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        if len(y) < RUST_FFT_SIZE: return None

        if augment and random.random() < 0.6:
            steps = random.randint(-2, 2)
            if steps != 0: y = librosa.effects.pitch_shift(y, sr=SR, n_steps=steps)
            if random.random() < 0.3: y = y + 0.005 * np.random.randn(len(y))
        
        # 1. CQT
        cqt_complex = librosa.cqt(y, sr=SR, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz(MIN_NOTE), 
                                  n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, 
                                  filter_scale=FILTER_SCALE)
        cqt_mag = np.abs(cqt_complex)
        n_frames = cqt_mag.shape[1]
        
        # 2. VECTORIZED GOERTZEL (POPRAWIONE BINY!)
        # E2=82.4Hz -> Bin 32 (przy 24 b/oct, C1=0)
        # F2=87.3Hz -> Bin 34 (+2 biny = 1 półton)
        target_freqs = [82.41, 87.31, 92.50, 98.00, 103.83]
        target_indices = [32, 34, 36, 38, 40] 
        
        pad_len = RUST_FFT_SIZE // 2
        y_padded = np.pad(y, (pad_len, pad_len), mode='constant')
        y_frames = librosa.util.frame(y_padded, frame_length=RUST_FFT_SIZE, hop_length=HOP_LENGTH)
        
        min_frames = min(n_frames, y_frames.shape[1])
        cqt_mag = cqt_mag[:, :min_frames]
        y_frames = y_frames[:, :min_frames]
        
        N = RUST_FFT_SIZE
        basis_matrix = np.zeros((len(target_freqs), N), dtype=np.complex64)
        t_vec = np.arange(N)
        for i, freq in enumerate(target_freqs):
            k = np.round(freq * N / SR)
            basis_matrix[i, :] = np.exp(-2j * np.pi * k * t_vec / N)
            
        goertzel_energies = np.abs(basis_matrix @ y_frames) * 0.5 
        
        for i, idx in enumerate(target_indices):
            g_en_row = goertzel_energies[i, :]
            
            # Harmonic Check (+24 biny = 1 oktawa)
            harm_idx = idx + 24
            if harm_idx < N_BINS:
                harm_row = cqt_mag[harm_idx, :]
                boost_mask = harm_row > (g_en_row * 0.2)
                g_en_row[boost_mask] *= 1.5
            
            fft_row = cqt_mag[idx, :]
            blend_mask = g_en_row > fft_row
            cqt_mag[idx, blend_mask] = fft_row[blend_mask] * 0.3 + g_en_row[blend_mask] * 0.7

        # 3. Cleanup
        cqt_mag[0:32, :] *= 0.1 # Tłumimy wszystko poniżej E2 (bin 32)
        
        ref_val = np.max(cqt_mag)
        if ref_val < 1e-9: ref_val = 1.0
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=ref_val)
        norm = np.clip((cqt_db + 80.0) / 80.0, 0, 1)

        # 4. SOFT SQUELCH (20% & Linear)
        # To przywróci wykrywanie pojedynczych nut!
        mask = norm < 0.20
        norm[mask] = 0.0
        norm[~mask] = (norm[~mask] - 0.20) / 0.80
        # Brak potęgowania!

        # 5. Chroma
        chroma = librosa.feature.chroma_cqt(C=norm, sr=SR, hop_length=HOP_LENGTH, 
                                            n_chroma=12, bins_per_octave=BINS_PER_OCTAVE)
        
        result = np.vstack([norm, chroma]).T.astype(np.float32)
        
        if np.isnan(result).any() or np.isinf(result).any(): return None
        return result

    except Exception: return None

# ==========================================
# 4. DATA PARSERS (Full)
# ==========================================
def split_chord_label(chord_str):
    if not isinstance(chord_str, str): return None, None
    chord_str = chord_str.strip()
    if chord_str in ["N", "Noise"]: return "Noise", ""
    match = re.match(r"^([A-G][#b]?)\s*(.*)$", chord_str)
    if not match: return None, None
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

def load_guitarset_data():
    data = []
    print("🔍 GuitarSet: Parsowanie JAMS...")
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
                        lbl = obs["value"]
                        if ":" in lbl:
                            r, q = lbl.split(":")
                            q = q.split("/")[0].split("(")[0]
                            r_final, q_final = split_chord_label(f"{r} {q}")
                            if r_final in ROOTS and q_final in QUALS:
                                data.append({"path": audio_path, "start": obs["time"], "end": obs["time"]+obs["duration"], 
                                             "root": ROOT_TO_IDX[r_final], "qual": QUAL_TO_IDX[q_final]})
                        elif lbl == "N":
                             data.append({"path": audio_path, "start": obs["time"], "end": obs["time"]+obs["duration"], 
                                          "root": ROOT_TO_IDX["Noise"], "qual": QUAL_TO_IDX[""]})
        except: pass
    return pd.DataFrame(data)

def load_custom_data(root_dir):
    data = []
    print(f"🔍 Custom Data: Szukanie w {root_dir}...")
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
            except Exception: pass
    return pd.DataFrame(data)

class ChordDataset(Dataset):
    def __init__(self, df):
        self.samples = []
        print(f"📦 Budowanie datasetu ({len(df)} regionów)...")
        grouped = df.groupby("path")
        for path, group in tqdm(grouped):
            feats = rust_dsp_simulation(path, augment=True)
            if feats is None: continue
            for _, row in group.iterrows():
                s = int(row['start'] * SR / HOP_LENGTH)
                e = int(row['end'] * SR / HOP_LENGTH)
                if e - s > CTX_FRAMES:
                    for i in range(s, e - CTX_FRAMES, 6):
                        c = feats[i : i+CTX_FRAMES]
                        if c.shape[0] == CTX_FRAMES:
                            self.samples.append((c, row['root'], row['qual']))
        gc.collect()

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, r, q = self.samples[i]
        if np.isnan(x).any(): x = np.zeros_like(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(r, dtype=torch.long), torch.tensor(q, dtype=torch.long)

# ==========================================
# 5. MODEL
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
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c); y = self.fc(y).view(b, c, 1, 1)
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
        self.proj = nn.Linear(256 * 12, 256)
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

    ds = ChordDataset(df_final)
    tr_len = int(0.9 * len(ds))
    tr, te = random_split(ds, [tr_len, len(ds)-tr_len])
    
    tr_l = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    te_l = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False)
    
    model = TransformerV10().to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit_r, crit_q = nn.CrossEntropyLoss(), FocalLoss(gamma=2.0)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, verbose=True)

    best_v = float('inf')
    print("\n🔥 START TRENINGU (V13)...")
    
    for ep in range(EPOCHS):
        model.train(); l_sum = 0
        loop = tqdm(tr_l, desc=f"Ep {ep+1}")
        for x, r, q in loop:
            try:
                x,r,q = x.to(device), r.to(device), q.to(device)
                opt.zero_grad(); or_, oq_ = model(x)
                loss = crit_r(or_, r) + crit_q(oq_, q)
                if torch.isnan(loss): continue
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                l_sum += loss.item(); loop.set_postfix(loss=loss.item())
            except: continue
        
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
                torch.onnx.export(m_save, torch.randn(1, CTX_FRAMES, N_BINS + 12).to(device), 
                                "chord_model_v13_final.onnx", input_names=["in"], output_names=["out_root", "out_qual"],
                                dynamic_axes={"in":{0:"b"}, "out_root":{0:"b"}, "out_qual":{0:"b"}}, opset_version=14)
                print("💾 Saved.")
