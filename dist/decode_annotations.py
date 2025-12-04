import soundfile as sf
import scipy.signal
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

# --- KONFIGURACJA ---
BEEPS_WAV = "beeps.wav"                
REF_CSV = "dataset_reference.csv"      
FINAL_CSV = "dataset_annotations.csv"  

BPM = 120
BEAT_DUR = 60 / BPM          # 0.5s
SIXTEENTH_DUR = BEAT_DUR / 4 # 0.125s (125ms)

# Struktura: [Pilot (Takt 1)] ... [Akord (Takt 3)]
OFFSET_TO_CHORD = 4.0 
CHORD_DURATION = 4.0 

def find_file(name):
    search_paths = [".", "/kaggle/input", "/content"]
    for path in search_paths:
        for root, dirs, files in os.walk(path):
            if name in files: return os.path.join(root, name)
    return None

BEEPS_PATH = find_file(BEEPS_WAV)
REF_PATH = find_file(REF_CSV)

if not BEEPS_PATH or not REF_PATH:
    sys.exit("❌ Brak plików!")

print(f"🚀 Dekoder Barcode (Z NORMALIZACJĄ)...")

# 1. Wczytanie
df_ref = pd.read_csv(REF_PATH)
id_map = {row['id']: row['label'] for _, row in df_ref.iterrows()}

audio, sr = sf.read(BEEPS_PATH)
if len(audio.shape) > 1: audio = np.mean(audio, axis=1) # Mono

# --- KLUCZOWA POPRAWKA: NORMALIZACJA ---
print("📊 Normalizowanie głośności...")
max_val = np.max(np.abs(audio))
if max_val > 0:
    audio = audio / max_val
    print(f"   Sygnał wzmocniony (Max był: {max_val:.4f})")
else:
    sys.exit("❌ Plik audio jest całkowicie pusty (cisza)!")

amplitude = np.abs(audio)

# 2. Szukanie Startów
print("🔍 Skanowanie impulsów...")
# Szukamy pików > 0.5 (skoro max to 1.0, to 0.5 jest bezpiecznym środkiem)
peaks, _ = scipy.signal.find_peaks(amplitude, height=0.5, distance=int(0.05 * sr))

valid_starts = []
last_time = -999.0

for p in peaks:
    t = p / sr
    # Szukamy przerwy > 3s (oznacza początek nowego taktu z kodem)
    if (t - last_time) > 3.0: 
        valid_starts.append(p)
    last_time = t

print(f"✅ Znaleziono sekwencji kodowych: {len(valid_starts)}")

# 3. Dekodowanie
print("🔓 Dekodowanie...")

csv_rows = []
success = 0
fail = 0

# Ile błędów wypisać w konsoli (żeby nie spamować)
debug_errors_left = 5

for start_samp in tqdm(valid_starts):
    bits = []
    
    # Skanujemy 12 okienek (bity danych)
    for i in range(1, 13):
        center_samp = start_samp + int(i * SIXTEENTH_DUR * sr)
        
        # Okno +/- 20ms
        win = int(0.02 * sr)
        s = max(0, center_samp - win)
        e = min(len(amplitude), center_samp + win)
        
        # Czy w tym oknie był JAKIKOLWIEK silny sygnał?
        # Używamy np.max zamiast np.mean, bo pisk może być krótki
        if np.max(amplitude[s:e]) > 0.5:
            bits.append('1')
        else:
            bits.append('0')
            
    bin_str = "".join(bits)
    try:
        decoded_id = int(bin_str, 2)
    except:
        decoded_id = -1
        
    if decoded_id in id_map:
        label = id_map[decoded_id]
        
        # Matematyka czasu (zgodna z generatorem GP5)
        t_pilot = start_samp / sr
        chord_start = t_pilot + OFFSET_TO_CHORD
        chord_end = chord_start + CHORD_DURATION
        
        csv_rows.append([f"{chord_start:.3f}", f"{chord_end - 0.5:.3f}", label])
        success += 1
    else:
        fail += 1
        if debug_errors_left > 0:
            print(f"\n❌ BŁĄD w {start_samp/sr:.1f}s: Odczytano binarnie: '{bin_str}' -> ID: {decoded_id}")
            print(f"   (Tego ID nie ma w CSV reference).")
            debug_errors_left -= 1

# 4. Zapis
df_final = pd.DataFrame(csv_rows, columns=["start", "end", "label"])
df_final.to_csv(FINAL_CSV, index=False)

print("\n" + "="*40)
print(f"WYNIK: {success} Dobrych / {fail} Złych")
print("="*40)

if success == 0:
    print("FATAL ERROR: Nadal nic. Sprawdź:")
    print("1. Czy 'beeps.wav' to na pewno ten plik z piskami?")
    print("2. Czy 'dataset_reference.csv' pasuje do tego generowania?")
    print("3. Czy BPM w skrypcie (120) zgadza się z GP5?")
else:
    print(f"✅ Plik gotowy: {FINAL_CSV}")
    print("Możesz trenować!")
