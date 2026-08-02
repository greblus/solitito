# ==========================================
# VERIFY ANNOTATIONS - do the labels describe what is in the audio?
# ==========================================
# Four checks, in order of increasing strength:
#   1. label histogram (a single dominant label shows up immediately),
#   2. REAL block period from the energy envelope, versus the annotated one,
#   3. offset: does the annotated window land on sound or on silence,
#   4. AGREEMENT: dominant pitch class in the window versus the labelled root.
#
# Checks 2 and 4 are complementary. Shifting the annotations by a whole block
# still lands on *some* attack, so the timing check will not see it - only the
# label comparison catches that.
#
# Loads slices only (offset/duration), so it is fast. CPU, no GPU.
# numpy is enough: its own WAV reader plus an FFT chroma. librosa is used when
# available because chroma_cqt is more accurate.
#
# KAGGLE: paste into a cell and call main(), or `!python verify_annotations.py`.
# ==========================================

import os
import re
import sys
import warnings
from collections import Counter, defaultdict

import numpy as np

try:
    import pandas as pd
except ImportError:
    import subprocess
    subprocess.call([sys.executable, "-m", "pip", "install", "pandas", "--quiet"])
    import pandas as pd

# librosa is OPTIONAL. On newer Pythons (3.13+) numba is often not ready and
# librosa fails - and numpy alone is enough to check label-audio agreement.
try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False


# ==========================================
# AUDIO READING WITHOUT LIBROSA (stdlib + numpy): PCM 16/24/32, float 32/64, mono/stereo
# ==========================================
def read_wav(path, offset=0.0, duration=None):
    import struct
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF": raise ValueError("not RIFF")
        f.read(4)
        if f.read(4) != b"WAVE": raise ValueError("not WAVE")
        fmt = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8: raise ValueError("no data chunk")
            cid, csz = struct.unpack("<4sI", hdr)
            if cid == b"fmt ":
                d = f.read(csz)
                afmt, ch, sr, _, _, bits = struct.unpack("<HHIIHH", d[:16])
                if afmt == 0xFFFE and len(d) >= 26:
                    afmt = struct.unpack("<H", d[24:26])[0]
                fmt = (afmt, ch, sr, bits)
            elif cid == b"data":
                afmt, ch, sr, bits = fmt
                bps = bits // 8
                skip = int(offset * sr) * ch * bps
                f.seek(skip, 1)
                avail = csz - skip
                nb = avail if duration is None else min(avail, int(duration * sr) * ch * bps)
                raw = f.read(max(0, nb))
                break
            else:
                f.seek(csz + (csz & 1), 1)

    if   afmt == 3 and bits == 32: a = np.frombuffer(raw[:len(raw)//4*4], "<f4").astype(np.float32)
    elif afmt == 3 and bits == 64: a = np.frombuffer(raw[:len(raw)//8*8], "<f8").astype(np.float32)
    elif afmt == 1 and bits == 16: a = np.frombuffer(raw[:len(raw)//2*2], "<i2").astype(np.float32)/32768
    elif afmt == 1 and bits == 32: a = np.frombuffer(raw[:len(raw)//4*4], "<i4").astype(np.float32)/2147483648
    elif afmt == 1 and bits == 24:
        b = np.frombuffer(raw[:len(raw)//3*3], np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        a = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float32) / 8388608
    else:
        raise ValueError(f"unsupported WAV (format={afmt}, {bits}-bit)")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def chroma_of_samples(x, sr):
    """Simple FFT chroma: energy summed per pitch class in the fundamental band.
    Enough to tell whether the labelled root is among the dominant notes."""
    if len(x) < 2048: return None
    if HAVE_LIBROSA:
        C = np.abs(librosa.cqt(x, sr=sr, hop_length=HOP_LENGTH,
                               fmin=librosa.note_to_hz(MIN_NOTE),
                               n_bins=N_BINS, bins_per_octave=BPO))
        return librosa.feature.chroma_cqt(C=C, sr=sr, hop_length=HOP_LENGTH,
                                          n_chroma=12, bins_per_octave=BPO).mean(1)
    n = 1 << int(np.ceil(np.log2(len(x))))
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)).astype(np.float32), n))
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    m = (fr > 65.0) & (fr < 1600.0)                 # guitar fundamental band
    f, mag = fr[m], X[m]
    if len(f) == 0: return None
    pc = (np.round(69 + 12 * np.log2(f / 440.0)).astype(int)) % 12
    ch = np.zeros(12)
    np.add.at(ch, pc, mag)
    return ch

warnings.filterwarnings("ignore")

SR          = 16000
HOP_LENGTH  = 256
MIN_NOTE    = 'C1'
N_BINS      = 144
BPO         = 24
INPUT_DIR   = "/kaggle/input"

SEGS_PER_FILE = 60          # segments sampled for the agreement test
PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NORM_MAP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def find_all(pattern_ext):
    out = []
    for base in [".", INPUT_DIR, "/kaggle/working"]:
        if not os.path.isdir(base): continue
        for r, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(pattern_ext):
                    out.append(os.path.join(r, f))
    return out


def label_root(lbl):
    """Root from a label ('C', 'E 7', 'Note C', 'A# m7b5') -> pitch class index."""
    t = str(lbl).strip()
    m = re.match(r"^(?:note\s+)?([A-G][#b]?)", t, re.IGNORECASE)
    if not m: return None
    r = m.group(1).upper().replace("B#", "B#")
    r = r[0].upper() + (r[1:] if len(r) > 1 else "")
    r = NORM_MAP.get(r, r)
    return PITCHES.index(r) if r in PITCHES else None


def block_period(path, probe_sec=240.0):
    """Real block period from the energy envelope (median gap between attacks)."""
    try:
        y, sr = read_wav(path, duration=probe_sec)
    except Exception as e:
        print(f"   ❌ Cannot read '{os.path.basename(path)}': {type(e).__name__}: {e}")
        return None, None
    if len(y) < sr * 10: return None, None
    hop = max(1, sr // 100)                                   # 10 ms windows
    nfr = len(y) // hop
    e = np.sqrt((y[:nfr*hop].reshape(nfr, hop) ** 2).mean(axis=1))
    if e.max() <= 0: return None, None
    # Schmitt trigger: high threshold to arm, low to release. Without it the
    # rippling envelope of a decaying chord produces many false attacks and the
    # block period comes out too short.
    hi, lo = 0.35 * e.max(), 0.08 * e.max()
    rises_idx, armed = [], True
    for i, v in enumerate(e):
        if armed and v > hi:
            rises_idx.append(i); armed = False
        elif not armed and v < lo:
            armed = True
    rises = np.array(rises_idx, dtype=float) * hop / sr
    if len(rises) < 3: return None, None
    gaps = np.diff(rises)
    gaps = gaps[gaps > 1.0]                 # ignore ripples within one block
    return (float(np.median(gaps)) if len(gaps) else None), rises


def main():
    csvs = [p for p in find_all(".csv") if "annotation" in os.path.basename(p).lower()]
    wavs = find_all(".wav")
    if not csvs: sys.exit("❌ No *annotations*.csv found")
    print(f"🔍 CSV: {[os.path.basename(c) for c in csvs]}")
    print(f"🔍 WAV: {len(wavs)} files\n")

    for csv_path in csvs:
        df = pd.read_csv(csv_path, sep=None, engine='python')
        cols = [str(c).strip().lower() for c in df.columns]
        df.columns = cols
        c_f = next((c for c in cols if 'file' in c or 'audio' in c or c == 'id'), None)
        c_l = next((c for c in cols if 'label' in c or 'chord' in c), None)
        c_s = next((c for c in cols if 'start' in c), None)
        c_e = next((c for c in cols if 'end' in c), None)
        print("=" * 84)
        print(f"CSV: {os.path.basename(csv_path)}  | kolumny: {cols}  | wierszy: {len(df)}")
        print("=" * 84)
        if not (c_l and c_s and c_e):
            print("  ⚠️ brak kolumn start/end/label — pomijam\n"); continue

        groups = df.groupby(c_f) if c_f else [("(brak kolumny file)", df)]
        for gid, g in groups:
            gid_s = str(gid).strip()
            labs = [str(x) for x in g[c_l]]
            hist = Counter(labs).most_common(6)
            top_lbl, top_n = hist[0]
            dom = 100.0 * top_n / len(labs)
            starts = np.sort(g[c_s].astype(float).values)
            ann_period = float(np.median(np.diff(starts))) if len(starts) > 2 else float('nan')

            print(f"\n── file/ID '{gid_s}'  ({len(g)} segments)")
            print("   etykiety: " + "  ".join(f"{l}×{n}" for l, n in hist))
            print(f"   dominacja jednej etykiety: {dom:.0f}%"
                  + ("   ⚠️ SUSPICIOUS (the decoder keeps returning the same ID)" if dom > 40 else ""))
            print(f"   block period per annotations: {ann_period:.2f}s")

            # match a wav to this ID (e.g. '04' -> 04_notes_clean.wav)
            cand = [w for w in wavs
                    if re.match(rf"^0*{re.escape(gid_s.lstrip('0') or '0')}[_.-]", os.path.basename(w))
                    and "_clean" in os.path.basename(w).lower()]
            if not cand:
                cand = [w for w in wavs if os.path.basename(w).startswith(gid_s)]
            if not cand:
                print("   (no matching wav found - skipping the audio test)"); continue
            wav = cand[0]
            print(f"   wav: {os.path.basename(wav)}")

            per, rises = block_period(wav)
            if per:
                print(f"   block period per AUDIO:      {per:.2f}s", end="")
                if not np.isnan(ann_period) and abs(per - ann_period) > 0.3:
                    print(f"   ⚠️ MISMATCH with the annotations ({ann_period:.2f}s)")
                else:
                    print("   ✓")
                if len(rises):
                    # offset: annotated start versus the nearest attack in audio
                    offs = []
                    for s in starts[:40]:
                        if s > rises[-1]: break
                        offs.append(s - rises[np.argmin(np.abs(rises - s))])
                    if offs:
                        mo = float(np.median(offs))
                        print(f"   start-to-attack offset:  {mo:+.2f}s"
                              + ("   ⚠️ the annotation MISSES the sound" if abs(mo) > 0.5 else "   ✓"))

            # --- AGREEMENT: dominant pitch in the window vs the labelled root ---
            idx = np.linspace(0, len(g) - 1, min(SEGS_PER_FILE, len(g))).astype(int)
            sub = g.iloc[idx]
            t1 = t3 = n = 0; energies = []; load_err = None; parse_skip = 0
            for _, row in sub.iterrows():
                r_lbl = label_root(row[c_l])
                if r_lbl is None:
                    parse_skip += 1; continue
                st, en = float(row[c_s]), float(row[c_e])
                dur = max(0.5, min(en - st, 3.0))
                try:
                    y, sr_w = read_wav(wav, offset=st, duration=dur)
                except Exception as e:
                    load_err = f"{type(e).__name__}: {e}"; break
                if len(y) < sr_w * 0.3: continue
                energies.append(float(np.sqrt(np.mean(y ** 2))))
                ch = chroma_of_samples(y, sr_w)
                if ch is None: continue
                order = np.argsort(ch)[::-1]
                n += 1
                if order[0] == r_lbl: t1 += 1
                if r_lbl in order[:3]: t3 += 1
            if load_err:
                print(f"   ❌ COULD NOT LOAD AUDIO: {load_err}")
                print("      Verification did NOT run - do not read this as 'OK'.")
            elif parse_skip and n == 0:
                print(f"   ❌ No label could be parsed ({parse_skip} attempts) - "
                      f"check the label column format.")
            elif n == 0:
                print("   ❌ Zero windows checked (segments too short?) - no verification.")
            if n:
                p1, p3 = 100.0 * t1 / n, 100.0 * t3 / n
                rms = float(np.median(energies)) if energies else 0.0
                print(f"   label-audio AGREEMENT (n={n}):  top1={p1:.0f}%   top3={p3:.0f}%"
                      f"   | median RMS={rms:.4f}")
                # Thresholds depend on the method. In a chord the root is NOT always
                # the loudest, so with the simple chroma top3 is the meaningful figure
                # (on known-good data: top3=100%, top1~39%). Chance: top1~8%, top3~25%.
                if HAVE_LIBROSA:
                    good, weak = p1 >= 45, p1 >= 20
                else:
                    good, weak = (p3 >= 75 and p1 >= 20), p3 >= 45
                if good:
                    print("      ✓ the annotations match the audio")
                elif weak:
                    print("      ⚠️ WEAK - partially misaligned")
                else:
                    print("      ❌ CHANCE LEVEL - the annotations do NOT describe this audio")
                if rms < 0.005:
                    print("      ❌ the annotated windows are nearly SILENT (wrong time offset)")
        print()

    print("=" * 84)
    if HAVE_LIBROSA:
        print("Metoda: librosa chroma_cqt.  OK = top1 >45%.  Losowo = top1 ~8%.")
    else:
        print("Method: simple FFT chroma (librosa unavailable).")
        print("  OK     = top3 >75%  (on correct data top3~100%, top1~39%)")
        print("  Chance = top3 ~25%, top1 ~8%  -> labels do not describe the audio, DO NOT TRAIN")
        print("  Note: with this method top1 is inherently low for chords -")
        print("         the root is often quieter than the third/fifth. Read top3.")
    print("On a timing mismatch, check the render offset calibration:")
    print("   python dataset_generator_v2.py --calibrate <wav>")


if __name__ == "__main__":
    main()
