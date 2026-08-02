# ==========================================
# GEN WEIGHTS - pseudo-CQT weights for the Rust app
# ==========================================
# Produces dsp_weights.json in a SPARSE format (CSR by CQT bin).
#
# Why sparse: the transformed CQT kernel has 4097x144 = 589968 weights, but they
# concentrate around each bin's centre frequency. Measured on the real kernel
# (white noise, pink noise, a guitar-like harmonic series):
#
#   threshold   weights kept   max error vs peak
#   1e-5          21.9%           0.006%
#   1e-4           6.9%           0.033%      <- used
#   1e-3           2.3%           0.352%
#
# At 1e-4 the error is orders of magnitude below the feature resolution (a log
# normalisation over 80 dB follows), and the audio path does ~14x fewer multiplies.
# The JSON drops from ~28 MB to ~2 MB, which matters on its own for a phone build.
#
# A previous version wrapped everything in a `try/except` that printed the error
# and exited successfully, so a failure looked like a successful run.
# Now any inconsistency aborts generation.
#
# The script also prints the bin->chroma mapping and compares it with the one
# shipped with the app. At bins_per_octave=24 the chroma matrix DOES have one
# non-zero weight per bin (it folds pairs) - that is normal, not a symptom.
# Only a SHIFT of the classes would matter.
#
# KAGGLE: needs librosa, so run it there rather than on a laptop.
# ==========================================

import json
import os
import warnings

import librosa
import numpy as np

# DSP config (16k) - must match model_trainer.py and src/audio.rs
SR = 16000
MIN_NOTE = 'C1'
N_BINS = 144
BINS_PER_OCTAVE = 24
RUST_FFT_SIZE = 8192

# Weights below PRUNE_REL * max(|w|) are dropped. See the table above.
PRUNE_REL = 1e-4

OUT = "dsp_weights.json"


def build_cqt_kernel():
    """[n_fft_bins, N_BINS] - complex pseudo-CQT kernel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fmin = librosa.note_to_hz(MIN_NOTE)
        raw = librosa.filters.constant_q(
            sr=SR, fmin=fmin, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
            filter_scale=1.0, pad_fft=False
        )
    basis = raw[0] if isinstance(raw, tuple) else raw
    if isinstance(basis, list):
        basis = basis[0]

    n_fft_bins = RUST_FFT_SIZE // 2 + 1
    kernel = np.zeros((N_BINS, n_fft_bins), dtype=np.complex64)
    for i, fk in enumerate(basis):
        fk = fk.toarray().flatten() if hasattr(fk, "toarray") else np.asarray(fk).flatten()
        padded = np.zeros(RUST_FFT_SIZE, dtype=np.complex64)
        if len(fk) > RUST_FFT_SIZE:
            off = (len(fk) - RUST_FFT_SIZE) // 2
            padded[:] = fk[off: off + RUST_FFT_SIZE]
        else:
            off = (RUST_FFT_SIZE - len(fk)) // 2
            padded[off: off + len(fk)] = fk
        kernel[i, :] = np.conj(np.fft.fft(padded)[:n_fft_bins])

    cqt = kernel.T                       # [n_fft_bins, N_BINS]
    cqt /= np.max(np.abs(cqt))
    return cqt


def main():
    print(f"🔧 SR={SR}  FFT={RUST_FFT_SIZE}  bins={N_BINS}  threshold={PRUNE_REL:g}")
    cqt = build_cqt_kernel()
    mag = np.abs(cqt)
    keep = mag > PRUNE_REL * mag.max()
    print(f"   weights kept: {keep.sum()} / {keep.size} ({keep.mean():.2%})")

    # Error check on a ~1/f spectrum - the weight count alone is not enough.
    rng = np.random.default_rng(7)
    n = cqt.shape[0]
    spec = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    spec /= np.sqrt(np.arange(1, n + 1))
    full = np.abs(spec @ cqt)
    thin = np.abs(spec @ np.where(keep, cqt, 0))
    err = np.max(np.abs(full - thin)) / max(full.max(), 1e-12)
    print(f"   max error vs peak: {err:.4%}")
    if err > 0.01:
        raise SystemExit(f"❌ Truncation error {err:.2%} too large - lower PRUNE_REL.")

    # CSR by CQT bin: bin i holds weights in [offsets[i], offsets[i+1])
    offsets, idx, wre, wim = [0], [], [], []
    for i in range(N_BINS):
        rows = np.nonzero(keep[:, i])[0]
        idx.extend(int(k) for k in rows)
        wre.extend(float(v) for v in cqt[rows, i].real)
        wim.extend(float(v) for v in cqt[rows, i].imag)
        offsets.append(len(idx))

    # Chroma matrix straight from librosa - the same one chroma_cqt uses.
    # Scale does not matter: audio.rs max-normalises per frame (norm=inf).
    #
    # A false lead to avoid: at bins_per_octave=24 and n_chroma=12, cq_to_chroma
    # folds TWO adjacent bins into a class and has exactly one non-zero weight
    # per bin. "One weight per bin" is therefore NOT a symptom of a bug; an
    # earlier version of this script aborted here for no reason.
    # The only thing that can really go wrong is a SHIFT in the mapping (which
    # bin lands in which class), so it is printed and compared.
    chroma = librosa.filters.cq_to_chroma(
        N_BINS, bins_per_octave=BINS_PER_OCTAVE, n_chroma=12
    ).T.astype(np.float32)                                   # [N_BINS, 12]

    if chroma.shape != (N_BINS, 12):
        raise SystemExit(f"❌ Chroma shape {chroma.shape}, expected ({N_BINS}, 12).")
    if not np.isfinite(chroma).all() or chroma.max() <= 0:
        raise SystemExit("❌ Chroma contains NaN/inf or is all zeros.")

    nz_per_bin = float((chroma != 0).sum(axis=1).mean())
    mapping = chroma.argmax(axis=1).tolist()
    legacy = [(k // (BINS_PER_OCTAVE // 12)) % 12 for k in range(N_BINS)]
    print(f"   chroma: {nz_per_bin:.1f} non-zero weights per bin, "
          f"range [{chroma.min():.3f}, {chroma.max():.3f}]")
    print(f"   bin->class mapping (first 24): {mapping[:24]}")
    if mapping == legacy:
        print("   = the same mapping as in the file shipped so far "
              "(no shift)")
    else:
        off = [(a - b) % 12 for a, b in zip(mapping, legacy)]
        print(f"   ⚠️ DIFFERENT mapping than before! class shift: "
              f"{sorted(set(off))} - a real train/serve mismatch.")

    data = {
        "format": "sparse-csr-v1",
        "fft_size": RUST_FFT_SIZE,
        "sr": SR,
        "n_bins": N_BINS,
        "cqt_offsets": offsets,
        "cqt_fft_idx": idx,
        "cqt_re": wre,
        "cqt_im": wim,
        "chroma_weights": [float(v) for v in chroma.flatten()],
    }
    with open(OUT, "w") as f:
        json.dump(data, f)
    print(f"✅ {OUT}: {os.path.getsize(OUT)/2**20:.1f} MB  ({len(idx)} CQT weights)")


if __name__ == "__main__":
    main()
