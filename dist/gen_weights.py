import numpy as np
import librosa
import json
import warnings

# ==========================================
# KONFIGURACJA V31 (16k)
# ==========================================
SR = 16000
MIN_NOTE = 'C1'
N_BINS = 144
BINS_PER_OCTAVE = 24
RUST_FFT_SIZE = 8192

print(f"🔧 Generowanie wag V31: SR={SR}, FFT={RUST_FFT_SIZE}, Bins={N_BINS}")

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fmin = librosa.note_to_hz(MIN_NOTE)
        
        # Filtry CQT (Scale 1.0)
        raw_result = librosa.filters.constant_q(
            sr=SR, fmin=fmin, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
            filter_scale=1.0, pad_fft=False
        )
        basis = raw_result[0] if isinstance(raw_result, tuple) else raw_result
        if isinstance(basis, list): basis = basis[0]

        n_fft_bins = RUST_FFT_SIZE // 2 + 1
        cqt_kernel = np.zeros((N_BINS, n_fft_bins), dtype=np.complex64)
        
        for i, filter_kernel in enumerate(basis):
            fk = np.array(filter_kernel).flatten() if not hasattr(filter_kernel, "toarray") else filter_kernel.toarray().flatten()
            padded = np.zeros(RUST_FFT_SIZE, dtype=np.complex64)
            if len(fk) > RUST_FFT_SIZE:
                padded[:] = fk[(len(fk)-RUST_FFT_SIZE)//2:(len(fk)-RUST_FFT_SIZE)//2+RUST_FFT_SIZE]
            else:
                padded[(RUST_FFT_SIZE-len(fk))//2:(RUST_FFT_SIZE-len(fk))//2+len(fk)] = fk
            cqt_kernel[i, :] = np.conj(np.fft.fft(padded)[:n_fft_bins])

        cqt_T = cqt_kernel.T 
        cqt_T /= np.max(np.abs(cqt_T))

        chroma = np.zeros((N_BINS, 12), dtype=np.float32)
        for i in range(N_BINS): chroma[i, int((i / BINS_PER_OCTAVE) * 12) % 12] = 1.0
        chroma /= (N_BINS / 12)

        data = {
            "fft_size": RUST_FFT_SIZE, "sr": SR, 
            "cqt_weights_re": cqt_T.real.flatten().tolist(), 
            "cqt_weights_im": cqt_T.imag.flatten().tolist(), 
            "chroma_weights": chroma.flatten().tolist()
        }
        with open("dsp_weights.json", "w") as f: json.dump(data, f)
    print("✅ Gotowe.")
except Exception as e: print(f"❌ BŁĄD: {e}")
