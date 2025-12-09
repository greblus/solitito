import numpy as np
import librosa
import json
import warnings

# ==========================================
# KONFIGURACJA V15 (High Fidelity)
# ==========================================
SR = 22050          
MIN_NOTE = 'C1'
N_BINS = 192        
BINS_PER_OCTAVE = 24 
RUST_FFT_SIZE = 8192 

# Używamy 0.85, żeby filtry były zwarte w czasie
FILTER_SCALE = 0.85 

print(f"🔧 Generowanie V15: SR={SR}, FFT={RUST_FFT_SIZE}, Scale={FILTER_SCALE}")

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fmin = librosa.note_to_hz(MIN_NOTE)
        
        raw_result = librosa.filters.constant_q(
            sr=SR, fmin=fmin, n_bins=N_BINS, 
            bins_per_octave=BINS_PER_OCTAVE,
            filter_scale=FILTER_SCALE, pad_fft=False
        )
        
        basis = raw_result[0] if isinstance(raw_result, tuple) else raw_result
        if isinstance(basis, list): basis = basis[0]

        n_fft_bins = RUST_FFT_SIZE // 2 + 1
        cqt_kernel = np.zeros((N_BINS, n_fft_bins), dtype=np.complex64)
        
        for i, filter_kernel in enumerate(basis):
            fk = np.array(filter_kernel).flatten() if not hasattr(filter_kernel, "toarray") else filter_kernel.toarray().flatten()
            k_len = len(fk)
            
            padded = np.zeros(RUST_FFT_SIZE, dtype=np.complex64)
            if k_len > RUST_FFT_SIZE:
                start = (k_len - RUST_FFT_SIZE) // 2
                padded[:] = fk[start : start + RUST_FFT_SIZE]
            else:
                start = (RUST_FFT_SIZE - k_len) // 2
                padded[start : start + k_len] = fk
            
            full_fft = np.fft.fft(padded)
            cqt_kernel[i, :] = np.conj(full_fft[:n_fft_bins])

        cqt_T = cqt_kernel.T 
        norm_factor = np.max(np.abs(cqt_T))
        if norm_factor > 0: cqt_T /= norm_factor

        chroma_matrix = np.zeros((N_BINS, 12), dtype=np.float32)
        for i in range(N_BINS):
            semitone = int((i / BINS_PER_OCTAVE) * 12) % 12
            chroma_matrix[i, semitone] = 1.0
        chroma_matrix /= (N_BINS / 12)

        data = {
            "fft_size": RUST_FFT_SIZE,
            "sr": SR,
            "cqt_weights_re": cqt_T.real.flatten().tolist(),
            "cqt_weights_im": cqt_T.imag.flatten().tolist(),
            "chroma_weights": chroma_matrix.flatten().tolist()
        }

        with open("dsp_weights.json", "w") as f: json.dump(data, f)

    print(f"✅ SUKCES V15.")

except Exception as e: print(f"❌ BŁĄD: {e}")
