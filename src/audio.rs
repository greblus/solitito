use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::process;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex};
use anyhow::Result;

// --- KONFIGURACJA ZGODNA Z MODELEM PYTHONOWYM ---
pub const LOG_BINS: usize = 216;   
const F_MIN: f32 = 65.4; 
const BINS_PER_OCTAVE: f32 = 36.0;

// FFT może być większe dla lepszej precyzji przy 48kHz
const FFT_SIZE: usize = 4096; 

// Parametry wizualizacji (bez zmian)
const GLOBAL_BIN_OFFSET: isize = 0; 
const DEFAULT_INPUT_GAIN: f32 = 5.0; // Podbiłem domyślny gain dla gitary

pub struct AudioAnalysis {
    pub raw_input_for_ai: [f32; LOG_BINS], 
    pub spectrum_visual: [f32; 48],
    pub chroma_sum: [f32; 12],
    pub bass_boost_enabled: bool,
    pub bass_boost_gain: f32,
    pub input_gain: f32,
    pub noise_gate: f32,
}

struct AudioProcessor {
    planner: FftPlanner<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
}

impl AudioProcessor {
    fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        
        // Okno Hanninga
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| {
                let n = i as f32;
                let size = (FFT_SIZE - 1) as f32;
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * n / size).cos())
            })
            .collect();

        Self { planner, fft, window }
    }

    fn process_frame(&mut self, chunk: &[f32], state: &Arc<Mutex<AudioAnalysis>>, sample_rate: u32, is_file: bool) {
        let (ui_gain, gate) = {
            let s = state.lock().unwrap();
            let g = if is_file { 1.0 } else { if s.input_gain < 0.1 { DEFAULT_INPUT_GAIN } else { s.input_gain } };
            (g, s.noise_gate)
        };

        // 1. Przygotowanie bufora FFT
        let mut buffer: Vec<Complex<f32>> = chunk.iter().zip(&self.window)
            .map(|(&s, &w)| Complex { re: s * ui_gain * w, im: 0.0 })
            .collect();
        
        if buffer.len() < FFT_SIZE {
            buffer.resize(FFT_SIZE, Complex { re: 0.0, im: 0.0 });
        }
        
        // 2. Wykonanie FFT
        self.fft.process(&mut buffer);
        
        // Obliczenie magnitud (tylko pierwsza połowa widma)
        let fft_mags: Vec<f32> = buffer.iter().take(FFT_SIZE/2).map(|c| c.norm()).collect();
        
        // Bramka szumów
        let avg = fft_mags.iter().sum::<f32>() / fft_mags.len() as f32;
        if avg < gate {
            if let Ok(mut s) = state.lock() {
                // Wyciszanie
                for x in &mut s.spectrum_visual { *x *= 0.8; }
                for x in &mut s.chroma_sum { *x *= 0.8; }
                for x in &mut s.raw_input_for_ai { *x = 0.0; } // Cisza dla AI
            }
            return; 
        }

        // --- MAPOWANIE DLA AI (KLUCZOWA ZMIANA) ---
        // Zamiast iterować po FFT, iterujemy po oczekiwanych binach AI (0..216)
        // i szukamy dla nich odpowiedniego prążka w FFT zależnie od Sample Rate.
        
        let mut ai_features = [0.0; LOG_BINS];
        let hz_per_bin_live = sample_rate as f32 / FFT_SIZE as f32;

        for i in 0..LOG_BINS {
            // Jaka częstotliwość odpowiada temu binowi w modelu AI?
            // Wzór z Pythona: center_freq = F_MIN * (2.0 ** (i / BINS_PER_OCTAVE))
            let center_freq = F_MIN * (2.0f32).powf(i as f32 / BINS_PER_OCTAVE);
            
            // Który to jest indeks w NASZYM aktualnym FFT (np. przy 48kHz)?
            let fft_idx = (center_freq / hz_per_bin_live).round() as usize;

            if fft_idx < fft_mags.len() {
                let mut val = fft_mags[fft_idx];
                
                // Smoothing (tak jak w Pythonie: val += prev*0.5 + next*0.5)
                // Pomaga to zbierać energię, jeśli prążek nie trafi idealnie w środek
                if fft_idx > 0 { val += fft_mags[fft_idx - 1] * 0.5; }
                if fft_idx < fft_mags.len() - 1 { val += fft_mags[fft_idx + 1] * 0.5; }
                
                ai_features[i] = val;
            }
        }

        // --- Logarytmizacja i Normalizacja (Zgodnie z Pythonem) ---
        // 1. Log1p
        for x in &mut ai_features {
            *x = (*x).ln_1p();
        }
        
        // 2. Normalize by max (per frame)
        let max_ai = ai_features.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_ai > 0.00001 {
            for x in &mut ai_features { *x /= max_ai; }
        }

        // --- WIZUALIZACJA I CHROMA (Stara logika, działa ok dla wizualiów) ---
        let mut local_chroma = [0.0; 12];
        let mut local_visual = [0.0; 48];
        
        for (i, &mag) in fft_mags.iter().enumerate().skip(1) {
            if mag < 0.001 { continue; }
            let freq = i as f32 * hz_per_bin_live;
            let midi = 12.0 * (freq / 440.0).log2() + 69.0;
            let tuned_midi = (midi.round() as isize) + GLOBAL_BIN_OFFSET;

            if tuned_midi > 0 {
                let chroma_idx = (tuned_midi % 12) as usize;
                local_chroma[chroma_idx] += mag;

                if tuned_midi >= 40 && tuned_midi < (40 + 48) {
                    let vis_idx = (tuned_midi - 40) as usize;
                    local_visual[vis_idx] += mag;
                }
            }
        }
        // Normalizacja visual
        let max_c = local_chroma.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_c > 0.0001 { for x in &mut local_chroma { *x /= max_c; } }
        let max_v = local_visual.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_v > 0.0001 { for x in &mut local_visual { *x /= max_v; } }

        // --- AKTUALIZACJA STANU ---
        if let Ok(mut s) = state.lock() {
            let alpha = if is_file { 0.8 } else { 0.6 }; 
            
            s.raw_input_for_ai = ai_features; // Kopiujemy poprawione dane
            
            for i in 0..12 {
                s.chroma_sum[i] = s.chroma_sum[i] * (1.0 - alpha) + local_chroma[i] * alpha;
            }
            for i in 0..48 {
                s.spectrum_visual[i] = s.spectrum_visual[i] * 0.6 + local_visual[i] * 0.4;
            }
        }
    }
}

pub fn start_audio_stream(shared_state: Arc<Mutex<AudioAnalysis>>) -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or_else(|| anyhow::anyhow!("Brak mikrofonu"))?;
    let config: cpal::StreamConfig = device.default_input_config()?.into();
    let sample_rate = config.sample_rate.0;
    let channels = config.channels as usize;

    println!("Input Device: {}", device.name().unwrap_or("?".into()));
    println!("Sample Rate: {} Hz", sample_rate);
    println!("FFT Size: {}", FFT_SIZE);
    
    let mut processor = AudioProcessor::new();
    let mut input_buffer: Vec<f32> = Vec::with_capacity(FFT_SIZE * 2);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            if channels == 2 {
                for f in data.chunks(2) { input_buffer.push((f[0]+f[1])*0.5); }
            } else {
                input_buffer.extend_from_slice(data);
            }

            // Przetwarzamy, gdy mamy pełny bufor
            while input_buffer.len() >= FFT_SIZE {
                // Pobieramy próbki (bez overlapa dla wydajności i prostoty synchronizacji)
                let chunk: Vec<f32> = input_buffer.drain(0..FFT_SIZE).collect();
                processor.process_frame(&chunk, &shared_state, sample_rate, false);
            }
        },
        |err| eprintln!("Audio Stream Err: {}", err),
        None 
    )?;

    stream.play()?;
    Ok(stream)
}

pub fn start_file_playback(path: String, shared_state: Arc<Mutex<AudioAnalysis>>) -> Result<()> {
    // ... (kod dla plików bez zmian, poza użyciem FFT_SIZE w buforowaniu)
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader.into_samples::<i16>().map(|s| s.unwrap_or(0) as f32 / 32768.0).collect()
    } else {
        reader.into_samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
    };

    let mut mono = Vec::new();
    if channels == 2 {
        for f in samples.chunks(2) { mono.push((f[0]+f[1])*0.5); }
    } else { mono = samples; }

    thread::spawn(move || {
        let mut processor = AudioProcessor::new();
        // Dla plików też używamy zdefiniowanego FFT_SIZE
        let chunk_size = FFT_SIZE; 
        let delay = Duration::from_secs_f32(chunk_size as f32 / sample_rate as f32);
        
        for chunk in mono.chunks(chunk_size) {
            if chunk.len() < chunk_size { break; }
            processor.process_frame(chunk, &shared_state, sample_rate, true);
            thread::sleep(delay);
        }
        process::exit(0);
    });
    Ok(())
}
