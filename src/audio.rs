// src/audio.rs
use std::sync::{Arc, Mutex};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex};
use anyhow::Result;

// Stałe zgodne z Pythonem V4
pub const LOG_BINS: usize = 128;
const F_MIN: f32 = 32.7; // C1
const BINS_PER_OCTAVE: f32 = 12.0;

pub struct AudioAnalysis {
    pub chroma_sum: [f32; 12],
    pub spectrum_visual: [f32; 48],
    pub raw_input_for_ai: [f32; LOG_BINS], // 128 Log Bins
    pub bass_boost_enabled: bool,
    pub bass_boost_gain: f32,
}

pub fn start_audio_stream(shared_state: Arc<Mutex<AudioAnalysis>>) -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or_else(|| anyhow::anyhow!("No input device found"))?;
    let config: cpal::StreamConfig = device.default_input_config()?.into();

    let mut planner = FftPlanner::new();
    let fft_len = 4096; // Duże FFT
    let fft = planner.plan_fft_forward(fft_len);
    let mut input_buffer: Vec<f32> = Vec::with_capacity(fft_len);

    // Pobieramy sample rate urządzenia (np. 44100 lub 48000)
    let sample_rate = config.sample_rate.0;

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            input_buffer.extend_from_slice(data);
            while input_buffer.len() >= fft_len {
                let chunk: Vec<f32> = input_buffer.drain(0..fft_len).collect();
                process_audio_chunk(&chunk, &fft, &shared_state, sample_rate);
            }
        },
        |err| eprintln!("Stream error: {}", err),
        None 
    )?;

    stream.play()?;
    Ok(stream)
}

fn process_audio_chunk(
    data: &[f32], 
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    state: &Arc<Mutex<AudioAnalysis>>,
    sample_rate: u32
) {
    let mut buffer: Vec<Complex<f32>> = data.iter().enumerate()
        .map(|(i, &sample)| {
            let win = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (data.len() - 1) as f32).cos());
            Complex { re: sample * win, im: 0.0 }
        })
        .collect();

    fft.process(&mut buffer);

    // Obliczamy Magnitudę FFT (tylko pierwsza połowa)
    let fft_size = buffer.len();
    let fft_mags: Vec<f32> = buffer.iter().take(fft_size/2).map(|c| c.norm()).collect();

    let mut ai_features = [0.0; LOG_BINS];
    let mut ui_features = [0.0; 48];

    // --- LOG MAPPING (Klucz do sukcesu) ---
    // Mapujemy 128 binów logarytmicznych na liniowe biny FFT
    let hz_per_bin = sample_rate as f32 / fft_size as f32;

    for i in 0..LOG_BINS {
        // Wzór: freq = f_min * 2^(i / 12)
        let center_freq = F_MIN * (2.0f32).powf(i as f32 / BINS_PER_OCTAVE);
        
        // Który to bin w FFT?
        let fft_idx = (center_freq / hz_per_bin).round() as usize;
        
        if fft_idx < fft_mags.len() {
            // Pobieramy energię (z lekkim rozmyciem jak w Pythonie)
            let mut val = fft_mags[fft_idx];
            if fft_idx > 0 { val += fft_mags[fft_idx-1] * 0.5; }
            if fft_idx < fft_mags.len() - 1 { val += fft_mags[fft_idx+1] * 0.5; }
            
            ai_features[i] = val;
        }
    }

    // Logarytm amplitudy (Log1p)
    for i in 0..LOG_BINS {
        ai_features[i] = (1.0 + ai_features[i]).ln();
    }
    
    // Normalizacja AI
    let max_val = ai_features.iter().fold(0.0f32, |a, &b| a.max(b));
    if max_val > 0.00001 {
        for x in &mut ai_features { *x /= max_val; }
    }

    // UI Features (proste mapowanie)
    // ... (można użyć ai_features, bo to też 12 binów na oktawę!)
    // Od binu 0 (C1) do binu 48+ (C5)
    for i in 0..48 {
        // Mapujemy ai_features na ui (startujemy od np. binu 12 czyli C2)
        let source_idx = i + 12; 
        if source_idx < LOG_BINS {
            ui_features[i] = ai_features[source_idx];
        }
    }

    // Zapis
    if let Ok(mut s) = state.lock() {
        s.raw_input_for_ai = ai_features;
        for i in 0..48 {
            s.spectrum_visual[i] = s.spectrum_visual[i] * 0.5 + ui_features[i] * 0.5;
            let c_idx = (i + 40) % 12; // 40 to E, ale tutaj mamy C jako 0. Uprośćmy: C=0
            // ui_features[0] to C2. 
            let chroma_idx = i % 12;
            s.chroma_sum[chroma_idx] = s.chroma_sum[chroma_idx] * 0.8 + ui_features[i] * 0.2;
        }
    }
}
