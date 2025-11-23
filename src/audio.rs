// src/audio.rs
use std::sync::{Arc, Mutex};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex};
use anyhow::Result;

pub struct AudioAnalysis {
    pub chroma_sum: [f32; 12],   
    pub spectrum_48: [f32; 48],  
    pub bass_boost_enabled: bool,
    pub bass_boost_gain: f32,
}

pub fn start_audio_stream(shared_state: Arc<Mutex<AudioAnalysis>>) -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;
        
    let config: cpal::StreamConfig = device.default_input_config()?.into();

    let mut planner = FftPlanner::new();
    let fft_len = 16384; 
    let fft = planner.plan_fft_forward(fft_len);
    let mut input_buffer: Vec<f32> = Vec::with_capacity(fft_len);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            input_buffer.extend_from_slice(data);
            while input_buffer.len() >= fft_len {
                let chunk: Vec<f32> = input_buffer.drain(0..fft_len).collect();
                process_audio_chunk(&chunk, &fft, &shared_state, config.sample_rate.0);
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
    let (boost_enabled, boost_gain) = {
        let s = state.lock().unwrap();
        (s.bass_boost_enabled, s.bass_boost_gain)
    };

    let windowed_data: Vec<Complex<f32>> = data.iter().enumerate()
        .map(|(i, &sample)| {
            let multiplier = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (data.len() - 1) as f32).cos());
            Complex { re: sample * multiplier, im: 0.0 }
        })
        .collect();

    let mut buffer = windowed_data;
    fft.process(&mut buffer);

    let bin_width = sample_rate as f32 / data.len() as f32;
    let mut local_spec48 = [0.0; 48];

    let useful_limit = (1600.0 / bin_width) as usize; 

    // 1. Budowanie surowego spektrum 48 binów
    for (i, bin) in buffer.iter().enumerate().take(useful_limit) {
        let mut magnitude = bin.norm();
        let frequency = i as f32 * bin_width;

        if frequency > 50.0 {
            if boost_enabled {
                if frequency < 90.0 { magnitude *= 1.0; } 
                else if frequency < 150.0 { magnitude *= boost_gain; } 
                else if frequency < 250.0 { magnitude *= boost_gain * 0.6; }
            }

            // Logarytmiczna kompresja dynamiki
            let scaled_val = (1.0 + magnitude * 0.1).ln();

            let midi_float = 12.0 * (frequency / 440.0).log2() + 69.0;
            let midi_note = midi_float.round() as i32;
            
            if midi_note >= 40 && midi_note < (40 + 48) {
                let spec_idx = (midi_note - 40) as usize;
                local_spec48[spec_idx] += scaled_val;
            }
        }
    }

    // --- HARMONIC CLEANUP (WYCINANIE DUCHÓW) ---
    // To jest kluczowe dla gitary. Iterujemy od dołu (basu).
    // Jeśli znajdziemy silny ton podstawowy, osłabiamy jego harmoniczne wyżej.
    
    // Kopia robocza do czytania (żeby nie modyfikować tego, co czytamy w pętli)
    let read_spec = local_spec48; 

    for i in 0..36 { // Nie sprawdzamy samej góry
        let root_energy = read_spec[i];
        
        // Jeśli w tym kubełku jest znacząca energia...
        if root_energy > 0.5 {
            // 1. Tłumimy II harmoniczną (Oktawa: +12 półtonów)
            // Gitara często ma głośniejszą oktawę niż bas, więc tłumimy ostrożnie.
            if i + 12 < 48 {
                local_spec48[i + 12] *= 0.6; 
            }

            // 2. Tłumimy III harmoniczną (Kwinta + Oktawa: ~+19 półtonów)
            // To jest główny winowajca mylenia nuty z Power Chordem. Tłumimy mocno.
            if i + 19 < 48 {
                local_spec48[i + 19] *= 0.4; 
            }

            // 3. Tłumimy V harmoniczną (Tercja + 2 Oktawy: ~+28 półtonów)
            // To sprawia, że pojedyncza nuta brzmi jak Dur. Tłumimy bardzo mocno.
            if i + 28 < 48 {
                local_spec48[i + 28] *= 0.3;
            }
        }
    }

    // 2. Budowanie Chromy (zwijanie 48 -> 12) PO wyczyszczeniu
    let mut local_chroma = [0.0; 12];
    for i in 0..48 {
        let chroma_idx = (i + 40) % 12; // +40 bo zaczynamy od MIDI 40 (E)
        // E (40) % 12 = 4. E=4 w naszej mapie (C=0). Zgadza się.
        local_chroma[chroma_idx] += local_spec48[i];
    }

    if let Ok(mut s) = state.lock() {
        // Update Chroma (Decay 0.85)
        for i in 0..12 {
            let current = s.chroma_sum[i];
            let target = local_chroma[i];
            if target > current { s.chroma_sum[i] = current * 0.3 + target * 0.7; }
            else { s.chroma_sum[i] = current * 0.85; }
        }
        
        // Update Spectrum (Decay 0.85) - też dodajemy mały decay dla stabilności AI
        for i in 0..48 {
            let current = s.spectrum_48[i];
            let target = local_spec48[i];
            if target > current { s.spectrum_48[i] = current * 0.3 + target * 0.7; }
            else { s.spectrum_48[i] = current * 0.85; }
        }
    }
}
