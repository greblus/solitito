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
    let mut local_chroma = [0.0; 12];
    let mut local_spec48 = [0.0; 48];

    let useful_limit = (1600.0 / bin_width) as usize; 

    for (i, bin) in buffer.iter().enumerate().take(useful_limit) {
        let mut magnitude = bin.norm();
        let frequency = i as f32 * bin_width;

        if frequency > 50.0 {
            if boost_enabled {
                if frequency < 90.0 { magnitude *= 1.0; } // Sub-bas bez zmian
                else if frequency < 150.0 { magnitude *= boost_gain; } 
                else if frequency < 250.0 { magnitude *= boost_gain * 0.6; }
            }

            // UPROSZCZONA MATEMATYKA:
            // Używamy logarytmu naturalnego, żeby spłaszczyć piki, ale wyciągnąć ciche dźwięki.
            // Dodajemy 1.0, żeby log(0) nie wybuchł.
            // Mnożnik 0.1 dopasowuje skalę do typowych wartości FFT.
            let scaled_val = (1.0 + magnitude * 0.1).ln();

            let midi_float = 12.0 * (frequency / 440.0).log2() + 69.0;
            let midi_note = midi_float.round() as i32;
            
            if midi_note > 0 {
                let chroma_idx = (midi_note as usize) % 12;
                local_chroma[chroma_idx] += scaled_val;

                if midi_note >= 40 && midi_note < (40 + 48) {
                    let spec_idx = (midi_note - 40) as usize;
                    local_spec48[spec_idx] += scaled_val;
                }
            }
        }
    }

    if let Ok(mut s) = state.lock() {
        // Decay ustawiony na 0.85 - dobry kompromis między stabilnością a szybkością
        for i in 0..12 {
            let current = s.chroma_sum[i];
            let target = local_chroma[i];
            if target > current { s.chroma_sum[i] = current * 0.3 + target * 0.7; }
            else { s.chroma_sum[i] = current * 0.85; }
        }
        
        for i in 0..48 {
            let current = s.spectrum_48[i];
            let target = local_spec48[i];
            if target > current { s.spectrum_48[i] = current * 0.3 + target * 0.7; }
            else { s.spectrum_48[i] = current * 0.85; }
        }
    }
}
