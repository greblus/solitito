// src/audio.rs
use std::sync::{Arc, Mutex};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex};
use anyhow::Result;

pub struct AudioAnalysis {
    pub chroma_energy: [f32; 12],
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

    let mut local_chroma = [0.0; 12];
    let bin_width = sample_rate as f32 / data.len() as f32;

    let useful_limit = (1100.0 / bin_width) as usize; 
    let mut sum = 0.0;
    for bin in buffer.iter().take(useful_limit) { sum += bin.norm(); }
    let avg = sum / useful_limit as f32;

    for (i, bin) in buffer.iter().enumerate().take(useful_limit) {
        let mut magnitude = bin.norm();
        let frequency = i as f32 * bin_width;

        if frequency > 55.0 {
            if boost_enabled {
                if frequency < 100.0 {
                    magnitude *= boost_gain; 
                } else if frequency < 150.0 {
                    magnitude *= boost_gain * 0.5; 
                }
            }

            if magnitude > avg * 2.0 {
                let scaled_val = magnitude / 1000.0;
                let midi_note = (12.0 * (frequency / 440.0).log2() + 69.0).round();
                if midi_note > 0.0 {
                    let index = (midi_note as usize) % 12;
                    local_chroma[index] += scaled_val;
                }
            }
        }
    }

    let active_bins = local_chroma.iter().filter(|&&v| v > 0.5).count();
    if active_bins > 8 {
        if let Ok(mut s) = state.lock() {
            for i in 0..12 { s.chroma_energy[i] *= 0.8; } 
        }
        return;
    }

    if let Ok(mut s) = state.lock() {
        for i in 0..12 {
            let current = s.chroma_energy[i];
            let target = local_chroma[i];

            if target > current {
                // Atak (szybki)
                s.chroma_energy[i] = current * 0.4 + target * 0.6;
            } else {
                // ZMIANA: Bardzo szybki decay (0.6). 
                // Dzięki temu po wytłumieniu struny ręką, wartość spadnie momentalnie.
                s.chroma_energy[i] = current * 0.60; 
            }
        }
    }
}
