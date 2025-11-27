// src/audio.rs
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use std::thread;
use std::time::Duration;
use std::process;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex};
use anyhow::Result;

// --- KONFIGURACJA CRNN ---
pub const LOG_BINS: usize = 216;   // 36 binów * 6 oktaw
pub const CTX_FRAMES: usize = 16;  // Kontekst czasowy

const F_MIN: f32 = 65.4; // C2
const BINS_PER_OCTAVE: f32 = 36.0;

// Ustawienia czułości
const DEFAULT_INPUT_GAIN: f32 = 4.0;
const NOISE_GATE_THRESHOLD: f32 = 0.0005;
const GLOBAL_BIN_OFFSET: isize = 0; // Model z augmentacją powinien stroić sam

pub struct AudioAnalysis {
    // Płaski wektor [16 * 216] dla AI
    pub raw_input_for_ai: Vec<f32>,
    pub spectrum_visual: [f32; 48],
    pub chroma_sum: [f32; 12],
    
    pub bass_boost_enabled: bool,
    pub bass_boost_gain: f32,
    pub input_gain: f32,
    pub noise_gate: f32,
}

// Wewnętrzny procesor z pamięcią historii
struct AudioProcessor {
    planner: FftPlanner<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    // Historia: Kolejka wektorów (każdy to jedna klatka widma)
    history: VecDeque<Vec<f32>>, 
}

impl AudioProcessor {
    fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(4096); // Wysoka rozdzielczość FFT
        
        let mut history = VecDeque::with_capacity(CTX_FRAMES);
        for _ in 0..CTX_FRAMES {
            history.push_back(vec![0.0; LOG_BINS]);
        }

        Self { planner, fft, history }
    }

    fn process_frame(&mut self, chunk: &[f32], state: &Arc<Mutex<AudioAnalysis>>, sample_rate: u32, is_file: bool) {
        let (ui_gain, gate) = {
            let s = state.lock().unwrap();
            // Dla plików wymuszamy 1.0, dla mikrofonu bierzemy suwak
            let g = if is_file { 1.0 } else { if s.input_gain < 0.1 { DEFAULT_INPUT_GAIN } else { s.input_gain } };
            (g, s.noise_gate)
        };

        // 1. FFT
        let mut buffer: Vec<Complex<f32>> = chunk.iter()
            .map(|&s| Complex { re: s * ui_gain, im: 0.0 })
            .collect();
        
        self.fft.process(&mut buffer);
        
        // Magnituda (połowa widma)
        let fft_mags: Vec<f32> = buffer.iter().take(buffer.len()/2).map(|c| c.norm()).collect();
        
        // Squelch (Bramka szumów)
        let avg = fft_mags.iter().sum::<f32>() / fft_mags.len() as f32;
        
        let mut current_frame = vec![0.0; LOG_BINS];
        let mut is_silence = false;

        if avg < gate {
            is_silence = true;
        } else {
            // 2. Log Mapping (36 bins/octave)
            let hz_per_bin = sample_rate as f32 / buffer.len() as f32;
            
            for i in 0..LOG_BINS {
                let center_freq = F_MIN * (2.0f32).powf(i as f32 / BINS_PER_OCTAVE);
                let idx = (center_freq / hz_per_bin).round() as usize;
                
                if idx < fft_mags.len() {
                    let mut val = fft_mags[idx];
                    // Wygładzanie (zbieranie energii z sąsiadów)
                    if idx > 0 { val += fft_mags[idx-1] * 0.5; }
                    if idx < fft_mags.len()-1 { val += fft_mags[idx+1] * 0.5; }
                    
                    // Ewentualny offset, jeśli bardzo trzeba
                    let target_idx = i as isize + GLOBAL_BIN_OFFSET;
                    if target_idx >= 0 && target_idx < LOG_BINS as isize {
                        current_frame[target_idx as usize] = val;
                    }
                }
            }

            // Log & Norm (Log1p)
            for x in &mut current_frame { *x = (1.0 + *x).ln(); }
            let max = current_frame.iter().fold(0.0f32, |a, &b| a.max(b));
            if max > 1e-6 { for x in &mut current_frame { *x /= max; } }
        }

        // 3. Aktualizacja Historii
        self.history.pop_front();
        self.history.push_back(current_frame.clone());

        // 4. Eksport do Stanu (Flattening)
        if let Ok(mut s) = state.lock() {
            // Spłaszczamy historię [16 klatek x 216 binów] do jednego długiego wektora
            let mut flat = Vec::with_capacity(LOG_BINS * CTX_FRAMES);
            for frame in &self.history {
                flat.extend_from_slice(frame);
            }
            s.raw_input_for_ai = flat;

            // UI Visuals (Zbijamy 216 -> 48 dla wykresu)
            if is_silence {
                for x in &mut s.spectrum_visual { *x *= 0.7; }
                for x in &mut s.chroma_sum { *x *= 0.8; }
            } else {
                for i in 0..48 {
                    // Bierzemy co 3 bin (36/12 = 3)
                    let src_idx = i * 3; 
                    if src_idx < LOG_BINS {
                        let val = current_frame[src_idx];
                        s.spectrum_visual[i] = s.spectrum_visual[i] * 0.5 + val * 0.5;
                        
                        // Chroma (i % 12)
                        let chroma_idx = i % 12;
                        s.chroma_sum[chroma_idx] = s.chroma_sum[chroma_idx] * 0.8 + val * 0.2;
                    }
                }
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

    println!("--- AUDIO CRNN (Live) ---");
    println!("Input: {:?} ({}Hz)", device.name().unwrap_or("?".into()), sample_rate);

    let mut processor = AudioProcessor::new();
    let mut input_buffer: Vec<f32> = Vec::with_capacity(4096);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            // Stereo Mix
            if channels == 2 {
                for f in data.chunks(2) { input_buffer.push((f[0]+f[1])*0.5); }
            } else {
                input_buffer.extend_from_slice(data);
            }

            // Okno 4096
            while input_buffer.len() >= 4096 {
                let chunk: Vec<f32> = input_buffer.drain(0..4096).collect();
                processor.process_frame(&chunk, &shared_state, sample_rate, false);
            }
        },
        |err| eprintln!("Err: {}", err),
        None 
    )?;

    stream.play()?;
    Ok(stream)
}

pub fn start_file_playback(path: String, shared_state: Arc<Mutex<AudioAnalysis>>) -> Result<()> {
    println!("--- AUDIO CRNN (File) ---");
    println!("File: {}", path);
    
    if let Ok(mut s) = shared_state.lock() { s.input_gain = 1.0; }

    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    // Load
    let samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader.into_samples::<i16>().map(|s| s.unwrap_or(0) as f32 / 32768.0).collect()
    } else {
        reader.into_samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
    };

    // Mix Mono
    let mut mono = Vec::new();
    if channels == 2 {
        for f in samples.chunks(2) { mono.push((f[0]+f[1])*0.5); }
    } else {
        mono = samples;
    }

    thread::spawn(move || {
        let mut processor = AudioProcessor::new();
        let chunk_size = 4096;
        let delay = Duration::from_secs_f32(chunk_size as f32 / sample_rate as f32);

        thread::sleep(Duration::from_millis(500));

        for chunk in mono.chunks(chunk_size) {
            if chunk.len() < chunk_size { break; }
            processor.process_frame(chunk, &shared_state, sample_rate, true);
            thread::sleep(delay);
        }
        
        println!("KONIEC PLIKU");
        process::exit(0);
    });
    Ok(())
}
