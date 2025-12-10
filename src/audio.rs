use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::BufReader;
use std::thread;
use std::time::{Duration, Instant};
use std::process;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex};
use anyhow::{Result, Context};
use serde::Deserialize;
use std::f32::consts::PI;

// --- STAŁE (ZGODNE Z V16) ---
pub const TOTAL_FEATURES: usize = 204; // 192 CQT + 12 Chroma
pub const CQT_BINS: usize = 192;       
pub const CHROMA_BINS: usize = 12;

const TARGET_SR: u32 = 22050; 
const FFT_SIZE: usize = 8192; 
const HOP_LENGTH: usize = 512; 

const MIN_REF_LEVEL: f32 = 0.0001; 

#[derive(Deserialize)]
struct DspConfig {
    fft_size: usize,
    #[allow(dead_code)] sr: u32,
    cqt_weights_re: Vec<f32>,
    cqt_weights_im: Vec<f32>,
    chroma_weights: Vec<f32>,
}

pub struct AudioAnalysis {
    pub raw_input_for_ai: [f32; TOTAL_FEATURES], 
    pub spectrum_visual: [f32; 48],
    pub chroma_sum: [f32; 12],
    pub bass_boost_enabled: bool,
    pub bass_boost_gain: f32,
    pub input_gain: f32,
    pub noise_gate: f32,
}

impl AudioAnalysis {
    pub fn copy_to_input(&mut self, data: &[f32]) {
        if data.len() == TOTAL_FEATURES {
            self.raw_input_for_ai.copy_from_slice(data);
        }
    }
}

// --- GOERTZEL FILTER ---
struct GoertzelFilter {
    coefficient: f32,
    target_idx: usize, 
}

impl GoertzelFilter {
    fn new(target_freq: f32, sr: u32, target_idx: usize) -> Self {
        // Obliczamy współczynnik k z zaokrągleniem (round) dla stabilności fazy
        let k = (target_freq * FFT_SIZE as f32 / sr as f32).round();
        let omega = (2.0 * PI * k) / FFT_SIZE as f32;
        Self { coefficient: 2.0 * omega.cos(), target_idx }
    }

    fn process(&self, samples: &[f32]) -> f32 {
        let mut q1 = 0.0;
        let mut q2 = 0.0;
        for &sample in samples {
            let q0 = self.coefficient * q1 - q2 + sample;
            q2 = q1;
            q1 = q0;
        }
        // Magnituda Goertzela
        (q1 * q1 + q2 * q2 - q1 * q2 * self.coefficient).sqrt()
    }
}

pub struct CqtAnalyzer {
    #[allow(dead_code)] planner: FftPlanner<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
    cqt_re: Vec<f32>,     
    cqt_im: Vec<f32>,     
    chroma_matrix: Vec<f32>,  
    fft_buffer: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    bass_filters: Vec<GoertzelFilter>,
    running_ref: f32,
}

impl CqtAnalyzer {
    pub fn new(json_path: &str) -> Result<Self> {
        println!("Loading DSP weights from {}...", json_path);
        let file = File::open(json_path).context("Nie znaleziono dsp_weights.json!")?;
        let reader = BufReader::new(file);
        let config: DspConfig = serde_json::from_reader(reader)?;

        if config.fft_size != FFT_SIZE {
            eprintln!("WARNING: JSON FFT Size {} != Rust FFT Size {}", config.fft_size, FFT_SIZE);
        }

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

        let scratch_len = fft.get_inplace_scratch_len();

        // --- GOERTZEL TARGETS V16 ---
        // Indeksy dla 24 bins/oct. E2 (82.4Hz) ląduje na indeksie 32.
        let mut bass_filters = Vec::new();
        bass_filters.push(GoertzelFilter::new(82.41, TARGET_SR, 32)); // E2
        bass_filters.push(GoertzelFilter::new(87.31, TARGET_SR, 34)); // F2
        bass_filters.push(GoertzelFilter::new(92.50, TARGET_SR, 36)); // F#2
        bass_filters.push(GoertzelFilter::new(98.00, TARGET_SR, 38)); // G2
        bass_filters.push(GoertzelFilter::new(103.83, TARGET_SR, 40)); // G#2

        Ok(Self { 
            planner, fft, window,
            cqt_re: config.cqt_weights_re,
            cqt_im: config.cqt_weights_im,
            chroma_matrix: config.chroma_weights,
            fft_buffer: vec![Complex{re:0.0, im:0.0}; FFT_SIZE],
            fft_scratch: vec![Complex{re:0.0, im:0.0}; scratch_len],
            bass_filters,
            running_ref: 0.05, 
        })
    }

    pub fn compute_cqt_chroma(&mut self, audio_chunk: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // 1. FFT
        for (i, &sample) in audio_chunk.iter().enumerate().take(FFT_SIZE) {
            self.fft_buffer[i] = Complex { re: sample * self.window[i], im: 0.0 };
        }
        for i in audio_chunk.len()..FFT_SIZE {
            self.fft_buffer[i] = Complex { re: 0.0, im: 0.0 };
        }
        
        self.fft.process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);
        let n_fft_bins = FFT_SIZE / 2 + 1;

        // 2. CQT Matrix (Complex)
        let mut cqt_vals = vec![0.0; CQT_BINS];
        for i in 0..CQT_BINS {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            for k in 0..n_fft_bins {
                let idx = k * CQT_BINS + i;
                let w_re = self.cqt_re[idx];
                let w_im = self.cqt_im[idx];
                if w_re.abs() > 1e-9 || w_im.abs() > 1e-9 {
                    let fft_val = self.fft_buffer[k];
                    sum_re += fft_val.re * w_re - fft_val.im * w_im;
                    sum_im += fft_val.re * w_im + fft_val.im * w_re;
                }
            }
            cqt_vals[i] = (sum_re * sum_re + sum_im * sum_im).sqrt();
        }

        // --- BASS DOCTOR V16 (Smart Hybrid) ---
        let base_scaling = 0.5;

        for filter in &self.bass_filters {
            let raw_goertzel = filter.process(audio_chunk);
            let mut goertzel_energy = raw_goertzel * base_scaling;
            
            // 1. HARMONIC PENALTY
            // Sprawdzamy oktawę wyżej (+24 biny). Jeśli harmonika jest słaba, to bas jest podejrzany.
            let harmonic_idx = filter.target_idx + 24;
            if harmonic_idx < cqt_vals.len() {
                let harmonic_energy = cqt_vals[harmonic_idx];
                // Jeśli harmonika < 10% Goertzela, tłumimy sygnał o połowę
                if harmonic_energy < goertzel_energy * 0.1 {
                    goertzel_energy *= 0.5;
                }
            }

            let fft_energy = cqt_vals[filter.target_idx];
            
            // 2. RATIO CHECK (Goertzel nie może dominować totalnie)
            let ratio = goertzel_energy / (fft_energy + 1e-6);
            
            // 3. SMART BLEND
            // Miksujemy tylko gdy Goertzel jest wyraźny (1.3x), ale nie absurdalny (3.0x)
            if ratio > 1.3 && ratio < 3.0 {
                // Zachowawczy mix: 60% FFT + 40% Goertzel
                cqt_vals[filter.target_idx] = fft_energy * 0.6 + goertzel_energy * 0.4;
            }
        }

        // Cleanup: Tłumimy sub-bas (poniżej E2, czyli indeks 32)
        for i in 0..32 { cqt_vals[i] *= 0.1; }

        // 3. AGC & SOFT SQUELCH
        let frame_max = cqt_vals.iter().fold(0.0f32, |a, &b| a.max(b));

        // Dynamika AGC
        if frame_max > self.running_ref {
            self.running_ref = self.running_ref * 0.5 + frame_max * 0.5;
        } else {
            self.running_ref = self.running_ref * 0.9995 + frame_max * 0.0005;
        }
        
        let effective_ref = self.running_ref.max(MIN_REF_LEVEL);

        for x in &mut cqt_vals {
            let val = x.max(1e-12);
            let db = 20.0 * (val / effective_ref).log10();
            let mut norm = (db + 80.0) / 80.0;
            norm = norm.clamp(0.0, 1.0);
            
            // SOFT SQUELCH 20% (LINIOWY)
            // Ucinamy szum tła, ale zachowujemy dynamikę reszty.
            // Bez potęgowania, żeby nie gubić cichych nut!
            if norm < 0.20 {
                norm = 0.0;
            } else {
                norm = (norm - 0.20) / 0.80;
            }
            *x = norm;
        }

        // 4. Chroma (z wyczyszczonego CQT)
        let mut chroma_vals = vec![0.0; CHROMA_BINS];
        for i in 0..CHROMA_BINS {
            let mut sum = 0.0;
            for k in 0..CQT_BINS {
                let weight = self.chroma_matrix[k * CHROMA_BINS + i];
                sum += cqt_vals[k] * weight;
            }
            chroma_vals[i] = sum;
        }

        let c_max = chroma_vals.iter().fold(0.0f32, |a, &b| a.max(b)).max(1e-9);
        if c_max > 0.0 { for x in &mut chroma_vals { *x /= c_max; } }

        (cqt_vals, chroma_vals)
    }
}

pub fn start_audio_stream(shared_state: Arc<Mutex<AudioAnalysis>>) -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or_else(|| anyhow::anyhow!("Brak mikrofonu"))?;
    let config: cpal::StreamConfig = device.default_input_config()?.into();
    let mic_sr = config.sample_rate.0;
    let channels = config.channels as usize;

    println!("Input: {} ({} Hz)", device.name().unwrap_or("?".into()), mic_sr);
    
    let mut analyzer = CqtAnalyzer::new("dsp_weights.json")?;
    
    let ratio = mic_sr as f32 / TARGET_SR as f32;
    let mut input_accumulator: Vec<f32> = Vec::with_capacity(8192 * 2);
    let mut processed_audio_buffer: Vec<f32> = Vec::with_capacity(FFT_SIZE * 2);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            let mut mono_chunk = Vec::with_capacity(data.len() / channels);
            if channels == 2 {
                for f in data.chunks(2) { mono_chunk.push((f[0]+f[1])*0.5); }
            } else {
                mono_chunk.extend_from_slice(data);
            }
            
            input_accumulator.extend_from_slice(&mono_chunk);
            let target_samples = (input_accumulator.len() as f32 / ratio).floor() as usize;
            
            if target_samples > 0 {
                for i in 0..target_samples {
                    let src_idx = i as f32 * ratio;
                    let idx_int = src_idx as usize;
                    let frac = src_idx - idx_int as f32;
                    
                    if idx_int + 1 < input_accumulator.len() {
                        let s0 = input_accumulator[idx_int];
                        let s1 = input_accumulator[idx_int+1];
                        processed_audio_buffer.push(s0 + frac * (s1 - s0));
                    }
                }
                let used_src = (target_samples as f32 * ratio) as usize;
                if used_src < input_accumulator.len() {
                    input_accumulator.drain(0..used_src);
                } else {
                    input_accumulator.clear();
                }
            }
            
            while processed_audio_buffer.len() >= FFT_SIZE {
                let chunk_to_process = &processed_audio_buffer[0..FFT_SIZE];
                
                let (gain, gate) = {
                    let s = shared_state.lock().unwrap();
                    (s.input_gain, s.noise_gate)
                };
                
                let rms = (chunk_to_process.iter().map(|x| x*x).sum::<f32>() / FFT_SIZE as f32).sqrt();
                
                if rms * gain > gate {
                    let amplified: Vec<f32> = chunk_to_process.iter().map(|&x| x * gain).collect();
                    let (cqt, chroma) = analyzer.compute_cqt_chroma(&amplified);
                    
                    if let Ok(mut state) = shared_state.lock() {
                        let mut frame = Vec::with_capacity(TOTAL_FEATURES);
                        frame.extend_from_slice(&cqt);
                        frame.extend_from_slice(&chroma);
                        state.copy_to_input(&frame);
                        
                        // Wizualizacja
                        for k in 0..48 {
                            let start = k * 4;
                            let avg = (cqt[start] + cqt[start+1] + cqt[start+2] + cqt[start+3]) / 4.0;
                            state.spectrum_visual[k] = state.spectrum_visual[k] * 0.7 + avg * 0.3;
                        }
                        for k in 0..12 {
                            state.chroma_sum[k] = state.chroma_sum[k] * 0.7 + chroma[k] * 0.3;
                        }
                    }
                } else {
                    if let Ok(mut state) = shared_state.lock() {
                        for x in &mut state.spectrum_visual { *x *= 0.8; }
                        for x in &mut state.chroma_sum { *x *= 0.8; }
                        state.raw_input_for_ai.fill(0.0);
                    }
                }
                
                processed_audio_buffer.drain(0..HOP_LENGTH);
            }
        },
        |err| eprintln!("Audio Err: {}", err),
        None 
    )?;

    stream.play()?;
    Ok(stream)
}

pub fn start_file_playback(path: String, shared_state: Arc<Mutex<AudioAnalysis>>) -> Result<()> {
    println!("Opening file: {}", path);
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let file_sr = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader.into_samples::<i16>().map(|s| s.unwrap_or(0) as f32 / 32768.0).collect()
    } else {
        reader.into_samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
    };

    let mut mono = Vec::with_capacity(samples.len() / channels);
    if channels == 2 {
        for f in samples.chunks(2) { mono.push((f[0]+f[1])*0.5); }
    } else { 
        mono = samples; 
    }

    thread::spawn(move || {
        let mut analyzer = match CqtAnalyzer::new("dsp_weights.json") {
            Ok(a) => a,
            Err(e) => { eprintln!("DSP Error: {}", e); return; }
        };

        let ratio = file_sr as f32 / TARGET_SR as f32;
        let frame_duration = Duration::from_secs_f32(HOP_LENGTH as f32 / TARGET_SR as f32);
        
        let target_len = (mono.len() as f32 / ratio) as usize;
        let mut resampled = Vec::with_capacity(target_len);
        for i in 0..target_len {
            let src_idx = i as f32 * ratio;
            let idx_int = src_idx as usize;
            let frac = src_idx - idx_int as f32;
            if idx_int + 1 < mono.len() {
                let val = mono[idx_int] * (1.0 - frac) + mono[idx_int+1] * frac;
                resampled.push(val);
            }
        }
        
        let mut pos = 0;
        while pos + FFT_SIZE < resampled.len() {
            let start_loop = Instant::now();
            let chunk = &resampled[pos .. pos + FFT_SIZE];
            
            let (cqt, chroma) = analyzer.compute_cqt_chroma(chunk);
            
            if let Ok(mut state) = shared_state.lock() {
                let mut frame = Vec::with_capacity(TOTAL_FEATURES);
                frame.extend_from_slice(&cqt);
                frame.extend_from_slice(&chroma);
                state.copy_to_input(&frame);
                
                for k in 0..48 { state.spectrum_visual[k] = cqt[k*4]; }
                state.chroma_sum = chroma.try_into().unwrap_or([0.0;12]);
            }
            
            pos += HOP_LENGTH;
            let elapsed = start_loop.elapsed();
            if frame_duration > elapsed { thread::sleep(frame_duration - elapsed); }
        }
        process::exit(0);
    });
    Ok(())
}
