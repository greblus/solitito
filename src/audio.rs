use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::Deserialize;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// --- CONSTANTS ---
pub const TOTAL_FEATURES: usize = 168; 
pub const CQT_BINS: usize = 144;       
pub const CHROMA_BINS: usize = 12;
pub const BASS_ENERGY_BINS: usize = 12;
pub const CTX_FRAMES: usize = 48; // model context

const BASS_BOOST_CUTOFF: usize = 36;

const TARGET_SR: u32 = 16000; 
const FFT_SIZE: usize = 8192; 
const HOP_LENGTH: usize = 256; 
const MIN_REF_LEVEL: f32 = 0.005; 

/// Input gain. A CONSTANT, not a knob: the CQT is max-normalised per frame, so
/// gain cancels out in the features. It only mattered where the level is
/// compared against the noise gate, and one knob is enough there.
pub const INPUT_GAIN: f32 = 2.0;

/// Sparse pseudo-CQT weights (CSR by CQT bin).
///
/// The kernel has 4097x144 weights, concentrated around each bin's centre
/// frequency. Dropping everything below 1e-4 of peak keeps 6.9% at 0.03% error
/// (measured in gen_weights.py): 28 MB -> ~2 MB, and ~14x fewer multiplications
/// per frame.
#[derive(Deserialize)]
struct DspConfig {
    /// Format marker. The old dense file lacks it - and also carried a different
    /// chroma mapping, so it is rejected with a message rather than accepted.
    #[serde(default)]
    pub format: String,
    /// 145 entries: bin i spans [offsets[i], offsets[i+1]).
    #[serde(default)]
    pub cqt_offsets: Vec<u32>,
    #[serde(default)]
    pub cqt_fft_idx: Vec<u32>,
    #[serde(default)]
    pub cqt_re: Vec<f32>,
    #[serde(default)]
    pub cqt_im: Vec<f32>,
    pub chroma_weights: Vec<f32>,
}

pub struct AudioAnalysis {
    pub input_history: [[f32; TOTAL_FEATURES]; CTX_FRAMES],
    /// Whether each history frame carries signal (true) or is silence pushed in
    /// by the noise gate (false). Parallel to `input_history`.
    pub frame_live: [bool; CTX_FRAMES],
    pub spectrum_visual: [f32; 48],
    pub chroma_sum: [f32; 12],
    pub bass_boost_enabled: bool,
    pub bass_boost_gain: f32,
    pub noise_gate: f32,
    /// Smoothed input level in the same units as `noise_gate`, so the UI meter is
    /// directly comparable with the threshold.
    pub input_level: f32,
    /// Increments on every detected attack. The app uses it to release the
    /// chord quality latch.
    pub onset_id: u64,
    /// Frames since the last attack. Below CTX_FRAMES the context window still
    /// contains part of the PREVIOUS chord.
    pub frames_since_onset: u32,
}

impl AudioAnalysis {
    pub fn push_frame(&mut self, data: &[f32]) {
        self.push(data, true);
    }

    /// Silence below the gate: history still advances, but the frame is empty.
    pub fn push_silence(&mut self) {
        self.push(&[0.0; TOTAL_FEATURES], false);
    }

    /// Reports an attack: resets the counter and bumps the event id.
    pub fn mark_onset(&mut self) {
        self.onset_id = self.onset_id.wrapping_add(1);
        self.frames_since_onset = 0;
    }

    fn push(&mut self, data: &[f32], live: bool) {
        if data.len() != TOTAL_FEATURES {
            return;
        }
        self.frames_since_onset = self.frames_since_onset.saturating_add(1);
        self.input_history.rotate_left(1);
        self.input_history[CTX_FRAMES - 1].copy_from_slice(data);
        self.frame_live.rotate_left(1);
        self.frame_live[CTX_FRAMES - 1] = live;
    }

    /// Fraction of the context window carrying real signal (0.0 - 1.0).
    ///
    /// In training the model only ever saw windows lying entirely inside a
    /// sustained chord (`range(start, end - 48)`), never "half silence, half
    /// chord". After the strings are struck the app feeds it exactly such a
    /// window for 48 frames (0.77 s) - input outside the training distribution.
    /// That is why chords appeared to resolve only "in the tail".
    pub fn history_fill(&self) -> f32 {
        let n = self.frame_live.iter().filter(|&&b| b).count();
        n as f32 / CTX_FRAMES as f32
    }
}

pub struct CqtAnalyzer {
    #[allow(dead_code)] planner: FftPlanner<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
    cqt_offsets: Vec<u32>,
    cqt_fft_idx: Vec<u32>,
    cqt_re: Vec<f32>,
    cqt_im: Vec<f32>,
    chroma_matrix: Vec<f32>,
    fft_buffer: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
}

impl CqtAnalyzer {
    pub fn new(json_path: &str) -> Result<Self> {
        println!("Loading DSP weights from {}...", json_path);
        let file = File::open(json_path).context("dsp_weights.json not found")?;
        let reader = BufReader::new(file);
        let config: DspConfig = serde_json::from_reader(reader)?;

        // The old format is rejected deliberately: it shipped a different chroma
        // mapping than cq_to_chroma, i.e. features the model was not trained on.
        // Accepting it silently would give an app that runs and gets chords wrong.
        if config.format != "sparse-csr-v1" {
            anyhow::bail!(
                "dsp_weights.json is in the old dense format.\n\
                 Generate a new one: python dist/gen_weights.py  (needs librosa)."
            );
        }
        if config.cqt_offsets.len() != CQT_BINS + 1 {
            anyhow::bail!(
                "dsp_weights.json: cqt_offsets has {} entries, expected {}.",
                config.cqt_offsets.len(), CQT_BINS + 1
            );
        }
        let nnz = config.cqt_fft_idx.len();
        if config.cqt_re.len() != nnz || config.cqt_im.len() != nnz {
            anyhow::bail!("dsp_weights.json: inconsistent CQT weight lengths.");
        }
        if config.chroma_weights.len() != CQT_BINS * CHROMA_BINS {
            anyhow::bail!(
                "dsp_weights.json: chroma has {} weights, expected {}.",
                config.chroma_weights.len(), CQT_BINS * CHROMA_BINS
            );
        }
        println!(
            "   Sparse CQT: {} weights ({:.1}% of {}), chroma {}x{}",
            nnz,
            100.0 * nnz as f32 / ((FFT_SIZE / 2 + 1) * CQT_BINS) as f32,
            (FFT_SIZE / 2 + 1) * CQT_BINS,
            CQT_BINS, CHROMA_BINS
        );

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (FFT_SIZE - 1) as f32).cos()))
            .collect();

        let scratch_len = fft.get_inplace_scratch_len();

        Ok(Self {
            planner, fft, window,
            cqt_offsets: config.cqt_offsets,
            cqt_fft_idx: config.cqt_fft_idx,
            cqt_re: config.cqt_re,
            cqt_im: config.cqt_im,
            chroma_matrix: config.chroma_weights,
            fft_buffer: vec![Complex{re:0.0, im:0.0}; FFT_SIZE],
            fft_scratch: vec![Complex{re:0.0, im:0.0}; scratch_len],
        })
    }

    pub fn compute_cqt_chroma(
        &mut self, 
        audio_chunk: &[f32], 
        boost_enabled: bool, 
        boost_gain: f32
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        
        // 1. FFT
        for (i, &sample) in audio_chunk.iter().enumerate().take(FFT_SIZE) {
            self.fft_buffer[i] = Complex { re: sample * self.window[i], im: 0.0 };
        }
        for i in audio_chunk.len()..FFT_SIZE {
            self.fft_buffer[i] = Complex { re: 0.0, im: 0.0 };
        }
        self.fft.process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);
        
        // 2. CQT - sparse multiply per bin.
        // The previous version scanned all 4097 FFT bins for each of the 144 CQT
        // bins (590k iterations per frame) and filtered zeros inside the loop.
        // Now we iterate the non-zeros directly: ~40k iterations, no branch.
        let mut cqt_mag = sparse_cqt_mag(
            &self.fft_buffer,
            &self.cqt_offsets,
            &self.cqt_fft_idx,
            &self.cqt_re,
            &self.cqt_im,
        );

        // --- BASS BOOST ---
        if boost_enabled {
            for i in 0..BASS_BOOST_CUTOFF {
                cqt_mag[i] *= boost_gain;
            }
        }

        // --- 3. Log Normalize (Per-Frame) ---
        let frame_max = cqt_mag.iter().fold(0.0f32, |a, &b| a.max(b));
        let ref_level = frame_max.max(MIN_REF_LEVEL); 

        let mut norm_cqt = cqt_mag.clone();
        for x in &mut norm_cqt {
            let val = x.max(1e-9);
            let db = 20.0 * (val / ref_level).log10();
            let n = (db + 80.0) / 80.0;
            *x = n.clamp(0.0, 1.0);
        }

        // 4. Chroma (cq_to_chroma matrix from dsp_weights.json)
        let mut chroma_vals = vec![0.0; CHROMA_BINS];
        for i in 0..CHROMA_BINS {
            let mut sum = 0.0;
            for k in 0..CQT_BINS {
                sum += norm_cqt[k] * self.chroma_matrix[k * CHROMA_BINS + i];
            }
            chroma_vals[i] = sum;
        }

        // Per-frame max normalisation of chroma, as in librosa.chroma_cqt norm=inf.
        let chroma_max = chroma_vals.iter().cloned().fold(0.0f32, f32::max);
        if chroma_max > 1e-9 {
            for v in &mut chroma_vals { *v /= chroma_max; }
        }

        // 5. Bass Energy
        let mut bass_energy = vec![0.0; BASS_ENERGY_BINS];
        for i in 0..12 {
            let idx = i * 2;
            if idx + 1 < CQT_BINS {
                bass_energy[i] = (norm_cqt[idx] + norm_cqt[idx+1]) / 2.0;
            }
        }
        
        let visual: Vec<f32> = norm_cqt.iter().skip(24).take(48).cloned().collect();

        (norm_cqt, chroma_vals, bass_energy, visual)
    }
}

pub fn start_audio_stream(shared_state: Arc<Mutex<AudioAnalysis>>) -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = host.default_input_device().context("Brak mikrofonu")?;
    let config: cpal::StreamConfig = device.default_input_config()?.into();
    let mic_sr = config.sample_rate.0;
    let channels = config.channels as usize;
    
    let mut analyzer = CqtAnalyzer::new("dsp_weights.json")?;
    let ratio = mic_sr as f32 / TARGET_SR as f32;
    
    // Attack detection: an energy jump above a slowly creeping envelope.
    const ATTACK_RATIO: f32 = 1.8;        // multiple of the baseline
    const ATTACK_FLOOR: f32 = 0.01;       // absolute floor, rejects noise
    const ATTACK_REFRACTORY: u32 = 12;    // ~0.2 s of detector silence after an attack
    let mut env_baseline: f32 = 0.0;
    let mut frames_since_attack: u32 = ATTACK_REFRACTORY;

    let mut input_acc = Vec::with_capacity(8192 * 2);
    let mut resampled = Vec::with_capacity(FFT_SIZE * 2);
    let mut read_pos: f32 = 0.0;

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            let mut mono = Vec::with_capacity(data.len() / channels);
            for f in data.chunks(channels) {
                mono.push(f.iter().sum::<f32>() / channels as f32);
            }
            input_acc.extend_from_slice(&mono);

            while read_pos + 1.0 < input_acc.len() as f32 {
                let idx_int = read_pos as usize;
                let frac = read_pos - idx_int as f32;
                
                let s0 = input_acc[idx_int];
                let s1 = input_acc[idx_int+1];
                resampled.push(s0 + frac * (s1 - s0));
                
                read_pos += ratio;
                
                if resampled.len() >= FFT_SIZE && resampled.len() % HOP_LENGTH == 0 {
                    let chunk = &resampled[resampled.len() - FFT_SIZE..];
                    
                    let (gate, boost_enabled, boost_gain) = {
                        let s = shared_state.lock().unwrap();
                        (s.noise_gate, s.bass_boost_enabled, s.bass_boost_gain)
                    };
                    let gain = INPUT_GAIN;
                    
                    let rms = (chunk.iter().map(|x| x*x).sum::<f32>() / FFT_SIZE as f32).sqrt();

                    // --- DETEKCJA ATAKU ---
                    // A strum is an energy jump above a slow envelope. Compared
                    // against an EMA baseline rather than an absolute threshold,
                    // which would depend on playing volume. The refractory period
                    // keeps one strum from firing several attacks.
                    // The gate and meter level come from RAW rms, without
                    // INPUT_GAIN, so the scale is true input dBFS (0 dB = full
                    // scale) rather than "dBFS plus 6", which pushed the whole
                    // useful range to the end of the slider on a loud microphone.
                    let level = rms;
                    if let Ok(mut st) = shared_state.lock() {
                        // Fast attack, slow release: the meter should show the
                        // strum peak, not the average of the gaps between chords.
                        st.input_level = if level > st.input_level {
                            level
                        } else {
                            st.input_level * 0.92 + level * 0.08
                        };
                    }
                    if level > env_baseline * ATTACK_RATIO && level > ATTACK_FLOOR
                        && frames_since_attack >= ATTACK_REFRACTORY
                    {
                        if let Ok(mut st) = shared_state.lock() { st.mark_onset(); }
                        frames_since_attack = 0;
                    } else {
                        frames_since_attack = frames_since_attack.saturating_add(1);
                    }
                    // The baseline falls fast and rises slowly; otherwise a long
                    // sustained chord would raise the threshold and swallow the
                    // next attack.
                    env_baseline = if level > env_baseline {
                        env_baseline * 0.90 + level * 0.10
                    } else {
                        env_baseline * 0.70 + level * 0.30
                    };

                    if level > gate {
                        let amplified: Vec<f32> = chunk.iter().map(|&x| x * gain).collect();
                        let (cqt, chroma, bass, visual) = analyzer.compute_cqt_chroma(
                            &amplified, boost_enabled, boost_gain
                        );
                        
                        if let Ok(mut state) = shared_state.lock() {
                            let mut frame = Vec::with_capacity(TOTAL_FEATURES);
                            frame.extend_from_slice(&cqt);
                            frame.extend_from_slice(&chroma);
                            frame.extend_from_slice(&bass);
                            
                            state.push_frame(&frame);
                            
                            for k in 0..48 {
                                if k < visual.len() { state.spectrum_visual[k] = visual[k]; }
                            }
                            state.chroma_sum = chroma.try_into().unwrap_or([0.0;12]);
                        }
                    } else if let Ok(mut state) = shared_state.lock() {
                        // In silence push an empty frame to advance the history
                        state.push_silence();
                        for x in &mut state.spectrum_visual { *x *= 0.7; }
                    }
                }
            }
            
            if read_pos > 0.0 {
                let consumed = (read_pos as usize).min(input_acc.len());
                input_acc.drain(0..consumed);
                read_pos -= consumed as f32;
            }
            
            if resampled.len() > FFT_SIZE * 2 {
                let keep = FFT_SIZE; 
                let drain_cnt = resampled.len().saturating_sub(keep);
                resampled.drain(0..drain_cnt);
            }
        },
        |_| {}, None
    )?;
    stream.play()?;
    Ok(stream)
}

pub fn start_file_playback(path: String, shared_state: Arc<Mutex<AudioAnalysis>>) -> Result<()> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader.into_samples::<i16>().map(|s| s.unwrap_or(0) as f32 / 32768.0).collect()
    } else {
        reader.into_samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
    };

    thread::spawn(move || {
        let mut analyzer = CqtAnalyzer::new("dsp_weights.json").unwrap();
        let mut pos = 0;
        
        while pos + FFT_SIZE < samples.len() {
            let start = Instant::now();
            let chunk = &samples[pos..pos+FFT_SIZE];
            let (cqt, chroma, bass, _) = analyzer.compute_cqt_chroma(chunk, true, 5.0);
            
            if let Ok(mut state) = shared_state.lock() {
                let mut frame = Vec::with_capacity(TOTAL_FEATURES);
                frame.extend_from_slice(&cqt);
                frame.extend_from_slice(&chroma);
                frame.extend_from_slice(&bass);
                state.push_frame(&frame);
            }
            pos += HOP_LENGTH;
            let sleep = Duration::from_secs_f32(HOP_LENGTH as f32 / TARGET_SR as f32);
            if sleep > start.elapsed() { thread::sleep(sleep - start.elapsed()); }
        }
    });
    Ok(())
}

/// CQT magnitude from CSR weights: bin `i` sums over `offsets[i]..offsets[i+1]`.
///
/// Split out of `compute_cqt_chroma` so it can be tested - a CSR indexing mistake
/// does not crash anything, it silently shifts the spectrum.
fn sparse_cqt_mag(
    fft: &[Complex<f32>],
    offsets: &[u32],
    fft_idx: &[u32],
    w_re: &[f32],
    w_im: &[f32],
) -> Vec<f32> {
    let n_bins = offsets.len().saturating_sub(1);
    let mut out = vec![0.0; n_bins];
    for i in 0..n_bins {
        let (from, to) = (offsets[i] as usize, offsets[i + 1] as usize);
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for j in from..to {
            let f = fft[fft_idx[j] as usize];
            sum_re += f.re * w_re[j] - f.im * w_im[j];
            sum_im += f.re * w_im[j] + f.im * w_re[j];
        }
        out[i] = (sum_re * sum_re + sum_im * sum_im).sqrt();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: the same multiply on a DENSE matrix, as before the format change.
    fn dense_cqt_mag(fft: &[Complex<f32>], re: &[f32], im: &[f32], n_bins: usize) -> Vec<f32> {
        let n_fft = fft.len();
        (0..n_bins)
            .map(|i| {
                let (mut sr, mut si) = (0.0f32, 0.0f32);
                for k in 0..n_fft {
                    let idx = k * n_bins + i;
                    let f = fft[k];
                    sr += f.re * re[idx] - f.im * im[idx];
                    si += f.re * im[idx] + f.im * re[idx];
                }
                (sr * sr + si * si).sqrt()
            })
            .collect()
    }

    #[test]
    fn csr_matches_the_dense_matrix() {
        let (n_fft, n_bins) = (32usize, 5usize);
        // deterministic "randomness" so the test is reproducible
        let pseudo = |s: usize| ((s * 2654435761usize) % 2000) as f32 / 1000.0 - 1.0;

        let mut re = vec![0.0f32; n_fft * n_bins];
        let mut im = vec![0.0f32; n_fft * n_bins];
        for k in 0..n_fft {
            for i in 0..n_bins {
                // sparsity: non-zero weights only near the "centre frequency"
                if (k as i32 - (i as i32 * 6 + 3)).abs() <= 2 {
                    re[k * n_bins + i] = pseudo(k * 7 + i);
                    im[k * n_bins + i] = pseudo(k * 13 + i + 1);
                }
            }
        }

        // dense -> CSR per bin
        let (mut offsets, mut idx, mut sre, mut sim) = (vec![0u32], vec![], vec![], vec![]);
        for i in 0..n_bins {
            for k in 0..n_fft {
                let j = k * n_bins + i;
                if re[j] != 0.0 || im[j] != 0.0 {
                    idx.push(k as u32);
                    sre.push(re[j]);
                    sim.push(im[j]);
                }
            }
            offsets.push(idx.len() as u32);
        }
        assert!(idx.len() < n_fft * n_bins, "the fixture was supposed to be sparse");

        let fft: Vec<Complex<f32>> = (0..n_fft)
            .map(|k| Complex { re: pseudo(k * 3), im: pseudo(k * 5 + 2) })
            .collect();

        let dense = dense_cqt_mag(&fft, &re, &im, n_bins);
        let sparse = sparse_cqt_mag(&fft, &offsets, &idx, &sre, &sim);

        assert_eq!(dense.len(), sparse.len());
        for (i, (d, s)) in dense.iter().zip(&sparse).enumerate() {
            assert!((d - s).abs() < 1e-5, "bin {i}: dense {d}, sparse {s}");
        }
    }

    #[test]
    fn an_empty_bin_yields_zero_not_a_panic() {
        let fft = vec![Complex { re: 1.0, im: -0.5 }; 8];
        // bin 0 has one weight, bin 1 has none
        let out = sparse_cqt_mag(&fft, &[0, 1, 1], &[3], &[2.0], &[0.0]);
        assert_eq!(out.len(), 2);
        assert!((out[0] - (1.0f32 * 2.0).hypot(-0.5 * 2.0)).abs() < 1e-6);
        assert_eq!(out[1], 0.0);
    }
}

#[cfg(test)]
mod fill_tests {
    use super::*;

    fn empty() -> AudioAnalysis {
        AudioAnalysis {
            input_history: [[0.0; TOTAL_FEATURES]; CTX_FRAMES],
            frame_live: [false; CTX_FRAMES],
            onset_id: 0,
            frames_since_onset: 0,
            spectrum_visual: [0.0; 48],
            chroma_sum: [0.0; 12],
            bass_boost_enabled: false,
            bass_boost_gain: 1.0,
            noise_gate: 0.0,
            input_level: 0.0,
        }
    }

    #[test]
    fn the_window_fills_only_after_a_full_context() {
        let mut a = empty();
        assert_eq!(a.history_fill(), 0.0, "start: silence only");

        let frame = [0.5f32; TOTAL_FEATURES];
        for i in 1..CTX_FRAMES {
            a.push_frame(&frame);
            let want = i as f32 / CTX_FRAMES as f32;
            assert!((a.history_fill() - want).abs() < 1e-6,
                    "after {i} frames expected {want}, got {}", a.history_fill());
        }
        a.push_frame(&frame);
        assert_eq!(a.history_fill(), 1.0, "a full context must report a full window");
    }

    #[test]
    fn silence_in_the_middle_lowers_the_fill() {
        let mut a = empty();
        let frame = [0.5f32; TOTAL_FEATURES];
        for _ in 0..CTX_FRAMES { a.push_frame(&frame); }
        assert_eq!(a.history_fill(), 1.0);

        a.push_silence();
        let want = (CTX_FRAMES - 1) as f32 / CTX_FRAMES as f32;
        assert!((a.history_fill() - want).abs() < 1e-6);

        // silence pushes the signal out of the whole window
        for _ in 0..CTX_FRAMES { a.push_silence(); }
        assert_eq!(a.history_fill(), 0.0);
    }

    #[test]
    fn the_live_flag_tracks_the_data() {
        let mut a = empty();
        let frame = [1.0f32; TOTAL_FEATURES];
        a.push_frame(&frame);
        a.push_silence();
        // the newest frame is silence...
        assert!(!a.frame_live[CTX_FRAMES - 1]);
        assert_eq!(a.input_history[CTX_FRAMES - 1][0], 0.0);
        // ...and the previous one carries signal
        assert!(a.frame_live[CTX_FRAMES - 2]);
        assert_eq!(a.input_history[CTX_FRAMES - 2][0], 1.0);
    }
}
