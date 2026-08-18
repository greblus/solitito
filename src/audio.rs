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

pub const TARGET_SR: u32 = 16000; 
pub const FFT_SIZE: usize = 8192; 
pub const HOP_LENGTH: usize = 256; 
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
    /// Pitch class of the single note sounding in the LAST frame, from the CQT
    /// alone - see `mono_pitch`. `None` while the gate is shut. The model cannot
    /// answer this: it looks at 0.77 s and note practice moves faster than that.
    pub cqt_pitch: Option<usize>,
    /// Whether the last frame carried signal. Cleared state downstream hangs on
    /// this: a pitch vector nobody refreshes stays true forever otherwise.
    pub gate_open: bool,
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

/// Bins per semitone in the CQT: 144 bins over 6 octaves from C1.
const BINS_PER_SEMITONE: usize = 2;

/// Offsets of the first six harmonics, in CQT bins: round(24 * log2(h)).
const HARMONIC_OFFSETS: [usize; 6] = [0, 24, 38, 48, 56, 62];

/// Least a reading has to score to be believed. In noise this estimator always
/// names something, so a weak answer is worth less than none - and the modes
/// that judge single notes have the model's pitch head to fall back on.
/// SOLITITO_EAR=1 prints the scores, this one included, before it is moved.
pub const MONO_MIN_SCORE: f32 = 0.5;

/// Search range for a guitar fundamental, in CQT bins from C1: E2 (bin 32) up to
/// roughly E5. Outside it lies nothing a guitar can sound as a fundamental, and
/// letting the search go there only invites octave errors.
const F0_LOW_BIN: usize = 30;
const F0_HIGH_BIN: usize = 112;

/// The pitch class of a SINGLE sounding note, straight from one CQT frame.
///
/// A harmonic sum over the log-magnitude CQT - which, the axis being
/// logarithmic, is the harmonic PRODUCT spectrum, the standard monophonic
/// estimator. It exists because the model cannot answer this question: its
/// context window is 48 frames, so it reports everything that sounded in the
/// last 0.77 s, and in note practice that is two or three notes at once. This
/// looks at 16 ms and has no memory at all.
///
/// Returns the pitch class (0 = C) and the score of the winning bin.
pub fn mono_pitch(norm_cqt: &[f32]) -> Option<(usize, f32)> {
    if norm_cqt.len() < CQT_BINS {
        return None;
    }
    let score_at = |b: usize| -> f32 {
        let mut sum = 0.0;
        let mut used = 0.0;
        for (h, off) in HARMONIC_OFFSETS.iter().enumerate() {
            let idx = b + off;
            if idx >= CQT_BINS {
                break;
            }
            // Later harmonics say less about which note it is, and are the ones
            // a neighbouring string is most likely to share.
            let w = 1.0 / (1.0 + h as f32 * 0.35);
            sum += w * norm_cqt[idx];
            used += w;
        }
        if used > 0.0 { sum / used } else { 0.0 }
    };

    // Every bin, quarter tones included. Restricting the search to semitone
    // centres was tried and starved the score: a real note sits a few cents off
    // and leaves most of its energy in the bin next door, which took a plain
    // reading from 1.00 down to 0.45 - under the gate, so no note reported at
    // all, and the app left waiting on the model's 0.77 s window. The quarter
    // tones are where the resolution is; what they must not do is decide the
    // semitone by rounding, which is what the interpolation below is for.
    let mut best = (F0_LOW_BIN, 0.0f32);
    for b in F0_LOW_BIN..F0_HIGH_BIN.min(CQT_BINS) {
        let s = score_at(b);
        if s > best.1 {
            best = (b, s);
        }
    }
    if best.1 <= 0.0 {
        return None;
    }

    // An octave down scores almost as well whenever the true fundamental's
    // harmonics land on its own - the classic failure of this method. Prefer the
    // higher candidate unless the lower one is clearly better on its own bin.
    let mut b = best.0;
    if b >= 24 + F0_LOW_BIN {
        let lower = b - 24;
        if score_at(lower) > best.1 * 0.98 && norm_cqt[lower] > norm_cqt[b] * 1.15 {
            b = lower;
        }
    }

    // Where the peak really lies, to a fraction of a bin. Three scores around
    // the winner describe a parabola, and its vertex says how far off centre
    // the note sits - which is what decides the semitone. Rounding the winning
    // bin instead answered every quarter-tone reading with the note ABOVE
    // (`round(18.5)` goes away from zero), so an F# a little sharp came back as
    // G, and in Formulas - where a mark never expires - the flat second lit up
    // every time the root was struck.
    let mut pos = b as f32;
    if b > 0 && b + 1 < CQT_BINS {
        let (l, c, r) = (score_at(b - 1), best.1, score_at(b + 1));
        let denom = l - 2.0 * c + r;
        if denom.abs() > 1e-6 {
            let delta = 0.5 * (l - r) / denom;
            // Beyond half a bin the parabola is describing something other than
            // this peak, and the shift is not to be trusted.
            if delta.abs() <= 0.5 {
                pos += delta;
            }
        }
    }
    let semitone = (pos / BINS_PER_SEMITONE as f32).round() as usize;
    Some((semitone % 12, best.1))
}

/// What the input stream actually opened with.
///
/// Reported back to the settings panel because on Windows the console is hidden
/// (`windows_subsystem = "windows"`), so a stream that failed to start looked
/// exactly like one that started and heard nothing.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct InputInfo {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub format: String,
    /// The chosen device could not be opened and the system default was used
    /// instead. Worth saying out loud: the fallback works, so nothing looks
    /// broken - the app is simply listening somewhere else than the picker
    /// claims, and the channel choice then belongs to a different device.
    pub fell_back: bool,
}

/// Silences ALSA's own error printing.
///
/// Opening a device or listing them makes libasound try every plugin it knows -
/// OSS emulation, dsnoop, route - and print a line to stderr for each one that
/// is not there. Ten lines of "Cannot open device /dev/dsp" before the app has
/// said anything is not a failure, but it reads as one, and it buries the line
/// that matters.
///
/// The handler is variadic, which Rust cannot express in a definition on
/// stable, so a fixed-arity no-op is cast to the right type. It is sound here
/// because the function never looks at its arguments: nothing reads the
/// variadic part, so there is nothing to get wrong about how it was passed.
#[cfg(target_os = "linux")]
pub fn hush_alsa() {
    use std::os::raw::{c_char, c_int};

    type Handler = unsafe extern "C" fn(*const c_char, c_int, *const c_char, c_int, *const c_char);
    unsafe extern "C" {
        fn snd_lib_error_set_handler(handler: Option<Handler>) -> c_int;
    }
    unsafe extern "C" fn silence(
        _file: *const c_char,
        _line: c_int,
        _function: *const c_char,
        _err: c_int,
        _fmt: *const c_char,
    ) {
    }
    unsafe {
        snd_lib_error_set_handler(Some(silence));
    }
}

/// Nothing to quieten anywhere else.
#[cfg(not(target_os = "linux"))]
pub fn hush_alsa() {}

/// Input devices the host can see, for the chooser in the settings panel.
///
/// On Linux one entry is usually enough - PipeWire routes whatever is plugged in
/// to the default device. Windows has no such layer, so the choice has to be
/// made here.
pub fn list_input_devices() -> Vec<String> {
    let Ok(devices) = cpal::default_host().input_devices() else {
        return Vec::new();
    };
    devices.filter_map(|d| d.name().ok()).collect()
}

/// Reports what a device says about itself, without opening a stream.
///
/// For `--devices`: a name alone does not say whether it is the interface or a
/// sound server standing in front of it.
pub fn probe_input(name: &str) -> Result<InputInfo> {
    let host = cpal::default_host();
    let device = host
        .input_devices()?
        .find(|d| d.name().map(|n| n == name).unwrap_or(false))
        .context("device is gone")?;
    let supported = device.default_input_config()?;
    Ok(InputInfo {
        name: name.to_string(),
        sample_rate: supported.sample_rate().0,
        channels: supported.channels(),
        format: format!("{:?}", supported.sample_format()),
        fell_back: false,
    })
}

/// Opens an input stream.
///
/// * `preferred` - device name to look for; falls back to the system default
///   when it is absent or no longer plugged in.
/// * `channel` - 1-based input to listen on, clamped to what the device has.
///   There is no mixing option: averaging the inputs of whatever interface
///   happens to be plugged in pulls in the other socket as well - hum from an
///   empty one, or a second instrument - and costs 6 dB besides.
pub fn start_audio_stream(
    shared_state: Arc<Mutex<AudioAnalysis>>,
    preferred: Option<&str>,
    channel: usize,
) -> Result<(cpal::Stream, InputInfo)> {
    let host = cpal::default_host();
    let picked = preferred.and_then(|want| {
        host.input_devices()
            .ok()?
            .find(|d| d.name().map(|n| n == want).unwrap_or(false))
    });
    // A card that something else already holds is not in the enumeration at
    // all - ALSA hands out a direct device once. So this is not only "unplugged
    // since last time": it is also "PipeWire is on it", which is the common case
    // and used to end here silently, on the default, with both channels alike.
    let fell_back = preferred.is_some() && picked.is_none();
    let device = picked
        .or_else(|| host.default_input_device())
        .context("No audio input device")?;

    // The sample FORMAT matters as much as the rate. Discarding it and building
    // an f32 stream on a device that reports i16 fails outright, and the app
    // then runs with no input at all - which looks like the interface being at
    // fault, because its own meters still show signal.
    let supported = device.default_input_config()?;
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let mic_sr = config.sample_rate.0;
    let channels = config.channels as usize;
    let info = InputInfo {
        name: device.name().unwrap_or_else(|_| "?".into()),
        sample_rate: mic_sr,
        channels: config.channels,
        format: format!("{sample_format:?}"),
        fell_back,
    };
    
    let mut analyzer = CqtAnalyzer::new("dsp_weights.json")?;
    let ratio = mic_sr as f32 / TARGET_SR as f32;
    
    // Attack detection: an energy jump above a slowly creeping envelope.
    const ATTACK_RATIO: f32 = 1.8;        // multiple of the baseline
    const ATTACK_FLOOR: f32 = 0.01;       // absolute floor, rejects noise
    const ATTACK_REFRACTORY: u32 = 12;    // ~0.2 s of detector silence after an attack
    let mut env_baseline: f32 = 0.0;
    let loud_ear = std::env::var("SOLITITO_EAR").is_ok();
    let mut frames_since_attack: u32 = ATTACK_REFRACTORY;

    let mut input_acc = Vec::with_capacity(8192 * 2);
    let mut resampled = Vec::with_capacity(FFT_SIZE * 2);
    let mut read_pos: f32 = 0.0;

    // Clamped here rather than per frame: a saved channel from a device with
    // more inputs than this one must not silently read past the end.
    let pick = channel.clamp(1, channels.max(1)) - 1;
    let mut process = move |data: &[f32]| {
            let mut mono = Vec::with_capacity(data.len() / channels);
            for f in data.chunks(channels) {
                mono.push(f.get(pick).copied().unwrap_or(0.0));
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
                        
                        // One frame, one note: no window memory, so a scale is
                        // tracked instead of smeared. Weak answers are dropped -
                        // in noise this estimator always names something.
                        let heard = mono_pitch(&cqt);
                        // SOLITITO_EAR=1: every reading with its score, the ones
                        // under the gate included. A note played softly has its
                        // upper partials down in the noise and scores lower for
                        // it, and this is the only way to see by how much before
                        // moving the gate.
                        if loud_ear {
                            if let Some((pc, s)) = heard {
                                println!(
                                    "ucho {:<3} wynik={s:.2}{}",
                                    crate::NOTE_NAMES[pc],
                                    if s >= MONO_MIN_SCORE { "" } else { "   << pod bramka" },
                                );
                            }
                        }
                        let mono = heard.filter(|&(_, s)| s >= MONO_MIN_SCORE).map(|(pc, _)| pc);
                        if let Ok(mut state) = shared_state.lock() {
                            state.cqt_pitch = mono;
                            state.gate_open = true;
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
                        state.cqt_pitch = None;
                        state.gate_open = false;
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
    };

    // One arm runs, so moving `process` into each is fine. Anything other than
    // these three is rare enough that saying so beats guessing at a conversion.
    let err = |e| eprintln!("audio input error: {e}");
    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            device.build_input_stream(&config, move |d: &[f32], _: &_| process(d), err, None)?
        }
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |d: &[i16], _: &_| {
                let v: Vec<f32> = d.iter().map(|&s| s as f32 / 32768.0).collect();
                process(&v);
            },
            err,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            move |d: &[u16], _: &_| {
                let v: Vec<f32> = d.iter().map(|&s| (s as f32 - 32768.0) / 32768.0).collect();
                process(&v);
            },
            err,
            None,
        )?,
        other => anyhow::bail!("unsupported sample format: {other:?}"),
    };
    stream.play()?;
    Ok((stream, info))
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
            cqt_pitch: None,
            gate_open: false,
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

    /// The stretched partials of a real string do not name the semitone above.
    ///
    /// A steel string is inharmonic: its partials run progressively sharp, so
    /// they land above where a perfect harmonic series would put them. The CQT
    /// has two bins to a semitone, so the QUARTER-TONE bin above the note
    /// collects that stretched series in full while the note's own bin only has
    /// the fundamental - and the winner used to be free to sit there, whereupon
    /// the semitone it rounded to was the one above. F# came back as G, which in
    /// Formulas lit the flat second every time the root was struck.
    #[test]
    fn stretched_partials_do_not_name_the_semitone_above() {
        // F#2: semitone 18 above C1, so bin 36 with two bins to a semitone.
        const F_SHARP_2: usize = 36;
        let mut cqt = vec![0.0f32; CQT_BINS];
        cqt[F_SHARP_2] = 1.0;
        for h in HARMONIC_OFFSETS.iter().skip(1) {
            let idx = F_SHARP_2 + h + 1; // a bin sharp, as a real string is
            if idx < CQT_BINS {
                cqt[idx] = 1.0;
            }
        }
        let (pc, _) = mono_pitch(&cqt).expect("a note with a fundamental present");
        assert_eq!(pc, 6, "F# was named {}", NOTE_NAMES_TEST[pc]);
    }


    /// A note a few cents sharp still scores high enough to be reported.
    ///
    /// The CQT has two bins to a semitone - 50 cents apart - so a string tuned
    /// a shade sharp spreads its energy over the note's own bin and the quarter
    /// tone above it, and every partial does the same. Searching semitone
    /// centres only, that spread has to still add up: the estimate is gated on
    /// its score, and a note that scores under the gate is reported as no note
    /// at all, which leaves the app waiting for the model's 0.77 s window.
    #[test]
    fn a_note_a_few_cents_sharp_still_scores() {
        // F#2: semitone 18 above C1, so bin 36 with two bins to a semitone.
        const F_SHARP_2: usize = 36;
        const GATE: f32 = 0.5; // the value the input thread filters on
        let mut cqt = vec![0.0f32; CQT_BINS];
        for h in HARMONIC_OFFSETS {
            let idx = F_SHARP_2 + h;
            if idx + 1 < CQT_BINS {
                // Sharp of centre: most of the energy one bin up.
                cqt[idx] = 0.45;
                cqt[idx + 1] = 1.0;
            }
        }
        let (pc, score) = mono_pitch(&cqt).expect("a note with every harmonic present");
        assert_eq!(pc, 6, "named {} instead of F#", NOTE_NAMES_TEST[pc]);
        assert!(score >= GATE, "scored {score:.2}, under the gate of {GATE}");
    }

    /// Energy leaking into the bin above does not become the note above.
    ///
    /// This is the failure that kept coming back on the guitar: the root and the
    /// flat second credited off one pluck. A note peaks in its own bin and
    /// spills into the quarter tone next to it, and that spill must not carry
    /// the reading a semitone up.
    #[test]
    fn a_spill_into_the_next_bin_is_not_the_next_note() {
        // D3: semitone 26 above C1, so bin 52.
        const D3: usize = 52;
        let mut cqt = vec![0.0f32; CQT_BINS];
        for h in HARMONIC_OFFSETS {
            let idx = D3 + h;
            if idx + 1 < CQT_BINS {
                cqt[idx] = 1.0;
                cqt[idx + 1] = 0.5;
            }
        }
        let (pc, _) = mono_pitch(&cqt).expect("a note with every harmonic present");
        assert_eq!(pc, 2, "D read as {}", NOTE_NAMES_TEST[pc]);
    }

    const NOTE_NAMES_TEST: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
}