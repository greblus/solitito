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
    /// Attacks the flux detector has reported. Pitch-blind by construction.
    pub flux_id: u64,
    /// Hops since the last of them, so a credit can ask whether a string was
    /// hit JUST NOW rather than merely at some point since.
    pub flux_age: u32,
    /// Hops of audio seen, gate or no gate: the recording's own clock.
    pub hops: u64,
    /// Which pitch classes are genuinely sounding, one bit each, from
    /// `voices::voices` - the spectrum explained rather than ranked, so a class
    /// that is only somebody else's harmonic is not in here.
    pub voices: u16,
    /// Strikes of each class, from `strike::Strikes`: THIS class was hit, which
    /// neither flux (a string was hit) nor the ear (this class sounds) can say.
    pub class_strikes: [u32; 12],
    /// Hops since each class was last struck.
    pub class_strike_age: [u32; 12],
    /// Whether that detector is running at all. Its model is a separate file,
    /// and without it the app judges as it did before it existed.
    pub strikes_live: bool,
    /// Frames since the last attack. Below CTX_FRAMES the context window still
    /// contains part of the PREVIOUS chord.
    pub frames_since_onset: u32,
    /// Pitch class of the single note sounding in the LAST frame, from the CQT
    /// alone - see `mono_pitch`. `None` while the gate is shut. The model cannot
    /// answer this: it looks at 0.77 s and note practice moves faster than that.
    pub cqt_pitch: Option<usize>,
    /// The same reading as an absolute semitone from the lowest CQT bin, so a
    /// note can be told from itself an octave down.
    pub cqt_semitone: Option<usize>,
    /// Whether the last frame carried signal. Cleared state downstream hangs on
    /// this: a pitch vector nobody refreshes stays true forever otherwise.
    pub gate_open: bool,
    /// Frames pushed since the app started, gated or not.
    ///
    /// The UI runs on its own clock, and anything that samples the per-frame
    /// estimate from there counts the same frame twice as often as not - which
    /// is enough for one bad reading to carry a vote. This is what says whether
    /// there is anything new to look at.
    pub frames_seen: u64,
}

impl AudioAnalysis {
    pub fn push_frame(&mut self, data: &[f32]) {
        self.push(data, true);
    }

    /// Silence below the gate: history still advances, but the frame is empty.
    pub fn push_silence(&mut self) {
        self.push(&[0.0; TOTAL_FEATURES], false);
    }

    /// Reports an attack the FLUX detector found, which is a different claim
    /// from `mark_onset`: that one compares whole-frame levels and a quiet
    /// string beside a ringing one does not move them.
    pub fn mark_flux(&mut self) {
        self.flux_id = self.flux_id.wrapping_add(1);
        self.flux_age = 0;
    }

    /// One hop's answer from the strike detector: every class ages a hop, and
    /// the ones struck on it start again from nothing.
    pub fn mark_strikes(&mut self, struck: &[usize]) {
        self.strikes_live = true;
        for age in &mut self.class_strike_age {
            *age = age.saturating_add(1);
        }
        for &pc in struck {
            self.class_strikes[pc] = self.class_strikes[pc].wrapping_add(1);
            self.class_strike_age[pc] = 0;
        }
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
        self.frames_seen = self.frames_seen.wrapping_add(1);
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

        check_weights(&config)?;

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

    /// One frame's features: the normalised CQT, the chroma, the bass energy,
    /// and the part of the CQT the spectrum display shows.
    pub fn compute_cqt_chroma(
        &mut self,
        audio_chunk: &[f32],
        boost_enabled: bool,
        boost_gain: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        self.windowed_fft(audio_chunk);

        // The CQT: a sparse multiply per bin, over the non-zero weights only -
        // ~40k iterations a frame instead of 590k over the dense matrix.
        let mut cqt_mag = sparse_cqt_mag(
            &self.fft_buffer,
            &self.cqt_offsets,
            &self.cqt_fft_idx,
            &self.cqt_re,
            &self.cqt_im,
        );
        if boost_enabled {
            for v in &mut cqt_mag[..BASS_BOOST_CUTOFF] {
                *v *= boost_gain;
            }
        }

        let norm_cqt = log_normalise(&cqt_mag);
        let chroma = self.chroma_of(&norm_cqt);
        let bass = bass_energy(&norm_cqt);
        let visual: Vec<f32> = norm_cqt.iter().skip(24).take(48).cloned().collect();
        (norm_cqt, chroma, bass, visual)
    }

    /// The chunk, under the Hann window and zero-padded, through the FFT into
    /// `fft_buffer`.
    fn windowed_fft(&mut self, audio_chunk: &[f32]) {
        for (i, &sample) in audio_chunk.iter().enumerate().take(FFT_SIZE) {
            self.fft_buffer[i] = Complex { re: sample * self.window[i], im: 0.0 };
        }
        for i in audio_chunk.len()..FFT_SIZE {
            self.fft_buffer[i] = Complex { re: 0.0, im: 0.0 };
        }
        self.fft.process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);
    }

    /// The chroma through the `cq_to_chroma` matrix from dsp_weights.json, with
    /// the per-frame max normalisation of librosa's `chroma_cqt` (norm=inf).
    fn chroma_of(&self, norm_cqt: &[f32]) -> Vec<f32> {
        let mut chroma = vec![0.0; CHROMA_BINS];
        for (i, value) in chroma.iter_mut().enumerate() {
            let mut sum = 0.0;
            for k in 0..CQT_BINS {
                sum += norm_cqt[k] * self.chroma_matrix[k * CHROMA_BINS + i];
            }
            *value = sum;
        }
        let chroma_max = chroma.iter().cloned().fold(0.0f32, f32::max);
        if chroma_max > 1e-9 {
            for v in &mut chroma {
                *v /= chroma_max;
            }
        }
        chroma
    }
}

/// The CQT in decibels below the frame's own peak, mapped from -80..0 dB onto
/// 0..1 - the features the model was trained on.
fn log_normalise(cqt_mag: &[f32]) -> Vec<f32> {
    let frame_max = cqt_mag.iter().fold(0.0f32, |a, &b| a.max(b));
    let ref_level = frame_max.max(MIN_REF_LEVEL);
    cqt_mag
        .iter()
        .map(|&x| {
            let db = 20.0 * (x.max(1e-9) / ref_level).log10();
            ((db + 80.0) / 80.0).clamp(0.0, 1.0)
        })
        .collect()
}

/// The bottom two octaves of the CQT, one value per semitone pair of bins.
fn bass_energy(norm_cqt: &[f32]) -> Vec<f32> {
    let mut bass = vec![0.0; BASS_ENERGY_BINS];
    for (i, value) in bass.iter_mut().enumerate().take(12) {
        let idx = i * 2;
        if idx + 1 < CQT_BINS {
            *value = (norm_cqt[idx] + norm_cqt[idx + 1]) / 2.0;
        }
    }
    bass
}

/// Rejects weights the model was not trained with.
///
/// The old format is rejected deliberately: it shipped a different chroma
/// mapping than cq_to_chroma, i.e. features the model was not trained on.
/// Accepting it silently would give an app that runs and gets chords wrong.
fn check_weights(config: &DspConfig) -> Result<()> {
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
    Ok(())
}

/// A copy of exactly the 16 kHz hops the detectors see, for measuring on.
///
/// SOLITITO_RECORD=<prefix> writes <prefix>-g<n>.wav, mono float32 at
/// TARGET_SR. This is the audio AFTER resampling and after the channel choice,
/// so what is measured later is what the app actually heard - no second guess
/// about which socket the instrument was in, which is the mistake an external
/// recorder invites.
///
/// Every hop, not only the ones past the noise gate: a file with the quiet
/// parts cut out has a step where each cut was, and a step reads as an attack.
pub struct Capture {
    writer: Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>,
    path: std::path::PathBuf,
    hops: u64,
    failed: bool,
}

impl Capture {
    /// None when SOLITITO_RECORD is unset, which is the normal case. The
    /// generation comes from `journal::base`, so the audio and the journal of
    /// one run always carry the same number.
    pub fn from_env() -> Option<Self> {
        let path = crate::journal::base()?.with_extension("wav");
        let file = match std::fs::File::create(&path) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("📼 nie mogę utworzyć {}: {error}", path.display());
                return None;
            }
        };
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: TARGET_SR,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        match hound::WavWriter::new(std::io::BufWriter::new(file), spec) {
            Ok(writer) => {
                eprintln!("📼 {} · {TARGET_SR} Hz · 1 ch · F32", path.display());
                Some(Self { writer: Some(writer), path, hops: 0, failed: false })
            }
            Err(error) => {
                eprintln!("📼 nie mogę pisać do {}: {error}", path.display());
                None
            }
        }
    }

    fn write_hop(&mut self, hop: &[f32]) -> Result<()> {
        let Some(writer) = self.writer.as_mut() else {
            return Ok(());
        };
        for &sample in hop {
            writer.write_sample(sample)?;
        }
        self.hops += 1;
        // The header stays readable if the process is interrupted.
        if self.hops % 64 == 0 {
            writer.flush()?;
        }
        Ok(())
    }

    /// One hop. Never panics and never fails twice loudly: this runs on the
    /// audio thread, where an unwrap would take the stream down mid-phrase.
    pub fn record(&mut self, hop: &[f32]) {
        if self.failed {
            return;
        }
        if let Err(error) = self.write_hop(hop) {
            self.failed = true;
            eprintln!("📼 zapis przerwany {}: {error:#}", self.path.display());
        }
    }
}

/// The windows the attack detector reads: 64 ms and 128 ms, against the CQT's
/// 512 ms. A window that long smears the very thing an attack detector looks
/// for, which is why this cannot reuse the features already being computed.
///
/// Both, not one. Measured on AtoA's 51 reviewed labels, the long window alone
/// finds 45 of them and the pair finds all 51: the short one carries the
/// transient and the long one the resolution to tell it from the room.
pub const SHORT_FFT: usize = 2048;
pub const SHORT_FFTS: [usize; 2] = [1024, SHORT_FFT];

/// Hann-windowed magnitude spectra of the most recent samples, concatenated.
///
/// It reads the tail of the chunk the CQT is already given, so no extra audio
/// is buffered and the two always describe the same instant.
pub struct ShortSpectrum {
    parts: Vec<(std::sync::Arc<dyn rustfft::Fft<f32>>, Vec<f32>, Vec<Complex<f32>>, Vec<Complex<f32>>, f32)>,
    magnitudes: Vec<f32>,
}

impl ShortSpectrum {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let parts: Vec<_> = SHORT_FFTS
            .iter()
            .map(|&size| {
                let fft = planner.plan_fft_forward(size);
                let scratch = vec![Complex { re: 0.0, im: 0.0 }; fft.get_inplace_scratch_len()];
                let window: Vec<f32> = (0..size)
                    .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
                    .collect();
                // Amplitude scaling, so the two windows are comparable: a
                // longer one sums more samples and would otherwise simply
                // weigh more, whatever the signal did.
                let scale = 2.0 / window.iter().sum::<f32>();
                (fft, window, vec![Complex { re: 0.0, im: 0.0 }; size], scratch, scale)
            })
            .collect();
        let bins = SHORT_FFTS.iter().map(|size| size / 4 + 1).sum();
        Self { parts, magnitudes: vec![0.0; bins] }
    }

    /// Linear magnitudes, not the log-compressed features: compression is what
    /// makes a quiet rise look like a loud one's rounding error.
    pub fn of(&mut self, audio: &[f32]) -> &[f32] {
        let mut offset = 0;
        for (fft, window, buffer, scratch, scale) in self.parts.iter_mut() {
            let size = window.len();
            let tail = audio.len().saturating_sub(size);
            for (i, slot) in buffer.iter_mut().enumerate() {
                let sample = audio.get(tail + i).copied().unwrap_or(0.0);
                *slot = Complex { re: sample * window[i], im: 0.0 };
            }
            fft.process_with_scratch(buffer, scratch);
            let bins = size / 4 + 1;
            for i in 0..bins {
                self.magnitudes[offset + i] = buffer[i].norm() * *scale;
            }
            offset += bins;
        }
        &self.magnitudes
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
/// last 0.77 s, and in note practice that is two or three notes at once.
///
/// It is not quicker than the model, though: one frame is 512 ms of audio, and
/// both name a new note about 0.6 s after the strike - see `docs/judge-measurements.md#mono_pitch`.
///
/// Returns the absolute semitone counted from the lowest CQT bin (`% 12` for the
/// class) and the score of the winning bin.
pub fn mono_pitch(norm_cqt: &[f32]) -> Option<(usize, f32)> {
    if norm_cqt.len() < CQT_BINS {
        return None;
    }
    let (winner, score) = strongest_bin(norm_cqt)?;
    let bin = true_fundamental(norm_cqt, winner, score);
    // The absolute semitone, counted from the lowest CQT bin, not the pitch
    // class: the octave is what tells a note struck again from the same note
    // still ringing an octave below it. Callers wanting a class take `% 12`.
    let semitone = (peak_position(norm_cqt, bin) / BINS_PER_SEMITONE as f32).round() as usize;
    Some((semitone, score))
}

/// How well bin `b` explains the frame as a fundamental: the weighted mean of
/// its first six harmonics.
fn harmonic_score(norm_cqt: &[f32], b: usize) -> f32 {
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
}

/// The bin with the best harmonic score in the guitar's range, and the score.
///
/// Every bin, quarter tones included. Restricting the search to semitone
/// centres was tried and starved the score: a real note sits a few cents off
/// and leaves most of its energy in the bin next door, which took a plain
/// reading from 1.00 down to 0.45 - under the gate, so no note reported at
/// all, and the app left waiting on the model's 0.77 s window. The quarter
/// tones are where the resolution is; what they must not do is decide the
/// semitone by rounding, which is what `peak_position` is for.
fn strongest_bin(norm_cqt: &[f32]) -> Option<(usize, f32)> {
    let mut best = (F0_LOW_BIN, 0.0f32);
    for b in F0_LOW_BIN..F0_HIGH_BIN.min(CQT_BINS) {
        let s = harmonic_score(norm_cqt, b);
        if s > best.1 {
            best = (b, s);
        }
    }
    (best.1 > 0.0).then_some(best)
}

/// The note whose partial won the sum, which may lie below the winner.
///
/// An impostor sitting on one of the note's partials collects its upper
/// harmonics as its own, so the winner is compared against every candidate that
/// would explain it, lowest first, and the first one that stands up on its own
/// bin takes it. A chromatic run of isolated plucks: 6 of 12 right before, 12 of
/// 12 after - see `docs/judge-measurements.md#true_fundamental`.
fn true_fundamental(norm_cqt: &[f32], winner: usize, score: f32) -> usize {
    const IMPOSTOR_OFFSETS: [usize; 6] = [62, 56, 48, 38, 24, 14];
    IMPOSTOR_OFFSETS
        .iter()
        .filter(|&&off| winner >= off + F0_LOW_BIN)
        .map(|&off| winner - off)
        // Looser than a plain "is it better": the bin being defended is the
        // quieter one by construction - that is why it lost - so it has to be
        // comparable rather than dominant. At 0.5 the rule fired too readily on
        // a real recording and pulled readings a fifth down; 0.8 keeps the
        // chromatic run at 12 of 12 and leaves the recording where it was.
        .find(|&lower| {
            harmonic_score(norm_cqt, lower) > score * 0.9
                && norm_cqt[lower] > norm_cqt[winner] * 0.8
        })
        .unwrap_or(winner)
}

/// Where the peak around bin `b` really lies, to a fraction of a bin.
///
/// Three scores around it describe a parabola, and its vertex says how far off
/// centre the note sits - which is what decides the semitone. Rounding the
/// winning bin instead answered every quarter-tone reading with the note ABOVE
/// (`round(18.5)` goes away from zero), so an F# a little sharp came back as
/// G, and in Formulas - where a mark never expires - the flat second lit up
/// every time the root was struck.
fn peak_position(norm_cqt: &[f32], b: usize) -> f32 {
    if b == 0 || b + 1 >= CQT_BINS {
        return b as f32;
    }
    let (l, c, r) = (
        harmonic_score(norm_cqt, b - 1),
        harmonic_score(norm_cqt, b),
        harmonic_score(norm_cqt, b + 1),
    );
    let denom = l - 2.0 * c + r;
    if denom.abs() <= 1e-6 {
        return b as f32;
    }
    let delta = 0.5 * (l - r) / denom;
    // Beyond half a bin the parabola is describing something other than this
    // peak, and the shift is not to be trusted.
    if delta.abs() <= 0.5 { b as f32 + delta } else { b as f32 }
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
    
    let mut process = live_processor(shared_state, mic_sr, channels, channel)?;

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

/// Everything the live input does with a buffer from the device, as the one
/// callback the stream is built with.
fn live_processor(
    shared: Arc<Mutex<AudioAnalysis>>,
    mic_sr: u32,
    channels: usize,
    channel: usize,
) -> Result<impl FnMut(&[f32])> {
    let mut listener = Listener::new(shared)?;
    // Clamped here rather than per frame: a saved channel from a device with
    // more inputs than this one must not silently read past the end.
    let pick = channel.clamp(1, channels.max(1)) - 1;
    let mut resampler = Resampler::new(mic_sr, channels, pick);
    Ok(move |data: &[f32]| resampler.push(data, |window, hop| listener.hear(window, hop)))
}

/// The chosen input channel, resampled to `TARGET_SR`, and handed on one hop
/// at a time together with the analysis window that ends with it.
struct Resampler {
    channels: usize,
    pick: usize,
    /// Input samples per output sample.
    ratio: f32,
    input: Vec<f32>,
    output: Vec<f32>,
    /// Where in `input` the next output sample is read, between two samples.
    read_pos: f32,
}

impl Resampler {
    fn new(mic_sr: u32, channels: usize, pick: usize) -> Self {
        Resampler {
            channels,
            pick,
            ratio: mic_sr as f32 / TARGET_SR as f32,
            input: Vec::with_capacity(8192 * 2),
            output: Vec::with_capacity(FFT_SIZE * 2),
            read_pos: 0.0,
        }
    }

    /// Takes one buffer from the device and calls `on_hop(window, hop)` for
    /// every hop it completes.
    fn push(&mut self, data: &[f32], mut on_hop: impl FnMut(&[f32], &[f32])) {
        let pick = self.pick;
        self.input
            .extend(data.chunks(self.channels).map(|frame| frame.get(pick).copied().unwrap_or(0.0)));

        // Linear interpolation between the two input samples either side.
        while self.read_pos + 1.0 < self.input.len() as f32 {
            let i = self.read_pos as usize;
            let frac = self.read_pos - i as f32;
            let (s0, s1) = (self.input[i], self.input[i + 1]);
            self.output.push(s0 + frac * (s1 - s0));
            self.read_pos += self.ratio;

            let len = self.output.len();
            if len >= FFT_SIZE && len % HOP_LENGTH == 0 {
                on_hop(&self.output[len - FFT_SIZE..], &self.output[len - HOP_LENGTH..]);
            }
        }
        self.drop_what_is_used();
    }

    /// Input already read past, and output older than one window.
    fn drop_what_is_used(&mut self) {
        if self.read_pos > 0.0 {
            let consumed = (self.read_pos as usize).min(self.input.len());
            self.input.drain(0..consumed);
            self.read_pos -= consumed as f32;
        }
        if self.output.len() > FFT_SIZE * 2 {
            let surplus = self.output.len().saturating_sub(FFT_SIZE);
            self.output.drain(0..surplus);
        }
    }
}

// Attack detection: an energy jump above a slowly creeping envelope.
const ATTACK_RATIO: f32 = 1.8; // multiple of the baseline
const ATTACK_FLOOR: f32 = 0.01; // absolute floor, rejects noise
const ATTACK_REFRACTORY: u32 = 12; // ~0.2 s of detector silence after an attack

/// A strum as an energy jump above a slow envelope. Compared against an EMA
/// baseline rather than an absolute threshold, which would depend on playing
/// volume. The refractory period keeps one strum from firing several attacks.
struct AttackDetector {
    baseline: f32,
    hops_since_attack: u32,
}

impl AttackDetector {
    fn new() -> Self {
        AttackDetector { baseline: 0.0, hops_since_attack: ATTACK_REFRACTORY }
    }

    /// Whether this hop's level is an attack.
    fn push(&mut self, level: f32) -> bool {
        let attack = level > self.baseline * ATTACK_RATIO
            && level > ATTACK_FLOOR
            && self.hops_since_attack >= ATTACK_REFRACTORY;
        if attack {
            self.hops_since_attack = 0;
        } else {
            self.hops_since_attack = self.hops_since_attack.saturating_add(1);
        }
        // The baseline falls fast and rises slowly; otherwise a long sustained
        // chord would raise the threshold and swallow the next attack.
        self.baseline = if level > self.baseline {
            self.baseline * 0.90 + level * 0.10
        } else {
            self.baseline * 0.70 + level * 0.30
        };
        attack
    }
}

/// What the live input makes of each hop: the meter, the three attack
/// detectors, and - past the noise gate - the model's features, the ear and
/// the voices, all written into the shared `AudioAnalysis`.
struct Listener {
    shared: Arc<Mutex<AudioAnalysis>>,
    analyzer: CqtAnalyzer,
    short: ShortSpectrum,
    flux: crate::flux::Flux,
    capture: Option<Capture>,
    /// Optional: without the file the app judges as it did before the
    /// detector existed, and says so once.
    strikes: Option<crate::strike::Strikes>,
    attack: AttackDetector,
    /// SOLITITO_EAR=1: print every reading of the ear - see `print_ear`.
    loud_ear: bool,
}

impl Listener {
    fn new(shared: Arc<Mutex<AudioAnalysis>>) -> Result<Self> {
        let analyzer = CqtAnalyzer::new("dsp_weights.json")?;
        let short = ShortSpectrum::new();
        let flux = crate::flux::Flux::default();
        let capture = Capture::from_env();
        let strikes = match crate::strike::Strikes::load_any() {
            Ok((s, path)) => {
                eprintln!("🎯 {path} · próg {:.2}", s.latch.threshold);
                Some(s)
            }
            Err(e) => {
                eprintln!("🎯 brak modelu uderzeń ({e}) - powtórzenia oceniane bez niego");
                None
            }
        };
        Ok(Listener {
            shared,
            analyzer,
            short,
            flux,
            capture,
            strikes,
            attack: AttackDetector::new(),
            loud_ear: std::env::var("SOLITITO_EAR").is_ok(),
        })
    }

    /// One hop: `window` is the last `FFT_SIZE` samples, `hop` its newest
    /// `HOP_LENGTH`.
    fn hear(&mut self, window: &[f32], hop: &[f32]) {
        if let Some(capture) = self.capture.as_mut() {
            capture.record(hop);
        }
        // The journal's clock. Bumped here and nowhere else, so a row's hop is
        // a sample position in the recording even though `frames_seen` counts
        // only what passed the gate.
        if let Ok(mut st) = self.shared.lock() {
            st.hops = st.hops.wrapping_add(1);
            st.flux_age = st.flux_age.saturating_add(1);
        }
        let (gate, boost_enabled, boost_gain) = {
            let s = self.shared.lock().unwrap();
            (s.noise_gate, s.bass_boost_enabled, s.bass_boost_gain)
        };

        // The gate and meter level come from RAW rms, without INPUT_GAIN, so
        // the scale is true input dBFS (0 dB = full scale) rather than "dBFS
        // plus 6", which pushed the whole useful range to the end of the
        // slider on a loud microphone.
        let level = (window.iter().map(|x| x * x).sum::<f32>() / FFT_SIZE as f32).sqrt();
        self.show_level(level);
        if self.attack.push(level) {
            if let Ok(mut st) = self.shared.lock() {
                st.mark_onset();
            }
        }
        // Outside the noise gate, as the envelope's own detector is: a quiet
        // attack is exactly what both are here to catch, and the gate is a
        // level test of the kind that misses it.
        if self.flux.push(self.short.of(window)).1 {
            if let Ok(mut st) = self.shared.lock() {
                st.mark_flux();
            }
        }
        self.listen_for_strikes(hop);

        if level > gate {
            self.analyse(window, boost_enabled, boost_gain);
        } else {
            self.go_quiet();
        }
    }

    /// Fast attack, slow release: the meter should show the strum peak, not
    /// the average of the gaps between chords.
    fn show_level(&self, level: f32) {
        if let Ok(mut st) = self.shared.lock() {
            st.input_level = if level > st.input_level {
                level
            } else {
                st.input_level * 0.92 + level * 0.08
            };
        }
    }

    fn listen_for_strikes(&mut self, hop: &[f32]) {
        let Some(detector) = self.strikes.as_mut() else {
            return;
        };
        match detector.push(hop) {
            Ok((_, struck)) => {
                if let Ok(mut st) = self.shared.lock() {
                    st.mark_strikes(&struck);
                }
            }
            Err(e) => {
                eprintln!("🎯 strike detector stopped: {e}");
                self.strikes = None;
            }
        }
    }

    /// Past the noise gate: the frame's features into the model's window, and
    /// what the ear and the voices make of it.
    fn analyse(&mut self, window: &[f32], boost_enabled: bool, boost_gain: f32) {
        let amplified: Vec<f32> = window.iter().map(|&x| x * INPUT_GAIN).collect();
        let (cqt, chroma, bass, visual) =
            self.analyzer.compute_cqt_chroma(&amplified, boost_enabled, boost_gain);

        // One frame, one note: no window memory, so a scale is tracked instead
        // of smeared. Weak answers are dropped - in noise this estimator always
        // names something.
        let heard = mono_pitch(&cqt);
        if self.loud_ear {
            print_ear(heard);
        }
        let mono = heard.filter(|&(_, s)| s >= MONO_MIN_SCORE).map(|(n, _)| n);
        // Four is a grip: more voices than that in one frame is not a thing
        // six strings under one hand can be.
        let sounding = crate::voices::voices(&cqt, 4)
            .iter()
            .fold(0u16, |mask, &(semitone, _)| mask | 1 << (semitone % 12));

        let Ok(mut state) = self.shared.lock() else {
            return;
        };
        state.voices = sounding;
        state.cqt_pitch = mono.map(|n| n % 12);
        state.cqt_semitone = mono;
        state.gate_open = true;
        let mut frame = Vec::with_capacity(TOTAL_FEATURES);
        frame.extend_from_slice(&cqt);
        frame.extend_from_slice(&chroma);
        frame.extend_from_slice(&bass);
        state.push_frame(&frame);
        for (shown, &v) in state.spectrum_visual.iter_mut().zip(&visual) {
            *shown = v;
        }
        state.chroma_sum = chroma.try_into().unwrap_or([0.0; 12]);
    }

    /// In silence an empty frame is pushed to advance the history. Everything
    /// read off a frame goes with it: `voices` used to keep the last sounding
    /// classes through a closed gate, and the judge's clock goes on ticking
    /// here, so it counted them as still in the air.
    fn go_quiet(&self) {
        let Ok(mut state) = self.shared.lock() else {
            return;
        };
        state.cqt_pitch = None;
        state.cqt_semitone = None;
        state.voices = 0;
        state.gate_open = false;
        state.push_silence();
        for x in &mut state.spectrum_visual {
            *x *= 0.7;
        }
    }
}

/// SOLITITO_EAR=1: every reading with its score, the ones under the gate
/// included. A note played softly has its upper partials down in the noise and
/// scores lower for it, and this is the only way to see by how much before
/// moving the gate.
fn print_ear(heard: Option<(usize, f32)>) {
    let Some((pc, score)) = heard else {
        return;
    };
    println!(
        "ucho {:<3} wynik={score:.2}{}",
        crate::NOTE_NAMES[pc % 12],
        if score >= MONO_MIN_SCORE { "" } else { "   << pod bramka" },
    );
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
        let mut short = ShortSpectrum::new();
        let mut flux = crate::flux::Flux::default();
        let mut strikes = crate::strike::Strikes::load_any().ok().map(|(s, _)| s);
        let mut pos = 0;
        
        while pos + FFT_SIZE < samples.len() {
            let start = Instant::now();
            let chunk = &samples[pos..pos+FFT_SIZE];
            let (cqt, chroma, bass, _) = analyzer.compute_cqt_chroma(chunk, true, 5.0);
            // The same ear and the same attack detector as the live path, so a
            // recording replayed here is judged the way it was when played.
            // Without these a replay pushes features and nothing else, and the
            // ear - which decides almost everything - is simply absent.
            let heard = mono_pitch(&cqt)
                .filter(|&(_, score)| score >= MONO_MIN_SCORE)
                .map(|(note, _)| note);
            let attack = flux.push(short.of(chunk)).1;
            let struck = strikes
                .as_mut()
                .and_then(|d| d.push(&chunk[FFT_SIZE - HOP_LENGTH..]).ok())
                .map(|(_, s)| s);
            
            let sounding = crate::voices::voices(&cqt, 4)
                .iter()
                .fold(0u16, |mask, &(semitone, _)| mask | 1 << (semitone % 12));
            if let Ok(mut state) = shared_state.lock() {
                state.voices = sounding;
                state.cqt_pitch = heard.map(|n| n % 12);
                state.cqt_semitone = heard;
                state.gate_open = true;
                state.hops = state.hops.wrapping_add(1);
                state.flux_age = state.flux_age.saturating_add(1);
                if attack {
                    state.mark_flux();
                    state.mark_onset();
                }
                if let Some(struck) = &struck {
                    state.mark_strikes(struck);
                }
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
            flux_id: 0,
            flux_age: u32::MAX,
            hops: 0,
            voices: 0,
            class_strikes: [0; 12],
            class_strike_age: [u32::MAX; 12],
            strikes_live: false,
            frames_since_onset: 0,
            spectrum_visual: [0.0; 48],
            chroma_sum: [0.0; 12],
            bass_boost_enabled: false,
            bass_boost_gain: 1.0,
            noise_gate: 0.0,
            input_level: 0.0,
            cqt_pitch: None,
            cqt_semitone: None,
            gate_open: false,
            frames_seen: 0,
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
        let (pc, _) = mono_pitch(&cqt).map(|(n, s)| (n % 12, s)).expect("a note with a fundamental present");
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
        let (pc, score) = mono_pitch(&cqt).map(|(n, s)| (n % 12, s)).expect("a note with every harmonic present");
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
        let (pc, _) = mono_pitch(&cqt).map(|(n, s)| (n % 12, s)).expect("a note with every harmonic present");
        assert_eq!(pc, 2, "D read as {}", NOTE_NAMES_TEST[pc]);
    }

    /// A weak fundamental does not hand the note to one of its own partials.
    ///
    /// On a low string the fundamental is often quieter than the partials above
    /// it, and an impostor sitting on one of those collects the rest of the
    /// series: a fifth above shares the note's third and sixth, and two octaves
    /// and a major third - the fifth harmonic - shares what is above that.
    /// Measured on a chromatic run of isolated plucks, the estimate named a
    /// fifth or a third above for six notes of twelve.
    #[test]
    fn a_quiet_fundamental_keeps_its_note() {
        // A2: semitone 21 above C1, so bin 42.
        const A2: usize = 42;
        for impostor in [14usize, 56] {
            let mut cqt = vec![0.0f32; CQT_BINS];
            for (h, off) in HARMONIC_OFFSETS.iter().enumerate() {
                let idx = A2 + off;
                if idx < CQT_BINS {
                    // The fundamental at a fraction of its own partials, which
                    // is what a low string actually looks like.
                    cqt[idx] = if h == 0 { 0.45 } else { 1.0 };
                }
            }
            // And the impostor's bin carrying something of its own, as it would
            // from the note's spread.
            if A2 + impostor < CQT_BINS {
                cqt[A2 + impostor] = 0.5;
            }
            let (pc, _) = mono_pitch(&cqt).map(|(n, s)| (n % 12, s)).expect("a note with its harmonics present");
            assert_eq!(pc, 9, "A read as {} with an impostor {impostor} bins up",
                       NOTE_NAMES_TEST[pc]);
        }
    }

    const NOTE_NAMES_TEST: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
}
#[cfg(test)]
mod voices_diagnostics {
    use super::*;

    /// How many pitch classes the CQT lights up, over a recording.
    ///
    /// The question the interval guard needs answered: can "several notes are
    /// sounding together" be told from "one note is sounding and the model is
    /// spreading it"? The model's own pitch head cannot say, because spreading
    /// is what it does. The CQT can, if the count separates.
    ///
    /// SOLITITO_VOICES_WAV=file.wav
    #[test]
    #[ignore = "diagnostic: requires SOLITITO_VOICES_WAV"]
    fn count_voices() -> anyhow::Result<()> {
        let path = std::env::var("SOLITITO_VOICES_WAV")?;
        let mut reader = hound::WavReader::open(&path)?;
        let spec = reader.spec();
        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
            _ => reader.samples::<i16>().map(|s| s.unwrap_or(0) as f32 / 32768.0).collect(),
        };
        let mono: Vec<f32> = raw.chunks(spec.channels as usize).map(|f| f[0]).collect();
        let ratio = spec.sample_rate as f32 / TARGET_SR as f32;
        let mut signal = Vec::with_capacity((mono.len() as f32 / ratio) as usize + 8);
        let mut read = 0.0f32;
        while read + 1.0 < mono.len() as f32 {
            let i = read as usize;
            let f = read - i as f32;
            signal.push(mono[i] + f * (mono[i + 1] - mono[i]));
            read += ratio;
        }
        let mut analyzer = CqtAnalyzer::new("dsp_weights.json")?;
        let fractions = [0.3f32, 0.5, 0.7];
        let mut counts: Vec<Vec<usize>> = vec![vec![0; 13]; fractions.len()];
        let mut frames = 0usize;
        let mut pos = 0usize;
        while pos + FFT_SIZE < signal.len() {
            let (_, chroma, _, _) = analyzer.compute_cqt_chroma(&signal[pos..pos + FFT_SIZE], true, 5.0);
            let peak = chroma.iter().cloned().fold(0.0f32, f32::max);
            // Only frames with something in them: the gaps say nothing.
            if peak > 1e-3 {
                frames += 1;
                for (f, &fraction) in fractions.iter().enumerate() {
                    let n = chroma.iter().filter(|&&v| v >= peak * fraction).count();
                    counts[f][n.min(12)] += 1;
                }
            }
            pos += HOP_LENGTH * 4;
        }
        println!("{}: {frames} ramek z dzwiekiem", path.rsplit('/').next().unwrap_or(&path));
        // The other candidate witness: does the monophonic ear MOVE between
        // classes after a strum, where after one pluck it would sit still?
        let mut pos2 = 0usize;
        let mut heard: Vec<Option<usize>> = Vec::new();
        while pos2 + FFT_SIZE < signal.len() {
            let (cqt, _, _, _) = analyzer.compute_cqt_chroma(&signal[pos2..pos2 + FFT_SIZE], true, 5.0);
            heard.push(mono_pitch(&cqt).map(|(pc, _)| pc));
            pos2 += HOP_LENGTH * 4;
        }
        // Distinct classes the ear names in each half-second window.
        let window = (0.5 * TARGET_SR as f32 / (HOP_LENGTH * 4) as f32) as usize;
        let mut distinct: Vec<usize> = Vec::new();
        for start in (0..heard.len().saturating_sub(window)).step_by(window) {
            let set: std::collections::BTreeSet<usize> =
                heard[start..start + window].iter().flatten().cloned().collect();
            if !set.is_empty() {
                distinct.push(set.len());
            }
        }
        distinct.sort_unstable();
        let at = |q: f32| distinct.get((distinct.len() as f32 * q) as usize).copied().unwrap_or(0);
        println!(
            "  ucho w oknie 0,5 s: mediana {} klas, 25% {} , 75% {} (okien {})",
            at(0.5), at(0.25), at(0.75), distinct.len(),
        );
        for (f, &fraction) in fractions.iter().enumerate() {
            let total: usize = counts[f].iter().sum();
            let median = {
                let mut seen = 0;
                (0..=12).find(|&n| { seen += counts[f][n]; seen * 2 >= total }).unwrap_or(0)
            };
            let share = |n: usize| 100.0 * counts[f][n] as f32 / total.max(1) as f32;
            println!(
                "  >= {fraction:.1} szczytu: mediana {median} klas, 1 klasa {:.0}%, 2 {:.0}%, 3+ {:.0}%",
                share(1), share(2), (3..=12).map(share).sum::<f32>(),
            );
        }
        Ok(())
    }
}
