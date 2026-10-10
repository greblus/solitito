//! Which string was struck, from the small causal onset model trained for it.
//!
//! `short_onset_rise.onnx`, trained by `dist/train_short_onset.py` on GuitarSet
//! (it came over from the rise branch). Two short windows of the newest audio -
//! 64 and 128 ms - log-compressed, 35 frames of history, and twelve per-class
//! probabilities for the newest frame. The onset branch of a take7 file has the
//! same contract and runs here too.
//!
//! What this is for: everything else in the app answers "a string was hit"
//! (flux, pitch-blind) or "this class is sounding" (the ear, `voices`), and the
//! repeat rules need the conjunction - THIS class was hit just now. The onset
//! head of the main model was asked that and answered it badly: on AtoA's 51
//! notes it was right 33 times and named the wrong class 50 times, because its
//! trunk is trained for chord identity, which is exactly what is invariant to
//! which string was struck. This model, on the same notes: 46 right, one wrong.

use anyhow::{ensure, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::collections::VecDeque;
use std::sync::Arc;

/// The rate the model was trained at. The app's own resampler feeds it; this
/// is for the diagnostics, which resample the way the trainer did.
#[cfg(test)]
pub const SR: u32 = 16_000;
pub const HOP: usize = 256;
const FEATURES: usize = 770;
const HISTORY: usize = 35;

/// The onset branch of `best_model_v2_take7_masking_v2.onnx`, cut out of the
/// combined file into one that does not run the chord trunk: 1 MB and 0.7 ms a
/// hop against 30 MB and the trunk's 38 ms. Trained on masking pairs - a note
/// struck under another one ringing - and it shows exactly there: on AtoA's
/// notes spliced with themselves it gave no repeat to a note that only rang,
/// where `short_onset_rise` gave three, and on the user's own playing it found
/// 81 of 87 notes against 76.
pub const MODEL: &str = "short_onset_masking_v2.onnx";

/// Tried in order: the first file present is used.
pub const MODELS: [&str; 2] = [MODEL, "short_onset_rise.onnx"];

impl Strikes {
    /// The best strike model on disk, or why there is none.
    pub fn load_any() -> Result<(Self, &'static str)> {
        let mut last = None;
        for path in MODELS {
            match Self::load(path) {
                Ok(s) => return Ok((s, path)),
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap_or_else(|| anyhow::anyhow!("no strike model")))
    }
}

/// The latch, as the model was evaluated with it: a class re-arms once its
/// probability falls under three tenths of the peak that fired it.
const REARM: f32 = 0.3;
const FLOOR: f32 = 0.1;

/// Hops after a strike of a class during which that class is not reported
/// again: 0.6 s.
///
/// The model fires on decaying notes: on the user's own playing, levelled, it
/// reported a class again less than 0.6 s after itself 58 times, and at those
/// moments the signal was LOSING energy - a median 0.95 of what it had been,
/// against 3.3 at first firings. In the exercises a class comes back only after
/// a credit, the 0.35 s the finished set is shown for and the player's reply,
/// so nothing real is lost to this.
///
/// A gate on the class's own energy rising was tried instead, twice, and both
/// times refused real re-strikes. The model fires about 26 ms after the attack,
/// when the 128 ms window holds little of the new note and the old one has just
/// been stopped by the pick, so the energy at that moment is FALLING: on AtoA's
/// notes struck again 0.8 s later it let 3 of 51 through where this lets 48.
pub const REFRACTORY: usize = 38;

/// A class reported again within this many hops of its last strike - 2 s -
/// has to show the energy RISING around the new one before it counts.
///
/// The refractory does not reach far enough: the model also fires on decaying
/// notes later than 0.6 s. On the user's three recordings it fired a class
/// again within 2 s of itself 37 times, and they fall in two groups with
/// nothing between: 29 at an energy ratio of 0.89 to 1.03 - the note dying
/// away, the false repeat the user reported among them, at 0.67 s - and 8 at
/// 3.7 to 48.8, a string struck again. Only one of the 29 happened to land when
/// that class was being asked for, which is why it was seen once.
const RECHECK: usize = 125;

/// Least energy rise a re-fire within `RECHECK` must show. The gap it sits in
/// runs from 1.03 to 3.7. At 1.5 it also refused 6 of 51 notes of AtoA struck
/// again after 0.8 s - a slower-decaying guitar, whose second attack stands
/// less far above the first one's sustain - and at 1.25 none.
const REFIRE_RISE: f32 = 1.25;

/// Hops waited after a re-fire before deciding it: 32 ms, so the attack the
/// model reacted to is in the measurement. A gate on energy tried earlier
/// failed exactly for deciding at the firing, when the new note had barely
/// entered the window.
const LOOK_AHEAD: usize = 2;

/// The level the model was measured at its best on: AtoA's playing RMS.
///
/// Its features are `ln(1 + 1000 |X|)`, so they are not level-invariant, and a
/// quieter input reads as a weaker attack. Measured on the user's own capture,
/// 11 dB quieter than AtoA: the right class peaked at a median 0.79 against a
/// threshold of 0.8, and half the notes were missed - 42 of 87 found, against
/// 76 once levelled.
const REFERENCE_RMS: f32 = 0.0325;

/// How slowly the level is followed, per hop of 16 ms: about four seconds to
/// settle. Slow on purpose - a follower fast enough to track a single note
/// would flatten the very rise the model is looking for.
const LEVEL_FOLLOW: f32 = 0.004;

/// Below this the window is the room, not the playing, and does not move the
/// level estimate.
const PLAYING_FLOOR: f32 = 0.002;

/// Limits on the correction: a fifth of the level to eight times it.
const GAIN_RANGE: (f32, f32) = (0.2, 8.0);

/// Brings the playing to the level the model was trained at, slowly.
#[derive(Default)]
pub struct Leveller {
    level: Option<f32>,
}

impl Leveller {
    /// The gain for this hop, from the level BEFORE it - a gain that moved
    /// with the hop it scales would read its own attack as a change of level -
    /// and then this hop folded into the level.
    pub fn gain_for(&mut self, hop: &[f32]) -> f32 {
        let gain = match self.level {
            Some(level) => (REFERENCE_RMS / level).clamp(GAIN_RANGE.0, GAIN_RANGE.1),
            None => 1.0,
        };
        let rms = (hop.iter().map(|v| v * v).sum::<f32>() / hop.len().max(1) as f32).sqrt();
        if rms > PLAYING_FLOOR {
            self.level = Some(match self.level {
                Some(level) => level + LEVEL_FOLLOW * (rms - level),
                None => rms,
            });
        }
        gain
    }
}

/// Turns twelve probabilities a hop into strikes: a crossing of the threshold,
/// once per peak, and not again for `refractory` hops.
pub struct Latch {
    pub threshold: f32,
    pub refractory: usize,
    armed: [bool; 12],
    peaks: [f32; 12],
    since: [usize; 12],
}

impl Latch {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            refractory: REFRACTORY,
            armed: [true; 12],
            peaks: [0.0; 12],
            since: [usize::MAX; 12],
        }
    }

    pub fn step(&mut self, p: &[f32; 12]) -> Vec<usize> {
        let mut struck = Vec::new();
        for pc in 0..12 {
            self.since[pc] = self.since[pc].saturating_add(1);
            if self.armed[pc] && p[pc] >= self.threshold {
                self.armed[pc] = false;
                self.peaks[pc] = p[pc];
                if self.since[pc] >= self.refractory {
                    struck.push(pc);
                    self.since[pc] = 0;
                }
            } else if p[pc] < (REARM * self.peaks[pc]).max(FLOOR) {
                self.armed[pc] = true;
            }
        }
        struck
    }
}

struct Spectrum {
    fft: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
    scale: f64,
    buffer: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl Spectrum {
    fn new(size: usize) -> Self {
        let fft = FftPlanner::new().plan_fft_forward(size);
        let window: Vec<f64> = (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos()))
            .collect();
        Self {
            scale: 2.0 / window.iter().sum::<f64>(),
            scratch: vec![Complex::default(); fft.get_inplace_scratch_len()],
            buffer: vec![Complex::default(); size],
            window,
            fft,
        }
    }
}

/// Holds a class fired again soon after its last strike until the energy has
/// shown whether it was struck or is dying away - see `REFIRE_RISE`.
pub struct Refires {
    /// Raw RMS of the last few hops, newest last.
    rms: VecDeque<f32>,
    /// Hops since each class's last ACCEPTED strike.
    since_strike: [usize; 12],
    /// Re-fires waiting for `LOOK_AHEAD`: the class and the hops waited so far.
    pending: Vec<(usize, usize)>,
    pub rise: f32,
}

impl Default for Refires {
    fn default() -> Self {
        Self {
            rms: VecDeque::with_capacity(16),
            since_strike: [usize::MAX; 12],
            pending: Vec::new(),
            rise: REFIRE_RISE,
        }
    }
}

impl Refires {
    /// One hop of raw audio: kept for the energy question below.
    pub fn hear(&mut self, hop: &[f32]) {
        if self.rms.len() == 16 {
            self.rms.pop_front();
        }
        self.rms.push_back((hop.iter().map(|v| v * v).sum::<f32>() / hop.len().max(1) as f32).sqrt());
    }

    /// First strikes of a class pass straight through; a class fired again
    /// within `RECHECK` of its last strike waits `LOOK_AHEAD` hops and passes
    /// only if the energy rose around it.
    pub fn confirm(&mut self, candidates: Vec<usize>) -> Vec<usize> {
        for since in &mut self.since_strike {
            *since = since.saturating_add(1);
        }
        let mut struck = Vec::new();
        let mut waiting = std::mem::take(&mut self.pending);
        waiting.retain_mut(|(pc, waited)| {
            *waited += 1;
            if *waited < LOOK_AHEAD {
                return true;
            }
            if self.rose() {
                struck.push(*pc);
                self.since_strike[*pc] = 0;
            }
            false
        });
        self.pending = waiting;
        for pc in candidates {
            if self.since_strike[pc] < RECHECK {
                self.pending.push((pc, 0));
            } else {
                struck.push(pc);
                self.since_strike[pc] = 0;
            }
        }
        struck
    }

    /// Whether the energy around a re-fire decided now - `LOOK_AHEAD` hops
    /// after it - rose: the loudest of the hops from two before it to now,
    /// against the mean of the four that ended four hops before it.
    fn rose(&self) -> bool {
        let len = self.rms.len();
        let fired = LOOK_AHEAD;
        if len < fired + 9 {
            return true;
        }
        let at = |back: usize| self.rms[len - 1 - back];
        let after = (0..=fired + 2).map(at).fold(0.0f32, f32::max);
        let before = (fired + 5..=fired + 8).map(at).sum::<f32>() / 4.0;
        after >= self.rise * before.max(1e-9)
    }
}

/// The model's input, one frame per hop: two Hann windows of the newest audio,
/// 1024 and 2048 samples, magnitudes up to 4 kHz, `ln(1 + 1000 a) / ln(1001)`,
/// rounded to binary16. The trainer's `onset_features` computes the same thing,
/// and a test holds the two to it.
pub struct ShortFeatures {
    spectra: [Spectrum; 2],
    audio: VecDeque<f32>,
}

impl Default for ShortFeatures {
    fn default() -> Self {
        Self {
            spectra: [Spectrum::new(1024), Spectrum::new(2048)],
            audio: VecDeque::from(vec![0.0; 2048]),
        }
    }
}

impl ShortFeatures {
    /// One hop in, the frame that ends with it out. The window starts as
    /// silence, so the first frames see zeros before the first sample - the
    /// trainer pads the same way.
    pub fn push(&mut self, hop: &[f32]) -> [f32; FEATURES] {
        self.audio.drain(..hop.len().min(2048));
        self.audio.extend(hop.iter().map(|&v| if v.is_finite() { v } else { 0.0 }));
        let audio = self.audio.make_contiguous();
        let mut features = [0.0f32; FEATURES];
        let mut offset = 0;
        for spectrum in &mut self.spectra {
            let size = spectrum.window.len();
            for (i, &sample) in audio[2048 - size..].iter().enumerate() {
                spectrum.buffer[i] = Complex::new(sample as f64 * spectrum.window[i], 0.0);
            }
            spectrum.fft.process_with_scratch(&mut spectrum.buffer, &mut spectrum.scratch);
            let bins = size / 4 + 1;
            for (i, value) in spectrum.buffer[..bins].iter().enumerate() {
                let magnitude = value.norm() * spectrum.scale;
                features[offset + i] = cache_precision((1000.0 * magnitude).ln_1p() / 1001f64.ln());
            }
            offset += bins;
        }
        features
    }
}

pub struct Strikes {
    session: Session,
    /// A take7 file carries the chord trunk beside the onset branch, and wants
    /// its input fed even though the onset answer does not depend on it
    /// (measured: identical to the last digit with two different trunk inputs).
    combined: bool,
    features: ShortFeatures,
    history: VecDeque<[f32; FEATURES]>,
    leveller: Leveller,
    pub latch: Latch,
    pub refires: Refires,
    /// Off, the input is fed as it comes - for measuring what levelling buys.
    pub levelled: bool,
}

impl Strikes {
    pub fn load(path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(path)?;
        // The trainer writes the operating point into the file; 0.8 is what it
        // used before it learned to.
        let threshold = match session.metadata()?.custom("onset_threshold")? {
            Some(value) => value.parse::<f32>()?,
            None => 0.8,
        };
        ensure!(threshold > 0.0 && threshold < 1.0, "onset_threshold out of range");
        let combined = session.inputs.iter().any(|i| i.name == "features");
        Ok(Self {
            session,
            combined,
            features: ShortFeatures::default(),
            history: VecDeque::with_capacity(HISTORY),
            leveller: Leveller::default(),
            latch: Latch::new(threshold),
            refires: Refires::default(),
            levelled: true,
        })
    }

    /// One hop of 16 kHz audio. Returns the twelve probabilities and the
    /// classes whose strike was reported on this hop.
    pub fn push(&mut self, hop: &[f32]) -> Result<([f32; 12], Vec<usize>)> {
        ensure!(hop.len() == HOP, "a hop is {HOP} samples");
        self.refires.hear(hop);
        let gain = self.leveller.gain_for(hop);
        let gain = if self.levelled { gain } else { 1.0 };
        let levelled: Vec<f32> = hop.iter().map(|&v| v * gain).collect();
        let features = self.features.push(&levelled);
        if self.history.len() == HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(features);

        let time = self.history.len();
        let mut flat = vec![0.0f32; FEATURES * time];
        for (t, row) in self.history.iter().enumerate() {
            for (bin, &value) in row.iter().enumerate() {
                flat[bin * time + t] = value;
            }
        }
        let input = Value::from_array((vec![1i64, FEATURES as i64, time as i64], flat))?;
        // The session's output borrows the session; the answer is copied out
        // and the borrow ends here, before the latch is touched.
        let probabilities = {
            let output = if self.combined {
                let trunk = Value::from_array((vec![1i64, 48, 168], vec![0.0f32; 48 * 168]))?;
                self.session.run(ort::inputs!["features" => trunk, "short_features" => input])?
            } else {
                self.session.run(ort::inputs!["short_features" => input])?
            };
            let (shape, logits) = output["onset_logits"].try_extract_tensor::<f32>()?;
            ensure!(shape.as_ref() == [1, 12, time as i64], "unexpected onset output {shape:?}");
            let mut p = [0.0f32; 12];
            for (pc, v) in p.iter_mut().enumerate() {
                *v = 1.0 / (1.0 + (-logits[pc * time + time - 1]).exp());
            }
            p
        };
        let candidates = self.latch.step(&probabilities);
        Ok((probabilities, self.refires.confirm(candidates)))
    }
}

/// The training cache rounds features to binary16, and the model learned on
/// that. Kept without a dependency for one rounding.
fn cache_precision(value: f64) -> f32 {
    let step = if value < 2f64.powi(-14) {
        2f64.powi(-24)
    } else {
        2f64.powi(value.log2().floor() as i32 - 10)
    };
    ((value / step).round_ties_even() * step) as f32
}

/// The trainer's resampling: linear, one input sample late. Matching it matters
/// less than it sounds - 23 us at 44.1 kHz - but a model is best measured on
/// what it was fed.
#[cfg(test)]
pub fn resample_like_trainer(input: &[f32], rate: u32) -> Vec<f32> {
    if rate == SR {
        return input.to_vec();
    }
    let total = input.len() as u64 * SR as u64 / rate as u64;
    (0..total)
        .map(|n| {
            let position = n as f64 * (rate as f64 / SR as f64) - 1.0;
            let lower = position.floor() as i64;
            let fraction = position - lower as f64;
            let at = |i: i64| if i < 0 { 0.0 } else { input.get(i as usize).copied().unwrap_or(0.0) as f64 };
            let a = at(lower);
            let b = if fraction == 0.0 { a } else { at(lower + 1) };
            (a + fraction * (b - a)) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A strike is a crossing, reported once per peak.
    #[test]
    fn a_class_is_reported_once_per_crossing() {
        let mut latch = Latch::new(0.8);
        let mut p = [0.0f32; 12];
        p[4] = 0.9;
        assert_eq!(latch.step(&p), vec![4]);
        // Still high: the same peak, not a new strike.
        assert!(latch.step(&p).is_empty());
    }

    /// The fault the refractory exists for: the model fires on a decaying
    /// note, and the probability dips and comes back within the same note.
    #[test]
    fn a_class_is_not_reported_again_within_the_refractory() {
        let mut latch = Latch::new(0.8);
        let mut p = [0.0f32; 12];
        p[7] = 0.95;
        assert_eq!(latch.step(&p), vec![7]);
        // Dips under the re-arm level and comes back, well inside 0.6 s.
        for _ in 0..10 {
            p[7] = 0.05;
            latch.step(&p);
        }
        p[7] = 0.95;
        assert!(latch.step(&p).is_empty(), "a decay re-fire was reported");
    }

    /// And after it, a real re-strike is reported.
    #[test]
    fn a_class_struck_again_after_the_refractory_is_reported() {
        let mut latch = Latch::new(0.8);
        let mut p = [0.0f32; 12];
        p[2] = 0.95;
        latch.step(&p);
        for _ in 0..REFRACTORY {
            p[2] = 0.05;
            latch.step(&p);
        }
        p[2] = 0.95;
        assert_eq!(latch.step(&p), vec![2]);
    }

    /// Another class is not held back by this one's refractory.
    #[test]
    fn the_refractory_is_per_class() {
        let mut latch = Latch::new(0.8);
        let mut p = [0.0f32; 12];
        p[0] = 0.95;
        latch.step(&p);
        p[0] = 0.05;
        p[5] = 0.95;
        assert_eq!(latch.step(&p), vec![5]);
    }

    fn hops(r: &mut Refires, level: f32, n: usize) {
        for _ in 0..n {
            r.hear(&vec![level; HOP]);
        }
    }

    /// The reported fault: a class fires again while its note dies away, the
    /// energy falling. That is not a strike.
    #[test]
    fn a_refire_on_a_dying_note_is_not_a_strike() {
        let mut r = Refires::default();
        hops(&mut r, 0.02, 12);
        assert_eq!(r.confirm(vec![5]), vec![5], "the first strike was held");
        let mut level = 0.02f32;
        let mut reported = Vec::new();
        for n in 0..60 {
            level *= 0.98;
            r.hear(&vec![level; HOP]);
            let candidates = if n == 40 { vec![5] } else { Vec::new() };
            reported.extend(r.confirm(candidates));
        }
        assert!(reported.is_empty(), "a dying note was reported struck: {reported:?}");
    }

    /// And a string struck again soon after is one, once the attack is in.
    #[test]
    fn a_refire_with_an_attack_is_a_strike() {
        let mut r = Refires::default();
        hops(&mut r, 0.02, 12);
        r.confirm(vec![5]);
        hops(&mut r, 0.01, 40);
        r.confirm(Vec::new());
        // The attack, and the model's report a hop into it.
        r.hear(&vec![0.05; HOP]);
        let first = r.confirm(vec![5]);
        assert!(first.is_empty(), "decided before looking ahead");
        let mut later = Vec::new();
        for _ in 0..LOOK_AHEAD {
            r.hear(&vec![0.05; HOP]);
            later.extend(r.confirm(Vec::new()));
        }
        assert_eq!(later, vec![5], "a re-strike was refused");
    }

    /// Long after its last strike, a class is not second-guessed at all.
    #[test]
    fn a_strike_long_after_the_last_passes_at_once() {
        let mut r = Refires::default();
        hops(&mut r, 0.02, 12);
        r.confirm(vec![5]);
        for _ in 0..RECHECK {
            r.hear(&vec![0.01; HOP]);
            r.confirm(Vec::new());
        }
        assert_eq!(r.confirm(vec![5]), vec![5]);
    }

    /// A quiet player is brought up to the level the model knows, slowly; the
    /// room does not move the estimate.
    #[test]
    fn a_quiet_player_is_brought_up_to_the_reference_level() {
        let mut lev = Leveller::default();
        // A constant is its own RMS: the user's playing level, 11 dB under
        // the reference.
        let quiet = vec![0.0087f32; HOP];
        let mut gain = 1.0;
        for _ in 0..2000 {
            gain = lev.gain_for(&quiet);
        }
        let reached = 0.0087 * gain;
        assert!((reached - REFERENCE_RMS).abs() / REFERENCE_RMS < 0.1, "reached {reached}");
        let room = vec![0.0001f32; HOP];
        let before = lev.gain_for(&room);
        for _ in 0..500 {
            lev.gain_for(&room);
        }
        assert!((lev.gain_for(&room) - before).abs() < 1e-3, "the room moved the level");
    }

    /// The gain is bounded both ways.
    #[test]
    fn the_correction_is_bounded() {
        let mut lev = Leveller::default();
        let faint = vec![0.0025f32; HOP];
        for _ in 0..3000 {
            lev.gain_for(&faint);
        }
        assert!(lev.gain_for(&faint) <= GAIN_RANGE.1 + 1e-6);
        let mut lev = Leveller::default();
        let loud = vec![0.9f32; HOP];
        for _ in 0..3000 {
            lev.gain_for(&loud);
        }
        assert!(lev.gain_for(&loud) >= GAIN_RANGE.0 - 1e-6);
    }

    /// A stored binary16 value, exactly.
    fn binary16(bits: u16) -> f32 {
        let fraction = (bits & 0x3ff) as f32;
        let value = match (bits >> 10) & 0x1f {
            0 => fraction * 2f32.powi(-24),
            exponent => (1.0 + fraction / 1024.0) * 2f32.powi(exponent as i32 - 15),
        };
        if bits & 0x8000 != 0 { -value } else { value }
    }

    /// The trainer's frames for the fixture audio (dist/test_strike_trainer.py
    /// holds the trainer to the same files): the same numbers, not close ones.
    #[test]
    fn features_are_the_trainers_to_the_last_bit() -> Result<()> {
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/dist/fixtures/");
        let audio: Vec<f32> = std::fs::read(format!("{fixtures}short_features.s16"))?
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();
        let expected: Vec<f32> = std::fs::read(format!("{fixtures}short_features.f16"))?
            .chunks_exact(2)
            .map(|b| binary16(u16::from_le_bytes([b[0], b[1]])))
            .collect();
        let mut features = ShortFeatures::default();
        let actual: Vec<f32> = audio.chunks_exact(HOP).flat_map(|hop| features.push(hop)).collect();
        assert_eq!(actual.len(), expected.len());
        let differing = actual.iter().zip(&expected).filter(|(a, e)| a != e).count();
        assert_eq!(differing, 0, "{differing} of {} values differ", expected.len());
        Ok(())
    }
}

#[cfg(test)]
mod diagnostics {
    use super::*;

    fn mono_of(path: &str, channel: usize) -> Result<(Vec<f32>, u32)> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let raw: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, _) => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
            (_, 16) => reader.samples::<i16>().map(|s| s.unwrap_or(0) as f32 / 32768.0).collect(),
            (_, bits) => {
                let full = (1i32 << (bits - 1)) as f32;
                reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / full).collect()
            }
        };
        let ch = channel.min(spec.channels as usize - 1);
        Ok((raw.chunks(spec.channels as usize).map(|f| f[ch]).collect(), spec.sample_rate))
    }

    /// The app's features for a recording, the first frames of them, so the
    /// trainer's `onset_features` can be held to them exactly.
    ///
    /// SOLITITO_FEATURES_WAV=file.wav SOLITITO_FEATURES_OUTPUT=out.json [SOLITITO_FEATURES_FRAMES=n]
    #[test]
    #[ignore = "diagnostic: requires SOLITITO_FEATURES_WAV"]
    fn export_features() -> Result<()> {
        let (mono, rate) = mono_of(&std::env::var("SOLITITO_FEATURES_WAV")?, 0)?;
        let signal = resample_like_trainer(&mono, rate);
        let frames: usize = std::env::var("SOLITITO_FEATURES_FRAMES")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(400);
        let mut features = ShortFeatures::default();
        let rows: Vec<Vec<f32>> = signal
            .chunks_exact(HOP)
            .take(frames)
            .map(|hop| features.push(hop).to_vec())
            .collect();
        std::fs::write(std::env::var("SOLITITO_FEATURES_OUTPUT")?, serde_json::to_vec(&rows)?)?;
        Ok(())
    }

    /// Every hop's twelve probabilities and every reported strike, for scoring
    /// against labelled notes.
    ///
    /// SOLITITO_STRIKE_WAV=file.wav SOLITITO_STRIKE_OUTPUT=out.json
    #[test]
    #[ignore = "diagnostic: requires SOLITITO_STRIKE_WAV"]
    fn export_strikes() -> Result<()> {
        let path = std::env::var("SOLITITO_STRIKE_WAV")?;
        let channel = std::env::var("SOLITITO_STRIKE_CHANNEL")
            .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).saturating_sub(1);
        let (mono, rate) = mono_of(&path, channel)?;
        let gain: f32 = std::env::var("SOLITITO_STRIKE_GAIN")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let signal: Vec<f32> = resample_like_trainer(&mono, rate).iter().map(|v| v * gain).collect();
        let model = std::env::var("SOLITITO_STRIKE_MODEL").unwrap_or_else(|_| MODEL.to_string());
        let mut strikes = Strikes::load(&model)?;
        if let Some(t) = std::env::var("SOLITITO_STRIKE_THRESHOLD").ok().and_then(|v| v.parse().ok()) {
            strikes.latch.threshold = t;
        }
        strikes.levelled = std::env::var("SOLITITO_STRIKE_LEVEL").map_or(true, |v| v != "0");
        if let Some(v) = std::env::var("SOLITITO_REFIRE_RISE").ok().and_then(|v| v.parse().ok()) {
            strikes.refires.rise = v;
        }
        if let Some(r) = std::env::var("SOLITITO_STRIKE_REFRACTORY").ok().and_then(|v| v.parse().ok()) {
            strikes.latch.refractory = r;
        }
        let mut events = Vec::new();
        let mut probs = Vec::new();
        for (n, hop) in signal.chunks_exact(HOP).enumerate() {
            let (p, struck) = strikes.push(hop)?;
            // The hop ends at (n + 1) * HOP: that is when this answer exists.
            let t = ((n + 1) * HOP) as f64 / SR as f64;
            for pc in struck {
                events.push(serde_json::json!({"t": t, "pc": pc, "p": p[pc]}));
            }
            probs.push(p.to_vec());
        }
        std::fs::write(
            std::env::var("SOLITITO_STRIKE_OUTPUT")?,
            serde_json::to_vec(&serde_json::json!({
                "threshold": strikes.latch.threshold, "events": events, "probs": probs,
            }))?,
        )?;
        Ok(())
    }
}
