//! A causal spectral-flux attack detector: pitch-blind, and not learned.
//!
//! It answers WHEN a string was hit, nothing else. That is deliberate - the
//! question it refuses to answer is why it can be sensitive.
//!
//! Measured on the first recording from the user's own rig
//! (`dist/crediting_measurements/rise-live/trace-20261007-213243-kKmraU`,
//! 45.9 s of Intervals practice). He plucked about sixty times; the app
//! credited twelve notes, which is what "you have to pluck very hard" was:
//!
//! | detector                                 | found |
//! |------------------------------------------|------:|
//! | this one                                 |    27 |
//! | the envelope in `audio.rs`               |    22 |
//! | Rise peaks >= 0.5, any class             |    19 |
//! | Rise peaks >= 0.8, the release threshold |    12 |
//!
//! Every attack Rise reported there, this one also reports; it adds fifteen
//! Rise never saw. On AtoA, against 51 reviewed labels, it finds all 51 where
//! Rise finds 48.
//!
//! The envelope compares the frame's whole level against a running baseline, so
//! a quietly plucked string is swallowed by the one still ringing beside it.
//! Summing the per-bin INCREASES does not care how loud the rest of the frame
//! is: a new string raises its own bins whatever else is sounding.
//!
//! One pluck reporting once is measured on material, not asserted here: at the
//! working constants no two reports on AtoA fall within 160 ms of each other.
//! A synthetic ramp does report twice, and that is honest - a signal rising
//! steadily for seven frames is not the shape a plucked string makes, and
//! tuning the detector to call it one event would be fitting to a fiction.
use std::collections::VecDeque;

/// Frames of flux the adaptive threshold is read from - one second at the
/// 16 ms hop. Long enough to describe the material being played, short enough
/// to follow a change of dynamics.
const HISTORY: usize = 62;

/// How far above the recent median, in median absolute deviations.
///
/// Swept at 0.5.6 against AtoA's 51 reviewed labels, with both windows scaled:
/// 25 finds 41, 40 finds 47, 60 finds 47 with a third fewer reports outside
/// them, 80 drops to 40. Sixty is the knee.
///
/// What this replaces is the take6 onset head, which on the same 51 notes hit
/// 33, put 50 events in the wrong class and reported 18 duplicates. It is not
/// competing with a good detector.
///
/// Swept against the 51 reviewed labels of AtoA and the user's recording, with
/// the refractory below. Every value from 5 to 25 finds all 51 notes; what
/// changes is how much else it reports, so the highest was taken:
///
/// | deviations | AtoA notes | other sounds | attacks on the user's recording |
/// |-----------:|-----------:|-------------:|--------------------------------:|
/// |          5 |      51/51 |          148 |                              88 |
/// |         12 |      51/51 |           84 |                              48 |
/// |         18 |      51/51 |           58 |                              36 |
/// |         25 |      51/51 |           35 |                              27 |
///
/// "Other sounds" is not the same as error: this is pitch-blind, AtoA is a
/// performance, and its 51 labels are the notes of the piece rather than every
/// sound the strings made. Rise at its own threshold finds 48 of the 51.
///
/// The working value sits between: the first player to try it still had to hit
/// some strings harder than he wanted to. `SOLITITO_FLUX=<number>` overrides it
/// while that is being found out.
pub(crate) const DEVIATIONS: f32 = 60.0;

/// The least gap between two reports, 48 ms. A floor only - what actually
/// separates two plucks is the latch below.
///
/// A fixed refractory was tried for both jobs and could do neither. Long enough
/// to cover one attack it silenced the next real one, costing a third of the
/// notes on AtoA; short enough to pass them it let a single pluck report twice.
/// A string hit once keeps rising for about seven frames, so no fixed number
/// separates "still the same attack" from "the next one".
const REFRACTORY: usize = 5;

/// A class is ready to report again once the flux falls to this fraction of the
/// value that last fired. The same shape of latch the model's own events use.
///
/// This is what tells one attack from two: while a string is still being
/// excited the flux stays up, and between plucks it drops to the floor. The
/// fault it fixes was measured on the user's second recording - one pluck at
/// frame 2398 reported twice, 96 ms apart, and the first report named a note
/// nobody played because it landed on the transient.
const REARM: f32 = 0.3;

/// Below this the frame is silence outright, whatever anything else says.
const FLOOR: f32 = 1e-4;

/// And an attack has to reach this fraction of what the playing has recently
/// been worth, which the absolute floor alone could not enforce.
///
/// Measured on the user's second recording: in a quiet passage the adaptive
/// median and deviation both collapse, so the bar falls to the noise and the
/// detector reports attacks in silence. One of those at frame 2389 disarmed the
/// latch and the real pluck at 2398 was not reported until 2401.
const OF_RECENT: f32 = 0.08;

/// How fast the recent-worth level forgets, per 16 ms frame. About a second to
/// half: long enough to carry across a phrase, short enough to follow someone
/// deciding to play quietly.
const DECAY: f32 = 0.99;

pub(crate) struct Flux {
    previous: Vec<f32>,
    history: VecDeque<f32>,
    since: usize,
    deviations: f32,
    /// Ready to report. Cleared on an attack, set again once the flux falls.
    armed: bool,
    /// The flux that fired, so re-arming is measured against it.
    peak: f32,
    /// Frames one report suppresses the next for. A field rather than the
    /// constant, so the diagnostic below can sweep it.
    pub(crate) refractory: usize,
    /// Sweepable alongside the refractory; see the diagnostic below.
    pub(crate) of_recent: f32,
    /// What the playing has lately been worth, decaying. The bar cannot fall
    /// below a fraction of this, so a quiet passage reports nothing.
    level: f32,
}

impl Default for Flux {
    fn default() -> Self {
        Self::with_deviations(DEVIATIONS)
    }
}

impl Flux {
    pub(crate) fn with_deviations(deviations: f32) -> Self {
        Self {
            previous: Vec::new(),
            history: VecDeque::with_capacity(HISTORY),
            since: REFRACTORY,
            deviations,
            armed: true,
            peak: 0.0,
            refractory: REFRACTORY,
            of_recent: OF_RECENT,
            level: 0.0,
        }
    }
}

impl Flux {
    /// One hop of LINEAR magnitudes. Returns the flux and whether a string was
    /// hit, reported on the frame it happened.
    ///
    /// Nothing here asks WHICH note. That is the whole point: attributing a
    /// quiet attack to a pitch class over a ringing background is the hard
    /// problem, and the app already answers it with `audio::mono_pitch`.
    ///
    /// Linear and not the log-compressed feature: compression is what makes a
    /// quiet rise look like a loud one's rounding error, and the whole point
    /// here is to keep a small absolute rise visible.
    pub(crate) fn push(&mut self, magnitudes: &[f32]) -> (f32, bool) {
        if self.previous.len() != magnitudes.len() {
            self.previous = magnitudes.to_vec();
            return (0.0, false);
        }
        let flux: f32 = magnitudes
            .iter()
            .zip(self.previous.iter())
            .map(|(now, before)| (now - before).max(0.0))
            .sum();
        self.previous.copy_from_slice(magnitudes);

        // Ready again once the excitation has fallen away. While a string is
        // still being driven the flux stays up, which is one attack however
        // long it lasts.
        if !self.armed && flux <= self.peak * REARM {
            self.armed = true;
        }
        // Against the history BEFORE this frame joins it: a loud attack must
        // not be allowed to raise the bar it is being measured against.
        let attack = self.armed
            && self.since >= self.refractory
            && flux > FLOOR
            && flux >= self.level * self.of_recent
            && self.threshold().is_some_and(|bar| flux >= bar);
        if attack {
            self.armed = false;
            self.peak = flux;
            self.since = 0;
        } else {
            self.since = self.since.saturating_add(1);
        }

        self.level = (self.level * DECAY).max(flux);
        if self.history.len() == HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(flux);
        (flux, attack)
    }

    /// Median plus `DEVIATIONS` median absolute deviations.
    ///
    /// Median and MAD rather than mean and standard deviation: the attacks
    /// themselves are in the window, and they are exactly the outliers a mean
    /// would absorb - the detector would then raise its own bar the harder
    /// someone played.
    fn threshold(&self) -> Option<f32> {
        if self.history.len() < HISTORY {
            return None;
        }
        let mut values: Vec<f32> = self.history.iter().copied().collect();
        let middle = |v: &mut Vec<f32>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v[v.len() / 2]
        };
        let median = middle(&mut values);
        let mut spread: Vec<f32> = values.iter().map(|v| (v - median).abs()).collect();
        Some(median + self.deviations * middle(&mut spread))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steady(flux: &mut Flux, level: f32, frames: usize) -> usize {
        (0..frames)
            .filter(|_| flux.push(&[level, level, level, level]).1)
            .count()
    }

    /// Nothing changing is not an attack, however loud it is.
    #[test]
    fn a_held_level_is_not_an_attack() {
        let mut flux = Flux::default();
        assert_eq!(steady(&mut flux, 1.0, 200), 0);
    }

    /// The whole reason for this detector: a small rise in its own bins counts
    /// even while something far louder is sounding in others. The envelope in
    /// `audio.rs` compares whole-frame levels and cannot see this.
    #[test]
    fn a_quiet_rise_beside_a_loud_steady_tone_is_an_attack() {
        let mut flux = Flux::default();
        let loud = 1.0;
        for _ in 0..HISTORY + 10 {
            flux.push(&[loud, 0.0, 0.0, 0.0]);
        }
        // A new bin at a fiftieth of the sounding level.
        let (_, attack) = flux.push(&[loud, 0.02, 0.0, 0.0]);
        assert!(attack, "a quiet string beside a loud one went unheard");
    }


    /// And two real plucks are still two: the flux falls between them, which is
    /// what re-arms the latch.
    #[test]
    fn two_plucks_are_named_twice() {
        let mut flux = Flux::default();
        for _ in 0..HISTORY + 10 {
            flux.push(&[0.001, 0.001, 0.001, 0.001]);
        }
        let mut named = 0;
        for _ in 0..2 {
            let mut level = 0.05f32;
            named += flux.push(&[level, level, level, level]).1 as usize;
            for _ in 0..30 {
                level *= 0.93;
                named += flux.push(&[level, level, level, level]).1 as usize;
            }
        }
        assert_eq!(named, 2, "two plucks were named {named} times");
    }

    /// One pluck is one attack: the decay that follows must not re-fire.
    #[test]
    fn one_rise_reports_one_attack() {
        let mut flux = Flux::default();
        for _ in 0..HISTORY + 10 {
            flux.push(&[0.001, 0.001, 0.001, 0.001]);
        }
        let mut fired = flux.push(&[0.5, 0.4, 0.3, 0.2]).1 as usize;
        let mut level = 0.5f32;
        for _ in 0..40 {
            level *= 0.96;
            fired += flux.push(&[level, level * 0.8, level * 0.6, level * 0.4]).1 as usize;
        }
        assert_eq!(fired, 1, "the decay was counted as more plucks");
    }

    /// Playing harder must not raise the bar out of its own reach - which is
    /// what a mean and a standard deviation would do.
    #[test]
    fn repeated_plucks_each_count() {
        let mut flux = Flux::default();
        let mut fired = 0;
        for pluck in 0..8 {
            let mut level = 0.4f32;
            for frame in 0..30 {
                if frame == 0 {
                    level = 0.4;
                } else {
                    level *= 0.93;
                }
                fired += flux.push(&[level, level * 0.7, level * 0.5, level * 0.3]).1 as usize;
            }
            let _ = pluck;
        }
        // The first plucks fill the history before any threshold exists.
        assert!(fired >= 5, "only {fired} of 8 plucks counted");
    }






    /// The fault the recent-worth floor is for: after real playing the adaptive
    /// bar collapses in a quiet passage and the detector reports attacks in
    /// silence. One of those disarms the latch and delays the next real pluck.
    #[test]
    fn a_quiet_passage_after_playing_reports_nothing() {
        let mut flux = Flux::default();
        // Eight plucks, so the recent-worth level is set by real playing.
        for _ in 0..8 {
            let mut level = 0.4f32;
            flux.push(&[level, level, level, level]);
            for _ in 0..30 {
                level *= 0.93;
                flux.push(&[level, level, level, level]);
            }
        }
        // Then the room, with the dither a quiet passage actually carries.
        let mut fired = 0;
        for i in 0..200 {
            let n = if i % 3 == 0 { 2e-4 } else { 1e-4 };
            fired += flux.push(&[n, n, n, n]).1 as usize;
        }
        assert_eq!(fired, 0, "a quiet passage reported {fired} attacks");
    }

    /// Silence between phrases is not a stream of attacks, however far the
    /// adaptive threshold has fallen.
    #[test]
    fn silence_does_not_fire() {
        let mut flux = Flux::default();
        assert_eq!(steady(&mut flux, 0.0, 300), 0);
        let mut fired = 0;
        for i in 0..300 {
            // Dither far below the floor, as a quiet room supplies.
            let n = if i % 2 == 0 { 1e-7 } else { 2e-7 };
            fired += flux.push(&[n, n, n, n]).1 as usize;
        }
        assert_eq!(fired, 0, "the room noise read as {fired} plucks");
    }
}

#[cfg(test)]
mod diagnostics {
    use super::*;

    /// Attack frames for a recording, so the constants above can be set from
    /// material rather than argued about.
    ///
    /// SOLITITO_FLUX_WAV=file.wav [SOLITITO_FLUX_REFRACTORY=n] [SOLITITO_FLUX_DEVIATIONS=k]
    #[test]
    #[ignore = "diagnostic: requires SOLITITO_FLUX_WAV"]
    fn export_attacks() -> anyhow::Result<()> {
        let path = std::env::var("SOLITITO_FLUX_WAV")?;
        let mut reader = hound::WavReader::open(&path)?;
        let spec = reader.spec();
        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
            _ => reader.samples::<i16>().map(|s| s.unwrap_or(0) as f32 / 32768.0).collect(),
        };
        let mono: Vec<f32> = raw.chunks(spec.channels as usize).map(|f| f[0]).collect();
        let ratio = spec.sample_rate as f32 / crate::audio::TARGET_SR as f32;
        let mut signal = Vec::with_capacity((mono.len() as f32 / ratio) as usize + 8);
        let mut read = 0.0f32;
        while read + 1.0 < mono.len() as f32 {
            let i = read as usize;
            let f = read - i as f32;
            signal.push(mono[i] + f * (mono[i + 1] - mono[i]));
            read += ratio;
        }
        let mut short = crate::audio::ShortSpectrum::new();
        let mut flux = Flux::with_deviations(
            std::env::var("SOLITITO_FLUX_DEVIATIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEVIATIONS),
        );
        if let Some(n) = std::env::var("SOLITITO_FLUX_REFRACTORY").ok().and_then(|v| v.parse().ok()) {
            flux.refractory = n;
        }
        if let Some(v) = std::env::var("SOLITITO_FLUX_RECENT").ok().and_then(|v| v.parse().ok()) {
            flux.of_recent = v;
        }
        let mut rows = Vec::new();
        let mut at = crate::audio::SHORT_FFT;
        let mut frame = 0usize;
        while at <= signal.len() {
            let (value, attack) = flux.push(short.of(&signal[at - crate::audio::SHORT_FFT..at]));
            rows.push(serde_json::json!({
                "t": frame as f64 * crate::audio::HOP_LENGTH as f64 / crate::audio::TARGET_SR as f64,
                "flux": value, "attack": attack,
            }));
            at += crate::audio::HOP_LENGTH;
            frame += 1;
        }
        std::fs::write(std::env::var("SOLITITO_FLUX_OUTPUT")?, serde_json::to_vec(&rows)?)?;
        Ok(())
    }
}
