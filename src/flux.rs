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
//! Nothing consults this yet. It answers WHEN and says nothing about WHICH
//! note, and on its own that cannot credit anything.
use std::collections::VecDeque;

/// Frames of flux the adaptive threshold is read from - one second at the
/// 16 ms hop. Long enough to describe the material being played, short enough
/// to follow a change of dynamics.
const HISTORY: usize = 62;

/// How far above the recent median, in median absolute deviations.
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
pub(crate) const DEVIATIONS: f32 = 15.0;

/// The least gap between two reports, 48 ms. A floor only - what actually
/// separates two plucks is the latch below.
///
/// A fixed refractory was tried for both jobs and could do neither. Long enough
/// to cover one attack it silenced the next real one, costing a third of the
/// notes on AtoA; short enough to pass them it let a single pluck report twice.
/// A string hit once keeps rising for about seven frames, so no fixed number
/// separates "still the same attack" from "the next one".
const REFRACTORY: usize = 3;

/// A class is ready to report again once the flux falls to this fraction of the
/// value that last fired. The same shape of latch the model's own events use.
///
/// This is what tells one attack from two: while a string is still being
/// excited the flux stays up, and between plucks it drops to the floor. The
/// fault it fixes was measured on the user's second recording - one pluck at
/// frame 2398 reported twice, 96 ms apart, and the first report named a note
/// nobody played because it landed on the transient.
const REARM: f32 = 0.3;

/// Frames between the attack and the question "which note was that": 96 ms.
///
/// The attack frame is the worst possible moment to ask. A string being struck
/// is a broadband click before it is a pitch, and on AtoA the answer changes
/// between the attack and 96 ms later for one attack in five. Waiting takes the
/// naming from 63 right / 33 wrong to 70 / 26, while the notes found stay at 50
/// of 51. Beyond this the curve flattens and only the delay grows.
///
/// The event still carries the time of the ATTACK, not of the naming: this
/// changes what is reported, never when it happened.
const SETTLE: usize = 6;

/// Below this the frame is silence, whatever the recent history says: without
/// an absolute floor the adaptive threshold would chase the noise down and fire
/// on it between phrases.
const FLOOR: f32 = 1e-4;

pub(crate) struct Flux {
    previous: Vec<f32>,
    history: VecDeque<f32>,
    since: usize,
    deviations: f32,
    /// Ready to report. Cleared on an attack, set again once the flux falls.
    armed: bool,
    /// The flux that fired, so re-arming is measured against it.
    peak: f32,
    /// Frames since an attack that has not been named yet, if any.
    waiting: Option<usize>,
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
            waiting: None,
        }
    }
}

impl Flux {
    /// One hop of LINEAR magnitudes. Returns the flux, and whether this frame
    /// is the one to NAME an attack on - which is `SETTLE` frames after the
    /// attack itself, not the attack frame.
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
            && self.since >= REFRACTORY
            && flux > FLOOR
            && self.threshold().is_some_and(|bar| flux >= bar);
        if attack {
            self.armed = false;
            self.peak = flux;
            self.since = 0;
            // A second attack while one is still settling replaces it: the
            // newer sound is the one about to be named, and reporting the older
            // one late would name this spectrum with that one's timing.
            self.waiting = Some(0);
        } else {
            self.since = self.since.saturating_add(1);
        }

        let name_now = match self.waiting {
            Some(age) if age >= SETTLE => {
                self.waiting = None;
                true
            }
            Some(age) => {
                self.waiting = Some(age + 1);
                false
            }
            None => false,
        };

        if self.history.len() == HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(flux);
        (flux, name_now)
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

/// Bins of the 2048-point spectrum, and the hertz each one stands for.
const BINS: usize = 513;
const HZ_PER_BIN: f32 = 16_000.0 / 2048.0;

/// The range a guitar plays in, E2 to E6.
const LOW_MIDI: i32 = 40;
const HIGH_MIDI: i32 = 88;

/// Harmonics summed, and how much each is worth. Later ones say less about
/// which note it is and are the ones a neighbouring string is likeliest to
/// share - the same weighting `audio::mono_pitch` uses on the CQT.
const HARMONICS: usize = 6;

/// How close a candidate below the winner has to come before it is preferred.
/// Any partial can win a harmonic sum outright, because an impostor sitting on
/// one collects the note's upper harmonics as its own.
///
/// Looser than a plain "is it better": the candidate being defended is the
/// quieter one by construction - that is why it lost - so it only has to be
/// comparable. The same 0.8 `audio::mono_pitch` settled on.
const SUBHARMONIC: f32 = 0.8;

/// Where the note might really be, if the winner is one of its partials:
/// harmonics two to six stand 12, 19, 24, 28 and 31 semitones above the
/// fundamental, and a fifth above shares the note's third and sixth.
///
/// Lowest candidate first, and the first one that stands up takes it. Each is
/// weighed against the ORIGINAL winner: tried in sequence against a winner that
/// keeps moving, the drops chain, and 12 then 19 then 24 lands fifty-five
/// semitones below where it started.
const IMPOSTORS: [usize; 6] = [31, 28, 24, 19, 12, 7];

/// Which pitch class the spectrum is built on, by a weighted harmonic sum.
///
/// Asked only at an instant something was struck, which is why it may answer
/// unconditionally. `audio::mono_pitch` carries a score gate because it is
/// asked on every frame and most frames are not a note; measured on the user's
/// recording, that gate leaves it silent at half of the attacks. Here the flux
/// detector has already said a string was hit.
///
/// Measured against the 51 reviewed labels of AtoA, naming the note at the
/// frame flux reports: 50 of 51, where Rise at its own threshold names 48.
pub(crate) fn harmonic_pitch(magnitudes: &[f32]) -> Option<usize> {
    if magnitudes.len() < BINS {
        return None;
    }
    let at = |hz: f32| -> f32 {
        let k = hz / HZ_PER_BIN;
        let low = k as usize;
        if low + 1 >= BINS {
            return 0.0;
        }
        let frac = k - low as f32;
        magnitudes[low] * (1.0 - frac) + magnitudes[low + 1] * frac
    };
    let score = |midi: i32| -> f32 {
        let f0 = 440.0 * 2f32.powf((midi - 69) as f32 / 12.0);
        (0..HARMONICS)
            .map(|h| at(f0 * (h + 1) as f32) / (1.0 + h as f32 * 0.35))
            .sum()
    };
    let scores: Vec<f32> = (LOW_MIDI..=HIGH_MIDI).map(score).collect();
    let winner = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?
        .0;
    if scores[winner] <= 0.0 {
        return None;
    }
    let best = IMPOSTORS
        .iter()
        .filter_map(|&drop| winner.checked_sub(drop))
        .find(|&lower| scores[lower] >= scores[winner] * SUBHARMONIC)
        .unwrap_or(winner);
    Some((LOW_MIDI as usize + best) % 12)
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
        flux.push(&[loud, 0.02, 0.0, 0.0]);
        let named = (0..SETTLE + 2)
            .filter(|_| flux.push(&[loud, 0.02, 0.0, 0.0]).1)
            .count();
        assert_eq!(named, 1, "a quiet string beside a loud one went unheard");
    }

    /// The reported fault, as a rule: a string hit once goes on getting louder
    /// for several frames, and that is still one attack. The old fixed
    /// refractory let it report twice 96 ms apart, and the first report landed
    /// on the transient and named a note nobody played.
    #[test]
    fn one_pluck_that_keeps_rising_is_named_once() {
        let mut flux = Flux::default();
        for _ in 0..HISTORY + 10 {
            flux.push(&[0.001, 0.001, 0.001, 0.001]);
        }
        let mut named = 0;
        // The measured shape: a jump, then seven frames still climbing.
        let mut level = 0.02f32;
        named += flux.push(&[level, level, level, level]).1 as usize;
        for _ in 0..7 {
            level *= 1.06;
            named += flux.push(&[level, level, level, level]).1 as usize;
        }
        // Then the decay, which adds nothing positive to sum.
        for _ in 0..40 {
            level *= 0.95;
            named += flux.push(&[level, level, level, level]).1 as usize;
        }
        assert_eq!(named, 1, "one pluck was named {named} times");
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

    /// A harmonic series names its own fundamental, not one of its partials.
    #[test]
    fn a_harmonic_series_names_its_fundamental() {
        for midi in [40i32, 45, 52, 57, 64, 69, 76] {
            let f0 = 440.0 * 2f32.powf((midi - 69) as f32 / 12.0);
            let mut spectrum = vec![0.0f32; BINS];
            for h in 0..HARMONICS {
                let k = (f0 * (h + 1) as f32 / HZ_PER_BIN) as usize;
                if k < BINS {
                    spectrum[k] = 1.0 / (h + 1) as f32;
                }
            }
            assert_eq!(
                harmonic_pitch(&spectrum),
                Some((midi % 12) as usize),
                "midi {midi} was misnamed"
            );
        }
    }

    /// The reported fault: a pluck's FIFTH harmonic stands 28 semitones above
    /// it, which is a major third away by pitch class, so a strong one names
    /// the third of the note actually played. Twenty-eight was missing from the
    /// offsets, and the ones that were there chained instead of being weighed
    /// against the winner.
    #[test]
    fn a_fifth_harmonic_does_not_name_the_third() {
        for midi in [40i32, 45, 50, 55, 59, 64] {
            let f0 = 440.0 * 2f32.powf((midi - 69) as f32 / 12.0);
            let mut spectrum = vec![0.0f32; BINS];
            // The fifth harmonic louder than the fundamental, as a bright
            // bridge pickup gives.
            for (h, level) in [(0usize, 0.35f32), (1, 0.5), (2, 0.6), (3, 0.7), (4, 1.0), (5, 0.5)] {
                let k = (f0 * (h + 1) as f32 / HZ_PER_BIN) as usize;
                if k < BINS {
                    spectrum[k] = level;
                }
            }
            let named = harmonic_pitch(&spectrum);
            assert_ne!(
                named,
                Some(((midi + 4) % 12) as usize),
                "midi {midi} was named as its own major third"
            );
            assert_eq!(named, Some((midi % 12) as usize), "midi {midi} was misnamed");
        }
    }

    /// The drops must be weighed against the winner, not against each other.
    #[test]
    fn the_subharmonic_guard_does_not_chain() {
        // One clean note: nothing below it should be preferred at all.
        let midi = 64i32;
        let f0 = 440.0 * 2f32.powf((midi - 69) as f32 / 12.0);
        let mut spectrum = vec![0.0f32; BINS];
        for h in 0..HARMONICS {
            let k = (f0 * (h + 1) as f32 / HZ_PER_BIN) as usize;
            if k < BINS {
                spectrum[k] = 1.0 / (h + 1) as f32;
            }
        }
        assert_eq!(harmonic_pitch(&spectrum), Some((midi % 12) as usize));
    }

    /// The trap the subharmonic guard is for: a fifth above shares the note's
    /// third and sixth harmonics, so it can win the sum outright.
    #[test]
    fn a_partial_does_not_win_over_the_note_it_belongs_to() {
        let midi = 52i32; // E3
        let f0 = 440.0 * 2f32.powf((midi - 69) as f32 / 12.0);
        let mut spectrum = vec![0.0f32; BINS];
        // A real pluck's upper partials are often louder than its fundamental.
        for (h, level) in [(0usize, 0.3f32), (1, 1.0), (2, 0.9), (3, 0.6), (4, 0.4)] {
            let k = (f0 * (h + 1) as f32 / HZ_PER_BIN) as usize;
            if k < BINS {
                spectrum[k] = level;
            }
        }
        assert_eq!(harmonic_pitch(&spectrum), Some((midi % 12) as usize));
    }

    #[test]
    fn an_empty_spectrum_names_nothing() {
        assert_eq!(harmonic_pitch(&vec![0.0; BINS]), None);
        assert_eq!(harmonic_pitch(&[1.0, 2.0]), None);
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
