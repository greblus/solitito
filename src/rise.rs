//! Which notes ROSE just now: the short spectrum against itself a moment
//! before.
//!
//! What is only ringing has not risen, whatever its level, and a string struck
//! beside it has - so the rise of the spectrum holds the note just struck
//! without the ones before it, which no reading of the spectrum itself does.
//! It is read as notes the way `voices` reads a spectrum: the strongest is
//! taken, its partials are struck out, and what is left is asked again.
//!
//! Measurements: `docs/judge-measurements.md#the-rise-of-the-short-spectrum`.

use std::collections::VecDeque;

use crate::audio::{SHORT_FFT, SHORT_FFTS, TARGET_SR};

/// Hops back the spectrum is compared with: the 2048 samples it is taken
/// from, so the two share no audio.
const REFERENCE: usize = 8;
/// The rise, summed, as a share of the spectrum now: below it nothing rose.
const FLOOR: f32 = 0.1;
/// A note is named from this share of the strongest note's rise.
const NEAR: f32 = 0.25;
/// The rise at a note's own fundamental, as a share of its mean partial's:
/// below it the note is somebody else's harmonics.
const OWN: f32 = 1.0;
/// Partials summed for a note.
const PARTIALS: usize = 5;
/// Notes looked for: E2 to E6, as MIDI numbers.
const NOTES: std::ops::RangeInclusive<usize> = 40..=88;
/// More than four notes risen at once is not a thing one hand strikes.
const VOICES: usize = 4;
/// Bins either side of a partial that are its own: the main lobe of the
/// spectrum's window.
const LOBE: usize = 2;

const BIN_HZ: f32 = TARGET_SR as f32 / SHORT_FFT as f32;

/// The rise of the short spectrum, hop by hop.
#[derive(Default)]
pub struct Rise {
    /// The last `REFERENCE` spectra of `SHORT_FFT` samples, oldest first.
    history: VecDeque<Vec<f32>>,
}

impl Rise {
    /// One hop: `spectrum` is `audio::ShortSpectrum`'s answer. Returns the
    /// classes that rose, strongest first, each with its rise - none until
    /// there is a spectrum `REFERENCE` hops old to compare with.
    pub fn push(&mut self, spectrum: &[f32]) -> Vec<(usize, f32)> {
        // The longer of the two spectra: the shorter cannot tell a semitone
        // from the next on the low strings.
        let now = spectrum[SHORT_FFTS[0] / 4 + 1..].to_vec();
        let risen = match self.history.front().filter(|_| self.history.len() == REFERENCE) {
            Some(before) => risen_classes(&now, before),
            None => Vec::new(),
        };
        self.history.push_back(now);
        if self.history.len() > REFERENCE {
            self.history.pop_front();
        }
        risen
    }
}

fn hz_of(midi: usize) -> f32 {
    440.0 * 2f32.powf((midi as f32 - 69.0) / 12.0)
}

/// `rise` between two bins, where a partial at `hz` falls.
fn rise_at(rise: &[f32], hz: f32) -> f32 {
    let bin = hz / BIN_HZ;
    let (low, part) = (bin as usize, bin.fract());
    match (rise.get(low), rise.get(low + 1)) {
        (Some(a), Some(b)) => a * (1.0 - part) + b * part,
        _ => 0.0,
    }
}

/// The summed rise of a note's partials, where its own fundamental rose with
/// them.
fn note_rise(rise: &[f32], midi: usize) -> Option<f32> {
    let hz = hz_of(midi);
    let sum: f32 = (1..=PARTIALS).map(|h| rise_at(rise, hz * h as f32)).sum();
    (sum > 0.0 && rise_at(rise, hz) >= OWN * sum / PARTIALS as f32).then_some(sum)
}

/// Every partial of `midi` the spectrum reaches, struck out of `rise`.
fn strike_out(rise: &mut [f32], midi: usize) {
    let hz = hz_of(midi);
    let mut partial = hz;
    while partial < BIN_HZ * rise.len() as f32 {
        let centre = (partial / BIN_HZ).round() as usize;
        for bin in centre.saturating_sub(LOBE)..(centre + LOBE + 1).min(rise.len()) {
            rise[bin] = 0.0;
        }
        partial += hz;
    }
}

fn risen_classes(now: &[f32], before: &[f32]) -> Vec<(usize, f32)> {
    let mut rise: Vec<f32> = now.iter().zip(before).map(|(n, b)| (n - b).max(0.0)).collect();
    let mut found: Vec<(usize, f32)> = Vec::new();
    if rise.iter().sum::<f32>() < FLOOR * now.iter().sum::<f32>() {
        return found;
    }
    for _ in 0..VOICES {
        let strongest = NOTES
            .filter_map(|midi| note_rise(&rise, midi).map(|sum| (midi, sum)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let Some((midi, sum)) = strongest else { break };
        if found.first().is_some_and(|first| sum < NEAR * first.1) {
            break;
        }
        if !found.iter().any(|f| f.0 == midi % 12) {
            found.push((midi % 12, sum));
        }
        strike_out(&mut rise, midi);
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{ShortSpectrum, FFT_SIZE, HOP_LENGTH};

    /// A plucked string: a fundamental and weaker harmonics, from `start` on.
    fn string(hz: f32, start: usize, level: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                if i < start {
                    return 0.0;
                }
                let t = (i - start) as f32 / TARGET_SR as f32;
                let decay = (-t * 2.0).exp();
                level * decay * (1..=6).map(|k| (2.0 * std::f32::consts::PI * hz * k as f32 * t).sin() / k as f32).sum::<f32>()
            })
            .collect()
    }

    /// The classes that rose on each hop of `audio`.
    fn risen(audio: &[f32]) -> Vec<Vec<usize>> {
        let mut short = ShortSpectrum::new();
        let mut rise = Rise::default();
        let mut window = vec![0.0f32; FFT_SIZE];
        audio
            .chunks_exact(HOP_LENGTH)
            .map(|hop| {
                window.drain(..HOP_LENGTH);
                window.extend_from_slice(hop);
                rise.push(short.of(&window)).iter().map(|c| c.0).collect()
            })
            .collect()
    }

    /// A G struck a quarter as loud under a ringing C is the note that rose;
    /// the C, struck half a second before, is not named again.
    #[test]
    fn a_quiet_note_struck_under_a_ringing_one_is_the_one_that_rose() {
        let len = TARGET_SR as usize;
        let second = len * 3 / 4;
        let c = string(130.81, len / 4, 1.0, len);
        let g = string(196.0, second, 0.25, len);
        let audio: Vec<f32> = c.iter().zip(&g).map(|(a, b)| a + b).collect();
        let hops = risen(&audio);
        let after = second / HOP_LENGTH + 3;
        for (hop, classes) in hops.iter().enumerate().take(after + 4).skip(after) {
            assert_eq!(classes.first(), Some(&7), "hop {hop}: {classes:?}");
            assert!(!classes.contains(&0), "the ringing C rose on hop {hop}: {classes:?}");
        }
    }

    /// A note left to ring rises once: nothing is named a second after it.
    #[test]
    fn a_note_only_ringing_does_not_rise() {
        let start = TARGET_SR as usize / 4;
        let audio = string(220.0, start, 1.0, 2 * TARGET_SR as usize);
        let hops = risen(&audio);
        let struck = start / HOP_LENGTH;
        assert!(hops[struck + 3].contains(&9), "{:?}", hops[struck + 3]);
        let late = struck + TARGET_SR as usize / HOP_LENGTH;
        assert!(hops[late..].iter().all(Vec::is_empty), "{:?}", &hops[late..late + 5]);
    }
}
