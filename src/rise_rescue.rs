//! Optional weak-onset confirmation. Reference experiment: dist/onset_rescue.py.
//! Uses the existing causal FFT; never accepts a note from its sound alone.
use super::Spectrum;
use rustfft::num_complex::Complex;
use std::collections::VecDeque;

const BINS: usize = 513;
const NOTES: usize = 49; // MIDI 40..88, the guitar range used by the probe.
const PAST: usize = 4;

pub(super) struct Rescue {
    dictionary: Vec<Vec<f64>>,
    gram: Vec<Vec<f64>>,
    past: VecDeque<Vec<f64>>,
    pending: [Option<f32>; 12],
    pending_background: Option<Vec<f64>>,
}

impl Rescue {
    pub fn new() -> Self {
        let mut fft = Spectrum::new(2048);
        let mut dictionary = Vec::with_capacity(NOTES);
        for midi in 40..=88 {
            let fundamental = 440.0 * 2f64.powf((midi as f64 - 69.0) / 12.0);
            let mut column = vec![0.0; BINS];
            for harmonic in 1..=8 {
                let frequency = fundamental * harmonic as f64;
                if frequency >= 8000.0 {
                    break;
                }
                for i in 0..2048 {
                    fft.buffer[i] = Complex::new(
                        fft.window[i]
                            * (2.0 * std::f64::consts::PI * frequency * i as f64 / 16000.0).sin(),
                        0.0,
                    );
                }
                fft.fft
                    .process_with_scratch(&mut fft.buffer, &mut fft.scratch);
                for (v, bin) in column.iter_mut().zip(&fft.buffer) {
                    *v += bin.norm() / harmonic as f64;
                }
            }
            let norm = dot(&column, &column).sqrt();
            for v in &mut column {
                *v /= norm;
            }
            dictionary.push(column);
        }
        let gram = dictionary
            .iter()
            .map(|a| dictionary.iter().map(|b| dot(a, b)).collect())
            .collect();
        Self {
            dictionary,
            gram,
            past: VecDeque::new(),
            pending: [None; 12],
            pending_background: None,
        }
    }

    pub fn reset(&mut self) {
        self.past.clear();
        self.pending = [None; 12];
        self.pending_background = None;
    }

    /// An expired candidate is never carried to another confirmation window.
    /// Values returned are the previous frame's model confidence, for the latch.
    #[cfg(test)]
    pub fn confirm(&mut self, magnitude: &[f64], probabilities: &[f32; 12]) -> [Option<f32>; 12] {
        self.confirm_with_threshold(magnitude, probabilities, 0.8)
    }

    pub fn confirm_with_threshold(
        &mut self,
        magnitude: &[f64],
        probabilities: &[f32; 12],
        threshold: f32,
    ) -> [Option<f32>; 12] {
        let mut confirmed = [None; 12];
        if self.pending.iter().any(Option::is_some) {
            if let Some(shares) = self.pitch_shares(magnitude) {
                for pc in 0..12 {
                    if shares[pc] > 0.5 {
                        confirmed[pc] = self.pending[pc];
                    }
                }
            }
            // A valid new note need not be louder than the old ringing note.
            // Reuse the candidate's background: advancing it would absorb the
            // very attack we are trying to confirm. Keep the same 60% novelty
            // pitch-share requirement in this second, independent frame.
            if (0..12).any(|pc| self.pending[pc].is_some() && confirmed[pc].is_none()) {
                if let Some(background) = &self.pending_background {
                    let residual: Vec<_> = magnitude
                        .iter()
                        .zip(background)
                        .map(|(&current, &old)| (current - old).max(0.0))
                        .collect();
                    if let Some(shares) = self.pitch_shares(&residual) {
                        for pc in 0..12 {
                            if shares[pc] >= 0.6 {
                                confirmed[pc] = self.pending[pc];
                            }
                        }
                    }
                }
            }
        }
        self.pending = [None; 12];
        self.pending_background = None;
        if probabilities.iter().any(|&p| (0.6..threshold).contains(&p)) {
            let background: Vec<_> = (0..BINS)
                .map(|bin| self.past.iter().map(|frame| frame[bin]).sum::<f64>() / PAST as f64)
                .collect();
            let novelty: Vec<_> = magnitude
                .iter()
                .zip(&background)
                .map(|(&current, &old)| (current - old).max(0.0))
                .collect();
            let fraction = novelty.iter().sum::<f64>() / magnitude.iter().sum::<f64>().max(1e-12);
            if fraction >= 0.15 {
                if let Some(shares) = self.pitch_shares(&novelty) {
                    for pc in 0..12 {
                        if probabilities[pc] >= 0.6 && shares[pc] >= 0.6 {
                            self.pending[pc] = Some(probabilities[pc]);
                        }
                    }
                }
            }
            if self.pending.iter().any(Option::is_some) {
                self.pending_background = Some(background);
            }
        }
        self.past.push_back(magnitude.to_vec());
        if self.past.len() > PAST {
            self.past.pop_front();
        }
        confirmed
    }

    fn pitch_shares(&self, magnitude: &[f64]) -> Option<[f64; 12]> {
        let rhs: Vec<_> = self.dictionary.iter().map(|c| dot(c, magnitude)).collect();
        let amplitudes = nonnegative_fit(&self.gram, &rhs)?;
        let mut shares = [0.0; 12];
        for (i, amplitude) in amplitudes.iter().enumerate() {
            shares[(i + 40) % 12] += amplitude;
        }
        let sum = shares.iter().sum::<f64>().max(1e-12);
        for v in &mut shares {
            *v /= sum;
        }
        Some(shares)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Coordinate descent for the small, fixed nonnegative least-squares problem.
/// Reject un-converged evidence rather than let numerical uncertainty rescue a note.
fn nonnegative_fit(gram: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let mut amplitudes = vec![0.0; rhs.len()];
    let mut gradient = rhs.to_vec();
    let tolerance = rhs.iter().copied().fold(0.0, f64::max) * 1e-10 + 1e-12;
    for _ in 0..1000 {
        for i in 0..rhs.len() {
            let next = (amplitudes[i] + gradient[i] / gram[i][i]).max(0.0);
            let delta = next - amplitudes[i];
            amplitudes[i] = next;
            if delta != 0.0 {
                for j in 0..rhs.len() {
                    gradient[j] -= gram[j][i] * delta;
                }
            }
        }
        let converged = (0..rhs.len()).all(|i| {
            if amplitudes[i] > 0.0 {
                gradient[i].abs() <= tolerance
            } else {
                gradient[i] <= tolerance
            }
        });
        if converged {
            return Some(amplitudes);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlapping_templates_are_separated_without_negative_amplitudes() {
        let gram = vec![vec![1.0, 0.8], vec![0.8, 1.0]];
        let a = nonnegative_fit(&gram, &[1.16, 1.0]).unwrap();
        assert!((a[0] - 1.0).abs() < 1e-8 && (a[1] - 0.2).abs() < 1e-8);
        let a = nonnegative_fit(&gram, &[1.0, 0.1]).unwrap();
        assert!((a[0] - 1.0).abs() < 1e-8 && a[1] == 0.0);
    }
    #[test]
    fn quiet_new_note_is_confirmed_over_a_louder_old_note() {
        let mut r = Rescue::new();
        let old: Vec<_> = r.dictionary[52 - 40].iter().map(|v| v * 4.0).collect();
        let new = r.dictionary[55 - 40].clone();
        for _ in 0..8 {
            r.confirm(&old, &[0.0; 12]);
        }
        let attack: Vec<_> = old.iter().zip(&new).map(|(a, b)| a + b).collect();
        let mut probabilities = [0.0; 12];
        probabilities[7] = 0.7;
        assert!(r
            .confirm(&attack, &probabilities)
            .iter()
            .all(Option::is_none));
        // The raw signal is still dominated by E, not by the newly struck G.
        let tail: Vec<_> = old.iter().zip(&new).map(|(a, b)| a + 0.2 * b).collect();
        assert!(r.pitch_shares(&tail).unwrap()[7] < 0.5);
        assert_eq!(r.confirm(&tail, &[0.0; 12])[7], Some(0.7));
        assert!(r.confirm(&old, &[0.0; 12]).iter().all(Option::is_none));
    }

    #[test]
    fn sustain_alone_and_a_different_new_pitch_cannot_confirm_a_candidate() {
        let mut r = Rescue::new();
        let old: Vec<_> = r.dictionary[52 - 40].iter().map(|v| v * 4.0).collect();
        for _ in 0..8 {
            r.confirm(&old, &[0.0; 12]);
        }
        let mut probabilities = [0.0; 12];
        probabilities[7] = 0.7;
        assert!(r.confirm(&old, &probabilities).iter().all(Option::is_none));
        assert!(r.confirm(&old, &probabilities).iter().all(Option::is_none));
        let attack: Vec<_> = old
            .iter()
            .zip(&r.dictionary[55 - 40])
            .map(|(a, b)| a + b)
            .collect();
        r.confirm(&attack, &probabilities);
        let other: Vec<_> = old
            .iter()
            .zip(&r.dictionary[58 - 40])
            .map(|(a, b)| a + b)
            .collect();
        assert!(r.confirm(&other, &[0.0; 12]).iter().all(Option::is_none));
    }

    #[test]
    fn clear_removes_a_pending_weak_attack() {
        let mut r = Rescue::new();
        r.pending[4] = Some(0.7);
        r.past.push_back(vec![1.0; BINS]);
        r.pending_background = Some(vec![1.0; BINS]);
        r.reset();
        assert!(r.past.is_empty() && r.pending.iter().all(Option::is_none));
        assert!(r.pending_background.is_none());
    }
}
