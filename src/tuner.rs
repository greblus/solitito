//! The tuner: the frequency of a single sounding string, and how far it lies
//! from the nearest note of equal temperament.
//!
//! YIN (de Cheveigné and Kawahara, 2002): the period is the lag at which the
//! signal best matches itself, found on the cumulative mean normalised
//! difference, refined between samples with a parabola.

/// A4, the reference the notes are counted from.
const A4_HZ: f32 = 440.0;
/// The lowest frequency looked for: below the low E (82.4 Hz) of a guitar
/// tuned down a whole step.
const LOWEST_HZ: f32 = 65.0;
/// The highest: above the 24th fret of the high e (1318.5 Hz).
const HIGHEST_HZ: f32 = 1400.0;
/// The first dip of the normalised difference below this is the period.
const THRESHOLD: f32 = 0.15;
/// A dip no lower than this is not a pitch: noise, or several strings.
const UNVOICED: f32 = 0.35;

/// The samples the tuner reads, at the end of the analysis window: two periods
/// of the lowest note and more, 128 ms at 16 kHz.
pub const SAMPLES: usize = 2048;

/// The frequency of the note in `samples`, or `None` where nothing periodic
/// sounds.
pub fn frequency(samples: &[f32], sample_rate: f32) -> Option<f32> {
    let shortest = (sample_rate / HIGHEST_HZ).floor() as usize;
    let longest = (sample_rate / LOWEST_HZ).ceil() as usize;
    let width = samples.len().checked_sub(longest)?;
    if width < longest {
        return None;
    }
    let normalised = normalised_difference(samples, longest, width);
    let lag = first_dip(&normalised, shortest)?;
    let exact = between_samples(&normalised, lag);
    Some(sample_rate / exact)
}

/// d'(τ): the squared difference of the signal with itself τ samples later,
/// over the mean of all shorter lags. 1 at τ = 0 by definition.
fn normalised_difference(samples: &[f32], longest: usize, width: usize) -> Vec<f32> {
    let mut out = vec![1.0; longest + 1];
    let mut running = 0.0;
    for lag in 1..=longest {
        let difference: f32 = (0..width).map(|i| (samples[i] - samples[i + lag]).powi(2)).sum();
        running += difference;
        out[lag] = if running > 0.0 { difference * lag as f32 / running } else { 1.0 };
    }
    out
}

/// The lag of the first dip below `THRESHOLD`, taken at its bottom; failing
/// that, the deepest dip, if it is deep enough to be a pitch.
fn first_dip(normalised: &[f32], shortest: usize) -> Option<usize> {
    let mut lag = shortest.max(2);
    while lag < normalised.len() - 1 {
        if normalised[lag] < THRESHOLD {
            while lag + 1 < normalised.len() - 1 && normalised[lag + 1] < normalised[lag] {
                lag += 1;
            }
            return Some(lag);
        }
        lag += 1;
    }
    let deepest = (shortest.max(2)..normalised.len() - 1).min_by(|&a, &b| normalised[a].total_cmp(&normalised[b]))?;
    (normalised[deepest] < UNVOICED).then_some(deepest)
}

/// The bottom of the parabola through the dip and its two neighbours.
fn between_samples(normalised: &[f32], lag: usize) -> f32 {
    let (a, b, c) = (normalised[lag - 1], normalised[lag], normalised[lag + 1]);
    let curve = a - 2.0 * b + c;
    if curve.abs() < f32::EPSILON {
        return lag as f32;
    }
    lag as f32 + 0.5 * (a - c) / curve
}

/// The note nearest to `hz`: its pitch class (0 = C), its octave in
/// scientific pitch notation, and how many cents `hz` lies above it.
pub fn nearest_note(hz: f32) -> (usize, i32, f32) {
    let semitones_from_a4 = 12.0 * (hz / A4_HZ).log2();
    let nearest = semitones_from_a4.round();
    let cents = (semitones_from_a4 - nearest) * 100.0;
    let midi = 69 + nearest as i32;
    (midi.rem_euclid(12) as usize, midi.div_euclid(12) - 1, cents)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 16000.0;

    /// A plucked string: a fundamental and weaker harmonics, decaying.
    fn string(hz: f32) -> Vec<f32> {
        (0..SAMPLES)
            .map(|i| {
                let t = i as f32 / SR;
                let decay = (-t * 3.0).exp();
                (1..=6)
                    .map(|k| (2.0 * std::f32::consts::PI * hz * k as f32 * t).sin() / k as f32)
                    .sum::<f32>()
                    * decay
            })
            .collect()
    }

    fn cents_between(a: f32, b: f32) -> f32 {
        1200.0 * (a / b).log2()
    }

    /// Every open string of standard tuning, read within a cent.
    #[test]
    fn the_open_strings_are_read_within_a_cent() {
        for hz in [82.41, 110.0, 146.83, 196.0, 246.94, 329.63] {
            let read = frequency(&string(hz), SR).expect("a pitch");
            assert!(cents_between(read, hz).abs() < 1.0, "{hz} Hz read as {read} Hz");
        }
    }

    /// A string 7 cents sharp is 7 cents sharp, not in tune.
    #[test]
    fn a_string_out_of_tune_says_by_how_much() {
        let hz = 110.0 * 2f32.powf(7.0 / 1200.0);
        let (class, octave, cents) = nearest_note(frequency(&string(hz), SR).expect("a pitch"));
        assert_eq!((class, octave), (9, 2), "A2");
        assert!((cents - 7.0).abs() < 1.0, "{cents} cents");
    }

    /// Silence and noise have no pitch.
    #[test]
    fn silence_and_noise_have_no_pitch() {
        assert_eq!(frequency(&vec![0.0; SAMPLES], SR), None);
        let mut seed = 12345u32;
        let noise: Vec<f32> = (0..SAMPLES)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (seed >> 16) as f32 / 32768.0 - 1.0
            })
            .collect();
        assert_eq!(frequency(&noise, SR), None);
    }

    #[test]
    fn notes_are_named_from_a4() {
        assert_eq!(nearest_note(440.0), (9, 4, 0.0));
        let (class, octave, cents) = nearest_note(329.63);
        assert_eq!((class, octave), (4, 4), "E4");
        assert!(cents.abs() < 0.1);
    }

    /// The tuner on the reviewed single notes of a real guitar: the note read
    /// 0.3, 0.6 and 1.0 s after each attack, against the label.
    ///
    /// SOLITITO_TUNER_WAV=AtoA.wav SOLITITO_TUNER_LABELS=atoa-reference-reviewed.csv
    #[test]
    #[ignore = "diagnostic: requires SOLITITO_TUNER_WAV and SOLITITO_TUNER_LABELS"]
    fn read_the_labelled_notes() -> anyhow::Result<()> {
        let mut reader = hound::WavReader::open(std::env::var("SOLITITO_TUNER_WAV")?)?;
        let spec = reader.spec();
        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
            _ => reader.samples::<i16>().map(|s| s.unwrap_or(0) as f32 / 32768.0).collect(),
        };
        let mono: Vec<f32> = raw.chunks(spec.channels as usize).map(|f| f[0]).collect();
        let ratio = spec.sample_rate as f32 / SR;
        let mut signal = Vec::new();
        let mut read = 0.0f32;
        while read + 1.0 < mono.len() as f32 {
            let i = read as usize;
            signal.push(mono[i] + (read - i as f32) * (mono[i + 1] - mono[i]));
            read += ratio;
        }
        let labels = std::fs::read_to_string(std::env::var("SOLITITO_TUNER_LABELS")?)?;
        let (mut right, mut octave_off, mut wrong, mut silent) = (0, 0, 0, 0);
        let mut cents_seen = Vec::new();
        for line in labels.lines().skip(1) {
            let fields: Vec<&str> = line.split(',').collect();
            let (Ok(t), Ok(midi)) = (fields[0].parse::<f32>(), fields[1].parse::<i32>()) else { continue };
            for after in [0.3f32, 0.6, 1.0] {
                let end = ((t + after) * SR) as usize;
                let Some(window) = end.checked_sub(SAMPLES).and_then(|start| signal.get(start..end)) else { continue };
                let Some(hz) = frequency(window, SR) else {
                    silent += 1;
                    continue;
                };
                let (class, octave, cents) = nearest_note(hz);
                let read_midi = (octave + 1) * 12 + class as i32;
                if read_midi == midi {
                    right += 1;
                    cents_seen.push(cents);
                } else if (read_midi - midi) % 12 == 0 {
                    octave_off += 1;
                    println!("t={t:.2}+{after}: {midi} read as {read_midi}");
                } else {
                    wrong += 1;
                    println!("t={t:.2}+{after}: {midi} read as {read_midi} ({hz:.1} Hz)");
                }
            }
        }
        cents_seen.sort_by(f32::total_cmp);
        let at = |q: f32| cents_seen[((cents_seen.len() - 1) as f32 * q) as usize];
        println!("right {right}, octave off {octave_off}, wrong {wrong}, no pitch {silent}");
        println!("cents of the right ones: 10% {:.1}, median {:.1}, 90% {:.1}", at(0.1), at(0.5), at(0.9));
        Ok(())
    }
}

