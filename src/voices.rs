//! Which notes are sounding, by explaining the spectrum rather than ranking it.
//!
//! Everything else in this project answers "which note" by scoring candidates
//! and taking the best. That cannot say how MANY notes there are, and it cannot
//! tell a note from another note's partial, because the two look alike: the
//! third harmonic of a root lands on the fifth, the fifth harmonic on the major
//! third, the seventh near the minor seventh.
//!
//! Three attempts at the question failed on exactly that. Counting pitch
//! classes in the chroma: ~12 classes for a single note and for a chord alike.
//! Counting peaks in the fundamental range: more peaks for one note than for a
//! chord, because a low note's partials live in that range too. Weighing a
//! rise against the energy standing in the bins it rose in: separates nothing
//! that the quiet attacks it costs were not worth more than.
//!
//! So this one does the arithmetic instead. Take the strongest candidate,
//! SUBTRACT the partial series it predicts, and ask the residual what is left.
//! A class fully explained as somebody else's harmonic leaves nothing behind
//! and is not a voice; a class that survives the subtraction was played.

use crate::audio::{mono_pitch, CQT_BINS};

/// Partial positions in CQT bins, two per semitone, as `audio::mono_pitch`
/// uses: the octave at +24, the fifth above it at +38, and so on.
const PARTIALS: [usize; 8] = [0, 24, 38, 48, 56, 62, 67, 72];

/// Bins either side of a partial that go with it. A real note sits a few cents
/// off centre and leaves energy in the bin next door - the same reason the ear
/// searches quarter tones rather than semitone centres.
const SPREAD: usize = 1;

/// How much of a partial's energy its fundamental is allowed to claim.
///
/// Not all of it: an octave-double or a shared partial would otherwise be
/// erased by whichever note was found first, and the second voice would vanish
/// for being a neighbour's harmonic. The fundamental itself is taken outright.
const CLAIM: f32 = 0.7;

/// Least a residual peak has to score to count as another voice, as a fraction
/// of the first voice's score. A voice far quieter than the one found first is
/// what is left of its own subtraction, not a string somebody struck.
///
/// Measured on AtoA's single notes: the spurious extra voices score 0.53 to
/// 0.67 of the first, so 0.4 let every one of them through and the count sat at
/// the cap for every note.
///
/// 0.75 reads AtoA's isolated notes better still - one voice 48 times of 51
/// against 46 - but loses a fifth sounding as loudly as the note below it,
/// because the two share a partial and the subtraction takes it. Losing a tone
/// that was played is worse here than two frames of a ghost: the point of this
/// is to confirm that several strings are sounding.
const NEXT_VOICE: f32 = 0.6;

/// A new voice has to carry this fraction of the first voice's own fundamental
/// bin, in ITS own fundamental bin.
///
/// The weighted sum over partials is not enough, because a candidate BELOW a
/// sounding note collects that note as one of its own partials and scores well
/// on borrowed energy - which is what put spurious voices 14, 17 and 18
/// semitones below the note on AtoA. Subtraction cannot answer it either: it
/// removes a note's partials upward, and a sub-harmonic's evidence lies above
/// it, in the note itself. So the candidate is asked for something only a
/// struck string has - energy where its own fundamental would be.
///
/// Swept on two recordings. At 0.7, AtoA's isolated notes report exactly one
/// voice 48 times of 51 - against 41 at 0.2 - while sesja-g1, where notes come
/// every half second and overlap, still reports two voices 30 times of 87. So
/// it does not buy the single notes by refusing real second voices.
const OWN_BIN: f32 = 0.7;

/// Semitones a new voice has to stand away from one already found. Below this
/// the second reading is the first note's energy in the bin next door, which
/// put 20 of AtoA's spurious voices exactly one semitone off.
const APART: usize = 2;

/// The notes sounding in this frame, strongest first, as semitones from C1 -
/// the same scale `cqt_semitone` carries.
///
/// `max` caps the search; a guitar grip wants 4 to 6.
pub fn voices(cqt: &[f32], max: usize) -> Vec<(usize, f32)> {
    voices_with(cqt, max, OWN_BIN)
}

/// `voices` with the own-bin share exposed, so the diagnostic can sweep it.
pub fn voices_with(cqt: &[f32], max: usize, own_bin: f32) -> Vec<(usize, f32)> {
    voices_tuned(cqt, max, own_bin, NEXT_VOICE)
}

pub fn voices_tuned(cqt: &[f32], max: usize, own_bin: f32, next_voice: f32) -> Vec<(usize, f32)> {
    if cqt.len() < CQT_BINS {
        return Vec::new();
    }
    let mut residual = cqt[..CQT_BINS].to_vec();
    let mut found: Vec<(usize, f32)> = Vec::new();
    let mut first_score = 0.0f32;
    let mut first_bin = 0.0f32;
    for _ in 0..max {
        let Some((semitone, score)) = mono_pitch(&residual) else {
            break;
        };
        if found.is_empty() {
            first_score = score;
            first_bin = residual[(semitone * 2).min(CQT_BINS - 1)];
        } else {
            if score < first_score * next_voice {
                break;
            }
            if residual[(semitone * 2).min(CQT_BINS - 1)] < first_bin * own_bin {
                break;
            }
            if found.iter().any(|&(s, _)| s.abs_diff(semitone) < APART) {
                break;
            }
        }
        subtract(&mut residual, semitone);
        found.push((semitone, score));
    }
    found
}

/// Removes what a note at `semitone` would put in the spectrum.
fn subtract(residual: &mut [f32], semitone: usize) {
    let base = semitone * 2;
    for (h, offset) in PARTIALS.iter().enumerate() {
        let centre = base + offset;
        if centre >= CQT_BINS {
            break;
        }
        // The fundamental goes entirely - it is what was identified. Higher
        // partials are shared property, and only a share of them is claimed.
        let claim = if h == 0 { 1.0 } else { CLAIM };
        let from = centre.saturating_sub(SPREAD);
        let to = (centre + SPREAD).min(CQT_BINS - 1);
        for bin in from..=to {
            residual[bin] *= 1.0 - claim;
        }
    }
}

#[cfg(test)]
mod diagnostics {
    use super::*;

    /// What this says about a recording, beside what the ear alone says.
    ///
    /// SOLITITO_VOICES_WAV=file.wav [SOLITITO_VOICES_MAX=n]
    #[test]
    #[ignore = "diagnostic: requires SOLITITO_VOICES_WAV"]
    fn export_voices() -> anyhow::Result<()> {
        let path = std::env::var("SOLITITO_VOICES_WAV")?;
        let mut reader = hound::WavReader::open(&path)?;
        let spec = reader.spec();
        let raw: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, _) => {
                reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
            }
            (_, 16) => reader.samples::<i16>().map(|s| s.unwrap_or(0) as f32 / 32768.0).collect(),
            (_, bits) => {
                let full = (1i32 << (bits - 1)) as f32;
                reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / full).collect()
            }
        };
        let channel = std::env::var("SOLITITO_VOICES_CHANNEL")
            .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1)
            .saturating_sub(1).min(spec.channels as usize - 1);
        let mono: Vec<f32> = raw.chunks(spec.channels as usize).map(|f| f[channel]).collect();
        let ratio = spec.sample_rate as f32 / crate::audio::TARGET_SR as f32;
        let mut signal = Vec::with_capacity((mono.len() as f32 / ratio) as usize + 8);
        let mut read = 0.0f32;
        while read + 1.0 < mono.len() as f32 {
            let i = read as usize;
            let f = read - i as f32;
            signal.push(mono[i] + f * (mono[i + 1] - mono[i]));
            read += ratio;
        }
        let max = std::env::var("SOLITITO_VOICES_MAX")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let mut analyzer = crate::audio::CqtAnalyzer::new("dsp_weights.json")?;
        let mut rows = Vec::new();
        let mut pos = 0usize;
        let mut frame = 0usize;
        while pos + crate::audio::FFT_SIZE < signal.len() {
            let (cqt, _, _, _) = analyzer.compute_cqt_chroma(
                &signal[pos..pos + crate::audio::FFT_SIZE], true, 5.0,
            );
            let found = voices_tuned(
                &cqt, max,
                std::env::var("SOLITITO_VOICES_OWN").ok().and_then(|v| v.parse().ok()).unwrap_or(OWN_BIN),
                std::env::var("SOLITITO_VOICES_NEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(NEXT_VOICE),
            );
            rows.push(serde_json::json!({
                "t": frame as f64 * crate::audio::HOP_LENGTH as f64
                    / crate::audio::TARGET_SR as f64,
                "ear": mono_pitch(&cqt)
                    .filter(|&(_, s)| s >= crate::audio::MONO_MIN_SCORE)
                    .map(|(n, _)| n),
                "voices": found.iter()
                    .map(|&(s, v)| serde_json::json!([s, v]))
                    .collect::<Vec<_>>(),
            }));
            pos += crate::audio::HOP_LENGTH;
            frame += 1;
        }
        std::fs::write(std::env::var("SOLITITO_VOICES_OUTPUT")?, serde_json::to_vec(&rows)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spectrum holding one note and its partial series.
    fn one_note(semitone: usize) -> Vec<f32> {
        let mut cqt = vec![0.0; CQT_BINS];
        for (h, offset) in PARTIALS.iter().enumerate() {
            let bin = semitone * 2 + offset;
            if bin < CQT_BINS {
                cqt[bin] = 1.0 / (1.0 + h as f32);
            }
        }
        cqt
    }

    /// The whole point: the fifth above a note carries that note's third
    /// harmonic, so ranking the spectrum finds it there. Explaining the
    /// spectrum does not, because nothing is left once the root is taken out.
    #[test]
    fn a_single_note_is_one_voice_not_its_harmonics() {
        let found = voices(&one_note(28), 4);
        assert_eq!(found.len(), 1, "one note came back as {} voices", found.len());
        assert_eq!(found[0].0, 28);
    }

    /// And a note that is genuinely there survives the subtraction - the third
    /// above, which is the case this was built for: the fifth harmonic of a
    /// root lands on the major third, so ranking cannot tell a played third
    /// from an unplayed one.
    #[test]
    fn a_struck_third_is_its_own_voice() {
        let mut cqt = one_note(28);
        for (bin, value) in one_note(32).iter().enumerate() {
            cqt[bin] += value;
        }
        let named: Vec<usize> = voices(&cqt, 4).iter().map(|&(s, _)| s).collect();
        assert!(named.contains(&28) && named.contains(&32), "found {named:?}");
    }

    /// A fifth as loud as the note below it, which shares a partial with it.
    #[test]
    fn a_struck_fifth_is_its_own_voice() {
        let mut cqt = one_note(28);
        for (bin, value) in one_note(35).iter().enumerate() {
            cqt[bin] += value;
        }
        let named: Vec<usize> = voices(&cqt, 4).iter().map(|&(s, _)| s).collect();
        assert!(named.contains(&35), "the fifth was taken for a partial: {named:?}");
    }

    /// The limit, written down so nobody looks for the bug: an octave cannot be
    /// resolved. Its fundamental sits exactly on the second partial of the note
    /// below, so the two are the same energy in the same bin and no harmonic
    /// model can separate them. A grip doubling a note at the octave therefore
    /// reads as one voice, and anything leaning on this has to allow for it.
    #[test]
    fn an_octave_cannot_be_told_from_a_partial() {
        let mut cqt = one_note(28);
        for (bin, value) in one_note(40).iter().enumerate() {
            cqt[bin] += value;
        }
        let named: Vec<usize> = voices(&cqt, 4).iter().map(|&(s, _)| s).collect();
        assert_eq!(named, vec![28], "the octave stopped being invisible: {named:?}");
    }

    /// A candidate BELOW a sounding note collects it as one of its own
    /// partials. Subtraction cannot answer that - it removes upward - so the
    /// candidate is asked for energy where its own fundamental would be.
    #[test]
    fn a_subharmonic_is_not_a_voice() {
        let found = voices(&one_note(40), 4);
        assert!(
            found.iter().all(|&(s, _)| s >= 40),
            "something below the note was reported: {found:?}",
        );
    }

    #[test]
    fn an_empty_spectrum_has_no_voices() {
        assert!(voices(&vec![0.0; CQT_BINS], 4).is_empty());
    }
}

