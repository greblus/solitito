//! Chords struck whole, judged in Formulas on real guitar audio.
//!
//! The whole app as it runs - features, the ear, voices, flux, the strike
//! detector, the chord model every 40 ms - drives the judge of Formulas with
//! In order off, Single notes on (the strict judge) and off (the loose one, the
//! setting a player strums the set in). For
//! every strum of `akordy-g1.wav` the formula is the chord struck, and what is
//! counted is how many of its four tones were credited before the next strum.

use super::*;
use super::tests::app;
use crate::audio::{mono_pitch, CqtAnalyzer, ShortSpectrum, FFT_SIZE, HOP_LENGTH, INPUT_GAIN, MONO_MIN_SCORE, TOTAL_FEATURES};

const SR: f32 = 16_000.0;
const MODEL_PERIOD: f32 = 0.040;
const MIN_FILL: f32 = 0.5;
const GATE: f32 = 0.001;

fn load(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    let raw: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, _) => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        (_, 16) => reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect(),
        (_, bits) => {
            let full = (1i32 << (bits - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / full).collect()
        }
    };
    let mono: Vec<f32> = raw.chunks(spec.channels as usize).map(|f| f[0]).collect();
    crate::strike::resample_like_trainer(&mono, spec.sample_rate)
}

/// The strums and their pitch classes, root first. A row with no tones - an
/// unknown chord, the end - is a time with nothing to credit.
fn labels(path: &str) -> Vec<(f32, Vec<usize>)> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (f[0].parse().unwrap(), f[2].split_whitespace().map(|p| p.parse().unwrap()).collect())
        })
        .collect()
}

/// For every strum, how many of its tones were credited.
fn credits(strict: bool) -> Vec<usize> {
    let signal = load("dist/crediting_measurements/akordy-g1.wav");
    let strums = labels("dist/crediting_measurements/akordy-g1-labels.csv");
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    // As the app sets it: the strict judge is Single notes on.
    a.single_notes = strict;
    a.formula_in_order = false;
    a.require_onset = false;
    a.strict_formulas = strict;

    let mut analyzer = CqtAnalyzer::new("dsp_weights.json").unwrap();
    let mut brain = crate::brain::ChordBrain::new(
        ["best_model_v2_take6_chords.onnx", "best_model_v2_take6_onset.onnx"]
            .into_iter().find(|f| std::path::Path::new(f).exists()).unwrap(),
    ).unwrap();
    let mut strikes = crate::strike::Strikes::load(crate::strike::MODEL).unwrap();
    let mut short = ShortSpectrum::new();
    let mut flux = crate::flux::Flux::default();
    let mut buffer = vec![0.0f32; FFT_SIZE];
    let mut next_model = 0.0f32;
    let dt = HOP_LENGTH as f32 / SR;
    let mut got = vec![0usize; strums.len()];
    let mut current: Option<usize> = None;

    for (n, hop) in signal.chunks_exact(HOP_LENGTH).enumerate() {
        let t = ((n + 1) * HOP_LENGTH) as f32 / SR;
        // A new strum: the formula becomes the chord struck, nothing yet heard.
        if let Some(k) = strums.iter().rposition(|s| s.0 - 0.05 <= t) {
            if current != Some(k) {
                current = Some(k);
                let pcs = &strums[k].1;
                if pcs.is_empty() {
                    a.formula_collected.clear();
                    continue;
                }
                let root = pcs[0];
                a.formula_root = root;
                a.formula_mask = pcs.iter().fold(0u16, |m, &p| m | 1 << ((p + 12 - root) % 12));
                a.formula_collected = vec![false; pcs.len()];
                a.lap_hold = 0.0;
            }
        }
        buffer.drain(..HOP_LENGTH);
        buffer.extend_from_slice(hop);
        let rms = (buffer.iter().map(|v| v * v).sum::<f32>() / FFT_SIZE as f32).sqrt();
        let attack = flux.push(short.of(&buffer)).1;
        let (_, struck) = strikes.push(hop).unwrap();
        {
            let mut st = a.analysis_state.lock().unwrap();
            st.hops += 1;
            st.flux_age = st.flux_age.saturating_add(1);
            if attack {
                st.mark_flux();
                st.mark_onset();
            }
            st.mark_strikes(&struck);
            if rms > GATE {
                let amplified: Vec<f32> = buffer.iter().map(|&x| x * INPUT_GAIN).collect();
                let (cqt, chroma, bass) = analyzer.compute_cqt_chroma(&amplified, true, 5.0);
                let heard = mono_pitch(&cqt).filter(|&(_, s)| s >= MONO_MIN_SCORE).map(|(x, _)| x);
                st.cqt_pitch = heard.map(|x| x % 12);
                st.cqt_semitone = heard;
                st.voices = crate::voices::voices(&cqt, 4).iter().fold(0u16, |m, &(s, _)| m | 1 << (s % 12));
                st.gate_open = true;
                let mut frame = Vec::with_capacity(TOTAL_FEATURES);
                frame.extend_from_slice(&cqt);
                frame.extend_from_slice(&chroma);
                frame.extend_from_slice(&bass);
                st.push_frame(&frame);
            } else {
                st.cqt_pitch = None;
                st.cqt_semitone = None;
                st.voices = 0;
                st.gate_open = false;
                st.push_silence();
            }
            st.frames_seen += 1;
        }
        a.sync_audio_settings();
        a.tick(dt);
        if t >= next_model {
            next_model += MODEL_PERIOD;
            let (history, fill) = {
                let st = a.analysis_state.lock().unwrap();
                (st.input_history, st.history_fill())
            };
            if fill >= MIN_FILL {
                let p = brain.predict(&history).unwrap();
                a.prev_pitches = a.last_pitches;
                a.last_pitches = p.pitches;
                a.check_progress_with_ai(MODEL_PERIOD, "...", p.confidence);
            }
        }
        // Counted while the formula is still the strum's: a finished one is
        // replaced by a drawn formula, which then counts for nothing.
        if let Some(k) = current.filter(|&k| !strums[k].1.is_empty()) {
            let root = strums[k].1[0];
            if a.formula_root == root && a.formula_collected.len() == strums[k].1.len() {
                got[k] = got[k].max(a.formula_collected.iter().filter(|&&c| c).count());
            }
            if t > strums[k].0 + 2.5 {
                current = Some(k);
            }
        }
    }
    got
}

#[test]
#[ignore = "diagnostic: needs akordy-g1.wav and both models"]
fn strummed_chords_in_formulas() {
    for strict in [true, false] {
        let strums = labels("dist/crediting_measurements/akordy-g1-labels.csv");
        let got = credits(strict);
        let played: Vec<usize> = got.iter().zip(&strums).filter(|(_, s)| !s.1.is_empty()).map(|(&g, _)| g).collect();
        let total: usize = played.iter().sum();
        println!("strict {strict}: tones credited per strum {played:?} - {total} of {}", played.len() * 4);
    }
}
