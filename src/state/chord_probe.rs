//! Chords built from AtoA's single notes, named by the model.
//!
//! Four real notes - each cut from its own place in AtoA, a sharp one taken
//! from the note a semitone off and resampled - are laid over each other 25 ms
//! apart, as a strum, and fed through the features and the model as the app
//! feeds them. What is counted is how the model names the chord.

use super::*;
use crate::audio::{CqtAnalyzer, FFT_SIZE, HOP_LENGTH, INPUT_GAIN, TOTAL_FEATURES};

const SR: f32 = 16_000.0;
const MODEL_PERIOD: f32 = 0.040;

/// A note of AtoA at `midi`: its own, or a neighbour resampled to it.
fn note(signal: &[f32], notes: &[(f32, usize)], midi: usize) -> Vec<f32> {
    let pick = |m: usize| notes.iter().find(|n| n.1 == m).map(|n| n.0);
    let (t, ratio) = match (pick(midi), pick(midi - 1), pick(midi + 1)) {
        (Some(t), _, _) => (t, 1.0),
        (None, Some(t), _) => (t, 2f32.powf(1.0 / 12.0)),
        (None, None, Some(t)) => (t, 2f32.powf(-1.0 / 12.0)),
        _ => panic!("no note near midi {midi}"),
    };
    let start = ((t - 0.03) * SR) as usize;
    let len = (2.0 * SR) as usize;
    (0..len)
        .map(|i| {
            let x = start as f32 + i as f32 * ratio;
            let j = x as usize;
            let f = x - j as f32;
            let a = signal.get(j).copied().unwrap_or(0.0);
            let b = signal.get(j + 1).copied().unwrap_or(0.0);
            a + (b - a) * f
        })
        .collect()
}

/// The chord as audio: its notes 25 ms apart, then silence.
fn strum(signal: &[f32], notes: &[(f32, usize)], midis: &[usize]) -> Vec<f32> {
    let lead = (0.5 * SR) as usize;
    let mut out = vec![0.0f32; lead + (3.0 * SR) as usize];
    for (k, &m) in midis.iter().enumerate() {
        let at = lead + (k as f32 * 0.025 * SR) as usize;
        for (i, v) in note(signal, notes, m).into_iter().enumerate() {
            if let Some(o) = out.get_mut(at + i) {
                *o += v * 0.5;
            }
        }
    }
    out
}

/// What the model calls the chord from 0.4 s to 2 s after the strum.
fn named(audio: &[f32]) -> Vec<String> {
    named_between(audio, 0.9, 2.5)
}

/// What the model calls the chord every 40 ms between `from` and `to` seconds.
fn named_between(audio: &[f32], from: f32, to: f32) -> Vec<String> {
    let mut analyzer = CqtAnalyzer::new("dsp_weights.json").unwrap();
    let mut brain = crate::brain::ChordBrain::new(
        ["best_model_v2_take6_chords.onnx", "best_model_v2_take6_onset.onnx"]
            .into_iter().find(|f| std::path::Path::new(f).exists()).unwrap(),
    ).unwrap();
    let app = super::tests::app();
    let mut state = app.analysis_state.lock().unwrap();
    let mut buffer = vec![0.0f32; FFT_SIZE];
    let mut next_model = 0.0f32;
    let mut names = Vec::new();
    for (n, hop) in audio.chunks_exact(HOP_LENGTH).enumerate() {
        let t = ((n + 1) * HOP_LENGTH) as f32 / SR;
        buffer.drain(..HOP_LENGTH);
        buffer.extend_from_slice(hop);
        let rms = (buffer.iter().map(|v| v * v).sum::<f32>() / FFT_SIZE as f32).sqrt();
        if rms > 0.001 {
            let amplified: Vec<f32> = buffer.iter().map(|&x| x * INPUT_GAIN).collect();
            let (cqt, chroma, bass) = analyzer.compute_cqt_chroma(&amplified, true, 5.0);
            let mut frame = Vec::with_capacity(TOTAL_FEATURES);
            frame.extend_from_slice(&cqt);
            frame.extend_from_slice(&chroma);
            frame.extend_from_slice(&bass);
            state.push_frame(&frame);
        } else {
            state.push_silence();
        }
        if t >= next_model {
            next_model += MODEL_PERIOD;
            if (from..to).contains(&t) && state.history_fill() >= 0.5 {
                let p = brain.predict(&state.input_history).unwrap();
                names.push(format!("{} {}", p.chord, if p.confidence < 0.30 { "weak" } else { "ok" }));
            }
        }
    }
    names
}

#[test]
#[ignore = "diagnostic: needs AtoA.wav and the model"]
fn chords_built_from_single_notes() {
    let (signal, _) = super::repeat_harness::load_atoa_pub();
    let notes: Vec<(f32, usize)> = std::fs::read_to_string("dist/crediting_measurements/atoa-reference-reviewed.csv")
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (f[0].parse().unwrap(), f[1].parse().unwrap())
        })
        .collect();
    let chords: [(&str, &[usize]); 9] = [
        ("Cm7", &[48, 55, 58, 63]),
        ("Dm7", &[50, 57, 60, 65]),
        ("F7", &[53, 57, 60, 63]),
        ("Bb Maj7", &[46, 53, 57, 62]),
        ("Db dim7 (x4565x)", &[49, 55, 58, 64]),
        ("Db dim7 (xx2323)", &[52, 58, 61, 67]),
        ("Db dim7 (9x898x)", &[49, 58, 64, 67]),
        ("C dim7", &[48, 54, 57, 63]),
        ("B dim7", &[47, 53, 56, 62]),
    ];
    for (name, midis) in chords {
        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        for n in named(&strum(&signal, &notes, midis)) {
            *counts.entry(n).or_default() += 1;
        }
        let mut list: Vec<(String, usize)> = counts.into_iter().collect();
        list.sort_by(|a, b| b.1.cmp(&a.1));
        println!("{name:<18} -> {list:?}");
    }
}

/// One chord struck after another, `gap` seconds apart: the readings after
/// the second strum, every 40 ms.
fn after_change(signal: &[f32], notes: &[(f32, usize)], first: &[usize], second: &[usize], gap: f32) -> Vec<String> {
    let mut audio = strum(signal, notes, first);
    let at = (0.5 + gap) * SR;
    audio.truncate(at as usize);
    let mut tail = strum(signal, notes, second);
    tail.drain(..(0.5 * SR) as usize);
    // The first chord rings on under the second until the hand lifts.
    let ring = strum(signal, notes, first);
    for (i, v) in tail.iter_mut().enumerate() {
        let fade = (1.0 - i as f32 / (0.08 * SR)).max(0.0);
        *v += ring.get(at as usize + i).copied().unwrap_or(0.0) * fade;
    }
    audio.extend(tail);
    named_between(&audio, 0.5 + gap, 0.5 + gap + 1.0)
        .into_iter()
        .map(|n| n.trim_end_matches(" ok").to_string())
        .collect()
}

#[test]
#[ignore = "diagnostic: needs AtoA.wav and the model"]
fn readings_after_a_change_of_chord() {
    let (signal, _) = super::repeat_harness::load_atoa_pub();
    let notes: Vec<(f32, usize)> = std::fs::read_to_string("dist/crediting_measurements/atoa-reference-reviewed.csv")
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (f[0].parse().unwrap(), f[1].parse().unwrap())
        })
        .collect();
    let dm7: &[usize] = &[50, 57, 60, 65];
    let cm7: &[usize] = &[48, 55, 58, 63];
    let f7: &[usize] = &[53, 57, 60, 63];
    let db_dim: &[usize] = &[49, 55, 58, 64];
    let g7: &[usize] = &[47, 53, 55, 62];
    let changes: [(&str, &[usize], &[usize]); 4] = [
        ("Dm7 -> Db dim7", dm7, db_dim),
        ("Db dim7 -> Cm7", db_dim, cm7),
        ("Cm7 -> F7", cm7, f7),
        ("G7 -> Cm7", g7, cm7),
    ];
    for gap in [2.0, 1.0] {
        for (name, a, b) in changes {
            let seen = after_change(&signal, &notes, a, b, gap);
            println!("gap {gap} {name:<16} {}", seen.join(" | "));
        }
    }
}

/// The model's answers - name and confidence - every 40 ms over a strum.
fn answers(audio: &[f32]) -> Vec<(String, f32)> {
    let mut analyzer = CqtAnalyzer::new("dsp_weights.json").unwrap();
    let mut brain = crate::brain::ChordBrain::new(
        ["best_model_v2_take6_chords.onnx", "best_model_v2_take6_onset.onnx"]
            .into_iter().find(|f| std::path::Path::new(f).exists()).unwrap(),
    ).unwrap();
    let app = super::tests::app();
    let mut state = app.analysis_state.lock().unwrap();
    let mut buffer = vec![0.0f32; FFT_SIZE];
    let mut next_model = 0.0f32;
    let mut out = Vec::new();
    for (n, hop) in audio.chunks_exact(HOP_LENGTH).enumerate() {
        let t = ((n + 1) * HOP_LENGTH) as f32 / SR;
        buffer.drain(..HOP_LENGTH);
        buffer.extend_from_slice(hop);
        let rms = (buffer.iter().map(|v| v * v).sum::<f32>() / FFT_SIZE as f32).sqrt();
        if rms > 0.001 {
            let amplified: Vec<f32> = buffer.iter().map(|&x| x * INPUT_GAIN).collect();
            let (cqt, chroma, bass) = analyzer.compute_cqt_chroma(&amplified, true, 5.0);
            let mut frame = Vec::with_capacity(TOTAL_FEATURES);
            frame.extend_from_slice(&cqt);
            frame.extend_from_slice(&chroma);
            frame.extend_from_slice(&bass);
            state.push_frame(&frame);
        } else {
            state.push_silence();
        }
        if t >= next_model {
            next_model += MODEL_PERIOD;
            if t >= 0.5 && state.history_fill() >= 0.5 {
                let p = brain.predict(&state.input_history).unwrap();
                out.push((p.chord, p.confidence));
            }
        }
    }
    out
}

/// Seconds from the strum until Chords mode moves on from `target`, as the
/// app judges it with its own settings; None when it never does.
fn chords_mode_passes(target: &str, answers: &[(String, f32)]) -> Option<f32> {
    let mut a = super::tests::app();
    a.set_mode(AppMode::Chords as i32);
    a.chords = crate::model::parse_chords_line_pub(&format!("{target} C7"));
    a.current_chord_index = 0;
    a.prev_chord_index = None;
    a.success_timer = 0.0;
    for (k, (name, conf)) in answers.iter().enumerate() {
        a.check_progress_with_ai(MODEL_PERIOD, name, *conf);
        if a.current_chord_index != 0 {
            return Some(k as f32 * MODEL_PERIOD);
        }
    }
    None
}

#[test]
#[ignore = "diagnostic: needs AtoA.wav and the model"]
fn chords_mode_on_the_diagram_grips() {
    let (signal, _) = super::repeat_harness::load_atoa_pub();
    let notes: Vec<(f32, usize)> = std::fs::read_to_string("dist/crediting_measurements/atoa-reference-reviewed.csv")
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (f[0].parse().unwrap(), f[1].parse().unwrap())
        })
        .collect();
    let grips: [(&str, &str, &[usize]); 9] = [
        ("Cm7", "Cm7 x3534x", &[48, 55, 58, 63]),
        ("Dm7", "Dm7 x5756x", &[50, 57, 60, 65]),
        ("F7", "F7 1x121x", &[41 + 12, 51, 57, 60]),
        ("BbMaj7", "BbMaj7 x1323x", &[46, 53, 57, 62]),
        ("Dbdim", "Db dim7 E shape 9x898x", &[49, 58, 64, 67]),
        ("Dbdim", "Db dim7 A shape x45353", &[49, 55, 58, 64, 67]),
        ("Dbdim", "Db dim7 D shape xx(11)(12)(11)(12)", &[61, 67, 70, 76]),
        ("Dbdim", "Db dim7 E shape on E 0x(-1)... as E dim 12x11 12 11x", &[52, 61, 67, 70]),
        ("Dbdim", "Db dim7 A shape on Bb x1(2)0 2 0", &[46, 52, 55, 61, 64]),
    ];
    for (target, name, midis) in grips {
        let heard = answers(&strum(&signal, &notes, midis));
        let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
        for (n, _) in &heard {
            *counts.entry(n.as_str()).or_default() += 1;
        }
        let mut list: Vec<(&str, usize)> = counts.into_iter().collect();
        list.sort_by(|a, b| b.1.cmp(&a.1));
        list.truncate(4);
        println!("{name:<36} passes after {:?} s; heard {list:?}", chords_mode_passes(target, &heard));
    }
}

/// Two chords of a song, the second struck `gap` s after the first while the
/// first still rings: how long after the second strum Chords mode leaves it.
fn second_passes(signal: &[f32], notes: &[(f32, usize)], line: &str, first: &[usize], second: &[usize], gap: f32) -> Option<f32> {
    let lead = 0.5 * SR;
    let at = lead + gap * SR;
    let mut audio = strum(signal, notes, first);
    audio.truncate(at as usize);
    let mut tail = strum(signal, notes, second);
    tail.drain(..lead as usize);
    let ring = strum(signal, notes, first);
    for (i, v) in tail.iter_mut().enumerate() {
        let fade = (1.0 - i as f32 / (0.08 * SR)).max(0.0);
        *v += ring.get(at as usize + i).copied().unwrap_or(0.0) * fade;
    }
    audio.extend(tail);
    let heard = answers(&audio);
    let mut a = super::tests::app();
    a.set_mode(AppMode::Chords as i32);
    a.chords = crate::model::parse_chords_line_pub(line);
    a.current_chord_index = 0;
    a.prev_chord_index = None;
    a.success_timer = 0.0;
    let mut left_first = None;
    for (k, (name, conf)) in heard.iter().enumerate() {
        let t = 0.5 + k as f32 * MODEL_PERIOD;
        a.check_progress_with_ai(MODEL_PERIOD, name, *conf);
        if a.current_chord_index == 1 && left_first.is_none() {
            left_first = Some(t);
        }
        if a.current_chord_index == 2 {
            return Some(t - (0.5 + gap));
        }
    }
    println!("    first left at {left_first:?}; after the change heard {:?}",
        heard.iter().skip(((gap) / MODEL_PERIOD) as usize).take(20).map(|h| h.0.as_str()).collect::<Vec<_>>());
    None
}

#[test]
#[ignore = "diagnostic: needs AtoA.wav and the model"]
fn chords_mode_after_the_chord_before() {
    let (signal, _) = super::repeat_harness::load_atoa_pub();
    let notes: Vec<(f32, usize)> = std::fs::read_to_string("dist/crediting_measurements/atoa-reference-reviewed.csv")
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (f[0].parse().unwrap(), f[1].parse().unwrap())
        })
        .collect();
    let dm7: &[usize] = &[50, 57, 60, 65];
    let cm7: &[usize] = &[48, 55, 58, 63];
    let f7: &[usize] = &[53, 57, 60, 63];
    let db_e: &[usize] = &[49, 58, 64, 67];
    let db_a: &[usize] = &[49, 55, 58, 64, 67];
    let db_d: &[usize] = &[61, 67, 70, 76];
    let cases: [(&str, &str, &[usize], &[usize]); 5] = [
        ("Dm7 -> Dbdim (E shape)", "Dm7 Dbdim Cm7", dm7, db_e),
        ("Dm7 -> Dbdim (A shape)", "Dm7 Dbdim Cm7", dm7, db_a),
        ("Dm7 -> Dbdim (D shape)", "Dm7 Dbdim Cm7", dm7, db_d),
        ("Dbdim -> Cm7", "Dbdim Cm7 F7", db_a, cm7),
        ("Cm7 -> F7", "Cm7 F7 BbMaj7", cm7, f7),
    ];
    for gap in [2.0, 1.0] {
        for (name, line, a, b) in cases {
            println!("gap {gap} {name:<24} passes after {:?} s", second_passes(&signal, &notes, line, a, b, gap));
        }
    }
}

/// A recording made with SOLITITO_RECORD, read by the model as the app reads
/// it: every run of one name, with when it started, how long it held, and
/// which pitch classes the pitch head heard in the middle of it.
#[test]
#[ignore = "diagnostic: SOLITITO_PROBE_WAV=<recording>.wav"]
fn a_recording_read_by_the_model() {
    const PCS: [&str; 12] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
    let path = std::env::var("SOLITITO_PROBE_WAV").expect("SOLITITO_PROBE_WAV");
    let mut reader = hound::WavReader::open(&path).unwrap();
    let audio: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap()).collect();
    let mut analyzer = CqtAnalyzer::new("dsp_weights.json").unwrap();
    let mut brain = crate::brain::ChordBrain::new(
        ["best_model_v2_take6_chords.onnx", "best_model_v2_take6_onset.onnx"]
            .into_iter().find(|f| std::path::Path::new(f).exists()).unwrap(),
    ).unwrap();
    let app = super::tests::app();
    let mut state = app.analysis_state.lock().unwrap();
    let mut buffer = vec![0.0f32; FFT_SIZE];
    let mut next_model = 0.0f32;
    // (t, name, confidence, pitches)
    let mut readings: Vec<(f32, String, f32, [f32; 12])> = Vec::new();
    for (n, hop) in audio.chunks_exact(HOP_LENGTH).enumerate() {
        let t = ((n + 1) * HOP_LENGTH) as f32 / SR;
        buffer.drain(..HOP_LENGTH);
        buffer.extend_from_slice(hop);
        let rms = (buffer.iter().map(|v| v * v).sum::<f32>() / FFT_SIZE as f32).sqrt();
        if rms > 0.001 {
            let amplified: Vec<f32> = buffer.iter().map(|&x| x * INPUT_GAIN).collect();
            let (cqt, chroma, bass) = analyzer.compute_cqt_chroma(&amplified, true, 5.0);
            let mut frame = Vec::with_capacity(TOTAL_FEATURES);
            frame.extend_from_slice(&cqt);
            frame.extend_from_slice(&chroma);
            frame.extend_from_slice(&bass);
            state.push_frame(&frame);
        } else {
            state.push_silence();
        }
        if t >= next_model {
            next_model += MODEL_PERIOD;
            if state.history_fill() >= 0.5 {
                let p = brain.predict(&state.input_history).unwrap();
                readings.push((t, p.chord, p.confidence, p.pitches));
            }
        }
    }
    // Chords mode judging the recording against one chord, asked for again
    // each time it passes.
    if let Ok(target) = std::env::var("SOLITITO_PROBE_TARGET") {
        let mut a = super::tests::app();
        a.set_mode(AppMode::Chords as i32);
        a.chords = crate::model::parse_chords_line_pub(&format!("{target} C7"));
        let mut passes = Vec::new();
        a.current_chord_index = 0;
        a.prev_chord_index = None;
        for (t, name, confidence, pitches) in &readings {
            a.last_pitches = *pitches;
            a.check_progress_with_ai(MODEL_PERIOD, name, *confidence);
            if a.current_chord_index != 0 {
                passes.push(format!("{t:.2}"));
                a.current_chord_index = 0;
                a.prev_chord_index = None;
                a.success_timer = 0.0;
            }
        }
        println!("{target}: {} passes at {}", passes.len(), passes.join(" "));
        return;
    }
    if std::env::var("SOLITITO_PROBE_FRAMES").is_ok() {
        for (t, name, confidence, pitches) in &readings {
            let heard: Vec<&str> = (0..12).filter(|&pc| pitches[pc] > 0.5).map(|pc| PCS[pc]).collect();
            println!("frame {t:.2} {} {confidence:.2} {}", name.replace(' ', "_"), heard.join(","));
        }
        return;
    }
    let mut start = 0;
    while start < readings.len() {
        let name = &readings[start].1;
        let end = (start..readings.len()).find(|&k| &readings[k].1 != name).unwrap_or(readings.len());
        let middle = &readings[(start + end) / 2];
        let heard: Vec<&str> = (0..12).filter(|&pc| middle.3[pc] > 0.5).map(|pc| PCS[pc]).collect();
        let confidence = readings[start..end].iter().map(|r| r.2).sum::<f32>() / (end - start) as f32;
        println!(
            "{:6.2} {:5.2}s  {:<9} conf {:.2}  hears {}",
            readings[start].0,
            readings[end - 1].0 - readings[start].0 + MODEL_PERIOD,
            name,
            confidence,
            heard.join(" ")
        );
        start = end;
    }
}
