//! Lines of single notes, judged in order on real guitar audio.
//!
//! The whole app as it runs - features, the ear, voices, flux, the strike
//! detector, the chord model every 40 ms - drives the judge of Scales, which
//! takes notes one at a time and in order, as the scale exercises of Intervals
//! do. The lines are spliced from AtoA's notes. The question is the one a scale
//! asks that an arpeggio does not: a step a semitone from the note before -
//! is it credited when it is played, or early, off the note before still
//! sounding in the analysis window?

use super::*;
use super::tests::app;
use crate::audio::{mono_pitch, CqtAnalyzer, ShortSpectrum, FFT_SIZE, HOP_LENGTH, INPUT_GAIN, MONO_MIN_SCORE, TOTAL_FEATURES};

const SR: f32 = 16_000.0;
const MODEL_PERIOD: f32 = 0.040;
const MIN_FILL: f32 = 0.5;
const GATE: f32 = 0.001;
/// Silence before the line, so the first note is not the start of the file.
const LEAD_IN: f32 = 0.5;

/// AtoA's notes with their MIDI numbers, in playing order.
fn atoa_notes() -> (Vec<f32>, Vec<(f32, u8)>) {
    let (signal, _) = super::repeat_harness::load_atoa_pub();
    let notes = std::fs::read_to_string("dist/crediting_measurements/atoa-reference-reviewed.csv")
        .unwrap()
        .lines()
        .skip(1)
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (f[3] == "played").then(|| (f[0].parse().unwrap(), f[1].parse().unwrap()))
        })
        .collect();
    (signal, notes)
}

/// The notes of a rising line through `pcs`, from the lowest `pcs[0]` AtoA
/// holds: each the nearest note above the one before with the class asked.
fn rising(notes: &[(f32, u8)], pcs: &[usize]) -> Vec<usize> {
    let mut picked = Vec::new();
    let mut floor = 0u8;
    for &pc in pcs {
        let next = notes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.1 > floor || picked.is_empty() && n.1 >= floor)
            .filter(|(_, n)| n.1 as usize % 12 == pc)
            .min_by_key(|(_, n)| n.1)
            .map(|(i, _)| i)
            .expect("AtoA has no such note");
        floor = notes[next].1;
        picked.push(next);
    }
    picked
}

/// The line as audio: each note `length` long, from just before its attack,
/// with `ring` more of it left under the next - a note on another string.
/// Returns the audio and when each note starts.
fn splice(signal: &[f32], notes: &[(f32, u8)], line: &[usize], length: f32, ring: f32) -> (Vec<f32>, Vec<f32>) {
    let step = (length * SR) as usize;
    let lead = (LEAD_IN * SR) as usize;
    let mut out = vec![0.0f32; lead + step * line.len() + (2.0 * SR) as usize];
    let mut onsets = Vec::new();
    for (k, &i) in line.iter().enumerate() {
        let from = ((notes[i].0 - 0.05) * SR) as usize;
        let span = ((length + ring) * SR) as usize;
        let fade = (0.01 * SR) as usize;
        let at = lead + k * step;
        for j in 0..span.min(signal.len() - from) {
            let gain = if j + fade > span { (span - j) as f32 / fade as f32 } else { 1.0 };
            out[at + j] += signal[from + j] * gain;
        }
        onsets.push((at as f32 / SR) + 0.05);
    }
    (out, onsets)
}

/// When each step of the scale was credited, run through the app as Scales
/// plays it.
fn credit_times(audio: &[f32], root: NoteName, names: &[&str], intervals: &[u8]) -> Vec<Option<f32>> {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    let def = ScaleDefinition {
        name: "line".into(),
        intervals: intervals.to_vec(),
        names: names.iter().map(|n| n.to_string()).collect(),
    };
    a.chords = vec![Chord { root, quality: ChordQuality::CustomScale(def) }];
    a.intervals_input = names.join(" ");
    a.scale_repeat_root = true;
    a.scale_descending = false;
    a.random_mode = false;
    a.reset_logic_state();
    a.scale_descending = false;
    a.current_chord_index = 0;

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
    let steps = names.len() + 1;
    let mut credited: Vec<Option<f32>> = vec![None; steps];
    let dt = HOP_LENGTH as f32 / SR;

    for (n, hop) in audio.chunks_exact(HOP_LENGTH).enumerate() {
        let t = ((n + 1) * HOP_LENGTH) as f32 / SR;
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
        if let Ok(window) = std::env::var("SOLITITO_LINE_TRACE") {
            let (from, to) = window.split_once('-').unwrap();
            let (from, to): (f32, f32) = (from.parse().unwrap(), to.parse().unwrap());
            if (from..to).contains(&t) {
                let due = a.current_note_step;
                let targets = a.degree_targets(&a.chords[0].clone());
                let steps = a.ordered_active_indices(&a.chords[0].clone());
                let pc = steps.get(due).map(|s| targets[s.degree] % 12).unwrap_or(99);
                let top: Vec<(usize, u8)> = {
                    let mut p: Vec<(usize, f32)> = a.last_pitches.iter().copied().enumerate().collect();
                    p.sort_by(|x, y| y.1.total_cmp(&x.1));
                    p.iter().take(3).map(|&(c, v)| (c, (v * 100.0) as u8)).collect()
                };
                println!(
                    "      {t:.2} due {due} pc {pc}: cqt {:?} steady {:?} voices {:012b} head {top:?} struck_age {} sounding {:?}",
                    a.heard.cqt_pitch, a.heard.steady_note(), a.heard.voices,
                    a.heard.class_strike_age.get(pc).copied().unwrap_or(0).min(999),
                    a.sounding_by(pc, None, 0.0)
                );
            }
        }
        // Once the whole scale is in, the app moves to a new key: stop there.
        if credited.iter().all(Option::is_some) {
            break;
        }
        for (k, slot) in credited.iter_mut().enumerate() {
            if slot.is_none() && a.collected_notes.get(k).copied().unwrap_or(false) {
                *slot = Some(t);
            }
        }
        // A finished scale starts again: its marks are cleared, and with the
        // shuffle on it moves to a new key as well.
        let cleared = credited.iter().any(Option::is_some) && !a.collected_notes.iter().any(|&c| c);
        if cleared || a.chords.first().is_none_or(|c| c.root != root) {
            // Advanced: the last step was credited on this hop.
            for slot in credited.iter_mut().filter(|s| s.is_none()) {
                *slot = Some(t);
            }
            break;
        }
    }
    credited
}

/// One line at one speed: how many steps were credited, how many before their
/// own note began, and the median delay of the rest.
fn measure(name: &str, root: NoteName, names: &[&str], intervals: &[u8], length: f32, ring: f32) {
    let (signal, notes) = atoa_notes();
    let root_pc = root as usize;
    let mut pcs: Vec<usize> = intervals.iter().map(|&i| (root_pc + i as usize) % 12).collect();
    pcs.push(root_pc);
    let line = rising(&notes, &pcs);
    let (audio, onsets) = splice(&signal, &notes, &line, length, ring);
    let credited = credit_times(&audio, root, names, intervals);
    let mut late = Vec::new();
    let mut early = Vec::new();
    for (k, (&on, at)) in onsets.iter().zip(&credited).enumerate() {
        match at {
            Some(t) if *t < on => early.push(format!("{k}:{:.0}ms", (on - t) * 1000.0)),
            Some(t) => late.push(t - on),
            None => {}
        }
    }
    late.sort_by(f32::total_cmp);
    let median = late.get(late.len() / 2).map(|d| d * 1000.0).unwrap_or(f32::NAN);
    if std::env::var("SOLITITO_LINE_DETAIL").is_ok() {
        let picked: Vec<u8> = line.iter().map(|&i| notes[i].1).collect();
        println!("    midi {picked:?}");
        for (k, (&on, at)) in onsets.iter().zip(&credited).enumerate() {
            println!("    step {k}: note at {on:.2}s, credited {}", at.map(|t| format!("{t:.2}s")).unwrap_or("-".into()));
        }
    }
    let got = credited.iter().filter(|c| c.is_some()).count();
    println!(
        "  {name:<16} {length:.2}s ring {ring:.1}: {got}/{} credited, early {early:?}, median delay {median:.0} ms",
        credited.len()
    );
}

#[test]
#[ignore = "diagnostic: needs AtoA.wav and both models"]
fn semitone_steps_in_rising_lines() {
    use NoteName::*;
    let dorian = (["1", "2", "b3", "4", "5", "6", "b7"], [0u8, 2, 3, 5, 7, 9, 10]);
    let mixolydian = (["1", "2", "3", "4", "5", "6", "b7"], [0u8, 2, 4, 5, 7, 9, 10]);
    let arpeggio = (["1", "b3", "5", "b7"], [0u8, 3, 7, 10]);
    for ring in [0.0, 0.3] {
        for length in [0.5, 0.45, 0.4, 0.35] {
            if std::env::var("SOLITITO_LINE_ONLY").map_or(true, |only| only == format!("{length}/{ring}")) {
                measure("D dorian", D, &dorian.0, &dorian.1, length, ring);
                measure("G mixolydian", G, &mixolydian.0, &mixolydian.1, length, ring);
                measure("D m7 arpeggio", D, &arpeggio.0, &arpeggio.1, length, ring);
            }
        }
        println!();
    }
}
