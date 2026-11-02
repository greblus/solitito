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

/// The notes of a falling line through `pcs`: the first the lowest `pcs[0]`
/// with room below it for the rest, each after it the nearest note below the
/// one before with the class asked.
fn falling(notes: &[(f32, u8, i8)], pcs: &[usize]) -> Vec<usize> {
    let lowest = notes.iter().map(|n| n.1).min().unwrap();
    let first = notes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.1 as usize % 12 == pcs[0] && n.1 >= lowest + 12)
        .min_by_key(|(_, n)| n.1)
        .map(|(i, _)| i)
        .expect("AtoA has no such note");
    let mut picked = vec![first];
    for &pc in &pcs[1..] {
        let ceiling = notes[*picked.last().unwrap()].1;
        let next = notes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.1 < ceiling && n.1 as usize % 12 == pc)
            // The first take of the note, as `rising` takes: AtoA holds some
            // notes twice, and the later A3 is one the judge does not credit
            // on its own - the line then stalls on the recording, not the step.
            .max_by_key(|&(i, n)| (n.1, std::cmp::Reverse(i)))
            .map(|(i, _)| i)
            .expect("AtoA has no such note");
        picked.push(next);
    }
    picked
}

/// AtoA holds only the white keys. The black ones are the note a semitone
/// below played a semitone faster - `(attack, midi, semitones shifted)` - so
/// a chromatic line can be spliced at all. An approximation: the attack and
/// the body come out a little shorter and brighter than a fretted note.
fn with_black_keys(notes: &[(f32, u8)]) -> Vec<(f32, u8, i8)> {
    let mut all: Vec<(f32, u8, i8)> = notes.iter().map(|&(t, m)| (t, m, 0)).collect();
    for &(t, m) in notes {
        if !notes.iter().any(|n| n.1 == m + 1) && !all.iter().any(|n| n.1 == m + 1) {
            all.push((t, m + 1, 1));
        }
    }
    all
}

/// `splice` for notes that may be shifted: read faster by 2^(shift/12).
fn splice_shifted(signal: &[f32], notes: &[(f32, u8, i8)], line: &[usize], length: f32, ring: f32) -> (Vec<f32>, Vec<f32>) {
    let step = (length * SR) as usize;
    let lead = (LEAD_IN * SR) as usize;
    let mut out = vec![0.0f32; lead + step * line.len() + (2.0 * SR) as usize];
    let mut onsets = Vec::new();
    for (k, &i) in line.iter().enumerate() {
        let (attack, _, shift) = notes[i];
        let ratio = 2f32.powf(shift as f32 / 12.0);
        let from = (attack - 0.05) * SR;
        let span = ((length + ring) * SR) as usize;
        let fade = (0.01 * SR) as usize;
        let at = lead + k * step;
        for j in 0..span {
            let x = from + j as f32 * ratio;
            let (i0, frac) = (x as usize, x.fract());
            if i0 + 1 >= signal.len() {
                break;
            }
            let sample = signal[i0] * (1.0 - frac) + signal[i0 + 1] * frac;
            let gain = if j + fade > span { (span - j) as f32 / fade as f32 } else { 1.0 };
            out[at + j] += sample * gain;
        }
        onsets.push((at as f32 / SR) + 0.05);
    }
    (out, onsets)
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
fn credit_times(audio: &[f32], root: NoteName, names: &[&str], intervals: &[u8], down: bool) -> Vec<Option<f32>> {
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
    if down {
        // One octave down from the top, as the line is spliced - not the
        // whole neck from wherever the hand was drawn to stand.
        a.set_scale_e_to_e(false);
        a.scale_direction = ScaleDirection::Down;
    }
    a.reset_logic_state();
    a.scale_descending = false;
    a.current_chord_index = 0;
    if std::env::var("SOLITITO_LINE_DETAIL").is_ok() {
        let chord = a.chords[0].clone();
        let targets = a.degree_targets(&chord);
        let pcs: Vec<usize> = a.ordered_active_indices(&chord).iter().map(|s| targets[s.degree]).collect();
        println!("    the app asks for {pcs:?}");
    }
    credit_with(a, audio, root, names.len() + 1)
}

/// When each step of the exercise `a` is set to was credited, the audio run
/// through the app as it runs. `root` is the chord's: the app moving to
/// another one ends the run.
fn credit_with(mut a: MyApp, audio: &[f32], root: NoteName, steps: usize) -> Vec<Option<f32>> {
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
    measure_way(name, root, names, intervals, length, ring, false);
}

fn measure_way(name: &str, root: NoteName, names: &[&str], intervals: &[u8], length: f32, ring: f32, down: bool) {
    let (signal, notes) = atoa_notes();
    let root_pc = root as usize;
    let mut pcs: Vec<usize> = intervals.iter().map(|&i| (root_pc + i as usize) % 12).collect();
    pcs.push(root_pc);
    let (line, audio, onsets) = if down {
        pcs.reverse();
        let all = with_black_keys(&notes);
        let line = falling(&all, &pcs);
        let (audio, onsets) = splice_shifted(&signal, &all, &line, length, ring);
        (line.iter().map(|&i| all[i].1).collect::<Vec<u8>>(), audio, onsets)
    } else {
        let line = rising(&notes, &pcs);
        let (audio, onsets) = splice(&signal, &notes, &line, length, ring);
        (line.iter().map(|&i| notes[i].1).collect::<Vec<u8>>(), audio, onsets)
    };
    let credited = credit_times(&audio, root, names, intervals, down);
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
        println!("    midi {line:?}");
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

/// The eight-note scales of the Harris lines, walked down from the octave as
/// the lines walk them: a semitone at the top of the dominant's (the major
/// seventh passing to the flat one) and of the sixth diminished (the sharp
/// fifth between the sixth and the fifth). Does the judge credit each step
/// when it is played, and not off the note before still sounding?
#[test]
#[ignore = "diagnostic: needs AtoA.wav and both models"]
fn harris_scales_falling() {
    use NoteName::*;
    let dominant = (["1", "2", "3", "4", "5", "6", "b7", "(7)"], [0u8, 2, 4, 5, 7, 9, 10, 11]);
    let sixth = (["1", "2", "3", "4", "5", "(#5)", "6", "7"], [0u8, 2, 4, 5, 7, 8, 9, 11]);
    let mixolydian = (["1", "2", "3", "4", "5", "6", "b7"], [0u8, 2, 4, 5, 7, 9, 10]);
    for ring in [0.0, 0.3] {
        for length in [0.5, 0.4, 0.3, 0.25] {
            if std::env::var("SOLITITO_LINE_ONLY").is_ok_and(|only| only != format!("{length}/{ring}")) {
                continue;
            }
            measure_way("G bebop down", G, &dominant.0, &dominant.1, length, ring, true);
            measure_way("C sixth dim down", C, &sixth.0, &sixth.1, length, ring, true);
            measure_way("G mixolydian down", G, &mixolydian.0, &mixolydian.1, length, ring, true);
        }
        println!();
    }
}

/// A Harris line over one chord, played as the app asks for it: each step's
/// pitch is the line's own (`harris::C3` and up). Returns the app, the MIDI
/// notes and the names of the steps.
fn harris_app(chord: Chord, recipe: crate::harris::Recipe) -> (MyApp, Vec<u8>, Vec<String>) {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = ArpExercise::HarrisLines;
    a.harris = recipe;
    a.random_mode = false;
    a.chords = vec![chord.clone()];
    a.play_order = vec![0];
    a.play_pos = 0;
    a.current_chord_index = 0;
    a.reset_logic_state();
    let steps = a.ordered_active_indices(&chord);
    let semis = a.degree_intervals(&chord);
    let names = a.degree_names(&chord);
    let midi = steps
        .iter()
        .map(|s| (crate::harris::C3 + chord.root as i32 + semis[s.degree] as i32 + 12 * s.octave as i32) as u8)
        .collect();
    (a, midi, steps.iter().map(|s| names[s.degree].clone()).collect())
}

/// The phrases of the Harris lines - the enclosure, the drop to the third,
/// the third into the root - with the half step below a chord tone that the
/// chord's scale may not hold, and the drop's leap of most of an octave. Is
/// each note credited when played?
#[test]
#[ignore = "diagnostic: needs AtoA.wav and both models"]
fn harris_enclosures() {
    use crate::harris::{Figure, Recipe, Start};
    let (signal, notes) = atoa_notes();
    let all = with_black_keys(&notes);
    let cases = [
        ("C7 from the root", Chord { root: NoteName::C, quality: ChordQuality::Dominant7 }, Start::Root),
        ("C6 from the third", Chord { root: NoteName::C, quality: ChordQuality::Major7 }, Start::Third),
        ("Dm7 from the third", Chord { root: NoteName::D, quality: ChordQuality::Minor7 }, Start::Third),
    ];
    let figures = [("enclosure", Figure::Enclosure), ("drop", Figure::DropToTheThird), ("into the root", Figure::ThirdIntoTheRoot)];
    for length in [0.5, 0.4] {
        for ((name, chord, start), (kind, figure)) in cases.iter().flat_map(|c| figures.iter().map(move |f| (c, f))) {
            let name = format!("{kind}, {name}");
            let recipe = Recipe { figure: *figure, start: *start, notes: 8 };
            let (a, midi, names) = harris_app(chord.clone(), recipe);
            // AtoA starts at A2: a phrase that drops below it is played an
            // octave up, the same notes.
            let midi: Vec<u8> = if midi.iter().all(|&m| all.iter().any(|n| n.1 == m)) {
                midi
            } else {
                midi.iter().map(|m| m + 12).collect()
            };
            // The first take of each note, as the other lines take it.
            let line: Vec<usize> = midi
                .iter()
                .map(|&m| all.iter().position(|n| n.1 == m).unwrap_or_else(|| panic!("no MIDI {m} to splice")))
                .collect();
            let (audio, onsets) = splice_shifted(&signal, &all, &line, length, 0.0);
            let credited = credit_with(a, &audio, chord.root, midi.len());
            let early: Vec<usize> =
                onsets.iter().zip(&credited).enumerate().filter(|(_, (on, at))| at.is_some_and(|t| t < **on)).map(|(k, _)| k).collect();
            let got = credited.iter().filter(|c| c.is_some()).count();
            let missed: Vec<&String> = names.iter().zip(&credited).filter(|(_, c)| c.is_none()).map(|(n, _)| n).collect();
            println!("  {name:<34} {length:.2}s: {got}/{} credited, early {early:?}, missed {missed:?}  midi {midi:?}", midi.len());
        }
    }
}
