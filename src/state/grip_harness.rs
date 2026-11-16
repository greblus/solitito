//! Chords in Intervals with free order, judged on real guitar audio.
//!
//! The whole app as it runs - features, the ear, voices, flux, the strike
//! detector, the chord model every 40 ms - drives the real exercise judge,
//! `1 3 5 7` over each labelled chord. The question is the one free order
//! exists for: a chord struck at once, are all its tones credited? And for
//! each tone that is not, which rule held it back.

use super::*;
use super::tests::app;
use crate::audio::{CqtAnalyzer, ShortSpectrum, FFT_SIZE, HOP_LENGTH};

const SR: f32 = 16_000.0;
const MIN_FILL: f32 = 0.5;
const MIN_FILL_CHORD: f32 = 0.9;
const GATE: f32 = 0.001;

/// One labelled stretch: when it starts, and the chord struck there - `None`
/// for a stretch left out (`?`) and for the end of the last one.
pub(super) struct Group {
    pub t: f32,
    pub chord: Option<Chord>,
}

/// The judge's settings for one run.
#[derive(Clone, Copy, Debug)]
pub(super) struct Rules {
    pub require_onset: bool,
    pub single_notes: bool,
    pub in_order: bool,
}

/// What became of one chord.
#[derive(Debug, Default)]
pub(super) struct Result {
    /// Classes credited while this chord was the target, in order.
    pub credited: Vec<usize>,
    /// For each tone never credited: why, as the rules saw it at their best moment.
    pub missed: Vec<(usize, Why)>,
    /// Classes the strike detector reported during the chord.
    pub struck: Vec<usize>,
    /// The classes the exercise asks for over this chord.
    pub asked: Vec<usize>,
}

/// For a tone not credited: the best each check managed over the whole span.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Why {
    /// `sounding_by` said yes at least once.
    pub sounding: bool,
    /// ... and on a frame where the repeat rule also allowed it.
    pub sounding_and_fresh: bool,
    /// `struck_since_credit` said yes at least once.
    pub fresh: bool,
    /// The class was credited in the chord before.
    pub in_previous: bool,
    /// The strike detector reported this class during the chord.
    pub struck: bool,
    /// `answer_for` - everything the step needs but time - said yes.
    pub answered: bool,
    /// Free order's `best_answer` picked this step at least once.
    pub chosen: bool,
    /// Longest run of consecutive hops this step was the one picked, in hops.
    pub longest_pick: u32,
    /// Longest run of consecutive hops `answer_for` said yes, picked or not.
    pub longest_answer: u32,
    /// The most hops `answer_for` said yes within any HOLD_WINDOW hops.
    pub best_window: u32,
    /// The checks that failed on the hop the longest answer ended - see `GATES`.
    pub answer_ended_by: [bool; GATES.len()],
    /// The conditions on the frame where the most of them held - see `GATES`.
    pub best_frame: [bool; GATES.len()],
}

/// The window `best_window` counts over: 12 hops, 0.19 s, against NOTE_HOLD's 8.
const HOLD_WINDOW: u32 = 12;

/// What a tone of a chord needs from the judge, in the order the rules ask.
pub(super) const GATES: [&str; 6] = [
    "fresh",          // struck_since_credit
    "ear not other",  // the ear names nothing else, or several strings sound
    "in voices",      // `voices` holds the class
    "struck now",     // the strike detector reported it lately (require_onset)
    "head >= thr",    // the pitch head over the note threshold
    "head >= 90%",    // ... and within 90% of the strongest class
];

fn gates(a: &MyApp, pc: usize) -> [bool; GATES.len()] {
    let p = a.last_pitches[pc];
    let p_max = a.last_pitches.iter().cloned().fold(0.0f32, f32::max);
    let ear_other = a.heard.cqt_pitch.is_some_and(|now| now != pc);
    let grip = a.heard.several_sounding() && !a.single_notes;
    [
        a.struck_since_credit(pc),
        !ear_other || grip,
        a.heard.voice_heard(pc),
        !a.require_onset || a.struck_just_now(pc),
        p >= a.note_threshold,
        p >= p_max * 0.9,
    ]
}

pub(super) fn chord_named(app: &MyApp, name: &str) -> Chord {
    let split = name
        .char_indices()
        .find(|&(i, c)| i > 0 && c != '#' && c != 'b')
        .map(|(i, _)| i)
        .unwrap_or(name.len());
    let root = app.str_to_note(&name[..split]).unwrap();
    let quality = match &name[split..] {
        "Maj7" => ChordQuality::Major7,
        "m7" => ChordQuality::Minor7,
        "7" => ChordQuality::Dominant7,
        "m7b5" => ChordQuality::HalfDiminished,
        "dim7" | "dim" => ChordQuality::Diminished,
        other => panic!("unknown quality {other}"),
    };
    Chord { root, quality }
}

pub(super) fn load(wav: &str, labels: &str) -> (Vec<f32>, Vec<Group>) {
    let signal: Vec<f32> = hound::WavReader::open(wav).unwrap().samples::<f32>().map(|s| s.unwrap()).collect();
    let a = app();
    let groups = std::fs::read_to_string(labels).unwrap().lines().skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let chord = (f[1] != "?" && f[1] != "end").then(|| chord_named(&a, f[1]));
            Group { t: f[0].parse().unwrap(), chord }
        })
        .collect();
    (signal, groups)
}

/// Runs `signal` through the app with `groups` as the progression, moving to
/// each chord the moment it is struck, and reports what each chord earned.
pub(super) fn run(signal: &[f32], groups: &[Group], rules: Rules, exercise: &str) -> Vec<Result> {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.chords = groups.iter().filter_map(|g| g.chord.clone()).collect();
    // Where each labelled group sits in the progression.
    let place: Vec<Option<usize>> = groups.iter()
        .scan(0usize, |next, g| Some(g.chord.as_ref().map(|_| { *next += 1; *next - 1 })))
        .collect();
    a.intervals_input = exercise.into();
    a.single_notes = rules.single_notes;
    a.interval_in_order = rules.in_order;
    a.require_onset = rules.require_onset;
    a.random_mode = false;
    a.shuffle_chords = false;
    a.reset_logic_state();
    a.current_chord_index = 0;

    let mut analyzer = CqtAnalyzer::new("dsp_weights.json").unwrap();
    let mut brain = crate::brain::ChordBrain::new(
        ["best_model_v2_take6_chords.onnx", "best_model_v2_take6_onset.onnx"]
            .into_iter().find(|f| std::path::Path::new(f).exists()).unwrap(),
    ).unwrap();
    let mut strikes = crate::strike::Strikes::load(&crate::strike::measured_model()).unwrap();
    let mut short = ShortSpectrum::new();
    let mut flux = crate::flux::Flux::default();
    let mut buffer = vec![0.0f32; FFT_SIZE];
    let mut noise_gate = crate::audio::Gate::default();
    let mut next_model = 0.0f32;
    let mut results: Vec<Result> = groups.iter().map(|_| Result::default()).collect();
    let mut whys: Vec<[Why; 12]> = groups.iter().map(|_| [Why::default(); 12]).collect();
    let mut runs: Vec<[u32; 12]> = groups.iter().map(|_| [0u32; 12]).collect();
    let mut answer_runs: Vec<[u32; 12]> = groups.iter().map(|_| [0u32; 12]).collect();
    let mut answer_history: Vec<[u32; 12]> = groups.iter().map(|_| [0u32; 12]).collect();
    let mut collected_before: Vec<bool> = Vec::new();
    let mut group: Option<usize> = None;
    let dt = HOP_LENGTH as f32 / SR;

    for (n, hop) in signal.chunks_exact(HOP_LENGTH).enumerate() {
        let t = ((n + 1) * HOP_LENGTH) as f32 / SR;
        // The chord struck now becomes the target, as the progression would
        // have it if the player had finished the one before.
        let due = groups.iter().rposition(|g| g.t - 0.05 <= t);
        if due != group {
            if let Some(index) = due.and_then(|g| place[g]) {
                while a.current_chord_index != index {
                    a.advance_chord();
                }
                collected_before = a.collected_notes.clone();
            }
            group = due;
        }

        buffer.drain(..HOP_LENGTH);
        buffer.extend_from_slice(hop);
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
            if noise_gate.hear(&buffer, GATE) {
                st.hear(&analyzer.listen(&buffer, true, 5.0));
            } else {
                st.go_quiet();
            }
            st.frames_seen += 1;
        }
        a.sync_audio_settings();
        a.tick(dt);

        // As often as the app would ask it in this mode.
        if let Some(period) = a.model_period().filter(|_| t >= next_model) {
            next_model += period;
            let (history, fill) = {
                let st = a.analysis_state.lock().unwrap();
                (st.input_history, st.history_fill())
            };
            if fill >= MIN_FILL {
                let p = brain.predict(&history).unwrap();
                a.prev_pitches = a.last_pitches;
                a.last_pitches = p.pitches;
                let shown = if fill >= MIN_FILL_CHORD { p.chord.clone() } else { "...".into() };
                a.check_progress_with_ai(period, &shown, p.confidence);
            }
        }

        let Some(g) = group else { continue };
        let Some(chord) = groups[g].chord.as_ref() else { continue };
        if Some(a.current_chord_index) != place[g] {
            continue;
        }
        let r = &mut results[g];
        for &c in &struck {
            if !r.struck.contains(&c) {
                r.struck.push(c);
            }
        }
        // Steps credited on this hop.
        let steps = a.ordered_active_indices(chord);
        let targets = chord.get_target_indices();
        let step_targets: Vec<usize> = steps.iter().map(|s| targets[s.degree]).collect();
        r.asked = step_targets.iter().map(|x| x % 12).collect();
        let best = a.best_answer(&step_targets, a.last_ai_root, a.last_ai_conf).map(|(k, _)| k);
        for (k, step) in steps.iter().enumerate() {
            let now = a.collected_notes.get(k).copied().unwrap_or(false);
            let was = collected_before.get(k).copied().unwrap_or(false);
            let pc = targets[step.degree] % 12;
            if now && !was && !r.credited.contains(&pc) {
                r.credited.push(pc);
            }
            if !now {
                let w = &mut whys[g][pc];
                let sounding = a.sounding_by(pc, a.last_ai_root, a.last_ai_conf).is_some();
                let fresh = a.struck_since_credit(pc);
                w.sounding |= sounding;
                w.fresh |= fresh;
                w.sounding_and_fresh |= sounding && fresh;
                let answered = a.answer_for(targets[step.degree], a.last_ai_root, a.last_ai_conf).is_some();
                w.answered |= answered;
                let window = (1u32 << HOLD_WINDOW) - 1;
                answer_history[g][pc] = ((answer_history[g][pc] << 1) | answered as u32) & window;
                w.best_window = w.best_window.max(answer_history[g][pc].count_ones());
                if answered {
                    answer_runs[g][pc] += 1;
                } else if answer_runs[g][pc] > 0 {
                    if answer_runs[g][pc] >= w.longest_answer {
                        w.answer_ended_by = gates(&a, pc).map(|held| !held);
                    }
                    answer_runs[g][pc] = 0;
                }
                w.longest_answer = w.longest_answer.max(answer_runs[g][pc]);
                let picked = best == Some(k);
                w.chosen |= picked;
                runs[g][pc] = if picked { runs[g][pc] + 1 } else { 0 };
                w.longest_pick = w.longest_pick.max(runs[g][pc]);
                let now = gates(&a, pc);
                let held = |g: &[bool]| g.iter().filter(|&&x| x).count();
                if held(&now) > held(&w.best_frame) {
                    w.best_frame = now;
                }
            }
        }
        collected_before = a.collected_notes.clone();
    }

    for (g, r) in results.iter_mut().enumerate() {
        if groups[g].chord.is_none() {
            continue;
        }
        let tones: Vec<usize> = r.asked.clone();
        let previous: Vec<usize> = g.checked_sub(1)
            .and_then(|p| groups[p].chord.as_ref())
            .map(|c| c.get_target_indices().iter().map(|x| x % 12).collect())
            .unwrap_or_default();
        for pc in tones {
            if r.credited.contains(&pc) {
                continue;
            }
            let mut w = whys[g][pc];
            w.in_previous = previous.contains(&pc);
            w.struck = r.struck.contains(&pc);
            r.missed.push((pc, w));
        }
    }
    results
}

fn name(pc: usize) -> &'static str {
    ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"][pc % 12]
}

/// SOLITITO_WAV and SOLITITO_LABELS (t,chord,pcs): each chord, what it earned,
/// and why the rest did not - under free order, with and without "credit only
/// what was struck".
#[test]
#[ignore = "diagnostic: needs a labelled chord recording and both models"]
fn free_order_on_chords() {
    let wav = std::env::var("SOLITITO_WAV").unwrap();
    let (signal, groups) = load(&wav, &std::env::var("SOLITITO_LABELS").unwrap());
    for require_onset in [true, false] {
        let rules = Rules { require_onset, single_notes: false, in_order: false };
        let results = run(&signal, &groups, rules, "1 3 5 7");
        let (got, total): (usize, usize) = results.iter().zip(&groups)
            .filter_map(|(r, g)| g.chord.as_ref().map(|c| (r.credited.len(), c.get_target_indices().len())))
            .fold((0, 0), |s, x| (s.0 + x.0, s.1 + x.1));
        println!("\n{wav} - free order, require_onset {require_onset}: {got} of {total} tones credited");
        println!("  missed tone: sounding / sounding when fresh / fresh at all / in chord before / struck reported / answered / chosen / longest pick (hops)");
        println!("               then: longest answer (hops) / most answered in {HOLD_WINDOW} hops / failed when the longest answer ended");
        let mut reasons = [0usize; 7];
        for (g, r) in results.iter().enumerate() {
            let Some(chord) = groups[g].chord.as_ref() else { continue };
            let credited: Vec<&str> = r.credited.iter().map(|&c| name(c)).collect();
            let struck: Vec<&str> = r.struck.iter().map(|&c| name(c)).collect();
            println!("  {:6.2} {:<6} credited {:?} struck {:?}", groups[g].t,
                     format!("{}{}", chord.root.to_string(), chord.quality.to_string()),
                     credited, struck);
            for (pc, w) in &r.missed {
                let ended: Vec<&str> = GATES.iter().zip(w.answer_ended_by).filter(|(_, f)| *f).map(|(g, _)| *g).collect();
                println!("         missed {:<3} {} {} {} {} {} {} {} {}  | {} {} {:?}", name(*pc), w.sounding as u8, w.sounding_and_fresh as u8,
                         w.fresh as u8, w.in_previous as u8, w.struck as u8, w.answered as u8, w.chosen as u8, w.longest_pick,
                         w.longest_answer, w.best_window, ended);
                for (i, f) in [w.sounding, w.sounding_and_fresh, w.fresh, w.in_previous, w.struck, w.answered, w.chosen].iter().enumerate() {
                    reasons[i] += *f as usize;
                }
            }
        }
        println!("  totals over missed tones: {reasons:?}");
    }
}

/// One note of AtoA, from just before its attack to just before the next.
fn atoa_note(signal: &[f32], labels: &[(f32, usize)], i: usize) -> Vec<f32> {
    let start = ((labels[i].0 - 0.05) * SR) as usize;
    let end = labels.get(i + 1).map(|n| ((n.0 - 0.05) * SR) as usize)
        .unwrap_or(signal.len())
        .min(start + (1.6 * SR) as usize);
    signal[start..end].to_vec()
}

/// Triads made of real AtoA notes - root, third, fifth, struck 15 ms apart
/// like a quick strum - with the fifth at `fifth_gain`. Every triad the scale
/// AtoA plays holds, from each occurrence of its root, the third and fifth
/// taken from the same occurrence of theirs. Three seconds apart, each its own
/// chord in the progression, `1 3 5` asked over it.
fn spliced_triads(fifth_gain: f32) -> (Vec<f32>, Vec<Group>) {
    let (atoa, labels) = super::repeat_harness::load_atoa_pub();
    let occurrences = |pc: usize| -> Vec<usize> {
        labels.iter().enumerate().filter(|(_, l)| l.1 == pc % 12).map(|(i, _)| i).collect()
    };
    let mut signal = vec![0.0f32; (1.0 * SR) as usize];
    let mut groups = Vec::new();
    for root in 0..12 {
        let roots = occurrences(root);
        let Some(third_up) = [4, 3].into_iter().find(|d| !occurrences(root + d).is_empty()) else { continue };
        let Some(fifth_up) = [7, 6].into_iter().find(|d| !occurrences(root + d).is_empty()) else { continue };
        let quality = match (third_up, fifth_up) {
            (4, _) => ChordQuality::Major7,
            (_, 7) => ChordQuality::Minor7,
            _ => ChordQuality::HalfDiminished,
        };
        let (thirds, fifths) = (occurrences(root + third_up), occurrences(root + fifth_up));
        for (n, &r) in roots.iter().enumerate() {
            let notes = [(r, 1.0), (thirds[n % thirds.len()], 1.0), (fifths[n % fifths.len()], fifth_gain)];
            let at = signal.len();
            signal.resize(at + (3.0 * SR) as usize, 0.0);
            for (k, (i, gain)) in notes.into_iter().enumerate() {
                let offset = at + k * (0.015 * SR) as usize;
                for (j, v) in atoa_note(&atoa, &labels, i).into_iter().enumerate() {
                    if let Some(slot) = signal.get_mut(offset + j) {
                        *slot += v * gain;
                    }
                }
            }
            groups.push(Group {
                t: at as f32 / SR + 0.05,
                chord: Some(Chord { root: NoteName::from_index(root), quality: quality.clone() }),
            });
        }
    }
    groups.push(Group { t: signal.len() as f32 / SR, chord: None });
    (signal, groups)
}

/// Triads of real notes with the fifth at full level, -6 and -12 dB: how many
/// tones free order credits, and how many fifths.
#[test]
#[ignore = "diagnostic: needs AtoA.wav and both models"]
fn quiet_fifth_on_spliced_triads() {
    for require_onset in [true, false] {
        println!("\nfree order, require_onset {require_onset}");
        for (db, gain) in [(0, 1.0f32), (-6, 0.5), (-12, 0.25), (-99, 0.0)] {
            let (signal, groups) = spliced_triads(gain);
            let rules = Rules { require_onset, single_notes: false, in_order: false };
            let results = run(&signal, &groups, rules, "1 3 5");
            let chords: Vec<(&Result, &Chord)> = results.iter().zip(&groups)
                .filter_map(|(r, g)| g.chord.as_ref().map(|c| (r, c))).collect();
            let tones: usize = chords.iter().map(|(r, _)| r.credited.len()).sum();
            let fifth_of = |c: &Chord| (c.root as usize + c.quality.intervals()[2] as usize) % 12;
            let fifths = chords.iter().filter(|(r, c)| r.credited.contains(&fifth_of(c))).count();
            let missed: Vec<&Why> = chords.iter()
                .filter_map(|(r, c)| r.missed.iter().find(|m| m.0 == fifth_of(c)).map(|m| &m.1))
                .collect();
            let picks: Vec<u32> = missed.iter().map(|w| w.longest_pick).collect();
            let answers: Vec<u32> = missed.iter().map(|w| w.longest_answer).collect();
            let windows: Vec<u32> = missed.iter().map(|w| w.best_window).collect();
            println!("  fifth {db:>3} dB: {tones} of {} tones, fifths {fifths} of {}; longest pick of each missed fifth {picks:?}",
                     chords.len() * 3, chords.len());
            println!("               longest answer {answers:?}");
            println!("               most answered in {HOLD_WINDOW} hops {windows:?}");
        }
    }
}

/// For every tone free order missed, the conditions that failed on the frame
/// where it came closest - with and without "credit only what was struck".
/// SOLITITO_WAV/SOLITITO_LABELS for a labelled recording; without them the
/// spliced triads with the fifth at -12 dB.
#[test]
#[ignore = "diagnostic: needs the recordings and both models"]
fn what_holds_back_missed_tones() {
    let (signal, groups, exercise) = match (std::env::var("SOLITITO_WAV"), std::env::var("SOLITITO_LABELS")) {
        (Ok(wav), Ok(labels)) => {
            let (s, g) = load(&wav, &labels);
            (s, g, "1 3 5 7")
        }
        _ => {
            let (s, g) = spliced_triads(0.25);
            (s, g, "1 3 5")
        }
    };
    for require_onset in [true, false] {
        let rules = Rules { require_onset, single_notes: false, in_order: false };
        let results = run(&signal, &groups, rules, exercise);
        let missed: Vec<&Why> = results.iter().flat_map(|r| r.missed.iter().map(|m| &m.1)).collect();
        let mut failed = [0usize; GATES.len()];
        let mut alone = [0usize; GATES.len()];
        for w in &missed {
            let failing: Vec<usize> = (0..GATES.len()).filter(|&i| !w.best_frame[i]).collect();
            for &i in &failing {
                failed[i] += 1;
            }
            if let [only] = failing[..] {
                alone[only] += 1;
            }
        }
        println!("\nrequire_onset {require_onset}: {} tones missed", missed.len());
        println!("  {:<14} {:>8} {:>14}", "condition", "failed", "the only one");
        for i in 0..GATES.len() {
            println!("  {:<14} {:>8} {:>14}", GATES[i], failed[i], alone[i]);
        }
    }
}

