//! Which evidence of a strike the "credit only what was struck" gate should
//! read: the take6 model's own onset head, or the strike detector.
//!
//! The whole pipeline over AtoA - features, the chord model every 40 ms, the
//! strike detector every hop - and `sounding_by` asked about every class after
//! each answer, once per source. Scored like `dist/latency_rules.py`: a class
//! credited that was not played in the last two seconds is false, a labelled
//! note never credited in its own span is missed.

use super::*;
use super::tests::app;
use crate::audio::{mono_pitch, CqtAnalyzer, ShortSpectrum, FFT_SIZE, HOP_LENGTH, INPUT_GAIN, MONO_MIN_SCORE, TOTAL_FEATURES};

const SR: f32 = 16_000.0;
const MODEL_PERIOD: f32 = 0.040;
const MIN_FILL: f32 = 0.5;
const MIN_FILL_CHORD: f32 = 0.9;

#[derive(Clone, Copy, Debug)]
enum Source {
    /// "Credit only what was struck" off.
    Ungated,
    /// On, with the strike detector running - the judge as the app runs it.
    Detector,
}

/// One answer of the chord model and what the strike detector knew then.
struct Answer {
    t: f32,
    pitches: [f32; 12],
    strike_age: [u32; 12],
    class_strikes: [u32; 12],
    cqt_pitch: Option<usize>,
    root: Option<NoteName>,
    confidence: f32,
}

/// One hop of the audio side: the ear, what is sounding, and the attacks.
struct Hop {
    t: f32,
    ear: Option<usize>,
    voices: u16,
    flux: bool,
    struck: Vec<usize>,
    strike_p: [f32; 12],
}

/// Runs the audio side once; every source is then judged on the same answers.
fn answers(signal: &[f32], gate: f32) -> (Vec<Answer>, Vec<Hop>) {
    let mut analyzer = CqtAnalyzer::new("dsp_weights.json").unwrap();
    let mut brain = crate::brain::ChordBrain::new(["best_model_v2_take6_chords.onnx", "best_model_v2_take6_onset.onnx"]
        .into_iter().find(|f| std::path::Path::new(f).exists()).unwrap()).unwrap();
    let mut strikes = crate::strike::Strikes::load(crate::strike::MODEL).unwrap();
    let mut short = ShortSpectrum::new();
    let mut flux = crate::flux::Flux::default();
    let a = app();
    let mut buffer = vec![0.0f32; FFT_SIZE];
    let (mut out, mut hops) = (Vec::new(), Vec::new());
    let mut next_model = 0.0f32;
    let parser = app();
    for (n, hop) in signal.chunks_exact(HOP_LENGTH).enumerate() {
        buffer.drain(..HOP_LENGTH);
        buffer.extend_from_slice(hop);
        let t = ((n + 1) * HOP_LENGTH) as f32 / SR;
        let rms = (buffer.iter().map(|v| v * v).sum::<f32>() / FFT_SIZE as f32).sqrt();
        let attack = flux.push(short.of(&buffer)).1;
        let (strike_p, struck) = strikes.push(hop).unwrap();
        let mut st = a.analysis_state.lock().unwrap();
        st.hops += 1;
        st.flux_age = st.flux_age.saturating_add(1);
        if attack {
            st.mark_flux();
        }
        st.mark_strikes(&struck);
        if rms > gate {
            let amplified: Vec<f32> = buffer.iter().map(|&x| x * INPUT_GAIN).collect();
            let (cqt, chroma, bass, _) = analyzer.compute_cqt_chroma(&amplified, true, 5.0);
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
        hops.push(Hop { t, ear: st.cqt_pitch, voices: st.voices, flux: attack, struck: struck.clone(), strike_p });
        if t >= next_model {
            next_model += MODEL_PERIOD;
            let fill = st.history_fill();
            if fill >= MIN_FILL {
                let p = brain.predict(&st.input_history).unwrap();
                let root = if fill >= MIN_FILL_CHORD { parser.parse_ai_prediction(&p.chord).0 } else { None };
                out.push(Answer {
                    t, pitches: p.pitches, strike_age: st.class_strike_age,
                    class_strikes: st.class_strikes,
                    cqt_pitch: st.cqt_pitch, root, confidence: p.confidence,
                });
            }
        }
    }
    (out, hops)
}

/// For each answer, the classes `sounding_by` lets through and by which
/// branch, and the strike counters `set_onsets` keeps from that source.
fn judged(answers: &[Answer], hops: &[Hop], source: Source, one_at_a_time: bool)
    -> Vec<(f32, Vec<(usize, u8)>, [u32; 12])>
{
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.single_notes = one_at_a_time;
    a.require_onset = !matches!(source, Source::Ungated);
    let mut h = 0;
    let mut out = Vec::new();
    for ans in answers {
        // The ear's per-hop history up to this answer, so `steady_note` is
        // what the app would have had.
        while h < hops.len() && hops[h].t <= ans.t {
            a.feed_estimate(h as u64 + 1, hops[h].ear);
            h += 1;
        }
        a.heard.cqt_pitch = ans.cqt_pitch;
        a.prev_pitches = a.last_pitches;
        a.last_pitches = ans.pitches;
        if matches!(source, Source::Detector) {
            a.heard.strikes_live = true;
            a.heard.class_strike_age = ans.strike_age;
            a.heard.class_strikes = ans.class_strikes;
        }
        let lit = (0..12)
            .filter_map(|c| a.sounding_by(c, ans.root, ans.confidence).map(|b| (c, b)))
            .collect();
        out.push((ans.t, lit, a.heard.class_strikes));
    }
    out
}

struct Score {
    false_pairs: usize,
    notes_with_false: usize,
    missed: usize,
    latency_ms: Option<f32>,
    by_branch: [usize; 5],
}

fn score(judged: &[(f32, Vec<(usize, u8)>, [u32; 12])], labels: &[(f32, usize)]) -> Score {
    let mut false_pairs = std::collections::BTreeSet::new();
    let mut by_branch = [0usize; 5];
    let (mut missed, mut lat) = (0, Vec::new());
    for (i, &(t0, pc)) in labels.iter().enumerate() {
        let end = labels.get(i + 1).map_or(t0 + 1.5, |n| n.0).min(t0 + 1.5);
        let mut first = None;
        for (t, lit, _) in judged.iter().filter(|(t, _, _)| *t >= t0 && *t <= end) {
            let played: Vec<usize> = labels.iter()
                .filter(|(tl, _)| *tl >= t - 2.0 && *tl <= t + 0.05).map(|l| l.1).collect();
            for &(c, b) in lit {
                if c == pc {
                    first.get_or_insert(t - t0);
                } else if !played.contains(&c) && false_pairs.insert((i, c)) {
                    by_branch[b as usize] += 1;
                }
            }
        }
        match first {
            Some(d) => lat.push(d),
            None => missed += 1,
        }
    }
    lat.sort_by(|x, y| x.partial_cmp(y).unwrap());
    Score {
        notes_with_false: false_pairs.iter().map(|p| p.0).collect::<std::collections::BTreeSet<_>>().len(),
        false_pairs: false_pairs.len(),
        missed,
        latency_ms: lat.get(lat.len() / 2).map(|d| d * 1000.0),
        by_branch,
    }
}

/// Every false credit on AtoA, by its distance above the note being played
/// and the branch that let it through - with the detector, as the app runs.
#[test]
#[ignore = "diagnostic: needs AtoA.wav, the chord model and the strike model"]
fn false_credits_by_interval() {
    const NAMES: [&str; 12] = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"];
    let (signal, labels) = super::repeat_harness::load_atoa_pub();
    let (answers, hops) = answers(&signal, 0.001);
    for one in [false, true] {
        let j = judged(&answers, &hops, Source::Detector, one);
        let mut seen = std::collections::BTreeSet::new();
        let mut by = std::collections::BTreeMap::<(usize, u8), u32>::new();
        for (i, &(t0, pc)) in labels.iter().enumerate() {
            let end = labels.get(i + 1).map_or(t0 + 1.5, |n| n.0).min(t0 + 1.5);
            for (t, lit, _) in j.iter().filter(|(t, _, _)| *t >= t0 && *t <= end) {
                let played: Vec<usize> = labels.iter()
                    .filter(|(tl, _)| *tl >= t - 2.0 && *tl <= t + 0.05).map(|l| l.1).collect();
                for &(c, b) in lit {
                    if c != pc && !played.contains(&c) && seen.insert((i, c)) {
                        *by.entry(((c + 12 - pc) % 12, b)).or_insert(0) += 1;
                    }
                }
            }
        }
        println!("\n{}: false credits by interval above the note played, and branch",
                 if one { "one note at a time" } else { "chords allowed" });
        for iv in 1..12 {
            let row: Vec<String> = (1..=4).map(|b| by.get(&(iv, b)).copied().unwrap_or(0).to_string()).collect();
            println!("  {:>3}: branch 1/2/3/4 = {}", NAMES[iv], row.join("/"));
        }
    }
}

#[test]
#[ignore = "diagnostic: needs AtoA.wav, the chord model and the strike model"]
fn credit_gate_by_source() {
    let (signal, labels) = super::repeat_harness::load_atoa_pub();
    let (answers, hops) = answers(&signal, 0.001);
    let fresh = answers.iter().filter(|a| a.strike_age.iter().any(|&x| x <= 22)).count();
    let strikes: usize = answers.windows(2).map(|w| (0..12).filter(|&c| w[1].strike_age[c] < w[0].strike_age[c]).count()).sum();
    println!("answers {}, with a strike in the last 22 hops {}, strike events seen {}", answers.len(), fresh, strikes);
    let sources = [Source::Ungated, Source::Detector];
    for one in [false, true] {
        println!("\n{} ({} labelled notes)", if one { "one note at a time" } else { "chords allowed" }, labels.len());
        println!("{:<14} {:>6} {:>6} {:>7} {:>8}   false by branch 1/2/3/4", "source", "false", "notes", "missed", "p50 ms");
        for s in sources {
            let r = score(&judged(&answers, &hops, s, one), &labels);
            println!("{:<14} {:>6} {:>6} {:>7} {:>8}   {:?}", format!("{s:?}"), r.false_pairs, r.notes_with_false,
                     r.missed, r.latency_ms.map_or("-".into(), |v| format!("{v:.0}")), &r.by_branch[1..]);
        }
    }
}

/// SOLITITO_WAV=<16 kHz float mono> SOLITITO_DUMP=<out.json>: the hops and
/// the answers, for labelling a recording the app made with SOLITITO_RECORD.
#[test]
#[ignore = "diagnostic: needs a recording"]
fn dump_recording() {
    let path = std::env::var("SOLITITO_WAV").unwrap();
    // "atoa" for the labelled reference, resampled as the harness reads it.
    let signal: Vec<f32> = if path == "atoa" {
        super::repeat_harness::load_atoa_pub().0
    } else {
        hound::WavReader::open(&path).unwrap().samples::<f32>().map(|s| s.unwrap()).collect()
    };
    let (answers, hops) = answers(&signal, 0.001);
    let doc = serde_json::json!({
        "hops": hops.iter().map(|h| serde_json::json!([h.t, h.ear, h.voices, h.flux, h.struck, h.strike_p])).collect::<Vec<_>>(),
        "answers": answers.iter().map(|a| serde_json::json!({
            "t": a.t, "pitches": a.pitches, "root": a.root.map(|r| r as usize),
            "confidence": a.confidence})).collect::<Vec<_>>(),
    });
    std::fs::write(std::env::var("SOLITITO_DUMP").unwrap(), serde_json::to_vec(&doc).unwrap()).unwrap();
}

/// Chord groups: start time and the classes of the chord played from then
/// on; `?` marks a stretch left out, `end` closes the last group.
fn chord_labels(path: &str) -> Vec<(f32, Option<Vec<usize>>)> {
    std::fs::read_to_string(path).unwrap().lines().skip(1).map(|l| {
        let f: Vec<&str> = l.split(',').collect();
        let pcs = (f[1] != "?" && f[1] != "end")
            .then(|| f[2].split_whitespace().map(|v| v.parse().unwrap()).collect());
        (f[0].parse().unwrap(), pcs)
    }).collect()
}

/// SOLITITO_WAV and SOLITITO_LABELS (see `chord_labels`): what each source
/// lets through the gate on chords, and whether it counts a strum.
#[test]
#[ignore = "diagnostic: needs a labelled chord recording"]
fn chord_gate_by_source() {
    let path = std::env::var("SOLITITO_WAV").unwrap();
    let signal: Vec<f32> = hound::WavReader::open(&path).unwrap().samples::<f32>().map(|s| s.unwrap()).collect();
    let groups = chord_labels(&std::env::var("SOLITITO_LABELS").unwrap());
    let (answers, hops) = answers(&signal, 0.001);
    let sources = [Source::Ungated, Source::Detector];
    let labelled = groups.iter().filter(|g| g.1.is_some()).count();
    for one in [false, true] {
        println!("\n{} - {path}, {labelled} chords", if one { "one note at a time" } else { "chords allowed" });
        println!("{:<16} {:>9} {:>7} {:>13} {:>13}", "source", "tones", "false", "strum >=2/>=1", "ring >=2/>=1");
        for s in sources {
            let j = judged(&answers, &hops, s, one);
            let (mut tones, mut total, mut false_lit) = (0, 0, 0);
            let (mut seen2, mut seen1, mut ring2, mut ring1) = (0, 0, 0, 0);
            for (g, (t0, pcs)) in groups.iter().enumerate() {
                let Some(pcs) = pcs else { continue };
                let t1 = groups[g + 1].0;
                let before = g.checked_sub(1).and_then(|p| groups[p].1.clone()).unwrap_or_default();
                let span: Vec<_> = j.iter().filter(|(t, _, _)| *t >= *t0 && *t < t1).collect();
                let lit: std::collections::BTreeSet<usize> = span.iter().flat_map(|(_, l, _)| l.iter().map(|x| x.0)).collect();
                total += pcs.len();
                tones += pcs.iter().filter(|c| lit.contains(c)).count();
                false_lit += lit.iter().filter(|c| !pcs.contains(c) && !before.contains(c)).count();
                // Strikes counted on the chord's own classes, as `chord_struck_since` does.
                let counted = |a: f32, b: f32| -> u32 {
                    let at = |x: f32| j.iter().filter(|(t, _, _)| *t <= x).last().map(|r| r.2).unwrap_or([0; 12]);
                    let (from, to) = (at(a), at(b));
                    pcs.iter().map(|&c| to[c].wrapping_sub(from[c])).sum()
                };
                let strum = counted(t0 - 0.05, t0 + 0.6);
                let ring = counted(t0 + 0.8, t1 - 0.05);
                seen2 += (strum >= 2) as u32;
                seen1 += (strum >= 1) as u32;
                ring2 += (ring >= 2) as u32;
                ring1 += (ring >= 1) as u32;
            }
            println!("{:<16} {:>4}/{:<4} {:>7} {:>6}/{:<6} {:>6}/{:<6}", format!("{s:?}"), tones, total, false_lit,
                     seen2, seen1, ring2, ring1);
        }
    }
}

/// SOLITITO_WAV and SOLITITO_NOTES (t,pc - the credits a session's journal
/// wrote): does each source still credit what the app credited live, and
/// how much later. Where the detector missed a quiet note, a rule leaning
/// on it shows here as a note lost.
#[test]
#[ignore = "diagnostic: needs a session recording and its journal"]
fn session_credits_by_source() {
    let path = std::env::var("SOLITITO_WAV").unwrap();
    let signal: Vec<f32> = hound::WavReader::open(&path).unwrap().samples::<f32>().map(|s| s.unwrap()).collect();
    let notes: Vec<(f32, usize)> = std::fs::read_to_string(std::env::var("SOLITITO_NOTES").unwrap())
        .unwrap().lines().skip(1)
        .map(|l| { let f: Vec<&str> = l.split(',').collect(); (f[0].parse().unwrap(), f[1].parse().unwrap()) })
        .collect();
    let (answers, hops) = answers(&signal, 0.001);
    for one in [true, false] {
        println!("\n{} - {path}, {} credits in the journal", if one { "one note at a time" } else { "chords allowed" }, notes.len());
        for s in [Source::Detector] {
            let j = judged(&answers, &hops, s, one);
            let (mut lost, mut late) = (0, 0);
            for &(t, pc) in &notes {
                let first = j.iter().filter(|(x, lit, _)| *x >= t - 0.6 && *x <= t + 0.6 && lit.iter().any(|l| l.0 == pc))
                    .map(|r| r.0).next();
                match first {
                    None => {
                        lost += 1;
                        if let Some(ans) = answers.iter().filter(|a| a.t <= t).last() {
                            let recent: Vec<(usize, u32)> = (0..12)
                                .filter(|&c| ans.strike_age[c] <= 188).map(|c| (c, ans.strike_age[c])).collect();
                            println!("  lost {:?}: t {t:.2} class {pc}, its last strike {} hops back, struck in 3 s: {recent:?}",
                                     s, ans.strike_age[pc]);
                        }
                    }
                    Some(x) if x > t + 0.1 => late += 1,
                    _ => {}
                }
            }
            println!("{:<16} lost {lost:>3}   more than 0.1 s later {late:>3}", format!("{s:?}"));
        }
    }
}
