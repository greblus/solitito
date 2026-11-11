//! How soon after a strike each witness names the note.
//!
//! Four witnesses read the same audio hop by hop: the strike detector, the
//! single-frame estimate on the 8192-sample window (the ear), and the tuner's
//! YIN on the last 2048 and 1024 samples. An estimate names a class once it
//! has said so for two hops running, as a credit asks of the ear.
//!
//! - `how_soon_a_note_is_named`: `line_harness`'s lines, spliced from AtoA's
//!   notes, at several speeds - when each witness first names a note, and
//!   what else it names while the note sounds.
//! - `how_much_sooner_on_real_sessions`: the user's sessions - how long
//!   before the app's credit each witness had named the note.
//! - `agreement_on_single_notes`: AtoA's notes left to ring - how often the
//!   detector and an estimate agree on a class nobody played.
//! - `one_model_pass_by_threads`: what one pass of the chord model costs.

use super::line_harness::{atoa_notes, rising, splice};
use crate::audio::{mono_pitch, CqtAnalyzer, FFT_SIZE, HOP_LENGTH, INPUT_GAIN, MONO_MIN_SCORE};

const SR: f32 = 16_000.0;
const GATE: f32 = 0.001;
/// Hops a reading has to hold to count: `CREDIT_TICKS`.
const HELD: usize = 2;
const WITNESSES: [&str; 5] = ["strike detector", "ear, 8192", "YIN, 2048", "YIN, 1024", "rise, 2048"];

/// What each witness says on every hop: the classes the detector reports
/// struck, and the class each of the four estimates names.
struct Readings {
    struck: Vec<Vec<usize>>,
    named: [Vec<Option<usize>>; 4],
}

fn read(audio: &[f32]) -> Readings {
    let mut analyzer = CqtAnalyzer::new("dsp_weights.json").unwrap();
    let mut strikes = crate::strike::Strikes::load(crate::strike::MODEL).unwrap();
    let mut buffer = vec![0.0f32; FFT_SIZE];
    let mut short = crate::audio::ShortSpectrum::new();
    let mut rise = crate::rise::Rise::default();
    let mut out = Readings { struck: Vec::new(), named: [Vec::new(), Vec::new(), Vec::new(), Vec::new()] };
    for hop in audio.chunks_exact(HOP_LENGTH) {
        buffer.drain(..HOP_LENGTH);
        buffer.extend_from_slice(hop);
        out.struck.push(strikes.push(hop).unwrap().1);
        let rms = (buffer.iter().map(|v| v * v).sum::<f32>() / FFT_SIZE as f32).sqrt();
        let ear = (rms > GATE)
            .then(|| {
                let amplified: Vec<f32> = buffer.iter().map(|&x| x * INPUT_GAIN).collect();
                let (cqt, _, _) = analyzer.compute_cqt_chroma(&amplified, true, 5.0);
                mono_pitch(&cqt).filter(|&(_, s)| s >= MONO_MIN_SCORE).map(|(n, _)| n % 12)
            })
            .flatten();
        out.named[0].push(ear);
        for (slot, samples) in [(1, 2048), (2, 1024)] {
            let end = &buffer[FFT_SIZE - samples..];
            let loud = (end.iter().map(|v| v * v).sum::<f32>() / samples as f32).sqrt() > GATE;
            let class = loud
                .then(|| crate::tuner::frequency(end, SR))
                .flatten()
                .map(|hz| crate::tuner::nearest_note(hz).0);
            out.named[slot].push(class);
        }
        out.named[3].push(rise.push(short.of(&buffer)).first().map(|class| class.0));
    }
    out
}

/// One witness over one line.
#[derive(Default)]
struct Tally {
    notes: usize,
    in_time: usize,
    delays: Vec<f32>,
    /// Hops, from 50 ms after a strike on, naming the note before / a class
    /// that is neither.
    previous: usize,
    other: usize,
}

fn hop_of(t: f32) -> usize {
    (t * SR / HOP_LENGTH as f32) as usize
}

fn tally(readings: &Readings, witness: usize, onsets: &[f32], pcs: &[usize], length: f32) -> Tally {
    let mut out = Tally { notes: onsets.len(), ..Tally::default() };
    for (k, (&on, &pc)) in onsets.iter().zip(pcs).enumerate() {
        let (from, to) = (hop_of(on), hop_of(on + length));
        let before = k.checked_sub(1).map(|j| pcs[j]);
        let first = if witness == 0 {
            (from.saturating_sub(1)..to).find(|&h| readings.struck[h].contains(&pc))
        } else {
            let named = &readings.named[witness - 1];
            (from + HELD - 1..to).find(|&h| (0..HELD).all(|back| named[h - back] == Some(pc)))
        };
        if let Some(h) = first {
            out.in_time += 1;
            out.delays.push(((h + 1) * HOP_LENGTH) as f32 / SR - on);
        }
        for h in hop_of(on + 0.05)..to {
            let said: Vec<usize> = if witness == 0 {
                readings.struck[h].clone()
            } else {
                readings.named[witness - 1][h].into_iter().collect()
            };
            for class in said.into_iter().filter(|&c| c != pc) {
                if Some(class) == before {
                    out.previous += 1;
                } else {
                    out.other += 1;
                }
            }
        }
    }
    out
}

#[test]
#[ignore = "diagnostic: needs AtoA.wav and the strike model"]
fn how_soon_a_note_is_named() {
    let (signal, notes) = atoa_notes();
    let lines: [(&str, Vec<usize>); 3] = [
        ("D dorian", vec![2, 4, 5, 7, 9, 11, 0, 2]),
        ("G mixolydian", vec![7, 9, 11, 0, 2, 4, 5, 7]),
        ("D m7 arpeggio", vec![2, 5, 9, 0, 2]),
    ];
    for ring in [0.0, 0.3] {
        for length in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15] {
            println!("{length:.2} s a note, the note before ringing {ring:.1} s under the next:");
            let mut totals: Vec<Tally> = (0..WITNESSES.len()).map(|_| Tally::default()).collect();
            for (_, pcs) in &lines {
                let line = rising(&notes, pcs);
                let (audio, onsets) = splice(&signal, &notes, &line, length, ring);
                let readings = read(&audio);
                for (witness, total) in totals.iter_mut().enumerate() {
                    let t = tally(&readings, witness, &onsets, pcs, length);
                    total.notes += t.notes;
                    total.in_time += t.in_time;
                    total.delays.extend(t.delays);
                    total.previous += t.previous;
                    total.other += t.other;
                }
            }
            for (name, t) in WITNESSES.iter().zip(&mut totals) {
                t.delays.sort_by(f32::total_cmp);
                let at = |q: f32| t.delays.get(((t.delays.len() as f32 - 1.0) * q) as usize).map_or(f32::NAN, |d| d * 1000.0);
                println!(
                    "  {name:<16} named {:>2}/{} in time, median {:>4.0} ms, slowest {:>4.0} ms; hops on the note before {:>3}, on another class {:>3}",
                    t.in_time, t.notes, at(0.5), at(1.0), t.previous, t.other
                );
            }
        }
        println!();
    }
}

/// The user's own sessions: for every note the app credited, how much sooner
/// each witness had named its class - looked for from the credit before it,
/// at most 1.5 s back.
#[test]
#[ignore = "diagnostic: needs the session recordings and the strike model"]
fn how_much_sooner_on_real_sessions() {
    for session in ["sesja-g1", "gitara-g1", "gitara-g2", "gitara-g3"] {
        let dir = "dist/crediting_measurements";
        let audio: Vec<f32> = hound::WavReader::open(format!("{dir}/{session}.wav"))
            .unwrap()
            .samples::<f32>()
            .map(|s| s.unwrap())
            .collect();
        let credits: Vec<(f32, usize)> = std::fs::read_to_string(format!("{dir}/{session}-notes.csv"))
            .unwrap()
            .lines()
            .skip(1)
            .filter_map(|l| l.split_once(','))
            .map(|(t, pc)| (t.parse().unwrap(), pc.parse().unwrap()))
            .collect();
        let readings = read(&audio);
        println!("{session}: {} credits", credits.len());
        let mut firsts: Vec<Vec<Option<f32>>> = vec![Vec::new(); WITNESSES.len()];
        for (k, &(t, pc)) in credits.iter().enumerate() {
            let floor = k.checked_sub(1).map_or(0.0, |j| credits[j].0).max(t - 1.5);
            let (from, to) = (hop_of(floor) + 1, hop_of(t).min(readings.struck.len() - 1));
            for (witness, first) in firsts.iter_mut().enumerate() {
                let hop = if witness == 0 {
                    (from..=to).find(|&h| readings.struck[h].contains(&pc))
                } else {
                    let named = &readings.named[witness - 1];
                    (from + HELD - 1..=to).find(|&h| (0..HELD).all(|back| named[h - back] == Some(pc)))
                };
                first.push(hop.map(|h| t - ((h + 1) * HOP_LENGTH) as f32 / SR));
            }
        }
        for (name, first) in WITNESSES.iter().zip(&firsts) {
            let mut leads: Vec<f32> = first.iter().flatten().copied().collect();
            leads.sort_by(f32::total_cmp);
            let at = |q: f32| leads.get(((leads.len() as f32 - 1.0) * q) as usize).map_or(f32::NAN, |d| d * 1000.0);
            println!(
                "  {name:<16} named {:>3}/{} before the credit; sooner by: quarter {:>4.0} ms, median {:>4.0} ms, three quarters {:>4.0} ms",
                leads.len(), first.len(), at(0.25), at(0.5), at(0.75)
            );
        }
        let both = |a: usize, b: usize| (0..credits.len()).filter(|&k| firsts[0][k].is_some() && (firsts[a][k].is_some() || firsts[b][k].is_some())).count();
        println!(
            "  the detector and: YIN 1024 {}, the rise {}, either {} of {}",
            both(3, 3), both(4, 4), both(3, 4), credits.len()
        );
    }
}

/// Hops after a strike report within which an estimate has to name the same
/// class for the two to count as agreeing: about 0.13 s.
const AGREE_WITHIN: usize = 8;

/// The other side: on AtoA's reviewed single notes, each left to ring, how
/// often the detector and an estimate agree on a class nobody played - and
/// how often on the one played, and how soon.
#[test]
#[ignore = "diagnostic: needs AtoA.wav and the strike model"]
fn agreement_on_single_notes() {
    let (signal, labels) = super::repeat_harness::load_atoa_pub();
    let readings = read(&signal);
    let hops = readings.struck.len();
    for (slot, name) in [(1, "YIN, 2048"), (2, "YIN, 1024"), (3, "rise, 2048"), (0, "ear, 8192")] {
        let named = &readings.named[slot];
        let held = |h: usize, pc: usize| h >= HELD - 1 && (0..HELD).all(|back| named[h - back] == Some(pc));
        let (mut right, mut wrong, mut alone_wrong, mut strikes_wrong) = (0, 0, 0, 0);
        let mut delays = Vec::new();
        for (i, &(t, pc)) in labels.iter().enumerate() {
            let (from, to) = (hop_of(t).saturating_sub(2), labels.get(i + 1).map_or(hops, |n| hop_of(n.0 - 0.05)).min(hops));
            let mut agreed_right = None;
            for h in from..to {
                for &class in &readings.struck[h] {
                    let agreed = (h..(h + AGREE_WITHIN).min(to)).find(|&at| held(at, class));
                    match (class == pc, agreed) {
                        (true, Some(at)) => agreed_right = agreed_right.or(Some(at)),
                        (false, Some(at)) => {
                            wrong += 1;
                            println!(
                                "  {name}: class {class} agreed {:.2} s after the strike of class {pc} at {t:.2} s",
                                ((at + 1) * HOP_LENGTH) as f32 / SR - t
                            );
                        }
                        (false, None) => strikes_wrong += 1,
                        (true, None) => {}
                    }
                }
            }
            if let Some(at) = agreed_right {
                right += 1;
                delays.push(((at + 1) * HOP_LENGTH) as f32 / SR - t);
            }
            // Runs of three hops or more on a class that is not the note's.
            let mut run = 0;
            for h in from..to {
                run = if named[h].is_some_and(|c| c != pc) && named[h] == named[h.saturating_sub(1)] { run + 1 } else { 0 };
                alone_wrong += (run == 2) as u32;
            }
        }
        delays.sort_by(f32::total_cmp);
        let median = delays.get(delays.len() / 2).map_or(f32::NAN, |d| d * 1000.0);
        println!(
            "{name:<10} with the detector: the note played {right}/{}, median {median:.0} ms after the strike; a class nobody played {wrong}. \
             Alone: {alone_wrong} runs of three hops on another class. Strikes of another class the estimate did not back: {strikes_wrong}",
            labels.len()
        );
    }
}

/// One pass of the chord model, by the threads it is given: the app gives it
/// one and asks every 40 ms.
#[test]
#[ignore = "diagnostic: needs the chord model"]
fn one_model_pass_by_threads() {
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::Value;
    for threads in [1usize, 2, 4] {
        let mut session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(threads)
            .unwrap()
            .commit_from_file("best_model_v2_take6_chords.onnx")
            .unwrap();
        let mut times = Vec::new();
        for pass in 0..45 {
            let input = Value::from_array((vec![1i64, 48, 168], vec![0.05f32; 48 * 168])).unwrap();
            let started = std::time::Instant::now();
            session.run(ort::inputs!["features" => input]).unwrap();
            if pass >= 5 {
                times.push(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        times.sort_by(f64::total_cmp);
        println!("{threads} thread(s): median {:.1} ms, slowest {:.1} ms", times[times.len() / 2], times[times.len() - 1]);
    }
}

/// The lines through the whole judge, as Scales judges them, down to speeds
/// the ordinary path does not reach. SOLITITO_QUICK=1 turns quick crediting on.
#[test]
#[ignore = "diagnostic: needs AtoA.wav and both models"]
fn lines_through_the_judge() {
    use crate::model::NoteName::{D, G};
    let dorian = (["1", "2", "b3", "4", "5", "6", "b7"], [0u8, 2, 3, 5, 7, 9, 10]);
    let mixolydian = (["1", "2", "3", "4", "5", "6", "b7"], [0u8, 2, 4, 5, 7, 9, 10]);
    let arpeggio = (["1", "b3", "5", "b7"], [0u8, 3, 7, 10]);
    for ring in [0.0, 0.3] {
        for length in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15] {
            super::line_harness::measure("D dorian", D, &dorian.0, &dorian.1, length, ring);
            super::line_harness::measure("G mixolydian", G, &mixolydian.0, &mixolydian.1, length, ring);
            super::line_harness::measure("D m7 arpeggio", D, &arpeggio.0, &arpeggio.1, length, ring);
        }
        println!();
    }
}

/// What the quick ear costs a hop: YIN on its samples and the rise of the
/// short spectrum, on a second of a plucked note.
#[test]
#[ignore = "diagnostic: timing"]
fn what_the_quick_ear_costs() {
    let audio: Vec<f32> = (0..FFT_SIZE + 4096 * HOP_LENGTH)
        .map(|i| {
            let t = i as f32 / SR;
            (1..=6).map(|k| (2.0 * std::f32::consts::PI * 110.0 * k as f32 * t).sin() / k as f32).sum::<f32>() * 0.2
        })
        .collect();
    let mut short = crate::audio::ShortSpectrum::new();
    let mut quick = crate::audio::QuickEar::default();
    let spectra: Vec<Vec<f32>> = (0..4096).map(|h| short.of(&audio[h * HOP_LENGTH..h * HOP_LENGTH + FFT_SIZE]).to_vec()).collect();
    let started = std::time::Instant::now();
    let mut named = 0;
    for (h, spectrum) in spectra.iter().enumerate() {
        let reading = quick.hear(&audio[h * HOP_LENGTH..h * HOP_LENGTH + FFT_SIZE], spectrum, 0.001);
        named += reading.pitch.is_some() as u32;
    }
    let micros = started.elapsed().as_secs_f64() * 1e6 / spectra.len() as f64;
    println!("{micros:.0} µs a hop of 16 ms ({named} hops named)");
}
