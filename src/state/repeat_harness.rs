//! Repeats judged on real guitar audio, with ground truth.
//!
//! Every reviewed note of AtoA is spliced into four signals: the note alone,
//! ringing out (a repeat must NOT be allowed), and the same note struck again
//! on top of its own decay 0.4, 0.8 and 1.2 s later (a repeat MUST be allowed,
//! and only after the second strike). The full audio pipeline runs on each -
//! features, the ear, voices, flux, the strike detector - into the real judge,
//! and the question asked is the one the rules exist for: after the first
//! credit, when does `struck_since_credit` first say yes?

use super::*;
use super::tests::app;
use crate::audio::{mono_pitch, CqtAnalyzer, ShortSpectrum, FFT_SIZE, HOP_LENGTH, MONO_MIN_SCORE};

const SR: f32 = 16_000.0;
const PRE: f32 = 0.5;
const LENGTH: f32 = 3.2;

pub(crate) struct Outcome {
    pub credited_at: Option<f32>,
    pub permitted_at: Option<f32>,
    /// Which branch of `struck_since_credit` said yes first.
    pub by: Option<&'static str>,
}

/// The branches of `struck_since_credit`, asked one at a time, so a false
/// permission can be traced to the rule that gave it.
fn why(a: &MyApp, pc: usize) -> Option<&'static str> {
    let c = a.credited[pc]?;
    if c.left && a.heard.steady_note() == Some(pc) {
        return Some("left+ear");
    }
    if a.heard.strikes_live {
        return Some("class strike");
    }
    if let (Some(now), Some(then)) = (a.heard.cqt_semitone, c.semitone) {
        if now % 12 == pc && (now >= then + 6 || now + 6 <= then) {
            return Some("octave");
        }
    }
    Some("attack")
}

fn load_atoa() -> (Vec<f32>, Vec<(f32, usize)>) {
    let mut reader = hound::WavReader::open("/home/greblus/Documents/AtoA.wav").unwrap();
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
    let signal = crate::strike::resample_like_trainer(&mono, spec.sample_rate);
    let labels = std::fs::read_to_string(
        "dist/crediting_measurements/atoa-reference-reviewed.csv",
    )
    .unwrap()
    .lines()
    .skip(1)
    .map(|l| {
        let f: Vec<&str> = l.split(',').collect();
        (f[0].parse::<f32>().unwrap(), f[2].parse::<usize>().unwrap())
    })
    .collect();
    (signal, labels)
}

pub(crate) fn load_atoa_pub() -> (Vec<f32>, Vec<(f32, usize)>) {
    load_atoa()
}

/// One note of AtoA, from just before its attack to just before the next.
fn note(signal: &[f32], labels: &[(f32, usize)], i: usize) -> Vec<f32> {
    let start = ((labels[i].0 - 0.05) * SR) as usize;
    let end = labels
        .get(i + 1)
        .map(|n| ((n.0 - 0.05) * SR) as usize)
        .unwrap_or(signal.len())
        .min(start + (1.6 * SR) as usize);
    signal[start..end].to_vec()
}

/// The note at PRE, and again at PRE + again if asked.
///
/// Struck again, the first note is DAMPED over 15 ms where the second
/// begins: a pick meeting a ringing string stops it before it starts it
/// again. Summing the two instead - an early version of this - made their
/// partials interfere, so the energy at a re-strike could FALL, and a test
/// on energy built on that material condemned a rule for a physics the
/// guitar does not have.
pub(crate) fn splice(seg: &[f32], again: Option<f32>) -> Vec<f32> {
    let mut out = vec![0.0f32; (LENGTH * SR) as usize];
    let first = (PRE * SR) as usize;
    let stop = again.map(|d| ((PRE + d) * SR) as usize);
    let fade = (0.015 * SR) as usize;
    for (k, &v) in seg.iter().enumerate() {
        let i = first + k;
        if i >= out.len() {
            break;
        }
        let gain = match stop {
            Some(s) if i >= s + fade => 0.0,
            Some(s) if i >= s => 1.0 - (i - s) as f32 / fade as f32,
            _ => 1.0,
        };
        out[i] += v * gain;
    }
    if let Some(s) = stop {
        for (k, &v) in seg.iter().enumerate() {
            if s + k < out.len() {
                out[s + k] += v;
            }
        }
    }
    out
}

/// Runs the whole pipeline over `signal` into the judge, credits `pc` the
/// first moment the ear holds it, and reports when a repeat is permitted.
pub(crate) fn judge(signal: &[f32], pc: usize, gate: f32) -> Outcome {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    // A chord without this class, so the judge never credits it itself.
    a.chords = vec![Chord { root: NoteName::from_index((pc + 1) % 12), quality: ChordQuality::Minor7 }];
    a.current_chord_index = 0;
    a.single_notes = true;
    a.interval_in_order = true;
    a.require_onset = true;
    a.reset_logic_state();

    let mut analyzer = CqtAnalyzer::new("dsp_weights.json").unwrap();
    let mut short = ShortSpectrum::new();
    let mut flux = crate::flux::Flux::default();
    let mut strikes = if std::env::var("SOLITITO_NO_STRIKES").is_ok() {
        None
    } else {
        crate::strike::Strikes::load(&std::env::var("SOLITITO_STRIKE_MODEL").unwrap_or_else(|_| crate::strike::MODEL.into())).ok().map(|mut d| {
            if let Some(v) = std::env::var("SOLITITO_REFIRE_RISE").ok().and_then(|v| v.parse().ok()) {
                d.refires.rise = v;
            }
            if let Some(r) = std::env::var("SOLITITO_STRIKE_REFRACTORY").ok().and_then(|v| v.parse().ok()) {
                d.latch.refractory = r;
            }
            d
        })
    };
    let mut buffer = vec![0.0f32; FFT_SIZE];
    let mut out = Outcome { credited_at: None, permitted_at: None, by: None };

    for (n, hop) in signal.chunks_exact(HOP_LENGTH).enumerate() {
        buffer.drain(..HOP_LENGTH);
        buffer.extend_from_slice(hop);
        let t = ((n + 1) * HOP_LENGTH) as f32 / SR;
        let rms = (buffer.iter().map(|v| v * v).sum::<f32>() / FFT_SIZE as f32).sqrt();
        let attack = flux.push(short.of(&buffer)).1;
        let struck = strikes.as_mut().and_then(|d| d.push(hop).ok()).map(|(_, s)| s);
        {
            let mut st = a.analysis_state.lock().unwrap();
            st.hops += 1;
            st.flux_age = st.flux_age.saturating_add(1);
            if attack {
                st.mark_flux();
                st.mark_onset();
            }
            if let Some(s) = &struck {
                st.mark_strikes(s);
            }
            if rms > gate {
                let (cqt, _, _) = analyzer.compute_cqt_chroma(&buffer, true, 5.0);
                let heard = mono_pitch(&cqt)
                    .filter(|&(_, s)| s >= MONO_MIN_SCORE)
                    .map(|(note, _)| note);
                st.cqt_pitch = heard.map(|n| n % 12);
                st.cqt_semitone = heard;
                st.voices = crate::voices::voices(&cqt, 4)
                    .iter()
                    .fold(0u16, |m, &(s, _)| m | 1 << (s % 12));
                st.gate_open = true;
                st.frames_seen += 1;
            } else {
                st.cqt_pitch = None;
                st.cqt_semitone = None;
                st.voices = 0;
                st.gate_open = false;
                st.frames_seen += 1;
            }
        }
        a.sync_audio_settings();
        a.tick(HOP_LENGTH as f32 / SR);
        if out.credited_at.is_none() {
            if t > PRE && a.heard.sounding_now() == Some(pc) {
                a.credit_class(pc, 0);
                out.credited_at = Some(t);
            }
        } else if out.permitted_at.is_none() && a.struck_since_credit(pc) {
            out.permitted_at = Some(t);
            out.by = why(&a, pc);
        }
    }
    out
}

/// The boundary through the whole judge: X rings, never credited, and the
/// chord changes to one that opens on X. Returns when X's step was credited.
fn judge_boundary(signal: &[f32], pc: usize, switch_at: f32, gate: f32) -> Option<f32> {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.chords = vec![
        Chord { root: NoteName::from_index((pc + 1) % 12), quality: ChordQuality::Minor7 },
        Chord { root: NoteName::from_index(pc), quality: ChordQuality::Major7 },
    ];
    a.current_chord_index = 0;
    a.single_notes = true;
    a.interval_in_order = true;
    a.require_onset = true;
    a.shuffle_chords = false;
    a.reset_logic_state();
    a.play_order = vec![0, 1];
    a.play_pos = 0;
    a.current_chord_index = 0;
    a.update_collected_notes_size();

    let mut analyzer = CqtAnalyzer::new("dsp_weights.json").unwrap();
    let mut short = ShortSpectrum::new();
    let mut flux = crate::flux::Flux::default();
    let mut strikes = if std::env::var("SOLITITO_NO_STRIKES").is_ok() {
        None
    } else {
        crate::strike::Strikes::load(&std::env::var("SOLITITO_STRIKE_MODEL").unwrap_or_else(|_| crate::strike::MODEL.into()))
            .ok()
            .map(|mut d| {
                if let Some(v) = std::env::var("SOLITITO_REFIRE_RISE").ok().and_then(|v| v.parse().ok()) {
                    d.refires.rise = v;
                }
                d
            })
    };
    let mut buffer = vec![0.0f32; FFT_SIZE];
    let mut switched = false;
    for (n, hop) in signal.chunks_exact(HOP_LENGTH).enumerate() {
        buffer.drain(..HOP_LENGTH);
        buffer.extend_from_slice(hop);
        let t = ((n + 1) * HOP_LENGTH) as f32 / SR;
        let rms = (buffer.iter().map(|v| v * v).sum::<f32>() / FFT_SIZE as f32).sqrt();
        let attack = flux.push(short.of(&buffer)).1;
        let struck = strikes.as_mut().and_then(|d| d.push(hop).ok()).map(|(_, s)| s);
        {
            let mut st = a.analysis_state.lock().unwrap();
            st.hops += 1;
            st.flux_age = st.flux_age.saturating_add(1);
            if attack {
                st.mark_flux();
                st.mark_onset();
            }
            if let Some(s) = &struck {
                st.mark_strikes(s);
            }
            if rms > gate {
                let (cqt, _, _) = analyzer.compute_cqt_chroma(&buffer, true, 5.0);
                let heard = mono_pitch(&cqt).filter(|&(_, s)| s >= MONO_MIN_SCORE).map(|(x, _)| x);
                st.cqt_pitch = heard.map(|x| x % 12);
                st.cqt_semitone = heard;
                st.voices = crate::voices::voices(&cqt, 4).iter().fold(0u16, |m, &(s, _)| m | 1 << (s % 12));
                st.gate_open = true;
            } else {
                st.cqt_pitch = None;
                st.cqt_semitone = None;
                st.voices = 0;
                st.gate_open = false;
            }
            st.frames_seen += 1;
        }
        a.sync_audio_settings();
        if !switched && t >= switch_at {
            if std::env::var("SOLITITO_BOUNDARY_DEBUG").is_ok() {
                eprintln!(
                    "przed zmiana t={t:.2}: zaliczone[X]={} glos={} ucho_stale={:?} wiek_uderzenia={} uderzen={} live={}",
                    a.credited[pc].is_some(), a.heard.voice_heard(pc), a.heard.steady_note(),
                    a.heard.class_strike_age[pc], a.heard.class_strikes[pc], a.heard.strikes_live,
                );
            }
            a.advance_chord();
            switched = true;
            if std::env::var("SOLITITO_BOUNDARY_DEBUG").is_ok() {
                eprintln!(
                    "po zmianie: akord={} zaliczone[X]={:?}",
                    a.current_chord_index, a.credited[pc].map(|c| (c.class_strike, c.left)),
                );
            }
        }
        if switched && std::env::var("SOLITITO_BOUNDARY_DEBUG").is_ok() && a.struck_since_credit(pc) {
            eprintln!(
                "  t={t:.2} dopuszczone przez: {:?}  (uderzen {}, wiek {}, left {:?})",
                why(&a, pc), a.heard.class_strikes[pc], a.heard.class_strike_age[pc], a.credited[pc].map(|c| c.left)
            );
        }
        a.tick(HOP_LENGTH as f32 / SR);
        if switched && a.current_chord_index == 1 && a.collected_notes.first() == Some(&true) {
            if std::env::var("SOLITITO_BOUNDARY_DEBUG").is_ok() {
                eprintln!(
                    "ZALICZONE t={t:.2}: zaliczone[X]={:?} uderzen={} wiek={} galaz_ucha={:?}",
                    a.credited[pc].map(|c| (c.class_strike, c.left)),
                    a.heard.class_strikes[pc], a.heard.class_strike_age[pc], a.heard.sounding_now(),
                );
            }
            return Some(t);
        }
    }
    None
}

/// The user's case: X credited in one chord, the chord changes while X
/// rings, and the next chord has X too - in any order. Returns when that
/// chord took a credit, which off a note only ringing it must not.
fn judge_ringing_credited(signal: &[f32], pc: usize, switch_at: f32, gate: f32) -> (Option<f32>, bool) {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.chords = vec![
        Chord { root: NoteName::from_index(pc), quality: ChordQuality::Major7 },
        Chord { root: NoteName::from_index((pc + 5) % 12), quality: ChordQuality::Major7 },
    ];
    a.single_notes = true;
    a.interval_in_order = false;
    a.require_onset = true;
    a.shuffle_chords = false;
    a.reset_logic_state();
    a.play_order = vec![0, 1];
    a.play_pos = 0;
    a.current_chord_index = 0;
    a.update_collected_notes_size();

    let mut analyzer = CqtAnalyzer::new("dsp_weights.json").unwrap();
    let mut short = ShortSpectrum::new();
    let mut flux = crate::flux::Flux::default();
    let mut strikes = crate::strike::Strikes::load(crate::strike::MODEL).ok();
    let mut buffer = vec![0.0f32; FFT_SIZE];
    let mut switched = false;
    let mut x_credited: Option<f32> = None;
    for (n, hop) in signal.chunks_exact(HOP_LENGTH).enumerate() {
        buffer.drain(..HOP_LENGTH);
        buffer.extend_from_slice(hop);
        let t = ((n + 1) * HOP_LENGTH) as f32 / SR;
        let rms = (buffer.iter().map(|v| v * v).sum::<f32>() / FFT_SIZE as f32).sqrt();
        let attack = flux.push(short.of(&buffer)).1;
        let struck = strikes.as_mut().and_then(|d| d.push(hop).ok()).map(|(_, s)| s);
        {
            let mut st = a.analysis_state.lock().unwrap();
            st.hops += 1;
            st.flux_age = st.flux_age.saturating_add(1);
            if attack {
                st.mark_flux();
                st.mark_onset();
            }
            if let Some(s) = &struck {
                st.mark_strikes(s);
            }
            if rms > gate {
                let (cqt, _, _) = analyzer.compute_cqt_chroma(&buffer, true, 5.0);
                let heard = mono_pitch(&cqt).filter(|&(_, s)| s >= MONO_MIN_SCORE).map(|(x, _)| x);
                st.cqt_pitch = heard.map(|x| x % 12);
                st.cqt_semitone = heard;
                st.voices = crate::voices::voices(&cqt, 4).iter().fold(0u16, |m, &(s, _)| m | 1 << (s % 12));
                st.gate_open = true;
            } else {
                st.cqt_pitch = None;
                st.cqt_semitone = None;
                st.voices = 0;
                st.gate_open = false;
            }
            st.frames_seen += 1;
        }
        a.sync_audio_settings();
        if x_credited.is_none() && a.credited[pc].is_some() {
            x_credited = Some(t);
        }
        // The app moves on once the chord is done and shown for 0.35 s:
        // never before X itself was credited there.
        if !switched && x_credited.is_some_and(|c| t >= switch_at.max(c + 0.35)) {
            a.advance_chord();
            switched = true;
        }
        a.tick(HOP_LENGTH as f32 / SR);
        if switched && a.current_chord_index == 1 && a.collected_notes.iter().any(|&c| c) {
            if std::env::var("SOLITITO_RING_DEBUG").is_ok() {
                let classes: Vec<usize> = (0..12).filter(|&c| a.credited[c].is_some()).collect();
                eprintln!(
                    "  X={pc} t={:.2}: ear {:?}, X strikes {} age {}, credit {:?}, why {:?}, credited {classes:?}",
                    t, a.heard.cqt_pitch, a.heard.class_strikes[pc], a.heard.class_strike_age[pc],
                    a.credited[pc].map(|c| (c.class_strike, c.left)), why(&a, pc),
                );
            }
            return (Some(t), true);
        }
    }
    (None, x_credited.is_some())
}

/// Real notes left ringing 3 s and more, from the user's sessions
/// (`dist/crediting_measurements/long-rings.csv`: wav,attack,pc,end), each
/// credited in one chord and still ringing into the next.
#[test]
#[ignore = "diagnostic: needs the session recordings"]
fn long_rings_into_the_next_chord() {
    let rows: Vec<(String, f32, usize, f32)> =
        std::fs::read_to_string("dist/crediting_measurements/long-rings.csv").unwrap()
            .lines().skip(1).map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                (f[0].to_string(), f[1].parse().unwrap(), f[2].parse().unwrap(), f[3].parse().unwrap())
            }).collect();
    let mut audio = std::collections::HashMap::<String, Vec<f32>>::new();
    let (mut bad, mut late, mut n, mut never) = (0, 0, 0, 0);
    for (wav, attack, pc, end) in &rows {
        let signal = audio.entry(wav.clone()).or_insert_with(|| {
            hound::WavReader::open(wav).unwrap().samples::<f32>().map(|s| s.unwrap()).collect()
        });
        let from = ((attack - 0.05) * SR) as usize;
        let to = ((*end * SR) as usize).min(signal.len());
        let mut seg = vec![0.0f32; (PRE * SR) as usize];
        seg.extend_from_slice(&signal[from..to]);
        n += 1;
        let (outcome, credited) = judge_ringing_credited(&seg, *pc, PRE + 0.7, 0.001);
        never += !credited as u32;
        if let Some(t) = outcome {
            bad += 1;
            late += (t - PRE >= 2.0) as u32;
            println!("credited while only ringing: {wav} at {attack:.2} s, class {pc}, {:.2} s after the strike", t - PRE);
        }
    }
    println!("rings credited in the next chord: {bad}/{n}, of them 2 s or more after the strike: {late}; X never credited in the first chord: {never}");
}

#[test]
#[ignore = "diagnostic"]
fn boundary_one_note() {
    let (signal, labels) = load_atoa();
    let i: usize = std::env::var("SOLITITO_NOTE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let seg = note(&signal, &labels, i);
    let r = judge_boundary(&splice(&seg, None), labels[i].1, PRE + 0.7, 0.001);
    eprintln!("nuta {i} klasa {}: {:?}", labels[i].1, r);
}

#[test]
#[ignore = "diagnostic: needs AtoA.wav and the strike model"]
fn ringing_into_the_next_chord() {
    let (signal, labels) = load_atoa();
    let gate = 0.001;
    let (mut ring_bad, mut again_ok, mut again_late, mut n) = (0, 0, 0, 0);
    for i in 0..labels.len() {
        let pc = labels[i].1;
        let seg = note(&signal, &labels, i);
        n += 1;
        // X struck at PRE; the chord moves on 0.7 s later while it rings -
        // the earliest it can in the app: ~0.3 s to credit the last note,
        // 0.35 s to show the finished set.
        if judge_boundary(&splice(&seg, None), pc, PRE + 0.7, gate).is_some() {
            ring_bad += 1;
        }
        // ...and X struck again half a second after the new chord appears.
        match judge_boundary(&splice(&seg, Some(1.2)), pc, PRE + 0.7, gate) {
            Some(t) if t >= PRE + 1.2 => again_ok += 1,
            Some(_) => again_late += 1, // credited before the second strike: carried
            None => {}
        }
    }
    println!("granica, X tylko dzwieczy: zaliczone w nowym akordzie {ring_bad}/{n}   <- musi byc 0");
    println!("granica, X uderzone znowu: zaliczone po uderzeniu {again_ok}/{n}, przed nim {again_late}");
}

/// SOLITITO_HARNESS_OUT=file.json for the per-note detail.
#[test]
#[ignore = "diagnostic: needs AtoA.wav and the strike model"]
fn repeats_on_spliced_notes() {
    let (signal, labels) = load_atoa();
    let gate = 0.001;
    let mut rows = Vec::new();
    let (mut ring_bad, mut ring_n) = (0, 0);
    let mut ring_by = std::collections::BTreeMap::<&str, u32>::new();
    let mut early_by = std::collections::BTreeMap::<&str, u32>::new();
    let mut again = std::collections::BTreeMap::<u32, (u32, u32, u32, u32)>::new();
    for i in 0..labels.len() {
        let pc = labels[i].1;
        let seg = note(&signal, &labels, i);
        let ring = judge(&splice(&seg, None), pc, gate);
        if ring.credited_at.is_some() {
            ring_n += 1;
            ring_bad += ring.permitted_at.is_some() as u32;
            if let Some(by) = ring.by {
                *ring_by.entry(by).or_insert(0) += 1;
            }
        }
        let mut row = serde_json::json!({
            "i": i, "pc": pc,
            "ring": [ring.credited_at, ring.permitted_at],
        });
        for d in [0.4f32, 0.8, 1.2] {
            let o = judge(&splice(&seg, Some(d)), pc, gate);
            let second = PRE + d;
            let e = again.entry((d * 10.0) as u32).or_default();
            if o.credited_at.is_some_and(|c| c < second) {
                e.3 += 1;
                match o.permitted_at {
                    Some(p) if p < second => {
                        e.2 += 1; // early: false
                        *early_by.entry(o.by.unwrap_or("?")).or_insert(0) += 1;
                    }
                    Some(p) if p <= second + 0.6 => e.0 += 1,         // in time
                    _ => e.1 += 1,                                    // missed
                }
            }
            row[format!("again{d}")] = serde_json::json!([o.credited_at, o.permitted_at]);
        }
        rows.push(row);
    }
    println!("tylko wybrzmiewa: powtorzenie dopuszczone {ring_bad}/{ring_n}   <- musi byc 0");
    println!("  ktora galaz: {ring_by:?}");
    println!("  przedwczesne przy ponownym uderzeniu, ktora galaz: {early_by:?}");
    for (d, (hit, miss, early, n)) in &again {
        println!(
            "uderzona znowu po {:.1} s: w czasie {hit}/{n}, przegapione {miss}, za wczesnie {early}",
            *d as f32 / 10.0
        );
    }
    if let Ok(path) = std::env::var("SOLITITO_HARNESS_OUT") {
        std::fs::write(path, serde_json::to_vec(&rows).unwrap()).unwrap();
    }
}
