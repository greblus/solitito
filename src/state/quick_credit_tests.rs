//! Quick crediting: the strike detector and the quick ear agreeing on the
//! note asked for, with no answer from the model and none from the ear.

use super::tests::app;
use super::*;
use crate::audio::QuickReading;

/// A scale of three notes from C, one octave up, with quick crediting on.
fn scale() -> MyApp {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    let names = ["1", "2", "3"];
    let def = ScaleDefinition {
        name: "line".into(),
        intervals: vec![0, 2, 4],
        names: names.iter().map(|n| n.to_string()).collect(),
    };
    a.chords = vec![Chord { root: NoteName::C, quality: ChordQuality::CustomScale(def) }];
    a.intervals_input = names.join(" ");
    a.random_mode = false;
    a.set_scale_e_to_e(false);
    a.reset_logic_state();
    a.current_chord_index = 0;
    a.quick_credit = true;
    a
}

/// One hop: the classes the detector reports struck and what the quick ear
/// reads, then the judge's own clock.
fn hop(a: &mut MyApp, struck: &[usize], quick: QuickReading) {
    {
        let mut st = a.analysis_state.lock().unwrap();
        st.mark_strikes(struck);
        st.frames_seen += 1;
        st.quick = quick;
    }
    a.sync_audio_settings();
    a.tick(0.016);
}

/// YIN reading `pc` for `hops` hops, the rise nothing.
fn yin(pc: usize, hops: u32) -> QuickReading {
    QuickReading { pitch: Some(pc), pitch_for: hops, ..QuickReading::default() }
}

/// The rise naming `pc` for `hops` hops, YIN nothing.
fn rise(pc: usize, hops: u32) -> QuickReading {
    QuickReading { risen: Some(pc), risen_for: hops, ..QuickReading::default() }
}

fn silent() -> QuickReading {
    QuickReading::default()
}

/// Struck, and named for two hops: the step is credited on that hop, the
/// model never asked.
#[test]
fn a_step_is_credited_the_hop_the_two_agree() {
    let mut a = scale();
    hop(&mut a, &[0], yin(0, 1));
    assert!(!a.is_collected(0), "one hop of the quick ear was believed");
    hop(&mut a, &[], yin(0, 2));
    assert!(a.is_collected(0), "struck and named, and not credited");
    assert_eq!(a.current_note_step, 1);
    // The next step, D, by the rise alone: a note struck beside a ringing one.
    hop(&mut a, &[2], rise(2, 1));
    hop(&mut a, &[], rise(2, 2));
    assert!(a.is_collected(1), "the rise named the note struck, and it was not credited");
}

/// The option off: the same sound credits nothing here.
#[test]
fn nothing_is_credited_quickly_with_the_option_off() {
    let mut a = scale();
    a.quick_credit = false;
    hop(&mut a, &[0], yin(0, 1));
    hop(&mut a, &[], yin(0, 2));
    assert!(!a.is_collected(0));
}

/// Neither witness alone, and not the two on different notes.
#[test]
fn one_witness_is_not_enough() {
    let mut a = scale();
    // Named and never struck.
    for hops in 1..6 {
        hop(&mut a, &[], yin(0, hops));
    }
    assert!(!a.is_collected(0), "a note named and never struck was credited");
    // Struck, and the quick ear reads something else.
    hop(&mut a, &[0], yin(5, 1));
    hop(&mut a, &[], yin(5, 2));
    assert!(!a.is_collected(0), "a strike with another note named was credited");
    // Struck too long ago to be the note now named.
    for _ in 0..QUICK_AGREE {
        hop(&mut a, &[], silent());
    }
    hop(&mut a, &[], yin(0, 1));
    hop(&mut a, &[], yin(0, 2));
    assert!(!a.is_collected(0), "a stale strike and a late name were taken for one note");
}

/// One strike credits one step: the class asked for again needs its own.
#[test]
fn a_strike_is_not_credited_twice() {
    let mut a = scale();
    hop(&mut a, &[0], yin(0, 1));
    hop(&mut a, &[], yin(0, 2));
    assert!(a.is_collected(0));
    assert!(!a.struck_and_named(0), "the strike just credited counts again");
    hop(&mut a, &[0], yin(0, 3));
    assert!(a.struck_and_named(0), "a new strike of the class was refused");
}

/// A pause is the player's turn: nothing is judged.
#[test]
fn nothing_is_credited_quickly_while_paused() {
    let mut a = scale();
    a.paused = true;
    hop(&mut a, &[0], yin(0, 1));
    hop(&mut a, &[], yin(0, 2));
    assert!(!a.is_collected(0));
}

/// Where chords may be played - Intervals without "single notes" - the quick
/// ear is not asked: it reads one note.
#[test]
fn chords_are_not_credited_quickly() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.single_notes = false;
    a.quick_credit = true;
    a.reset_logic_state();
    let chord = a.chords[a.current_chord_index].clone();
    let root = chord.root as usize;
    hop(&mut a, &[root], yin(root, 1));
    hop(&mut a, &[], yin(root, 2));
    assert!(a.collected_notes.iter().all(|&done| !done));
    assert!(!a.analysis_state.lock().unwrap().quick_ear, "the quick ear was asked for where it is not used");
}

/// The fretboard trainer: the note asked for, struck and named, is passed
/// and held on screen.
#[test]
fn the_fretboard_note_is_passed_quickly() {
    let mut a = app();
    a.set_mode(AppMode::Fretboard as i32);
    a.quick_credit = true;
    let target = a.fret_target.expect("a note to play");
    hop(&mut a, &[target], yin(target, 1));
    assert_eq!(a.fret_hold, 0.0);
    hop(&mut a, &[], yin(target, 2));
    assert!(a.fret_hold > 0.0, "struck and named, and not passed");
    assert_eq!(a.match_status, MatchStatus::Exact);
}

/// Formulas, one note at a time: a function struck and named is marked, and
/// in order only the lowest still wanted.
#[test]
fn a_function_of_a_formula_is_marked_quickly() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.single_notes = true;
    a.quick_credit = true;
    a.formula_exercise = FormulaExercise::InKey;
    a.formula_root = 0;
    a.formula_mask = 1 | 1 << 4 | 1 << 7;
    a.formula_collected = vec![false; 3];
    a.formula_in_order = true;
    a.lap_hold = 0.0;
    // The fifth first: not its turn.
    hop(&mut a, &[7], yin(7, 1));
    hop(&mut a, &[], yin(7, 2));
    assert_eq!(a.formula_collected, [false, false, false], "a function out of order was marked");
    hop(&mut a, &[0], yin(0, 1));
    hop(&mut a, &[], yin(0, 2));
    assert_eq!(a.formula_collected, [true, false, false]);
    a.formula_in_order = false;
    hop(&mut a, &[7], rise(7, 1));
    hop(&mut a, &[], rise(7, 2));
    assert_eq!(a.formula_collected, [true, false, true]);
}
