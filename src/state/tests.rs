use super::*;
use crate::audio::{CTX_FRAMES, TOTAL_FEATURES};

// Exercise the production clock: audio continues even when inference has
// no full window to answer. Do not manufacture model results for CQT.
fn interval_audio(a: &mut MyApp, pitch: Option<usize>, frames: usize) {
    for _ in 0..frames {
        {
            let mut audio = a.analysis_state.lock().unwrap();
            audio.frames_seen += 1;
            audio.cqt_pitch = pitch;
            audio.gate_open = pitch.is_some();
        }
        a.sync_audio_settings();
        a.tick(0.016);
    }
}

#[test]
fn intervals_credit_audio_without_an_inference_result() {
    for ordered in [false, true] {
        let mut a = app();
        a.set_mode(AppMode::Intervals as i32);
        a.interval_in_order = ordered;
        let chord = &a.chords[a.current_chord_index];
        let steps = a.ordered_active_indices(chord);
        let k = if ordered { 0 } else { steps.len() - 1 };
        let pc = chord.get_target_indices()[steps[k].degree];
        interval_audio(&mut a, Some(pc), 12);
        assert!(a.collected_notes[k], "audio was waiting for the model");
        assert_eq!(a.note_color(k), (50, 255, 50), "text hid a credit");
        if !ordered {
            assert_ne!(a.note_color(0), (50, 255, 50), "text invented a credit");
        }
    }
}

#[test]
fn stepping_forward_while_paused_marks_nothing_as_played() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.interval_in_order = false;
    a.transition_delay = 0.0;
    let index = a.current_chord_index;
    let chord = &a.chords[index];
    let all = chord.get_target_indices();
    let pitches: Vec<_> = a.ordered_active_indices(chord).iter().map(|s| all[s.degree]).collect();
    for pc in pitches {
        interval_audio(&mut a, Some(pc), 12);
    }
    interval_audio(&mut a, None, 60);
    assert_ne!(a.current_chord_index, index, "the set did not advance");
    assert_eq!(a.pass_result(-1), MatchStatus::Exact, "the played chord lost its mark");

    a.paused = true;
    a.step_chord(1);
    a.step_chord(1);
    assert_eq!(a.pass_result(-1), MatchStatus::None, "a chord stepped over shows as played");
    assert_eq!(a.pass_result(-2), MatchStatus::None, "a chord stepped over shows as played");
    assert_eq!(a.pass_result(-3), MatchStatus::Exact, "the played chord lost its mark");
}

#[test]
fn intervals_show_the_complete_set_before_advancing_even_in_silence() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.interval_in_order = false; // this one is about free order
    a.transition_delay = 0.0;
    let index = a.current_chord_index;
    let chord = &a.chords[index];
    let all = chord.get_target_indices();
    let pitches: Vec<_> = a.ordered_active_indices(chord).iter()
        .map(|s| all[s.degree]).collect();
    for pc in pitches.into_iter().rev() {
        interval_audio(&mut a, Some(pc), 12);
    }
    assert_eq!(a.current_chord_index, index, "last green mark was erased");
    assert!(a.collected_notes.iter().all(|&done| done));
    a.paused = true;
    interval_audio(&mut a, None, 40);
    assert_eq!(a.current_chord_index, index, "pause did not hold the result");
    a.paused = false;
    interval_audio(&mut a, None, 30);
    assert_ne!(a.current_chord_index, index, "silence stopped the transition");
    assert_eq!(a.prev_status(), MatchStatus::Exact);
    assert!(a.collected_notes.iter().all(|&done| !done));
}

#[test]
fn intervals_keep_the_last_mark_when_model_results_are_arriving() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    let index = a.current_chord_index;
    let chord = &a.chords[index];
    let all = chord.get_target_indices();
    let pitches: Vec<_> = a.ordered_active_indices(chord).iter()
        .map(|s| all[s.degree]).collect();
    for pc in pitches {
        for _ in 0..12 {
            a.hears(Some(pc));
            a.check_progress_with_ai(0.016, "Noise", 0.0);
            interval_audio(&mut a, Some(pc), 1);
        }
    }
    assert_eq!(a.current_chord_index, index, "last credit vanished before drawing");
    assert!(a.collected_notes.iter().all(|&done| done));
    for step in 0..a.collected_notes.len() {
        assert_eq!(a.note_color(step), (50, 255, 50));
    }
}

#[test]
fn intervals_do_not_reuse_one_audio_frame_after_a_stall() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    let root = a.chords[a.current_chord_index].root as usize;
    interval_audio(&mut a, Some(root), 1);
    for _ in 0..30 { a.tick(0.016); }
    assert!(a.collected_notes.iter().all(|&done| !done));
    interval_audio(&mut a, Some(root), 1);
    assert!(a.collected_notes.iter().all(|&done| !done));
}

#[test]
fn intervals_can_repeat_a_muted_note_without_an_onset_prediction() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.intervals_input = "1 1 3".into();
    a.reset_logic_state();
    let root = a.chords[a.current_chord_index].root as usize;
    interval_audio(&mut a, Some(root), 40);
    assert_eq!(&a.collected_notes[..2], &[true, false], "one pluck counted twice");
    interval_audio(&mut a, None, 16);
    interval_audio(&mut a, Some(root), 12);
    assert_eq!(&a.collected_notes[..2], &[true, true], "a muted repeat stayed blocked");
}

#[test]
fn intervals_do_not_confuse_a_missing_estimate_with_muting() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    let root = a.chords[a.current_chord_index].root as usize;
    interval_audio(&mut a, Some(root), 12);
    for _ in 0..50 {
        a.cqt_pitch = None;
        a.steady_pitch = None;
        a.audio_frames += 1;
        a.tick(0.016); // the gate stays open
    }
    a.hears(Some(root));
    assert!(!a.struck_since_credit(root));
}

#[test]
fn intervals_discard_a_stale_model_answer_before_it_can_finish_a_set() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.audio_gate_open = true;
    a.last_pitches = [0.99; 12];
    a.check_progress_with_ai(0.016, "Noise", 0.0);
    for _ in 0..60 {
        a.audio_frames += 1;
        a.tick(0.016);
    }
    assert_eq!(a.collected_notes.iter().filter(|&&v| v).count(), 1);
    assert_eq!(a.interval_hold, 0.0);
}

#[test]
fn intervals_accept_a_strum_but_not_in_single_note_mode() {
    for single in [false, true] {
        let mut a = app();
        a.set_mode(AppMode::Intervals as i32);
        a.single_notes = single;
        let chord = a.chords[a.current_chord_index].clone();
        let root = chord.root as usize;
        let name = format!("{} {}", chord.root.to_string(), chord.quality.to_string());
        // A strum: every tone of the grip genuinely in the air, which is
        // what `voices` reports and what the exception now asks for.
        let sounding = chord.get_target_indices().iter()
            .fold(0u16, |mask, &pc| mask | 1 << (pc % 12));
        for _ in 0..28 {
            a.last_pitches = [0.0; 12];
            for pc in chord.get_target_indices() { a.last_pitches[pc] = 0.99; }
            a.check_progress_with_ai(0.016, &name, 0.99);
            {
                let mut audio = a.analysis_state.lock().unwrap();
                audio.voices = sounding;
            }
            interval_audio(&mut a, Some(root), 1);
        }
        let count = a.collected_notes.iter().filter(|&&done| done).count();
        assert_eq!(count, if single { 1 } else { a.collected_notes.len() });
    }
}

/// The reported fault, refused at last: the root is plucked, the model
/// recognises the chord from it and the pitch head offers the whole shape -
/// but only one string is sounding, so only one tone counts.
///
/// The old exception asked whether the chord had been named, and one
/// plucked root names it. This one asks what is in the air.
#[test]
fn a_plucked_root_does_not_credit_the_third_above_it() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.chords = vec![Chord { root: NoteName::C, quality: ChordQuality::Minor7 }];
    a.current_chord_index = 0;
    a.single_notes = false;
    a.note_threshold = 0.5;
    a.require_onset = false;
    a.reset_logic_state();
    for _ in 0..40 {
        // The model answers with the whole chord it thinks it hears.
        a.last_pitches = [0.0; 12];
        for pc in [0usize, 3, 7, 10] { a.last_pitches[pc] = 0.95; }
        a.last_pitches[0] = 1.0;
        a.check_progress_with_ai(0.016, "C m7", 0.99);
        {
            // But one string is sounding, and it is the root.
            let mut audio = a.analysis_state.lock().unwrap();
            audio.voices = 1 << 0;
        }
        interval_audio(&mut a, Some(0), 1);
    }
    let credited: Vec<usize> = (0..12).filter(|&pc| a.credited[pc].is_some()).collect();
    assert_eq!(credited, vec![0], "credited {credited:?} off one plucked root");
}

#[test]
fn intervals_shared_note_must_be_played_again_across_a_chord_change() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.interval_in_order = false;
    // "Only what was struck" on, so the carry-over rules apply at all: with
    // every guard off the mode is deliberately permissive, see
    // `relaxed_chords`.
    a.require_onset = true;
    a.chords = vec![
        Chord { root: NoteName::C, quality: ChordQuality::Major7 },
        Chord { root: NoteName::F, quality: ChordQuality::Major7 },
    ];
    a.current_chord_index = 0;
    a.reset_logic_state();
    for pc in [7, 4, 0] { interval_audio(&mut a, Some(pc), 12); }
    interval_audio(&mut a, Some(0), 40); // C rings through the transition
    assert_eq!(a.current_chord_index, 1);
    assert!(a.collected_notes.iter().all(|&done| !done));
    for pc in [5, 9, 0] { interval_audio(&mut a, Some(pc), 12); }
    assert_eq!(a.current_chord_index, 1);
    assert!(a.collected_notes.iter().all(|&done| done), "the shared C stayed blocked");
}

/// Frames in which a string is HIT, which is what the flux detector reports
/// and what `interval_audio` alone never claims.
fn interval_pluck(a: &mut MyApp, pitch: Option<usize>, frames: usize) {
    {
        let mut audio = a.analysis_state.lock().unwrap();
        audio.flux_id += 1;
    }
    interval_audio(a, pitch, frames);
}

/// The reported fault: the root is played and the minor third is offered up
/// with it, because route 2 asks only that the class be within a tenth of
/// the loudest - and a head trained on chord shapes answers the whole chord
/// when it hears one note of it.
///
/// Asked of `sounding_by` directly, which is where the willingness lives.
/// Not the attack gate: the third was never credited, so the repeat gate
/// returns true without being consulted. This is older than that change.
///
/// "One note at a time" is the measured lever: over 49 notes the four
/// routes together credited 110 things nobody played, the steady estimate
/// alone 33, and it missed nothing.
#[test]
fn the_model_offers_a_third_nobody_played_unless_notes_come_one_at_a_time() {
    for single in [false, true] {
        let mut a = app();
        a.set_mode(AppMode::Intervals as i32);
        a.chords = vec![Chord { root: NoteName::C, quality: ChordQuality::Minor7 }];
        a.current_chord_index = 0;
        a.single_notes = single;
        a.note_threshold = 0.5;
        a.require_onset = false;
        a.reset_logic_state();
        // The ear names the ROOT: that is the note being played. The model
        // lights the third nearly as loudly beside it.
        interval_audio(&mut a, Some(0), CQT_STEADY_TICKS as usize);
        a.last_pitches = [0.0; 12];
        a.last_pitches[0] = 1.0;
        a.last_pitches[3] = 0.95;
        let third = a.sounding_by(3, Some(NoteName::C), 0.99);
        let root = a.sounding_by(0, Some(NoteName::C), 0.99);
        assert_eq!(root, Some(1), "the note actually played stopped counting");
        if single {
            assert_eq!(third, None, "one note at a time still offered the third");
        } else {
            assert_eq!(
                third, Some(2),
                "route 2 no longer spreads a chord onto its third",
            );
        }
    }
}

/// The regression the pitch-blind gate risks, in the configuration the user
/// actually runs: Intervals with "one note at a time" off, where the guard
/// against the ear naming something else is switched off too.
///
/// A credited note that merely rings must not be credited again because
/// some OTHER string was hit. The per-class counter this replaced was bad
/// at classes, but it was not nothing.
#[test]
fn an_attack_on_another_string_does_not_recredit_a_ringing_note() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.single_notes = false;
    a.reset_logic_state();
    let chord = a.chords[a.current_chord_index].clone();
    let all = chord.get_target_indices();
    let steps = a.ordered_active_indices(&chord);
    let first = all[steps[0].degree];
    let other = all[steps[1].degree];
    interval_pluck(&mut a, Some(first), 12);
    assert!(a.collected_notes[0], "the first note was not credited at all");
    let before = a.collected_notes.iter().filter(|&&d| d).count();
    // Another string is hit, and the ear names THAT one. The first note is
    // only ringing on underneath.
    for _ in 0..6 {
        interval_pluck(&mut a, Some(other), 12);
    }
    let credited = a.collected_notes.iter().filter(|&&d| d).count();
    assert_eq!(
        credited - before, 1,
        "{} notes were credited by attacks on another string", credited - before,
    );
}

/// The reported fault, with the new evidence: the shared note rings on into
/// the next chord and must not be credited there until it is hit again.
///
/// The older gate let this through because `strike_id` came from a per-class
/// head that fired on the wrong class; this one asks only whether a string
/// was hit, which over a sustain it was not.
#[test]
fn a_shared_interval_ringing_into_the_next_chord_waits_to_be_hit() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.interval_in_order = false;
    // "Only what was struck" on, so the carry-over rules apply at all: with
    // every guard off the mode is deliberately permissive, see
    // `relaxed_chords`.
    a.require_onset = true;
    a.chords = vec![
        Chord { root: NoteName::C, quality: ChordQuality::Major7 },
        Chord { root: NoteName::F, quality: ChordQuality::Major7 },
    ];
    a.current_chord_index = 0;
    a.reset_logic_state();
    for pc in [7, 4, 0] { interval_pluck(&mut a, Some(pc), 12); }
    // C rings through the change: the ear goes on naming it, no new attack.
    interval_audio(&mut a, Some(0), 40);
    assert_eq!(a.current_chord_index, 1);
    assert!(a.collected_notes.iter().all(|&done| !done));
    // More ringing is still not a pluck.
    for _ in 0..4 {
        interval_audio(&mut a, Some(0), 15);
    }
    assert!(
        a.collected_notes.iter().all(|&done| !done),
        "a note that only rang on was credited in the next chord",
    );
    // Hit again, and it counts.
    for pc in [5, 9, 0] { interval_pluck(&mut a, Some(pc), 12); }
    assert!(
        a.collected_notes.iter().all(|&done| done),
        "the shared C stayed blocked after being played again",
    );
}

/// A hop of audio in which the strike detector reports `struck`.
fn strike_hop(a: &mut MyApp, struck: &[usize], ear: Option<usize>) {
    {
        let mut st = a.analysis_state.lock().unwrap();
        st.mark_strikes(struck);
        st.frames_seen += 1;
        st.cqt_pitch = ear;
        st.gate_open = ear.is_some();
    }
    a.sync_audio_settings();
}

/// The repeat rule with the strike detector running: a class credited once
/// counts again only on a strike of THAT class, and only while it is fresh.
#[test]
fn a_credited_class_needs_its_own_fresh_strike_to_count_again() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.single_notes = true;
    strike_hop(&mut a, &[4], Some(4));
    a.credit_class(4, 0);
    assert!(!a.struck_since_credit(4), "credited and counted again off one strike");
    // Ringing on, other strings struck: not a strike of this class.
    for _ in 0..10 {
        strike_hop(&mut a, &[], Some(4));
    }
    strike_hop(&mut a, &[9], Some(4));
    assert!(!a.struck_since_credit(4), "another string's strike let a ringing note count");
    // Struck again: it counts.
    strike_hop(&mut a, &[4], Some(4));
    assert!(a.struck_since_credit(4), "a real re-strike was refused");
    // ...but not for ever: a strike long gone is not "just now".
    for _ in 0..=CLASS_FRESH {
        strike_hop(&mut a, &[], Some(4));
    }
    assert!(!a.struck_since_credit(4), "a stale strike still counted");
}

/// Without the detector, the octave branch was the leak: the estimate
/// flips an octave on a decaying note and the rule took the flip for a new
/// pluck - 50 of 58 false permissions on AtoA's notes spliced with
/// themselves. A flip now needs an attack behind it.
#[test]
fn an_octave_flip_without_an_attack_is_not_a_new_pluck() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.single_notes = true;
    {
        let mut st = a.analysis_state.lock().unwrap();
        st.mark_flux();
        st.frames_seen += 1;
        st.cqt_pitch = Some(9);
        st.cqt_semitone = Some(33);
        st.gate_open = true;
    }
    a.sync_audio_settings();
    a.credit_class(9, 0);
    // The decaying note read an octave up, with no attack anywhere near.
    for _ in 0..30 {
        let mut st = a.analysis_state.lock().unwrap();
        st.flux_age = st.flux_age.saturating_add(1);
        st.frames_seen += 1;
        st.cqt_pitch = Some(9);
        st.cqt_semitone = Some(45);
        drop(st);
        a.sync_audio_settings();
    }
    assert!(!a.struck_since_credit(9), "an octave flip passed for a pluck");
    // The same octave with an attack: a note played an octave away.
    a.analysis_state.lock().unwrap().mark_flux();
    a.sync_audio_settings();
    assert!(a.struck_since_credit(9), "an octave played with an attack was refused");
}

/// The boundary through the whole judge: a note that was never credited
/// rings into a chord that wants it. In 0.5.7 that was credited 47 times of
/// 51 on AtoA's notes; it now has to be struck again.
#[test]
fn what_rings_into_the_next_chord_needs_its_own_strike() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.chords = vec![
        Chord { root: NoteName::C, quality: ChordQuality::Minor7 },
        Chord { root: NoteName::B, quality: ChordQuality::Major7 },
    ];
    a.single_notes = true;
    a.interval_in_order = true;
    a.shuffle_chords = false;
    a.reset_logic_state();
    a.play_order = vec![0, 1];
    a.play_pos = 0;
    a.current_chord_index = 0;
    // B struck - not a tone of C m7 - and left ringing past the moment.
    strike_hop(&mut a, &[11], Some(11));
    for _ in 0..(JUST_STRUCK + 5) {
        strike_hop(&mut a, &[], Some(11));
    }
    a.analysis_state.lock().unwrap().voices = 1 << 11;
    a.sync_audio_settings();
    assert!(a.credited[11].is_none(), "B was credited in a chord that does not have it");
    a.advance_chord();
    assert_eq!(a.current_chord_index, 1);
    assert!(!a.struck_since_credit(11), "the ringing B was free to answer B Maj7");
    strike_hop(&mut a, &[11], Some(11));
    assert!(a.struck_since_credit(11), "B struck again was refused");
}

/// Played early - while the finished chord is still shown - is played for
/// the next one, not left over from the last.
#[test]
fn a_note_struck_during_the_hold_counts_for_the_next_chord() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.chords = vec![
        Chord { root: NoteName::C, quality: ChordQuality::Minor7 },
        Chord { root: NoteName::B, quality: ChordQuality::Major7 },
    ];
    a.single_notes = true;
    a.reset_logic_state();
    a.play_order = vec![0, 1];
    a.play_pos = 0;
    a.current_chord_index = 0;
    strike_hop(&mut a, &[11], Some(11));
    a.analysis_state.lock().unwrap().voices = 1 << 11;
    a.sync_audio_settings();
    a.advance_chord();
    assert!(a.struck_since_credit(11), "an anticipated note was refused");
}

/// The fault the user found with shuffle on: a note credited in one chord
/// is credited again in the next without being struck, because it is still
/// ringing and happens to be the loudest thing when the new chord asks for
/// its class.
///
/// Shuffle only exposes it. It takes two consecutive chords sharing a class
/// at the boundary, which a fixed progression may never do and a random
/// order does often.
#[test]
fn a_class_still_ringing_is_not_credited_again_in_the_next_chord() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    // E m7 opens on E, which C Maj7 has just had credited.
    a.chords = vec![
        Chord { root: NoteName::C, quality: ChordQuality::Major7 },
        Chord { root: NoteName::E, quality: ChordQuality::Minor7 },
    ];
    a.current_chord_index = 0;
    a.interval_in_order = true;
    a.single_notes = false;
    a.reset_logic_state();
    // The first chord, played one note at a time. Each note goes on
    // ringing under the next, which is what `voices` reports and what a
    // guitar does.
    let mut ringing = 0u16;
    for pc in [0usize, 4, 7, 11] {
        ringing |= 1 << pc;
        {
            let mut audio = a.analysis_state.lock().unwrap();
            audio.flux_id += 1;
            audio.flux_age = 0;
            audio.voices = ringing;
        }
        interval_audio(&mut a, Some(pc), 14);
    }
    assert!(a.collected_notes.iter().all(|&d| d), "the first chord was not completed");
    // Past the hold the finished set is shown for, into the next chord.
    for _ in 0..200 {
        if a.current_chord_index == 1 {
            break;
        }
        interval_audio(&mut a, None, 1);
    }
    assert_eq!(a.current_chord_index, 1, "the first chord did not hand over");
    let before = a.collected_notes.iter().filter(|&&done| done).count();
    // Now nothing is played. The chord rings on, and E outlasts the rest
    // and becomes the loudest thing - so the ear names it, and the new
    // chord's first step wants E.
    {
        let mut audio = a.analysis_state.lock().unwrap();
        audio.voices = ringing;
        audio.flux_age = u32::MAX;
    }
    interval_audio(&mut a, Some(4), 60);
    let after = a.collected_notes.iter().filter(|&&done| done).count();
    assert_eq!(
        after, before,
        "a ringing E was credited in the next chord without being struck",
    );
}

/// With all three guards off the mode is deliberately permissive: the
/// model decides and nothing argues with it, carry-over included.
///
/// This is a choice, not an oversight. The rules exist to make a test
/// honest, and someone who has turned off single notes, "only what was
/// struck" and the fixed order is playing chords for fun. The strict path
/// is the default and the measured one.
#[test]
fn with_every_guard_off_the_model_decides_and_carry_over_is_allowed() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.chords = vec![
        Chord { root: NoteName::C, quality: ChordQuality::Major7 },
        Chord { root: NoteName::E, quality: ChordQuality::Minor7 },
    ];
    a.current_chord_index = 0;
    a.single_notes = false;
    a.require_onset = false;
    a.interval_in_order = false;
    a.reset_logic_state();
    assert!(a.relaxed_chords(), "the relaxed path was not entered");
    // The first chord, played and credited.
    let mut ringing = 0u16;
    for pc in [0usize, 4, 7, 11] {
        ringing |= 1 << pc;
        {
            let mut audio = a.analysis_state.lock().unwrap();
            audio.flux_id += 1;
            audio.flux_age = 0;
            audio.voices = ringing;
        }
        interval_audio(&mut a, Some(pc), 14);
    }
    for _ in 0..200 {
        if a.current_chord_index == 1 {
            break;
        }
        interval_audio(&mut a, None, 1);
    }
    assert_eq!(a.current_chord_index, 1, "the first chord did not hand over");
    // E rings on into the next chord, which opens on E. Strictly this is
    // the carry-over the other tests refuse; here it counts.
    {
        let mut audio = a.analysis_state.lock().unwrap();
        audio.voices = ringing;
        audio.flux_age = u32::MAX;
    }
    let before = a.collected_notes.iter().filter(|&&d| d).count();
    interval_audio(&mut a, Some(4), 60);
    assert!(
        a.collected_notes.iter().filter(|&&d| d).count() > before,
        "the relaxed path refused a ringing note anyway",
    );
}

/// A strum where the witness only sees part of the grip, which is the
/// realistic case: an octave in the shape cannot be resolved at all and a
/// quiet fifth is missed, so `voices` comes back short.
///
/// Counting them rather than looking for the target is what makes this
/// work - two voices is already more than one string can be.
#[test]
fn a_strum_counts_even_when_the_witness_sees_only_part_of_it() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.chords = vec![Chord { root: NoteName::C, quality: ChordQuality::Major7 }];
    a.current_chord_index = 0;
    a.single_notes = false;
    a.note_threshold = 0.5;
    a.reset_logic_state();
    let tones = a.chords[0].get_target_indices();
    let mut best = 0;
    for _ in 0..120 {
        a.last_pitches = [0.0; 12];
        for &pc in &tones {
            a.last_pitches[pc] = 0.95;
        }
        a.check_progress_with_ai(0.016, "Noise", 0.0);
        {
            // Two of the grip's tones, not all of them.
            let mut audio = a.analysis_state.lock().unwrap();
            audio.voices = (1 << tones[0]) | (1 << tones[1]);
        }
        // The ear names the root throughout, as it does over a strum.
        interval_audio(&mut a, Some(tones[0]), 1);
        best = best.max(a.collected_notes.iter().filter(|&&d| d).count());
    }
    assert_eq!(
        best, a.collected_notes.len(),
        "a strum credited {best} of {} tones", a.collected_notes.len(),
    );
}

#[test]
fn intervals_do_not_transfer_partial_confirmation_between_notes() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    let chord = &a.chords[a.current_chord_index];
    let all = chord.get_target_indices();
    let steps = a.ordered_active_indices(chord);
    let pcs = [all[steps[0].degree], all[steps[1].degree]];
    for _ in 0..4 {
        for pc in pcs { interval_audio(&mut a, Some(pc), 4); }
    }
    assert!(a.collected_notes.iter().all(|&done| !done));
    interval_audio(&mut a, Some(pcs[0]), 12);
    assert!(a.collected_notes[0]);
    assert!(!a.collected_notes[1]);
}

/// One note played and heard: the single-frame estimate holding still on it,
/// which is the only way into a formula's marks.
pub(crate) fn ear_hears(a: &mut MyApp, pc: usize) {
    ready(a);
    for _ in 0..CQT_STEADY_TICKS {
        ear_frame(a, Some(pc % 12));
    }
}

/// One reading, as one new audio frame carrying it.
pub(crate) fn ear_frame(a: &mut MyApp, pc: Option<usize>) {
    a.cqt_pitch = pc;
    a.audio_frames += 1;
    a.tick(0.016);
}

/// Past the hold a new lap begins with - which real playing walks through
/// without noticing, and a test would otherwise have to wait out.
pub(crate) fn ready(a: &mut MyApp) {
    a.lap_hold = 0.0;
}

/// The ear letting go: answers arriving with nothing heard in them.
pub(crate) fn ear_silent(a: &mut MyApp) {
    for _ in 0..EAR_WINDOW {
        ear_frame(a, None);
    }
}

/// Frames passing with nothing new from the model - which is what happens
/// the moment the player stops playing.
pub(crate) fn frames_pass(a: &mut MyApp, n: usize, dt: f32) {
    for _ in 0..n {
        a.tick(dt);
    }
}

/// MyApp needs the shared audio state; nothing here touches it.
/// A credit as the tests write one: the counters, nothing heard. `flux` is
/// left at zero and these tests never move `flux_id`, so they go on
/// exercising the older evidence - which is the fallback path and wants
/// testing too.
fn credit(onset: u64) -> Credit {
    Credit { onset, flux: 0, class_strike: 0, octave: 0, semitone: None, settle: 0.0, left: false }
}

pub(crate) fn app() -> MyApp {
    let analysis = Arc::new(Mutex::new(AudioAnalysis {
        cqt_semitone: None,
        input_history: [[0.0; TOTAL_FEATURES]; CTX_FRAMES],
        frame_live: [false; CTX_FRAMES],
        spectrum_visual: [0.0; 48],
        chroma_sum: [0.0; 12],
        bass_boost_enabled: false,
        bass_boost_gain: 1.0,
        noise_gate: 0.0,
        input_level: 0.0,
        onset_id: 0,
        flux_id: 0,
        flux_age: u32::MAX,
        hops: 0,
        voices: 0,
        class_strikes: [0; 12],
        class_strike_age: [u32::MAX; 12],
        strikes_live: false,
        frames_since_onset: 0,
        cqt_pitch: None,
        gate_open: false,
        frames_seen: 0,
    }));
    let mut a = MyApp::new(analysis);
    a.chords = vec![
        Chord { root: NoteName::C, quality: ChordQuality::Major7 },
        Chord { root: NoteName::D, quality: ChordQuality::Minor7 },
        Chord { root: NoteName::G, quality: ChordQuality::Dominant7 },
        Chord { root: NoteName::A, quality: ChordQuality::Minor7 },
    ];
    a.reset_logic_state();
    a
}

/// A standard is worked on for weeks and read from four modes. Losing it
/// on the way to Scales and back was the app forgetting what the session
/// was about.
#[test]
fn the_tune_follows_from_mode_to_mode() {
    let mut a = app();
    if a.song_library.len() < 2 {
        return;                       // a build with one tune has nothing to lose
    }
    a.set_mode(AppMode::Chords as i32);
    a.item_selected(1);
    let title = a.song_title.clone();
    a.set_mode(AppMode::Scales as i32);
    a.set_mode(AppMode::Intervals as i32);
    assert_eq!(a.selected_library_idx, 1, "the tune snapped back to the first row");
    assert_eq!(a.song_title, title);
    // Arpeggios over the changes reads the same library; the studies do not,
    // and their own row is remembered separately.
    a.arp_exercise = 1;
    a.set_mode(AppMode::Arpeggios as i32);
    assert_eq!(a.song_title, title, "over the changes it is the same tune");
}

/// One scale is worked on for days, and Scales opened on the first row of
/// its library however long that had been going on.
#[test]
fn the_scale_is_kept_too() {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    assert!(a.scale_definitions.len() > 2, "a library to choose from");
    a.item_selected(2);
    let name = a.song_title.clone();
    a.set_mode(AppMode::Chords as i32);
    a.set_mode(AppMode::Scales as i32);
    assert_eq!(a.selected_library_idx, 2, "back on the first row");
    assert_eq!(a.song_title, name);
    // And the name is what the settings hand back, not the row.
    let mut b = app();
    assert!(!b.select_scale("No Such Scale"));
    assert!(b.select_scale(&name));
    b.set_mode(AppMode::Scales as i32);
    assert_eq!(b.song_title, name);
}

/// What the settings hand back is a NAME, and the library is read from disk.
#[test]
fn a_tune_that_is_gone_leaves_the_choice_alone() {
    let mut a = app();
    a.set_mode(AppMode::Chords as i32);
    let before = a.song_title.clone();
    assert!(!a.select_song("Nothing Of The Sort"), "it cannot have been found");
    assert_eq!(a.song_title, before);
    assert_eq!(a.selected_library_idx, 0);
    let known = a.song_library.last().expect("a library").title.clone();
    assert!(a.select_song(&known));
    assert_eq!(a.song_title, known);
}

#[test]
fn sequential_order_without_randomisation() {
    let mut a = app();
    a.set_random_mode(false);
    let seen: Vec<usize> = (0..4).map(|_| { a.advance_chord(); a.current_chord_index }).collect();
    assert_eq!(seen, vec![1, 2, 3, 0]);
}

/// The same chord twice in a row reads as the app having missed the first one.
#[test]
fn randomised_order_never_repeats_immediately() {
    let mut a = app();
    a.set_random_mode(true);
    let mut prev = a.current_chord_index;
    for _ in 0..300 {
        a.advance_chord();
        assert_ne!(a.current_chord_index, prev, "the same chord came up twice in a row");
        prev = a.current_chord_index;
    }
}

/// A single-chord list has no alternative - it must not loop forever or panic.
#[test]
fn randomisation_survives_a_one_chord_list() {
    let mut a = app();
    a.chords.truncate(1);
    a.reset_logic_state();
    a.set_random_mode(true);
    a.advance_chord();
    assert_eq!(a.current_chord_index, 0);
}

#[test]
fn order_is_untouched_when_randomisation_is_off() {
    let mut a = app();
    a.set_random_mode(false);
    let chord = a.chords[a.current_chord_index].clone();
    assert_eq!(a.ordered_active_indices(&chord), a.get_active_indices(&chord));
}

/// Shuffling must not drop or duplicate a tone - every step still gets played.
#[test]
fn shuffled_steps_are_a_permutation() {
    let mut a = app();
    a.intervals_input = "1 3 5 7".to_string();
    a.set_random_mode(true);
    for _ in 0..50 {
        let chord = a.chords[a.current_chord_index].clone();
        let plain = a.get_active_indices(&chord);
        let mut ordered = a.ordered_active_indices(&chord);
        let mut expect = plain.clone();
        ordered.sort();
        expect.sort();
        assert_eq!(ordered, expect, "the shuffle changed the set of tones");
        a.advance_chord();
    }
}

/// Editing the interval list leaves a stale permutation behind; the length
/// check must fall back to the plain order instead of indexing out of bounds.
#[test]
fn stale_permutation_falls_back_instead_of_panicking() {
    let mut a = app();
    a.intervals_input = "1 3 5 7".to_string();
    a.set_random_mode(true);
    a.intervals_input = "1 3".to_string();      // shorter, no reroll yet
    let chord = a.chords[a.current_chord_index].clone();
    assert_eq!(a.ordered_active_indices(&chord), a.get_active_indices(&chord));
}

/// Builds a Scales-mode app the way reload_library_content does.
fn scales_app() -> MyApp {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    a
}

/// A scale is one "chord", so finishing it must move the KEY, not the index.
#[test]
fn scales_draw_a_new_key_on_every_pass() {
    let mut a = scales_app();
    assert!(!a.chords.is_empty(), "no scale loaded - the test would pass vacuously");
    a.set_random_mode(true);
    let mut keys = std::collections::HashSet::new();
    let mut prev = a.chords[0].root as usize;
    for _ in 0..200 {
        a.advance_chord();
        let now = a.chords[0].root as usize;
        assert_ne!(now, prev, "the same key came up twice in a row");
        keys.insert(now);
        prev = now;
    }
    assert!(keys.len() >= 10, "keys barely varied: {} of 12", keys.len());
}

/// The combo has to follow, or it would name a key the app is not asking for.
#[test]
fn key_combo_index_tracks_the_drawn_key() {
    let mut a = scales_app();
    assert!(!a.chords.is_empty(), "no scale loaded - the test would pass vacuously");
    a.set_random_mode(true);
    for _ in 0..50 {
        a.advance_chord();
        assert_eq!(a.secondary_index, a.chords[0].root as usize);
    }
}

#[test]
fn scales_keep_their_key_without_randomisation() {
    let mut a = scales_app();
    assert!(!a.chords.is_empty(), "no scale loaded - the test would pass vacuously");
    a.set_random_mode(false);
    let key = a.chords[0].root as usize;
    for _ in 0..20 { a.advance_chord(); }
    assert_eq!(a.chords[0].root as usize, key, "the key moved with randomisation off");
}

/// Over the changes the second combo is not a key and must not be moved:
/// the progression already says which chord comes next.
#[test]
fn the_changes_exercise_keeps_its_second_combo() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = 1;
    a.set_random_mode(true);
    let sec = a.secondary_index;
    for _ in 0..20 { a.advance_chord(); }
    assert_eq!(a.secondary_index, sec, "something moved the combo behind our back");
}

/// A study has one chord and no progression, so a finished pass means a new
/// key - the same as a finished scale, and the same switch decides it.
#[test]
fn a_study_draws_a_new_key_when_a_pass_ends() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = 0;
    a.reload_library();
    assert!(!a.chords.is_empty(), "the study has no chord to stand on");
    a.set_random_mode(true);
    let key = a.chords[0].root as usize;
    let mut moved = false;
    for _ in 0..20 {
        a.advance_chord();
        if a.chords[0].root as usize != key {
            moved = true;
            break;
        }
    }
    assert!(moved, "the key never moved between passes");

    a.set_random_mode(false);
    let held = a.chords[0].root as usize;
    for _ in 0..20 { a.advance_chord(); }
    assert_eq!(a.chords[0].root as usize, held, "the key moved with randomisation off");
}

/// Pause must stop progression WITHOUT stopping the feedback - the whole
/// point is to sit on one chord and keep seeing whether it is right.
#[test]
fn pause_holds_the_chord_but_keeps_scoring() {
    let mut a = app();
    // Straight to the field: set_mode would reload the library and replace
    // the test chords. The default mode is Fretboard, which never reaches
    // the chord branch at all.
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.paused = true;
    a.transition_delay = 0.05;
    let start = a.current_chord_index;
    for _ in 0..40 {
        a.check_progress_with_ai(0.1, "C Maj7", 0.99);
    }
    assert_eq!(a.current_chord_index, start, "paused, yet it moved on");
    assert_eq!(a.match_status, MatchStatus::Exact, "paused stopped the colours too");
}

/// A chord struck and released is right or wrong within a few frames of the
/// attack. The hold timer cannot see that: the decay stops feeding it and it
/// drains, so a correct chick never reaches the threshold.
#[test]
fn short_verdict_passes_a_chord_that_stops_ringing() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = true;
    a.transition_delay = 0.25;          // far longer than the strum lasts
    let start = a.current_chord_index;
    // One frame of a clean reading, then the sound dies away.
    a.check_progress_with_ai(0.02, "C Maj7", 0.99);
    assert_ne!(a.current_chord_index, start, "a clean strum did not count");
}

/// Nothing runs away when green advances at once: moving on changes the
/// target, and the chord still ringing no longer matches it. So a chord held
/// down does not walk through the whole song.
#[test]
fn a_ringing_chord_does_not_walk_through_the_song() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = true;
    a.transition_delay = 0.25;
    let start = a.current_chord_index;
    for _ in 0..40 {
        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
    }
    assert_eq!(a.current_chord_index, (start + 1) % a.chords.len(),
               "one chord ringing moved the song on more than once");
}

/// Two of the same chord in a row: the first pass is earned, the second has
/// to be strummed again. The ring of the first would otherwise match the
/// target the instant it is set.
#[test]
fn the_same_chord_twice_has_to_be_played_twice() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = true;
    a.transition_delay = 0.25;
    a.chords = vec![
        Chord { root: NoteName::C, quality: ChordQuality::Major7 },
        Chord { root: NoteName::C, quality: ChordQuality::Major7 },
    ];
    a.reset_logic_state();

    a.check_progress_with_ai(0.02, "C Maj7", 0.99);
    assert_eq!(a.current_chord_index, 1, "the first strum did not pass");

    // The same chord keeps ringing, unplayed.
    for _ in 0..40 {
        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
    }
    assert_eq!(a.current_chord_index, 1, "the ring passed the repeat by itself");

    // Struck again: without the strike detector, the envelope says so.
    a.analysis_state.lock().unwrap().mark_onset();
    a.sync_audio_settings();
    a.check_progress_with_ai(0.02, "C Maj7", 0.99);
    assert_ne!(a.current_chord_index, 1, "a real second strum was refused");
}

/// The strip of the progression: what was played, what is asked, what is
/// to come - in the order of this pass, never wrapping into the next one.
#[test]
fn the_progress_window_follows_the_order_of_play() {
    let mut a = app();
    a.play_order = vec![3, 1, 4, 0, 2];
    a.play_pos = 0;
    assert_eq!(a.progress_window(2, 2), vec![(3, 0), (1, 1), (4, 2)]);
    a.play_pos = 3;
    assert_eq!(a.progress_window(2, 4), vec![(1, -2), (4, -1), (0, 0), (2, 1)]);
    a.play_order.clear();
    assert!(a.progress_window(2, 4).is_empty());
}

/// The detector's strikes, as the audio thread reports them, one hop of
/// playing - the noise gate open, or the sync clears the model's answer.
fn detector_hop(a: &mut MyApp, struck: &[usize]) {
    {
        let mut st = a.analysis_state.lock().unwrap();
        st.gate_open = true;
        st.mark_strikes(struck);
    }
    a.sync_audio_settings();
}

/// With the strike detector running, a repeated chord waits for its strike,
/// however long the chord rings.
#[test]
fn a_ringing_chord_is_not_a_strum_to_the_strike_detector() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = true;
    a.transition_delay = 0.25;
    a.chords = vec![
        Chord { root: NoteName::C, quality: ChordQuality::Major7 },
        Chord { root: NoteName::C, quality: ChordQuality::Major7 },
    ];
    detector_hop(&mut a, &[]);
    a.reset_logic_state();
    a.check_progress_with_ai(0.02, "C Maj7", 0.99);
    assert_eq!(a.current_chord_index, 1, "the first strum did not pass");

    // Ringing: the detector stays quiet.
    for _ in 0..20 {
        detector_hop(&mut a, &[]);
        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
    }
    assert_eq!(a.current_chord_index, 1, "a ringing chord passed as a strum");

    // One class reported is a strum: the detector rarely reports more.
    detector_hop(&mut a, &[4]);
    a.check_progress_with_ai(0.02, "C Maj7", 0.99);
    assert_ne!(a.current_chord_index, 1, "the detector's strike was refused");
}

/// "Credit only what was struck" asks the detector which class was struck:
/// the model's pitch head may credit that class, and only while the strike
/// is fresh.
#[test]
fn the_strike_detector_decides_what_the_model_may_credit() {
    let mut a = app();
    a.require_onset = true;
    a.single_notes = false;
    a.note_threshold = 0.5;
    a.last_pitches = [0.0; 12];
    a.last_pitches[7] = 1.0;
    a.last_pitches[4] = 0.95;
    detector_hop(&mut a, &[7]);
    assert!(a.note_is_sounding(7, None, 0.0), "the struck class was refused");
    assert!(!a.note_is_sounding(4, None, 0.0), "an unstruck class passed");
    for _ in 0..STRUCK_FRESH {
        detector_hop(&mut a, &[]);
    }
    assert!(a.note_is_sounding(7, None, 0.0), "the strike went stale too early");
    detector_hop(&mut a, &[]);
    assert!(!a.note_is_sounding(7, None, 0.0), "a stale strike still counted as just now");
}

/// Just after a strike the ear may name only what was struck: its window
/// still holds the note before. Once it has settled, it is believed again,
/// so a note the detector did not report is late, not lost.
#[test]
fn the_ear_settles_after_a_strike_of_another_class() {
    let mut a = app();
    a.single_notes = true;
    detector_hop(&mut a, &[2]);                // D struck
    a.hears(Some(5));                          // the ear reads F
    assert!(!a.note_is_sounding(5, None, 0.0), "the ear credited F right after D was struck");
    a.hears(Some(2));
    assert!(a.note_is_sounding(2, None, 0.0), "the struck D was refused");
    for _ in 0..EAR_SETTLE {
        detector_hop(&mut a, &[]);
    }
    a.hears(Some(5));
    assert!(!a.note_is_sounding(5, None, 0.0), "the ear was believed before it settled");
    detector_hop(&mut a, &[]);
    a.hears(Some(5));
    assert!(a.note_is_sounding(5, None, 0.0), "the ear was not believed once it settled");
}

/// In Formulas the ear credits only a class the detector struck within 3 s:
/// a dying note read as another one is not a mark. Elsewhere the ear is
/// believed as before.
#[test]
fn in_formulas_the_ear_needs_a_strike_behind_it() {
    for formulas in [true, false] {
        let mut a = app();
        if formulas {
            a.app_mode = AppMode::Formulas;
        }
        a.single_notes = true;
        detector_hop(&mut a, &[10]);           // Bb struck
        for _ in 0..=EAR_SETTLE {
            detector_hop(&mut a, &[]);
        }
        a.hears(Some(10));
        assert!(a.note_is_sounding(10, None, 0.0), "the struck Bb was refused");
        a.hears(Some(3));                      // the dying Bb read as Eb
        assert_eq!(a.note_is_sounding(3, None, 0.0), !formulas, "formulas = {formulas}");
        for _ in 0..EAR_BACKED {
            detector_hop(&mut a, &[]);
        }
        a.hears(Some(10));
        assert_eq!(a.note_is_sounding(10, None, 0.0), !formulas,
                   "a strike 3 s old still backed the ear, formulas = {formulas}");
    }
}

/// Without the detector there is nothing to settle after.
#[test]
fn without_the_detector_the_ear_is_believed_at_once() {
    let mut a = app();
    a.single_notes = true;
    a.hears(Some(5));
    assert!(a.note_is_sounding(5, None, 0.0));
}

/// A m7b5 has no shell of its own: the shell is root, seventh and third,
/// and the flat fifth is the one note it leaves out. With shells the only
/// shapes on screen, the m7 shell IS the shape asked for.
#[test]
fn the_shell_of_a_half_diminished_passes_as_itself() {
    for (shells_only, want) in [(true, MatchStatus::Exact), (false, MatchStatus::Partial)] {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = false;
        a.shells_only = shells_only;
        a.chords = vec![Chord { root: NoteName::A, quality: ChordQuality::HalfDiminished }];
        a.reset_logic_state();
        for _ in 0..40 { a.check_progress_with_ai(0.02, "A m7", 0.99); }
        assert_eq!(a.prev_status(), want, "shells_only = {shells_only}");
    }
}

/// Passing has to be visible, and on the right chord. Advancing changes
/// which chord is current, so the green belongs on the previous one - the
/// one that earned it. Lighting the current chord marked the one that had
/// not been played yet.
#[test]
fn a_pass_lights_the_chord_that_earned_it() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = true;
    a.transition_delay = 0.25;
    let played = a.current_chord_index;

    a.check_progress_with_ai(0.02, "C Maj7", 0.99);

    assert_eq!(a.prev_status(), MatchStatus::Exact, "the pass was not shown at all");
    assert_eq!(a.prev_chord_index, Some(played), "the wrong chord was lit");
    assert_ne!(a.current_chord_index, played, "it did not move on");
}

/// The green stays put. It marked a pass for a fifth of a second before,
/// which was too brief to catch; there is nothing to time out, because the
/// next pass moves the mark along by itself.
#[test]
fn the_green_stays_until_the_next_pass_moves_it() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = true;
    a.transition_delay = 0.25;

    let first = a.current_chord_index;
    a.check_progress_with_ai(0.02, "C Maj7", 0.99);
    assert_eq!(a.prev_chord_index, Some(first));

    // Whatever comes next - silence, a wrong chord, a long wait - the mark
    // stays on the chord that earned it.
    for _ in 0..40 {
        a.check_progress_with_ai(0.05, "F# m", 0.10);
    }
    assert_eq!(a.prev_status(), MatchStatus::Exact, "the mark disappeared on its own");
    assert_eq!(a.prev_chord_index, Some(first), "the mark wandered off");

    // Passing the chord now current moves it along, and only then.
    let second = a.current_chord_index;
    let name = a.chord_label(second);
    let (root, qual) = name.split_at(name.find(' ').unwrap_or(name.len()));
    a.check_progress_with_ai(0.02, &format!("{root}{qual}"), 0.99);
    assert_eq!(a.prev_chord_index, Some(second), "the next pass did not move the mark");
    assert_eq!(a.prev_status(), MatchStatus::Exact);
}

/// A chord passed on a triad or a substitution went yellow on the way
/// through. Reporting it green afterwards would be a kinder account than
/// what actually happened.
#[test]
fn the_mark_says_how_the_chord_was_passed() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = false;          // the hold timer, so a partial can pass
    a.transition_delay = 0.05;
    let played = a.current_chord_index;

    // A plain triad against a Maj7 target: yellow, and enough to pass.
    for _ in 0..10 {
        a.check_progress_with_ai(0.02, "C", 0.99);
    }

    assert_ne!(a.current_chord_index, played, "the triad never passed");
    assert_eq!(a.prev_chord_index, Some(played));
    assert_eq!(a.prev_status(), MatchStatus::Partial,
               "a chord passed on a triad was reported as an exact match");
}

/// The same four notes have four names. Which one comes back depends on
/// the voicing, so the app cannot insist on one of them.
#[test]
fn a_diminished_seventh_is_the_same_chord_from_all_four_roots() {
    for named in ["Eb dim", "Gb dim", "A dim"] {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = false;
        a.transition_delay = 0.05;
        a.chords = vec![Chord { root: NoteName::C, quality: ChordQuality::Diminished }];
        a.reset_logic_state();

        for _ in 0..10 {
            a.check_progress_with_ai(0.02, named, 0.99);
        }
        assert_eq!(
            a.prev_status(),
            MatchStatus::Exact,
            "{named} is C dim7 played from another of its notes"
        );
    }

    // A minor third is the interval that makes them one chord; a semitone
    // away is a different chord and stays wrong.
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = false;
    a.transition_delay = 0.05;
    a.chords = vec![Chord { root: NoteName::C, quality: ChordQuality::Diminished }];
    a.reset_logic_state();
    for _ in 0..10 {
        a.check_progress_with_ai(0.02, "Db dim", 0.99);
    }
    assert_ne!(a.prev_status(), MatchStatus::Exact, "Db dim7 is not C dim7");
}

/// A shell voicing of a m7b5 is the m7 shell, note for note - the fifth is
/// the only place the two chords differ, and a shell has no fifth. The app
/// draws that shape in shell mode, so it has to accept hearing it.
#[test]
fn a_half_diminished_passes_on_its_shell() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = false;
    a.transition_delay = 0.05;
    a.chords = vec![Chord { root: NoteName::D, quality: ChordQuality::HalfDiminished }];
    a.reset_logic_state();

    for _ in 0..10 {
        a.check_progress_with_ai(0.02, "D m7", 0.99);
    }
    assert_eq!(
        a.prev_status(),
        MatchStatus::Partial,
        "the shell the app itself draws was not accepted"
    );

    // And it is not mistaken for the chord: green is still only the chord.
    let mut b = app();
    b.app_mode = AppMode::Chords;
    b.set_random_mode(false);
    b.short_verdict = false;
    b.transition_delay = 0.05;
    b.chords = vec![Chord { root: NoteName::D, quality: ChordQuality::HalfDiminished }];
    b.reset_logic_state();
    for _ in 0..10 {
        b.check_progress_with_ai(0.02, "D m7b5", 0.99);
    }
    assert_eq!(b.prev_status(), MatchStatus::Exact);
}

/// Shuffle used to draw a chord at each step, so there was no "next" to
/// show and stepping back landed anywhere. The order is now laid out when
/// the toggle goes on, and a lap visits every chord exactly once.
#[test]
fn shuffle_lays_out_the_whole_song_up_front() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(true);
    let len = a.chords.len();

    let mut seen = vec![a.current_chord_index];
    for _ in 1..len {
        a.advance_chord();
        seen.push(a.current_chord_index);
    }
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), len, "a lap did not cover every chord exactly once");
}

/// Turning it off goes back to the written order, from the top.
#[test]
fn turning_shuffle_off_restarts_the_song() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(true);
    a.advance_chord();
    a.advance_chord();

    a.set_random_mode(false);

    assert_eq!(a.current_chord_index, 0, "it did not start the song again");
    a.advance_chord();
    assert_eq!(a.current_chord_index, 1, "it is not walking the written order");
}

/// What the strip promises is what actually arrives - in shuffled order the
/// chord after this one is not the next one in the song.
#[test]
fn the_strip_promises_the_chord_that_arrives() {
    for random in [false, true] {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(random);
        for _ in 0..(a.chords.len() + 2) {
            let promised = a.next_chord_index();
            a.advance_chord();
            assert_eq!(a.current_chord_index, promised,
                       "random={random}: the strip showed a chord that did not come");
        }
    }
}

/// And stepping back while paused returns to the chord actually played
/// before, not the one written before it in the song.
#[test]
fn stepping_back_follows_the_order_in_force() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(true);
    let first = a.current_chord_index;
    a.advance_chord();
    assert_ne!(a.current_chord_index, first);

    a.step_chord(-1);

    assert_eq!(a.current_chord_index, first, "stepping back left the shuffled order");
}

/// Stepping back to a chord that has gone by is for practising it again -
/// it must not look or count as though it had just been played.
#[test]
fn stepping_back_is_not_a_pass() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = true;
    a.check_progress_with_ai(0.02, "C Maj7", 0.99);
    let after_pass = a.current_chord_index;

    a.step_chord(-1);

    assert_ne!(a.current_chord_index, after_pass, "stepping back did nothing");
    assert_eq!(a.prev_status(), MatchStatus::None, "stepping back lit a chord as passed");
    assert_eq!(a.match_status, MatchStatus::None);
    assert_eq!(a.success_timer, 0.0, "stepping banked progress");
}

/// Stepping by hand has to draw a new string to start from, like arriving
/// by playing does. It did not, so shuffle plus pause plus the arrows left
/// the suggestion frozen on whichever string it happened to be.
#[test]
fn stepping_draws_a_new_string_hint() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(true);

    let mut seen = std::collections::HashSet::new();
    for _ in 0..12 {
        a.step_chord(1);
        seen.insert(a.start_hint);
    }
    assert!(a.start_hint.is_some(), "stepping left no suggestion at all");
    assert!(seen.len() > 1, "the suggestion never changed while stepping: {seen:?}");
}

/// And it wraps, so the strip can be walked round a short progression.
#[test]
fn stepping_wraps_both_ways() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    let len = a.chords.len();
    a.current_chord_index = 0;
    a.step_chord(-1);
    assert_eq!(a.current_chord_index, len - 1, "stepping back off the start did not wrap");
    a.step_chord(1);
    assert_eq!(a.current_chord_index, 0, "stepping forward did not wrap back");
}

/// The point of the option: it must not turn a wrong chord into a pass.
#[test]
fn short_verdict_still_rejects_the_wrong_chord() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = true;
    a.transition_delay = 0.25;
    let start = a.current_chord_index;
    for _ in 0..20 {
        a.check_progress_with_ai(0.02, "F# m", 0.99);
    }
    assert_eq!(a.current_chord_index, start, "a wrong chord was accepted");
}

/// A correct chord read too quietly to be trusted is not a verdict either -
/// that is what the confidence threshold is for.
#[test]
fn short_verdict_needs_the_confidence_threshold() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = true;
    a.chord_confidence = 0.80;
    a.transition_delay = 0.25;
    let start = a.current_chord_index;
    for _ in 0..20 {
        a.check_progress_with_ai(0.02, "C Maj7", 0.50);
    }
    assert_eq!(a.current_chord_index, start, "an unsure reading counted");
}

/// With the option off nothing changes: holding is still what advances.
#[test]
fn without_the_option_a_single_frame_is_not_enough() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.short_verdict = false;
    a.transition_delay = 0.25;
    let start = a.current_chord_index;
    a.check_progress_with_ai(0.02, "C Maj7", 0.99);
    assert_eq!(a.current_chord_index, start, "one frame advanced without the option");
}

/// The timer must not keep running while paused, or the next chord would
/// jump the instant play resumes.
#[test]
fn pause_does_not_bank_up_progress() {
    let mut a = app();
    a.app_mode = AppMode::Chords;
    a.set_random_mode(false);
    a.transition_delay = 0.05;
    a.paused = true;
    for _ in 0..100 { a.check_progress_with_ai(0.1, "C Maj7", 0.99); }
    assert!(a.success_timer <= a.transition_delay + 1e-6,
            "timer ran up to {} while paused", a.success_timer);
    let start = a.current_chord_index;
    a.paused = false;
    a.check_progress_with_ai(0.1, "C Maj7", 0.99);
    assert_ne!(a.current_chord_index, start, "did not resume after unpausing");
}

#[test]
fn pause_freezes_the_fretboard_target() {
    let mut a = app();
    a.set_mode(AppMode::Fretboard as i32);
    a.paused = true;
    a.transition_delay = 0.05;
    a.note_threshold = 0.5;
    let target = a.fret_target.unwrap();
    a.last_pitches = [0.0; 12];
    a.last_pitches[target] = 1.0;
    for _ in 0..40 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
    assert_eq!(a.fret_target, Some(target), "paused, yet it drew a new note");
}

/// The behaviour the option exists to protect: a strummed chord lights every
/// one of its tones in the pitch head at once and the interval modes walk
/// through them - something a monophonic estimate cannot do, so it must not
/// be allowed to overrule it while the option is off.
#[test]
fn a_strummed_chord_still_walks_its_intervals() {
    let mut a = app();
    a.note_threshold = 0.6;
    a.single_notes = false;              // the default
    a.last_pitches = [0.0; 12];
    a.last_pitches[0] = 0.95;            // C
    a.last_pitches[4] = 0.92;            // E
    a.last_pitches[7] = 0.90;            // G
    a.cqt_pitch = Some(0);               // the estimate can only name one
    for pc in [0usize, 4, 7] {
        assert!(a.note_is_sounding(pc, None, 0.0), "chord tone {pc} stopped passing");
    }
    a.single_notes = true;
    assert!(
        !a.note_is_sounding(4, None, 0.0),
        "with the option on, E has to be played on its own"
    );
}

/// The reported case: not the first chord of a run but every one after it.
#[test]
fn a_new_chord_does_not_inherit_the_old_ones_ringing() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.last_pitches = [0.9; 12];
    a.prev_pitches = [0.8; 12];
    a.credited[3] = Some(credit(7));

    a.advance_chord();

    assert_eq!(a.last_pitches, [0.0; 12], "the previous chord came along");
    // The credit is the exception: it outlives the chord on purpose, so
    // that the same note asked for on both sides of the boundary needs the
    // string struck again. It can only refuse, never credit.
    assert_eq!(a.credited[3], Some(credit(7)), "the credit was dropped at the boundary");
}

/// Entering a mode is a fresh start for the ear as well as for the exercise.
#[test]
fn a_mode_starts_without_what_was_heard_before_it() {
    let mut a = app();
    a.last_pitches = [0.9; 12];
    a.prev_pitches = [0.8; 12];
    a.credited[3] = Some(credit(7));
    a.cqt_pitch = Some(3);

    a.set_mode(AppMode::Intervals as i32);

    assert_eq!(a.last_pitches, [0.0; 12], "the model's last answer came along");
    assert_eq!(a.prev_pitches, [0.0; 12]);
    assert_eq!(a.credited[3], None, "a note credited in another mode still counted");
    // And nothing passes on it: the pitch head has nothing to say until the
    // next audio frame arrives.
    a.cqt_pitch = None;
    assert!(!a.note_is_sounding(3, None, 0.0));
}

/// The note before, still ringing inside the model's window, is what the
/// option is aimed at: the model reports it as sounding and it is, but it
/// was not struck.
#[test]
fn what_was_struck_can_be_asked_for() {
    let mut a = app();
    a.note_threshold = 0.6;
    a.single_notes = false;
    a.last_pitches = [0.0; 12];
    a.last_pitches[0] = 0.95;            // C, being played
    a.last_pitches[10] = 0.93;           // Bb, the note before, still ringing
    a.cqt_pitch = Some(0);

    // Without the option both pass, which is the complaint.
    assert!(a.note_is_sounding(0, None, 0.0));
    assert!(a.note_is_sounding(10, None, 0.0));

    a.require_onset = true;
    detector_hop(&mut a, &[0]);          // only C was struck
    a.cqt_pitch = Some(0);
    assert!(a.note_is_sounding(0, None, 0.0), "the struck note stopped passing");
    assert!(!a.note_is_sounding(10, None, 0.0), "the ringing note still counted");
    // With the CQT silent the model's own branch answers for C - this gate
    // deliberately leaves the estimate alone, so it is checked without it.
    a.cqt_pitch = None;
    assert!(a.note_is_sounding(0, None, 0.0), "the model lost its own branch");
}

/// Dwa różne losowania pod jednym przyciskiem: kolejność akordów i kolejność
/// dźwięków w akordzie. Rozdzielone tylko tam, gdzie oba istnieją.
#[test]
fn the_shuffle_reaches_the_chord_order_only_where_asked() {
    let mut a = app();

    // Chords: nothing else for the switch to mean, so the option is ignored.
    a.set_mode(AppMode::Chords as i32);
    a.set_random_mode(true);
    assert!(a.shuffles_chord_order(), "shuffle stopped working in Chords");
    a.set_shuffle_chords(true);
    assert!(a.shuffles_chord_order());

    // Intervals: the progression is a tune, so it stays put unless asked.
    a.set_mode(AppMode::Intervals as i32);
    a.set_shuffle_chords(false);
    assert!(
        !a.shuffles_chord_order(),
        "the progression was reordered without being asked"
    );
    a.set_shuffle_chords(true);
    assert!(a.shuffles_chord_order(), "the option did not reach the chord order");

    // And nothing is shuffled at all with the switch off.
    a.set_random_mode(false);
    assert!(!a.shuffles_chord_order());
}

/// Zachowanie, o które chodzi: interwały lecą losowo, progresja zostaje
/// zapisana - z tego robi się melodia.
#[test]
fn shuffled_tones_walk_the_written_progression() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.set_random_mode(true);
    a.set_shuffle_chords(false);
    let n = a.chords.len();
    assert!(n > 1, "test potrzebuje progresji dłuższej niż jeden akord");
    assert_eq!(
        a.play_order,
        (0..n).collect::<Vec<_>>(),
        "kolejność akordów miała zostać taka, jak zapisana"
    );
}

/// One CQT frame carries no memory, so it answers "what is sounding NOW"
/// where the model answers "what sounded in the last 0.77 s".
#[test]
fn the_cqt_alone_credits_the_note_being_played() {
    let mut a = app();
    a.note_threshold = 0.6;
    a.last_pitches = [0.0; 12];          // model still says nothing
    a.cqt_pitch = Some(4);               // E is sounding
    assert!(a.note_is_sounding(4, None, 0.0), "the CQT knew and was ignored");
    assert!(!a.note_is_sounding(7, None, 0.0), "credited a note nobody played");
}

/// The failure this whole path exists for: the window still holds the note
/// before the one being played, and it is the LOUDER of the two.
#[test]
fn a_note_that_has_been_left_behind_stops_counting() {
    let mut a = app();
    a.single_notes = true;
    a.note_threshold = 0.6;
    a.last_pitches = [0.0; 12];
    a.last_pitches[2] = 0.95;            // D, played a moment ago, still loud
    a.last_pitches[4] = 0.70;            // E, the one under the fingers now
    a.hears(Some(4));
    assert!(!a.note_is_sounding(2, None, 0.0), "credited the previous note");
    assert!(a.note_is_sounding(4, None, 0.0), "did not credit the current one");
}

/// Without the CQT to arbitrate, the new note is the one that ROSE - being
/// loudest belongs to whichever has had more of the window to itself.
#[test]
fn a_new_note_under_a_louder_old_one_counts() {
    let mut a = app();
    a.note_threshold = 0.6;
    a.cqt_pitch = None;                  // gate open, estimate too weak to rule
    a.prev_pitches = [0.05; 12];
    a.prev_pitches[2] = 0.90;            // D was already there
    a.last_pitches = [0.0; 12];
    a.last_pitches[2] = 0.95;            // and barely moved
    a.last_pitches[4] = 0.70;            // E arrived
    assert!(a.note_is_sounding(4, None, 0.0), "the note that arrived did not count");
}

/// Nothing refreshes the pitch vector while the gate is shut, so it has to be
/// cleared - otherwise a note that stopped sounding still reads as played.
#[test]
fn the_gate_closing_clears_the_pitch_memory() {
    let mut a = app();
    a.last_pitches = [0.5; 12];
    a.prev_pitches = [0.5; 12];
    a.analysis_state.lock().unwrap().gate_open = false;
    a.sync_audio_settings();
    assert_eq!(a.last_pitches, [0.0; 12], "stale pitches survived the gate closing");
    assert_eq!(a.prev_pitches, [0.0; 12]);
}

/// Every study is written out exactly as it is played in its source.
///
/// The phrases below were read out of the Guitar Pro files note by note and
/// turned into degrees against each one's own chord; this test is what
/// keeps the table honest. The reported case was a study picked by name and
/// drawn short: the minor one runs two octaves AND a third, so its last
/// three notes sit on the top string - 5 8 5 in A - and a table entry that
/// turned round at the seventh had none there at all.
#[test]
fn a_book_study_is_written_out_as_it_is_played() {
    let studies = [
        (
            "Minor (Two Octaves and a Third)",
            "1 3 5 7 1' 3' 5' 7' 1'' 3'' 1'' 7' 5' 3' 1' 7 5 3 1",
        ),
        (
            "Major (Leading Tone)",
            "1 3 5 7 1' 3' 5' 7' 1'' 7' 5' 3' 1' 7 5 3 1 7, 1",
        ),
        (
            "Dominant (Approach from Below)",
            "1 3 5 7 1' 3' 5' 7' 5' 3' 1' 7 5 3 1 7, 5, 7, 1",
        ),
        (
            "Skipping Notes (Fifths and Fourths)",
            "1 5 3 7 5 1' 7 3' 1' 5' 3' 7' 5' 1'' 7' 3'' 1'' 5' 7' 3' 5' 1' 3' 7 1' 5 7 3 5 1",
        ),
        (
            "Triplets (Up-Down)",
            "1 3 5 3 5 7 5 7 1' 7 1' 3' 1' 3' 5' 3' 5' 3' 5' 7' 5' 7' 1'' 3'' 1'' 7' 5' 7' \
             5' 3' 5' 3' 1' 3' 1' 7 1' 7 5 7 5 7 5 3 5 3 1",
        ),
    ];
    let patterns = crate::model::load_arpeggio_patterns();
    for (name, written) in studies {
        let want: Vec<&str> = written.split_whitespace().collect();
        let found = patterns
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} is not in the list at all"));
        assert_eq!(
            found.names, want,
            "{name} is not what is played in the source"
        );
    }
}

/// And the minor study reaches the top string, which is the whole of what
/// tells it from a two-octave run: its last climb is the third above the
/// second root.
#[test]
fn the_minor_study_reaches_the_top_string() {
    let patterns = crate::model::load_arpeggio_patterns();
    let study = patterns
        .iter()
        .find(|p| p.name == "Minor (Two Octaves and a Third)")
        .expect("the minor study is missing");
    let steps = crate::model::steps_of(&study.names);
    let names: Vec<String> = ["1", "3", "5", "7"].iter().map(|s| s.to_string()).collect();
    let spots = crate::tab::place(9, &[0, 3, 7, 10], &steps, &names);   // A m7
    let top: Vec<i32> = spots.iter().filter(|s| s.string == 5).map(|s| s.fret).collect();
    assert_eq!(top, vec![5, 8, 5], "the top string does not read 5 8 5");
}

/// A scale is drawn with the degrees IT writes, not with the nearest
/// interval name: the altered scale spells its second `#2`, and reading
/// `♭3` on the neck would be a different degree of the same pitch.
#[test]
fn the_neck_spells_a_scale_the_way_the_scale_does() {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    let altered = a
        .scale_definitions
        .iter()
        .position(|d| d.name.starts_with("Altered"))
        .expect("the altered scale is missing");
    a.item_selected(altered as i32);
    a.scale_start = 0;
    let chord = a.chords[0].clone();
    let steps = a.get_active_indices(&chord);
    let spots = crate::tab::place(
        chord.root as usize,
        &chord.quality.intervals(),
        &steps,
        &chord.quality.interval_names(),
    );
    let labels: Vec<String> = spots.iter().map(|s| s.label()).collect();
    assert!(labels.contains(&"♯2".to_string()), "the neck reads {labels:?}");
    assert!(!labels.contains(&"♭3".to_string()), "a degree was respelled: {labels:?}");
}

/// What the shuffle does in Scales: it draws the key and where on the neck
/// to take it, and leaves the degrees in order. A scale not walked in
/// order is not a scale.
#[test]
fn the_shuffle_moves_a_scale_without_scattering_it() {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    let chord = a.chords[0].clone();
    let written = a.get_active_indices(&chord);
    assert!(written.len() >= 5, "the scale is too short to tell");

    a.set_random_mode(true);
    // No permutation: whatever the pass drew, the strip is the written run
    // - the same list `get_active_indices` gives, forwards or backwards.
    assert_eq!(
        a.ordered_active_indices(&chord),
        a.get_active_indices(&chord),
        "the shuffle dealt the scale out of order"
    );
    let _ = &written;

    // The key moves between passes, and so does the string it starts from.
    let key = a.chords[0].root as usize;
    let mut keys = false;
    let mut strings = std::collections::HashSet::new();
    for _ in 0..6 {
        a.advance_chord();
        keys |= a.chords[0].root as usize != key;
        strings.insert(a.scale_start);
    }
    assert!(keys, "the key never moved");
    assert_eq!(strings.len(), 3, "the scale did not go through the strings");
}

/// Up and down in turn, and through the strings with them: what is wanted
/// is coverage, and a fair coin does not give it - twelve drawn passes came
/// out with five descending in a row.
#[test]
fn a_scale_runs_both_ways() {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    let chord = a.chords[0].clone();
    let mut ways = Vec::new();
    let mut pairs = std::collections::HashSet::new();
    for _ in 0..6 {
        a.advance_chord();
        let steps = a.get_active_indices(&chord);
        ways.push(steps.first().unwrap().degree < steps.last().unwrap().degree);
        pairs.insert((a.scale_start, a.scale_descending));
    }
    assert!(ways.windows(2).all(|w| w[0] != w[1]), "two passes ran the same way");
    assert_eq!(pairs.len(), 6, "six passes did not cover the six ways to take it");

    // Whichever way, it is the written run and not a shuffled one.
    a.scale_descending = true;
    let down = a.get_active_indices(&chord);
    a.scale_descending = false;
    let up = a.get_active_indices(&chord);
    let mut reversed = down.clone();
    reversed.reverse();
    assert_eq!(reversed, up, "the descent is not the climb turned round");
}

/// And without the shuffle the string still moves, while the key stays: the
/// shape is the same everywhere, so where it is taken is the exercise.
#[test]
fn a_scale_starts_from_a_drawn_string_even_unshuffled() {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    a.set_random_mode(false);
    let key = a.chords[0].root as usize;
    let mut strings = std::collections::HashSet::new();
    for _ in 0..6 {
        a.advance_chord();
        strings.insert(a.scale_start);
        assert_eq!(a.chords[0].root as usize, key, "the key moved unasked");
    }
    assert_eq!(strings.len(), 3, "the scale did not go through the strings");
}

/// What twenty passes of a scale actually draw.
/// `cargo test --release -- --ignored --nocapture the_scale_passes`
#[test]
#[ignore]
fn the_scale_passes() {
    for shuffled in [false, true] {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        a.set_random_mode(shuffled);
        println!("--- losowanie {}", if shuffled { "wlaczone" } else { "wylaczone" });
        for _ in 0..12 {
            a.advance_chord();
            let chord = a.chords[0].clone();
            let steps = a.get_active_indices(&chord);
            println!(
                "  tonacja {:<3} struna {} kierunek {:<5} pierwszy stopien {}",
                chord.root.to_string(),
                a.scale_start,
                if a.scale_descending { "dol" } else { "gora" },
                steps.first().map(|s| s.degree).unwrap_or(0),
            );
        }
    }
}

/// Renders a scale on the neck to stdout, for looking at it. Ignored in
/// normal runs; `cargo test -- --ignored --nocapture the_scale_picture`.
/// Where each step of a study lands, for reading a screenshot back.
/// `cargo test --release -- --ignored --nocapture the_study_places`
#[test]
#[ignore]
fn the_study_places() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = 0;
    a.item_selected(0);
    a.secondary_item_selected(0);             // C
    let chord = a.chords[0].clone();
    let steps = a.get_active_indices(&chord);
    let spots = crate::tab::place_near(
        chord.root as usize,
        &chord.quality.intervals(),
        &steps,
        &chord.quality.interval_names(),
        None,
    );
    for (i, spot) in spots.iter().enumerate() {
        println!("krok {i:2}: struna {} próg {:2}  {}", spot.string, spot.fret, spot.label());
    }
}

#[test]
#[ignore]
fn the_scale_picture() {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    a.set_random_mode(true);
    let chord = a.chords[0].clone();
    let steps = a.ordered_active_indices(&chord);
    let spots = crate::tab::place_near(
        chord.root as usize,
        &chord.quality.intervals(),
        &steps,
        &chord.quality.interval_names(),
        Some(a.voicing_anchor),
    );
    let done: Vec<bool> = (0..spots.len()).map(|i| i < 2).collect();
    println!("{}", crate::tab::neck(&spots, &done, 2, false));
}

/// The reported stall, with the attack head silent throughout: the study
/// asks for the flat seventh at step 7, three other notes are played, and
/// step 11 asks for it again. Without something to say the note had gone
/// quiet in between, the phrase could not get past that step.
#[test]
fn a_phrase_gets_past_a_note_it_has_already_used() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = 0;
    a.item_selected(0);
    a.note_threshold = 0.5;
    let seventh = 10usize;                    // over the default m7

    a.hears(Some(seventh));
    a.credit_class(seventh, 0);
    assert!(!a.struck_since_credit(seventh), "the same pluck counted twice");

    // Three other notes, each held steadily, and no attack reported at all.
    let mut frame = 0u64;
    for pc in [0usize, 3, 0] {
        for _ in 0..4 {
            frame += 1;
            a.audio_frames = frame - 1;
            a.feed_estimate(frame, Some(pc));
        }
        a.check_progress_with_ai(0.3, "Noise", 0.0);
    }
    // The phrase comes back to it and the player plays it.
    for _ in 0..4 {
        frame += 1;
        a.audio_frames = frame - 1;
        a.feed_estimate(frame, Some(seventh));
    }
    assert!(
        a.struck_since_credit(seventh),
        "the phrase could not come back to a note it had used"
    );
}

/// An arpeggio played through, note by note, finishes and starts again.
#[test]
fn an_arpeggio_played_through_comes_round() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = 0;
    a.set_random_mode(false);
    a.item_selected(0);                       // the minor study
    a.note_threshold = 0.5;
    let chord = a.chords[0].clone();
    let steps = a.get_active_indices(&chord);
    let all = chord.get_target_indices();
    assert!(steps.len() > 8, "the phrase is too short to tell");

    let mut frame = 0u64;
    for (n, step) in steps.iter().enumerate() {
        let pc = all[step.degree] % 12;
        // Played: the string struck, the estimate on it, the model too.
        detector_hop(&mut a, &[pc]);
        a.last_pitches = [0.0; 12];
        a.last_pitches[pc] = 1.0;
        for _ in 0..4 {
            frame += 1;
            a.audio_frames = frame - 1;
            a.feed_estimate(frame, Some(pc));
        }
        for _ in 0..10 {
            a.check_progress_with_ai(0.1, "Noise", 0.0);
        }
        let expected = if n + 1 == steps.len() { 0 } else { n + 1 };
        assert_eq!(
            a.current_note_step, expected,
            "step {n} of {} did not count", steps.len()
        );
    }
}

/// Played one at a time, nothing but the ear counts - and it has to have
/// held. The model answers about 0.77 s of audio, which is where the notes
/// nobody played come from.
#[test]
fn one_at_a_time_believes_the_ear_alone() {
    let mut a = app();
    a.single_notes = true;
    a.note_threshold = 0.5;
    // The model is sure, loudly, and the ear says nothing: no credit.
    a.last_pitches = [0.0; 12];
    a.last_pitches[7] = 1.0;
    // The ear says nothing at all, so the model's pitch head may answer -
    // a note too quiet for the estimate has to stay playable.
    assert!(a.note_is_sounding(7, None, 0.0), "a quiet note became uncreditable");
    // But the ear naming something ELSE is a refusal.
    a.hears(Some(2));
    assert!(!a.note_is_sounding(7, None, 0.0), "the model overruled the ear");
    // And the chord name never credits a note here.
    a.hears(None);
    a.last_pitches = [0.0; 12];
    assert!(
        !a.note_is_sounding(7, Some(NoteName::G), 1.0),
        "the root head credited a note nobody played"
    );
    // The ear, steadily: that is the whole test.
    a.hears(Some(7));
    assert!(a.note_is_sounding(7, None, 0.0), "the ear was not believed");
    // One frame of it is not enough.
    a.steady_for = 1;
    assert!(!a.note_is_sounding(7, None, 0.0), "a single frame counted");
}

/// Arpeggios are one note at a time by definition, ticked or not.
#[test]
fn an_arpeggio_is_always_one_note_at_a_time() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.single_notes = false;                  // and it makes no difference
    a.note_threshold = 0.5;
    a.last_pitches = [0.0; 12];
    a.last_pitches[7] = 1.0;
    a.hears(Some(2));                        // the ear on another note
    assert!(!a.note_is_sounding(7, None, 0.0), "the model overruled the ear");
    a.hears(Some(7));
    assert!(a.note_is_sounding(7, None, 0.0), "the ear was not believed");
}

/// And "credit only what was struck" holds back every way in, the chord
/// name included: with a microphone in a room the model goes on naming
/// chords, and their roots were being credited as notes.
#[test]
fn the_root_head_needs_an_attack_too() {
    let mut a = app();
    a.single_notes = false;
    a.require_onset = true;
    a.note_threshold = 0.5;
    assert!(
        !a.note_is_sounding(7, Some(NoteName::G), 1.0),
        "a chord name credited a note with no attack behind it"
    );
    detector_hop(&mut a, &[7]);
    assert!(
        a.note_is_sounding(7, Some(NoteName::G), 1.0),
        "a struck note went uncredited"
    );
}

/// One pluck credits one step. A note credited a moment ago and still
/// ringing must not count again for the next step that asks for it - the
/// reported case being `3` played and `1` credited along with it.
#[test]
fn a_ringing_note_does_not_count_for_the_next_step() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    let (root, third) = (0usize, 4usize);

    a.hears(Some(root));
    a.credit_class(root, 0);

    // The third is played on top; the root rings on, and the model - which
    // is polyphonic - still reports it.
    let mut frame = 0u64;
    for _ in 0..6 {
        frame += 1;
        a.audio_frames = frame - 1;
        a.feed_estimate(frame, Some(third));
    }
    a.tick(0.6);
    assert!(
        !a.struck_since_credit(root),
        "the ringing root counted again while the third was being played"
    );

    // Struck again - the estimate names it, because it is loudest now.
    for _ in 0..4 {
        frame += 1;
        a.audio_frames = frame - 1;
        a.feed_estimate(frame, Some(root));
    }
    assert!(a.struck_since_credit(root), "a real second pluck did not count");
}

/// A note asked for again, after something else has been played, counts
/// when it is played - even where the attack head missed the pluck.
///
/// The reported case: the fretboard trainer lighting green on a note that
/// was plainly right and never moving on. The note had been credited a few
/// steps earlier, the head did not report the new attack, and the rule that
/// stops one pluck counting twice refused it for ever.
#[test]
fn a_note_that_went_quiet_counts_when_it_comes_back() {
    let mut a = app();
    a.set_mode(AppMode::Fretboard as i32);
    let pc = 4usize;
    a.hears(Some(pc));
    a.credit_class(pc, 0);
    assert!(!a.struck_since_credit(pc), "an unstruck repeat counted at once");

    // Something else, steadily, and then silence: the note has gone.
    for frame in 1..=4u64 {
        a.audio_frames = frame - 1;
        a.feed_estimate(frame, Some(9));
    }
    a.check_progress_with_ai(0.6, "Noise", 0.0);   // past the settling window
    // And now it is played again: the estimate names it, because a struck
    // string is what is loudest.
    for frame in 5..=8u64 {
        a.audio_frames = frame - 1;
        a.feed_estimate(frame, Some(pc));
    }
    assert!(
        a.struck_since_credit(pc),
        "the note stayed uncreditable after it stopped sounding"
    );
}

/// A drawing follows a reading that has held, not every frame: at a
/// microphone, with something playing in the room, the neck flickered.
#[test]
fn a_stray_reading_does_not_reach_the_drawing() {
    let mut a = app();
    assert_eq!(a.steady_note(), None, "nothing has been heard yet");

    // One frame of a note is not a note.
    for (frame, pitch) in [(1u64, Some(3usize)), (2, Some(7)), (3, Some(3))] {
        a.audio_frames = frame - 1;
        a.feed_estimate(frame, pitch);
        assert_eq!(a.steady_note(), None, "a single frame reached the drawing");
    }
    // Three frames of the same one is.
    for frame in 4..=6u64 {
        a.audio_frames = frame - 1;
        a.feed_estimate(frame, Some(9));
    }
    assert_eq!(a.steady_note(), Some(9), "a reading that held did not count");

    // And it lets go as soon as something else is heard.
    a.audio_frames = 6;
    a.feed_estimate(7, Some(2));
    assert_eq!(a.steady_note(), None, "the old reading outlived what replaced it");
}

/// A note quiet enough that the estimate loses it every other frame was
/// uncreditable: the run of three was rarely reached, and the answer was to
/// play louder. See `CREDIT_TICKS` for what was measured.
#[test]
fn two_frames_are_enough_to_credit_but_not_to_take_over() {
    let mut a = app();
    a.app_mode = AppMode::Scales;
    assert_eq!(a.sounding_now(), None, "nothing heard yet");

    a.audio_frames = 0;
    a.feed_estimate(1, Some(9));
    assert_eq!(a.sounding_now(), None, "one frame is still not a note");

    a.audio_frames = 1;
    a.feed_estimate(2, Some(9));
    assert_eq!(a.sounding_now(), Some(9), "two frames of the same reading");
    // But the longer measure - the one that says a credited note has gone
    // quiet - has not been reached, so a repeat cannot ride on this.
    assert_eq!(a.steady_note(), None, "two frames must not count as taking over");

    a.audio_frames = 2;
    a.feed_estimate(3, Some(9));
    assert_eq!(a.steady_note(), Some(9), "three frames still means what it did");
}

/// A grip is three notes under one hand: which finger lands first is not
/// the exercise, so the set is taken in any order unless asked otherwise.
#[test]
fn intervals_take_the_notes_in_any_order() {
    for shuffled in [false, true] {
        let mut a = app();
        a.set_mode(AppMode::Intervals as i32);
    a.interval_in_order = false; // this one is about free order
        a.set_random_mode(shuffled);
        a.note_threshold = 0.5;
        assert!(a.free_order(), "free order is the default");
        assert_eq!(a.note_due(), None, "nothing is due when any of them will do");

        let chord = a.chords[a.current_chord_index].clone();
        let steps = a.ordered_active_indices(&chord);
        let all = chord.get_target_indices();
        assert!(steps.len() >= 3, "a set to take out of order");
        // The LAST step first, which in order would not be answerable yet.
        let last = steps.len() - 1;
        let pc = all[steps[last].degree] % 12;
        a.last_pitches = [0.0; 12];
        a.last_pitches[pc] = 1.0;
        for _ in 0..15 {
            a.check_progress_with_ai(0.016, "Noise", 0.0);
            a.audio_gate_open = true;
            a.audio_frames += 1;
            a.tick(0.016);
        }
        assert!(
            a.collected_notes[last],
            "the last step was not credited first (shuffled: {shuffled})"
        );
        assert!(!a.collected_notes[0], "it credited a step nobody played");
    }
}

/// The reported case: the root is played, the model's pitch head spreads
/// onto the b3 as well, and free order asked every step still wanted - so
/// whichever of them came first in the deal was credited.
#[test]
fn a_neighbour_is_not_credited_while_the_root_is_sounding() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.set_random_mode(false);
    a.note_threshold = 0.5;
    a.intervals_input = "1 b3 5".to_string();
    a.reset_logic_state();
    let chord = a.chords[a.current_chord_index].clone();
    let steps = a.ordered_active_indices(&chord);
    let all = chord.get_target_indices();
    let root = all[steps[0].degree] % 12;
    let third = all[steps[1].degree] % 12;
    assert_ne!(root, third);

    // The root is played: the estimate names it, and the head answers for
    // the third as well - which is what it does on a real pluck.
    a.last_pitches = [0.0; 12];
    a.last_pitches[root] = 1.0;
    a.last_pitches[third] = 0.9;
    a.hears(Some(root));
    for _ in 0..15 {
        a.check_progress_with_ai(0.016, "Noise", 0.0);
        a.audio_gate_open = true;
        a.audio_frames += 1;
        a.tick(0.016);
    }
    assert!(a.collected_notes[0], "the note that was played was not credited");
    assert!(!a.collected_notes[1], "the b3 was credited off the root");
}

/// And with the option on, only the step due answers.
#[test]
fn the_option_puts_the_intervals_back_in_order() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.set_random_mode(false);
    a.interval_in_order = true;
    a.note_threshold = 0.5;
    assert!(!a.free_order());
    assert_eq!(a.note_due(), Some(0), "the picture has a note to ring");

    let chord = a.chords[a.current_chord_index].clone();
    let steps = a.ordered_active_indices(&chord);
    let all = chord.get_target_indices();
    let last = steps.len() - 1;
    let pc = all[steps[last].degree] % 12;
    a.last_pitches = [0.0; 12];
    a.last_pitches[pc] = 1.0;
    for _ in 0..15 {
        a.check_progress_with_ai(0.016, "Noise", 0.0);
        a.audio_gate_open = true;
        a.audio_frames += 1;
        a.tick(0.016);
    }
    assert!(!a.collected_notes[last], "a step out of turn was credited");
    assert_eq!(a.current_note_step, 0, "the exercise moved off the step it asked for");
}

/// A phrase keeps its order, whichever phrase is showing: the notes of a
/// study dealt at random are no longer that study.
#[test]
fn the_shuffle_does_not_reorder_an_arpeggio() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = 0;
    a.item_selected(0);
    a.set_random_mode(true);
    let chord = a.chords[0].clone();
    assert_eq!(
        a.ordered_active_indices(&chord),
        a.get_active_indices(&chord),
        "the shuffle dealt the phrase a new order"
    );
}

/// What the shuffle means there: the KEY. The study is the player's choice
/// and the switch does not touch it.
#[test]
fn the_shuffle_moves_the_key_and_leaves_the_study() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = 0;
    a.item_selected(2);
    let study = a.arpeggio_patterns[2].name.clone();
    let key = a.chords[0].root as usize;

    a.set_random_mode(true);
    assert_eq!(
        a.arpeggio_patterns[a.selected_library_idx].name, study,
        "the shuffle changed the study"
    );

    let mut moved = false;
    for _ in 0..20 {
        a.advance_chord();
        if a.chords[0].root as usize != key {
            moved = true;
            break;
        }
    }
    assert!(moved, "the key never moved between passes");
    assert_eq!(
        a.arpeggio_patterns[a.selected_library_idx].name, study,
        "a pass changed the study"
    );
}

/// And the studies named for a chord know which one: the phrase is the
/// book's only when it is read over the chord it was written for.
#[test]
fn a_study_named_for_a_chord_names_the_right_one() {
    use crate::model::study_quality;
    assert_eq!(study_quality("Minor (Two Octaves and a Third)"), Some(0));
    assert_eq!(study_quality("Major (Leading Tone)"), Some(1));
    assert_eq!(study_quality("Dominant (Approach from Below)"), Some(2));
    // The rest name no quality, and must not set one.
    assert_eq!(study_quality("Skipping Notes (Fifths and Fourths)"), None);
    assert_eq!(study_quality("Triplets (Up-Down)"), None);
    assert_eq!(study_quality("Two Octaves Up-Down"), None);
    // The index has to mean what the settings list means by it.
    for (i, want) in [(0, "m7"), (1, "Maj7"), (2, "7")] {
        assert_eq!(ARP_QUALITIES[i].0, want, "the quality list moved under it");
    }
}

/// The study on screen is the study that was chosen, in every key.
///
/// The reported case: with the key on A - the tenth row of the key list,
/// and the generator sits in the tenth row of the PATTERN list - the phrase
/// was quietly replaced by a freshly generated one, so the name above said
/// "Up-Down, Approach from Below" while the neck showed something else
/// entirely, two octaves shorter and never reaching the top string.
#[test]
fn a_study_plays_the_phrase_it_names() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = 0;
    let patterns = a.arpeggio_patterns.clone();
    for (i, pattern) in patterns.iter().enumerate() {
        if pattern.name == crate::model::GENERATOR_NAME {
            continue;                       // that one is meant to change
        }
        for key in 0..12 {
            a.item_selected(i as i32);
            a.secondary_item_selected(key);
            assert_eq!(
                a.intervals_input,
                pattern.names.join(" "),
                "{} in key {key} was swapped for something else",
                pattern.name
            );
            assert_eq!(a.song_title, pattern.name, "the name above does not match");
            assert_eq!(a.chords[0].root as usize, key as usize, "the key was not taken");
        }
    }
}

/// Over the changes the phrase is built, and the descending one is the
/// shape the studies use: from the root DOWNWARDS through the chord's own
/// tones, not the ascending phrase read backwards.
#[test]
fn the_changes_run_matches_the_studies() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = 1;
    let chord = Chord { root: NoteName::D, quality: ChordQuality::Dominant7 };

    a.arp_direction = 0;
    let up: Vec<(usize, i8)> =
        a.get_active_indices(&chord).iter().map(|s| (s.degree, s.octave)).collect();
    assert_eq!(
        up,
        vec![(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
        "1 3 5 7 1' 3' 5' 7'"
    );

    a.arp_direction = 1;
    let down: Vec<(usize, i8)> =
        a.get_active_indices(&chord).iter().map(|s| (s.degree, s.octave)).collect();
    assert_eq!(
        down,
        vec![(0, 0), (3, -1), (2, -1), (1, -1), (0, -1), (3, -2), (2, -2), (1, -2)],
        "1 7, 5, 3, 1, 7,, 5,, 3,,"
    );
}

/// Alternating: one chord one way, the next the other, from whichever side
/// the option starts on.
#[test]
fn alternating_turns_the_run_round_at_every_chord() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = 1;
    a.set_random_mode(false);
    let chord = a.chords[a.current_chord_index].clone();
    let is_down = |a: &MyApp, c: &Chord| a.get_active_indices(c)[1].octave < 0;

    a.arp_direction = 2;                    // from the descending one
    assert!(is_down(&a, &chord), "the first chord did not start downwards");
    a.play_pos = 1;
    assert!(!is_down(&a, &chord), "the second chord did not turn round");

    a.arp_direction = 3;                    // and the other way about
    assert!(is_down(&a, &chord), "starting up, the second chord runs down");
    a.play_pos = 0;
    assert!(!is_down(&a, &chord), "the first chord did not start upwards");
}

/// Every token of every arpeggio pattern is a chord tone the mode can ask
/// for. A typo drops the step silently - the phrase just comes out shorter
/// than it was written - so the count is what is checked. Some patterns run
/// one way only, so where they end is not.
#[test]
fn the_arpeggio_patterns_are_playable_as_written() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    let chord = Chord { root: NoteName::C, quality: ChordQuality::Minor7 };
    for pattern in crate::model::load_arpeggio_patterns() {
        if pattern.name == crate::model::GENERATOR_NAME {
            continue;                       // built fresh every pass
        }
        let written = pattern.names.join(" ");
        a.intervals_input = written.clone();
        let steps = a.get_active_indices(&chord);
        assert_eq!(
            steps.len(),
            pattern.names.len(),
            "{}: {} of {} tokens are playable - {written}",
            pattern.name,
            steps.len(),
            pattern.names.len()
        );
    }
}

/// Every built-in scale asks for its own notes, in its own order.
///
/// The reported case: the altered scale showed `1 b2 #2 #2 b5 #5 b7` - the
/// major third asked for as `#2`, and the note itself never asked for at
/// all. Ten of the twenty-six scales were walking a step twice like that,
/// wherever they spell both an altered degree and its natural neighbour.
#[test]
fn a_scale_walks_the_degrees_it_is_written_with() {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    a.scale_descending = false;              // the written order, not a pass
    let scales = crate::model::load_all_scale_definitions();
    assert!(scales.len() >= 20, "the built-in scales did not load");
    for def in scales {
        let written: Vec<String> = def.names.clone();
        a.intervals_input = written.join(" ");
        let chord = Chord {
            root: NoteName::C,
            quality: ChordQuality::CustomScale(def.clone()),
        };
        let walked: Vec<usize> =
            a.get_active_indices(&chord).iter().map(|s| s.degree).collect();
        assert_eq!(
            walked,
            (0..written.len()).collect::<Vec<_>>(),
            "{} walks {:?} of {:?}",
            def.name,
            walked.iter().map(|&i| &written[i]).collect::<Vec<_>>(),
            written
        );
    }
}

/// And the loose rules still answer for a token the set spells otherwise:
/// `3` over a minor seventh is its flat third, not nothing at all.
#[test]
fn a_plain_degree_still_finds_the_one_the_chord_has() {
    let mut a = app();
    a.set_mode(AppMode::Intervals as i32);
    a.intervals_input = "1 3 5 7".to_string();
    let chord = Chord { root: NoteName::C, quality: ChordQuality::Minor7 };
    let walked: Vec<usize> =
        a.get_active_indices(&chord).iter().map(|s| s.degree).collect();
    assert_eq!(walked, vec![0, 1, 2, 3], "b3 and b7 stopped answering for 3 and 7");
}

/// The scale run ends where it started, when the option asks for it.
#[test]
fn a_scale_can_end_on_its_root_again() {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    a.scale_descending = false;              // the closing root is the top one
    let chord = a.chords[0].clone();
    let plain = a.get_active_indices(&chord);
    a.scale_repeat_root = true;
    let looped = a.get_active_indices(&chord);
    assert_eq!(looped.len(), plain.len() + 1, "no extra step");
    assert_eq!(
        looped.last().unwrap().degree,
        plain.first().unwrap().degree,
        "the run does not end on the note it started from"
    );
    assert_eq!(looped.last().unwrap().octave, plain.first().unwrap().octave + 1,
               "the closing root is not marked an octave up");
    // And it is the scales' own: the other modes are untouched.
    a.set_mode(AppMode::Intervals as i32);
    let chord = a.chords[a.current_chord_index].clone();
    assert_eq!(a.get_active_indices(&chord).len(),
               { a.scale_repeat_root = false; a.get_active_indices(&chord).len() });
}

/// An arpeggio may ask for the same pitch class twice in a row. One pluck
/// must credit one step, or a single ringing string walks through both.
///
/// No longer gated on "play the notes one at a time": a repeat off one
/// pluck was wrong in every mode, and the test never fires on a strummed
/// chord because there the successive steps are different classes.
#[test]
fn one_pluck_credits_one_step() {
    let mut a = app();
    a.single_notes = false;                    // the default, and it still holds
    a.set_mode(AppMode::Fretboard as i32);
    a.transition_delay = 0.05;
    a.note_threshold = 0.5;
    let target = a.fret_target.unwrap();
    a.last_pitches = [0.0; 12];
    a.last_pitches[target % 12] = 1.0;
    a.credit_class(target, 0);                // this pluck already counted
    a.onset_id = 0;
    for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
    assert_eq!(a.fret_target, Some(target), "the same pluck counted twice");

    a.onset_id = 1;                            // strings struck again
    for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
    assert_eq!(
        a.credited[target % 12], Some(credit(1)),
        "a fresh attack on the right note did not count"
    );
}

/// A scale that ends on the note it starts from asks for that note twice
/// over the turn of the lap. The string is still ringing, so the first step
/// of the next pass has to wait for it to be played again.
#[test]
fn the_closing_root_does_not_open_the_next_pass() {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    a.set_random_mode(false);
    a.scale_repeat_root = true;
    a.note_threshold = 0.5;
    let chord = a.chords[0].clone();
    let steps = a.get_active_indices(&chord);
    let root_pc = chord.get_target_indices()[steps[0].degree] % 12;

    // Standing on the closing root and playing it: the lap ends and the
    // string goes on ringing, which is where the run leaves the player.
    a.current_note_step = steps.len() - 1;
    a.last_pitches = [0.0; 12];
    a.last_pitches[root_pc] = 1.0;

    for _ in 0..20 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
    assert_eq!(a.current_note_step, 0, "the closing root did not end the lap");

    // The string is still ringing, so the model goes on reporting it.
    a.last_pitches[root_pc] = 1.0;
    for _ in 0..20 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
    assert_eq!(a.current_note_step, 0, "the ringing root opened the next pass");

    detector_hop(&mut a, &[root_pc]);           // played again
    for _ in 0..20 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
    assert_eq!(a.current_note_step, 1, "a real second pluck did not count");
}

/// The whole run, played with every note left ringing. The closing root is
/// the note the run started from, six notes back and still sounding, and
/// the six plucks in between move the envelope's attack counter - which is
/// why the envelope cannot be what answers "was this struck again".
#[test]
fn the_closing_root_is_not_credited_to_the_opening_one() {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    a.set_random_mode(false);
    a.scale_repeat_root = true;
    a.note_threshold = 0.5;
    let chord = a.chords[0].clone();
    let steps = a.get_active_indices(&chord);
    let all = chord.get_target_indices();
    let pcs: Vec<usize> = steps.iter().map(|st| all[st.degree] % 12).collect();
    assert_eq!(pcs.first(), pcs.last(), "the run does not close on its root");

    // The run played note by note, each one left ringing. The estimate
    // follows the newest string.
    let low = 24 + pcs[0];                      // where the run starts
    for (i, pc) in pcs.iter().take(steps.len() - 1).enumerate() {
        a.last_pitches[*pc] = 1.0;
        a.hears(Some(*pc));
        a.cqt_semitone = Some(low + [0, 2, 4, 5, 7, 9, 11][i]);
        a.onset_id += 1;
        for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
    }
    assert_eq!(a.current_note_step, steps.len() - 1, "the run did not reach its last step");

    // Nothing is played now. The opening root goes on ringing, the
    // envelope's counter has moved, and the estimate is reading the seventh
    // degree - the string that was hit last.
    for _ in 0..20 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
    assert_eq!(a.current_note_step, steps.len() - 1, "the opening root closed the run");

    // Played where the run asks for it, an octave up: the estimate says so,
    // and that is enough on its own.
    a.hears(Some(pcs[0]));
    a.cqt_semitone = Some(low + 12);
    for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
    assert_eq!(a.current_note_step, 0, "playing the closing root did not end the run");
}

/// And the run that closes where it started, in the same octave - a shape
/// that does not reach up. The estimate cannot prove anything there, so the
/// note has to be struck again, which is the whole of what is asked.
#[test]
fn a_root_repeated_in_its_own_octave_still_counts() {
    let mut a = app();
    a.set_mode(AppMode::Scales as i32);
    a.set_random_mode(false);
    a.scale_repeat_root = true;
    a.note_threshold = 0.5;
    let chord = a.chords[0].clone();
    let steps = a.get_active_indices(&chord);
    let all = chord.get_target_indices();
    let pc = all[steps[0].degree] % 12;
    let low = 24 + pc;

    a.current_note_step = steps.len() - 1;
    a.last_pitches[pc] = 1.0;
    a.hears(Some(pc));
    a.cqt_semitone = Some(low);
    a.credit_class(pc, 0);                      // credited on the opening root

    // Ringing on, nothing struck: it must not close the run.
    for _ in 0..20 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
    assert_eq!(a.current_note_step, steps.len() - 1, "the ring closed the run");

    // Struck again, in the same octave, with the estimate reading it.
    detector_hop(&mut a, &[pc]);
    a.hears(Some(pc));
    a.cqt_semitone = Some(low);
    for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
    assert_eq!(a.current_note_step, 0, "a note played again in its own octave never counted");
}

/// A second strike of the note just credited is a second pluck; the note
/// lingering is not.
#[test]
fn a_second_strike_is_seen_by_the_detector() {
    let mut a = app();
    a.single_notes = false;
    a.set_mode(AppMode::Fretboard as i32);
    a.transition_delay = 0.05;
    a.note_threshold = 0.5;
    let target = a.fret_target.unwrap();
    let pc = target % 12;
    detector_hop(&mut a, &[pc]);
    a.last_pitches = [0.0; 12];
    a.last_pitches[pc] = 1.0;
    a.credit_class(pc, 0);                       // credited on that strike

    // The note lingering does not count as another strike.
    for _ in 0..10 {
        detector_hop(&mut a, &[]);
        a.check_progress_with_ai(0.1, "Noise", 0.0);
        a.tick(0.1);
    }
    assert_eq!(a.fret_target, Some(target), "a lingering note counted as a new pluck");

    // Struck again: that is a second pluck.
    detector_hop(&mut a, &[pc]);
    for _ in 0..10 {
        a.check_progress_with_ai(0.1, "Noise", 0.0);
        a.tick(0.1);
    }
    assert_ne!(a.fret_target, Some(target), "the second pluck did not count");
}

/// Feeds the note the exercise is currently asking for, once.
fn play_current_note(a: &mut MyApp, dt: f32) {
    let chord = a.chords[a.current_chord_index].clone();
    let steps = a.ordered_active_indices(&chord);
    let targets = chord.get_target_indices();
    if let Some(step) = steps.get(a.current_note_step) {
        let pc = targets[step.degree] % 12;
        a.last_pitches = [0.0; 12];
        a.last_pitches[pc] = 1.0;
    }
    a.check_progress_with_ai(dt, "Noise", 0.0);
}

/// Scales: the key is drawn afresh once the whole scale has been played.
#[test]
fn scales_change_key_after_the_scale_is_finished() {
    let mut a = scales_app();
    a.set_random_mode(true);
    a.note_threshold = 0.5;
    let first = a.chords[0].root as usize;
    let steps = a.ordered_active_indices(&a.chords[0].clone()).len();
    for _ in 0..steps { play_current_note(&mut a, 0.5); }
    assert_ne!(a.chords[0].root as usize, first, "the key did not move after a full pass");
}

/// Paused, the step never completes, so the scale never finishes and the key
/// stays put. This is the chain the pause button relies on.
#[test]
fn paused_scales_keep_their_key_and_their_step() {
    let mut a = scales_app();
    a.set_random_mode(true);
    a.note_threshold = 0.5;
    a.paused = true;
    let key = a.chords[0].root as usize;
    let step = a.current_note_step;
    for _ in 0..200 { play_current_note(&mut a, 0.5); }
    assert_eq!(a.current_note_step, step, "paused, yet it stepped through the scale");
    assert_eq!(a.chords[0].root as usize, key, "paused, yet the key changed");
}

/// ...and unpausing lets it run again.
#[test]
fn unpausing_lets_the_scale_finish() {
    let mut a = scales_app();
    a.set_random_mode(true);
    a.note_threshold = 0.5;
    a.paused = true;
    for _ in 0..50 { play_current_note(&mut a, 0.5); }
    let key = a.chords[0].root as usize;
    a.paused = false;
    let steps = a.ordered_active_indices(&a.chords[0].clone()).len();
    for _ in 0..steps { play_current_note(&mut a, 0.5); }
    assert_ne!(a.chords[0].root as usize, key, "still stuck after unpausing");
}

#[test]
fn hint_appears_only_with_randomisation() {
    let mut a = app();
    a.set_random_mode(false);
    assert!(a.start_hint.is_none());
    a.set_random_mode(true);
    assert!(a.start_hint.is_some());
    assert!(a.start_hint.unwrap() < START_STRINGS.len());
}

/// The ways the model may name the chord asked for, and what each is worth.
#[test]
fn chord_readings_against_the_chord_asked_for() {
    use NoteName::*;
    let chord = |root, quality| Chord { root, quality };
    let read = |target: &Chord, root, heard| chord_reading(target, root, heard, false);

    let a_half_dim = chord(A, ChordQuality::HalfDiminished);
    assert_eq!(read(&a_half_dim, A, "m7b5"), ChordReading::Exact);
    assert_eq!(read(&a_half_dim, A, "dim"), ChordReading::Partial);
    assert_eq!(read(&a_half_dim, A, "m7"), ChordReading::Partial);
    assert_eq!(chord_reading(&a_half_dim, A, "m7", true), ChordReading::Exact, "the m7 shell is the shell asked for");
    assert_eq!(read(&a_half_dim, C, "m"), ChordReading::Partial, "C m is A m7b5 without its root");
    assert_eq!(read(&a_half_dim, C, "m7"), ChordReading::Wrong);

    let c_major = chord(C, ChordQuality::Major7);
    assert_eq!(read(&c_major, C, ""), ChordReading::Partial);
    assert_eq!(read(&c_major, E, "m7"), ChordReading::Partial, "E m7 is C Maj7 without its root");
    assert_eq!(read(&c_major, D, "m7"), ChordReading::Wrong);

    let c_dim = chord(C, ChordQuality::Diminished);
    for root in [C, Ef, Fsh, A] {
        assert_eq!(read(&c_dim, root, "dim"), ChordReading::Exact, "{root:?} dim is the same chord");
    }
    assert_eq!(read(&c_dim, D, "dim"), ChordReading::Wrong);

    assert_eq!(read(&chord(G, ChordQuality::Dominant7), G, ""), ChordReading::Partial);
    assert_eq!(read(&chord(D, ChordQuality::Minor7), D, "m"), ChordReading::Partial);
}
