use super::tests::*;
use super::*;

fn pick_generator(a: &mut MyApp) {
    a.set_mode(AppMode::Arpeggios as i32);
    a.arp_exercise = 0;
    let idx = a.arpeggio_patterns.iter()
        .position(|p| p.name == crate::model::GENERATOR_NAME)
        .expect("generator entry missing from the arpeggio list");
    // The phrase is the first combo in the studies; the second holds the key.
    a.item_selected(idx as i32);
}

/// The generator sits last, after the hand-written phrases.
#[test]
fn generator_is_the_last_entry() {
    let a = app();
    let last = a.arpeggio_patterns.last().expect("no arpeggio patterns");
    assert_eq!(last.name, crate::model::GENERATOR_NAME);
}

/// Selecting it must replace the placeholder with a real phrase.
#[test]
fn selecting_the_generator_builds_a_phrase() {
    let mut a = app();
    pick_generator(&mut a);
    let n = a.intervals_input.split_whitespace().count();
    assert!(n >= 8, "phrase of only {n} steps: {:?}", a.intervals_input);
}

/// A new phrase after every pass - that is the whole point of the entry.
#[test]
fn each_pass_brings_a_different_phrase() {
    let mut a = app();
    pick_generator(&mut a);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..30 {
        seen.insert(a.intervals_input.clone());
        a.advance_chord();
    }
    assert!(seen.len() > 3, "only {} distinct phrases in 30 passes", seen.len());
}

/// The fixed patterns must be left alone.
#[test]
fn hand_written_patterns_are_not_regenerated() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.secondary_item_selected(0);
    let first = a.intervals_input.clone();
    for _ in 0..10 { a.advance_chord(); }
    assert_eq!(a.intervals_input, first, "a fixed pattern was overwritten");
}

/// Leaving the mode and coming back keeps the generator selected AND brings
/// a new phrase - both halves matter, and the selection half was broken:
/// set_mode zeroed the pattern index.
#[test]
fn leaving_and_returning_regenerates() {
    let mut a = app();
    pick_generator(&mut a);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..30 {
        seen.insert(a.intervals_input.clone());
        a.set_mode(AppMode::Chords as i32);
        a.set_mode(AppMode::Arpeggios as i32);
        assert_eq!(
            a.arpeggio_patterns[a.selected_library_idx].name,
            crate::model::GENERATOR_NAME,
            "the generator selection was lost on the way back"
        );
    }
    assert!(seen.len() > 3, "only {} distinct phrases across 30 round trips", seen.len());
}

/// A fixed pattern must survive the round trip too, unchanged.
#[test]
fn a_fixed_pattern_survives_the_round_trip() {
    let mut a = app();
    a.set_mode(AppMode::Arpeggios as i32);
    a.item_selected(1);
    let name = a.arpeggio_patterns[1].name.clone();
    let phrase = a.intervals_input.clone();
    a.set_mode(AppMode::Scales as i32);
    a.set_mode(AppMode::Arpeggios as i32);
    assert_eq!(a.arpeggio_patterns[a.selected_library_idx].name, name);
    assert_eq!(a.intervals_input, phrase, "a fixed pattern changed on the way back");
}

/// Whatever it builds has to be playable by the exercise logic.
#[test]
fn generated_phrases_drive_the_exercise() {
    let mut a = app();
    pick_generator(&mut a);
    for _ in 0..20 {
        let chord = a.chords[a.current_chord_index].clone();
        let steps = a.ordered_active_indices(&chord);
        assert!(!steps.is_empty(), "phrase {:?} yielded no steps", a.intervals_input);
        let names = chord.quality.interval_names();
        for s in &steps {
            assert!(s.degree < names.len(), "step points past the chord");
        }
        a.advance_chord();
    }
}

/// A formula played through STAYS. It is the exercise itself, not a step
/// inside one, so finishing it starts another lap rather than drawing a
/// replacement; a different formula comes from the settings.
#[test]
fn finishing_a_formula_keeps_it() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    let mask = a.formula_mask;
    assert!(mask != 0, "entering the mode should draw a formula");

    for (n, pc) in a.formula_pitches().into_iter().enumerate() {
        a.onset_id = n as u64 + 1;
        ear_hears(&mut a, pc);
    }
    assert!(a.formula_collected.iter().all(|&c| c), "the formula was not collected");

    // Past the hold time, and once the last note has died away, the lap
    // ends - see `a_lap_waits_for_its_own_notes_to_die`.
    ear_silent(&mut a);
    frames_pass(&mut a, 1, FORMULA_LAP_PAUSE + 0.01);
    assert_eq!(a.formula_mask, mask, "a new formula was drawn");
    assert!(
        a.formula_collected.iter().all(|&c| !c),
        "the marks did not start again"
    );
}

/// A single frame of the fast estimate credits nothing.
///
/// It names a neighbour now and then - the root crediting the semitone
/// beside it was the shape this took on the guitar - and a mark here never
/// expires. Held for four ticks, the same reading counts.
#[test]
fn one_frame_of_the_fast_estimate_is_not_enough() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.strict_formulas = true;
    a.onset_id = 2;
    let pitches = a.formula_pitches();

    // Flickering between two of them: neither ever settles.
    ready(&mut a);
    for _ in 0..8 {
        for &pc in pitches.iter().take(2) {
            ear_frame(&mut a, Some(pc));
        }
    }
    assert!(
        a.formula_collected.iter().all(|&c| !c),
        "a flickering estimate marked functions off"
    );

    // Held still, it counts.
    ear_hears(&mut a, pitches[0]);
    assert!(a.formula_collected[0], "a steady reading did not count");
}

/// The strict rule is what the mode runs on, and the loose one - what the
/// note modes use - is still there to compare against. It credits whatever
/// is sounding, which after a few notes is most of the formula: 110 things
/// nobody played over 49 notes, against 33.
#[test]
fn strict_is_the_default_and_loose_still_works() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    assert!(a.strict_formulas, "the strict rule is what the mode runs on");

    a.strict_formulas = false;
    a.onset_id = 2;
    let pc = a.formula_pitches()[0];
    // The model's pitch head alone, on the first frame after it is seen.
    ready(&mut a);
    a.last_pitches = [0.0; 12];
    a.last_pitches[pc % 12] = 1.0;
    ear_frame(&mut a, None);
    assert!(a.formula_collected[0], "the note modes would have counted this");
}

/// A finished lap moves on even when the player has stopped playing.
///
/// The judge is only called when the model answers, and the model is only
/// asked while the context window is nine tenths full - so when the hands
/// come off the strings it stops being called, and the lap that timed itself
/// there stopped with it: every function green and no next formula, until
/// something was played again.
#[test]
fn a_finished_lap_moves_on_in_silence() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.strict_formulas = true;
    for (n, &pc) in a.formula_pitches().iter().enumerate() {
        a.onset_id = n as u64 + 1;
        ear_hears(&mut a, pc);
    }
    assert!(a.formula_collected.iter().all(|&c| c), "the formula was not collected");

    // Silence: the ear lets go, then nothing is asked of the model at all -
    // only frames go by, which is what happens when the hands come off.
    ear_silent(&mut a);
    frames_pass(&mut a, 200, 0.016);
    assert!(
        a.formula_collected.iter().all(|&c| !c),
        "the lap never moved on with nobody playing"
    );
}

/// Paused in Formulas is "your turn": nothing is judged until it is let go.
#[test]
fn a_paused_formula_judges_nothing() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.strict_formulas = true;
    a.onset_id = 2;
    let pc = a.formula_pitches()[0];
    ready(&mut a);
    a.paused = true;
    for _ in 0..40 {
        ear_frame(&mut a, Some(pc));
    }
    assert!(
        a.formula_collected.iter().all(|&c| !c),
        "a paused formula was still being judged"
    );

    a.paused = false;
    ear_hears(&mut a, pc);
    assert!(a.formula_collected[0], "letting go did not start it again");
}

/// A finished lap moves on after its pause, and holds for a moment so that
/// what was ringing when it ended cannot walk into it.
///
/// The hold is timed rather than tied to the next attack. Tied to the
/// attack it blocked everything until one was detected, and an attack comes
/// from a transient, which soft playing does not always give - so the first
/// function of a new lap could take seconds.
#[test]
fn a_lap_moves_on_and_holds_for_a_moment() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.strict_formulas = true;
    let pitches = a.formula_pitches();
    for (n, &pc) in pitches.iter().enumerate() {
        a.onset_id = n as u64 + 1;
        ear_hears(&mut a, pc);
    }
    assert!(a.formula_collected.iter().all(|&c| c), "the formula was not collected");

    frames_pass(&mut a, 1, FORMULA_LAP_PAUSE + 0.01);
    assert!(a.formula_collected.iter().all(|&c| !c), "the lap did not move on");

    // The last note still ringing, straight away: held.
    for _ in 0..EAR_WINDOW {
        ear_frame(&mut a, Some(pitches[0]));
    }
    assert!(
        a.formula_collected.iter().all(|&c| !c),
        "the decay of the last lap walked into the new one"
    );

    // A moment later the same note, played, counts like any other.
    frames_pass(&mut a, 1, LAP_HOLD);
    for _ in 0..EAR_WINDOW {
        ear_frame(&mut a, Some(pitches[0]));
    }
    assert!(a.formula_collected[0], "the hold never let go");
}

/// One reading per audio frame: no more, no less.
///
/// Sampled off the UI clock the same frame was counted two or three times,
/// and one bad reading could carry the vote by itself. Sampled off the
/// model's answers it stopped whenever the model did - and the model is
/// asked only while the context window is nine tenths full, which playing
/// one note at a time never manages. Both were measured on the guitar, and
/// both were worse.
#[test]
fn a_frame_votes_once_however_often_it_is_looked_at() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.strict_formulas = true;
    a.onset_id = 2;
    let stray = a.formula_pitches()[1];

    // One frame, looked at twenty times: one vote, so nothing counts.
    ready(&mut a);
    a.cqt_pitch = Some(stray);
    a.audio_frames += 1;
    for _ in 0..20 {
        a.tick(0.0);
    }
    assert!(
        a.formula_collected.iter().all(|&c| !c),
        "one frame filled the vote by being looked at repeatedly"
    );

    // Four more frames of it, and it is evidence.
    for _ in 0..(EAR_VOTES - 1) {
        ear_frame(&mut a, Some(stray));
    }
    assert!(a.formula_collected[1], "four readings of five did not count");
}

/// A kept formula comes back as a set, with nothing marked off - and in
/// whatever key is on screen, because a formula has none of its own.
#[test]
fn a_favourite_comes_back_whole() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    let mask = crate::formulas::parse("1 b3 5 b7").unwrap();
    let key = a.formula_key_name.clone();
    a.load_formula(mask);
    assert_eq!(a.formula_mask, mask, "the set did not come back");
    assert_eq!(a.formula_key_name, key, "it brought a key of its own");
    assert_eq!(a.formula_collected.len(), 4, "the marks do not fit the set");
    assert!(a.formula_collected.iter().all(|&c| !c), "it came back half played");
}

/// A settled reading counts. Judging runs where the answers arrive, so each
/// reading in the vote is a new one - sampling it on the UI clock instead
/// counted the same audio frame twice and let one bad reading through.
#[test]
fn a_settled_reading_counts() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.strict_formulas = true;
    a.onset_id = 2;
    let pc = a.formula_pitches()[0];

    ear_hears(&mut a, pc);
    assert!(a.formula_collected[0], "a settled reading did not count");
}

/// In order: the set is played low to high, and a function out of turn
/// waits. Nothing is refused loudly - it simply does not count yet.
#[test]
fn in_order_takes_the_lowest_function_first() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.strict_formulas = true;
    a.formula_in_order = true;
    a.onset_id = 2;
    let pitches = a.formula_pitches();

    // The second function first: it waits.
    ear_hears(&mut a, pitches[1]);
    assert!(
        a.formula_collected.iter().all(|&c| !c),
        "a function out of turn was counted"
    );

    // The first, then the second: both count, in that order.
    a.onset_id += 1;
    ear_hears(&mut a, pitches[0]);
    assert!(a.formula_collected[0], "the function due did not count");
    a.onset_id += 1;
    ear_hears(&mut a, pitches[1]);
    assert!(a.formula_collected[1], "the next one due did not count");
}

/// Off, which is the default, the set is a set: any of them, any order.
#[test]
fn out_of_order_is_the_default() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.strict_formulas = true;
    assert!(!a.formula_in_order, "a formula is unordered unless asked");
    a.onset_id = 2;
    let pitches = a.formula_pitches();
    ear_hears(&mut a, pitches[2]);
    assert!(a.formula_collected[2], "the set refused a function it should take");
}

/// One stray frame does not undo a steady reading. A run had to start over
/// on any flicker, which at half a second to the note was rarely finished
/// before the next note began - the vote survives it.
#[test]
fn a_stray_frame_does_not_reset_the_evidence() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.strict_formulas = true;
    a.onset_id = 2;
    let pitches = a.formula_pitches();
    let (played, stray) = (pitches[0], pitches[1]);

    // The note, one stray reading, then the note again: four of five.
    ready(&mut a);
    let frames = [played, stray, played, played, played];
    for (n, pc) in frames.iter().enumerate() {
        ear_frame(&mut a, Some(*pc));
        if n + 1 < frames.len() {
            assert!(!a.formula_collected[0], "counted before the evidence was in");
        }
    }
    assert!(a.formula_collected[0], "a stray frame threw away four good ones");
}

/// A chord name credits nothing here, however sure the model is of it. The
/// marks never expire, so a function lit by a name nobody played would stay
/// lit for the rest of the exercise.
#[test]
fn a_chord_name_alone_credits_nothing() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.strict_formulas = true;
    a.formula_random_key = false;
    a.formula_key_setting = "C".to_string();
    a.rekey_formula();
    // Nothing heard: no single frame, no pitch vector. Only the name.
    a.cqt_pitch = None;
    a.last_pitches = [0.0; 12];
    a.prev_pitches = [0.0; 12];
    for n in 0..20u64 {
        a.onset_id = n;
        a.check_progress_with_ai(0.05, "C Maj7", 0.99);
    }
    assert!(
        a.formula_collected.iter().all(|&c| !c),
        "the chord name credited a function on its own"
    );
}

/// And it stays in its key. A lap ending is not a reason to move the
/// exercise; the key is drawn for a new formula, not for a new lap of the
/// one being played.
#[test]
fn a_lap_does_not_move_the_key() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.formula_random_key = true;
    let key = a.formula_key_name.clone();
    let mask = a.formula_mask;

    for lap in 0..5u64 {
        for (n, pc) in a.formula_pitches().into_iter().enumerate() {
            a.onset_id = lap * 12 + n as u64 + 1;
            ear_hears(&mut a, pc);
        }
        a.cqt_pitch = None;
        a.check_progress_with_ai(FORMULA_LAP_PAUSE + 0.01, "...", 0.0);
        assert_eq!(a.formula_key_name, key, "the lap moved the key");
        assert_eq!(a.formula_mask, mask, "the lap moved the formula");
    }
}

#[test]
fn over_a_chord_the_lap_moves_the_placement_and_keeps_the_chord() {
    let mut a = app();
    a.formula_exercise = 1;
    a.set_mode(AppMode::Formulas as i32);
    let chord = a.formula_chord.clone().expect("no chord to play over");
    let mask = a.formula_mask;
    let mut degrees = std::collections::HashSet::new();
    for _ in 0..40 {
        degrees.insert(a.formula_degree);
        a.restart_formula();
        assert_eq!(a.formula_mask, mask, "the formula moved with the placement");
        let now = a.formula_chord.clone().unwrap();
        assert_eq!(now.root, chord.root, "the chord moved");
    }
    assert!(degrees.len() > 1, "the placement never moved");
    // And the root the exercise is read from is the chord's root plus the
    // degree, which is what makes a placement a placement.
    let root = a.formula_chord.as_ref().unwrap().root as usize;
    assert_eq!(a.formula_root, (root + a.formula_degree) % 12);
}

#[test]
fn over_the_changes_the_lap_moves_the_chord() {
    let mut a = app();
    a.formula_exercise = 2;
    a.set_mode(AppMode::Formulas as i32);
    // Whatever the library gave it, the tune has to be walked in order.
    a.chords = vec![
        Chord { root: NoteName::C, quality: ChordQuality::Minor7 },
        Chord { root: NoteName::F, quality: ChordQuality::Dominant7 },
        Chord { root: NoteName::Bf, quality: ChordQuality::Major7 },
    ];
    a.current_chord_index = 0;
    let mask = a.formula_mask;
    let mut seen = vec![];
    for _ in 0..6 {
        a.restart_formula();
        seen.push(a.formula_chord.as_ref().unwrap().root);
        assert_eq!(a.formula_mask, mask, "the formula did not survive the chord");
    }
    assert_eq!(
        seen,
        vec![NoteName::F, NoteName::Bf, NoteName::C,
             NoteName::F, NoteName::Bf, NoteName::C],
        "the changes are walked in order and round again",
    );
}

#[test]
fn asking_for_a_kind_of_placement_gets_that_kind() {
    let mut a = app();
    a.formula_exercise = 1;
    a.formula_placement_want = 1; // spells the chord out
    a.set_mode(AppMode::Formulas as i32);
    // A chord and a formula chosen rather than drawn: over some pairs no
    // placement spells the chord out at all, and the draw then answers with
    // what exists rather than refusing - see `draw_placement`. That
    // fallback is the subject of its own test; this one is about getting
    // the kind that was asked for.
    a.formula_mask = crate::formulas::parse("1 b3 5 b7").unwrap();
    a.formula_collected = vec![false; 4];
    a.formula_chord = Some(Chord { root: NoteName::C, quality: ChordQuality::Minor7 });
    a.place_over_chord();
    for _ in 0..30 {
        assert_eq!(
            a.formula_verdict,
            Some(crate::formulas::Verdict::Defines),
            "degree {}",
            a.formula_degree,
        );
        a.restart_formula();
    }
}

/// The key does move when the settings say so - random draws a fresh one.
#[test]
fn a_random_key_is_drawn_when_asked_for() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.formula_random_key = true;
    let mask = a.formula_mask;
    let mut keys = std::collections::HashSet::new();
    for _ in 0..40 {
        keys.insert(a.formula_key_name.clone());
        a.rekey_formula();
        assert_eq!(a.formula_mask, mask, "the formula changed along with the key");
    }
    assert!(keys.len() > 1, "the key never moved");
}

/// A key typed into the settings is a key asked for.
#[test]
fn a_chosen_key_stays_put() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    a.formula_random_key = false;
    a.formula_key_setting = "Ab".to_string();
    for _ in 0..10 {
        a.rekey_formula();
        assert_eq!(a.formula_key_name, "Ab", "the key wandered off the chosen one");
    }
}

/// The arrows are the only way to ask for a different formula by hand, so
/// they have to draw one whichever way they point.
#[test]
fn the_arrows_draw_another_formula() {
    let mut a = app();
    a.set_mode(AppMode::Formulas as i32);
    let mut masks = std::collections::HashSet::new();
    for i in 0..40 {
        masks.insert(a.formula_mask);
        a.step_chord(if i % 2 == 0 { 1 } else { -1 });
        assert_eq!(
            a.formula_collected.len(),
            a.formula_mask.count_ones() as usize,
            "the marks do not match the formula drawn"
        );
    }
    assert!(masks.len() > 1, "the arrows never changed the formula");
}
