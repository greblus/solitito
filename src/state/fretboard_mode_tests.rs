use super::tests::*;
use super::*;

/// The region is drawn once per entry into the mode, not per note - the
/// whole point is to stay in one hand position.
#[test]
fn region_holds_while_notes_change() {
    let mut a = app();
    a.set_mode(AppMode::Fretboard as i32);
    let region = a.region;
    let mut notes = std::collections::HashSet::new();
    for _ in 0..40 {
        notes.insert(a.fret_target.expect("a target must be drawn"));
        a.next_fret_target();
        assert_eq!(a.region.strings, region.strings, "the region moved mid-exercise");
        assert_eq!(a.region.fret_from, region.fret_from, "the region moved mid-exercise");
    }
    assert!(notes.len() > 1, "the note never changed");
}

#[test]
fn every_target_is_playable_in_the_region() {
    let mut a = app();
    for _ in 0..30 {
        a.set_mode(AppMode::Fretboard as i32);
        for _ in 0..30 {
            let pc = a.fret_target.expect("a target must be drawn");
            assert!(
                !a.region.positions_of(pc).is_empty(),
                "{:?}: asked for {pc}, not reachable there", a.region
            );
            a.next_fret_target();
        }
    }
}

/// A window running off the end of the neck would ask for notes that cannot
/// be played on a 15-fret range.
#[test]
fn random_region_stays_on_the_neck() {
    let mut a = app();
    for _ in 0..300 {
        a.randomize_region();
        assert!(
            a.region.fret_to() <= crate::fretboard::MAX_FRET,
            "region {:?} runs past fret {}", a.region, crate::fretboard::MAX_FRET
        );
        assert_eq!(a.region.fret_span, 4, "span should be one finger per fret");
    }
}

/// Entering the mode repeatedly must give different regions, otherwise the
/// randomisation is decorative.
#[test]
fn re_entering_the_mode_gives_a_new_region() {
    let mut a = app();
    let mut seen = std::collections::HashSet::new();
    for _ in 0..40 {
        a.set_mode(AppMode::Fretboard as i32);
        seen.insert((a.region.strings as i32, a.region.fret_from));
    }
    assert!(seen.len() > 3, "regions barely varied: {seen:?}");
}

/// The fretboard branch runs before the chord guard; with no chords loaded
/// the old code path would have returned early and the mode would be dead.
#[test]
fn progress_runs_without_any_chords() {
    let mut a = app();
    a.set_mode(AppMode::Fretboard as i32);
    assert!(a.chords.is_empty(), "the trainer should not load a song");
    let target = a.fret_target.unwrap();
    a.last_pitches = [0.0; 12];
    a.last_pitches[target] = 1.0;
    a.note_threshold = 0.5;
    // The trainer holds for a fixed 0.12 s, as the note modes do - the
    // chord delay is not its measure.
    a.check_progress_with_ai(0.1, "Noise", 0.0);
    assert_eq!(a.fret_target, Some(target), "credited before the hold was up");
    a.check_progress_with_ai(0.1, "Noise", 0.0);
    // Passed - and now SHOWN, green, before the next note is drawn.
    assert_eq!(a.match_status, MatchStatus::Exact, "the verdict was not shown");
    assert_eq!(a.fret_target, Some(target), "it moved on before anything was seen");
    run_show_clock(&mut a);
    assert_ne!(a.fret_target, Some(target), "a played note did not advance");
}

/// The show clock runs in `tick`, once per UI frame, so it cannot stop when
/// the model does. Sixty frames is a second - more than `FRET_SHOW`.
fn run_show_clock(a: &mut MyApp) {
    for _ in 0..60 {
        a.tick(0.016);
    }
}

/// The green has to survive long enough to be drawn. It used to be cleared
/// by `next_fret_target` in the same call that set it, so a note passed
/// inside one answer from the model was never green on screen at all.
#[test]
fn a_passed_note_is_shown_before_the_next_is_drawn() {
    let mut a = app();
    a.set_mode(AppMode::Fretboard as i32);
    let target = a.fret_target.unwrap();
    a.last_pitches = [0.0; 12];
    a.last_pitches[target] = 1.0;
    a.note_threshold = 0.5;
    // One answer from the model, long enough to pass the hold on its own -
    // which is the case that showed nothing.
    a.check_progress_with_ai(0.25, "Noise", 0.0);
    assert_eq!(a.match_status, MatchStatus::Exact);
    assert_eq!(a.fret_target, Some(target), "the verdict was gone in the same call");
    // And it stays through the frames that follow, model answering or not.
    a.tick(0.016);
    assert_eq!(a.match_status, MatchStatus::Exact, "the green did not last a frame");
    assert_eq!(a.fret_target, Some(target));
    run_show_clock(&mut a);
    assert_ne!(a.fret_target, Some(target), "the trainer stuck on the note it passed");
    assert_eq!(a.match_status, MatchStatus::None, "the next note came up green");
}
