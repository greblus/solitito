use super::tests::*;
use super::*;
use crate::interval_levels::LEVELS;

fn chord(root: NoteName, quality: ChordQuality) -> Chord {
    Chord { root, quality }
}

/// The pitch classes a level asks for over `chord`, in play order.
fn played(a: &MyApp, chord: &Chord) -> Vec<usize> {
    let targets = a.degree_targets(chord);
    a.ordered_active_indices(chord).iter().map(|s| targets[s.degree]).collect()
}

fn on_level(index: usize) -> MyApp {
    let mut a = app();
    a.app_mode = AppMode::Intervals;
    a.interval_level = Some(index);
    a.level_rotation = 0;
    a
}

fn level_of(group: crate::interval_levels::Group, nth: usize) -> usize {
    LEVELS.iter().enumerate().filter(|(_, l)| l.group == group).nth(nth).map(|(i, _)| i).unwrap()
}

use crate::interval_levels::Group;
use NoteName::*;

const C_: usize = 0;
const D_: usize = 2;
const EB: usize = 3;
const E_: usize = 4;
const F_: usize = 5;
const GB: usize = 6;
const G_: usize = 7;
const AB: usize = 8;
const A_: usize = 9;
const BB: usize = 10;
const B_: usize = 11;

#[test]
fn every_token_of_every_level_lands_on_every_chord() {
    let qualities = [
        ChordQuality::Major7,
        ChordQuality::Dominant7,
        ChordQuality::Minor7,
        ChordQuality::HalfDiminished,
        ChordQuality::Diminished,
    ];
    for (index, level) in LEVELS.iter().enumerate().filter(|(_, l)| l.run.is_none()) {
        let a = on_level(index);
        for quality in &qualities {
            let written = level.cell(quality, 0).len();
            let steps = a.get_active_indices(&chord(C, quality.clone()));
            assert_eq!(steps.len(), written, "level {index} lost a token over {}", quality.to_string());
        }
    }
}

#[test]
fn the_cell_from_the_root_follows_the_kind_of_chord() {
    let a = on_level(level_of(Group::RootToFifth, 0));
    assert_eq!(played(&a, &chord(C, ChordQuality::Major7)), [C_, D_, E_, G_]);
    assert_eq!(played(&a, &chord(G, ChordQuality::Dominant7)), [G_, A_, B_, D_]);
    assert_eq!(played(&a, &chord(D, ChordQuality::Minor7)), [D_, F_, G_, A_]);
    // Half-diminished with the natural ninth; its cell is the minor one.
    assert_eq!(played(&a, &chord(B, ChordQuality::HalfDiminished)), [B_, D_, E_, F_]);
    assert_eq!(played(&a, &chord(C, ChordQuality::Diminished)), [C_, D_, EB, GB]);
}

#[test]
fn the_cell_from_the_fifth_brings_the_ninth_and_the_sixth() {
    let a = on_level(level_of(Group::FifthToNinth, 0));
    assert_eq!(played(&a, &chord(C, ChordQuality::Major7)), [G_, A_, B_, D_]);
    assert_eq!(played(&a, &chord(C, ChordQuality::Dominant7)), [G_, A_, BB, D_]);
    assert_eq!(played(&a, &chord(C, ChordQuality::Minor7)), [G_, BB, C_, D_]);
    assert_eq!(played(&a, &chord(C, ChordQuality::Diminished)), [GB, AB, A_, D_]);
}

/// Over II-V-I every cell ends a step from where the next begins: the
/// seventh of one chord falls to the third of the next.
#[test]
fn lines_through_ii_v_i_move_by_step_between_chords() {
    let ii_v_i = [
        chord(D, ChordQuality::Minor7),
        chord(G, ChordQuality::Dominant7),
        chord(C, ChordQuality::Major7),
    ];
    let a = on_level(level_of(Group::VoiceLeading, 0));
    let lines: Vec<Vec<usize>> = ii_v_i.iter().map(|c| played(&a, c)).collect();
    assert_eq!(lines, [vec![F_, E_, D_, C_], vec![B_, A_, G_, F_], vec![E_, D_, C_, B_]]);
    for k in 0..5 {
        let a = on_level(level_of(Group::VoiceLeading, k));
        let lines: Vec<Vec<usize>> = ii_v_i.iter().map(|c| played(&a, c)).collect();
        for pair in lines.windows(2) {
            let from = *pair[0].last().unwrap() as i32;
            let to = pair[1][0] as i32;
            let distance = (to - from).rem_euclid(12).min((from - to).rem_euclid(12));
            assert!((1..=2).contains(&distance) || k >= 3, "line {k} leaps {distance} between chords");
        }
    }
}

#[test]
fn a_major_seventh_takes_the_lydian_fourth_but_a_suspension_the_plain_one() {
    let names = ChordQuality::Major7.scale_names();
    let intervals = ChordQuality::Major7.scale_intervals();
    let sharp_four = names.iter().position(|n| n == "#4").unwrap();
    assert_eq!(intervals[sharp_four], 6);
    let a = on_level(level_of(Group::Suspensions, 1));
    assert_eq!(played(&a, &chord(C, ChordQuality::Major7)), [F_, E_, C_]);
    assert_eq!(played(&a, &chord(C, ChordQuality::Minor7)), [F_, EB, C_]);
}

#[test]
fn a_rotated_level_starts_where_drawn_and_climbs_on() {
    let index = LEVELS.iter().position(|l| l.rotated && l.group == Group::ThreeTones).unwrap();
    let mut a = on_level(index);
    a.level_rotation = 1;
    let c = chord(C, ChordQuality::Major7);
    assert_eq!(played(&a, &c), [E_, G_, C_]);
    let octaves: Vec<i8> = a.ordered_active_indices(&c).iter().map(|s| s.octave).collect();
    assert_eq!(octaves, [0, 0, 1], "the root carried round did not go up");
}

/// The first cell of each structure is played in order but dealt in another
/// by the shuffle; the cells after it are those orders and stay as written.
#[test]
fn the_shuffle_deals_the_first_cell_of_each_structure_only() {
    let mut a = on_level(level_of(Group::RootToFifth, 0));
    a.interval_in_order = false;
    a.random_mode = true;
    a.step_order = vec![3, 2, 1, 0];
    assert!(!a.free_order());
    let c = chord(C, ChordQuality::Major7);
    assert_eq!(played(&a, &c), [G_, E_, D_, C_]);
    a.interval_level = Some(level_of(Group::FifthToNinth, 0));
    assert_eq!(played(&a, &c), [D_, B_, A_, G_]);
    for (group, nth) in [(Group::RootToFifth, 1), (Group::FifthToNinth, 1), (Group::VoiceLeading, 0), (Group::Suspensions, 1)] {
        a.interval_level = Some(level_of(group, nth));
        a.step_order = (0..a.get_active_indices(&c).len()).rev().collect();
        assert_eq!(a.ordered_active_indices(&c), a.get_active_indices(&c), "{group:?} {nth} was dealt");
    }
}

/// A cell is an order: the shuffle does not scatter it and the player cannot
/// take it in any order. A set of chord tones still can.
#[test]
fn a_cell_keeps_its_order_and_a_set_does_not_have_to() {
    let mut a = on_level(level_of(Group::RootToFifth, 1));
    a.interval_in_order = false;
    a.random_mode = true;
    a.step_order = vec![3, 2, 1, 0];
    assert!(!a.free_order());
    assert_eq!(played(&a, &chord(C, ChordQuality::Major7)), [D_, C_, G_, E_]);

    a.interval_level = Some(level_of(Group::FourTones, 0));
    assert!(a.free_order());
    assert_eq!(played(&a, &chord(C, ChordQuality::Major7)), [B_, G_, E_, C_]);
}

/// Outside Intervals, or with no level chosen, the chord's own four tones
/// are what a degree indexes, as before.
#[test]
fn without_a_level_the_degrees_are_the_chords_own() {
    let mut a = app();
    a.app_mode = AppMode::Intervals;
    a.interval_level = None;
    a.intervals_input = "1 3 5".into();
    let c = chord(C, ChordQuality::Major7);
    assert_eq!(a.degree_names(&c).len(), 4);
    assert_eq!(played(&a, &c), [C_, E_, G_]);
    a.interval_level = Some(0);
    a.app_mode = AppMode::Scales;
    assert!(a.level().is_none());
}

#[test]
fn the_list_names_a_level_by_its_major_cell_and_voice_leading_by_number() {
    let t = crate::i18n::strings(crate::i18n::Lang::Pl);
    let labels = crate::interval_levels::labels(&crate::interval_levels::Names {
        groups: &t.level_groups,
        any_chord_tone: t.level_any_chord_tone,
        any_scale_tone: t.level_any_scale_tone,
    });
    assert_eq!(labels.len(), LEVELS.len());
    assert!(labels.contains(&"Struktura 5–9: 6 5 9 7".to_string()));
    assert!(labels.contains(&"Trzy dźwięki: 1 3 5 ↻".to_string()));
    assert!(labels.contains(&"Prowadzenie głosów 5".to_string()));
    assert!(!labels.iter().any(|l| l.contains("Δ7")));
}

/// The standards' m6 and mMaj7 were read as dominant sevenths - a major
/// third where the chord has a minor one.
#[test]
fn the_songs_keep_their_minor_sixths_and_minor_major_sevenths() {
    let chords: Vec<ChordQuality> = crate::model::load_songs()
        .into_iter()
        .flat_map(|s| s.chords.into_iter().map(|c| c.quality))
        .collect();
    assert!(chords.contains(&ChordQuality::Minor6));
    assert!(chords.contains(&ChordQuality::MinorMajor7));
}

#[test]
fn a_minor_sixth_and_a_minor_major_seventh_take_the_minor_cells() {
    let a = on_level(level_of(Group::RootToFifth, 0));
    assert_eq!(played(&a, &chord(C, ChordQuality::Minor6)), [C_, EB, F_, G_]);
    assert_eq!(played(&a, &chord(C, ChordQuality::MinorMajor7)), [C_, EB, F_, G_]);
    // "7" over a sixth chord is its sixth; over mMaj7 the major seventh.
    let a = on_level(level_of(Group::FourTones, 0));
    assert_eq!(played(&a, &chord(C, ChordQuality::Minor6)), [C_, EB, G_, A_]);
    assert_eq!(played(&a, &chord(C, ChordQuality::MinorMajor7)), [C_, EB, G_, B_]);
}

/// The chord model names neither chord: the triad passes in part, and a m6
/// heard as the m7b5 a minor third below - the same four notes - in full.
#[test]
fn chords_mode_hears_a_minor_sixth_through_what_the_model_can_name() {
    use crate::state::chords_mode::{chord_reading, ChordReading};
    let a6 = chord(A, ChordQuality::Minor6);
    assert_eq!(chord_reading(&a6, Fsh, "m7b5", false), ChordReading::Exact);
    assert_eq!(chord_reading(&a6, A, "m", false), ChordReading::Partial);
    assert_eq!(chord_reading(&chord(C, ChordQuality::MinorMajor7), C, "m", false), ChordReading::Partial);
    assert_eq!(chord_reading(&a6, A, "7", false), ChordReading::Wrong);
}

/// Voice leading with the shuffle on: the chord starts on its tone nearest
/// the last note played, a step away and never the note itself.
#[test]
fn voice_leading_starts_the_chord_a_step_from_the_last_note() {
    let mut a = on_level(level_of(Group::FourTones, 0));
    a.chords = vec![chord(G, ChordQuality::Dominant7)];
    a.current_chord_index = 0;
    a.random_mode = true;
    a.lead_line = true;
    for (last, wanted) in [(C_, B_), (G_, F_), (EB, D_)] {
        for _ in 0..20 {
            a.last_played = Some(last);
            a.reroll();
            assert_eq!(played(&a, &a.chords[0].clone())[0], wanted, "after {last}");
        }
    }
    // Off, the first tone is whatever the shuffle drew.
    a.lead_line = false;
    let firsts: std::collections::HashSet<usize> = (0..60)
        .map(|_| {
            a.last_played = Some(C_);
            a.reroll();
            played(&a, &a.chords[0].clone())[0]
        })
        .collect();
    assert!(firsts.len() > 1);
}

const DB: usize = 1;
const FS: usize = 6;
const AB_: usize = 8;

#[test]
fn a_chord_scale_runs_an_octave_from_where_its_level_says() {
    let a = on_level(level_of(Group::ChordScales, 0));
    // Lydian over a major seventh.
    assert_eq!(played(&a, &chord(C, ChordQuality::Major7)), [C_, D_, E_, FS, G_, A_, B_, C_]);
    let a = on_level(level_of(Group::ChordScales, 1));
    assert_eq!(played(&a, &chord(G, ChordQuality::Dominant7)), [B_, C_, D_, E_, F_, G_, A_, B_]);
    // Locrian with a natural ninth.
    let a = on_level(level_of(Group::ChordScales, 0));
    assert_eq!(played(&a, &chord(B, ChordQuality::HalfDiminished)), [B_, DB, D_, E_, F_, G_, A_, B_]);
    let octaves: Vec<i8> = a.ordered_active_indices(&chord(C, ChordQuality::Major7)).iter().map(|s| s.octave).collect();
    assert_eq!(octaves, [0, 0, 0, 0, 0, 0, 0, 1]);
}

/// The passing note makes eight notes to the octave, and from a chord tone
/// every chord tone lands on every other note - on the beat.
#[test]
fn the_passing_note_puts_the_chord_on_the_beat() {
    let a = on_level(level_of(Group::PassingNoteScales, 0));
    assert_eq!(played(&a, &chord(G, ChordQuality::Dominant7)), [G_, A_, B_, C_, D_, E_, F_, FS, G_]);
    assert_eq!(played(&a, &chord(C, ChordQuality::Major7)), [C_, D_, E_, FS, G_, AB_, A_, B_, C_]);
    // The diminished chord's passing note completes the half-whole scale.
    assert_eq!(played(&a, &chord(C, ChordQuality::Diminished)), [C_, D_, EB, F_, GB, AB_, A_, B_, C_]);
    for quality in [ChordQuality::Dominant7, ChordQuality::Minor7, ChordQuality::HalfDiminished, ChordQuality::Diminished] {
        let c = chord(D, quality);
        let chord_tones = c.get_target_indices();
        let line = played(&a, &c);
        for (k, pc) in line.iter().enumerate() {
            assert_eq!(chord_tones.contains(pc), k % 2 == 0, "{} note {k} off the beat", c.quality.to_string());
        }
    }
    let a = on_level(level_of(Group::PassingNoteScales, 1));
    assert_eq!(played(&a, &chord(D, ChordQuality::Minor7)), [F_, G_, A_, B_, C_, DB, D_, E_, F_]);
}

#[test]
fn a_run_from_a_drawn_tone_starts_on_a_chord_tone_or_a_scale_tone() {
    let c = chord(C, ChordQuality::Major7);
    let mut a = on_level(level_of(Group::ChordScales, 4));
    for rotation in 0..8 {
        a.level_rotation = rotation;
        let line = played(&a, &c);
        assert!(c.get_target_indices().contains(&line[0]));
        assert_eq!(line.first(), line.last());
    }
    let mut a = on_level(level_of(Group::ChordScales, 5));
    let starts: std::collections::HashSet<usize> = (0..7)
        .map(|rotation| {
            a.level_rotation = rotation;
            played(&a, &c)[0]
        })
        .collect();
    assert_eq!(starts.len(), 7, "every tone of the scale can start the run");
}
