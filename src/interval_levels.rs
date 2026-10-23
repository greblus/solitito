//! The exercises of Intervals: which tones of each chord, and in what order.
//!
//! A level is written once per kind of chord, because the same idea lands on
//! different tones: the four-note cell from the root is 1 2 3 5 over a major or
//! a dominant chord and 1 3 4 5 over a minor one, where the second would rub
//! against the minor third. Tokens are read against the chord's scale - see
//! `ChordQuality::scale_names` - so `3` and `7` are whatever third and seventh
//! the chord has, and `2`, `4`, `6` and `9` come from its scale.

use crate::model::{split_octave, with_octave, ChordQuality};

/// The headings the levels are listed under, in order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Group {
    OneTone,
    TwoTones,
    ThreeTones,
    FourTones,
    RootToFifth,
    FifthToNinth,
    VoiceLeading,
    Suspensions,
}

pub const GROUPS: usize = 8;

/// One exercise.
pub struct Level {
    pub group: Group,
    /// Over a major seventh, a dominant seventh, a minor seventh, a
    /// half-diminished and a diminished chord.
    cells: [&'static str; 5],
    /// Played as written. A set of chord tones may come in any order; a cell
    /// or a line through the changes is that order or it is not the exercise.
    pub ordered: bool,
    /// The shuffle may deal the tones in another order: every set, and the
    /// first cell of each structure. The cells after it are its other orders,
    /// written out as levels of their own; voice leading would stop joining one
    /// chord to the next, a suspension would have nothing to resolve, and a
    /// level that already turns or climbs past the octave would lose its shape.
    pub shuffles: bool,
    /// Started from a tone drawn for every chord, the rest following round.
    pub rotated: bool,
}

const fn same(cell: &'static str) -> [&'static str; 5] {
    [cell; 5]
}

const fn tones(cell: &'static str) -> Level {
    set(Group::OneTone, cell)
}

const fn set(group: Group, cell: &'static str) -> Level {
    Level { group, cells: same(cell), ordered: false, shuffles: true, rotated: false }
}

const fn line(group: Group, cells: [&'static str; 5]) -> Level {
    Level { group, cells, ordered: true, shuffles: false, rotated: false }
}

/// The first cell of a structure: played in order, dealt in another one by
/// the shuffle.
const fn dealt(group: Group, cells: [&'static str; 5]) -> Level {
    Level { group, cells, ordered: true, shuffles: true, rotated: false }
}

const fn turned(group: Group, cells: [&'static str; 5]) -> Level {
    Level { group, cells, ordered: true, shuffles: false, rotated: true }
}

/// Major and dominant, minor and half-diminished share their cells; the
/// diminished chord gets its own.
const fn by_kind(major: &'static str, minor: &'static str, diminished: &'static str) -> [&'static str; 5] {
    [major, major, minor, minor, diminished]
}

use Group::*;

pub const LEVELS: [Level; 39] = [
    tones("1"),
    tones("3"),
    tones("5"),
    tones("7"),
    set(TwoTones, "1 3"),
    set(TwoTones, "1 5"),
    set(TwoTones, "1 7"),
    set(TwoTones, "3 5"),
    set(TwoTones, "3 7"),
    set(ThreeTones, "1 3 5"),
    set(ThreeTones, "1 3 7"),
    set(ThreeTones, "3 5 7"),
    turned(ThreeTones, same("1 3 5")),
    set(FourTones, "1 3 5 7"),
    line(FourTones, same("3 5 7 1'")),
    line(FourTones, same("5 7 1' 3'")),
    line(FourTones, same("7 1' 3' 5'")),
    turned(FourTones, same("1 3 5 7")),
    line(FourTones, same("1 3 5 7 1'")),
    line(FourTones, same("3 5 7 1' 3'")),
    line(FourTones, same("5 7 1' 3' 5'")),
    line(FourTones, same("7 1' 3' 5' 7'")),
    // Four notes of the chord's pentatonic sound, in four orders and turned.
    dealt(RootToFifth, by_kind("1 2 3 5", "1 3 4 5", "1 2 3 5")),
    line(RootToFifth, by_kind("2 1 5 3", "3 1 5 4", "2 1 5 3")),
    line(RootToFifth, by_kind("3 5 2 1", "4 5 3 1", "3 5 2 1")),
    line(RootToFifth, by_kind("5 1 2 3", "5 1 3 4", "5 1 2 3")),
    turned(RootToFifth, by_kind("1 2 3 5", "1 3 4 5", "1 2 3 5")),
    // The same shapes built on the fifth: the hand plays what it knows and
    // the ear hears the ninth, the sixth and the seventh.
    dealt(FifthToNinth, by_kind("5 6 7 9", "5 7 1' 9", "5 b6 7 9")),
    line(FifthToNinth, by_kind("6 5 9 7", "7 5 9 1'", "b6 5 9 7")),
    line(FifthToNinth, by_kind("7 9 6 5", "1' 9 7 5", "7 9 b6 5")),
    line(FifthToNinth, by_kind("9 7 6 5", "9 1' 7 5", "9 7 b6 5")),
    turned(FifthToNinth, by_kind("5 6 7 9", "5 7 1' 9", "5 b6 7 9")),
    // Lines for II-V-I and other cycles: each chord's cell ends a step from
    // where the next one starts - the seventh falling to the next third.
    line(VoiceLeading, same("3 2 1 7,")),
    line(VoiceLeading, ["5, 1 2 3", "3 5 7 9", "3 2 1 7,", "3 2 1 7,", "3 2 1 7,"]),
    line(VoiceLeading, ["3 2 1 7,", "3 2 1 7,", "1 3 5 7", "1 3 5 7", "3 2 1 7,"]),
    line(VoiceLeading, ["3 2 1 7,", "3 5 7 9", "1 3 5 7", "1 3 5 7", "3 2 1 7,"]),
    line(VoiceLeading, ["3 2 1 7,", "3 5 7 9", "5 3 1 7,", "5 3 1 7,", "3 2 1 7,"]),
    line(Suspensions, same("2 3 1")),
    line(Suspensions, same("4 3 1")),
];

/// The level a new install starts on: the triad, as the text field used to.
pub const DEFAULT: usize = 9;

impl Level {
    /// The tokens over `quality`, turned to start `rotation` tones in for a
    /// rotated level. A tone carried round the end goes up an octave, so the
    /// line keeps climbing.
    pub fn cell(&self, quality: &ChordQuality, rotation: usize) -> Vec<String> {
        let tokens: Vec<&str> = self.cells[kind(quality)].split_whitespace().collect();
        if !self.rotated {
            return tokens.iter().map(|t| t.to_string()).collect();
        }
        let start = rotation % tokens.len();
        tokens[start..]
            .iter()
            .map(|t| t.to_string())
            .chain(tokens[..start].iter().map(|t| {
                let (name, octave) = split_octave(t);
                with_octave(name, octave + 1)
            }))
            .collect()
    }
}

/// How the levels read in the list, in its order. A level is named by its
/// group and the major chord's cell - over other chords the tones follow the
/// chord, and the exercise shows them as it is played. Voice leading is
/// numbered instead: its cells differ between chords, and over a major chord
/// four of the five are the same.
pub fn labels(groups: &[&str; GROUPS]) -> Vec<String> {
    LEVELS
        .iter()
        .enumerate()
        .map(|(index, level)| {
            let group = groups[level.group as usize];
            if level.group == VoiceLeading {
                let nth = LEVELS[..=index].iter().filter(|l| l.group == VoiceLeading).count();
                return format!("{group} {nth}");
            }
            let turn = if level.rotated { " ↻" } else { "" };
            format!("{group}: {}{turn}", level.cells[0])
        })
        .collect()
}

/// Which of a level's five cells a chord takes. A scale played as a chord is
/// read as a major one.
fn kind(quality: &ChordQuality) -> usize {
    match quality {
        ChordQuality::Major7 | ChordQuality::CustomScale(_) => 0,
        ChordQuality::Dominant7 => 1,
        ChordQuality::Minor7 => 2,
        ChordQuality::HalfDiminished => 3,
        ChordQuality::Diminished => 4,
        // Minor chords both: the third is the minor one, and that is what
        // decides which cell fits.
        ChordQuality::Minor6 | ChordQuality::MinorMajor7 => 2,
    }
}
