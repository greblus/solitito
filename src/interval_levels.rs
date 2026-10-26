//! The exercises of Intervals: which tones of each chord, and in what order.
//!
//! A level is written once per kind of chord, because the same idea lands on
//! different tones: the four-note cell from the root is 1 2 3 5 over a major or
//! a dominant chord and 1 3 4 5 over a minor one, where the second would rub
//! against the minor third. Tokens are read against the chord's scale - see
//! `ChordQuality::scale_names` - so `3` and `7` are whatever third and seventh
//! the chord has, and `2`, `4`, `6` and `9` come from its scale.

use crate::model::{split_octave, with_octave, ChordQuality, Step};

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
    ChordScales,
    PassingNoteScales,
}

pub const GROUPS: usize = 10;

/// Where a scale run starts: on a tone of the chord, or anywhere in its scale.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum From {
    Root,
    Third,
    Fifth,
    Seventh,
    /// A chord tone drawn for every chord.
    AnyChordTone,
    /// A tone of the scale drawn for every chord.
    AnyScaleTone,
}

/// A level that is the chord's scale run through an octave rather than a
/// written cell - see `ChordQuality::scale_in_order`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Run {
    pub from: From,
    /// With the passing note, so that started on a chord tone, the chord's
    /// tones fall on every other note.
    pub passing: bool,
}

/// How the list names the levels: the group headings and the two words a
/// scale run drawn from anywhere needs.
pub struct Names<'a> {
    pub groups: &'a [&'a str; GROUPS],
    pub any_chord_tone: &'a str,
    pub any_scale_tone: &'a str,
}

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
    /// A scale run instead of the cells.
    pub run: Option<Run>,
}

const fn same(cell: &'static str) -> [&'static str; 5] {
    [cell; 5]
}

const fn tones(cell: &'static str) -> Level {
    set(Group::OneTone, cell)
}

const fn set(group: Group, cell: &'static str) -> Level {
    Level { group, cells: same(cell), ordered: false, shuffles: true, rotated: false, run: None }
}

const fn line(group: Group, cells: [&'static str; 5]) -> Level {
    Level { group, cells, ordered: true, shuffles: false, rotated: false, run: None }
}

/// The first cell of a structure: played in order, dealt in another one by
/// the shuffle.
const fn dealt(group: Group, cells: [&'static str; 5]) -> Level {
    Level { group, cells, ordered: true, shuffles: true, rotated: false, run: None }
}

const fn turned(group: Group, cells: [&'static str; 5]) -> Level {
    Level { group, cells, ordered: true, shuffles: false, rotated: true, run: None }
}

const fn scale(group: Group, from: From) -> Level {
    let passing = matches!(group, PassingNoteScales);
    Level { group, cells: same(""), ordered: true, shuffles: false, rotated: false, run: Some(Run { from, passing }) }
}

/// Major and dominant, minor and half-diminished share their cells; the
/// diminished chord gets its own.
const fn by_kind(major: &'static str, minor: &'static str, diminished: &'static str) -> [&'static str; 5] {
    [major, major, minor, minor, diminished]
}

use Group::*;

pub const LEVELS: [Level; 50] = [
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
    // The chord's scale through an octave, from where its name says.
    scale(ChordScales, From::Root),
    scale(ChordScales, From::Third),
    scale(ChordScales, From::Fifth),
    scale(ChordScales, From::Seventh),
    scale(ChordScales, From::AnyChordTone),
    scale(ChordScales, From::AnyScaleTone),
    // And with the passing note: played in eighths from a downbeat, every
    // tone of the chord lands on a beat.
    scale(PassingNoteScales, From::Root),
    scale(PassingNoteScales, From::Third),
    scale(PassingNoteScales, From::Fifth),
    scale(PassingNoteScales, From::Seventh),
    scale(PassingNoteScales, From::AnyChordTone),
];

/// The level a new install starts on: the triad, as the text field used to.
pub const DEFAULT: usize = 9;

impl Level {
    /// A scale run's steps over `quality`: from its starting tone up through
    /// the octave to the same tone again. `rotation` picks the tone where the
    /// level draws one.
    pub fn run_steps(&self, quality: &ChordQuality, rotation: usize) -> Option<Vec<Step>> {
        let run = self.run?;
        let line = quality.scale_in_order(run.passing);
        let start_degree = match run.from {
            From::Root => 0,
            From::Third => 1,
            From::Fifth => 2,
            From::Seventh => 3,
            From::AnyChordTone => rotation % 4,
            From::AnyScaleTone => line[rotation % line.len()],
        };
        let start = line.iter().position(|&d| d == start_degree).unwrap_or(0);
        Some(
            (0..=line.len())
                .map(|k| Step { degree: line[(start + k) % line.len()], octave: ((start + k) / line.len()) as i8 })
                .collect(),
        )
    }

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
/// group and the major chord's cell, and the minor and diminished chords'
/// cells where they differ. Voice leading is
/// numbered instead: its cells differ between chords, and over a major chord
/// four of the five are the same.
pub fn labels(names: &Names) -> Vec<String> {
    LEVELS
        .iter()
        .enumerate()
        .map(|(index, level)| {
            let group = names.groups[level.group as usize];
            if let Some(run) = level.run {
                let from = match run.from {
                    From::Root => "1",
                    From::Third => "3",
                    From::Fifth => "5",
                    From::Seventh => "7",
                    From::AnyChordTone => names.any_chord_tone,
                    From::AnyScaleTone => names.any_scale_tone,
                };
                return format!("{group}: {from}");
            }
            if level.group == VoiceLeading {
                let nth = LEVELS[..=index].iter().filter(|l| l.group == VoiceLeading).count();
                return format!("{group} {nth}");
            }
            let turn = if level.rotated { " ↻" } else { "" };
            // Where a minor or a diminished chord takes other tones, they are
            // named too: the list said 1 2 3 5 while the neck asked for the
            // minor chord's 1 b3 4 5.
            let (major, minor, diminished) = (level.cells[0], level.cells[2], level.cells[4]);
            let mut label = format!("{group}: {}", shown_cell(major));
            // Spelled as those chords have them, as the neck shows them: the
            // minor chord's third is b3.
            if minor != major {
                label += &format!(" · m: {}", spelled_cell(minor, &ChordQuality::Minor7));
            }
            if diminished != major && diminished != minor {
                label += &format!(" · °: {}", spelled_cell(diminished, &ChordQuality::Diminished));
            }
            label + turn
        })
        .collect()
}

/// A cell as the list writes it: plain degrees, the octave left out - a 9 is
/// the 2 and `1'` the 1. The judge hears which note and not in which octave,
/// and the grip on the neck takes each one where the hand has it, so a
/// compound number would ask for something nothing checks.
fn shown_cell(cell: &str) -> String {
    cell.split_whitespace()
        .map(|token| plain_degree(split_octave(token).0))
        .collect::<Vec<_>>()
        .join(" ")
}

/// A cell spelled over `quality`, plain: each token as the degree it asks of
/// that chord, read the way the judge reads it.
fn spelled_cell(cell: &str, quality: &ChordQuality) -> String {
    let names = quality.scale_names();
    cell.split_whitespace()
        .map(|token| {
            let name = split_octave(token).0;
            match crate::state::level_degree(name, &names) {
                Some(degree) => names[degree].clone(),
                None => plain_degree(name),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A degree brought inside the octave: `9` -> `2`, `b13` -> `b6`.
pub fn plain_degree(name: &str) -> String {
    let accidental = name.trim_end_matches(|c: char| c.is_ascii_digit());
    match name[accidental.len()..].parse::<u32>() {
        Ok(number) if number > 7 => format!("{accidental}{}", (number - 1) % 7 + 1),
        _ => name.to_string(),
    }
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
