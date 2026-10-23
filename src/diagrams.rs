//! Chord shape diagrams.
//!
//! The shapes are labelled with DEGREES, not finger numbers, so one diagram
//! covers all twelve keys - the shape is movable and the red dot marks the root.
//! That is the same idea the whole app runs on.
//!
//! The files are compiled into the binary rather than shipped in a directory.
//! At 136 kB for the lot it costs nothing, and it removes a class of failure the
//! release packages would otherwise have: a missing or half-copied asset folder.
//! Same reasoning as the compiled-in UI strings in `i18n`.
//!
//! Colours are left exactly as generated - blue lines and dots
//! (`rgba(74,144,226)`) with a red root - except the one full-size background
//! rect, black in the files, which is drawn in `CARD` so a shape sits on the
//! same card as the rest of the UI.

use crate::model::ChordQuality;


/// The card every drawing sits on: the colour of the cards in the UI
/// (`#161A22` in `appwindow.slint`).
pub const CARD: &str = "rgba(22, 26, 34, 1)";

/// One shape: which string carries the root, and the drawing itself.
pub struct Diagram {
    /// Root string the shape sits on, low to high. Not drawn any more - the
    /// captions were noise under a 96px thumbnail - but it documents the table
    /// and names the shape in test failures.
    pub label: &'static str,
    pub svg: &'static str,
}

impl Diagram {
    /// The drawing on the UI's card instead of the black it was generated on.
    /// The generator's half-black credit line goes with the background, so it
    /// stays as invisible as it was on black.
    pub fn on_card(&self) -> String {
        let card = format!("fill=\"{CARD}\"");
        let svg = self
            .svg
            .replace("fill=\"rgba(0, 0, 0, 1)\"", &card)
            .replace("fill=\"rgba(0, 0, 0, 0.5)\"", &card);
        crate::tab::with_accidentals_font(&svg)
    }
}

macro_rules! diagram {
    ($label:expr, $file:expr) => {
        Diagram { label: $label, svg: include_str!(concat!("assets/chord_diagrams/", $file)) }
    };
}

// Ordered low string to high, which is how a guitarist scans for a shape.
// Where a quality has two shapes off the same string they follow each other.

static MAJ7: &[Diagram] = &[
    diagram!("E", "maj7_E.svg"),
    diagram!("A", "maj7_A.svg"),
    diagram!("D", "maj7_D.svg"),
];

static MIN7: &[Diagram] = &[
    diagram!("E", "m7_E.svg"),
    diagram!("A", "m7_A_1.svg"),
    diagram!("A", "m7_A_2.svg"),
    diagram!("D", "m7_D.svg"),
];

static DOM7: &[Diagram] = &[
    diagram!("E", "7_E.svg"),
    diagram!("A", "7_A_1.svg"),
    diagram!("A", "7_A_2.svg"),
    diagram!("D", "7_D.svg"),
];

static HALF_DIM: &[Diagram] = &[
    diagram!("E", "m7b5_E.svg"),
    diagram!("A", "m7b5_A.svg"),
    diagram!("D", "m7b5_D.svg"),
];

// Drawn by dist/make_chord_diagrams.py in the style of the others.
static MIN6: &[Diagram] = &[
    diagram!("E", "m6_E.svg"),
    diagram!("A", "m6_A.svg"),
    diagram!("D", "m6_D.svg"),
];

static MIN_MAJ7: &[Diagram] = &[
    diagram!("E", "mmaj7_E.svg"),
    diagram!("A", "mmaj7_A.svg"),
    diagram!("D", "mmaj7_D.svg"),
];

static DIM: &[Diagram] = &[
    diagram!("E", "dim7_E.svg"),
    diagram!("A", "dim7_A.svg"),
    diagram!("D", "dim7_D.svg"),
];

// Shell voicings: the third and the seventh, with the root, and no fifth. They
// answer a different question from the grips above - not "how does the whole
// chord sound" but "what do I play to accompany without getting in the way" -
// which is why they are a choice rather than more entries in the same row. Six
// shapes beside four would also leave every thumbnail too small to read the
// fret off.

static S_MAJ7: &[Diagram] = &[
    diagram!("E", "s_maj7_E.svg"),
    diagram!("E", "s_maj7_E_1.svg"),
    diagram!("A", "s_maj7_A.svg"),
    diagram!("A", "s_maj7_A_1.svg"),
    diagram!("D", "s_maj7_D.svg"),
    diagram!("D", "s_maj7_D_1.svg"),
];

static S_MIN7: &[Diagram] = &[
    diagram!("E", "s_m7_E.svg"),
    diagram!("E", "s_m7_E_1.svg"),
    diagram!("A", "s_m7_A.svg"),
    diagram!("A", "s_m7_A_1.svg"),
    diagram!("D", "s_m7_D.svg"),
    diagram!("D", "s_m7_D_1.svg"),
];

static S_DOM7: &[Diagram] = &[
    diagram!("E", "s_7_E.svg"),
    diagram!("E", "s_7_E_1.svg"),
    diagram!("A", "s_7_A.svg"),
    diagram!("A", "s_7_A_1.svg"),
    diagram!("D", "s_7_D.svg"),
    diagram!("D", "s_7_D_1.svg"),
];

/// Two rows in columns by root string, when the shapes come in pairs off the
/// same string - the shells, E E A A D D. The plain split put E E A over
/// A D D, and the eye had to find which shape belonged to which string. Here
/// the first of each pair goes on top and the second under it, so a column is
/// one string. `None` when the shapes do not pair up.
pub fn by_root_string<T: Clone>(shapes: &[(&'static str, T)]) -> Option<(Vec<T>, Vec<T>)> {
    if shapes.len() < 4 || shapes.len() % 2 != 0 {
        return None;
    }
    let pairs: Vec<_> = shapes.chunks(2).collect();
    if pairs.iter().any(|p| p[0].0 != p[1].0) {
        return None;
    }
    let top = pairs.iter().map(|p| p[0].1.clone()).collect();
    let bottom = pairs.iter().map(|p| p[1].1.clone()).collect();
    Some((top, bottom))
}

/// Which set of shapes to draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shapes {
    /// The full grip, root and all.
    Full,
    /// Third and seventh over the root - and nothing for the qualities that
    /// have no shell of their own, which is honest: a diminished seventh is
    /// four notes a minor third apart and has nothing to leave out.
    Shell,
    /// Both, the full grips first. Ten thumbnails in a row that holds four
    /// comfortably, so this is for comparing the two, not for playing from.
    Both,
}

fn full(quality: &ChordQuality) -> &'static [Diagram] {
    match quality {
        ChordQuality::Major7 => MAJ7,
        ChordQuality::Minor7 => MIN7,
        ChordQuality::Dominant7 => DOM7,
        ChordQuality::HalfDiminished => HALF_DIM,
        ChordQuality::Diminished => DIM,
        ChordQuality::Minor6 => MIN6,
        ChordQuality::MinorMajor7 => MIN_MAJ7,
        ChordQuality::CustomScale(_) => &[],
    }
}

/// Shells where they exist; where they do not, the full grip.
///
/// A m7b5 and a dim7 have no shell of their own worth drawing. The shell of a
/// m7b5 is root, third and seventh - which is the m7 shell exactly, since the
/// two chords differ only in the fifth the shell leaves out - and a diminished
/// seventh has nothing to leave out at all. Rather than an empty row, the mode
/// falls back to the chord's own shapes: something to play that the app also
/// accepts, which the m7 shell in that position would not be.

/// What the shell row is, when it is not the chord's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellNote {
    /// The shapes belong to the chord being asked for.
    Own,
    /// A m7 shell standing in for a m7b5. Root, third and seventh are the same
    /// three notes in both - a shell leaves out the fifth, which is the only
    /// place those two chords differ - so this is the substitution a player
    /// makes anyway, and the application accepts it as a partial match.
    MinorForHalfDim,
    /// No simpler form exists: a diminished seventh is four notes a minor third
    /// apart and has nothing to leave out.
    None_,
    /// The chord model has no name for this chord, so what it accepts is a
    /// substitute: the minor triad, and for a m6 also the m7b5 a minor third
    /// below, which is the same four notes. Said under every row, full or
    /// shell, since it is about what the app will hear rather than the shape.
    HeardAsMinor,
}

/// Which of the three the shell row is, for whoever has to caption it. A player
/// told nothing would take a substitute for the chord itself.
pub fn shell_note(quality: &ChordQuality) -> ShellNote {
    match quality {
        ChordQuality::Major7 | ChordQuality::Minor7 | ChordQuality::Dominant7 => ShellNote::Own,
        ChordQuality::HalfDiminished => ShellNote::MinorForHalfDim,
        ChordQuality::Minor6 | ChordQuality::MinorMajor7 => ShellNote::HeardAsMinor,
        _ => ShellNote::None_,
    }
}

fn shell(quality: &ChordQuality) -> &'static [Diagram] {
    match quality {
        ChordQuality::Major7 => S_MAJ7,
        ChordQuality::Minor7 => S_MIN7,
        ChordQuality::Dominant7 => S_DOM7,
        // The m7 shell IS the m7b5 shell, note for note. Drawing the full grip
        // instead would answer a mode meant to simplify with the harder shape.
        ChordQuality::HalfDiminished => S_MIN7,
        other => full(other),
    }
}

/// Shapes for a chord quality, or nothing when there are none.
///
/// Scales carry a `CustomScale` quality and have no shape, which is why the
/// caller gets a list rather than an error - nothing to show is normal. The
/// same holds for the shell of a chord that has none.
pub fn for_quality(quality: &ChordQuality, shapes: Shapes) -> Vec<&'static Diagram> {
    match shapes {
        Shapes::Full => full(quality).iter().collect(),
        Shapes::Shell => shell(quality).iter().collect(),
        // Where the fallback IS the full grip, one row of it is enough.
        // Where the fallback IS the full grip, one row of it is enough.
        Shapes::Both if shell_note(quality) == ShellNote::None_ => {
            full(quality).iter().collect()
        }
        Shapes::Both => full(quality).iter().chain(shell(quality)).collect(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn shells_go_in_columns_by_root_string() {
        let shells: Vec<(&str, &str)> = S_MIN7.iter().map(|d| (d.label, d.svg)).collect();
        let (top, bottom) = by_root_string(&shells).expect("the shells pair up");
        // Top row the first shape off E, A and D; under each, the second.
        assert_eq!(top, vec![S_MIN7[0].svg, S_MIN7[2].svg, S_MIN7[4].svg]);
        assert_eq!(bottom, vec![S_MIN7[1].svg, S_MIN7[3].svg, S_MIN7[5].svg]);
        // The full grips are one per string and stay as they are.
        let full: Vec<(&str, &str)> = MAJ7.iter().map(|d| (d.label, d.svg)).collect();
        assert!(by_root_string(&full).is_none());
    }

    use super::*;

    /// include_str! would fail the build on a missing file, so reaching here
    /// means every path resolved. This checks the content is a real drawing.
    #[test]
    fn every_diagram_is_a_usable_svg() {
        for q in [
            ChordQuality::Major7,
            ChordQuality::Minor7,
            ChordQuality::Dominant7,
            ChordQuality::HalfDiminished,
            ChordQuality::Diminished,
            ChordQuality::Minor6,
            ChordQuality::MinorMajor7,
        ] {
            for kind in [Shapes::Full, Shapes::Shell, Shapes::Both] {
                let shapes = for_quality(&q, kind);
                assert!(
                    !shapes.is_empty(),
                    "{:?} has no shapes as {:?}", q.to_string(), kind
                );
                for d in &shapes {
                    assert!(d.svg.contains("<svg"), "{}: not an SVG", d.label);
                    assert!(d.svg.contains("</svg>"), "{}: truncated", d.label);
                    assert!(
                        d.svg.contains("finger-circle"),
                        "{} {}: no finger dots - wrong file?", q.to_string(), d.label
                    );
                }
            }
        }
    }

    /// The red dot is the only thing telling the player where the root sits.
    #[test]
    fn every_diagram_marks_the_root() {
        for q in [
            ChordQuality::Major7, ChordQuality::Minor7, ChordQuality::Dominant7,
            ChordQuality::HalfDiminished, ChordQuality::Diminished,
            ChordQuality::Minor6, ChordQuality::MinorMajor7,
        ] {
            for d in for_quality(&q, Shapes::Both) {
                assert!(
                    d.svg.contains("rgba(208, 2, 27, 1)"),
                    "{} {}: no red root dot", q.to_string(), d.label
                );
            }
        }
    }

    /// Scales have no shape and must not blow up on the way past.
    #[test]
    fn custom_scales_have_no_shapes() {
        let scale = ChordQuality::CustomScale(crate::model::ScaleDefinition {
            name: "test".into(), intervals: vec![0, 2, 4], names: vec!["1".into()],
        });
        assert!(for_quality(&scale, Shapes::Full).is_empty());
        assert!(for_quality(&scale, Shapes::Both).is_empty());
    }

    /// The random-order hint names a string to start from, and every string it
    /// can name has to have a shape behind it. It offered the G string while the
    /// shapes only cover E, A and D, which read as the diagrams being wrong.
    #[test]
    fn every_suggested_string_has_shapes() {
        let labels: Vec<&str> = [
            ChordQuality::Major7,
            ChordQuality::Minor7,
            ChordQuality::Dominant7,
            ChordQuality::HalfDiminished,
            ChordQuality::Diminished,
        ]
        .iter()
        .flat_map(|q| for_quality(q, Shapes::Full).into_iter().map(|d| d.label))
        .collect();

        for s in crate::state::START_STRINGS {
            assert!(labels.contains(&s), "the hint can suggest {s}, which has no shape");
        }
    }

    /// Labels drive nothing but the caption, yet a wrong one would send the
    /// player to the wrong string.
    #[test]
    fn labels_are_root_strings() {
        for q in [ChordQuality::Major7, ChordQuality::Minor7, ChordQuality::Dominant7] {
            for d in for_quality(&q, Shapes::Both) {
                assert!(
                    ["E", "A", "D"].contains(&d.label),
                    "unexpected label {:?}", d.label
                );
            }
        }
    }
}
