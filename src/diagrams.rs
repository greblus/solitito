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
//! Colours are left exactly as generated. The lines and dots are blue
//! (`rgba(74,144,226)`) with a red root, which reads fine on the app's black
//! background; the only black element is a full-size background rect that
//! disappears against it.

use crate::model::ChordQuality;


/// One shape: which string carries the root, and the drawing itself.
pub struct Diagram {
    /// Root string the shape sits on, low to high. Not drawn any more - the
    /// captions were noise under a 96px thumbnail - but it documents the table
    /// and names the shape in test failures.
    #[allow(dead_code)]
    pub label: &'static str,
    pub svg: &'static str,
}

macro_rules! diagram {
    ($label:expr, $file:expr) => {
        Diagram { label: $label, svg: include_str!(concat!("chord_diagrams/", $file)) }
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

static DIM: &[Diagram] = &[
    diagram!("E", "dim7_E.svg"),
    diagram!("A", "dim7_A.svg"),
    diagram!("D", "dim7_D.svg"),
];

/// Shapes for a chord quality, or an empty slice when there are none.
///
/// Scales carry a `CustomScale` quality and have no shape, which is why the
/// caller gets a slice rather than an error - nothing to show is normal.
pub fn for_quality(quality: &ChordQuality) -> &'static [Diagram] {
    match quality {
        ChordQuality::Major7 => MAJ7,
        ChordQuality::Minor7 => MIN7,
        ChordQuality::Dominant7 => DOM7,
        ChordQuality::HalfDiminished => HALF_DIM,
        ChordQuality::Diminished => DIM,
        ChordQuality::CustomScale(_) => &[],
    }
}

#[cfg(test)]
mod tests {
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
        ] {
            let shapes = for_quality(&q);
            assert!(!shapes.is_empty(), "{:?} has no shapes", q.to_string());
            for d in shapes {
                assert!(d.svg.contains("<svg"), "{}: not an SVG", d.label);
                assert!(d.svg.contains("</svg>"), "{}: truncated", d.label);
                assert!(
                    d.svg.contains("finger-circle"),
                    "{} {}: no finger dots - wrong file?", q.to_string(), d.label
                );
            }
        }
    }

    /// The red dot is the only thing telling the player where the root sits.
    #[test]
    fn every_diagram_marks_the_root() {
        for q in [
            ChordQuality::Major7, ChordQuality::Minor7, ChordQuality::Dominant7,
            ChordQuality::HalfDiminished, ChordQuality::Diminished,
        ] {
            for d in for_quality(&q) {
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
        assert!(for_quality(&scale).is_empty());
    }

    /// Labels drive nothing but the caption, yet a wrong one would send the
    /// player to the wrong string.
    #[test]
    fn labels_are_root_strings() {
        for q in [ChordQuality::Major7, ChordQuality::Minor7, ChordQuality::Dominant7] {
            for d in for_quality(&q) {
                assert!(
                    ["E", "A", "D"].contains(&d.label),
                    "unexpected label {:?}", d.label
                );
            }
        }
    }
}

