//! Arpeggios drawn as tablature, with intervals in place of fret numbers.
//!
//! Six string lines and a dot per note, the degree written inside it: the shape
//! on the neck is then visible directly, and the octave a note sits in needs no
//! marker - it is which string the dot is on.
//!
//! Drawn in the same language as the chord diagrams in `diagrams`: black
//! ground, blue `rgba(74, 144, 226)` lines, a red root, white labels in Arial,
//! dots of the same radius. The two are read side by side, so they have to look
//! like one thing.
//!
//! WHERE the notes go is not a matter of taste here. A note is placed on the
//! lowest string on which it falls inside a four-fret box around the root - and
//! that one rule reproduces, note for note, the fingerings of the three
//! arpeggio studies this was checked against.

use crate::model::Step;

/// Standard tuning, low string first.
pub const TUNING: [i32; 6] = [40, 45, 50, 55, 59, 64];

/// Degree names as the chord diagrams spell them: proper flats and sharps.
const DEGREE: [&str; 12] = [
    "1", "♭2", "2", "♭3", "3", "4", "♭5", "5", "♭6", "6", "♭7", "7",
];

/// One note of the phrase, placed on the neck.
#[derive(Debug, Clone, PartialEq)]
pub struct Spot {
    /// 0 is the lowest string, 5 the highest - the order `TUNING` is in.
    pub string: usize,
    pub fret: i32,
    /// Semitones above the chord's root, so the label and the colour follow.
    pub interval: i32,
}

impl Spot {
    pub fn label(&self) -> &'static str {
        DEGREE[self.interval.rem_euclid(12) as usize]
    }
    pub fn is_root(&self) -> bool {
        self.interval.rem_euclid(12) == 0
    }
}

/// Where a note of this pitch class is first found: the lowest string carrying
/// it inside the first seven frets, which is where a movable shape is read from.
pub fn first_position(pc: usize) -> (usize, i32) {
    for string in 0..6 {
        let fret = (pc as i32 - TUNING[string]).rem_euclid(12);
        if (1..=7).contains(&fret) {
            return (string, fret);
        }
    }
    (0, (pc as i32 - TUNING[0]).rem_euclid(12))
}

/// The phrase laid on the neck.
///
/// `intervals` are the chord's own, in the order its degrees are written, so
/// `steps` index into them exactly as the note modes do.
pub fn place(root_pc: usize, intervals: &[u8], steps: &[Step]) -> Vec<Spot> {
    let written: Vec<i32> = steps
        .iter()
        .filter_map(|s| intervals.get(s.degree).map(|&i| i as i32 + 12 * s.octave as i32))
        .collect();
    if written.is_empty() {
        return Vec::new();
    }
    // Anchored by the phrase's LOWEST note, not by its root. A phrase written
    // downwards from the root - `1 7, 5, 3,` and further - runs two octaves
    // under it, and anchoring on the root pushed all of that off the bottom of
    // the neck, where it piled onto the two lowest strings. Anchoring on the
    // bottom note puts the same shape where a hand would take it, and leaves
    // every upward phrase exactly where it was: there the root IS the lowest.
    let bottom = *written.iter().min().unwrap_or(&0);
    let bottom_pc = (root_pc as i32 + bottom).rem_euclid(12) as usize;
    let (bottom_string, bottom_fret) = first_position(bottom_pc);
    let root_pitch = TUNING[bottom_string] + bottom_fret - bottom;
    // A hand covers four frets, and the shape is read from one below its
    // lowest note: that is the box every one of the studies stays inside.
    let (lo, hi) = (bottom_fret - 1, bottom_fret + 3);
    let (mut lo, mut hi) = (lo, hi);
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        let Some(&semi) = intervals.get(step.degree) else { continue };
        let interval = semi as i32 + 12 * step.octave as i32;
        let pitch = root_pitch + interval;
        // The lowest string where it falls inside the box the hand is in.
        let string = match (0..6).find(|&s| (lo..=hi).contains(&(pitch - TUNING[s]))) {
            Some(s) => s,
            None => {
                // Out of reach: the hand SHIFTS, it does not stretch. A phrase
                // of three octaves cannot be held in one four-fret box, and
                // reaching for the nearest playable string instead scattered
                // the notes across the neck - four frets apart on neighbouring
                // strings, which is what read as unplayable. Take the string
                // that moves the hand least, and carry the box there.
                let s = (0..6)
                    .filter(|&s| pitch - TUNING[s] >= 0)
                    .min_by_key(|&s| {
                        let fret = pitch - TUNING[s];
                        let shift = if fret < lo { lo - fret } else { fret - hi };
                        // A shift up the neck on the string already in use is
                        // cheaper than crossing strings for the same distance.
                        shift * 2 + (5 - s) as i32
                    })
                    .unwrap_or(0);
                let fret = pitch - TUNING[s];
                lo = fret - 1;
                hi = fret + 3;
                s
            }
        };
        out.push(Spot { string, fret: pitch - TUNING[string], interval });
    }
    out
}

// The drawing, in the numbers the chord diagrams use.
const STRING_GAP: f32 = 48.0;
const DOT_R: f32 = 21.36;
const STEP_X: f32 = 56.0;
const MARGIN_X: f32 = 34.0;
const MARGIN_Y: f32 = 30.0;
const BLUE: &str = "rgba(74, 144, 226, 1)";
const RED: &str = "rgba(208, 2, 27, 1)";
/// Darker than the green of the interval strip, which is drawn on the black
/// ground and has nothing written on it. Here the degree sits INSIDE the dot,
/// and white on `rgb(50, 255, 50)` cannot be read at all.
const GREEN: &str = "rgba(24, 148, 58, 1)";
const FONT: &str = "Arial, &quot;Helvetica Neue&quot;, Helvetica, sans-serif";

/// How wide the strip is against its height, for laying it out: the drawing
/// grows with the window, and the window has to know its shape to give it a
/// box of the right proportions.
pub fn aspect(notes: usize) -> f32 {
    let n = notes.max(1) as f32;
    (MARGIN_X * 2.0 + (n - 1.0) * STEP_X) / (MARGIN_Y * 2.0 + STRING_GAP * 5.0)
}

/// The strip as an SVG, ready for `Image::load_from_svg_data`.
///
/// `done` is parallel to `spots`: a note already played is green, the same
/// green the interval strip above it uses, so the two rows read as one answer.
///
/// `frets` writes the fret number in each dot instead of the degree - ordinary
/// tablature, for reading a shape onto the neck rather than hearing what it is.
/// The colours do not change with it: the root stays the root.
pub fn svg(spots: &[Spot], done: &[bool], frets: bool) -> String {
    let n = spots.len().max(1) as f32;
    let w = MARGIN_X * 2.0 + (n - 1.0) * STEP_X;
    let h = MARGIN_Y * 2.0 + STRING_GAP * 5.0;
    let mut s = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" \
         preserveAspectRatio=\"xMidYMid meet\" viewBox=\"0 0 {w} {h}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"rgba(0, 0, 0, 1)\"></rect>"
    );
    // The strings, high one at the top - the way tab is read.
    for i in 0..6 {
        let y = MARGIN_Y + i as f32 * STRING_GAP;
        s.push_str(&format!(
            "<line x1=\"0\" y1=\"{y}\" x2=\"{w}\" y2=\"{y}\" stroke-width=\"2\" stroke=\"{BLUE}\"></line>"
        ));
    }
    for (i, spot) in spots.iter().enumerate() {
        let x = MARGIN_X + i as f32 * STEP_X;
        let y = MARGIN_Y + (5 - spot.string) as f32 * STRING_GAP;
        let fill = if done.get(i).copied().unwrap_or(false) {
            GREEN
        } else if spot.is_root() {
            RED
        } else {
            BLUE
        };
        // A ring of background under the dot, so the string line does not run
        // through the label.
        s.push_str(&format!(
            "<circle r=\"{r}\" cx=\"{x}\" cy=\"{y}\" fill=\"{fill}\" stroke-width=\"0\"></circle>\
             <text x=\"{x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"24\" \
             text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"#ffffff\">{label}</text>",
            r = DOT_R,
            label = if frets { spot.fret.to_string() } else { spot.label().to_string() },
        ));
    }
    s.push_str("</svg>");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fingerings of the three arpeggio studies, note for note. This is the
    /// whole justification for the placing rule: it was not designed, it was
    /// checked against what a player actually wrote.
    #[test]
    fn the_placing_rule_reproduces_the_studies() {
        // A m7 from the fifth fret, G maj7 from the third, D7 from the fifth.
        let cases: [(usize, &[u8], &[(usize, i32)]); 3] = [
            (9, &[0, 3, 7, 10], &[(0, 5), (0, 8), (1, 7), (2, 5), (2, 7), (3, 5), (4, 5), (4, 8), (5, 5), (5, 8)]),
            (7, &[0, 4, 7, 11], &[(0, 3), (1, 2), (1, 5), (2, 4), (2, 5), (3, 4), (4, 3), (5, 2), (5, 3)]),
            (2, &[0, 4, 7, 10], &[(1, 5), (2, 4), (2, 7), (3, 5), (3, 7), (4, 7), (5, 5), (5, 8)]),
        ];
        for (root, intervals, book) in cases {
            // The studies run 1 3 5 7 through two octaves and a little over.
            let steps: Vec<Step> = (0..book.len())
                .map(|i| Step { degree: i % 4, octave: (i / 4) as i8 })
                .collect();
            let placed = place(root, intervals, &steps);
            let got: Vec<(usize, i32)> = placed.iter().map(|s| (s.string, s.fret)).collect();
            assert_eq!(got, book.to_vec(), "root {root}");
        }
    }

    #[test]
    fn the_root_is_red_and_the_labels_are_the_degrees() {
        let steps: Vec<Step> = (0..4).map(|i| Step { degree: i, octave: 0 }).collect();
        let spots = place(9, &[0, 3, 7, 10], &steps);
        assert_eq!(
            spots.iter().map(|s| s.label()).collect::<Vec<_>>(),
            vec!["1", "♭3", "5", "♭7"]
        );
        assert!(spots[0].is_root() && !spots[1].is_root());
        let out = svg(&spots, &[], false);
        assert!(out.starts_with("<svg") && out.ends_with("</svg>"));
        assert_eq!(out.matches("<circle").count(), 4, "one dot per note");
        assert_eq!(out.matches("<line").count(), 6, "six strings");
        assert!(out.contains(RED), "the root is not marked");
        // And a step played lights green, root or not.
        let lit = svg(&spots, &[true, true, false, false], false);
        assert_eq!(lit.matches(GREEN).count(), 2, "the played steps are not lit");
        assert!(!lit.contains(RED), "the root stayed red after it was played");

        // The same drawing with fret numbers: A m7 from the fifth fret.
        let numbered = svg(&spots, &[], true);
        assert!(numbered.contains(">5<"), "the fifth fret is not written");
        assert!(!numbered.contains("♭3"), "a degree was written where a fret was asked for");
        assert!(numbered.contains(RED), "the root stopped being marked");
    }
}
