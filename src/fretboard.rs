//! Fretboard trainer: learning where the notes sit in one hand position.
//!
//! The exercise fixes a REGION of the fretboard - a set of strings and a span of
//! frets - and then asks for notes that can be found inside it. The region holds
//! for the whole session; only the note changes. That is the point: you settle
//! into one hand position and learn what lies under the fingers there, rather
//! than chasing a target around the neck.
//!
//! WHAT THE APP CAN AND CANNOT CHECK
//!
//! The model reports 12 pitch classes with no string, fret or octave (see
//! `brain::Prediction::pitches`). So the app verifies that the right note is
//! sounding and nothing more - a D played anywhere counts as a D. The position
//! is a prompt the player follows on trust.
//!
//! That is less of a hole than it sounds. Within one string and a four-fret span
//! every pitch class appears at most once, so the target is unambiguous, and
//! missing it INSIDE the region produces a different pitch class and shows up as
//! a wrong answer. Only playing the same note somewhere else entirely goes
//! unnoticed.

use crate::rng::Rng;

/// Open string MIDI numbers in standard tuning, low to high: E2 A2 D3 G3 B3 E4.
pub const OPEN_MIDI: [u8; 6] = [40, 45, 50, 55, 59, 64];

/// String names, low to high. Index matches `OPEN_MIDI`.
pub const STRING_NAMES: [&str; 6] = ["E", "A", "D", "G", "B", "e"];

/// Highest fret the exercise will use.
pub const MAX_FRET: u8 = 15;

/// Which strings the exercise covers.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StringSet {
    All = 0,
    /// E A D - the bass side.
    LowThree = 1,
    /// G B e - the treble side.
    HighThree = 2,
}

impl From<i32> for StringSet {
    fn from(v: i32) -> Self {
        match v {
            1 => StringSet::LowThree,
            2 => StringSet::HighThree,
            _ => StringSet::All,
        }
    }
}

impl StringSet {
    /// Indices into `OPEN_MIDI` / `STRING_NAMES`.
    pub fn strings(self) -> &'static [usize] {
        match self {
            StringSet::All => &[0, 1, 2, 3, 4, 5],
            StringSet::LowThree => &[0, 1, 2],
            StringSet::HighThree => &[3, 4, 5],
        }
    }
}

/// A fixed region of the fretboard: some strings, a span of frets.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub strings: StringSet,
    pub fret_from: u8,
    /// Number of frets covered, e.g. 4 for an index-to-little-finger span.
    pub fret_span: u8,
}

impl Default for Region {
    fn default() -> Self {
        Self { strings: StringSet::All, fret_from: 1, fret_span: 4 }
    }
}

impl Region {
    /// Last fret in the region, clamped to the neck.
    pub fn fret_to(&self) -> u8 {
        self.fret_from
            .saturating_add(self.fret_span.saturating_sub(1))
            .min(MAX_FRET)
    }

    /// Pitch classes reachable in this region, ascending, without duplicates.
    ///
    /// Drawing from this rather than from all twelve matters: asking for a note
    /// that is not in the region would be unanswerable, and the player would be
    /// left hunting for something that is not there.
    pub fn pitch_classes(&self) -> Vec<usize> {
        let mut seen = [false; 12];
        for &s in self.strings.strings() {
            for f in self.fret_from..=self.fret_to() {
                seen[(OPEN_MIDI[s] as usize + f as usize) % 12] = true;
            }
        }
        (0..12).filter(|&pc| seen[pc]).collect()
    }

    /// Where the note can be played inside the region, as `(string index, fret)`.
    ///
    /// Not used by the app - the display names the region, not the individual
    /// spots - but it is what the tests check every drawn note against, so the
    /// invariant "we never ask for an unreachable note" stays enforced.
    #[allow(dead_code)]
    pub fn positions_of(&self, pc: usize) -> Vec<(usize, u8)> {
        let mut out = Vec::new();
        for &s in self.strings.strings() {
            for f in self.fret_from..=self.fret_to() {
                if (OPEN_MIDI[s] as usize + f as usize) % 12 == pc % 12 {
                    out.push((s, f));
                }
            }
        }
        out
    }

    /// The line shown under the note, e.g. "E A D · 5-8".
    pub fn describe(&self) -> String {
        let names: Vec<&str> = self.strings.strings().iter().map(|&s| STRING_NAMES[s]).collect();
        format!("{} · {}-{}", names.join(" "), self.fret_from, self.fret_to())
    }

    /// Draws the next note, never repeating the previous one.
    ///
    /// Returns `None` only for an empty region, which the constructors make
    /// impossible - a span of at least one fret always yields pitch classes.
    pub fn draw(&self, rng: &mut Rng, previous: Option<usize>) -> Option<usize> {
        let pcs = self.pitch_classes();
        match pcs.len() {
            0 => None,
            1 => Some(pcs[0]),
            n => {
                let prev_pos = previous.and_then(|p| pcs.iter().position(|&x| x == p));
                let idx = match prev_pos {
                    Some(i) => rng.below_excluding(n, i),
                    None => rng.below(n),
                };
                Some(pcs[idx])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_strings_are_standard_tuning() {
        // E2 A2 D3 G3 B3 E4 as pitch classes: E A D G B E
        let pcs: Vec<usize> = OPEN_MIDI.iter().map(|&m| m as usize % 12).collect();
        assert_eq!(pcs, vec![4, 9, 2, 7, 11, 4]);
    }

    #[test]
    fn fret_to_respects_span_and_neck() {
        let r = Region { strings: StringSet::All, fret_from: 5, fret_span: 4 };
        assert_eq!(r.fret_to(), 8, "5,6,7,8 is a four-fret span");
        let long = Region { strings: StringSet::All, fret_from: 14, fret_span: 5 };
        assert_eq!(long.fret_to(), MAX_FRET, "must not run off the neck");
    }

    /// A four-fret span on one string gives four consecutive semitones.
    #[test]
    fn single_string_span_gives_consecutive_notes() {
        let r = Region { strings: StringSet::LowThree, fret_from: 5, fret_span: 4 };
        let d = Region { strings: StringSet::All, fret_from: 5, fret_span: 4 };
        assert!(!r.pitch_classes().is_empty());
        assert!(!d.pitch_classes().is_empty());
        // D string (open D = pc 2), frets 5..8 -> G G# A A#
        let only_d = Region { strings: StringSet::All, fret_from: 5, fret_span: 4 };
        let pos = only_d.positions_of(7); // G
        assert!(pos.contains(&(2, 5)), "G must sit at D string fret 5, got {pos:?}");
    }

    /// The claim the whole design rests on: inside one string and a four-fret
    /// span, no pitch class repeats, so the target is unambiguous.
    #[test]
    fn no_duplicate_pitch_class_on_one_string_within_four_frets() {
        for s in 0..6usize {
            for from in 0..=(MAX_FRET - 3) {
                let mut seen = std::collections::HashSet::new();
                for f in from..from + 4 {
                    let pc = (OPEN_MIDI[s] as usize + f as usize) % 12;
                    assert!(seen.insert(pc), "string {s}, frets {from}..: {pc} twice");
                }
            }
        }
    }

    #[test]
    fn drawn_notes_are_always_reachable_in_the_region() {
        let mut rng = Rng::with_seed(1);
        for &set in &[StringSet::All, StringSet::LowThree, StringSet::HighThree] {
            let r = Region { strings: set, fret_from: 3, fret_span: 4 };
            for _ in 0..200 {
                let pc = r.draw(&mut rng, None).expect("region is not empty");
                assert!(
                    !r.positions_of(pc).is_empty(),
                    "{set:?}: drew {pc}, which cannot be played in the region"
                );
            }
        }
    }

    /// Asking for the same note twice in a row reads as the app having missed it.
    #[test]
    fn draw_does_not_repeat_the_previous_note() {
        let mut rng = Rng::with_seed(2);
        let r = Region { strings: StringSet::All, fret_from: 5, fret_span: 4 };
        let mut prev = r.draw(&mut rng, None).unwrap();
        for _ in 0..300 {
            let next = r.draw(&mut rng, Some(prev)).unwrap();
            assert_ne!(next, prev);
            prev = next;
        }
    }

    /// A one-fret span on one string offers a single note; repetition is then
    /// unavoidable and must not hang or panic.
    #[test]
    fn degenerate_region_still_yields_a_note() {
        let mut rng = Rng::with_seed(3);
        let r = Region { strings: StringSet::LowThree, fret_from: 7, fret_span: 1 };
        let pcs = r.pitch_classes();
        assert_eq!(pcs.len(), 3, "three strings, one fret -> three notes");
        assert!(r.draw(&mut rng, Some(pcs[0])).is_some());
    }

    #[test]
    fn describe_names_strings_and_frets() {
        let r = Region { strings: StringSet::LowThree, fret_from: 5, fret_span: 4 };
        assert_eq!(r.describe(), "E A D · 5-8");
        let hi = Region { strings: StringSet::HighThree, fret_from: 1, fret_span: 4 };
        assert_eq!(hi.describe(), "G B e · 1-4");
    }

    /// Six strings over four frets cover most of the octave; the exercise would
    /// be thin if this collapsed to a couple of notes.
    #[test]
    fn full_width_region_is_rich_enough_to_practise() {
        let r = Region { strings: StringSet::All, fret_from: 5, fret_span: 4 };
        assert!(r.pitch_classes().len() >= 9, "got {:?}", r.pitch_classes());
    }
}
