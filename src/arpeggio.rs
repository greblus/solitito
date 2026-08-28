//! Arpeggio pattern generator.
//!
//! Every phrase in the bundled set is a walk along the same LADDER: the chord
//! tones stacked upwards, so rung 0 is the root, 1 the third, 2 the fifth, 3 the
//! seventh, 4 the root an octave up, and so on. What tells the patterns apart is
//! the rule for stepping along it.
//!
//! Reading the supplied phrases back as rung numbers:
//!
//! ```text
//! Two Octaves Up-Down   0 1 2 3 4 5 6 7          +1 each time
//! Broken Thirds         0 2 1 3 2 4 3 5          +2, -1, +2, -1
//! Triplet Sequence      0 1 2  1 2 3  2 3 4      threes, each starting a rung up
//! ```
//!
//! So a generator needs three choices: which step rule, how far up, and how to
//! come back down. The tests check that the rules reproduce the hand-written
//! phrases exactly - if they did not, the rules would be a guess rather than a
//! reading.
//!
//! Output is the app's positional degree notation (`1 3 5 7 1'`), which is why it
//! fits any chord quality: rung 1 is "the third of whatever this chord is".

use crate::rng::Rng;

/// Degree tokens by rung within one octave. Positional, as the trainer expects.
const RUNGS: [&str; 4] = ["1", "3", "5", "7"];

/// How to walk up the ladder.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Step {
    /// One rung at a time: a plain arpeggio.
    Straight,
    /// Up two, back one. The classic broken-thirds shape.
    BrokenThirds,
    /// Overlapping groups of three, each starting one rung higher.
    Triplets,
    /// Up three, back two - wider skips, still climbing.
    BrokenFourths,
}

impl Step {
    /// Rungs visited on the way up to `top`, starting from rung 0.
    fn ascend(self, top: usize) -> Vec<usize> {
        let mut out = Vec::new();
        match self {
            Step::Straight => out.extend(0..=top),
            Step::BrokenThirds => {
                let mut p = 0usize;
                let mut up = true;
                while p <= top {
                    out.push(p);
                    // The pair (+2, -1) nets one rung per two notes.
                    p = if up { p + 2 } else { p.saturating_sub(1) };
                    up = !up;
                    if !up && p > top { break; }
                }
            }
            Step::Triplets => {
                for start in 0..top.saturating_sub(1) {
                    out.extend([start, start + 1, start + 2]);
                }
            }
            Step::BrokenFourths => {
                let mut p = 0usize;
                let mut up = true;
                while p <= top {
                    out.push(p);
                    p = if up { p + 3 } else { p.saturating_sub(2) };
                    up = !up;
                    if !up && p > top { break; }
                }
            }
        }
        out.retain(|&p| p <= top);
        out
    }

    /// Rungs on the way back down, starting one below the peak so the turning
    /// note is not played twice.
    ///
    /// This applies the rule in reverse rather than replaying the ascent
    /// backwards - those are not the same walk. Broken thirds climb (+2, -1) and
    /// descend (-2, +1); reversing the ascent list instead produces a stutter at
    /// the turn, which is how the hand-written phrase caught the mistake.
    fn descend(self, top: usize) -> Vec<usize> {
        let start = top.saturating_sub(1);
        let mut out = Vec::new();
        match self {
            Step::Straight => out.extend((0..=start).rev()),
            Step::BrokenThirds | Step::BrokenFourths => {
                let (down, up) = if self == Step::BrokenThirds { (2, 1) } else { (3, 2) };
                let mut p = start as isize;
                let mut going_down = true;
                while p >= 0 {
                    out.push(p as usize);
                    // The phrase lands on the root. Without this the "+1" half of
                    // the rule bounces back off rung 0 and adds a stray third.
                    if p == 0 { break; }
                    p += if going_down { -(down as isize) } else { up as isize };
                    going_down = !going_down;
                }
            }
            Step::Triplets => {
                for start_rung in (0..=start.saturating_sub(2)).rev() {
                    out.extend([start_rung + 2, start_rung + 1, start_rung]);
                }
            }
        }
        out.retain(|&p| p <= top);
        out
    }
}

/// What happens after the top rung.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Shape {
    /// Stop at the top. Not drawn at random - a phrase that climbs and then just
    /// stops leaves the exercise hanging - but it is what the bundled
    /// "Two Octaves Up" is, and the tests build it.
    #[allow(dead_code)]
    Up,
    /// Come back down the same way.
    UpDown,
}

/// How the phrase finishes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tail {
    None,
    /// `7, 1` - the seventh below the root, resolving up into it.
    LeadingTone,
    /// `7, 5, 7, 1` - a longer turn under the root before landing.
    ApproachFromBelow,
}

pub struct Recipe {
    pub step: Step,
    pub shape: Shape,
    pub tail: Tail,
    /// Highest rung. 8 is two octaves above the root (rungs 0..8).
    pub top: usize,
}

impl Default for Recipe {
    fn default() -> Self {
        Self { step: Step::Straight, shape: Shape::UpDown, tail: Tail::None, top: 7 }
    }
}

fn token(rung: usize) -> String {
    let mark = "'".repeat(rung / RUNGS.len());
    format!("{}{}", RUNGS[rung % RUNGS.len()], mark)
}

/// Renders a recipe as degree tokens, ready for the pattern file.
pub fn build(recipe: &Recipe) -> Vec<String> {
    let up = recipe.step.ascend(recipe.top);
    let mut rungs = up.clone();

    if recipe.shape == Shape::UpDown {
        rungs.extend(recipe.step.descend(recipe.top));
    }

    let mut out: Vec<String> = rungs.iter().map(|&r| token(r)).collect();
    match recipe.tail {
        Tail::None => {}
        Tail::LeadingTone => out.extend(["7,".into(), "1".into()]),
        Tail::ApproachFromBelow => {
            out.extend(["7,".into(), "5,".into(), "7,".into(), "1".into()])
        }
    }

    // A degree repeated back to back would be credited twice the moment it
    // sounds, so the phrase would skip. Collapse any such pair.
    out.dedup_by(|a, b| a.trim_end_matches(['\'', ',']) == b.trim_end_matches(['\'', ',']));
    out
}

/// A fresh phrase, drawn from the same vocabulary as the bundled ones.
pub fn random(rng: &mut Rng) -> Vec<String> {
    // Broken fourths are in the vocabulary but not in the draw: up three rungs
    // is a seventh, and a phrase built of sevenths is a reading exercise rather
    // than something a hand falls into. The three rules below are the ones the
    // studies are actually written in.
    let step = match rng.below(3) {
        0 => Step::Straight,
        1 => Step::BrokenThirds,
        _ => Step::Triplets,
    };
    // Triplets triple the note count, so keep their range shorter or the strip
    // runs to a dozen pages.
    // Two octaves, or a little over. Three octaves cannot be held in one hand
    // position, so the phrase would spend itself shifting up the neck instead
    // of being played.
    let top = if step == Step::Triplets { 5 + rng.below(2) } else { 7 + rng.below(2) };
    let tail = match rng.below(3) {
        0 => Tail::None,
        1 => Tail::LeadingTone,
        _ => Tail::ApproachFromBelow,
    };
    build(&Recipe { step, shape: Shape::UpDown, tail, top })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joined(r: &Recipe) -> String {
        build(r).join(" ")
    }

    /// The rules are a READING of the supplied phrases, so they have to
    /// reproduce them. If this drifts, the generator has stopped speaking the
    /// same language as the hand-written set.
    #[test]
    fn reproduces_two_octaves_up_down() {
        let got = joined(&Recipe { step: Step::Straight, shape: Shape::UpDown,
                                   tail: Tail::None, top: 7 });
        assert_eq!(got, "1 3 5 7 1' 3' 5' 7' 5' 3' 1' 7 5 3 1");
    }

    #[test]
    fn reproduces_two_octaves_up() {
        let got = joined(&Recipe { step: Step::Straight, shape: Shape::Up,
                                   tail: Tail::None, top: 7 });
        assert_eq!(got, "1 3 5 7 1' 3' 5' 7'");
    }

    /// arp1, straight from the user's Guitar Pro file.
    #[test]
    fn reproduces_broken_thirds() {
        let got = joined(&Recipe { step: Step::BrokenThirds, shape: Shape::UpDown,
                                   tail: Tail::None, top: 9 });
        assert_eq!(
            got,
            "1 5 3 7 5 1' 7 3' 1' 5' 3' 7' 5' 1'' 7' 3'' 1'' 5' 7' 3' 5' 1' 3' 7 1' 5 7 3 5 1"
        );
    }

    /// The leading-tone ending of arp4.
    #[test]
    fn reproduces_leading_tone_finish() {
        let got = joined(&Recipe { step: Step::Straight, shape: Shape::UpDown,
                                   tail: Tail::LeadingTone, top: 8 });
        assert_eq!(got, "1 3 5 7 1' 3' 5' 7' 1'' 7' 5' 3' 1' 7 5 3 1 7, 1");
    }

    /// The longer turn under the root, from arp3.
    #[test]
    fn reproduces_approach_from_below() {
        let got = joined(&Recipe { step: Step::Straight, shape: Shape::UpDown,
                                   tail: Tail::ApproachFromBelow, top: 7 });
        assert_eq!(got, "1 3 5 7 1' 3' 5' 7' 5' 3' 1' 7 5 3 1 7, 5, 7, 1");
    }

    /// Triplets climb in overlapping threes, as in arp2.
    #[test]
    fn triplets_climb_in_overlapping_threes() {
        let up = Step::Triplets.ascend(5);
        assert_eq!(&up[..9], &[0, 1, 2, 1, 2, 3, 2, 3, 4]);
    }

    /// The one rule the app enforces: the same degree twice running would be
    /// credited twice as soon as it sounds.
    #[test]
    fn no_generated_phrase_repeats_a_degree() {
        let mut rng = Rng::with_seed(4);
        for _ in 0..500 {
            let p = random(&mut rng);
            for w in p.windows(2) {
                let a = w[0].trim_end_matches(['\'', ',']);
                let b = w[1].trim_end_matches(['\'', ',']);
                assert_ne!(a, b, "repeated degree in {}", p.join(" "));
            }
        }
    }

    /// Every token has to be something the trainer can ask for.
    #[test]
    fn generated_tokens_are_valid_degrees() {
        let mut rng = Rng::with_seed(5);
        for _ in 0..300 {
            for t in random(&mut rng) {
                let base = t.trim_end_matches(['\'', ',']);
                assert!(RUNGS.contains(&base), "odd token {t:?}");
            }
        }
    }

    /// Long enough to be an exercise, short enough to stay playable.
    #[test]
    fn generated_phrases_are_a_sensible_length() {
        let mut rng = Rng::with_seed(6);
        for _ in 0..300 {
            let n = random(&mut rng).len();
            assert!((8..=60).contains(&n), "phrase of {n} steps");
        }
    }

    /// A phrase that always came out the same would not be a generator.
    #[test]
    fn successive_phrases_differ() {
        let mut rng = Rng::with_seed(7);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..40 {
            seen.insert(random(&mut rng).join(" "));
        }
        assert!(seen.len() > 5, "only {} distinct phrases in 40 draws", seen.len());
    }
}
