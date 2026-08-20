//! Interval formulas: every subset of the twelve chromatic functions that
//! contains the root.
//!
//! A formula is a bitmask in a `u16`, bit `i` standing for `FUNCS[i]`, with bit
//! 0 - the root - always set. That gives 2^11 = 2048 formulas in total, and
//! C(11, n-1) of them with exactly `n` notes. Nothing here is stored: the whole
//! set is cheaper to enumerate than to keep.
//!
//! Formulas are written as functions rather than note names, so one formula is
//! the same exercise in all twelve keys - which is also why the root has to be
//! stated separately. The same pitches read from a different root are a
//! different formula.

// Not wired into the UI yet - the Formulas mode is the next piece of work, and
// this is the machine it will run on.
#![allow(dead_code)]

use crate::model::ScaleDefinition;
use crate::rng::Rng;

/// Function names, in chromatic order. The five notes outside the major scale
/// are written as flats throughout, so every formula has one spelling.
pub const FUNCS: [&str; 12] = [
    "1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7",
];

/// For each function: which degree of the major scale it sits on, and whether
/// it is flattened. Needed to spell a note: `b3` in C is `Eb`, not `D#`,
/// because it is the third degree lowered.
const DEGREE: [(usize, i32); 12] = [
    (0, 0),  // 1
    (1, -1), // b2
    (1, 0),  // 2
    (2, -1), // b3
    (2, 0),  // 3
    (3, 0),  // 4
    (4, -1), // b5
    (4, 0),  // 5
    (5, -1), // b6
    (5, 0),  // 6
    (6, -1), // b7
    (6, 0),  // 7
];

const LETTERS: [char; 7] = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
const LETTER_PITCH: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
const MAJOR: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];

const NAMES_FLAT: [&str; 12] =
    ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
const NAMES_SHARP: [&str; 12] =
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/// Keys to draw from, one per pitch class, spelled the way players write them.
pub const KEY_POOL: [&str; 12] =
    ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];

// ---------------------------------------------------------------- the set

/// Every formula with exactly `n` notes, the root included, as bitmasks.
///
/// Enumerated rather than stored: the largest group has C(11, 5) = 462
/// entries and building it costs less than reading a file would.
pub fn group(n: usize) -> Vec<u16> {
    let mut out = Vec::new();
    if !(1..=12).contains(&n) {
        return out;
    }
    let mut pick: Vec<usize> = Vec::with_capacity(n - 1);
    build(1, n - 1, &mut pick, &mut out);
    out
}

fn build(start: usize, left: usize, pick: &mut Vec<usize>, out: &mut Vec<u16>) {
    if left == 0 {
        let mut mask: u16 = 1; // the root is in every formula
        for &i in pick.iter() {
            mask |= 1 << i;
        }
        out.push(mask);
        return;
    }
    // Stop early where too few functions are left to complete the pick.
    for i in start..=(12 - left) {
        pick.push(i);
        build(i + 1, left - 1, pick, out);
        pick.pop();
    }
}

/// Indices of the functions in a formula, ascending.
pub fn functions_of(mask: u16) -> Vec<usize> {
    (0..12).filter(|i| mask & (1 << i) != 0).collect()
}

/// `"1 b3 5 b7"`.
pub fn to_text(mask: u16) -> String {
    functions_of(mask)
        .iter()
        .map(|&i| FUNCS[i])
        .collect::<Vec<_>>()
        .join(" ")
}

/// Reads `"1 b3 5 b7"`, `"1b35b7"` or `"1 #4 5"`. Sharps are accepted on input
/// and folded onto their flat spelling; the root is added whether written or
/// not. `None` when a token is not a function.
pub fn parse(text: &str) -> Option<u16> {
    let mut mask: u16 = 1;
    let mut token = String::new();
    let flush = |t: &mut String, mask: &mut u16| -> bool {
        if t.is_empty() {
            return true;
        }
        let norm = match t.as_str() {
            "#1" => "b2", "#2" => "b3", "#4" => "b5", "#5" => "b6", "#6" => "b7",
            other => other,
        };
        let found = FUNCS.iter().position(|&f| f == norm);
        t.clear();
        match found {
            Some(i) => {
                *mask |= 1 << i;
                true
            }
            None => false,
        }
    };

    for c in text.chars() {
        match c {
            'b' | '#' => {
                // an accidental opens a new token
                if !flush(&mut token, &mut mask) {
                    return None;
                }
                token.push(c);
            }
            '1'..='7' => {
                token.push(c);
                if !flush(&mut token, &mut mask) {
                    return None;
                }
            }
            ' ' | ',' | '\t' => {
                if !flush(&mut token, &mut mask) {
                    return None;
                }
            }
            _ => return None,
        }
    }
    if !flush(&mut token, &mut mask) {
        return None;
    }
    Some(mask)
}

/// Does the formula contain every function of `required`?
pub fn contains_all(mask: u16, required: u16) -> bool {
    mask & required == required
}

/// Draws one formula of `n` notes containing everything in `required`.
///
/// `None` when nothing satisfies the filter - asking for four notes including
/// five different functions has no answer, and silently handing back something
/// else would be worse than saying so.
pub fn draw(rng: &mut Rng, n: usize, required: u16) -> Option<u16> {
    let pool: Vec<u16> = group(n)
        .into_iter()
        .filter(|&m| contains_all(m, required))
        .collect();
    if pool.is_empty() {
        return None;
    }
    Some(pool[rng.below(pool.len())])
}

// ------------------------------------------------------------------ keys

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key {
    letter: usize,
    acc: i32,
}

impl Key {
    pub fn pitch(&self) -> i32 {
        (LETTER_PITCH[self.letter] + self.acc).rem_euclid(12)
    }

    pub fn name(&self) -> String {
        let a = match self.acc {
            -1 => "b",
            1 => "#",
            _ => "",
        };
        format!("{}{}", LETTERS[self.letter], a)
    }
}

/// Reads `C`, `Eb`, `F#`. Case-insensitive on the letter.
pub fn parse_key(s: &str) -> Option<Key> {
    let mut ch = s.trim().chars();
    let head = ch.next()?.to_ascii_uppercase();
    let letter = LETTERS.iter().position(|&l| l == head)?;
    let tail: String = ch.collect();
    let acc = match tail.as_str() {
        "" => 0,
        "#" | "♯" => 1,
        "b" | "B" | "♭" => -1,
        _ => return None,
    };
    Some(Key { letter, acc })
}

fn accidental(n: i32) -> Option<&'static str> {
    match n {
        -2 => Some("bb"),
        -1 => Some("b"),
        0 => Some(""),
        1 => Some("#"),
        2 => Some("##"),
        _ => None,
    }
}

/// The note a function names in a key.
///
/// The letter comes from the degree of the major scale, so the spelling follows
/// the harmony: `b3` in C is `Eb`. Where that lands on a name nobody uses -
/// double accidentals, `Fb`, `E#` - the enharmonic equivalent is substituted,
/// spelled to match the key. A player reading `E` off the neck does not care
/// that it is formally `Fb`.
pub fn note_name(key: &Key, func: usize) -> String {
    let (deg, alt) = DEGREE[func];
    let letter = (key.letter + deg) % 7;
    let target = (key.pitch() + MAJOR[deg]).rem_euclid(12);

    let mut diff = (target - LETTER_PITCH[letter]).rem_euclid(12);
    if diff > 6 {
        diff -= 12;
    }
    let acc = diff + alt;

    let pc = (target + alt).rem_euclid(12) as usize;

    match accidental(acc) {
        Some(a) => {
            // The degree spelling stands whenever it is a name a player would
            // write, whichever accidental it carries: `b3` in F# is Ab, not A#,
            // because the function is a flattened third. Keying this off the
            // key's own signature instead - a sharp key gets the sharp table -
            // is what put A# and C# under flat functions.
            let by_degree = format!("{}{}", LETTERS[letter], a);
            let usable = NAMES_FLAT.contains(&by_degree.as_str())
                || NAMES_SHARP.contains(&by_degree.as_str());
            if usable {
                by_degree
            } else {
                // Fb, E#, double accidentals: nobody reads those off a neck.
                NAMES_FLAT[pc].to_string()
            }
        }
        None => NAMES_FLAT[pc].to_string(),
    }
}

/// The notes of a formula in a key, in order.
///
/// For display alongside the functions, and meant to be switched OFF by
/// default: the functions are the exercise, and reading note names is the habit
/// the whole scheme exists to break. Useful all the same when checking yourself
/// on the neck, so it stays available.
pub fn note_names(mask: u16, key: &Key) -> Vec<String> {
    functions_of(mask).iter().map(|&f| note_name(key, f)).collect()
}

/// One exercise: a formula and the key to read it in.
#[derive(Clone, Debug, PartialEq)]
pub struct Drawn {
    pub mask: u16,
    pub key: Key,
}

/// Draws the next exercise.
///
/// `key` of `None` means a fresh one per formula, taken from `KEY_POOL` - one
/// spelling per pitch class, so the names that appear are the ones players
/// write. Drawing the key alongside the formula rather than once per session is
/// deliberate: the formula is the constant, and hearing it land somewhere else
/// each time is what stops it turning into a shape under the fingers.
pub fn next(rng: &mut Rng, notes: usize, required: u16, key: Option<Key>) -> Option<Drawn> {
    let mask = draw(rng, notes, required)?;
    let key = match key {
        Some(k) => k,
        None => draw_key(rng)?,
    };
    Some(Drawn { mask, key })
}

/// One key from `KEY_POOL`, for moving a formula that already stands.
pub fn draw_key(rng: &mut Rng) -> Option<Key> {
    parse_key(KEY_POOL[rng.below(KEY_POOL.len())])
}

// -------------------------------------------------------- against scales

/// How a formula sits next to a scale the player already knows.
#[derive(Clone, Debug, PartialEq)]
pub struct Neighbour {
    /// Index into the list of scales that was searched.
    pub scale: usize,
    /// How far the formula departs: functions it has that the scale has not.
    /// This is the sort key, because it answers "how much of this is new".
    pub outside_count: u32,
    /// Functions in one and not the other, counting both ways. Larger for a
    /// formula with fewer notes, so it says how different the two SETS are
    /// rather than how far the formula strays.
    pub distance: u32,
    /// Every function of the formula is in the scale - so it is a fragment of
    /// something familiar rather than a departure from it.
    pub subset: bool,
    /// Functions the formula has and the scale does not.
    pub outside: u16,
}

/// A scale definition as a formula mask, so the two can be compared.
///
/// Scale files spell raised degrees with sharps (`#4`) where formulas use flats
/// (`b5`); both land on the same pitch class, and the mask is what matters.
pub fn scale_mask(scale: &ScaleDefinition) -> u16 {
    scale
        .intervals
        .iter()
        .fold(1u16, |m, &semitone| m | 1 << (semitone as usize % 12))
}

/// Scales nearest to a formula, closest first, excluding exact matches.
///
/// The point is a formula that is *almost* something known: near enough to hear
/// where it came from, different enough to be worth practising. Ranked by how
/// many functions fall OUTSIDE the scale, not by how much the two sets differ -
/// a five-note formula sitting entirely inside the major scale is a fragment of
/// something familiar, and counting the four notes it does not use as
/// "distance" would bury it under scales it actually contradicts. An identical
/// set teaches nothing new, so it is dropped.
pub fn nearest_scales(mask: u16, scales: &[ScaleDefinition], limit: usize) -> Vec<Neighbour> {
    let mut out: Vec<Neighbour> = scales
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let sm = scale_mask(s);
            let outside = mask & !sm;
            Neighbour {
                scale: i,
                outside_count: outside.count_ones(),
                distance: (mask ^ sm).count_ones(),
                subset: outside == 0,
                outside,
            }
        })
        .filter(|n| n.distance > 0)
        .collect();
    out.sort_by_key(|n| (n.outside_count, n.distance, n.scale));
    out.truncate(limit);
    out
}

// -------------------------------------------------------- against chords

/// Chord shapes looked for inside a formula, as semitones from their own root.
///
/// Sevenths, sixths and triads - the shapes a guitarist reaches for, not every
/// name in the books. The order is the order they are offered in: a seventh says
/// more about where a formula sits than the triad hiding inside it, so the
/// bigger shapes come first. A major triad carries no suffix, as on a chart.
pub const CHORD_SHAPES: [(&str, &[u8]); 14] = [
    ("Maj7", &[0, 4, 7, 11]),
    ("m7", &[0, 3, 7, 10]),
    ("7", &[0, 4, 7, 10]),
    ("m7b5", &[0, 3, 6, 10]),
    ("dim7", &[0, 3, 6, 9]),
    // Bracketed on purpose: "mMaj7" reads like a typo or a contradiction,
    // and it is neither - the "m" is the third, the "maj7" the seventh.
    ("m(maj7)", &[0, 3, 7, 11]),
    ("6", &[0, 4, 7, 9]),
    ("m6", &[0, 3, 7, 9]),
    ("", &[0, 4, 7]),
    ("m", &[0, 3, 7]),
    ("dim", &[0, 3, 6]),
    ("aug", &[0, 4, 8]),
    ("sus2", &[0, 2, 7]),
    ("sus4", &[0, 5, 7]),
];

/// Degrees for chords, in the same order as `FUNCS`.
///
/// Roman where the functions above are arabic, which is how chords are written
/// over a key anyway - and it keeps the two rows apart: read as arabic, a
/// dominant seventh on the second degree comes out "27".
pub const ROMAN: [&str; 12] = [
    "I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII", "VII",
];

/// A chord that fits entirely inside a formula.
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    /// Function of the formula the chord is rooted on.
    pub degree: usize,
    /// Index into `CHORD_SHAPES`.
    pub shape: usize,
}

impl Fit {
    /// How it reads on screen: the degree it stands on, then the suffix.
    pub fn name(&self) -> String {
        format!("{}{}", ROMAN[self.degree], CHORD_SHAPES[self.shape].0)
    }

    /// The same chord spelled with note names, e.g. `AbMaj7` - for the line
    /// under the degrees, where the note names under the functions are.
    pub fn named_in(&self, key: &Key) -> String {
        format!("{}{}", note_name(key, self.degree), CHORD_SHAPES[self.shape].0)
    }

    /// Which functions of the formula the chord is built from - what to pick out
    /// of the row above when this one is pointed at.
    pub fn functions(&self) -> Vec<usize> {
        CHORD_SHAPES[self.shape]
            .1
            .iter()
            .map(|&semi| (self.degree + semi as usize) % 12)
            .collect()
    }
}

/// Chords playable without leaving the formula, the fullest first.
///
/// "Fits" means every note of the chord is in the set - the formula need not use
/// all of it. That is the useful direction: it says what can be sounded under
/// the exercise without stepping outside it.
///
/// Rooted on the formula's own functions only. A chord rooted on a note the
/// formula does not contain can still have all its notes inside it, but it is
/// not something the formula offers to play.
///
/// Only the fullest are kept. Everything smaller that fits inside one of them is
/// already being played when it is: the triad inside a seventh, the seventh that
/// is that seventh read from another of its notes. Over the major scale this
/// leaves exactly the seven diatonic sevenths, which is the answer; without it
/// the same four notes came back three times over as `IV7`, `IV6`, `IV`.
///
/// That covers symmetry too, where a shape reads the same from several of its
/// own notes - a diminished seventh from each of its four.
pub fn chords_inside(mask: u16, limit: usize) -> Vec<Fit> {
    let mut kept: Vec<(Fit, u16)> = Vec::new();
    for (shape, (_, intervals)) in CHORD_SHAPES.iter().enumerate() {
        for &degree in functions_of(mask).iter() {
            let pcs = intervals
                .iter()
                .fold(0u16, |m, &semi| m | 1 << ((degree + semi as usize) % 12));
            if pcs & !mask != 0 {
                continue;
            }
            // Bigger shapes come first, so anything this is contained in has
            // been seen already.
            if kept.iter().any(|(_, other)| pcs & !other == 0) {
                continue;
            }
            kept.push((Fit { degree, shape }, pcs));
        }
    }
    let mut out: Vec<Fit> = kept.into_iter().map(|(f, _)| f).collect();
    out.sort_by_key(|f| (f.shape, f.degree));
    out.truncate(limit);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_whole_set_is_two_thousand_and_forty_eight() {
        // Every subset of the eleven functions above the root.
        let total: usize = (1..=12).map(|n| group(n).len()).sum();
        assert_eq!(total, 2048);
        assert_eq!(group(1).len(), 1, "only the root");
        assert_eq!(group(2).len(), 11, "the root and one other");
        assert_eq!(group(12).len(), 1, "the whole chromatic scale");
        assert_eq!(group(7).len(), 462, "C(11, 6)");
        assert!(group(0).is_empty(), "a formula without a root is not one");
        assert!(group(13).is_empty());
    }

    #[test]
    fn every_formula_carries_the_root() {
        for n in 1..=12 {
            for m in group(n) {
                assert_eq!(m & 1, 1, "{m:012b} has no root");
                assert_eq!(m.count_ones() as usize, n);
            }
        }
    }

    #[test]
    fn text_survives_a_round_trip() {
        for text in ["1 b3 5 b7", "1 2 3 4 5 6 7", "1"] {
            let m = parse(text).expect(text);
            assert_eq!(to_text(m), text);
        }
        // Written tight, and with sharps, which fold onto the flat spelling.
        assert_eq!(parse("1b35b7"), parse("1 b3 5 b7"));
        assert_eq!(parse("1 #4 5"), parse("1 b5 5"));
        // The root need not be written; it is in every formula anyway.
        assert_eq!(parse("b3 5"), parse("1 b3 5"));
        assert_eq!(parse("1 x"), None);
    }

    #[test]
    fn a_filter_that_cannot_be_met_says_so() {
        let mut rng = Rng::with_seed(1);
        let required = parse("1 b3 5 b7").unwrap();
        let drawn = draw(&mut rng, 5, required).expect("five notes can hold four");
        assert!(contains_all(drawn, required));
        assert_eq!(drawn.count_ones(), 5);
        // Four functions demanded, three notes on offer.
        assert_eq!(draw(&mut rng, 3, required), None);
    }

    #[test]
    fn the_key_is_either_chosen_or_drawn() {
        let mut rng = Rng::with_seed(7);
        let required = parse("b3 b7").unwrap();

        // Chosen: every draw reads in that key, the formula keeps changing.
        let eb = parse_key("Eb").unwrap();
        for _ in 0..20 {
            let d = next(&mut rng, 5, required, Some(eb)).unwrap();
            assert_eq!(d.key, eb);
            assert!(contains_all(d.mask, required));
            assert_eq!(d.mask.count_ones(), 5);
        }

        // Drawn: keys come from the pool, and over twenty draws more than one
        // of them shows up - otherwise "random" would be a fixed key in disguise.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..20 {
            let d = next(&mut rng, 4, 1, None).unwrap();
            assert!(KEY_POOL.contains(&d.key.name().as_str()), "{} is off the pool", d.key.name());
            seen.insert(d.key.name());
        }
        assert!(seen.len() > 1, "the drawn key never moved");

        // An impossible filter fails here too, rather than looping for a key.
        assert_eq!(next(&mut rng, 2, parse("1 b3 5 b7").unwrap(), None), None);
    }

    #[test]
    fn notes_are_spelled_by_degree() {
        let c = parse_key("C").unwrap();
        assert_eq!(note_names(parse("1 b3 5 b7").unwrap(), &c), ["C", "Eb", "G", "Bb"]);
        let eb = parse_key("Eb").unwrap();
        assert_eq!(note_name(&eb, 0), "Eb");
        assert_eq!(note_name(&eb, 3), "Gb", "b3 of Eb is the third lowered");
        // Fb is formally right for b2 of Eb and nobody reads it that way.
        assert_eq!(note_name(&eb, 1), "E");
        let fs = parse_key("F#").unwrap();
        assert_eq!(note_name(&fs, 0), "F#");
        assert_eq!(note_names(parse("1 3 5").unwrap(), &fs), ["F#", "A#", "C#"]);
        // A flattened function keeps its flat in a sharp key: the spelling
        // follows the degree, not the key signature. b5 of G reads Db, and
        // reading the key signature instead had it as C#.
        let g = parse_key("G").unwrap();
        assert_eq!(note_names(parse("1 3 b5 5 6").unwrap(), &g), ["G", "B", "Db", "D", "E"]);
        assert!(parse_key("H").is_none());
    }

    fn scale(name: &str, semis: &[u8]) -> ScaleDefinition {
        ScaleDefinition {
            name: name.to_string(),
            intervals: semis.to_vec(),
            names: vec![],
        }
    }

    #[test]
    fn the_nearest_scale_is_the_one_worth_showing() {
        let scales = vec![
            scale("Major", &[0, 2, 4, 5, 7, 9, 11]),
            scale("Harmonic minor", &[0, 2, 3, 5, 7, 8, 11]),
        ];
        // Dorian: two functions apart from major, no closer to harmonic minor.
        let dorian = parse("1 2 b3 4 5 6 b7").unwrap();
        let near = nearest_scales(dorian, &scales, 5);
        assert_eq!(near[0].scale, 0, "major is the nearer of the two");
        assert_eq!(near[0].outside_count, 2, "b3 and b7 fall outside");
        assert_eq!(near[0].distance, 4, "and major has 3 and 7 that dorian has not");
        assert!(!near[0].subset, "dorian leaves the major scale");
        assert_eq!(to_text(near[0].outside), "b3 b7");

        // A fragment of the major scale is a subset: familiar, not a departure.
        let triad = parse("1 3 5").unwrap();
        let near = nearest_scales(triad, &scales, 5);
        assert!(near[0].subset, "1 3 5 lives inside the major scale");
        assert_eq!(near[0].outside_count, 0);
        assert_eq!(near[0].distance, 4, "four scale notes it does not use");

        // The scale itself is dropped - it has nothing new to teach.
        let major = parse("1 2 3 4 5 6 7").unwrap();
        assert!(nearest_scales(major, &scales, 5).iter().all(|n| n.scale != 0));
    }

    /// "Fits" means every note of the chord is inside the formula - which for
    /// four notes leaves very little room.
    #[test]
    fn a_chord_fits_only_when_all_of_it_is_inside() {
        let m = parse("1 b3 5 b7").expect("a minor seventh reads as a formula");
        let names: Vec<String> = chords_inside(m, 20).iter().map(|f| f.name()).collect();
        // One chord, not four: the triad inside it, the major triad on b3 and
        // the same four notes read from b3 as a sixth are all being played
        // already whenever the seventh is.
        assert_eq!(names, vec!["Im7".to_string()], "{names:?}");
    }

    /// The rule stated at its plainest: over the major scale, the chords that
    /// fit are the seven diatonic sevenths. Anything more is one of them under
    /// another name, anything less is a piece of one.
    #[test]
    fn the_major_scale_yields_its_seven_sevenths() {
        let m = parse("1 2 3 4 5 6 7").expect("the major scale reads as a formula");
        let names: Vec<String> = chords_inside(m, 7).iter().map(|f| f.name()).collect();
        assert_eq!(
            names,
            vec!["IMaj7", "IVMaj7", "IIm7", "IIIm7", "VIm7", "V7", "VIIm7b5"],
            "{names:?}"
        );
    }

    /// The fullest shapes are offered first - a seventh says more about where a
    /// formula sits than the triad hiding inside it.
    #[test]
    fn sevenths_come_before_triads() {
        let m = parse("1 b2 2 b3 3 4 b5 5 b6 6 b7 7").expect("the whole chromatic set");
        let fits = chords_inside(m, 4);
        assert!(
            fits.iter().all(|f| CHORD_SHAPES[f.shape].1.len() == 4),
            "a triad got in front of the sevenths"
        );
    }

    /// A diminished seventh reads the same from each of its four notes; naming
    /// it four times would fill the line with one chord.
    #[test]
    fn symmetrical_shapes_are_named_once() {
        let m = parse("1 b2 2 b3 3 4 b5 5 b6 6 b7 7").expect("the whole chromatic set");
        let fits = chords_inside(m, 200);
        let dim7 = fits.iter().filter(|f| CHORD_SHAPES[f.shape].0 == "dim7").count();
        assert_eq!(dim7, 3, "there are only three diminished sevenths in twelve notes");
        // None at all, and rightly: an augmented triad sits inside the
        // m(maj7) a third below it, so wherever one fits, so does that.
        let aug = fits.iter().filter(|f| CHORD_SHAPES[f.shape].0 == "aug").count();
        assert_eq!(aug, 0, "an augmented triad on its own is never the fullest");
    }
}
