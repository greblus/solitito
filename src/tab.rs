//! Arpeggios drawn as tablature, with intervals in place of fret numbers.
//!
//! Six string lines and a dot per note, the degree written inside it: the shape
//! on the neck is then visible directly, and the octave a note sits in needs no
//! marker - it is which string the dot is on.
//!
//! Drawn in the same language as the chord diagrams in `diagrams`: the UI's
//! card for ground (`diagrams::CARD`), blue `rgba(74, 144, 226)` lines, a red root, white labels in Arial,
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
/// Used where the caller has no name of its own to give.
const DEGREE: [&str; 12] = [
    "1", "♭2", "2", "♭3", "3", "4", "♭5", "5", "♭6", "6", "♭7", "7",
];

/// A degree as the exercise writes it, in the diagrams' typography: `b3` and
/// `#2` are the same pitch and NOT the same degree, and a scale that spells the
/// altered second as `#2` has to read `#2` on the neck as well.
fn spell(name: &str) -> String {
    name.trim_end_matches(['\'', ','])
        .replace('b', "♭")
        .replace('#', "♯")
}

/// One note of the phrase, placed on the neck.
#[derive(Debug, Clone, PartialEq)]
pub struct Spot {
    /// 0 is the lowest string, 5 the highest - the order `TUNING` is in.
    pub string: usize,
    pub fret: i32,
    /// Semitones above the chord's root, so the colour follows.
    pub interval: i32,
    /// What goes in the dot: the degree as the exercise writes it, without the
    /// octave marker - which string the dot sits on IS the octave. `None` where
    /// the caller passed no names, and then the interval names itself.
    pub name: Option<String>,
}

impl Spot {
    pub fn label(&self) -> String {
        match &self.name {
            Some(n) => n.clone(),
            None => DEGREE[self.interval.rem_euclid(12) as usize].to_string(),
        }
    }
    pub fn is_root(&self) -> bool {
        self.interval.rem_euclid(12) == 0
    }
}


/// The phrase laid on the neck.
///
/// `intervals` are the chord's own, in the order its degrees are written, so
/// `steps` index into them exactly as the note modes do.
pub fn place(root_pc: usize, intervals: &[u8], steps: &[Step], names: &[String]) -> Vec<Spot> {
    place_near(root_pc, intervals, steps, names, None)
}

/// The same, with a fret to prefer where two positions cost the hand the same.
///
/// A scale played always from the lowest place it fits is half an exercise: the
/// shape is the same in every position, and the point is to know it wherever
/// the hand lands. `anchor` is drawn by the caller, once per pass.
pub fn place_near(
    root_pc: usize,
    intervals: &[u8],
    steps: &[Step],
    names: &[String],
    anchor: Option<i32>,
) -> Vec<Spot> {
    place_from(root_pc, intervals, steps, names, anchor, None)
}

/// Where the root of `root_pc` is taken as the steps' octave 0: its lowest
/// place at or above the open low E.
pub fn root_above_low_e(root_pc: usize) -> i32 {
    TUNING[0] + (root_pc as i32 - TUNING[0]).rem_euclid(12)
}

/// A phrase laid under a hand that does not move: every note at the lowest
/// fret that takes it between frets `lo` and `hi` - only the G and B strings,
/// a third apart, ever both reach one note, and then an open B is not moved
/// to the fourth fret of the G. Octave 0 of the steps is `root_above_low_e`.
/// A note the hand cannot reach is left out.
pub fn place_in_hand(
    root_pc: usize,
    intervals: &[u8],
    steps: &[Step],
    names: &[String],
    (lo, hi): (i32, i32),
) -> Vec<Spot> {
    let root = root_above_low_e(root_pc);
    steps
        .iter()
        .filter_map(|step| {
            let interval = *intervals.get(step.degree)? as i32 + 12 * step.octave as i32;
            let pitch = root + interval;
            let string = (0..6)
                .filter(|&s| (lo..=hi).contains(&(pitch - TUNING[s])))
                .min_by_key(|&s| pitch - TUNING[s])?;
            Some(Spot {
                string,
                fret: pitch - TUNING[string],
                interval,
                name: names.get(step.degree).map(|n| spell(n)),
            })
        })
        .collect()
}

/// The same, and starting from a given string where one is asked for.
///
/// A scale is the same shape in every position AND on every pair of strings it
/// will fit; drawing the string it starts on is what stops the exercise from
/// being one place on the neck practised over and over.
pub fn place_from(
    root_pc: usize,
    intervals: &[u8],
    steps: &[Step],
    names: &[String],
    anchor: Option<i32>,
    start_string: Option<usize>,
) -> Vec<Spot> {
    let written: Vec<i32> = steps
        .iter()
        .filter_map(|s| intervals.get(s.degree).map(|&i| i as i32 + 12 * s.octave as i32))
        .collect();
    if written.is_empty() {
        return Vec::new();
    }
    // The phrase is laid from its LOWEST note, not from its root: one written
    // downwards - `1 7, 5, 3,` and further - runs two octaves under the root,
    // and anchoring on the root pushed all of that off the bottom of the neck.
    // For an upward phrase the two are the same note anyway.
    let bottom = *written.iter().min().unwrap_or(&0);
    let bottom_pc = (root_pc as i32 + bottom).rem_euclid(12) as usize;

    // Every place on the neck that note can be taken, and the whole phrase laid
    // out from each. A player chooses the position ONCE, looking at the phrase;
    // laying it out greedily from the first position that fits made the hand
    // wander up and back down the neck inside a single study.
    let mut best: Option<(Cost, Vec<Spot>)> = None;
    for string in 0..6 {
        for fret in 0..=12 {
            if (TUNING[string] + fret).rem_euclid(12) != bottom_pc as i32 {
                continue;
            }
            let Some((cost, spots)) =
                lay(TUNING[string] + fret - bottom, fret, steps, intervals, names)
            else {
                continue;
            };
            // Where the caller named a string to start from, that comes
            // first: it is the shape's place on the neck, not a preference
            // between two equal ones. A fret named instead is the last word
            // between two positions the hand finds equally easy.
            let cost = match (start_string, anchor) {
                (Some(want), _) => {
                    // The BOTTOM of the shape, not the note played first: a
                    // scale run downwards starts at its top, and "starting from
                    // the A string" is about where the shape sits.
                    let from = spots
                        .iter()
                        .min_by_key(|s| TUNING[s.string] + s.fret)
                        .map(|s| s.string)
                        .unwrap_or(0);
                    (
                        (from as i32 - want as i32).unsigned_abs() as usize,
                        cost.0 as i32,
                        cost.1,
                    )
                }
                (None, Some(a)) => (cost.0, cost.1, (fret - a).abs()),
                (None, None) => cost,
            };
            if best.as_ref().is_none_or(|(b, _)| cost < *b) {
                best = Some((cost, spots));
            }
        }
    }
    best.map(|(_, spots)| spots).unwrap_or_default()
}

/// What a position costs the hand: how many times it has to move, how far in
/// total, and how far up the neck it ends up. In that order - a shift is worth
/// more than any number of frets, and between two positions that move the hand
/// equally the lower one is the one a player takes.
type Cost = (usize, i32, i32);

/// The phrase laid out from one position, or `None` if it cannot be played
/// there at all.
fn lay(
    root_pitch: i32,
    start_fret: i32,
    steps: &[Step],
    intervals: &[u8],
    names: &[String],
) -> Option<(Cost, Vec<Spot>)> {
    let (mut lo, mut hi) = hand_from(start_fret);
    let (mut shifts, mut distance, mut highest) = (0usize, 0i32, 0i32);
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        let &semi = intervals.get(step.degree)?;
        let interval = semi as i32 + 12 * step.octave as i32;
        let pitch = root_pitch + interval;
        let string = match (0..6).find(|&s| (lo..=hi).contains(&(pitch - TUNING[s]))) {
            Some(s) => s,
            None => {
                // Out of reach: the hand SHIFTS, it does not stretch. Take the
                // string that moves it least, and carry the box there.
                let s = (0..6)
                    .filter(|&s| pitch - TUNING[s] >= 0)
                    .min_by_key(|&s| {
                        let fret = pitch - TUNING[s];
                        let move_by = if fret < lo { lo - fret } else { fret - hi };
                        move_by * 2 + (5 - s) as i32
                    })?;
                let fret = pitch - TUNING[s];
                shifts += 1;
                distance += if fret < lo { lo - fret } else { fret - hi };
                (lo, hi) = hand_from(fret);
                s
            }
        };
        let fret = pitch - TUNING[string];
        // Nothing below the nut, and nothing past where the neck joins the body.
        if !(0..=17).contains(&fret) {
            return None;
        }
        highest = highest.max(fret);
        out.push(Spot {
            string,
            fret,
            interval,
            name: names.get(step.degree).map(|n| spell(n)),
        });
    }
    Some(((shifts, distance, highest), out))
}

/// The frets a hand covers from a note on `fret`: one below it to three
/// above. From an open string that is first position, frets 0 to 4 - one below
/// would be under the nut, where E major's G#2 went to the A string at fret -1
/// and no position was left to draw it in.
pub fn hand_from(fret: i32) -> (i32, i32) {
    if fret == 0 { (0, 4) } else { (fret - 1, fret + 3) }
}

/// A chord laid out as a VOICING: one note per string, close together, and as
/// near as possible to where the last one stood.
///
/// The note modes walk a phrase and `place` lays it under one hand; a set of
/// three or four intervals is not a phrase but a grip, and a player takes it
/// with a string for each note. What matters from chord to chord is then not
/// the position on the neck but the voices: `1 3 7` over one chord and over the
/// next should sit where the fingers barely move.
///
/// `prev` is the grip before this one, and the voices are led from it. With
/// none - the first chord, or voice leading switched off - the grip is taken on
/// `from_string` and near `anchor`, both drawn by the caller, so the grips move
/// about the neck instead of settling on the bottom strings.
pub fn place_voiced(
    root_pc: usize,
    intervals: &[u8],
    steps: &[Step],
    names: &[String],
    prev: Option<&[Spot]>,
    anchor: i32,
    from_string: usize,
    close_only: bool,
) -> Vec<Spot> {
    place_grip_voiced(root_pc, intervals, steps, names, prev, anchor, from_string, close_only, !close_only)
}

/// The same, and saying whether the note asked for first has to be the bass.
/// It does where the exercise is an inversion - the tones of the chord in
/// the order written; in a line, a structure or a suspension it is a melody,
/// and holding its first note under the rest cost a stretch for nothing.
#[allow(clippy::too_many_arguments)]
pub fn place_grip_voiced(
    root_pc: usize,
    intervals: &[u8],
    steps: &[Step],
    names: &[String],
    prev: Option<&[Spot]>,
    anchor: i32,
    from_string: usize,
    close_only: bool,
    bass_first: bool,
) -> Vec<Spot> {
    // Each note as its degree and its pitch class.
    let notes: Vec<(usize, i32)> = steps
        .iter()
        .filter_map(|s| {
            intervals
                .get(s.degree)
                .map(|&i| (s.degree, (root_pc as i32 + i as i32 + 12 * s.octave as i32).rem_euclid(12)))
        })
        .collect();
    // Wider than four voices is not a grip any more: past that the strings run
    // out and the phrase is better read as a phrase.
    if notes.is_empty() || notes.len() > 4 {
        return place(root_pc, intervals, steps, names);
    }
    let mut best: Option<(i32, Vec<Spot>)> = None;
    // Every order the notes can be stacked in, bottom up, on every set of
    // strings and at every hand position. The order they are PLAYED in is the
    // exercise's; the order they are stacked in only has to fit the hand.
    // Stacked in the order played, 3 5 7 8 over Ab maj7 had to put the Ab a
    // ninth above the G - a skipped string and a stretch - where the Ab on the
    // low E makes a grip of four neighbouring strings.
    for stack_order in orders(notes.len()) {
        // In an inversion the note asked for first is the bass: started on the
        // b7, a grip with the root under it drew root position and only the
        // ring said otherwise. Shuffled, or in a melody, the order says nothing
        // about the bass and the grip is simply the closest one. Either way the
        // rest stack in any order.
        if bass_first && stack_order[0] != 0 {
            continue;
        }
        let stack: Vec<(usize, i32)> = stack_order.iter().map(|&i| notes[i]).collect();
        // Unshuffled, the order played breaks ties; shuffled it is drawn and
        // must not choose the shape.
        let reordered = !close_only && stack_order.windows(2).any(|w| w[0] > w[1]);
        for strings in string_sets(notes.len()) {
            let skipped = strings.windows(2).filter(|w| w[1] > w[0] + 1).count() as i32;
            for base in 0..=15 {
                let Some(spots) = stack_grip(&stack, &strings, base, intervals, names) else {
                    continue;
                };
                // A skipped string costs more than any other difference, a
                // stretch to a fifth fret next, and the order played breaks
                // ties with the order stacked.
                let stretched = spots.iter().any(|s| s.fret == base + 4) && spots.iter().any(|s| s.fret == base && base > 0);
                // An open string costs too. They made the closest grips at
                // the nut, so a chain of led grips slid down to it chord by
                // chord, and nothing pulled it back up.
                let open = spots.iter().filter(|s| s.fret == 0).count() as i32;
                let shape = skipped * 60 + stretched as i32 * 30 + open * 12 + reordered as i32 * 8;
                let cost = shape
                    + match prev {
                        // Led from the grip before, and drawn gently back to
                        // the place asked for: a progression led by the
                        // nearest voice keeps stepping down, and over a tune
                        // the hand ended at the nut and stayed there.
                        Some(prev) => {
                            voice_travel(prev, &in_dealt_order(spots.clone(), &stack_order)) + (base - anchor).abs() * 2
                        }
                        // Nothing to lead from: take it where the caller pointed -
                        // the string first, then the fret. Without the string the
                        // lowest set of strings won every tie and a whole progression
                        // sat on the bottom of the neck.
                        // Close forms: the narrowest placement of the stack wins and
                        // the hand position only breaks ties, so the shape still walks
                        // the neck as the anchor is drawn.
                        // Narrow, but not at any price: a semitone narrower
                        // is not worth a stretch to a fifth fret.
                        // The place drawn weighs six to a fret: at one, the
                        // costs of the shape outweighed it and the first grip
                        // of a tune sat at the nut wherever it was asked for.
                        None if close_only => grip_width(&spots) * 8 + (base - anchor).abs() * 6,
                        None => (strings[0] as i32 - from_string as i32).abs() * 24 + (base - anchor).abs() * 6,
                    };
                if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                    best = Some((cost, in_dealt_order(spots, &stack_order)));
                }
            }
        }
    }
    best.map(|(_, s)| s).unwrap_or_else(|| place(root_pc, intervals, steps, names))
}

/// Every order of `n` items, as indices.
fn orders(n: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = vec![Vec::new()];
    for _ in 0..n {
        out = out
            .iter()
            .flat_map(|done| {
                (0..n).filter(|i| !done.contains(i)).map(|i| {
                    let mut next = done.clone();
                    next.push(i);
                    next
                })
            })
            .collect();
    }
    out
}

/// The stack laid on `strings`, one note each, with the hand at fret `base` -
/// or `None` where it does not fit under one hand.
fn stack_grip(
    stack: &[(usize, i32)],
    strings: &[usize],
    base: i32,
    intervals: &[u8],
    names: &[String],
) -> Option<Vec<Spot>> {
    let mut spots = Vec::with_capacity(stack.len());
    let mut last_pitch = i32::MIN;
    for (&(degree, pc), &string) in stack.iter().zip(strings) {
        // The lowest fret at or above the hand's position that plays it.
        let fret = base + (pc - TUNING[string] - base).rem_euclid(12);
        let pitch = TUNING[string] + fret;
        // A grip covers four frets - one for each finger - or five with a
        // stretch, and a voicing rises with the strings: two notes crossing
        // over read as a mistake on paper even when the hand could take them.
        if fret > base + 4 || fret > 17 || pitch <= last_pitch {
            return None;
        }
        last_pitch = pitch;
        spots.push(Spot {
            string,
            fret,
            // The degree, not the distance from the bottom string.
            interval: intervals.get(degree).copied().unwrap_or(0) as i32,
            name: names.get(degree).map(|n| spell(n)),
        });
    }
    Some(spots)
}

/// Voice leading: how far each finger has to travel from the grip before.
/// Strings count too, or a grip two strings over with the same frets would
/// look like no movement at all.
fn voice_travel(prev: &[Spot], spots: &[Spot]) -> i32 {
    prev.iter()
        .zip(spots)
        .map(|(a, b)| (a.fret - b.fret).abs() + (a.string as i32 - b.string as i32).abs() * 2)
        .sum()
}

/// Semitones from the lowest voice of a grip to the highest.
fn grip_width(spots: &[Spot]) -> i32 {
    let pitch = |spot: &Spot| TUNING[spot.string] + spot.fret;
    match (spots.first(), spots.last()) {
        (Some(low), Some(high)) => pitch(high) - pitch(low),
        _ => 0,
    }
}

/// The grip back in the order the steps were dealt in: `collected_notes` is
/// indexed by that, so a grip handed back stacked would light the wrong dots
/// green.
fn in_dealt_order(spots: Vec<Spot>, order: &[usize]) -> Vec<Spot> {
    let mut out = vec![None; spots.len()];
    for (spot, &i) in spots.into_iter().zip(order) {
        out[i] = Some(spot);
    }
    out.into_iter().flatten().collect()
}

/// Sets of `n` strings, low to high, that a hand can take at once: neighbours,
/// or with one string skipped where a voicing wants the room.
fn string_sets(n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(n);
    fn walk(from: usize, n: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == n {
            out.push(current.clone());
            return;
        }
        for s in from..6 {
            // At most one string skipped between voices: further apart and the
            // grip stops being one hand's.
            if current.last().is_some_and(|&last| s > last + 2) {
                break;
            }
            current.push(s);
            walk(s + 1, n, current, out);
            current.pop();
        }
    }
    walk(0, n, &mut current, &mut out);
    out
}

// The drawing, in the numbers the chord diagrams use.
const STRING_GAP: f32 = 48.0;
const DOT_R: f32 = 21.36;
const STEP_X: f32 = 56.0;
const MARGIN_X: f32 = 34.0;
const MARGIN_Y: f32 = 30.0;
/// Room around a grip box: at the left for the nut, and above it for the fret
/// the shape starts on. At the left it also holds an open string's note before
/// the nut, ringed when it is due - at 52 the ring ran off the picture.
const GRIP_LEFT: f32 = 62.0;
/// Room above the box for the fret number - and enough of it that the number
/// clears a dot sitting on the top string, whose centre is ON the box's edge.
const GRIP_TOP: f32 = 88.0;
const FRET_GAP: f32 = 56.0;
/// How far the white ring of the note due reaches from the note's centre:
/// its radius and half its stroke, and a little air.
const RING_REACH: f32 = DOT_R + 7.0 + 2.5 + 3.0;
use crate::diagrams::CARD;
const BLUE: &str = "rgba(74, 144, 226, 1)";
const RED: &str = "rgba(208, 2, 27, 1)";
/// Darker than the green of the interval strip, which is drawn on the black
/// ground and has nothing written on it. Here the degree sits INSIDE the dot,
/// and white on `rgb(50, 255, 50)` cannot be read at all.
const GREEN: &str = "rgba(24, 148, 58, 1)";
/// A string that is on the guitar but not in the question.
const DIM: &str = "rgba(74, 144, 226, 0.55)";
const FONT: &str = "Arial, &quot;Helvetica Neue&quot;, Helvetica, sans-serif";

/// The font the app bundles for ♭, ♯ and ♮ (`assets/Accidentals.ttf`,
/// imported in `appwindow.slint`).
///
/// Arial has none of them, nor does the app's own Noto Sans. Linux finds them
/// in some installed font when the glyph is missing; Windows does not, and the
/// sign was simply not drawn. Slint's SVG renderer looks for a missing glyph
/// only among the generic families, never in the `font-family` list, so the
/// sign has to sit in a span whose own font has it.
const ACCIDENTALS_FONT: &str = "Solitito Accidentals";

/// The text of an SVG with every accidental set in `ACCIDENTALS_FONT`.
pub fn with_accidentals_font(svg: &str) -> String {
    let mut out = String::with_capacity(svg.len());
    for c in svg.chars() {
        if matches!(c, '♭' | '♯' | '♮') {
            out.push_str(&format!("<tspan font-family=\"{ACCIDENTALS_FONT}\">{c}</tspan>"));
        } else {
            out.push(c);
        }
    }
    out
}

/// The start of a drawing `w` by `h`, on the card's colour. The caller closes
/// it with `</svg>`.
fn open_svg(w: f32, h: f32) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" \
         preserveAspectRatio=\"xMidYMid meet\" viewBox=\"0 0 {w} {h}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"{CARD}\"></rect>"
    )
}

fn line(x1: f32, y1: f32, x2: f32, y2: f32, colour: &str) -> String {
    format!(
        "<line x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y2}\" \
         stroke-width=\"2\" stroke=\"{colour}\"></line>"
    )
}

/// The white ring around the note due now.
fn ring(x: f32, y: f32, r: f32, width: u32) -> String {
    format!(
        "<circle r=\"{r}\" cx=\"{x}\" cy=\"{y}\" fill=\"none\" \
         stroke-width=\"{width}\" stroke=\"#ffffff\"></circle>"
    )
}

/// A filled dot with `label` written in it.
fn dot(x: f32, y: f32, fill: &str, label: &str) -> String {
    format!(
        "<circle r=\"{DOT_R}\" cx=\"{x}\" cy=\"{y}\" fill=\"{fill}\" \
         stroke-width=\"0\"></circle>{}",
        dot_label(x, y, label)
    )
}

/// The white writing inside a dot.
fn dot_label(x: f32, y: f32, label: &str) -> String {
    format!(
        "<text x=\"{x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"24\" \
         text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"#ffffff\">{}</text>",
        with_accidentals_font(label)
    )
}

/// Green once played, red for the root, blue for the rest.
fn dot_colour(played: bool, spot: &Spot) -> &'static str {
    if played {
        GREEN
    } else if spot.is_root() {
        RED
    } else {
        BLUE
    }
}

/// The fret a box starts on, written over its first column.
fn fret_number(base: i32) -> String {
    fret_number_at(base, base)
}

/// The inlays of a neck in a box starting at fret `base`: one dot on the 3rd,
/// 5th, 7th, 9th and 15th frets, between the D and G strings, and two on the
/// 12th, a string further out each way. Under the notes, so they never hide
/// one.
fn inlays(base: i32, columns: i32) -> String {
    const INLAY: &str = "rgba(200, 205, 215, 0.22)";
    let between = |low: usize| (string_y(low) + string_y(low + 1)) / 2.0;
    let dot = |fret: i32, y: f32| {
        let x = GRIP_LEFT + FRET_GAP * ((fret - base) as f32 + 0.5);
        format!("<circle cx=\"{x}\" cy=\"{y}\" r=\"9\" fill=\"{INLAY}\"/>")
    };
    let shown = |fret: i32| (base..base + columns).contains(&fret);
    let mut out: String = [3, 5, 7, 9, 15].into_iter().filter(|&f| shown(f)).map(|f| dot(f, between(2))).collect();
    if shown(12) {
        out += &dot(12, between(1));
        out += &dot(12, between(3));
    }
    out
}

/// The number of `fret` over its column, in a box starting at fret `base`.
fn fret_number_at(fret: i32, base: i32) -> String {
    format!(
        "<text x=\"{x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"26\" \
         text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"{BLUE}\">{fret}</text>",
        x = GRIP_LEFT + FRET_GAP * ((fret - base) as f32 + 0.5),
        y = GRIP_TOP - 44.0,
    )
}

/// The size of a box - the grip, the neck, the region - `columns` frets wide.
fn box_size(columns: i32) -> (f32, f32) {
    (
        GRIP_LEFT + FRET_GAP * columns as f32 + MARGIN_X,
        // The ring on the low string reaches past the margin the strip
        // keeps, and was cut along the bottom.
        GRIP_TOP + STRING_GAP * 5.0 + MARGIN_Y.max(RING_REACH),
    )
}

fn box_aspect(columns: i32) -> f32 {
    let (w, h) = box_size(columns);
    w / h
}

/// Where a string runs across a box: the lowest at the bottom.
fn string_y(string: usize) -> f32 {
    GRIP_TOP + (5 - string) as f32 * STRING_GAP
}

/// The middle of the column a fret falls in, kept inside the box.
fn fret_x(fret: i32, base: i32, columns: i32) -> f32 {
    GRIP_LEFT + ((fret - base).clamp(0, columns - 1) as f32 + 0.5) * FRET_GAP
}

/// The fret lines standing up across a box.
fn fret_lines(columns: i32) -> String {
    (0..=columns)
        .map(|i| {
            let x = GRIP_LEFT + i as f32 * FRET_GAP;
            line(x, GRIP_TOP, x, GRIP_TOP + STRING_GAP * 5.0, BLUE)
        })
        .collect()
}

/// How wide the strip is against its height, for laying it out: the drawing
/// grows with the window, and the window has to know its shape to give it a
/// box of the right proportions.
pub fn aspect(notes: usize) -> f32 {
    let n = notes.max(1) as f32;
    (MARGIN_X * 2.0 + (n - 1.0) * STEP_X) / (MARGIN_Y * 2.0 + STRING_GAP * 5.0)
}

/// A grip as a chord box: strings across, frets down, exactly the way the chord
/// diagrams draw one.
///
/// The strip below puts TIME on the horizontal axis - which note comes next -
/// and that says nothing about where the fingers go. For a grip the axis has to
/// be the neck itself, or three notes an octave apart look the same as three
/// notes under one hand.
pub fn grip(spots: &[Spot], done: &[bool], current: usize, frets: bool) -> String {
    let base = spots
        .iter()
        .map(|s| s.fret)
        .min()
        .unwrap_or(1)
        .max(1)
        .min(15);
    let (w, h) = box_size(5);
    let mut out = open_svg(w, h);
    // Laid the way the chord diagrams lie: the neck across the picture, the
    // strings along it with the lowest at the bottom, the frets standing up.
    for i in 0..6 {
        let y = GRIP_TOP + i as f32 * STRING_GAP;
        out += &line(GRIP_LEFT, y, GRIP_LEFT + FRET_GAP * 5.0, y, BLUE);
        let x = GRIP_LEFT + i as f32 * FRET_GAP;
        out += &line(x, GRIP_TOP, x, GRIP_TOP + STRING_GAP * 5.0, BLUE);
    }
    // Which position on the neck, over the first fret - but only where the
    // dots carry fret numbers too. With degrees in them the exercise is the
    // shape and not the place: the same three intervals are the same three
    // intervals wherever the hand is, and a fret number is the crutch that
    // switch exists to put down.
    if frets {
        out += &fret_number(base);
    }
    for (i, spot) in spots.iter().enumerate() {
        let played = done.get(i).copied().unwrap_or(false);
        // The note due now wears a white ring, as it does on the neck.
        //
        // The grip used to say the order by its shape: it was stacked in the
        // order the steps are dealt, so bottom to top WAS the order to play. A
        // close form is not stacked that way - that is the point of it - so
        // without this the picture no longer says which note is being asked
        // for, and the mode asks for them one at a time.
        out += &grip_dot(spot, base, dot_colour(played, spot), i == current, frets);
    }
    out.push_str("</svg>");
    out
}

/// One note of a grip. An open string sits to the left of the nut, as an
/// empty ring: it used to be clamped into the first fret, which put a correct
/// interval label over the wrong physical note.
fn grip_dot(spot: &Spot, base: i32, colour: &str, is_due: bool, frets: bool) -> String {
    let open = spot.fret == 0;
    let x = if open {
        GRIP_LEFT - DOT_R - 7.0
    } else {
        fret_x(spot.fret, base, 5)
    };
    let y = string_y(spot.string);
    // The open-string symbol always names the interval. A bare zero inside it
    // would say less than the empty circle already says.
    let label = if frets && !open { spot.fret.to_string() } else { spot.label() };
    let (fill, stroke_width) = if open { ("none", 5) } else { (colour, 0) };
    let due_ring = if is_due { ring(x, y, DOT_R + 7.0, 5) } else { String::new() };
    format!(
        "{due_ring}<circle r=\"{DOT_R}\" cx=\"{x}\" cy=\"{y}\" fill=\"{fill}\" \
         stroke=\"{colour}\" stroke-width=\"{stroke_width}\"></circle>{}",
        dot_label(x, y, &label)
    )
}

/// The first fret drawn on the neck, and how many columns the box has.
///
/// A fret of neck on either side of the shape: a box cut exactly to the notes
/// reads as if the hand could not move, and where the position sits is half of
/// what the picture is for.
fn neck_columns(spots: &[Spot]) -> (i32, i32) {
    let highest = spots.iter().map(|s| s.fret).max().unwrap_or(0);
    (1, highest.max(crate::fretboard::MAX_FRET as i32))
}

/// A phrase on the neck: the whole of it, from the open strings to the
/// fifteenth fret - or further, where the phrase goes - with its strings named
/// and its inlays, every note drawn where the finger goes, and the one due now
/// ringed in white. A grip is drawn the same way.
///
/// The strip has the ORDER on its horizontal axis, which is right for reading a
/// line but makes a thirty-note study thirty columns wide - and at that width
/// the dots are too small to read. Here the axis is the neck itself, so the
/// picture is the size of a hand however long the phrase is, and what moves is
/// the ring, not the page.
pub fn neck(spots: &[Spot], done: &[bool], current: usize, frets: bool) -> String {
    let (base, columns) = neck_columns(spots);
    let (w, h) = box_size(columns);
    let mut out = open_svg(w, h);
    // An open string's note stands before the nut, where its name would.
    for string in (0..6).rev() {
        let open = spots.iter().any(|s| s.string == string && s.fret == 0);
        out += &region_string(string, columns, true, !open);
    }
    out += &fret_lines(columns);
    out += &inlays(base, columns);
    // One dot per PLACE, not per step: a phrase comes back to the same finger
    // over and over, and six dots on top of each other are one dot with a
    // muddy edge. The step due now decides what the place looks like.
    let mut drawn: Vec<(usize, i32)> = Vec::new();
    for spot in spots {
        let place = (spot.string, spot.fret);
        if drawn.contains(&place) {
            continue;
        }
        drawn.push(place);
        let here: Vec<usize> = (0..spots.len())
            .filter(|&j| (spots[j].string, spots[j].fret) == place)
            .collect();
        // Green from the first time it is played. The other reading - green
        // only once the phrase has no visit left to that place - is truer to
        // the marks, but it leaves the whole climb of an up-and-down phrase
        // colourless, with the ring alone moving. Feedback on the way up is
        // worth more than the distinction, and the ring is what says where the
        // player is on the way back.
        let played = here.iter().any(|&j| done.get(j).copied().unwrap_or(false));
        // An open string is drawn before the nut, as in a grip. In the first
        // column it sat on the first fret, and a note there hid it - the ring
        // then stood on a note other than the one being asked for.
        if spot.fret == 0 {
            out += &grip_dot(&spots[here[0]], base, dot_colour(played, spot), here.contains(&current), frets);
            continue;
        }
        let x = fret_x(spot.fret, base, columns);
        let y = string_y(spot.string);
        // The note due now wears a white ring: the phrase is a line, and the
        // picture has to say where in the line the player is.
        if here.contains(&current) {
            out += &ring(x, y, DOT_R + 7.0, 5);
        }
        let label = if frets { spot.fret.to_string() } else { spots[here[0]].label() };
        out += &dot(x, y, dot_colour(played, spot), &label);
    }
    out.push_str("</svg>");
    out
}

/// The first fret drawn for a region, and how many columns the box has.
fn region_columns(fret_from: i32, fret_to: i32) -> (i32, i32) {
    let base = fret_from.max(1).min(15);
    (base, (fret_to - base + 1).clamp(4, 15))
}

/// The region a fretboard exercise is asking within: the frets it covers and
/// the strings that are in play, with the note marked once it has been found.
///
/// Nothing is marked until the note asked for is played: where it lies IS the
/// exercise, and drawing it first turns the mode into copying a picture. What
/// is drawn then is the answer - the note, green, at every place inside the
/// region it lies. A wrong note is not drawn at all: the one being asked for is
/// on the screen already, and marking strays was more to read than to learn
/// from. `right` is what the colour follows, for a caller that wants both.
///
/// The strings are named down the left so the region is legible at a glance -
/// brighter for the ones in play, dim for the rest of the guitar, which is
/// drawn because leaving it out would move the strings that matter to the wrong
/// place on the neck.
pub fn region(
    strings: &[usize],
    fret_from: i32,
    fret_to: i32,
    marks: &[(usize, i32)],
    label: &str,
    right: bool,
) -> String {
    let fill = if right { GREEN } else { RED };
    region_marked(strings, fret_from, fret_to, marks, label, fill, false)
}

/// The frets a fretboard hint shows: all a region can be drawn in, the open
/// strings to the fifteenth fret.
pub const HINT_FRETS: (i32, i32) = (0, crate::fretboard::MAX_FRET as i32);

/// A hint for the fretboard trainer: the open strings to fret 15, the region's strings
/// bright, and the note asked for at every place on them, in blue - not the
/// green of an answer, since nothing has been played. The neck's inlays
/// count the frets, so no number is written over them.
pub fn neck_hint(strings: &[usize], marks: &[(usize, i32)], label: &str) -> String {
    region_marked(strings, HINT_FRETS.0, HINT_FRETS.1, marks, label, BLUE, true)
}

fn region_marked(
    strings: &[usize],
    fret_from: i32,
    fret_to: i32,
    marks: &[(usize, i32)],
    label: &str,
    fill: &str,
    whole_neck: bool,
) -> String {
    let (base, columns) = region_columns(fret_from, fret_to);
    let (w, h) = box_size(columns);
    let mut out = open_svg(w, h);
    // A note on an open string is drawn before the nut, where the string's
    // name stands - and an open string sounds its own name, so the mark says
    // it in its place.
    let open = |string: usize| marks.contains(&(string, 0));
    for string in (0..6).rev() {
        out += &region_string(string, columns, strings.contains(&string), !open(string));
    }
    out += &fret_lines(columns);
    // A few frets say where they are with the number of the first; the
    // whole neck says it with its inlays.
    if whole_neck {
        out += &inlays(base, columns);
    } else {
        out += &fret_number(base);
    }
    for &(string, fret) in marks {
        out += &if fret == 0 {
            open_dot(string_y(string), fill, label)
        } else {
            dot(fret_x(fret, base, columns), string_y(string), fill, label)
        };
    }
    out.push_str("</svg>");
    out
}

/// A note on an open string, before the nut: an empty circle, as in a grip.
fn open_dot(y: f32, colour: &str, label: &str) -> String {
    let x = GRIP_LEFT - DOT_R - 7.0;
    format!(
        "<circle r=\"{DOT_R}\" cx=\"{x}\" cy=\"{y}\" fill=\"none\" stroke=\"{colour}\" \
         stroke-width=\"5\"></circle>{}",
        dot_label(x, y, label)
    )
}

/// One string of a region, named at the left unless a note there stands in
/// its place: bright when it is in play, dim when it is only there to keep
/// the others in their place.
fn region_string(string: usize, columns: i32, in_play: bool, named: bool) -> String {
    // The high string small, as the region's own line writes it.
    const STRING_NAMES: [&str; 6] = ["E", "A", "D", "G", "B", "e"];
    let colour = if in_play { BLUE } else { DIM };
    let y = string_y(string);
    format!(
        "{}<text x=\"{name_x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"22\" \
         text-anchor=\"end\" dominant-baseline=\"central\" fill=\"{colour}\">{name}</text>",
        line(GRIP_LEFT, y, GRIP_LEFT + FRET_GAP * columns as f32, y, colour),
        name_x = GRIP_LEFT - 12.0,
        name = if named { STRING_NAMES[string] } else { "" },
    )
}

/// How wide a region box is against its height.
pub fn region_aspect(fret_from: i32, fret_to: i32) -> f32 {
    box_aspect(region_columns(fret_from, fret_to).1)
}

/// How wide a neck box is against its height, for the columns it needs.
pub fn neck_aspect(spots: &[Spot]) -> f32 {
    box_aspect(neck_columns(spots).1)
}


/// The strip as an SVG, ready for `Image::load_from_svg_data`.
///
/// `done` is parallel to `spots`: a note already played is green, the same
/// green the interval strip above it uses, so the two rows read as one answer.
///
/// `frets` writes the fret number in each dot instead of the degree - ordinary
/// tablature, for reading a shape onto the neck rather than hearing what it is.
/// The colours do not change with it: the root stays the root.
///
/// `current` is the step due now, ringed in white as it is on the neck. Without
/// it a shuffled scale gave the player no way of knowing which note was being
/// asked for - the dots are in the drawn order, and the order is the point.
pub fn svg(spots: &[Spot], done: &[bool], current: usize, frets: bool) -> String {
    let n = spots.len().max(1) as f32;
    let w = MARGIN_X * 2.0 + (n - 1.0) * STEP_X;
    let h = MARGIN_Y * 2.0 + STRING_GAP * 5.0;
    let mut out = open_svg(w, h);
    // The strings, high one at the top - the way tab is read.
    for i in 0..6 {
        let y = MARGIN_Y + i as f32 * STRING_GAP;
        out += &line(0.0, y, w, y, BLUE);
    }
    for (i, spot) in spots.iter().enumerate() {
        let x = MARGIN_X + i as f32 * STEP_X;
        let y = MARGIN_Y + (5 - spot.string) as f32 * STRING_GAP;
        let played = done.get(i).copied().unwrap_or(false);
        if i == current {
            out += &ring(x, y, DOT_R + 6.0, 4);
        }
        let label = if frets { spot.fret.to_string() } else { spot.label() };
        out += &dot(x, y, dot_colour(played, spot), &label);
    }
    out.push_str("</svg>");
    out
}

#[cfg(test)]
mod tests {
    /// The hint marks the note at every place to the fifteenth fret, in blue,
    /// an open string before the nut in place of its name. `SOLITITO_HINT_SVG` writes it out
    /// to look at.
    #[test]
    fn the_fretboard_hint_marks_every_place_in_blue() {
        let marks: Vec<(usize, i32)> = (0..6)
            .flat_map(|string| (HINT_FRETS.0..=HINT_FRETS.1).map(move |fret| (string, fret)))
            .filter(|&(string, fret)| (TUNING[string] + fret) % 12 == 0)
            .collect();
        assert_eq!(marks.len(), 8, "C on every string, twice on the e strings: {marks:?}");
        let svg = neck_hint(&[0, 1, 2], &marks, "C");
        assert!(svg.matches(BLUE).count() >= 6, "the marks are not blue");
        assert!(!svg.contains(GREEN) && !svg.contains(RED), "a hint is not an answer");
        if let Ok(path) = std::env::var("SOLITITO_HINT_SVG") {
            std::fs::write(path, &svg).unwrap();
        }
        // E on the open low and high strings: two empty circles before the nut.
        let marks: Vec<(usize, i32)> = (0..6)
            .flat_map(|string| (HINT_FRETS.0..=HINT_FRETS.1).map(move |fret| (string, fret)))
            .filter(|&(string, fret)| (TUNING[string] + fret) % 12 == 4)
            .collect();
        assert!(marks.contains(&(0, 0)) && marks.contains(&(5, 0)), "{marks:?}");
        let svg = neck_hint(&[0, 1, 2], &marks, "E");
        assert_eq!(svg.matches("fill=\"none\"").count(), 2, "the open strings are not before the nut");
        if let Ok(path) = std::env::var("SOLITITO_HINT_SVG_OPEN") {
            std::fs::write(path, &svg).unwrap();
        }
    }

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
            let placed = place(root, intervals, &steps, &[]);
            let got: Vec<(usize, i32)> = placed.iter().map(|s| (s.string, s.fret)).collect();
            assert_eq!(got, book.to_vec(), "root {root}");
        }
    }

    /// The fretboard region says which strings are in play, names them, and
    /// marks every place the note lies inside it.
    #[test]
    fn a_region_names_its_strings_and_answers_what_was_played() {
        let strings = [0usize, 1, 2];
        // Nothing played yet: the region alone, and no answer in it.
        let asking = region(&strings, 5, 8, &[], "E", false);
        assert_eq!(asking.matches("<circle").count(), 0, "the answer was given away");
        assert!(asking.contains(DIM), "the strings out of play are not dimmed");
        assert!(asking.contains(">A<") && asking.contains(">G<"), "a string is unnamed");

        // The note asked for: green wherever it lies inside the region.
        let right = region(&strings, 5, 8, &[(0, 7), (1, 7)], "E", true);
        assert_eq!(right.matches(GREEN).count(), 2, "a right answer is not green");
        // Something else: red, and named, so the player sees what they played.
        let wrong = region(&strings, 5, 8, &[(2, 6)], "F", false);
        assert_eq!(wrong.matches(RED).count(), 1, "a wrong answer is not red");
        assert!(wrong.contains(">F<"), "the note played is not named");
    }

    /// A grip is one note per string, and the strings rise with the pitch.
    #[test]
    fn a_voicing_takes_a_string_for_each_note() {
        let steps: Vec<Step> = (0..3).map(|i| Step { degree: i, octave: 0 }).collect();
        for root in 0..12 {
            let spots = place_voiced(root, &[0, 4, 10], &steps, &[], None, 5, 0, false);
            assert_eq!(spots.len(), 3, "root {root}");
            let strings: Vec<usize> = spots.iter().map(|s| s.string).collect();
            let mut sorted = strings.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 3, "root {root}: two notes on one string {strings:?}");
            assert!(strings.windows(2).all(|w| w[0] < w[1]), "root {root}: {strings:?}");
            let frets: Vec<i32> = spots.iter().map(|s| s.fret).collect();
            let span = frets.iter().max().unwrap() - frets.iter().min().unwrap();
            assert!(span <= 4, "root {root}: frets {frets:?} are not one grip");
        }
    }

    /// Taken without leading, a grip goes where the neck is asked for it - so
    /// a progression does not settle on the bottom strings.
    #[test]
    fn a_grip_can_be_asked_for_higher_up() {
        let steps: Vec<Step> = (0..3).map(|i| Step { degree: i, octave: 0 }).collect();
        let low = place_voiced(0, &[0, 4, 10], &steps, &[], None, 5, 0, false);
        let high = place_voiced(0, &[0, 4, 10], &steps, &[], None, 5, 3, false);
        assert_eq!(low[0].string, 0, "the grip did not start where it was asked");
        assert_eq!(high[0].string, 3, "the grip ignored the string it was given");
        // Both are still grips: a string each, rising with the pitch.
        for grip in [&low, &high] {
            let strings: Vec<usize> = grip.iter().map(|s| s.string).collect();
            assert!(strings.windows(2).all(|w| w[0] < w[1]), "{strings:?}");
        }
    }

    /// And from chord to chord the fingers barely move: that is the whole point
    /// of leading the voices rather than taking each grip on its own.
    /// Shuffled, the grip is a close form: the set stacked in its written
    /// order and rotated so the note asked for first is in the bass - root
    /// position and its two inversions.
    ///
    /// Dealt in the shuffled order instead, a set coming out 5 1 3 has to climb
    /// in that order. Measured over 12 roots x 3 chords x all six orders three
    /// notes can be dealt in: 17 to 18 semitones wide before, 9 after; shells
    /// 18 to 20 before, 10 to 11 after. Every one of them now sits inside an
    /// octave with the fingers in four frets.
    #[test]
    fn a_shuffled_grip_is_a_close_form() {
        let orders = [[0usize, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [2, 1, 0], [1, 0, 2]];
        let sets: [[u8; 3]; 6] = [
            [0, 4, 7], [0, 3, 7], [0, 3, 6],          // triads
            [0, 4, 11], [0, 4, 10], [0, 3, 10],       // shells
        ];
        let mut checked = 0;
        for order in orders {
            let steps: Vec<Step> =
                order.iter().map(|&d| Step { degree: d, octave: 0 }).collect();
            for root in 0..12usize {
                for iv in sets {
                    let g = place_voiced(root, &iv, &steps, &[], None, 5, 0, true);
                    assert_eq!(g.len(), 3, "no grip for {order:?} {iv:?}");
                    checked += 1;
                    let p: Vec<i32> = g.iter().map(|s| TUNING[s.string] + s.fret).collect();
                    let (lo, hi) = (*p.iter().min().unwrap(), *p.iter().max().unwrap());
                    assert!(hi - lo < 12, "{} semitones ({order:?} {iv:?} root {root})", hi - lo);
                    let f: Vec<i32> = g.iter().map(|s| s.fret).collect();
                    let stretch = f.iter().max().unwrap() - f.iter().min().unwrap();
                    assert!(stretch <= 4, "{stretch} frets ({order:?} {iv:?} root {root})");
                    // Each dot still belongs to the step that asked for it.
                    for (spot, &d) in g.iter().zip(order.iter()) {
                        assert_eq!(
                            spot.interval, iv[d] as i32,
                            "the grip came back stacked, not in the dealt order"
                        );
                    }
                }
            }
        }
        assert_eq!(checked, 6 * 12 * 6);
    }

    #[test]
    fn every_generated_grip_labels_the_pitch_under_its_dot() {
        let sets: [&[u8]; 5] = [
            &[0, 4, 7], &[0, 3, 7], &[0, 4, 10], &[0, 3, 10], &[0, 3, 6],
        ];
        let orders = [
            [0usize, 1, 2], [0, 2, 1], [1, 0, 2],
            [1, 2, 0], [2, 0, 1], [2, 1, 0],
        ];
        for root in 0..12usize {
            for intervals in sets {
                for order in orders {
                    let steps: Vec<Step> = order.iter()
                        .map(|&degree| Step { degree, octave: 0 })
                        .collect();
                    for anchor in 1..=9 {
                        for from_string in 0..4 {
                            for close_only in [false, true] {
                                let grip = place_voiced(
                                    root, intervals, &steps, &[], None,
                                    anchor, from_string, close_only,
                                );
                                assert_eq!(grip.len(), steps.len());
                                for (spot, step) in grip.iter().zip(steps.iter()) {
                                    assert!((0..=17).contains(&spot.fret));
                                    let actual = (TUNING[spot.string] + spot.fret).rem_euclid(12);
                                    let expected = (root as i32 + intervals[step.degree] as i32)
                                        .rem_euclid(12);
                                    assert_eq!(
                                        actual, expected,
                                        "root {root}, intervals {intervals:?}, order {order:?}, \
                                         anchor {anchor}, string {from_string}, close {close_only}: \
                                         {:?} says {}", spot, spot.label()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn an_open_string_is_drawn_outside_the_nut_as_an_empty_interval_dot() {
        let spots = vec![Spot {
            string: 5,
            fret: 0,
            interval: 7,
            name: Some("5".into()),
        }];
        let svg = grip(&spots, &[false], 1, true);
        let open_x = GRIP_LEFT - DOT_R - 7.0;
        assert!(svg.contains(&format!("cx=\"{open_x}\"")), "open dot is not left of the nut");
        assert!(
            svg.contains("fill=\"none\" stroke=\"rgba(74, 144, 226, 1)\" stroke-width=\"5\""),
            "open dot is not an empty circle"
        );
        assert!(svg.contains(">5</text>"), "open dot lost its interval label");
        assert!(!svg.contains(">0</text>"), "open dot shows a fret instead of an interval");
    }

    #[test]
    fn the_open_fifth_in_the_a7_grip_is_not_drawn_on_a_fret() {
        let steps: Vec<Step> = (0..3)
            .map(|degree| Step { degree, octave: 0 })
            .collect();
        let names = vec!["1".into(), "3".into(), "5".into()];
        let spots = place_voiced(9, &[0, 4, 7], &steps, &names, None, 1, 3, false);
        assert_eq!((spots[0].string, spots[0].fret), (3, 2), "A root moved off G2");
        assert_eq!((spots[2].string, spots[2].fret), (5, 0), "open E fifth moved");
        let svg = grip(&spots, &[false; 3], 3, false);
        let open_x = GRIP_LEFT - DOT_R - 7.0;
        assert!(
            svg.contains(&format!("cx=\"{open_x}\" cy=\"88\" fill=\"none\"")),
            "the open E fifth was drawn inside the fretboard"
        );
    }

    /// A close form no longer says the order by its shape, so the picture has
    /// to say which note is due - the mode asks for them one at a time, and
    /// without the ring there was nothing to read it from.
    #[test]
    fn the_grip_rings_the_note_that_is_due() {
        let steps: Vec<Step> = (0..3).map(|d| Step { degree: d, octave: 0 }).collect();
        let spots = place_voiced(0, &[0, 4, 7], &steps, &[], None, 5, 0, true);
        let done = vec![false; spots.len()];
        let rings = |svg: &str| svg.matches("fill=\"none\"").count();
        for step in 0..spots.len() {
            assert_eq!(rings(&grip(&spots, &done, step, false)), 1, "step {step}");
        }
        // Past the last one - the lap is over - nothing is ringed.
        assert_eq!(rings(&grip(&spots, &done, spots.len(), false)), 0);
        // And the ring sits on the dot the step asks for, not on the lowest.
        let one = grip(&spots, &done, 1, false);
        let y = |s: &Spot| GRIP_TOP + (5 - s.string) as f32 * STRING_GAP;
        let ringed = one
            .split("fill=\"none\"")
            .next()
            .and_then(|head| head.rsplit("cy=\"").next())
            .and_then(|t| t.split('"').next())
            .and_then(|t| t.parse::<f32>().ok())
            .expect("a ring with a position");
        assert!(
            (ringed - y(&spots[1])).abs() < 0.5,
            "the ring is on the wrong string: {ringed} vs {}",
            y(&spots[1])
        );
    }

    /// The position number is a fret number like the ones in the dots, and
    /// goes with them: with degrees showing, the exercise is the shape and not
    /// the place.
    #[test]
    fn the_position_shows_only_with_the_fret_numbers() {
        let steps: Vec<Step> = (0..3).map(|d| Step { degree: d, octave: 0 }).collect();
        let spots = place_voiced(0, &[0, 3, 7], &steps, &[], None, 5, 0, true);
        let done = vec![false; spots.len()];
        // By its own type size, not by its text: a dot can carry the same
        // number as the position, and a plain search finds that instead.
        let positions = |svg: &str| svg.matches("font-size=\"26\"").count();
        assert_eq!(
            positions(&grip(&spots, &done, 0, true)), 1,
            "the position is gone with the fret numbers on"
        );
        assert_eq!(
            positions(&grip(&spots, &done, 0, false)), 0,
            "the position is still drawn with the degrees showing"
        );
    }

    /// Shuffled, the order is drawn and the grip is simply the closest one:
    /// the same shape whichever note comes first, each dot still the step
    /// that asked for it.
    #[test]
    fn a_shuffled_grip_keeps_its_shape() {
        let mut shapes = std::collections::HashSet::new();
        for lead in 0..3usize {
            let mut d: Vec<usize> = (0..3).collect();
            d.rotate_left(lead);
            let steps: Vec<Step> =
                d.iter().map(|&x| Step { degree: x, octave: 0 }).collect();
            let g = place_voiced(0, &[0, 4, 7], &steps, &[], None, 5, 0, true);
            for (spot, &x) in g.iter().zip(&d) {
                assert_eq!(spot.interval, [0, 4, 7][x], "a dot moved to another step");
            }
            let mut places: Vec<(usize, i32)> = g.iter().map(|s| (s.string, s.fret)).collect();
            places.sort();
            shapes.insert(places);
        }
        assert_eq!(shapes.len(), 1, "the shuffle changed the shape: {shapes:?}");
    }

    /// The hand position is still what moves the shape along the neck - it only
    /// stopped being the FIRST thing the search satisfies.
    #[test]
    fn a_close_grip_still_walks_the_neck() {
        let steps: Vec<Step> = (0..3).map(|d| Step { degree: d, octave: 0 }).collect();
        let mut positions = std::collections::HashSet::new();
        for anchor in 1..=9 {
            let g = place_voiced(0, &[0, 4, 7], &steps, &[], None, anchor, 0, true);
            positions.insert(g.iter().map(|s| s.fret).min().unwrap_or(0));
        }
        assert!(positions.len() >= 2, "every position drew the same grip: {positions:?}");
    }

    #[test]
    fn the_voices_are_led_from_one_chord_to_the_next() {
        let steps: Vec<Step> = (0..3).map(|i| Step { degree: i, octave: 0 }).collect();
        // ii V I in C: D m7, G7, C Maj7 - the shell of each, 1 3 7.
        let progression = [(2usize, [0u8, 3, 10]), (7, [0, 4, 10]), (0, [0, 4, 11])];
        let mut prev: Option<Vec<Spot>> = None;
        let mut travelled = 0;
        let mut alone = 0;
        for (root, intervals) in progression {
            let led = place_voiced(root, &intervals, &steps, &[], prev.as_deref(), 5, 0, false);
            let by_itself = place_voiced(root, &intervals, &steps, &[], None, 5, 0, false);
            if let Some(prev) = &prev {
                let cost = |grip: &[Spot]| -> i32 {
                    prev.iter()
                        .zip(grip.iter())
                        .map(|(a, b)| (a.fret - b.fret).abs())
                        .sum()
                };
                travelled += cost(&led);
                alone += cost(&by_itself);
            }
            prev = Some(led);
        }
        assert!(
            travelled < alone,
            "leading the voices moved the hand {travelled} frets, taking each grip on its own {alone}"
        );
    }

    /// A phrase that fits one hand position is given one, in every key and over
    /// every chord. Measured before this: laying it out from the first position
    /// that fits made the hand move 342 times over the whole set of patterns,
    /// twelve keys and three qualities each; choosing the position for the
    /// phrase instead brings that to 266, and the two-octave runs - which do
    /// fit a hand - to none at all.
    #[test]
    fn a_two_octave_phrase_stays_in_one_position() {
        let phrases = [
            "1 3 5 7 1' 3' 5' 7' 5' 3' 1' 7 5 3 1",
            "1 3 5 7 1' 3' 5' 7'",
            "7' 5' 3' 1' 7 5 3 1",
            "1 3 5 7 1' 3' 5' 7' 1'' 7' 5' 3' 1' 7 5 3 1 7, 1",
        ];
        let qualities: [&[u8]; 3] = [&[0, 3, 7, 10], &[0, 4, 7, 11], &[0, 4, 7, 10]];
        for phrase in phrases {
            let names: Vec<String> = phrase.split_whitespace().map(String::from).collect();
            let steps = crate::model::steps_of(&names);
            for quality in qualities {
                for root in 0..12 {
                    let spots = place(root, quality, &steps, &[]);
                    assert_eq!(spots.len(), steps.len(), "{phrase} in {root} lost a note");
                    let frets: Vec<i32> = spots.iter().map(|s| s.fret).collect();
                    let (lo, hi) = (
                        *frets.iter().min().unwrap(),
                        *frets.iter().max().unwrap(),
                    );
                    assert!(
                        hi - lo <= 4,
                        "{phrase} over root {root} spans frets {lo}..{hi} - the hand has to move"
                    );
                    assert!(lo >= 0 && hi <= 17, "{phrase} over root {root} runs off the neck");
                }
            }
        }
    }

    #[test]
    fn the_root_is_red_and_the_labels_are_the_degrees() {
        let steps: Vec<Step> = (0..4).map(|i| Step { degree: i, octave: 0 }).collect();
        let spots = place(9, &[0, 3, 7, 10], &steps, &[]);
        assert_eq!(
            spots.iter().map(|s| s.label()).collect::<Vec<_>>(),
            vec!["1", "♭3", "5", "♭7"]
        );
        assert!(spots[0].is_root() && !spots[1].is_root());
        let out = svg(&spots, &[], usize::MAX, false);
        assert!(out.starts_with("<svg") && out.ends_with("</svg>"));
        assert_eq!(out.matches("<circle").count(), 4, "one dot per note");
        assert_eq!(out.matches("<line").count(), 6, "six strings");
        assert!(out.contains(RED), "the root is not marked");
        // And a step played lights green, root or not.
        let lit = svg(&spots, &[true, true, false, false], usize::MAX, false);
        assert_eq!(lit.matches(GREEN).count(), 2, "the played steps are not lit");
        assert!(!lit.contains(RED), "the root stayed red after it was played");

        // The same drawing with fret numbers instead. Which frets they are is
        // the placer's business - see the tests above - so what is checked here
        // is that numbers are what gets written, and that the root is still
        // marked by colour when its degree is no longer on the dot.
        let numbered = svg(&spots, &[], usize::MAX, true);
        for spot in &spots {
            assert!(
                numbered.contains(&format!(">{}<", spot.fret)),
                "fret {} is not written",
                spot.fret
            );
        }
        assert!(!numbered.contains("♭3"), "a degree was written where a fret was asked for");
        assert!(numbered.contains(RED), "the root stopped being marked");
    }
}

#[cfg(test)]
mod open_string_tests {
    use super::*;

    fn spot(string: usize, fret: i32, interval: i32, name: &str) -> Spot {
        Spot { string, fret, interval, name: Some(name.to_string()) }
    }

    /// The open G and the A flat on its first fret are two places, and both
    /// are drawn: the open one before the nut.
    #[test]
    fn an_open_string_is_not_drawn_on_the_first_fret() {
        let spots = [spot(3, 0, 9, "6"), spot(3, 1, 10, "♭7")];
        let svg = neck(&spots, &[false, false], 0, false);
        let xs: Vec<&str> = svg.match_indices("cx=\"").map(|(i, _)| &svg[i + 4..i + 12]).collect();
        let circles = svg.matches("<circle").count();
        assert!(circles >= 3, "a dot is missing: {circles} circles");
        assert!(svg.contains(&format!("cx=\"{}\"", GRIP_LEFT - DOT_R - 7.0)), "the open string is not before the nut: {xs:?}");
    }
}
