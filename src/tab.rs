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
    // A hand covers four frets, read from one below the note it starts on.
    let (mut lo, mut hi) = (start_fret - 1, start_fret + 3);
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
                lo = fret - 1;
                hi = fret + 3;
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
    let order = voice_order(&notes, close_only);
    let stack: Vec<(usize, i32)> = order.iter().map(|&i| notes[i]).collect();

    let mut best: Option<(i32, Vec<Spot>)> = None;
    // Every set of strings, low to high, and every hand position on the neck.
    for strings in string_sets(notes.len()) {
        for base in 0..=15 {
            let Some(spots) = stack_grip(&stack, &strings, base, intervals, names) else {
                continue;
            };
            let cost = match prev {
                Some(prev) => voice_travel(prev, &spots),
                // Nothing to lead from: take it where the caller pointed -
                // the string first, then the fret. Without the string the
                // lowest set of strings won every tie and a whole progression
                // sat on the bottom of the neck.
                // Close forms: the narrowest placement of the stack wins and
                // the hand position only breaks ties, so the shape still walks
                // the neck as the anchor is drawn. Which string the bass lands
                // on is not asked for - the rotation has already chosen which
                // NOTE is in the bass, and asking for both left the search
                // nothing it could satisfy.
                None if close_only => grip_width(&spots) * 24 + (base - anchor).abs(),
                None => (strings[0] as i32 - from_string as i32).abs() * 24 + (base - anchor).abs(),
            };
            if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                best = Some((cost, in_dealt_order(spots, &order)));
            }
        }
    }
    best.map(|(_, s)| s).unwrap_or_else(|| place(root_pc, intervals, steps, names))
}

/// Which voice sits where, bottom up, as indices into `notes`.
///
/// Normally the order the steps are DEALT in, so a set written 1 3 5 is
/// stacked in that order, each voice above the one below it.
///
/// That is what spread the grips out under the shuffle: a set dealt 5 1 3 has
/// to climb in that order, which is an octave and a half wide and not a grip
/// anybody plays. Close forms instead stack the set in its written order and
/// ROTATE it, so the note asked for first is the bass - root position and its
/// two inversions for a triad, and the same for a shell. Measured over 12 roots
/// x 3 qualities x 3 inversions: every triad comes out between 6 and 9
/// semitones wide.
fn voice_order(notes: &[(usize, i32)], close_only: bool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..notes.len()).collect();
    let mut degrees: Vec<usize> = notes.iter().map(|&(degree, _)| degree).collect();
    degrees.sort_unstable();
    let distinct = degrees.windows(2).all(|w| w[0] != w[1]);
    if !close_only || !distinct {
        return order;
    }
    order.sort_by_key(|&i| notes[i].0);
    let lead = order.iter().position(|&i| i == 0).unwrap_or(0);
    order.rotate_left(lead);
    order
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
        // A grip covers four frets, and a voicing rises with the strings: two
        // notes crossing over read as a mistake on paper even when the hand
        // could take them.
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
/// the shape starts on.
const GRIP_LEFT: f32 = 52.0;
/// Room above the box for the fret number - and enough of it that the number
/// clears a dot sitting on the top string, whose centre is ON the box's edge.
const GRIP_TOP: f32 = 88.0;
const FRET_GAP: f32 = 56.0;
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
         text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"#ffffff\">{label}</text>"
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
    format!(
        "<text x=\"{x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"26\" \
         text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"{BLUE}\">{base}</text>",
        x = GRIP_LEFT + FRET_GAP * 0.5,
        y = GRIP_TOP - 44.0,
    )
}

/// The size of a box - the grip, the neck, the region - `columns` frets wide.
fn box_size(columns: i32) -> (f32, f32) {
    (
        GRIP_LEFT + FRET_GAP * columns as f32 + MARGIN_X,
        GRIP_TOP + STRING_GAP * 5.0 + MARGIN_Y,
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
    let lowest = spots.iter().map(|s| s.fret).min().unwrap_or(1);
    let highest = spots.iter().map(|s| s.fret).max().unwrap_or(5);
    let base = (lowest - 1).max(1).min(15);
    (base, (highest - base + 2).clamp(6, 13))
}

/// A phrase on the neck: the position it is played in, every note of it drawn
/// where the finger goes, and the one due now ringed in white.
///
/// The strip has the ORDER on its horizontal axis, which is right for reading a
/// line but makes a thirty-note study thirty columns wide - and at that width
/// the dots are too small to read. Here the axis is the neck itself, so the
/// picture is the size of a hand however long the phrase is, and what moves is
/// the ring, not the page.
pub fn neck(spots: &[Spot], done: &[bool], current: usize, frets: bool) -> String {
    if spots.is_empty() {
        return grip(spots, done, current, frets);
    }
    let (base, columns) = neck_columns(spots);
    let (w, h) = box_size(columns);
    let mut out = open_svg(w, h);
    for i in 0..6 {
        let y = GRIP_TOP + i as f32 * STRING_GAP;
        out += &line(GRIP_LEFT, y, GRIP_LEFT + FRET_GAP * columns as f32, y, BLUE);
    }
    out += &fret_lines(columns);
    out += &fret_number(base);
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
    (base, (fret_to - base + 1).clamp(4, 13))
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
    let (base, columns) = region_columns(fret_from, fret_to);
    let (w, h) = box_size(columns);
    let mut out = open_svg(w, h);
    for string in (0..6).rev() {
        out += &region_string(string, columns, strings.contains(&string));
    }
    out += &fret_lines(columns);
    out += &fret_number(base);
    let fill = if right { GREEN } else { RED };
    for &(string, fret) in marks {
        out += &dot(fret_x(fret, base, columns), string_y(string), fill, label);
    }
    out.push_str("</svg>");
    out
}

/// One string of a region, named at the left: bright when it is in play, dim
/// when it is only there to keep the others in their place.
fn region_string(string: usize, columns: i32, in_play: bool) -> String {
    const STRING_NAMES: [&str; 6] = ["E", "A", "D", "G", "B", "E"];
    let colour = if in_play { BLUE } else { DIM };
    let y = string_y(string);
    format!(
        "{}<text x=\"{name_x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"22\" \
         text-anchor=\"end\" dominant-baseline=\"central\" fill=\"{colour}\">{name}</text>",
        line(GRIP_LEFT, y, GRIP_LEFT + FRET_GAP * columns as f32, y, colour),
        name_x = GRIP_LEFT - 12.0,
        name = STRING_NAMES[string],
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

/// How wide a grip box is against its height - see `aspect` for the strip.
pub fn grip_aspect() -> f32 {
    box_aspect(5)
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
        assert!(svg.contains("cx=\"23.64\""), "open dot is not left of the nut");
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
        assert!(
            svg.contains("cx=\"23.64\" cy=\"88\" fill=\"none\""),
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

    /// And the shuffle still changes the shape: three inversions, not one grip
    /// drawn over and over.
    #[test]
    fn the_shuffle_chooses_which_note_is_in_the_bass() {
        let mut basses = std::collections::HashSet::new();
        for lead in 0..3usize {
            let mut d: Vec<usize> = (0..3).collect();
            d.rotate_left(lead);
            let steps: Vec<Step> =
                d.iter().map(|&x| Step { degree: x, octave: 0 }).collect();
            let g = place_voiced(0, &[0, 4, 7], &steps, &[], None, 5, 0, true);
            let low = g.iter().min_by_key(|s| TUNING[s.string] + s.fret).unwrap();
            basses.insert(low.interval);
        }
        assert_eq!(basses.len(), 3, "the shuffle drew the same inversion: {basses:?}");
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
