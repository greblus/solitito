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
            // Where the caller named a fret, that is the last word between two
            // positions the hand finds equally easy.
            let cost = match anchor {
                Some(a) => (cost.0, cost.1, (fret - a).abs()),
                None => cost,
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
/// none - the first chord, or the shuffle asking for no leading at all - the
/// grip is taken near `anchor`, which is a fret the caller drew.
pub fn place_voiced(
    root_pc: usize,
    intervals: &[u8],
    steps: &[Step],
    names: &[String],
    prev: Option<&[Spot]>,
    anchor: i32,
) -> Vec<Spot> {
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
    if notes.is_empty() || notes.len() > 4 || notes.len() > 6 {
        return place(root_pc, intervals, steps, names);
    }

    let mut best: Option<(i32, Vec<Spot>)> = None;
    // Every set of strings, low to high, and every hand position on the neck.
    for strings in string_sets(notes.len()) {
        for base in 0..=15 {
            let mut spots = Vec::with_capacity(notes.len());
            let mut last_pitch = i32::MIN;
            let mut ok = true;
            for (i, &(degree, pc)) in notes.iter().enumerate() {
                let string = strings[i];
                // The lowest fret at or above the hand's position that plays it.
                let fret = base + (pc - TUNING[string] - base).rem_euclid(12);
                let pitch = TUNING[string] + fret;
                // A grip covers four frets, and a voicing rises with the
                // strings: two notes crossing over read as a mistake on paper
                // even when the hand could take them.
                if fret > base + 4 || fret > 17 || pitch <= last_pitch {
                    ok = false;
                    break;
                }
                last_pitch = pitch;
                spots.push(Spot {
                    string,
                    fret,
                    interval: pitch - (TUNING[strings[0]] + base),
                    name: names.get(degree).map(|n| spell(n)),
                });
            }
            if !ok {
                continue;
            }
            // The interval each dot carries is its degree, not its distance
            // from the bottom string - recomputed here now the grip is settled.
            for (spot, &(degree, _)) in spots.iter_mut().zip(notes.iter()) {
                spot.interval = intervals.get(degree).copied().unwrap_or(0) as i32;
            }
            let cost = match prev {
                // Voice leading: how far each finger has to travel. Strings
                // count too, or a grip two strings over with the same frets
                // would look like no movement at all.
                Some(prev) => prev
                    .iter()
                    .zip(spots.iter())
                    .map(|(a, b)| (a.fret - b.fret).abs() + (a.string as i32 - b.string as i32).abs() * 2)
                    .sum::<i32>(),
                // Nothing to lead from: take it where the caller pointed.
                None => (base - anchor).abs(),
            };
            if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                best = Some((cost, spots));
            }
        }
    }
    best.map(|(_, s)| s).unwrap_or_else(|| place(root_pc, intervals, steps, names))
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
const BLUE: &str = "rgba(74, 144, 226, 1)";
const RED: &str = "rgba(208, 2, 27, 1)";
/// Darker than the green of the interval strip, which is drawn on the black
/// ground and has nothing written on it. Here the degree sits INSIDE the dot,
/// and white on `rgb(50, 255, 50)` cannot be read at all.
const GREEN: &str = "rgba(24, 148, 58, 1)";
/// A string that is on the guitar but not in the question.
const DIM: &str = "rgba(74, 144, 226, 0.55)";
const FONT: &str = "Arial, &quot;Helvetica Neue&quot;, Helvetica, sans-serif";

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
pub fn grip(spots: &[Spot], done: &[bool], frets: bool) -> String {
    let base = spots
        .iter()
        .map(|s| s.fret)
        .min()
        .unwrap_or(1)
        .max(1)
        .min(15);
    let w = GRIP_LEFT + FRET_GAP * 5.0 + MARGIN_X;
    let h = GRIP_TOP + STRING_GAP * 5.0 + MARGIN_Y;
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" \
         preserveAspectRatio=\"xMidYMid meet\" viewBox=\"0 0 {w} {h}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"rgba(0, 0, 0, 1)\"></rect>"
    );
    // Laid the way the chord diagrams lie: the neck across the picture, the
    // strings along it with the lowest at the bottom, the frets standing up.
    for i in 0..6 {
        let y = GRIP_TOP + i as f32 * STRING_GAP;
        out.push_str(&format!(
            "<line x1=\"{left}\" y1=\"{y}\" x2=\"{right}\" y2=\"{y}\" \
             stroke-width=\"2\" stroke=\"{BLUE}\"></line>",
            left = GRIP_LEFT,
            right = GRIP_LEFT + FRET_GAP * 5.0,
        ));
        let x = GRIP_LEFT + i as f32 * FRET_GAP;
        out.push_str(&format!(
            "<line x1=\"{x}\" y1=\"{top}\" x2=\"{x}\" y2=\"{bottom}\" \
             stroke-width=\"2\" stroke=\"{BLUE}\"></line>",
            top = GRIP_TOP,
            bottom = GRIP_TOP + STRING_GAP * 5.0,
        ));
    }
    // Which position on the neck, over the first fret - a movable shape says
    // nothing without it.
    out.push_str(&format!(
        "<text x=\"{x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"26\" \
         text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"{BLUE}\">{base}</text>",
        x = GRIP_LEFT + FRET_GAP * 0.5,
        y = GRIP_TOP - 44.0,
    ));
    for (i, spot) in spots.iter().enumerate() {
        // Anything outside the five frets drawn is put on the nearest edge
        // rather than off the picture; `place_voiced` keeps grips inside four.
        let column = ((spot.fret - base) as f32).clamp(0.0, 4.0);
        let x = GRIP_LEFT + (column + 0.5) * FRET_GAP;
        let y = GRIP_TOP + (5 - spot.string) as f32 * STRING_GAP;
        let fill = if done.get(i).copied().unwrap_or(false) {
            GREEN
        } else if spot.is_root() {
            RED
        } else {
            BLUE
        };
        out.push_str(&format!(
            "<circle r=\"{DOT_R}\" cx=\"{x}\" cy=\"{y}\" fill=\"{fill}\" stroke-width=\"0\"></circle>\
             <text x=\"{x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"24\" \
             text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"#ffffff\">{label}</text>",
            label = if frets { spot.fret.to_string() } else { spot.label() },
        ));
    }
    out.push_str("</svg>");
    out
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
        return grip(spots, done, frets);
    }
    let lowest = spots.iter().map(|s| s.fret).min().unwrap_or(1);
    let highest = spots.iter().map(|s| s.fret).max().unwrap_or(5);
    // A fret of neck on either side of the shape: a box cut exactly to the
    // notes reads as if the hand could not move, and where the position sits
    // is half of what the picture is for.
    let base = (lowest - 1).max(1).min(15);
    let columns = (highest - base + 2).clamp(6, 13);
    let w = GRIP_LEFT + FRET_GAP * columns as f32 + MARGIN_X;
    let h = GRIP_TOP + STRING_GAP * 5.0 + MARGIN_Y;
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" \
         preserveAspectRatio=\"xMidYMid meet\" viewBox=\"0 0 {w} {h}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"rgba(0, 0, 0, 1)\"></rect>"
    );
    for i in 0..6 {
        let y = GRIP_TOP + i as f32 * STRING_GAP;
        out.push_str(&format!(
            "<line x1=\"{left}\" y1=\"{y}\" x2=\"{right}\" y2=\"{y}\" \
             stroke-width=\"2\" stroke=\"{BLUE}\"></line>",
            left = GRIP_LEFT,
            right = GRIP_LEFT + FRET_GAP * columns as f32,
        ));
    }
    for i in 0..=columns {
        let x = GRIP_LEFT + i as f32 * FRET_GAP;
        out.push_str(&format!(
            "<line x1=\"{x}\" y1=\"{top}\" x2=\"{x}\" y2=\"{bottom}\" \
             stroke-width=\"2\" stroke=\"{BLUE}\"></line>",
            top = GRIP_TOP,
            bottom = GRIP_TOP + STRING_GAP * 5.0,
        ));
    }
    out.push_str(&format!(
        "<text x=\"{x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"26\" \
         text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"{BLUE}\">{base}</text>",
        x = GRIP_LEFT + FRET_GAP * 0.5,
        y = GRIP_TOP - 44.0,
    ));
    // One dot per PLACE, not per step: a phrase comes back to the same finger
    // over and over, and six dots on top of each other are one dot with a
    // muddy edge. The step due now decides what the place looks like.
    let mut drawn: Vec<(usize, i32)> = Vec::new();
    for (i, spot) in spots.iter().enumerate() {
        let place = (spot.string, spot.fret);
        let here: Vec<usize> = spots
            .iter()
            .enumerate()
            .filter(|(_, s)| (s.string, s.fret) == place)
            .map(|(j, _)| j)
            .collect();
        if drawn.contains(&place) {
            continue;
        }
        drawn.push(place);
        let is_current = here.contains(&current);
        // Green from the first time it is played. The other reading - green
        // only once the phrase has no visit left to that place - is truer to
        // the marks, but it leaves the whole climb of an up-and-down phrase
        // colourless, with the ring alone moving. Feedback on the way up is
        // worth more than the distinction, and the ring is what says where the
        // player is on the way back.
        let played = here.iter().any(|&j| done.get(j).copied().unwrap_or(false));
        let fill = if played {
            GREEN
        } else if spot.is_root() {
            RED
        } else {
            BLUE
        };
        let x = GRIP_LEFT + ((spot.fret - base).clamp(0, columns - 1) as f32 + 0.5) * FRET_GAP;
        let y = GRIP_TOP + (5 - spot.string) as f32 * STRING_GAP;
        // The note due now wears a white ring: the phrase is a line, and the
        // picture has to say where in the line the player is.
        let ring = if is_current {
            format!(
                "<circle r=\"{r}\" cx=\"{x}\" cy=\"{y}\" fill=\"none\" \
                 stroke-width=\"5\" stroke=\"#ffffff\"></circle>",
                r = DOT_R + 7.0,
            )
        } else {
            String::new()
        };
        let label = if frets {
            spot.fret.to_string()
        } else {
            spots[here[0]].label()
        };
        out.push_str(&format!(
            "{ring}<circle r=\"{DOT_R}\" cx=\"{x}\" cy=\"{y}\" fill=\"{fill}\" \
             stroke-width=\"0\"></circle>\
             <text x=\"{x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"24\" \
             text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"#ffffff\">{label}</text>",
        ));
        let _ = i;
    }
    out.push_str("</svg>");
    out
}

/// The region a fretboard exercise is asking within: the frets it covers and
/// the strings that are in play, with the note marked once it has been found.
///
/// Nothing is marked until something is played: where the note lies IS the
/// exercise, and drawing it first turns the mode into copying a picture. What
/// is drawn is the ANSWER - green where what sounded is the note asked for, red
/// where it is not, at every place inside the region the sounding note lies.
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
    let base = fret_from.max(1).min(15);
    let columns = (fret_to - base + 1).clamp(4, 13);
    let w = GRIP_LEFT + FRET_GAP * columns as f32 + MARGIN_X;
    let h = GRIP_TOP + STRING_GAP * 5.0 + MARGIN_Y;
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" \
         preserveAspectRatio=\"xMidYMid meet\" viewBox=\"0 0 {w} {h}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"rgba(0, 0, 0, 1)\"></rect>"
    );
    // Strings outside the region are drawn dim: they are part of the guitar and
    // not part of the question, and leaving them out entirely would move the
    // ones that matter to the wrong place on the neck.
    const STRING_NAMES: [&str; 6] = ["E", "A", "D", "G", "B", "E"];
    for i in 0..6 {
        let string = 5 - i;
        let y = GRIP_TOP + i as f32 * STRING_GAP;
        let in_play = strings.contains(&string);
        let colour = if in_play { BLUE } else { DIM };
        out.push_str(&format!(
            "<line x1=\"{left}\" y1=\"{y}\" x2=\"{right}\" y2=\"{y}\" \
             stroke-width=\"2\" stroke=\"{colour}\"></line>\
             <text x=\"{name_x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"22\" \
             text-anchor=\"end\" dominant-baseline=\"central\" fill=\"{colour}\">{name}</text>",
            left = GRIP_LEFT,
            right = GRIP_LEFT + FRET_GAP * columns as f32,
            name_x = GRIP_LEFT - 12.0,
            name = STRING_NAMES[string],
        ));
    }
    for i in 0..=columns {
        let x = GRIP_LEFT + i as f32 * FRET_GAP;
        out.push_str(&format!(
            "<line x1=\"{x}\" y1=\"{top}\" x2=\"{x}\" y2=\"{bottom}\" \
             stroke-width=\"2\" stroke=\"{BLUE}\"></line>",
            top = GRIP_TOP,
            bottom = GRIP_TOP + STRING_GAP * 5.0,
        ));
    }
    out.push_str(&format!(
        "<text x=\"{x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"26\" \
         text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"{BLUE}\">{base}</text>",
        x = GRIP_LEFT + FRET_GAP * 0.5,
        y = GRIP_TOP - 44.0,
    ));
    for &(string, fret) in marks {
        let x = GRIP_LEFT + ((fret - base).clamp(0, columns - 1) as f32 + 0.5) * FRET_GAP;
        let y = GRIP_TOP + (5 - string) as f32 * STRING_GAP;
        let fill = if right { GREEN } else { RED };
        out.push_str(&format!(
            "<circle r=\"{DOT_R}\" cx=\"{x}\" cy=\"{y}\" fill=\"{fill}\" \
             stroke-width=\"0\"></circle>\
             <text x=\"{x}\" y=\"{y}\" font-family=\"{FONT}\" font-size=\"24\" \
             text-anchor=\"middle\" dominant-baseline=\"central\" fill=\"#ffffff\">{label}</text>",
        ));
    }
    out.push_str("</svg>");
    out
}

/// How wide a region box is against its height.
pub fn region_aspect(fret_from: i32, fret_to: i32) -> f32 {
    let base = fret_from.max(1).min(15);
    let columns = (fret_to - base + 1).clamp(4, 13);
    (GRIP_LEFT + FRET_GAP * columns as f32 + MARGIN_X) / (GRIP_TOP + STRING_GAP * 5.0 + MARGIN_Y)
}

/// How wide a neck box is against its height, for the columns it needs.
pub fn neck_aspect(spots: &[Spot]) -> f32 {
    let lowest = spots.iter().map(|s| s.fret).min().unwrap_or(1);
    let highest = spots.iter().map(|s| s.fret).max().unwrap_or(5);
    let base = (lowest - 1).max(1).min(15);
    let columns = (highest - base + 2).clamp(6, 13);
    (GRIP_LEFT + FRET_GAP * columns as f32 + MARGIN_X) / (GRIP_TOP + STRING_GAP * 5.0 + MARGIN_Y)
}

/// How wide a grip box is against its height - see `aspect` for the strip.
pub fn grip_aspect() -> f32 {
    (GRIP_LEFT + FRET_GAP * 5.0 + MARGIN_X) / (GRIP_TOP + STRING_GAP * 5.0 + MARGIN_Y)
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
        if i == current {
            s.push_str(&format!(
                "<circle r=\"{ring}\" cx=\"{x}\" cy=\"{y}\" fill=\"none\" \
                 stroke-width=\"4\" stroke=\"#ffffff\"></circle>",
                ring = DOT_R + 6.0,
            ));
        }
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
            let spots = place_voiced(root, &[0, 4, 10], &steps, &[], None, 5);
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

    /// And from chord to chord the fingers barely move: that is the whole point
    /// of leading the voices rather than taking each grip on its own.
    #[test]
    fn the_voices_are_led_from_one_chord_to_the_next() {
        let steps: Vec<Step> = (0..3).map(|i| Step { degree: i, octave: 0 }).collect();
        // ii V I in C: D m7, G7, C Maj7 - the shell of each, 1 3 7.
        let progression = [(2usize, [0u8, 3, 10]), (7, [0, 4, 10]), (0, [0, 4, 11])];
        let mut prev: Option<Vec<Spot>> = None;
        let mut travelled = 0;
        let mut alone = 0;
        for (root, intervals) in progression {
            let led = place_voiced(root, &intervals, &steps, &[], prev.as_deref(), 5);
            let by_itself = place_voiced(root, &intervals, &steps, &[], None, 5);
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
