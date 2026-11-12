//! Lines over the changes in the manner Barry Harris taught them.
//!
//! The idea everything here rests on: an eight-note scale played in eighth
//! notes from a chord tone on the beat keeps every chord tone on a beat,
//! because chord tones and the notes between them alternate. A plain
//! seven-note scale drifts off the beat after four notes; one chromatic
//! passing note puts it back.
//!
//! Each chord therefore gets a ring of eight notes, written from its own root,
//! in which every EVEN place is a tone on the beat:
//!
//! - a dominant seventh: the bebop dominant scale, the major seventh passing
//!   between the flat seventh and the root;
//! - a major seventh, a minor sixth and a minor-major seventh: the sixth
//!   diminished scale - the sixth chord on the beats, the diminished chord
//!   built on the seventh between them;
//! - a minor seventh: thought of as the dominant a fourth up, its scale
//!   played from the minor chord's root, so the major third passes;
//! - a half-diminished seventh: thought of as the dominant a major third
//!   below, the ninth chord it is the top of;
//! - a diminished seventh: the diminished scale, eight notes of its own.
//!
//! A figure is a path through the ring, counted in its places: down the
//! scale, up it, the arpeggio up and the scale down, the arpeggio from the
//! third of the chord the line thinks in or from the chord's own, and a tone
//! enclosed - the scale note above it and the chromatic one below. Places
//! are plain integers - eight up is the same place an octave higher. A note
//! the ring does not hold is a semitone above the root instead, and a step
//! names it past the ring's eight - see `names`.
//!
//! Before the lines come what they are made of. Six figures are cells on the
//! chord's own tones - the third, the triad, the seventh chord, the third
//! with a note leading into it, the arpeggio with one, the arpeggio turned
//! on its root - and are taken through the chords of a key, degree by
//! degree, as the scale is played "in chords", or over the chords of a tune.

use crate::model::{Chord, ChordQuality, NoteName, Step};

/// The eight notes over one chord, from its root up, and which chord the line
/// thinks in.
pub struct Ring {
    /// Each note named from the chord's root. A chromatic passing note is in
    /// brackets, as the passing note of the Intervals scales is.
    pub names: [&'static str; 8],
    /// Semitones above the chord's root, in the order of `names`.
    pub semitones: [u8; 8],
    /// The place of the root of the chord the line thinks in: the dominant a
    /// fourth up over a minor seventh, a major third down over a
    /// half-diminished one, the chord itself everywhere else.
    pub thought_root: i32,
}

const DOMINANT: Ring = Ring {
    names: ["1", "2", "3", "4", "5", "6", "b7", "(7)"],
    semitones: [0, 2, 4, 5, 7, 9, 10, 11],
    thought_root: 0,
};
const MAJOR_SIXTH_DIMINISHED: Ring = Ring {
    names: ["1", "2", "3", "4", "5", "(#5)", "6", "7"],
    semitones: [0, 2, 4, 5, 7, 8, 9, 11],
    thought_root: 0,
};
const MINOR_SIXTH_DIMINISHED: Ring = Ring {
    names: ["1", "2", "b3", "4", "5", "(#5)", "6", "7"],
    semitones: [0, 2, 3, 5, 7, 8, 9, 11],
    thought_root: 0,
};
/// G minor seventh played as C seventh: G A Bb B C D E F.
const MINOR_AS_DOMINANT: Ring = Ring {
    names: ["1", "2", "b3", "(3)", "4", "5", "6", "b7"],
    semitones: [0, 2, 3, 4, 5, 7, 9, 10],
    thought_root: 4,
};
/// A half-diminished played as F seventh: A Bb C D Eb E F G.
const HALF_DIMINISHED_AS_DOMINANT: Ring = Ring {
    names: ["1", "b2", "b3", "4", "b5", "(5)", "b6", "b7"],
    semitones: [0, 1, 3, 5, 6, 7, 8, 10],
    thought_root: 6,
};
const DIMINISHED: Ring = Ring {
    names: ["1", "2", "b3", "4", "b5", "b6", "bb7", "7"],
    semitones: [0, 2, 3, 5, 6, 8, 9, 11],
    thought_root: 0,
};

/// The ring over a chord; none over a chord of a user's own scale.
pub fn ring(quality: &ChordQuality) -> Option<&'static Ring> {
    match quality {
        ChordQuality::Dominant7 => Some(&DOMINANT),
        ChordQuality::Major7 => Some(&MAJOR_SIXTH_DIMINISHED),
        ChordQuality::Minor6 | ChordQuality::MinorMajor7 => Some(&MINOR_SIXTH_DIMINISHED),
        ChordQuality::Minor7 => Some(&MINOR_AS_DOMINANT),
        ChordQuality::HalfDiminished => Some(&HALF_DIMINISHED_AS_DOMINANT),
        ChordQuality::Diminished => Some(&DIMINISHED),
        ChordQuality::CustomScale(_) => None,
    }
}

impl Ring {
    /// Semitones above the chord's root of a place, octaves included.
    fn pitch(&self, place: i32) -> i32 {
        self.semitones[place.rem_euclid(8) as usize] as i32 + 12 * place.div_euclid(8)
    }

    /// The ring without its passing note: the chord's plain scale, in
    /// semitones above the root.
    fn scale(&self) -> Vec<i32> {
        self.semitones.iter().zip(self.names).filter(|(_, name)| !name.starts_with('(')).map(|(&s, _)| s as i32).collect()
    }
}

/// The twelve semitones above the root by name, for the notes a figure takes
/// from outside the ring - the chromatic note under an enclosed tone, the
/// tones of the chord's own arpeggio. Spelled with sharps, as notes leading
/// upwards are, except the chord's own tones - see `names`.
const CHROMATIC: [&str; 12] = ["1", "#1", "2", "#2", "3", "4", "#4", "5", "#5", "6", "#6", "7"];

/// What a step of a Harris line indexes: the ring's eight notes, then the
/// twelve semitones of `CHROMATIC` - where one of them is a tone of the
/// chord, named as the chord names it: over Dm7 the F of its own arpeggio is
/// its `b3`, not a `#2`.
#[cfg(test)]
pub fn names(ring: &Ring, quality: &ChordQuality) -> Vec<String> {
    names_in_scale(ring, quality, &ring.scale())
}

/// `names` over a chord of a line played over `harmony`: the notes of the
/// scale the line walks there are named as its degrees - over Em7 in C the F
/// is `b2`, not `#1`.
pub fn names_over(chord: &Chord, harmony: Harmony) -> Option<Vec<String>> {
    let ring = ring(&chord.quality)?;
    Some(names_in_scale(ring, &chord.quality, &scale_over(chord, ring, harmony)))
}

fn names_in_scale(ring: &Ring, quality: &ChordQuality, scale: &[i32]) -> Vec<String> {
    let mut chromatic: Vec<String> = CHROMATIC.iter().map(|n| n.to_string()).collect();
    // A scale of seven notes names each by its number, flat or sharp of the
    // major scale's.
    if scale.len() == 7 {
        for (degree, &semitone) in scale.iter().enumerate() {
            let sign = match semitone - MAJOR[degree] {
                -2 => "bb",
                -1 => "b",
                1 => "#",
                _ => "",
            };
            chromatic[semitone as usize % 12] = format!("{sign}{}", degree + 1);
        }
    }
    for (name, &semitone) in quality.interval_names().iter().zip(quality.intervals().iter()) {
        chromatic[semitone as usize % 12] = name.clone();
    }
    ring.names.iter().map(|n| n.to_string()).chain(chromatic).collect()
}

/// The major scale, in semitones.
const MAJOR: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];

/// What a line is played over, beyond the chord under it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Harmony {
    /// The chords of a tune, and its tonic where one is known: a chord that
    /// lies in the tune's key takes its steps from that key's scale.
    Tune(Option<usize>),
    /// The chords of a major key, degree by degree - see `chords_of_key`: one
    /// cell of a figure over each, every step from the key's scale.
    Key(usize),
}

impl Default for Harmony {
    fn default() -> Self {
        Harmony::Tune(None)
    }
}

/// The seventh chords of a major key from its tonic up to the octave: the
/// scale "in chords".
pub fn chords_of_key(key: usize) -> Vec<Chord> {
    use ChordQuality::{Dominant7, HalfDiminished, Major7, Minor7};
    let qualities = [Major7, Minor7, Minor7, Major7, Dominant7, Minor7, HalfDiminished, Major7];
    MAJOR
        .iter()
        .chain(&MAJOR[..1])
        .zip(qualities)
        .map(|(&degree, quality)| Chord { root: NoteName::from_index(key + degree as usize), quality })
        .collect()
}

/// The scale a line's steps walk over `chord`, in semitones above its root
/// and ascending from it: the key's where the chord lies in the key - a minor
/// tune's by its relative major - and the ring's own, less its passing note,
/// everywhere else.
fn scale_over(chord: &Chord, ring: &Ring, harmony: Harmony) -> Vec<i32> {
    let root = chord.root as i32;
    let keys: Vec<i32> = match harmony {
        Harmony::Key(key) => vec![key as i32],
        Harmony::Tune(Some(tonic)) => vec![tonic as i32, tonic as i32 + 3],
        Harmony::Tune(None) => Vec::new(),
    };
    let tones: Vec<i32> = chord.quality.intervals().iter().map(|&i| i as i32).collect();
    for key in keys {
        let mut scale: Vec<i32> = MAJOR.iter().map(|degree| (key + degree - root).rem_euclid(12)).collect();
        scale.sort_unstable();
        if tones.iter().all(|tone| scale.contains(&tone.rem_euclid(12))) {
            return scale;
        }
    }
    ring.scale()
}

/// Semitones above the root, in the order of `names`.
pub fn semitones(ring: &Ring) -> Vec<u8> {
    ring.semitones.iter().copied().chain(0..12).collect()
}

/// The path a figure takes. Saved as its index, 0 to 13, in the order of the
/// list: the six cells, then the lines. The plain arpeggios are the arpeggio
/// over the changes, not a Harris figure, and not saved.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum Figure {
    /// A note and the third above it in the scale: over C in C major C E,
    /// and on up the scale - D F, E G.
    Thirds,
    /// The triad on a chord tone: C E G.
    Triads,
    /// The seventh chord on a chord tone: C E G B, and from the third the
    /// arpeggio that adds the ninth - E G B D.
    Sevenths,
    /// A third with the note a half step under it leading in: B C E.
    LedThirds,
    /// The seventh chord with that leading note before it: B C E G B.
    BebopArpeggio,
    /// The seventh chord turned on its root - the root an octave up, the
    /// rest from below - and a step down the scale to end: C' E G B A.
    Pivot,
    /// Down the scale from a tone on the beat.
    #[default]
    ScaleDown,
    /// Up the scale from a tone on the beat.
    ScaleUp,
    /// Up the arpeggio, the octave on the beat, and down the scale.
    ArpeggioUpScaleDown,
    /// Up the arpeggio from the third of the chord the line thinks in - its
    /// third, fifth, seventh and ninth - and down the scale from the octave.
    FromTheThird,
    /// Up the arpeggio from the chord's OWN third - over Dm7 F A C E, the
    /// Fmaj7 inside it - and down the scale.
    FromTheChordsThird,
    /// A chord tone enclosed - the scale note above it, the note a half step
    /// below - then up its arpeggio to the seventh and a step down the scale:
    /// over Cmaj7 D B C E G B A.
    Enclosure,
    /// From a chord tone down to the half step under the next third an octave
    /// lower, into that third, up to the seventh and two steps down the
    /// scale: over Cmaj7 C D# E B A G.
    DropToTheThird,
    /// From the third into the root, enclosed - the half step below it, the
    /// scale note above: over Cmaj7 E B D C.
    ThirdIntoTheRoot,
    /// The chord's own four tones up from a chord tone, two octaves - the
    /// arpeggio over the changes.
    ArpeggioUp,
    /// The same down.
    ArpeggioDown,
}

impl From<usize> for Figure {
    fn from(index: usize) -> Self {
        match index {
            1 => Figure::Triads,
            2 => Figure::Sevenths,
            3 => Figure::LedThirds,
            4 => Figure::BebopArpeggio,
            5 => Figure::Pivot,
            6 => Figure::ScaleDown,
            7 => Figure::ScaleUp,
            8 => Figure::ArpeggioUpScaleDown,
            9 => Figure::FromTheThird,
            10 => Figure::FromTheChordsThird,
            11 => Figure::Enclosure,
            12 => Figure::DropToTheThird,
            13 => Figure::ThirdIntoTheRoot,
            14 => Figure::ArpeggioUp,
            15 => Figure::ArpeggioDown,
            _ => Figure::Thirds,
        }
    }
}

/// Where the figure over each chord starts. Saved as its index, 0 to 3.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum Start {
    /// The chord's root, every chord.
    #[default]
    Root,
    /// The chord's third, every chord.
    Third,
    /// The tones on the beat in turn: root, third, then the other two.
    InTurn,
    /// One line through the whole tune: each chord starts on its tone on the
    /// beat nearest the last note, the way the line was going.
    Continuous,
}

impl From<usize> for Start {
    fn from(index: usize) -> Self {
        match index {
            1 => Start::Third,
            2 => Start::InTurn,
            3 => Start::Continuous,
            _ => Start::Root,
        }
    }
}

/// One note of a figure: a place in the ring, or a semitone above the
/// chord's root that the ring may not hold.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Note {
    Place(i32),
    Semitone(i32),
}

/// The chord a figure is played over: its ring, its own four tones, its
/// root, and the scale its steps walk.
struct Over<'a> {
    ring: &'a Ring,
    own: [i32; 4],
    root: i32,
    /// In semitones above the root, ascending from it - see `scale_over`.
    scale: Vec<i32>,
    /// One cell of a figure over this chord, not as many as a bar holds: the
    /// chords of a key walk the scale themselves.
    one_cell: bool,
}

impl Over<'_> {
    fn new(chord: &Chord, harmony: Harmony) -> Option<Over<'_>> {
        let ring = ring(&chord.quality)?;
        let tones = chord.quality.intervals();
        let own = [0, 1, 2, 3].map(|k| tones.get(k).copied().unwrap_or(0) as i32);
        let scale = scale_over(chord, ring, harmony);
        Some(Over { ring, own, root: chord.root as i32, scale, one_cell: matches!(harmony, Harmony::Key(_)) })
    }

    /// The pitch of a note, in semitones from C in the octave the line counts
    /// from - see `C3`.
    fn pitch(&self, note: Note) -> i32 {
        self.root
            + match note {
                Note::Place(p) => self.ring.pitch(p),
                Note::Semitone(s) => s,
            }
    }

    /// A tone of the chord stacked in thirds from the root: 0 the root, 1 the
    /// third, 2 the fifth, 3 the seventh, then the ninth, eleventh and
    /// thirteenth, and on again two octaves up.
    ///
    /// The three above the seventh are the scale's: over Em7 in C the ninth
    /// is F. A scale that is not seven notes - the diminished - has the
    /// major ninth, the eleventh and the thirteenth instead.
    fn tone(&self, k: i32) -> i32 {
        let upper = |n: i32, plain: i32| self.degree(n).unwrap_or(plain);
        let stack = [self.own[0], self.own[1], self.own[2], self.own[3], upper(8, 14), upper(10, 17), upper(12, 21)];
        stack[k.rem_euclid(7) as usize] + 24 * k.div_euclid(7)
    }

    /// The note `n` steps up the scale from the root, where the scale is
    /// seven notes.
    fn degree(&self, n: i32) -> Option<i32> {
        (self.scale.len() == 7).then(|| self.scale[n.rem_euclid(7) as usize] + 12 * n.div_euclid(7))
    }

    /// Which step of the scale chord tone `k` is - the root none, the third
    /// two - by where the tone lies in it: over a minor sixth the fourth tone
    /// is the sixth, five steps up.
    fn degree_of_tone(&self, k: i32) -> i32 {
        let tone = self.own[k.rem_euclid(4) as usize];
        self.scale.iter().position(|&note| note == tone).map_or(2 * k, |at| at as i32)
    }

    /// Whether `semitones` above the root is in the scale the steps walk.
    fn in_scale(&self, semitones: i32) -> bool {
        self.scale.contains(&semitones.rem_euclid(12))
    }

    /// The next note of the scale above, and below, `semitones`.
    fn scale_above(&self, semitones: i32) -> i32 {
        (semitones + 1..semitones + 13).find(|&s| self.in_scale(s)).unwrap_or(semitones + 2)
    }

    fn scale_below(&self, semitones: i32) -> i32 {
        (semitones - 12..semitones).rev().find(|&s| self.in_scale(s)).unwrap_or(semitones - 2)
    }

    /// The highest tone on the beat below `semitones` above the root: where
    /// the scale goes down from, so its tones on the beat stay on the beat.
    fn place_below(&self, semitones: i32) -> i32 {
        (-32..48).filter(|&p| p % 2 == 0 && self.ring.pitch(p) < semitones).max().unwrap_or(0)
    }
}

impl Figure {
    /// The figure from `start`. The scale figures are eight notes - a bar of
    /// eighth notes - and `start` is a place in the ring; the arpeggio from
    /// the chord's own third takes it as the octave. The phrases - the
    /// enclosure, the drop, the third into the root - are shorter, built on
    /// the chord's own tones and its plain scale, and `start` is a chord tone
    /// with its octave: 0 to 3 the root to the seventh, 4 up an octave.
    fn notes(self, over: &Over, start: i32) -> Vec<Note> {
        use Note::{Place, Semitone};
        let s = start;
        // A chord tone `k` tones above the one the phrase starts from.
        let (k, octave) = (s.rem_euclid(4), 12 * s.div_euclid(4));
        let tone = |up: i32| over.tone(k + up) + octave;
        // The cells a bar holds; one where the chords walk a key.
        let cells = |bar: i32| if over.one_cell { 1 } else { bar };
        // The note `thirds` thirds above the one `up` steps up the scale from
        // the chord tone. A scale that is not seven notes - the diminished -
        // has no steps to count: there the chord's own tones, a tone further
        // on with every cell.
        let stacked = |up: i32, thirds: i32| match over.degree(over.degree_of_tone(k) + up + 2 * thirds) {
            Some(note) => note + octave,
            None => tone(up + thirds),
        };
        match self {
            // The thirds and the triads go on up the SCALE from the chord
            // tone, a cell on each note of it, as they do through a key:
            // stacked on the chord's tones instead, the second triad begins
            // with two notes of the first.
            Figure::Thirds => {
                (0..cells(4)).flat_map(|i| [stacked(i, 0), stacked(i, 1)]).map(Semitone).collect()
            }
            Figure::Triads => {
                (0..cells(2)).flat_map(|i| [stacked(i, 0), stacked(i, 1), stacked(i, 2)]).map(Semitone).collect()
            }
            Figure::LedThirds => {
                (0..cells(2)).flat_map(|i| [stacked(i, 0) - 1, stacked(i, 0), stacked(i, 1)]).map(Semitone).collect()
            }
            // The seventh chords go on up the CHORD: the second is the one
            // from the third, three notes the chord's own and the ninth.
            Figure::Sevenths => {
                (0..cells(2)).flat_map(|i| [tone(i), tone(i + 1), tone(i + 2), tone(i + 3)]).map(Semitone).collect()
            }
            Figure::BebopArpeggio => [tone(0) - 1, tone(0), tone(1), tone(2), tone(3)].map(Semitone).to_vec(),
            Figure::Pivot => {
                let seventh = tone(3);
                [tone(0) + 12, tone(1), tone(2), seventh, over.scale_below(seventh)].map(Semitone).to_vec()
            }
            Figure::ScaleDown => (0..8).map(|i| Place(s - i)).collect(),
            Figure::ScaleUp => (0..8).map(|i| Place(s + i)).collect(),
            Figure::ArpeggioUpScaleDown => [0, 2, 4, 6, 8, 7, 6, 5].map(|i| Place(s + i)).to_vec(),
            // The ninth is one place above the octave of the root; from it
            // the octave of the root is the next tone on the beat.
            Figure::FromTheThird => [0, 2, 4, 7, 6, 5, 4, 3].map(|i| Place(s + i)).to_vec(),
            Figure::FromTheChordsThird => {
                let [_, third, fifth, seventh] = over.own;
                let octave = 12 * s;
                let ninth = octave + 14;
                let below = over.place_below(ninth);
                vec![
                    Semitone(octave + third),
                    Semitone(octave + fifth),
                    Semitone(octave + seventh),
                    Semitone(ninth),
                    Place(below),
                    Place(below - 1),
                    Place(below - 2),
                    Place(below - 3),
                ]
            }
            Figure::Enclosure => {
                let target = tone(0);
                let seventh = tone(3);
                [over.scale_above(target), target - 1, target, tone(1), tone(2), seventh, over.scale_below(seventh)]
                    .map(Semitone)
                    .to_vec()
            }
            Figure::DropToTheThird => {
                let third = tone(1) - 12;
                let seventh = tone(3) - 12;
                let sixth = over.scale_below(seventh);
                [tone(0), third - 1, third, seventh, sixth, over.scale_below(sixth)].map(Semitone).to_vec()
            }
            Figure::ThirdIntoTheRoot => {
                let target = tone(0);
                [tone(1), target - 1, over.scale_above(target), target].map(Semitone).to_vec()
            }
            // The four tones round and round, an octave up every time round.
            Figure::ArpeggioUp | Figure::ArpeggioDown => {
                let way = if self == Figure::ArpeggioUp { 1 } else { -1 };
                (0..8)
                    .map(|i| {
                        let at = k + way * i;
                        Semitone(over.own[at.rem_euclid(4) as usize] + 12 * at.div_euclid(4) + octave)
                    })
                    .collect()
            }
        }
    }

    /// The same path the other way, for a scale; the arpeggio figures have
    /// no other way.
    fn turned(self) -> Option<Figure> {
        match self {
            Figure::ScaleDown => Some(Figure::ScaleUp),
            Figure::ScaleUp => Some(Figure::ScaleDown),
            Figure::ArpeggioUp => Some(Figure::ArpeggioDown),
            Figure::ArpeggioDown => Some(Figure::ArpeggioUp),
            _ => None,
        }
    }

    /// Whether the figure starts from a third and only its octave is chosen.
    fn from_a_third(self) -> bool {
        matches!(self, Figure::FromTheThird | Figure::FromTheChordsThird)
    }

    /// Whether the figure is a phrase on the chord's own tones, its start a
    /// chord tone and an octave - see `notes`.
    fn is_phrase(self) -> bool {
        self.cell().is_some()
            || matches!(
                self,
                Figure::Enclosure
                    | Figure::DropToTheThird
                    | Figure::ThirdIntoTheRoot
                    | Figure::ArpeggioUp
                    | Figure::ArpeggioDown
            )
    }

    /// The notes of one cell, for the figures that are cells on the chord's
    /// tones: a line plays whole cells.
    pub fn cell(self) -> Option<usize> {
        match self {
            Figure::Thirds => Some(2),
            Figure::Triads | Figure::LedThirds => Some(3),
            Figure::Sevenths => Some(4),
            Figure::BebopArpeggio | Figure::Pivot => Some(5),
            _ => None,
        }
    }

    /// How far apart two starts of the figure are that play the same notes an
    /// octave apart: eight places of the ring, four chord tones, or one
    /// octave for the arpeggio from the chord's own third.
    fn octave(self) -> i32 {
        match self {
            Figure::FromTheChordsThird => 1,
            _ if self.is_phrase() => 4,
            _ => 8,
        }
    }
}

/// The settings a line is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Recipe {
    pub figure: Figure,
    pub start: Start,
    /// Eight notes over a chord that lasts a bar, four over one of two beats.
    pub notes: usize,
    /// Up over one chord, down over the next: the arpeggio over the changes
    /// taken alternately.
    pub alternate: bool,
    /// The fret a hand stands at for the whole pass, where the line is kept
    /// in one position: every note within the frets `tab::hand_from` gives,
    /// on any string.
    pub position: Option<i32>,
    /// A tune's chords or a key's - see `Harmony`.
    pub harmony: Harmony,
}

impl Default for Recipe {
    fn default() -> Self {
        Recipe {
            figure: Figure::ScaleDown,
            start: Start::Root,
            notes: 8,
            alternate: false,
            position: None,
            harmony: Harmony::default(),
        }
    }
}

impl Recipe {
    /// The lowest and highest the line may go, in semitones from C3: the
    /// guitar's three octaves, or what one position reaches - from the low E
    /// at its lowest fret to the high E at its highest.
    fn range(&self) -> (i32, i32) {
        match self.position {
            Some(fret) => {
                let (lo, hi) = crate::tab::hand_from(fret);
                (crate::tab::TUNING[0] + lo - C3, crate::tab::TUNING[5] + hi - C3)
            }
            None => (LOWEST, HIGHEST),
        }
    }
}

/// The lowest and highest a continuous line may go, in semitones from C3
/// (MIDI 48): the low E string open, E2, to the high one at the twelfth fret,
/// E5. A pitch of the line is `C3 + root + the note's semitones`, so the
/// drawing can lay the line on the neck where it really is.
const LOWEST: i32 = -8;
const HIGHEST: i32 = 28;
/// MIDI of the pitch the line counts from.
pub const C3: i32 = 48;

/// The steps over each chord of `chords`, in the order given - the order they
/// are played in. A chord without a ring gets `None`, and the line resumes
/// after it from where it would have been.
pub fn line<'a>(chords: impl IntoIterator<Item = &'a Chord>, recipe: Recipe) -> Vec<Option<Vec<Step>>> {
    let mut last: Option<i32> = None;
    // Through a key the cells climb the scale - see `climbing_start`: the
    // pitch of the chord tone the cell before was built on.
    let climbs = matches!(recipe.harmony, Harmony::Key(_)) && recipe.start != Start::Continuous;
    let mut built_on: Option<i32> = None;
    // The figure now being played: one line turns a scale round where it
    // reaches the end of the range, so the scale up becomes the scale down.
    let mut figure = recipe.figure;
    chords
        .into_iter()
        .enumerate()
        .map(|(at, chord)| {
            let over = Over::new(chord, recipe.harmony)?;
            // Alternately, every other chord the other way round.
            if recipe.alternate {
                figure = if at % 2 == 1 { recipe.figure.turned().unwrap_or(recipe.figure) } else { recipe.figure };
            }
            let cell = climbs && figure.cell().is_some();
            let (start, now) = start_place(&over, recipe, figure, at, if cell { built_on } else { last });
            figure = now;
            let notes = now.notes(&over, start);
            // Whole cells, as many as the notes asked for hold and one at least.
            let played = match now.cell() {
                Some(cell) => (recipe.notes / cell).max(1) * cell,
                None => recipe.notes,
            };
            let notes = &notes[..played.clamp(1, notes.len())];
            built_on = cell.then(|| over.pitch(Figure::Sevenths.notes(&over, start)[0]));
            last = notes.last().map(|&n| over.pitch(n));
            Some(notes.iter().map(|&n| step(n)).collect())
        })
        .collect()
}

fn step(note: Note) -> Step {
    match note {
        Note::Place(p) => Step { degree: p.rem_euclid(8) as usize, octave: p.div_euclid(8) as i8 },
        Note::Semitone(s) => Step { degree: 8 + s.rem_euclid(12) as usize, octave: s.div_euclid(12) as i8 },
    }
}

// --- In one position --------------------------------------------------
//
// The exercises after Ted Greene: what matters is not how many octaves an
// arpeggio spans but what lies under the hand. Every tone of the chord within
// one position - five frets, all six strings - from the lowest to the
// highest, and the figure walks that list: right through it and back, or a
// group at a time.

/// What the line walks in one position.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum PositionFigure {
    /// Every tone in the position, one after another, turning at either end.
    #[default]
    Sweep,
    /// A group at a time - three, four or five tones, `PositionRecipe::group`
    /// - each a tone further on: up, then down from the top where the
    /// position ends.
    GroupsUp,
    /// The same from the top down.
    GroupsDown,
    /// Groups of four through the position, each played in the order of the
    /// pass - see `order`.
    Orders,
}

impl From<usize> for PositionFigure {
    fn from(index: usize) -> Self {
        match index {
            1 => PositionFigure::GroupsUp,
            2 => PositionFigure::GroupsDown,
            3 => PositionFigure::Orders,
            _ => PositionFigure::Sweep,
        }
    }
}

/// The settings a line in one position is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PositionRecipe {
    pub figure: PositionFigure,
    /// The tensions too - see `tensions`.
    pub tensions: bool,
    /// The fret the hand stands at; `tab::hand_from` gives the frets.
    pub fret: i32,
    /// Notes over each chord, as many whole groups as they hold; `None` the
    /// whole figure: the position through and back, or every group.
    pub notes: Option<usize>,
    /// Tones in a group of `GroupsUp` and `GroupsDown`: 3, 4 or 5.
    pub group: usize,
    /// Which pass of the tune this is: picks the order of `Orders`, the same
    /// within a pass, the next one the pass after.
    pub seed: u64,
}

impl Default for PositionRecipe {
    fn default() -> Self {
        PositionRecipe { figure: PositionFigure::default(), tensions: false, fret: 0, notes: None, group: 4, seed: 0 }
    }
}

impl PositionRecipe {
    /// Tones in a group of the figure; none where it walks the position whole.
    pub fn group_len(&self) -> Option<usize> {
        match self.figure {
            PositionFigure::Sweep => None,
            PositionFigure::GroupsUp | PositionFigure::GroupsDown => Some(self.group.clamp(3, 5)),
            PositionFigure::Orders => Some(4),
        }
    }
}

/// The orders four tones can be played in.
pub const ORDERS: u64 = 24;

/// The order a group of four is played in on pass `pass`: which of its tones,
/// counted from the lowest, comes first, second, third and last. The 24 run
/// as Ted Greene's sheet "Permutations of 4-Note Groups" numbers them: the
/// first from the top tone straight down, the last from the bottom straight up.
pub fn order(pass: u64) -> [usize; 4] {
    let mut left = vec![3, 2, 1, 0];
    let mut k = (pass % ORDERS) as usize;
    let mut order = [0; 4];
    // Six orders begin on each tone, two of those go on to each of the three
    // left, and the last two tones come either way.
    for (place, orders) in [6, 2, 1, 1].into_iter().enumerate() {
        order[place] = left.remove(k / orders);
        k %= orders;
    }
    order
}

/// The tensions added to a chord's own tones, in semitones: the ninth and
/// sixth over a major seventh, the ninth and thirteenth over a dominant, the
/// ninth and eleventh over a minor, the eleventh over a half-diminished.
fn tensions(quality: &ChordQuality) -> &'static [u8] {
    match quality {
        ChordQuality::Major7 => &[2, 9],
        ChordQuality::Dominant7 => &[2, 9],
        ChordQuality::Minor7 | ChordQuality::Minor6 | ChordQuality::MinorMajor7 => &[2, 5],
        ChordQuality::HalfDiminished => &[5],
        ChordQuality::Diminished | ChordQuality::CustomScale(_) => &[],
    }
}

/// The lowest and highest pitch a position reaches, in semitones from C3:
/// the low E at its lowest fret to the high E at its highest.
fn position_range(fret: i32) -> (i32, i32) {
    let (lo, hi) = crate::tab::hand_from(fret);
    (crate::tab::TUNING[0] + lo - C3, crate::tab::TUNING[5] + hi - C3)
}

/// Every tone of the chord within the position, lowest first, in semitones
/// above its root: the pitch is `C3 + root + semitones`, as in `line`.
pub fn tones_in_position(chord: &Chord, fret: i32, with_tensions: bool) -> Vec<i32> {
    let (lowest, highest) = position_range(fret);
    let root = chord.root as i32;
    let mut classes: Vec<u8> = chord.quality.intervals().iter().map(|i| i % 12).collect();
    if with_tensions {
        classes.extend(tensions(&chord.quality));
    }
    (lowest..=highest)
        .map(|pitch| pitch - root)
        .filter(|s| classes.contains(&(s.rem_euclid(12) as u8)))
        .collect()
}

/// A walk through `len` tones from `at`, `n` of them, turning at either end;
/// `up` is the way it goes and is left the way it went.
fn sweep(len: usize, at: usize, up: &mut bool, n: usize) -> Vec<usize> {
    let mut i = at.min(len.saturating_sub(1));
    let mut out = vec![i];
    while out.len() < n && len > 1 {
        if *up && i + 1 == len {
            *up = false;
        } else if !*up && i == 0 {
            *up = true;
        }
        i = if *up { i + 1 } else { i - 1 };
        out.push(i);
    }
    out
}

/// Groups of `size` tones from `at`, each a tone further on, `n` notes of
/// them; at the end of the position they turn and go back the other way.
fn groups(len: usize, at: usize, up: &mut bool, n: usize, size: usize) -> Vec<usize> {
    if len < size {
        return (0..len).collect();
    }
    let reach = size - 1;
    let mut start = at.min(len - 1);
    let mut out = Vec::new();
    while out.len() < n {
        // A group that would run past the end turns the walk round where it
        // is - the group starting from the same tone the other way - so it
        // neither jumps across the position nor repeats the note just played.
        if *up && start + size > len {
            *up = false;
            start = start.max(reach);
        } else if !*up && start < reach {
            *up = true;
            start = start.min(len - size);
        }
        out.extend((0..size).map(|k| if *up { start + k } else { start - k }));
        start = if *up { start + 1 } else { start - 1 };
    }
    out.truncate(n);
    out
}

/// Groups of four through the position, each played in `order`, `n` notes of
/// them, the first group the one that begins on tone `first` or nearest it.
/// The groups climb a tone at a time and come back down from the top. A group
/// that would begin on the note just played, or on its octave, is passed
/// over: the judge hears pitch classes, and a class played twice running
/// stopped the line in `greene_in_position`. `tones` are the position's, in
/// semitones.
fn ordered_groups(tones: &[i32], first: usize, up: &mut bool, n: usize, order: [usize; 4]) -> Vec<usize> {
    let len = tones.len();
    if len < 4 {
        return (0..len).collect();
    }
    let same_class = |a: usize, b: usize| (tones[a] - tones[b]).rem_euclid(12) == 0;
    // A group is named by its lowest tone.
    let highest = len - 4;
    let mut lowest = first.saturating_sub(order[0]).min(highest);
    let mut out: Vec<usize> = Vec::new();
    let mut passed_over = 0;
    while out.len() < n {
        let repeats = out.last().is_some_and(|&last| same_class(last, lowest + order[0]));
        // Were every group of a position passed over, the line would never
        // end: then the group is played all the same.
        if !repeats || passed_over > len {
            out.extend(order.iter().map(|tone| lowest + tone));
            passed_over = 0;
        } else {
            passed_over += 1;
        }
        if highest == 0 {
            continue;
        }
        if *up && lowest == highest {
            *up = false;
        } else if !*up && lowest == 0 {
            *up = true;
        }
        lowest = if *up { lowest + 1 } else { lowest - 1 };
    }
    out.truncate(n);
    out
}

/// The steps over each chord in one position, in the order given. The line
/// goes on from chord to chord: each starts on its tone nearest the last
/// note, the way the line was going.
pub fn position_line<'a>(chords: impl IntoIterator<Item = &'a Chord>, recipe: PositionRecipe) -> Vec<Option<Vec<Step>>> {
    let mut last: Option<i32> = None;
    let mut up = recipe.figure != PositionFigure::GroupsDown;
    let group = recipe.group_len();
    chords
        .into_iter()
        .map(|chord| {
            ring(&chord.quality)?;
            let tones = tones_in_position(chord, recipe.fret, recipe.tensions);
            if tones.is_empty() {
                return None;
            }
            let root = chord.root as i32;
            let len = tones.len();
            let at = match last {
                None if up => 0,
                None => len - 1,
                Some(last) => {
                    let onward = |i: &usize| if up { root + tones[*i] > last } else { root + tones[*i] < last };
                    let nearest = |i: &usize| (root + tones[*i] - last).abs();
                    (0..len).filter(onward).min_by_key(nearest)
                        .or_else(|| (0..len).filter(|&i| root + tones[i] != last).min_by_key(nearest))
                        .unwrap_or(0)
                }
            };
            let n = match (recipe.notes, group) {
                (None, None) => 2 * len - 1,
                // Every group once: one from each tone that has a whole group above it.
                (None, Some(size)) => size * (len + 1).saturating_sub(size).max(1),
                (Some(notes), None) => notes,
                // Whole groups only: a group cut short is no group.
                (Some(notes), Some(size)) => (notes / size).max(1) * size,
            };
            let just_played = last.and_then(|l| (0..len).find(|&i| root + tones[i] == l));
            // A walk that turns at the end of the position may start on the
            // very note the last chord ended on - the judge could take that
            // one, still ringing, for the new. Then from the next tone in.
            let walk_from = |at: usize, up: &mut bool| match recipe.figure {
                PositionFigure::Sweep => sweep(len, at, up, n),
                PositionFigure::GroupsUp | PositionFigure::GroupsDown => groups(len, at, up, n, group.unwrap_or(4)),
                PositionFigure::Orders => ordered_groups(&tones, at, up, n, order(recipe.seed)),
            };
            // From `at`, or failing that the nearest start that does not
            // begin on the note just played.
            let mut starts: Vec<usize> = (0..len).collect();
            starts.sort_by_key(|&i| i.abs_diff(at));
            let walk = starts
                .into_iter()
                .map(|from| {
                    let mut way = up;
                    let walk = walk_from(from, &mut way);
                    (walk, way)
                })
                .find(|(walk, _)| walk.first().copied() != just_played)
                .map(|(walk, way)| {
                    up = way;
                    walk
                })
                .unwrap_or_default();
            last = walk.last().map(|&i| root + tones[i]);
            Some(walk.iter().map(|&i| step(Note::Semitone(tones[i]))).collect())
        })
        .collect()
}

/// Semitones above the chord's root of a step of a Harris line.
#[cfg(test)]
fn step_semitones(ring: &Ring, step: Step) -> i32 {
    let octave = 12 * step.octave as i32;
    match step.degree {
        d if d < 8 => ring.semitones[d] as i32 + octave,
        d => (d - 8) as i32 + octave,
    }
}

/// Where the figure over a chord starts, and the figure itself - a
/// continuous line may have turned it round.
fn start_place(over: &Over, recipe: Recipe, figure: Figure, at: usize, last: Option<i32>) -> (i32, Figure) {
    // The arpeggio from a third starts on that third whatever the setting,
    // and only its octave is chosen: for the dominant's third a place, for
    // the chord's own the octave itself.
    let third = over.ring.thought_root + 2;
    // A phrase counts its start in chord tones, a scale figure in places of
    // the ring, where the chord tones are every other one.
    let tone = if figure.is_phrase() { 1 } else { 2 };
    let fixed = match recipe.start {
        _ if figure == Figure::FromTheThird => third,
        _ if figure == Figure::FromTheChordsThird => 0,
        Start::Root | Start::Continuous => 0,
        Start::Third => tone,
        Start::InTurn => tone * (at as i32 % 4),
    };
    let (lowest, highest) = recipe.range();
    if recipe.start != Start::Continuous {
        // Kept in one position, the start moves by octaves until the figure
        // lies within it - as near its middle as it will go.
        if over.one_cell && figure.cell().is_some() {
            return (climbing_start(over, figure, fixed, (lowest, highest), last), figure);
        }
        if recipe.position.is_none() {
            return (nearest_octave(over, figure, fixed, (lowest, highest), last), figure);
        }
        let middle = (lowest + highest) / 2;
        let first = |s: i32| over.pitch(figure.notes(over, s)[0]);
        let wanted = first(fixed);
        // Where no octave of it fits - two octaves of a chord from a root
        // high in the position - the start nearest it that does.
        let start = (-4..=4)
            .map(|n| fixed + n * figure.octave())
            .filter(|&s| fits(over, figure, s, (lowest, highest)))
            .min_by_key(|&s| (first(s) - middle).abs())
            .or_else(|| {
                (-32..32)
                    .filter(|&s| fits(over, figure, s, (lowest, highest)))
                    .min_by_key(|&s| (first(s) - wanted).abs())
            })
            .unwrap_or(fixed);
        return (start, figure);
    }
    let first = |f: Figure, s: &i32| over.pitch(f.notes(over, *s)[0]);
    // Starts a few octaves round the last note whose figure lies within the
    // range: tones on the beat, the dominant's third, or the octaves of the
    // chord's own third.
    //
    // Never the last note again, in any octave, where there is a choice: the
    // judge could take the old one still ringing for the new. A figure from a
    // third has none - over a V going to its I, the dominant's sixth is the
    // tonic's third - and avoids only the very same note.
    let candidates = |f: Figure| -> Vec<i32> {
        let starts: Vec<i32> = match f {
            Figure::FromTheThird => (-24..24).filter(|p| (p - third).rem_euclid(8) == 0).collect(),
            Figure::FromTheChordsThird => (-3..=3).collect(),
            _ if f.is_phrase() => (-12..12).collect(),
            _ => (-24..24).filter(|p| p % 2 == 0).collect(),
        };
        starts
            .into_iter()
            .filter(|&s| fits(over, f, s, (lowest, highest)))
            .filter(|s| {
                last.is_none_or(|last| {
                    if f.from_a_third() { first(f, s) != last } else { (first(f, s) - last).rem_euclid(12) != 0 }
                })
            })
            .collect()
    };
    // Where a line sets out: at the end of the range away from where the
    // figure drifts - the top for the scale down, the bottom for the scale
    // up and for the arpeggio and the enclosure, which climb a little with
    // every chord - so it has the range to run through. A figure from a
    // third barely moves and sets out in the middle.
    let afresh = |f: Figure, c: &[i32]| -> Option<i32> {
        match f {
            Figure::ScaleDown | Figure::ArpeggioDown => c.iter().max_by_key(|s| first(f, s)).copied(),
            Figure::ScaleUp | Figure::ArpeggioUp | Figure::ArpeggioUpScaleDown => {
                c.iter().min_by_key(|s| first(f, s)).copied()
            }
            _ => c.iter().min_by_key(|s| (first(f, s) - (lowest + highest) / 2).abs()).copied(),
        }
    };
    let mut here = candidates(figure);
    // A narrow range - one position - may hold no octave of a third the
    // figure is tied to: then any start that fits, so no note falls outside.
    if here.is_empty() {
        here = (-32..32).filter(|&s| fits(over, figure, s, (lowest, highest))).collect();
    }
    let Some(last) = last else {
        return (afresh(figure, &here).unwrap_or(fixed), figure);
    };
    // On the way the line was going, a step or two. A figure from a third has
    // its tone fixed and takes the octave nearest.
    // A phrase starts on the chord tone nearest, whichever way.
    let onward = |f: Figure, c: &[i32]| -> Option<i32> {
        if f.from_a_third() || f.is_phrase() {
            return c.iter().min_by_key(|s| (first(f, s) - last).abs()).copied();
        }
        let next = match f {
            Figure::ScaleUp => c.iter().filter(|s| first(f, s) > last).min_by_key(|s| first(f, s)),
            _ => c.iter().filter(|s| first(f, s) < last).max_by_key(|s| first(f, s)),
        };
        next.filter(|s| (first(f, s) - last).abs() <= 4).copied()
    };
    if let Some(start) = onward(figure, &here) {
        return (start, figure);
    }
    // The end of the range: a scale turns round and goes on by step; an
    // arpeggio figure starts again from the far end. The arpeggio up and the
    // scale down climbs about a fifth a chord: started an octave back it
    // leapt 11-13 every third chord, from the nearest start that fits a
    // seventh every chord, from the far end 28 every sixth.
    if let Some(turned) = figure.turned() {
        if let Some(start) = onward(turned, &candidates(turned)) {
            return (start, turned);
        }
    }
    (afresh(figure, &here).unwrap_or(fixed), figure)
}

/// The octave of `fixed` whose figure lies within the range, its first note
/// nearest the last note played, and not that very note where another
/// octave fits; the first chord in the middle of the range. Without it every
/// chord started in the octave its root names from C3, and the lines leapt
/// up to two octaves between chords and ran past the twelfth fret.
fn nearest_octave(over: &Over, figure: Figure, fixed: i32, range: (i32, i32), last: Option<i32>) -> i32 {
    let first = |s: i32| over.pitch(figure.notes(over, s)[0]);
    let target = last.unwrap_or((range.0 + range.1) / 2);
    let fitting: Vec<i32> =
        (-4..=4).map(|n| fixed + n * figure.octave()).filter(|&s| fits(over, figure, s, range)).collect();
    let fresh: Vec<i32> = fitting.iter().copied().filter(|&s| Some(first(s)) != last).collect();
    let pool = if fresh.is_empty() { &fitting } else { &fresh };
    pool.iter().copied().min_by_key(|&s| (first(s) - target).abs()).unwrap_or(fixed)
}

/// Through a key, where a cell starts: on the octave of its chord tone
/// nearest the tone the cell before was built on - a step above it, while the
/// range lasts - and the first cell on the lowest from which the chord's own
/// arpeggio fits the range. Counted on the chord tone, not on the cell's
/// first note: the turned arpeggio begins an octave above its tone and lies
/// where the plain one does.
fn climbing_start(over: &Over, figure: Figure, fixed: i32, range: (i32, i32), built_on: Option<i32>) -> i32 {
    let tone = |s: i32| over.pitch(Figure::Sevenths.notes(over, s)[0]);
    let octaves = || (-4..=4).map(|n| fixed + 4 * n);
    let lowest = || octaves().filter(|&s| fits(over, Figure::Sevenths, s, range)).map(tone).min();
    let Some(by) = built_on.or_else(lowest) else {
        return fixed;
    };
    octaves().filter(|&s| fits(over, figure, s, range)).min_by_key(|&s| (tone(s) - by).abs()).unwrap_or(fixed)
}

/// Whether the whole figure from `start` lies within the range - every note
/// even where four are played, so the four are the first half of it.
fn fits(over: &Over, figure: Figure, start: i32, (lowest, highest): (i32, i32)) -> bool {
    figure.notes(over, start).iter().all(|&n| (lowest..=highest).contains(&over.pitch(n)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NoteName, parse_chords_line_pub};

    fn chords(line: &str) -> Vec<Chord> {
        parse_chords_line_pub(line)
    }

    const QUALITIES: [ChordQuality; 7] = [
        ChordQuality::Major7,
        ChordQuality::Dominant7,
        ChordQuality::Minor7,
        ChordQuality::HalfDiminished,
        ChordQuality::Diminished,
        ChordQuality::Minor6,
        ChordQuality::MinorMajor7,
    ];
    const FIGURES: [Figure; 8] = [
        Figure::ScaleDown,
        Figure::ScaleUp,
        Figure::ArpeggioUpScaleDown,
        Figure::FromTheThird,
        Figure::FromTheChordsThird,
        Figure::Enclosure,
        Figure::DropToTheThird,
        Figure::ThirdIntoTheRoot,
    ];

    fn chord_on_c(quality: ChordQuality) -> Chord {
        Chord { root: NoteName::C, quality }
    }

    /// The pitch classes of a line's steps over `chord`.
    fn classes(chord: &Chord, steps: &[Step]) -> Vec<i32> {
        let ring = ring(&chord.quality).unwrap();
        steps.iter().map(|&s| (chord.root as i32 + step_semitones(ring, s)).rem_euclid(12)).collect()
    }
    const STARTS: [Start; 4] = [Start::Root, Start::Third, Start::InTurn, Start::Continuous];

    /// The tones the line thinks of as the chord: the even places.
    fn on_the_beat(ring: &Ring) -> Vec<u8> {
        (0..4).map(|k| ring.semitones[2 * k]).collect()
    }

    /// The whole point: whatever the figure, the chord or where it starts,
    /// every note on a beat - every other note - is a tone of the chord the
    /// line thinks in.
    #[test]
    fn every_beat_falls_on_a_chord_tone() {
        for quality in QUALITIES {
            let chord = chord_on_c(quality.clone());
            let over = Over::new(&chord, Harmony::default()).unwrap();
            // The phrases are counted in chord tones, not beats: see their
            // own tests.
            for figure in FIGURES.into_iter().filter(|f| !f.is_phrase()) {
                let starts: Vec<i32> = match figure {
                    Figure::FromTheThird => vec![over.ring.thought_root + 2],
                    Figure::FromTheChordsThird => vec![0],
                    _ => vec![0, 2, 4, 6],
                };
                for start in starts {
                    let notes = figure.notes(&over, start);
                    for k in (0..8).step_by(2) {
                        // The chord's own arpeggio has its own tones on the
                        // first two beats.
                        let lead_in = figure == Figure::FromTheChordsThird && k < 4;
                        if lead_in {
                            continue;
                        }
                        let Note::Place(p) = notes[k] else {
                            panic!("{quality:?} {figure:?} from {start}: note {k} outside the ring");
                        };
                        assert_eq!(p.rem_euclid(2), 0, "{quality:?} {figure:?} from {start}: note {k} off the beat");
                    }
                }
            }
        }
    }

    /// The phrases, note for note as the examples they come from are
    /// written: the pitch classes over each chord, from the root and from
    /// the third.
    fn phrase(figure: Figure, chord: &str, start: Start) -> Vec<i32> {
        let chord = chords(chord)[0].clone();
        let steps = line(std::slice::from_ref(&chord), Recipe { figure, start, notes: 8, ..Default::default() })[0].clone().unwrap();
        classes(&chord, &steps)
    }

    const C: i32 = 0;
    const CS: i32 = 1;
    const D: i32 = 2;
    const DS: i32 = 3;
    const E: i32 = 4;
    const F: i32 = 5;
    const FS: i32 = 6;
    const G: i32 = 7;
    const GS: i32 = 8;
    const A: i32 = 9;
    const AS: i32 = 10;
    const B: i32 = 11;

    #[test]
    fn an_enclosure_then_the_arpeggio_and_a_step_down() {
        assert_eq!(phrase(Figure::Enclosure, "CMaj7", Start::Root), [D, B, C, E, G, B, A]);
        assert_eq!(phrase(Figure::Enclosure, "Dm7", Start::Root), [E, CS, D, F, A, C, B]);
        assert_eq!(phrase(Figure::Enclosure, "G7", Start::Root), [A, FS, G, B, D, F, E]);
        // From the third: over Dm7 the Fmaj7 inside it.
        assert_eq!(phrase(Figure::Enclosure, "Dm7", Start::Third), [G, E, F, A, C, E, D]);
        let chord = chord_on_c(ChordQuality::Major7);
        let steps = line(std::slice::from_ref(&chord), Recipe { figure: Figure::Enclosure, ..Recipe::default() })[0].clone().unwrap();
        let ring = ring(&chord.quality).unwrap();
        let names: Vec<String> = steps.iter().map(|s| names(ring, &chord.quality)[s.degree].clone()).collect();
        assert_eq!(names, ["2", "7", "1", "3", "5", "7", "6"]);
    }

    #[test]
    fn a_drop_to_the_third_with_its_leading_note() {
        assert_eq!(phrase(Figure::DropToTheThird, "CMaj7", Start::Root), [C, DS, E, B, A, G]);
        assert_eq!(phrase(Figure::DropToTheThird, "Dm7", Start::Root), [D, E, F, C, B, A]);
        assert_eq!(phrase(Figure::DropToTheThird, "G7", Start::Root), [G, AS, B, F, E, D]);
        assert_eq!(phrase(Figure::DropToTheThird, "FMaj7", Start::Root), [F, GS, A, E, D, C]);
        // The third really drops: an octave below the root.
        let chord = chords("CMaj7")[0].clone();
        let steps = line(std::slice::from_ref(&chord), Recipe { figure: Figure::DropToTheThird, ..Recipe::default() })[0].clone().unwrap();
        let ring = ring(&chord.quality).unwrap();
        assert!(step_semitones(ring, steps[2]) < step_semitones(ring, steps[0]) - 6);
    }

    #[test]
    fn the_third_into_the_root() {
        assert_eq!(phrase(Figure::ThirdIntoTheRoot, "CMaj7", Start::Root), [E, B, D, C]);
        assert_eq!(phrase(Figure::ThirdIntoTheRoot, "Dm7", Start::Root), [F, CS, E, D]);
        assert_eq!(phrase(Figure::ThirdIntoTheRoot, "G7", Start::Root), [B, FS, A, G]);
        // From the third: over Dm7 the Fmaj7 version, into F.
        assert_eq!(phrase(Figure::ThirdIntoTheRoot, "Dm7", Start::Third), [A, E, G, F]);
    }

    /// The chord's own third: over Dm7 the Fmaj7 inside it, F A C E, then
    /// down the scale the line thinks in, G7's - D C B A.
    #[test]
    fn the_chords_own_third_plays_the_arpeggio_inside_it() {
        let chord = chords("Dm7")[0].clone();
        let recipe = Recipe { figure: Figure::FromTheChordsThird, ..Recipe::default() };
        let steps = line(std::slice::from_ref(&chord), recipe)[0].clone().unwrap();
        assert_eq!(classes(&chord, &steps), vec![5, 9, 0, 4, 2, 0, 11, 9]);
    }

    /// The rings are what the module says they are: eight notes, rising
    /// within the octave, the root first.
    #[test]
    fn each_ring_is_eight_notes_rising_from_the_root() {
        for quality in QUALITIES {
            let ring = ring(&quality).unwrap();
            assert_eq!(ring.semitones[0], 0);
            assert!(ring.semitones.windows(2).all(|w| w[0] < w[1]), "{quality:?}");
            assert_eq!(ring.thought_root % 2, 0, "{quality:?}: the thought root off the beat");
        }
    }

    /// The minor seventh thinks in the dominant a fourth up: over Gm7 the
    /// beats are C7's tones. The half-diminished thinks in the dominant a
    /// major third down: over Am7b5 the beats are F7's.
    #[test]
    fn minor_and_half_diminished_think_in_their_dominants() {
        let pcs = |ring: &Ring, root: u8| -> Vec<u8> {
            let mut v: Vec<u8> = on_the_beat(ring).iter().map(|s| (s + root) % 12).collect();
            v.sort();
            v
        };
        // G = 7: C E G Bb.
        assert_eq!(pcs(ring(&ChordQuality::Minor7).unwrap(), 7), vec![0, 4, 7, 10]);
        // A = 9: F A C Eb.
        assert_eq!(pcs(ring(&ChordQuality::HalfDiminished).unwrap(), 9), vec![0, 3, 5, 9]);
    }

    /// Down from the root of C7: C B Bb A G F E D - Harris's first rule.
    #[test]
    fn the_scale_down_from_the_root_of_a_dominant() {
        let line = line(&chords("C7"), Recipe::default());
        let steps = line[0].as_ref().unwrap();
        let names: Vec<&str> = steps.iter().map(|s| DOMINANT.names[s.degree]).collect();
        assert_eq!(names, ["1", "(7)", "b7", "6", "5", "4", "3", "2"]);
    }

    /// The arpeggio from the third of the dominant: E G Bb D over C7, then
    /// down from C. Over Gm7 the same notes, thought in C7.
    #[test]
    fn from_the_third_plays_the_dominants_third_up() {
        let recipe = Recipe { figure: Figure::FromTheThird, ..Recipe::default() };
        let pcs = |line: &str| -> Vec<i32> {
            let chord = &chords(line)[0];
            let ring = ring(&chord.quality).unwrap();
            let steps = super::line(std::slice::from_ref(chord), recipe)[0].clone().unwrap();
            steps
                .iter()
                .map(|&s| (chord.root as i32 + step_semitones(ring, s)).rem_euclid(12))
                .collect()
        };
        // E G Bb D C B Bb A
        assert_eq!(pcs("C7"), vec![4, 7, 10, 2, 0, 11, 10, 9]);
        assert_eq!(pcs("Gm7"), vec![4, 7, 10, 2, 0, 11, 10, 9]);
    }

    /// One line through the tune: every chord after the first starts on a
    /// tone on the beat a step or two from where the last one ended - or
    /// that an octave away, where the line would leave the guitar's range -
    /// and the line keeps to the range.
    #[test]
    fn a_continuous_line_moves_by_step_into_each_chord() {
        let tune = chords("Cm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7 Gm7 Cm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7 Gm7");
        for figure in FIGURES {
            let recipe = Recipe { figure, start: Start::Continuous, notes: 8, ..Default::default() };
            let line = line(&tune, recipe);
            let mut last: Option<i32> = None;
            for (chord, steps) in tune.iter().zip(&line) {
                let ring = ring(&chord.quality).unwrap();
                let pitches: Vec<i32> = steps
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|&s| chord.root as i32 + step_semitones(ring, s))
                    .collect();
                assert!(pitches.iter().all(|p| (LOWEST..=HIGHEST).contains(p)), "{figure:?} out of range: {pitches:?}");
                if let Some(last) = last {
                    let leap = (pitches[0] - last).abs();
                    let same = if figure.from_a_third() { leap == 0 } else { leap % 12 == 0 };
                    let nearest = figure.from_a_third() || figure.is_phrase();
                    assert!(!same, "{figure:?} into {chord:?}: the same note again");
                    // The third has no choice of tone, only of octave.
                    let (leap, most) = if nearest { (leap.min((leap - 12).abs()), 6) } else { (leap, 4) };
                    // Or the line starts afresh from the far end of the range.
                    let afresh = pitches.iter().min().unwrap() < &(LOWEST + 5)
                        || pitches.iter().max().unwrap() > &(HIGHEST - 5);
                    assert!(leap <= most || afresh, "{figure:?} into {chord:?}: {last} -> {}", pitches[0]);
                }
                last = pitches.last().copied();
            }
        }
    }

    /// Four notes over a chord of two beats: the first half of the figure.
    #[test]
    fn four_notes_are_the_first_half() {
        for figure in FIGURES {
            for start in STARTS {
                let eight = line(&chords("C7 F7"), Recipe { figure, start, notes: 8, ..Default::default() });
                let four = line(&chords("C7"), Recipe { figure, start, notes: 4, ..Default::default() });
                assert_eq!(four[0].as_ref().unwrap()[..], eight[0].as_ref().unwrap()[..4]);
            }
        }
    }

    /// A chord of the user's own scale has no ring and no line.
    #[test]
    fn a_custom_chord_has_no_line() {
        let custom = Chord {
            root: NoteName::C,
            quality: ChordQuality::CustomScale(crate::model::ScaleDefinition {
                name: "x".into(),
                intervals: vec![0, 3, 7],
                names: vec!["1".into(), "b3".into(), "5".into()],
            }),
        };
        assert!(line(&[custom], Recipe::default())[0].is_none());
    }

    /// The first chorus of Autumn Leaves in every figure, written out for a
    /// person to read: `cargo test --release harris_print -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn harris_print() {
        let tune = chords("Cm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7 Gm7");
        for figure in FIGURES {
            for start in STARTS {
                println!("\n{figure:?}, {start:?}");
                for (chord, steps) in tune.iter().zip(line(&tune, Recipe { figure, start, notes: 8, ..Default::default() })) {
                    let ring = ring(&chord.quality).unwrap();
                    let steps = steps.unwrap();
                    let notes: Vec<String> = steps
                        .iter()
                        .map(|&s| format!("{:?}", NoteName::from_index((chord.root as i32 + step_semitones(ring, s)).rem_euclid(12) as usize)))
                        .collect();
                    let degrees: Vec<String> = steps.iter().map(|s| names(ring, &chord.quality)[s.degree].clone()).collect();
                    println!("{:>8}  {:<32} {}", short(chord), notes.join(" "), degrees.join(" "));
                }
            }
        }
    }

    fn short(chord: &Chord) -> String {
        let q = match chord.quality {
            ChordQuality::Major7 => "maj7",
            ChordQuality::Dominant7 => "7",
            ChordQuality::Minor7 => "m7",
            ChordQuality::HalfDiminished => "m7b5",
            ChordQuality::Diminished => "dim7",
            ChordQuality::Minor6 => "m6",
            ChordQuality::MinorMajor7 => "mMaj7",
            ChordQuality::CustomScale(_) => "?",
        };
        format!("{:?}{q}", chord.root)
    }
    /// The notes of a line as letters, a group to a chord: `C E G B | D F A C`.
    fn spelled(chords: &[Chord], recipe: Recipe) -> String {
        const LETTERS: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        line(chords, recipe)
            .iter()
            .zip(chords)
            .map(|(steps, chord)| {
                classes(chord, steps.as_ref().unwrap()).iter().map(|&pc| LETTERS[pc as usize]).collect::<Vec<_>>().join(" ")
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// The pitches of a line, in semitones from C3, chord by chord.
    fn pitches(chords: &[Chord], recipe: Recipe) -> Vec<Vec<i32>> {
        line(chords, recipe)
            .iter()
            .zip(chords)
            .map(|(steps, chord)| {
                let ring = ring(&chord.quality).unwrap();
                steps.as_ref().unwrap().iter().map(|&s| chord.root as i32 + step_semitones(ring, s)).collect()
            })
            .collect()
    }

    fn in_c(figure: Figure) -> Recipe {
        Recipe { figure, harmony: Harmony::Key(0), ..Recipe::default() }
    }

    /// The scale in chords: the seventh chord on every degree of C major.
    #[test]
    fn the_chords_of_a_key_are_its_seven_sevenths() {
        let names: Vec<String> =
            chords_of_key(0).iter().map(|c| format!("{}{}", c.root.to_string(), c.quality.to_string())).collect();
        assert_eq!(names.join(" "), "CMaj7 Dm7 Em7 FMaj7 G7 Am7 Bm7b5 CMaj7");
        assert_eq!(chords_of_key(10)[4].root, NoteName::F, "the fifth degree of Bb");
    }

    /// The six cells through C major, a cell to a degree, as the exercise is
    /// taught: thirds, triads, the chords, thirds and chords led into by a
    /// half step, and the chord turned on its root with a step down to end.
    #[test]
    fn the_cells_through_a_key() {
        let key = chords_of_key(0);
        assert_eq!(spelled(&key, in_c(Figure::Thirds)), "C E | D F | E G | F A | G B | A C | B D | C E");
        assert_eq!(
            spelled(&key, in_c(Figure::Triads)),
            "C E G | D F A | E G B | F A C | G B D | A C E | B D F | C E G"
        );
        assert_eq!(
            spelled(&key, in_c(Figure::Sevenths)),
            "C E G B | D F A C | E G B D | F A C E | G B D F | A C E G | B D F A | C E G B"
        );
        assert_eq!(
            spelled(&key, in_c(Figure::LedThirds)),
            "B C E | C# D F | D# E G | E F A | F# G B | G# A C | A# B D | B C E"
        );
        assert_eq!(
            spelled(&key, in_c(Figure::BebopArpeggio)),
            "B C E G B | C# D F A C | D# E G B D | E F A C E | F# G B D F | G# A C E G | A# B D F A | B C E G B"
        );
        assert_eq!(
            spelled(&key, in_c(Figure::Pivot)),
            "C E G B A | D F A C B | E G B D C | F A C E D | G B D F E | A C E G F | B D F A G | C E G B A"
        );
    }

    /// Through a key the cells climb the scale: each begins a step above the
    /// one before, the whole way up the octave, in every key and in a
    /// position as on the open neck - and the turned chord begins on its
    /// root an octave up, above everything after it.
    #[test]
    fn the_cells_climb_the_scale() {
        for key in 0..12 {
            for position in [None, Some(3), Some(7)] {
                for figure in [Figure::Thirds, Figure::Triads, Figure::Sevenths, Figure::LedThirds, Figure::BebopArpeggio, Figure::Pivot] {
                    let recipe = Recipe { figure, harmony: Harmony::Key(key), position, ..Recipe::default() };
                    let cells = pitches(&chords_of_key(key), recipe);
                    let (lowest, highest) = recipe.range();
                    assert!(cells.iter().flatten().all(|p| (lowest..=highest).contains(p)), "key {key} {figure:?} {position:?}: {cells:?}");
                    let firsts: Vec<i32> = cells.iter().map(|c| c[0]).collect();
                    // From the bottom: the plain arpeggio has no octave below
                    // it in the range, and the turned one lies on its notes.
                    let plain = &pitches(&chords_of_key(key), Recipe { figure: Figure::Sevenths, ..recipe })[0];
                    assert!(plain.iter().any(|p| !(lowest..=highest).contains(&(p - 12))), "key {key} {position:?}: begins an octave up, on {plain:?}");
                    if figure == Figure::Pivot {
                        assert_eq!(cells[0][1..4], plain[1..4], "key {key} {position:?}: the turned arpeggio lies elsewhere");
                    }
                    let steps = firsts.windows(2).filter(|w| matches!(w[1] - w[0], 1 | 2)).count();
                    // A position two octaves and a third wide has no room for
                    // every key's octave: there the line drops an octave once.
                    let drops = firsts.windows(2).filter(|w| w[1] < w[0]).count();
                    assert!(steps + drops == 7 && drops <= 1, "key {key} {figure:?} {position:?}: begins on {firsts:?}");
                    if figure == Figure::Pivot {
                        assert!(cells.iter().all(|c| c[0] > c[3]), "the root is not on top: {cells:?}");
                    }
                }
            }
        }
    }

    /// Over a tune a bar holds as many whole cells as fit: the second seventh
    /// chord is the one from the third, with the ninth - the key's ninth
    /// where the chord lies in the tune's key.
    #[test]
    fn the_cells_over_the_chords_of_a_tune() {
        let tune = chords("Dm7 G7 CMaj7");
        let over = |figure, start, notes| Recipe { figure, start, notes, harmony: Harmony::Tune(Some(0)), ..Recipe::default() };
        assert_eq!(spelled(&tune, over(Figure::Sevenths, Start::Root, 8)), "D F A C F A C E | G B D F B D F A | C E G B E G B D");
        assert_eq!(spelled(&tune, over(Figure::Sevenths, Start::Root, 4)), "D F A C | G B D F | C E G B");
        // The second triad a step up the scale, not on the chord's third.
        assert_eq!(spelled(&tune, over(Figure::Triads, Start::Root, 8)), "D F A E G B | G B D A C E | C E G D F A");
        assert_eq!(spelled(&tune, over(Figure::Triads, Start::Third, 8)), "F A C G B D | B D F C E G | E G B F A C");
        assert_eq!(spelled(&tune, over(Figure::Thirds, Start::Root, 8)), "D F E G F A G B | G B A C B D C E | C E D F E G F A");
        assert_eq!(spelled(&tune, over(Figure::LedThirds, Start::Root, 8)), "C# D F D# E G | F# G B G# A C | B C E C# D F");
        // The arpeggio from the third with its leading note: over Dm7 the
        // F major seventh, as in the opening of a well-known solo.
        assert_eq!(spelled(&tune, over(Figure::BebopArpeggio, Start::Third, 8)), "E F A C E | A# B D F A | D# E G B D");
        assert_eq!(spelled(&tune, over(Figure::Pivot, Start::Third, 4)), "F A C E D | B D F A G | E G B D C");
        // The third degree in C: its ninth is the key's F, not its own F#.
        let third = chords("Em7");
        assert_eq!(spelled(&third, over(Figure::Sevenths, Start::Root, 8)), "E G B D G B D F");
        let alone = Recipe { harmony: Harmony::Tune(None), ..over(Figure::Sevenths, Start::Root, 8) };
        assert_eq!(spelled(&third, alone), "E G B D G B D F#");
    }

    /// The steps of the key's scale are named as its degrees: over Em7 in C
    /// the F is the flat second and the C the flat sixth; on its own the
    /// chord has the second and the sixth of its own scale.
    #[test]
    fn a_scale_note_is_named_as_its_degree() {
        let chord = chords("Em7").remove(0);
        let in_c = names_over(&chord, Harmony::Key(0)).unwrap();
        assert_eq!(&in_c[8..], ["1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"]);
        let alone = names_over(&chord, Harmony::Tune(None)).unwrap();
        assert_eq!(&alone[8..], ["1", "#1", "2", "b3", "3", "4", "#4", "5", "#5", "6", "b7", "7"]);
    }
}

#[cfg(test)]
mod in_position {
    use super::*;
    use crate::model::parse_chords_line_pub;

    fn pitches(chord: &Chord, steps: &[Step]) -> Vec<i32> {
        let ring = ring(&chord.quality).unwrap();
        steps.iter().map(|&s| C3 + chord.root as i32 + step_semitones(ring, s)).collect()
    }

    /// The tones of a chord in a position are every one the hand reaches,
    /// and only those: over Cmaj7 at the fifth fret, every C E G B from the
    /// low E at the fourth fret to the high E at the eighth.
    #[test]
    fn the_tones_in_a_position_are_all_under_the_hand() {
        let chord = parse_chords_line_pub("CMaj7")[0].clone();
        let tones = tones_in_position(&chord, 5, false);
        let midi: Vec<i32> = tones.iter().map(|s| C3 + s).collect();
        let (lo, hi) = crate::tab::hand_from(5);
        assert!(midi.iter().all(|&m| (40 + lo..=64 + hi).contains(&m)), "{midi:?}");
        let expected: Vec<i32> = (40 + lo..=64 + hi).filter(|m| [0, 4, 7, 11].contains(&(m % 12))).collect();
        assert_eq!(midi, expected);
        // With the tensions the ninth and the sixth come in.
        let more = tones_in_position(&chord, 5, true);
        assert!(more.len() > tones.len());
        assert!(more.iter().any(|s| s.rem_euclid(12) == 2) && more.iter().any(|s| s.rem_euclid(12) == 9));
    }

    fn recipe(figure: PositionFigure, notes: Option<usize>) -> PositionRecipe {
        PositionRecipe { figure, tensions: false, fret: 5, notes, group: 4, seed: 7 }
    }

    /// The whole position, one chord: the condensed arpeggio is every tone up
    /// and back down; the groups are four at a time, each a tone on.
    #[test]
    fn the_whole_position_on_one_chord() {
        let chord = parse_chords_line_pub("CMaj7")[0].clone();
        let tones: Vec<i32> = tones_in_position(&chord, 5, false).iter().map(|s| C3 + s).collect();
        let len = tones.len();

        let sweep = pitches(&chord, &position_line([&chord], recipe(PositionFigure::Sweep, None))[0].clone().unwrap());
        let mut there_and_back = tones.clone();
        there_and_back.extend(tones.iter().rev().skip(1));
        assert_eq!(sweep, there_and_back);

        let up = pitches(&chord, &position_line([&chord], recipe(PositionFigure::GroupsUp, None))[0].clone().unwrap());
        assert_eq!(up.len(), 4 * (len - 3));
        assert_eq!(up[..8], [tones[0], tones[1], tones[2], tones[3], tones[1], tones[2], tones[3], tones[4]]);

        let down = pitches(&chord, &position_line([&chord], recipe(PositionFigure::GroupsDown, None))[0].clone().unwrap());
        assert_eq!(down[..4], [tones[len - 1], tones[len - 2], tones[len - 3], tones[len - 4]]);
    }

    /// Groups of three and of five walk the position as those of four do,
    /// and a chord given eight notes takes whole groups only: two of three,
    /// one of five.
    #[test]
    fn groups_of_three_and_of_five() {
        let chord = parse_chords_line_pub("CMaj7")[0].clone();
        let tones: Vec<i32> = tones_in_position(&chord, 5, false).iter().map(|s| C3 + s).collect();
        let len = tones.len();
        let line = |group, notes| {
            let r = PositionRecipe { group, ..recipe(PositionFigure::GroupsUp, notes) };
            pitches(&chord, &position_line([&chord], r)[0].clone().unwrap())
        };
        assert_eq!(line(3, None).len(), 3 * (len - 2));
        assert_eq!(line(3, None)[..6], [tones[0], tones[1], tones[2], tones[1], tones[2], tones[3]]);
        assert_eq!(line(5, None).len(), 5 * (len - 4));
        assert_eq!(line(5, None)[..5], tones[..5]);
        assert_eq!(line(3, Some(8)).len(), 6);
        assert_eq!(line(3, Some(4)).len(), 3);
        assert_eq!(line(5, Some(8)).len(), 5);
        assert_eq!(line(5, Some(4)).len(), 5);
        assert_eq!(line(4, Some(8)).len(), 8);
    }

    /// The 24 orders run as on Ted Greene's sheet: the first from the top
    /// tone straight down, the second with the two lowest changed round, the
    /// seventh the first to begin a tone lower, the last straight up - and
    /// no two are the same.
    #[test]
    fn the_orders_run_as_on_the_sheet() {
        assert_eq!(order(0), [3, 2, 1, 0]);
        assert_eq!(order(1), [3, 2, 0, 1]);
        assert_eq!(order(5), [3, 0, 1, 2]);
        assert_eq!(order(6), [2, 3, 1, 0]);
        assert_eq!(order(18), [0, 3, 2, 1]);
        assert_eq!(order(23), [0, 1, 2, 3]);
        assert_eq!(order(24), order(0));
        let mut all: Vec<[usize; 4]> = (0..ORDERS).map(order).collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 24);
    }

    /// A pass plays every group in one order, the next pass in the next:
    /// each group is four tones in a row, in the order of the pass, and no
    /// group begins on the note the last one ended on, or on its octave.
    #[test]
    fn a_pass_plays_its_groups_in_one_order() {
        let chord = parse_chords_line_pub("G7")[0].clone();
        let tones: Vec<i32> = tones_in_position(&chord, 5, false).iter().map(|s| C3 + chord.root as i32 + s).collect();
        for pass in 0..ORDERS {
            let r = PositionRecipe { seed: pass, ..recipe(PositionFigure::Orders, None) };
            let line = pitches(&chord, &position_line([&chord], r)[0].clone().unwrap());
            assert_eq!(line.len(), 4 * (tones.len() - 3));
            for group in line.chunks(4) {
                let at: Vec<usize> = group.iter().map(|p| tones.iter().position(|t| t == p).unwrap()).collect();
                let lowest = *at.iter().min().unwrap();
                let played: Vec<usize> = at.iter().map(|tone| tone - lowest).collect();
                assert_eq!(played, order(pass), "pass {pass}: {group:?}");
            }
            assert!(line.windows(2).all(|w| (w[0] - w[1]) % 12 != 0), "pass {pass} plays a class twice running: {line:?}");
        }
        // The last order is the groups straight up.
        let straight = PositionRecipe { seed: 23, ..recipe(PositionFigure::Orders, None) };
        assert_eq!(position_line([&chord], straight), position_line([&chord], recipe(PositionFigure::GroupsUp, None)));
    }

    /// Through a tune the line goes on from chord to chord: each starts on a
    /// tone a few semitones from the last note, never the same note, and
    /// stays in the position.
    #[test]
    fn through_a_tune_the_line_goes_on() {
        let tune = parse_chords_line_pub("Cm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7 Gm7");
        for figure in [PositionFigure::Sweep, PositionFigure::GroupsUp, PositionFigure::GroupsDown, PositionFigure::Orders] {
            for (fret, pass) in [1, 5, 9].into_iter().flat_map(|fret| (0..ORDERS).map(move |pass| (fret, pass))) {
                let r = PositionRecipe { fret, seed: pass, ..recipe(figure, Some(8)) };
                let (lo, hi) = crate::tab::hand_from(fret);
                let mut last: Option<i32> = None;
                for (chord, steps) in tune.iter().zip(position_line(&tune, r)) {
                    let p = pitches(chord, &steps.unwrap());
                    assert_eq!(p.len(), 8);
                    assert!(p.iter().all(|&m| (40 + lo..=64 + hi).contains(&m)), "{figure:?} at {fret}: {p:?}");
                    // An order may begin on the top tone of its group, and
                    // the group cannot lie above the position: its leap is
                    // as short as the position lets it be, not counted here.
                    let furthest = if figure == PositionFigure::Orders { 12 } else { 5 };
                    if let Some(last) = last {
                        let leap = (p[0] - last).abs();
                        assert!(leap > 0 && leap <= furthest, "{figure:?} at {fret} into {chord:?}: {last} -> {}", p[0]);
                    }
                    last = p.last().copied();
                }
            }
        }
    }
}

#[cfg(test)]
mod examples {
    use super::*;
    use crate::model::parse_chords_line_pub;

    /// The guide's examples as the program builds them, one line a chord:
    /// `chord|figure|start|pitch classes|names`.
    /// `cargo test --release harris_examples -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn harris_examples() {
        let cases = [
            ("C7", 0, 0), ("Cm7 F7 BbMaj7 EbMaj7", 0, 3), ("Am7b5 D7 Gm7", 0, 0),
            ("F7", 2, 0), ("Cm7 F7 BbMaj7", 3, 0), ("Cm7 F7 BbMaj7 EbMaj7", 0, 2),
            ("Cm7 F7 BbMaj7 EbMaj7", 1, 3), ("C7 Cm7 Am7b5 CMaj7 Cdim7", 0, 0),
            ("Dm7 G7 CMaj7", 3, 0), ("Dm7 G7 CMaj7", 4, 0), ("Dm7 G7 CMaj7", 5, 0), ("Dm7 G7 CMaj7", 6, 0),
            ("Dm7 G7 CMaj7", 7, 0), ("Dm7 G7 CMaj7", 5, 1), ("Dm7 G7 CMaj7", 6, 1), ("Dm7 G7 CMaj7", 7, 1),
        ];
        // The arpeggio over the changes, a chord at a time: four notes. Its
        // figure is printed as 800 or 900 plus the notes, to tell it apart.
        // Greene's figures at the seventh fret: printed as 1000 plus the
        // figure, then the tensions, the tones in a group, the pass and the
        // notes over a chord - 0 the whole position.
        let greenes = [
            ("CMaj7", 0usize, false, 4, 0, None), ("CMaj7", 0, true, 4, 0, None), ("CMaj7", 1, false, 4, 0, None),
            ("CMaj7", 2, false, 4, 0, None), ("CMaj7", 1, false, 3, 0, None), ("CMaj7", 1, false, 5, 0, None),
            ("CMaj7", 3, false, 4, 0, None), ("CMaj7", 3, false, 4, 8, None), ("CMaj7", 3, false, 4, 15, None),
            ("CMaj7 F7 CMaj7 F7", 0, false, 4, 0, Some(8)), ("CMaj7 F7 CMaj7 F7", 0, true, 4, 0, Some(8)),
            ("CMaj7 A7 Dm7 G7", 1, false, 4, 0, Some(4)), ("CMaj7 A7 Dm7 G7", 3, false, 4, 8, Some(4)),
        ];
        for (tune, figure, tensions, group, seed, notes) in greenes {
            let chords = parse_chords_line_pub(tune);
            let recipe = PositionRecipe { figure: figure.into(), tensions, fret: 7, notes, group, seed };
            for (chord, steps) in chords.iter().zip(position_line(&chords, recipe)) {
                let ring = ring(&chord.quality).unwrap();
                let steps = steps.unwrap();
                let pcs: Vec<String> = steps.iter().map(|&s| (chord.root as i32 + step_semitones(ring, s)).rem_euclid(12).to_string()).collect();
                let names: Vec<String> = steps.iter().map(|s| super::names(ring, &chord.quality)[s.degree].clone()).collect();
                let settings = format!("{}|{}|{group}|{seed}|{}", 1000 + figure, tensions as u8, notes.unwrap_or(0));
                println!("EX|{tune}|{settings}|{:?}|{:?}|{}|{}", chord.root, chord.quality, pcs.join(" "), names.join(" "));
            }
        }
        let fours = [("Dm7 G7 CMaj7", 8, 0), ("Dm7 G7 CMaj7", 8, 1), ("Dm7 G7 CMaj7", 8, 2), ("Dm7 G7 CMaj7", 8, 3)];
        let cases = cases.iter().map(|&(t, f, s)| (t, f, s, 8)).chain(fours.iter().map(|&(t, f, s)| (t, f, s, 4)));
        for (tune, figure, start, notes) in cases {
            let chords = parse_chords_line_pub(tune);
            // Counted from the scale down, the first of the lines: the six
            // cells stand before it in the list.
            let recipe = Recipe { figure: (figure + 6).into(), start: start.into(), notes, ..Default::default() };
            let figure = if figure >= 8 { figure * 100 + notes } else { figure };
            for (chord, steps) in chords.iter().zip(line(&chords, recipe)) {
                let ring = ring(&chord.quality).unwrap();
                let steps = steps.unwrap();
                let pcs: Vec<String> = steps.iter().map(|&s| (chord.root as i32 + step_semitones(ring, s)).rem_euclid(12).to_string()).collect();
                let names: Vec<String> = steps.iter().map(|s| super::names(ring, &chord.quality)[s.degree].clone()).collect();
                println!("EX|{tune}|{figure}|{start}|{:?}|{:?}|{}|{}", chord.root, chord.quality, pcs.join(" "), names.join(" "));
            }
        }
    }

}

