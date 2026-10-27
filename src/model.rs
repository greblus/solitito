use std::fs;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoteName {
    C, Df, D, Ef, E, F, Fsh, G, Af, A, Bf, B
}

pub const ALL_NOTES: [NoteName; 12] = [
    NoteName::C, NoteName::Df, NoteName::D, NoteName::Ef, NoteName::E, NoteName::F,
    NoteName::Fsh, NoteName::G, NoteName::Af, NoteName::A, NoteName::Bf, NoteName::B
];

impl NoteName {
    pub fn from_index(i: usize) -> Self {
        ALL_NOTES[i % 12]
    }
    
    pub fn to_string(&self) -> &str {
        match self {
            NoteName::C => "C",  NoteName::Df => "Db", NoteName::D => "D",
            NoteName::Ef => "Eb", NoteName::E => "E",  NoteName::F => "F",
            NoteName::Fsh => "F#", NoteName::G => "G",  NoteName::Af => "Ab",
            NoteName::A => "A",  NoteName::Bf => "Bb", NoteName::B => "B",
        }
    }
}

/// One step of an exercise: which chord degree, and in which octave.
///
/// The octave is DISPLAY ONLY. The model reports 12 pitch classes with no
/// octave (`brain::Prediction::pitches`), so `1` and `1'` are verified
/// identically - the marker tells the player where to go, nothing more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Step {
    /// Index into the chord's `interval_names()`.
    pub degree: usize,
    /// 0 = base octave, 1 = one up (`'`), -1 = one down (`,`).
    pub octave: i8,
}

/// Splits an octave marker off a degree token.
///
/// `"b3''"` -> `("b3", 2)`, `"5,"` -> `("5", -1)`, `"7"` -> `("7", 0)`.
/// Mixing `'` and `,` in one token is nonsense and yields their sum, which is
/// as good an answer as any for input nobody should write.
pub fn split_octave(token: &str) -> (&str, i8) {
    let base = token.trim_end_matches(['\'', ',']);
    let marks = &token[base.len()..];
    let up = marks.matches('\'').count() as i8;
    let down = marks.matches(',').count() as i8;
    (base, up - down)
}

/// Appends the octave marker back onto a name for display: `("b3", 1)` -> `"b3'"`.
pub fn with_octave(name: &str, octave: i8) -> String {
    match octave {
        0 => name.to_string(),
        n if n > 0 => format!("{name}{}", "'".repeat(n as usize)),
        n => format!("{name}{}", ",".repeat((-n) as usize)),
    }
}

/// A degree as the strip, the tiles and the exercise list show it. Above the
/// octave it takes the compound number intervals are written with - octave 8,
/// ninth 9, tenth 10, eleventh 11 and on, 15 the double octave - rather than a
/// mark: `("1", 1)` -> `"8"`, `("b3", 1)` -> `"b10"`, `("2", 1)` -> `"9"`. Below
/// the root it is the plain degree; which way the line goes is in its order,
/// and where on the neck in the drawing. A name that is not a numbered degree
/// keeps its marks, as `with_octave` writes them.
pub fn shown_degree(name: &str, octave: i8) -> String {
    let (open, inner, close) = match name.strip_prefix('(').and_then(|n| n.strip_suffix(')')) {
        Some(inner) => ("(", inner, ")"),
        None => ("", name, ""),
    };
    let accidental = inner.trim_end_matches(|c: char| c.is_ascii_digit());
    let Ok(number) = inner[accidental.len()..].parse::<i32>() else {
        return with_octave(name, octave);
    };
    let compound = number + 7 * i32::from(octave.max(0));
    format!("{open}{accidental}{compound}{close}")
}

/// The degrees of a scale as jazz writes them over its chord. What lies
/// between the chord's tones is a tension above it - a 2 is a 9, a 4 an 11, a
/// 6 a 13, each with its accidental - and nothing is counted past the octave:
/// the root on top is 1 again, and where on the neck is in the drawing. A
/// plain 4 over a major third stays 4: a semitone above the third, it is the
/// note to pass over there, not a tension to hold.
pub fn jazz_names(names: &[String]) -> Vec<String> {
    let major_third = names.iter().any(|n| n == "3");
    names.iter().map(|n| jazz_degree(n, major_third)).collect()
}

fn jazz_degree(name: &str, major_third: bool) -> String {
    let (open, inner, close) = match name.strip_prefix('(').and_then(|n| n.strip_suffix(')')) {
        Some(inner) => ("(", inner, ")"),
        None => ("", name, ""),
    };
    let accidental = inner.trim_end_matches(|c: char| c.is_ascii_digit());
    let Ok(number) = inner[accidental.len()..].parse::<i32>() else {
        return name.to_string();
    };
    let jazz = match (number - 1) % 7 + 1 {
        2 => 9,
        4 if accidental.is_empty() && major_third => 4,
        4 => 11,
        6 => 13,
        plain => plain,
    };
    format!("{open}{accidental}{jazz}{close}")
}

/// A written phrase as the text field shows it: tones above the octave by
/// their compound numbers - `"1 3 5 7 1' 3'"` -> `"1 3 5 7 8 10"`. Tones below
/// the root keep their commas: the field is edited, and a line going below the
/// root has no other way to say so. See `written_phrase` for the way back.
pub fn shown_phrase(text: &str) -> String {
    text.split_whitespace()
        .map(|token| {
            let (name, octave) = split_octave(token);
            if octave > 0 && plain_degree(name).is_some() {
                shown_degree(name, octave)
            } else {
                token.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A phrase typed or shown with compound numbers, back in the octave marks
/// the exercises are read in: 8 -> 1', 10 -> 3', 15 -> 1''. The ninth,
/// eleventh and thirteenth stay as written: a 9 means the tension, which is
/// what it meant before.
pub fn written_phrase(text: &str) -> String {
    text.split_whitespace()
        .map(|token| {
            let accidental = token.trim_end_matches(|c: char| c.is_ascii_digit());
            match token[accidental.len()..].parse::<i8>() {
                Ok(number) if number >= 8 && !matches!(number, 9 | 11 | 13) => {
                    let base = (number - 1) % 7 + 1;
                    with_octave(&format!("{accidental}{base}"), (number - 1) / 7)
                }
                _ => token.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The number of a degree written within the octave, 1 to 7, with any
/// accidental before it.
fn plain_degree(name: &str) -> Option<i8> {
    let accidental = name.trim_end_matches(|c: char| c.is_ascii_digit());
    name[accidental.len()..].parse::<i8>().ok().filter(|n| (1..=7).contains(n))
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScaleDefinition {
    pub name: String,
    pub intervals: Vec<u8>,
    pub names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChordQuality {
    Major7, Minor7, Dominant7, HalfDiminished, Diminished,
    /// The minor triad with a major sixth - and the m7b5 a minor third below
    /// it, note for note: A m6 is F# m7b5.
    Minor6,
    /// The minor triad with a major seventh.
    MinorMajor7,
    CustomScale(ScaleDefinition),
}

impl ChordQuality {
    pub fn to_string(&self) -> String {
        match self {
            ChordQuality::Major7 => "Maj7".to_string(),
            ChordQuality::Minor7 => "m7".to_string(),
            ChordQuality::Dominant7 => "7".to_string(),
            ChordQuality::HalfDiminished => "m7b5".to_string(),
            // "dim" is what brain.rs::quality_suffix emits, and state.rs matches
            // the two as strings - they have to agree.
            ChordQuality::Diminished => "dim".to_string(),
            ChordQuality::Minor6 => "m6".to_string(),
            ChordQuality::MinorMajor7 => "mMaj7".to_string(),
            ChordQuality::CustomScale(def) => def.name.clone(),
        }
    }

    pub fn intervals(&self) -> Vec<u8> {
        match self {
            ChordQuality::Major7 => vec![0, 4, 7, 11],
            ChordQuality::Minor7 => vec![0, 3, 7, 10],
            ChordQuality::Dominant7 => vec![0, 4, 7, 10],
            ChordQuality::HalfDiminished => vec![0, 3, 6, 10],
            // bb7 is 9 semitones, not 10 - that is the whole difference from m7b5.
            ChordQuality::Diminished => vec![0, 3, 6, 9],
            ChordQuality::Minor6 => vec![0, 3, 7, 9],
            ChordQuality::MinorMajor7 => vec![0, 3, 7, 11],
            ChordQuality::CustomScale(def) => def.intervals.clone(),
        }
    }
    
    pub fn interval_names(&self) -> Vec<String> {
        match self {
            ChordQuality::Major7 => vec!["1", "3", "5", "7"].iter().map(|s| s.to_string()).collect(),
            ChordQuality::Minor7 => vec!["1", "b3", "5", "b7"].iter().map(|s| s.to_string()).collect(),
            ChordQuality::Dominant7 => vec!["1", "3", "5", "b7"].iter().map(|s| s.to_string()).collect(),
            ChordQuality::HalfDiminished => vec!["1", "b3", "b5", "b7"].iter().map(|s| s.to_string()).collect(),
            ChordQuality::Diminished => vec!["1", "b3", "b5", "bb7"].iter().map(|s| s.to_string()).collect(),
            ChordQuality::Minor6 => vec!["1", "b3", "5", "6"].iter().map(|s| s.to_string()).collect(),
            ChordQuality::MinorMajor7 => vec!["1", "b3", "5", "7"].iter().map(|s| s.to_string()).collect(),
            ChordQuality::CustomScale(def) => def.names.clone(),
        }
    }
}

/// The scale a chord is played in, beyond the chord's own four tones.
///
/// Major sevenths take the Lydian scale and half-diminished chords Locrian
/// with a natural ninth - the sounds that leave the chord unclouded. Diminished
/// chords take the half-whole scale, its major seventh as the passing note -
/// the name `7` would clash with the chord's own seventh in any other role.
/// The two minor chords with a sixth or a major seventh take melodic minor.
struct ScaleSpec {
    /// The scale's tones that are not the chord's, as names and semitones.
    tensions: &'static [(&'static str, u8)],
    /// Tones outside the scale an exercise still asks for: a major seventh's
    /// plain fourth, which a suspension resolving to the third needs.
    outside: &'static [(&'static str, u8)],
    /// The chromatic note added to make an eight-note scale whose chord tones
    /// fall every other note: the major seventh between the flat seventh and
    /// the root, or the sharp fifth between the fifth and the sixth. Written
    /// in brackets, as it is shown.
    passing: (&'static str, u8),
}

const SEVENTH_PASSING: (&str, u8) = ("(7)", 11);
const FIFTH_PASSING: (&str, u8) = ("(#5)", 8);

const SCALES: [ScaleSpec; 7] = [
    // Major seventh: Lydian.
    ScaleSpec { tensions: &[("2", 2), ("#4", 6), ("6", 9)], outside: &[("4", 5)], passing: FIFTH_PASSING },
    // Dominant seventh: Mixolydian.
    ScaleSpec { tensions: &[("2", 2), ("4", 5), ("6", 9)], outside: &[], passing: SEVENTH_PASSING },
    // Minor seventh: Dorian.
    ScaleSpec { tensions: &[("2", 2), ("4", 5), ("6", 9)], outside: &[], passing: SEVENTH_PASSING },
    // Half-diminished: Locrian with a natural ninth.
    ScaleSpec { tensions: &[("2", 2), ("4", 5), ("b6", 8)], outside: &[], passing: SEVENTH_PASSING },
    // Diminished seventh: half-whole.
    ScaleSpec { tensions: &[("2", 2), ("4", 5), ("b6", 8)], outside: &[], passing: SEVENTH_PASSING },
    // Minor sixth: melodic minor.
    ScaleSpec { tensions: &[("2", 2), ("4", 5), ("7", 11)], outside: &[], passing: FIFTH_PASSING },
    // Minor-major seventh: melodic minor.
    ScaleSpec { tensions: &[("2", 2), ("4", 5), ("6", 9)], outside: &[], passing: FIFTH_PASSING },
];

impl ChordQuality {
    /// What an interval exercise may ask for over the chord: its own four
    /// tones first, so a step written against the chord alone means the same
    /// here, then the rest of its scale, the tones outside it, and the passing
    /// note - see `ScaleSpec`.
    pub fn scale_names(&self) -> Vec<String> {
        let mut names = self.interval_names();
        names.extend(self.extra_tones().map(|(name, _)| name.to_string()));
        names
    }

    /// Semitones above the root, in the order of `scale_names`.
    pub fn scale_intervals(&self) -> Vec<u8> {
        let mut intervals = self.intervals();
        intervals.extend(self.extra_tones().map(|(_, semitone)| semitone));
        intervals
    }

    /// The scale from the root up, as indices into `scale_names`: the chord's
    /// tones and its scale's, without the tones outside it, with the passing
    /// note where `passing` asks for it.
    pub fn scale_in_order(&self, passing: bool) -> Vec<usize> {
        let Some(spec) = self.spec() else {
            return (0..self.intervals().len()).collect();
        };
        let intervals = self.scale_intervals();
        let in_scale = 4 + spec.tensions.len();
        let passing_at = in_scale + spec.outside.len();
        let mut order: Vec<usize> = (0..in_scale).collect();
        if passing {
            order.push(passing_at);
        }
        order.sort_by_key(|&i| intervals[i]);
        order
    }

    fn extra_tones(&self) -> impl Iterator<Item = (&'static str, u8)> {
        self.spec()
            .into_iter()
            .flat_map(|spec| spec.tensions.iter().chain(spec.outside).chain(std::iter::once(&spec.passing)))
            .copied()
    }

    fn spec(&self) -> Option<&'static ScaleSpec> {
        let row = match self {
            ChordQuality::Major7 => 0,
            ChordQuality::Dominant7 => 1,
            ChordQuality::Minor7 => 2,
            ChordQuality::HalfDiminished => 3,
            ChordQuality::Diminished => 4,
            ChordQuality::Minor6 => 5,
            ChordQuality::MinorMajor7 => 6,
            ChordQuality::CustomScale(_) => return None,
        };
        Some(&SCALES[row])
    }
}

#[derive(Debug, Clone)]
pub struct Chord {
    pub root: NoteName,
    pub quality: ChordQuality,
}

impl Chord {
    pub fn get_target_indices(&self) -> Vec<usize> {
        let root_idx = self.root as usize;
        self.quality.intervals().iter()
            .map(|interval| (root_idx + *interval as usize) % 12)
            .collect()
    }

    /// Pitch classes in the order of `ChordQuality::scale_names`.
    pub fn scale_targets(&self) -> Vec<usize> {
        let root_idx = self.root as usize;
        self.quality.scale_intervals().iter()
            .map(|interval| (root_idx + *interval as usize) % 12)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Song {
    pub title: String,
    pub chords: Vec<Chord>,
}

// --- ARPEGGIO PATTERNS ---
// Two octaves, not one. A single-octave four-note arpeggio is a finger
// exercise; jazz practice runs the shape through the range and turns it over.
// Degrees carry an octave marker - see `split_octave`.
/// The three chord-quality studies come first: they are the ones the app was
/// built around, and the ones a player reaches for. Each is written out exactly
/// as it is played in its source - `a_book_study_is_written_out_as_it_is_played`
/// checks every one against the phrase read out of the file it came from. The
/// plain up and down runs are not studies; they are the simplest thing to hand
/// a beginner, and the changes exercise builds its own.
pub const ARPEGGIOS_PATTERNS_DEF: &str = r#"
Minor (Two Octaves and a Third)
1 3 5 7 1' 3' 5' 7' 1'' 3'' 1'' 7' 5' 3' 1' 7 5 3 1

Major (Leading Tone)
1 3 5 7 1' 3' 5' 7' 1'' 7' 5' 3' 1' 7 5 3 1 7, 1

Dominant (Approach from Below)
1 3 5 7 1' 3' 5' 7' 5' 3' 1' 7 5 3 1 7, 5, 7, 1

Skipping Notes (Fifths and Fourths)
1 5 3 7 5 1' 7 3' 1' 5' 3' 7' 5' 1'' 7' 3'' 1'' 5' 7' 3' 5' 1' 3' 7 1' 5 7 3 5 1

Triplets (Up-Down)
1 3 5 3 5 7 5 7 1' 7 1' 3' 1' 3' 5' 3' 5' 3' 5' 7' 5' 7' 1'' 3'' 1'' 7' 5' 7' 5' 3' 5' 3' 1' 3' 1' 7 1' 7 5 7 5 7 5 3 5 3 1

Two Octaves Up-Down
1 3 5 7 1' 3' 5' 7' 5' 3' 1' 7 5 3 1

Two Octaves Up
1 3 5 7 1' 3' 5' 7'

Two Octaves Down
7' 5' 3' 1' 7 5 3 1

Two Octaves Down from the Root
1 7, 5, 3, 1, 7,, 5,, 3,,
"#;

const BUILTIN_SCALES_DEF: &str = r#"
--- BASIC SCALES ---
Major Scale (Ionian)
1 2 3 4 5 6 7

Natural Minor (Aeolian)
1 2 b3 4 5 b6 b7

Harmonic Minor
1 2 b3 4 5 b6 7

Melodic Minor (Jazz)
1 2 b3 4 5 6 7

--- PENTATONIC / BLUES ---
Pentatonic Minor
1 b3 4 5 b7

Pentatonic Major
1 2 3 5 6

Pentatonic Dominant
1 3 4 5 b7

Pentatonic Minor 6
1 b3 4 5 6

Blues Scale
1 b3 4 #4 5 b7

Major Blues Scale
1 2 b3 3 5 6

--- SYMMETRIC SCALES ---
Whole-Half Diminished
1 2 b3 4 b5 #5 6 7

Half-Whole Diminished (Dominant)
1 b2 #2 3 #4 5 6 b7

Whole Tone
1 2 3 #4 #5 b7

Augmented Scale
1 #2 3 5 #5 7

--- JAZZ MODES ---
Altered Scale (Super Locrian)
1 b2 #2 3 b5 #5 b7

Lydian Dominant
1 2 3 #4 5 6 b7

Lydian Augmented
1 2 3 #4 #5 6 7

Phrygian Dominant
1 b2 3 4 5 b6 b7

Locrian #2
1 2 b3 4 b5 b6 b7

Dorian b2
1 b2 b3 4 5 6 b7

Mixolydian b6
1 2 3 4 5 b6 b7

--- STANDARD MODES ---
Dorian
1 2 b3 4 5 6 b7

Mixolydian
1 2 3 4 5 6 b7

Lydian
1 2 3 #4 5 6 7

Phrygian
1 b2 b3 4 5 b6 b7

Locrian
1 b2 b3 4 b5 b6 b7

--- HARMONIC MINOR MODES ---
Locrian Natural 6
1 b2 b3 4 b5 6 b7

Ionian #5
1 2 3 4 #5 6 7

Dorian #4
1 2 b3 #4 5 6 b7

Lydian #9
1 #2 3 #4 5 6 7

Superlocrian bb7
1 b2 b3 b4 b5 b6 bb7

--- HARMONIC MAJOR MODES ---
Harmonic Major
1 2 3 4 5 b6 7

Dorian b5
1 2 b3 4 b5 6 b7

Phrygian b4
1 b2 b3 b4 5 b6 b7

Lydian b3
1 2 b3 #4 5 6 7

Mixolydian b2
1 b2 3 4 5 6 b7

Lydian Augmented #2
1 #2 3 #4 #5 6 7

Locrian bb7
1 b2 b3 4 b5 b6 bb7

--- BEBOP SCALES ---
Bebop Dominant
1 2 3 4 5 6 b7 7

Bebop Major
1 2 3 4 5 #5 6 7

Bebop Dorian
1 2 b3 3 4 5 6 b7

Bebop Dorian (major 7th)
1 2 b3 4 5 6 b7 7

Bebop Dominant b9 b13
1 b2 3 4 5 b6 b7 7

Bebop Tonic Minor (Dorian)
1 2 b3 4 5 b6 6 b7

--- BARRY HARRIS (6th Dim) ---
Major 6 Diminished
1 2 3 4 5 b6 6 7

Minor 6 Diminished
1 2 b3 4 5 b6 6 7

Dominant 7th Diminished
1 2 3 4 5 b6 b7 7
"#;

const SONGS_DB: &str = r#"
Autumn Leaves
Cm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7 Gm7 Cm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7 Gm7 Am7b5 D7 Gm7 Gm7 Cm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7

All The Things You Are
Fm7 Bbm7 Eb7 AbMaj7 DbMaj7 G7 CMaj7 CMaj7 Cm7 Fm7 Bb7 EbMaj7 AbMaj7 Am7 D7 GMaj7 GMaj7 Am7 D7 GMaj7 GMaj7 F#m7 B7 EMaj7 C7+5 Fm7 Bbm7 Eb7 AbMaj7 DbMaj7 Gbm7 C7 Fm7 Fm7

Blue Bossa
Cm7 Cm7 Fm7 Fm7 Dm7b5 G7 Cm7 Cm7 Ebm7 Ab7 DbMaj7 DbMaj7 Dm7b5 G7 Cm7 Dm7b5 G7

Take The A Train
CMaj7 CMaj7 D7b5 D7b5 Dm7 G7 CMaj7 Dm7 G7 CMaj7 CMaj7 D7b5 D7b5 Dm7 G7 CMaj7 CMaj7 FMaj7 FMaj7 FMaj7 FMaj7 D7 D7 Dm7 G7 CMaj7 CMaj7 D7b5 D7b5 Dm7 G7 CMaj7 CMaj7

Stella By Starlight
Em7b5 A7 Cm7 F7 Fm7 Bb7 EbMaj7 Ab7 BbMaj7 Em7b5 A7 Dm7b5 G7 Cm7b5 F7 BbMaj7 Em7b5 A7 Dm7b5 G7 Cm7b5 F7 BbMaj7 BbMaj7

Satin Doll
Dm7 G7 Dm7 G7 Em7 A7 Em7 A7 D7 Db7 CMaj7 CMaj7 Dm7 G7 Dm7 G7 Em7 A7 Em7 A7 D7 Db7 CMaj7 CMaj7 GMaj7 Gm7 C7 FMaj7 FMaj7 Am7 D7 GMaj7 GMaj7 Dm7 G7 Dm7 G7 Em7 A7 Em7 A7 D7 Db7 CMaj7 CMaj7

Girl From Ipanema
FMaj7 FMaj7 G7 G7 Gm7 Gb7 FMaj7 Gb7 FMaj7 FMaj7 G7 G7 Gm7 Gb7 FMaj7 FMaj7 GbMaj7 GbMaj7 B7 B7 F#m7 F#m7 D7 D7 Gm7 Gm7 Eb7 Eb7 Am7 D7 Gm7 C7

Black Orpheus
Am7 Bm7b5 E7 Am7 Dm7 G7 CMaj7 FMaj7 Bm7b5 E7 Am7 E7 Am7 Bm7b5 E7 Am7 Dm7 G7 CMaj7 FMaj7 Bm7b5 E7 Am7 Am7 Bm7b5 E7 Am7 Dm7 G7 CMaj7 FMaj7 Bm7b5 E7 Am7 Am7

Misty
EbMaj7 Cm7 Fm7 Bb7 Gm7 C7 Fm7 Bb7 EbMaj7 Cm7 Fm7 Bb7 EbMaj7 EbMaj7 Bbm7 Eb7 AbMaj7 Abm7 Db7 EbMaj7 Cm7 Fm7 Bb7 EbMaj7 EbMaj7

My Funny Valentine
Cm CmMaj7 Cm7 Cm6 AbMaj7 Fm7 Dm7b5 G7 Cm CmMaj7 Cm7 Cm6 AbMaj7 Fm7 Dm7b5 G7 EbMaj7 Fm7 Gm7 Fm7 EbMaj7 Fm7 Gm7 Fm7 AbMaj7 Fm7 Dm7b5 G7 Cm7 Bbm7 A7 AbMaj7 Dm7b5 G7 Cm7 G7

Someday My Prince Will Come
BbMaj7 D7 EbMaj7 G7 Cm7 G7 Cm7 F7 Dm7 Dbdim Cm7 F7 BbMaj7 G7 Cm7 F7 BbMaj7 D7 EbMaj7 G7 Cm7 G7 Cm7 F7 Dm7 Dbdim Cm7 F7 BbMaj7 F7 BbMaj7 BbMaj7

Yesterdays
Dm7b5 G7 Cm7 F7 BbMaj7 BbMaj7 Em7b5 A7 Dm7 Dm7 Dm7 Dm7 Em7b5 A7 Dm7 Dm7 Dm7 Dm7

Song For My Father
Fm7 Fm7 Eb7 Eb7 Db7 C7 Fm7 Fm7 Fm7 Fm7 Eb7 Eb7 Db7 C7 Fm7 Fm7 Eb9 Eb9 Db9 Db9 Fm7 Fm7 Eb9 Eb9 Db9 C7 Fm7 Fm7

Maiden Voyage
D9 D9 D9 D9 F9 F9 F9 F9 Eb9 Eb9 Eb9 Eb9 Db9 Db9 Db9 Db9

Cantaloupe Island
Fm7 Fm7 Fm7 Fm7 Db7 Db7 Db7 Db7 Dm7 Dm7 Dm7 Dm7 Fm7 Fm7 Fm7 Fm7

Watermelon Man
F7 F7 F7 F7 Bb7 Bb7 F7 F7 C7 Bb7 F7 C7

Impressions
Dm7 Dm7 Dm7 Dm7 Dm7 Dm7 Dm7 Dm7 Ebm7 Ebm7 Ebm7 Ebm7 Dm7 Dm7 Dm7 Dm7

Solar
CmMaj7 CmMaj7 Gm7 C7 FMaj7 FMaj7 Fm7 Bb7 EbMaj7 EbMaj7 Ebm7 Ab7 DbMaj7 Dm7b5 G7

Tune Up
Em7 A7 DMaj7 DMaj7 Dm7 G7 CMaj7 CMaj7 Cm7 F7 BbMaj7 BbMaj7 Em7 A7 DMaj7 DMaj7

So What
Dm7 Dm7 Dm7 Dm7 Dm7 Dm7 Dm7 Dm7 Ebm7 Ebm7 Ebm7 Ebm7 Dm7 Dm7 Dm7 Dm7

Giant Steps
BMaj7 D7 GMaj7 Bb7 EbMaj7 Am7 D7 GMaj7 Bb7 EbMaj7 F#7 BMaj7 Fm7 Bb7 EbMaj7 Am7 D7 GMaj7 C#m7 F#7 BMaj7 Fm7 Bb7 EbMaj7 C#m7 F#7

Body And Soul
Ebm7 Bb7 Ebm7 D7 DbMaj7 Gb7 Fm7 E7 EbMaj7 Em7 A7 DMaj7 Em7 A7 DMaj7 Dm7 G7 CMaj7 Ebdim Dm7 G7 CMaj7 B7 Em7 A7 D7 Gm7 C7 Fm7 Bb7 EbMaj7 Ab7 DbMaj7

There Is No Greater Love
BbMaj7 BbMaj7 Eb7 Eb7 BbMaj7 G7 Cm7 F7 Dm7 G7 Cm7 F7 BbMaj7 G7 Cm7 F7 BbMaj7 BbMaj7 Eb7 Ab7 Dm7 G7 Cm7 F7 BbMaj7 G7 Cm7 F7

All Blues
G7 G7 G7 G7 Gm7 Gm7 G7 G7 D7 Eb7 D7 G7 G7

Footprints
Cm7 Cm7 Cm7 Cm7 Fm7 Fm7 Cm7 Cm7 D7 Db7 Cm7 Cm7

Four
EbMaj7 EbMaj7 Ebm7 Ab7 Fm7 Bb7 Bbm7 Eb7 AbMaj7 AbMaj7 Abm7 Db7 EbMaj7 C7 Fm7 Bb7

Have You Met Miss Jones
FMaj7 F#dim Gm7 C7 Am7 Dm7 Gm7 C7 BbMaj7 Abm7 Db7 GbMaj7 Em7 A7 DMaj7 Abm7 Db7 GbMaj7 Gm7 C7 FMaj7 F#dim Gm7 C7 FMaj7

How High The Moon
GMaj7 GMaj7 Gm7 C7 FMaj7 FMaj7 Fm7 Bb7 EbMaj7 Am7b5 D7 Gm7 Am7b5 D7 GMaj7 Gm7 C7 FMaj7 Fm7 Bb7 EbMaj7 Am7b5 D7 GMaj7

Just Friends
FMaj7 FMaj7 FMaj7 FMaj7 Fm7 Bb7 C7 C7 A7 A7 D7 D7 Dm7 G7 CMaj7 CMaj7

Lady Bird
CMaj7 CMaj7 Fm7 Bb7 CMaj7 CMaj7 Bbm7 Eb7 AbMaj7 AbMaj7 Am7 D7 Dm7 G7 CMaj7 EbMaj7 AbMaj7 DbMaj7

Night And Day
EbMaj7 G7b5 CMaj7 C7 Fm7 Bb7 EbMaj7 EbMaj7 Dm7b5 G7 CMaj7 C7 Fm7 Bb7 EbMaj7 EbMaj7 Fm7b5 Bb7 EbMaj7 EbMaj7 Fm7b5 Bb7 EbMaj7 EbMaj7

Oleo
BbMaj7 Gm7 Cm7 F7 BbMaj7 Gm7 Cm7 F7 E7 A7 D7 G7 C7 F7 BbMaj7 Gm7 Cm7 F7

On Green Dolphin Street
EbMaj7 EbMaj7 EbMaj7 EbMaj7 GbMaj7 GbMaj7 GbMaj7 GbMaj7 FMaj7 FMaj7 EMaj7 EMaj7 EbMaj7 EbMaj7 Dm7 G7 C7 C7 Fm7 Bb7 EbMaj7 Dm7 G7

Recorda Me
Am7 Am7 Cm7 Cm7 Cm7 F7 BbMaj7 BbMaj7 Bbm7 Eb7 AbMaj7 AbMaj7 Gm7 C7 FMaj7 E7

St Thomas
CMaj7 CMaj7 Em7b5 A7 Dm7 G7 CMaj7 C7 FMaj7 Fm7 Em7 A7 Dm7 G7 CMaj7 G7

Wave
DMaj7 DMaj7 Bm7b5 E7 Em7 A7 DMaj7 Ddim Am7 D7 GMaj7 Gm6 F#7 F#7 B7 B7 E7 E7 A7 A7 Dm7 G7

Yardbird Suite
CMaj7 Fm7 Bb7 CMaj7 Bb7 A7 D7 D7 Dm7 G7 CMaj7 Fm7 Bb7 CMaj7 Bb7 A7 Dm7 G7 CMaj7 E7
"#;

pub fn load_all_scale_definitions() -> Vec<ScaleDefinition> {
    let mut scales = Vec::new();
    scales.extend(parse_scale_definitions(BUILTIN_SCALES_DEF));
    if let Ok(user_content) = fs::read_to_string("user_scales_def.txt") {
        scales.extend(parse_scale_definitions(&user_content));
    }
    scales
}

/// The degree tokens of a written phrase as steps, the way the note modes read
/// them: which chord tone, and how many octaves off the root.
pub fn steps_of(names: &[String]) -> Vec<Step> {
    names
        .iter()
        .filter_map(|token| {
            let (base, octave) = split_octave(token);
            let degree = match base {
                "1" | "8" => 0,
                "3" => 1,
                "5" => 2,
                "7" => 3,
                "9" => 4,
                _ => return None,
            };
            Some(Step { degree, octave })
        })
        .collect()
}

/// The chord a study is written over, as an index into `ARP_QUALITIES`.
///
/// The three named for a quality are played over it in the source: the minor
/// study over `m7`, the major one over `Maj7`, the dominant one over `7`.
/// Picking one therefore sets the chord as well - the shape on the neck is the
/// book's phrase only when it is read over the chord it was written for. The
/// rest name no quality and set none; a study is a sequence of degrees, and any
/// of them can be heard over any chord on purpose.
pub fn study_quality(name: &str) -> Option<usize> {
    match name.split_whitespace().next()? {
        "Minor" => Some(0),
        "Major" => Some(1),
        "Dominant" => Some(2),
        _ => None,
    }
}

pub fn load_arpeggio_patterns() -> Vec<ScaleDefinition> {
    let mut out = parse_scale_definitions(ARPEGGIOS_PATTERNS_DEF);
    // Last in the list: not a fixed phrase but a fresh one every pass, built from
    // the same vocabulary as the ones above it. The degrees here are only a
    // placeholder - state.rs swaps them out on selection and after each pass.
    out.push(ScaleDefinition {
        name: GENERATOR_NAME.to_string(),
        intervals: vec![0, 4, 7, 11],
        names: vec!["1".into(), "3".into(), "5".into(), "7".into()],
    });
    out
}

/// Marks the generated entry in the arpeggio list.
pub const GENERATOR_NAME: &str = "Generator (new phrase each pass)";

pub fn load_songs() -> Vec<Song> {
    let mut all = Vec::new();
    all.extend(parse_songs(SONGS_DB));
    if let Ok(c) = fs::read_to_string("user_songs.txt") { 
        all.extend(parse_songs(&c)); 
    }
    all
}

fn parse_scale_definitions(content: &str) -> Vec<ScaleDefinition> {
    let mut defs = Vec::new();
    let lines: Vec<&str> = content.trim().split('\n').filter(|l| !l.trim().is_empty()).collect();
    let valid_lines: Vec<&str> = lines.into_iter().filter(|l| !l.starts_with("---")).collect();
    
    for chunk in valid_lines.chunks(2) {
        if chunk.len() < 2 { break; }
        let name = chunk[0].trim().to_string();
        let (intervals, names) = parse_intervals_string(chunk[1].trim());
        if !intervals.is_empty() {
            defs.push(ScaleDefinition { name, intervals, names });
        }
    }
    defs
}

fn parse_intervals_string(s: &str) -> (Vec<u8>, Vec<String>) {
    let mut semitones = Vec::new();
    let mut names = Vec::new();
    for part in s.split_whitespace() {
        // The marker is not part of the interval name for pitch purposes, but it
        // MUST survive into `names`: that list is joined back into
        // `intervals_input`, so stripping it here would lose the octave on the
        // round trip.
        let (base, _oct) = split_octave(part);
        let semitone = match base {
            "1" | "8" => 0, 
            "b2" => 1, 
            "2" | "9" => 2, 
            "b3" | "#2" | "#9" => 3, 
            "3" | "b4" => 4, 
            "4" | "11" => 5,
            "b5" | "#4" | "#11" => 6, 
            "5" => 7, 
            "#5" | "b6" | "b13" => 8, 
            "6" | "bb7" | "13" => 9,
            "b7" | "#6" => 10, 
            "7" => 11, 
            "b9" => 1, 
            _ => 0,
        };
        semitones.push(semitone);
        names.push(part.to_string());
    }
    (semitones, names)
}

fn parse_songs(content: &str) -> Vec<Song> {
    let mut songs = Vec::new();
    let lines: Vec<&str> = content.trim().split('\n').filter(|l| !l.trim().is_empty()).collect();
    for chunk in lines.chunks(2) {
        if chunk.len() < 2 { continue; }
        songs.push(Song { title: chunk[0].trim().to_string(), chords: parse_chords_line(chunk[1]) });
    }
    songs
}

fn parse_chords_line(line: &str) -> Vec<Chord> {
    line.split_whitespace().filter_map(parse_single_chord).collect()
}

fn parse_single_chord(s: &str) -> Option<Chord> {
    let (root_str, qual_str) = if s.len() > 1 && (s.chars().nth(1).unwrap() == 'b' || s.chars().nth(1).unwrap() == '#') {
        (&s[0..2], &s[2..])
    } else {
        (&s[0..1], &s[1..])
    };
    let root = match root_str {
        "C" => NoteName::C, "C#" | "Db" => NoteName::Df, "D" => NoteName::D, "D#" | "Eb" => NoteName::Ef,
        "E" => NoteName::E, "F" => NoteName::F, "F#" | "Gb" => NoteName::Fsh, "G" => NoteName::G,
        "G#" | "Ab" => NoteName::Af, "A" => NoteName::A, "A#" | "Bb" => NoteName::Bf, "B" => NoteName::B,
        _ => return None, 
    };
    let quality = match qual_str.to_lowercase().as_str() {
        "maj7" | "m7" | "7" | "m7b5" => map_chord_quality(qual_str),
        _ => map_chord_quality(qual_str),
    };
    Some(Chord { root, quality })
}

fn map_chord_quality(s: &str) -> ChordQuality {
    match s {
        "Maj7" | "maj7" | "M7" | "Δ7" => ChordQuality::Major7,
        "m7" | "min7" | "-" | "-7" => ChordQuality::Minor7,
        "7" | "dom7" | "7b9" | "7#9" | "7alt" | "7b13" => ChordQuality::Dominant7,
        "m7b5" | "hdim" | "ø" => ChordQuality::HalfDiminished,
        // Without this the diminished chords in the standards fell through to the
        // catch-all and were taught as dominant sevenths.
        "dim" | "dim7" | "o" | "o7" | "°" => ChordQuality::Diminished,
        // The same catch-all made these dominant sevenths too: a major third
        // where the chord has a minor one.
        "m6" | "min6" | "-6" => ChordQuality::Minor6,
        "mMaj7" | "mmaj7" | "mM7" | "m(maj7)" | "-Δ7" => ChordQuality::MinorMajor7,
        _ => ChordQuality::Dominant7,
    }
}

#[cfg(test)]
mod octave_tests {
    use super::*;

    #[test]
    fn plain_token_has_no_marker() {
        assert_eq!(split_octave("7"), ("7", 0));
        assert_eq!(split_octave("b3"), ("b3", 0));
    }

    #[test]
    fn apostrophes_go_up_commas_go_down() {
        assert_eq!(split_octave("1'"), ("1", 1));
        assert_eq!(split_octave("1''"), ("1", 2));
        assert_eq!(split_octave("5,"), ("5", -1));
    }

    /// The marker must not be mistaken for part of the interval name, or "b7'"
    /// would fall through to the catch-all and silently become the root.
    #[test]
    fn accidentals_survive_the_split() {
        assert_eq!(split_octave("b7'"), ("b7", 1));
        assert_eq!(split_octave("#11''"), ("#11", 2));
    }

    #[test]
    fn display_puts_the_marker_back() {
        assert_eq!(with_octave("b3", 0), "b3");
        assert_eq!(with_octave("b3", 1), "b3'");
        assert_eq!(with_octave("1", 2), "1''");
        assert_eq!(with_octave("5", -1), "5,");
    }

    #[test]
    fn split_and_rejoin_round_trip() {
        for t in ["1", "b3'", "5''", "b7,", "#9'"] {
            let (name, oct) = split_octave(t);
            assert_eq!(with_octave(name, oct), t);
        }
    }

    /// A pattern written with octave markers must still yield the right
    /// semitones - the marker is display, not pitch.
    #[test]
    fn markers_do_not_change_the_semitones() {
        let (plain, _) = parse_intervals_string("1 b3 5 b7");
        let (marked, _) = parse_intervals_string("1' b3' 5' b7'");
        assert_eq!(plain, marked);
        assert_eq!(plain, vec![0, 3, 7, 10]);
    }
}

#[cfg(test)]
mod arpeggio_pattern_tests {
    use super::*;

    fn pattern(name: &str) -> ScaleDefinition {
        load_arpeggio_patterns()
            .into_iter()
            .find(|d| d.name == name)
            .unwrap_or_else(|| panic!("pattern {name:?} not found"))
    }

    /// The two-octave shape from the reference tab: up through both octaves,
    /// turn, come back down. Fifteen steps - the case the scrolling exists for.
    #[test]
    fn two_octave_pattern_parses_whole() {
        let d = pattern("Two Octaves Up-Down");
        assert_eq!(d.names.len(), 15, "got {:?}", d.names);
        assert_eq!(d.names[0], "1");
        assert_eq!(d.names[4], "1'", "the fifth step should start the upper octave");
        assert_eq!(d.names[14], "1", "it should land back on the root");
    }

    /// Semitones must ignore the markers - the octave is display only.
    #[test]
    fn octave_markers_do_not_leak_into_semitones() {
        let d = pattern("Two Octaves Up-Down");
        assert_eq!(d.intervals[0], d.intervals[4], "1 and 1' are the same pitch class");
        assert!(d.intervals.iter().all(|&s| s < 12), "a marker escaped into the semitones");
    }

    /// Two identical degrees in a row would both complete at once: the pitch
    /// class is already sounding when the second one is asked for.
    #[test]
    fn no_pattern_repeats_a_degree_back_to_back() {
        for d in load_arpeggio_patterns() {
            for w in d.names.windows(2) {
                let (a, _) = split_octave(&w[0]);
                let (b, _) = split_octave(&w[1]);
                assert_ne!(a, b, "pattern {:?} repeats {a} back to back", d.name);
            }
        }
    }
}

#[cfg(test)]
mod scale_library_tests {
    use super::*;

    /// Every scale reads as distinct notes rising from the root - a degree
    /// the parser did not know would read as 0 and repeat the root.
    #[test]
    fn every_built_in_scale_rises_through_distinct_notes() {
        let scales = parse_scale_definitions(BUILTIN_SCALES_DEF);
        assert_eq!(scales.len(), 47);
        for scale in &scales {
            assert_eq!(scale.intervals[0], 0, "{}", scale.name);
            assert!(scale.intervals.windows(2).all(|w| w[0] < w[1]), "{}: {:?}", scale.name, scale.intervals);
        }
    }

    #[test]
    fn the_harmonic_major_and_minor_modes_spell_out() {
        let scales = parse_scale_definitions(BUILTIN_SCALES_DEF);
        let of = |name: &str| scales.iter().find(|s| s.name == name).unwrap().intervals.clone();
        assert_eq!(of("Superlocrian bb7"), [0, 1, 3, 4, 6, 8, 9]);
        assert_eq!(of("Phrygian b4"), [0, 1, 3, 4, 7, 8, 10]);
        assert_eq!(of("Harmonic Major"), [0, 2, 4, 5, 7, 8, 11]);
        assert_eq!(of("Augmented Scale"), [0, 3, 4, 7, 8, 11]);
    }
}

#[cfg(test)]
mod jazz_names_tests {
    use super::jazz_names;

    fn jazz(text: &str) -> String {
        let names: Vec<String> = text.split_whitespace().map(str::to_string).collect();
        jazz_names(&names).join(" ")
    }

    #[test]
    fn tensions_are_named_above_the_chord() {
        assert_eq!(jazz("1 b2 3 4 5 b6 b7"), "1 b9 3 4 5 b13 b7");
        assert_eq!(jazz("1 2 b3 4 5 6 b7"), "1 9 b3 11 5 13 b7");
        assert_eq!(jazz("1 2 3 #4 5 6 7"), "1 9 3 #11 5 13 7");
        assert_eq!(jazz("1 b2 #2 3 b5 #5 b7"), "1 b9 #9 3 b5 #5 b7");
    }

    #[test]
    fn nothing_is_counted_past_the_octave() {
        assert_eq!(jazz("8 9 10 11 b13"), "1 9 3 11 b13");
    }

    #[test]
    fn a_passing_note_keeps_its_brackets() {
        assert_eq!(jazz("1 3 5 7 2 #4 6 4 (#5)"), "1 3 5 7 9 #11 13 4 (#5)");
        assert_eq!(jazz("1 b3 5 b7 2 4 6 (7)"), "1 b3 5 b7 9 11 13 (7)");
    }
}

#[cfg(test)]
mod shown_degree_tests {
    use super::shown_degree;

    #[test]
    fn above_the_octave_degrees_take_their_compound_numbers() {
        assert_eq!(shown_degree("1", 1), "8");
        assert_eq!(shown_degree("2", 1), "9");
        assert_eq!(shown_degree("b3", 1), "b10");
        assert_eq!(shown_degree("#4", 1), "#11");
        assert_eq!(shown_degree("5", 1), "12");
        assert_eq!(shown_degree("b6", 1), "b13");
        assert_eq!(shown_degree("bb7", 1), "bb14");
        assert_eq!(shown_degree("1", 2), "15");
        assert_eq!(shown_degree("(7)", 1), "(14)");
        assert_eq!(shown_degree("2", 0), "2");
        // Below the root: the plain degree.
        assert_eq!(shown_degree("7", -1), "7");
        assert_eq!(shown_degree("b3", -2), "b3");
    }
}

#[cfg(test)]
mod phrase_text_tests {
    use super::*;

    #[test]
    fn the_text_field_shows_compound_numbers_and_reads_them_back() {
        assert_eq!(shown_phrase("1 3 5 7 1' 3' 5' 7' 1''"), "1 3 5 7 8 10 12 14 15");
        assert_eq!(shown_phrase("1 7, 5, 3,"), "1 7, 5, 3,");
        assert_eq!(written_phrase("1 3 5 7 8 10 12 14 15"), "1 3 5 7 1' 3' 5' 7' 1''");
        assert_eq!(written_phrase("1 3 5 9 b10"), "1 3 5 9 b3'");
        // Every phrase the app ships comes back exactly as it was written.
        for phrase in parse_scale_definitions(ARPEGGIOS_PATTERNS_DEF) {
            let text = phrase.names.join(" ");
            assert_eq!(written_phrase(&shown_phrase(&text)), text, "{}", phrase.name);
        }
        for scale in parse_scale_definitions(BUILTIN_SCALES_DEF) {
            let text = scale.names.join(" ");
            assert_eq!(written_phrase(&shown_phrase(&text)), text, "{}", scale.name);
        }
    }
}
