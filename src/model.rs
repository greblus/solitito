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

#[derive(Debug, Clone, PartialEq)]
pub struct ScaleDefinition {
    pub name: String,
    pub intervals: Vec<u8>,
    pub names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChordQuality {
    Major7, Minor7, Dominant7, HalfDiminished, Diminished,
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
            ChordQuality::CustomScale(def) => def.names.clone(),
        }
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

Blues Scale
1 b3 4 #4 5 b7

--- SYMMETRIC SCALES ---
Whole-Half Diminished
1 2 b3 4 b5 #5 6 7

Half-Whole Diminished (Dominant)
1 b2 #2 3 #4 5 6 b7

Whole Tone
1 2 3 #4 #5 b7

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

--- BEBOP SCALES ---
Bebop Dominant
1 2 3 4 5 6 b7 7

Bebop Major
1 2 3 4 5 #5 6 7

Bebop Dorian
1 2 b3 3 4 5 6 b7

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
            "3" => 4, 
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
