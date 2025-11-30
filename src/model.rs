// src/model.rs
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
    
    pub fn to_index(&self) -> usize {
        *self as usize
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

#[derive(Debug, Clone, PartialEq)]
pub struct ScaleDefinition {
    pub name: String,
    pub intervals: Vec<u8>,
    pub names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChordQuality {
    Major7, Minor7, Dominant7, HalfDiminished,
    CustomScale(ScaleDefinition),
}

impl ChordQuality {
    pub fn to_string(&self) -> String {
        match self {
            ChordQuality::Major7 => "Maj7".to_string(),
            ChordQuality::Minor7 => "m7".to_string(),
            ChordQuality::Dominant7 => "7".to_string(),
            ChordQuality::HalfDiminished => "m7b5".to_string(),
            ChordQuality::CustomScale(def) => def.name.clone(),
        }
    }

    pub fn intervals(&self) -> Vec<u8> {
        match self {
            ChordQuality::Major7 => vec![0, 4, 7, 11],
            ChordQuality::Minor7 => vec![0, 3, 7, 10],
            ChordQuality::Dominant7 => vec![0, 4, 7, 10],
            ChordQuality::HalfDiminished => vec![0, 3, 6, 10],
            ChordQuality::CustomScale(def) => def.intervals.clone(),
        }
    }
    
    pub fn interval_names(&self) -> Vec<String> {
        match self {
            ChordQuality::Major7 => vec!["1", "3", "5", "7"].iter().map(|s| s.to_string()).collect(),
            ChordQuality::Minor7 => vec!["1", "b3", "5", "b7"].iter().map(|s| s.to_string()).collect(),
            ChordQuality::Dominant7 => vec!["1", "3", "5", "b7"].iter().map(|s| s.to_string()).collect(),
            ChordQuality::HalfDiminished => vec!["1", "b3", "b5", "b7"].iter().map(|s| s.to_string()).collect(),
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
    
    // TA METODA JEST WYMAGANA PRZEZ STATE.RS
    pub fn get_components(&self) -> Vec<usize> {
        self.get_target_indices()
    }
}

#[derive(Debug, Clone)]
pub struct Song {
    pub title: String,
    pub chords: Vec<Chord>,
}

const BUILTIN_SCALES_DEF: &str = r#"
Major Scale (Ionian)
1 2 3 4 5 6 7

Minor Scale (Aeolian)
1 2 b3 4 5 b6 b7

Dorian
1 2 b3 4 5 6 b7

Mixolydian
1 2 3 4 5 6 b7

Phrygian
1 b2 b3 4 5 b6 b7

Lydian
1 2 3 #4 5 6 7

Locrian
1 b2 b3 4 b5 b6 b7

Pentatonic Minor
1 b3 4 5 b7

Pentatonic Major
1 2 3 5 6

Blues Scale
1 b3 4 #4 5 b7
"#;

const SONGS_DB: &str = r#"
Giant Steps
BMaj7 D7 GMaj7 Bb7 EbMaj7 Am7 D7 GMaj7 Bb7 EbMaj7 F#7 BMaj7 Fm7 Bb7 EbMaj7 Am7 D7 GMaj7 C#m7 F#7 BMaj7 Fm7 Bb7 EbMaj7 C#m7 F#7

Autumn Leaves
Cm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7 Gm7 Cm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7 Gm7 Am7b5 D7 Gm7 Gm7 Cm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7

Blue Bossa
Cm7 Cm7 Fm7 Fm7 Dm7b5 G7 Cm7 Cm7 Ebm7 Ab7 DbMaj7 DbMaj7 Dm7b5 G7 Cm7 Dm7b5 G7
"#;

pub fn load_all_scale_definitions() -> Vec<ScaleDefinition> {
    let mut scales = Vec::new();
    scales.extend(parse_scale_definitions(BUILTIN_SCALES_DEF));
    if let Ok(user_content) = fs::read_to_string("user_scales_def.txt") {
        scales.extend(parse_scale_definitions(&user_content));
    }
    scales
}

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
    for chunk in lines.chunks(2) {
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
        let semitone = match part {
            "1" => 0, "b2" => 1, "2" => 2, "b3" | "#2" => 3, "3" => 4, "4" => 5,
            "b5" | "#4" => 6, "5" => 7, "b6" | "#5" => 8, "6" | "bb7" => 9,
            "b7" | "#6" => 10, "7" => 11, _ => continue,
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
        _ => ChordQuality::Dominant7,
    }
}
