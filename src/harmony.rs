//! Where each chord of a tune stands and where it leads.
//!
//! Two things, kept apart. The degree: where a chord stands against the
//! tune's tonic, `♭VII` or `iiø`. The lead: where a chord goes when that is
//! not home - `V7/IV` for a C7 going to F in C minor, `ii/♭III` for the Fm7
//! before the B♭7 that goes to E♭. A chord is read against the chords around
//! it, so a whole tune is analysed at once.
//!
//! The tune is read as the strip plays it, round and round: the chord after
//! the last is the first.

use crate::model::{Chord, ChordQuality};

/// A chord's place against the tonic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Degree {
    /// Semitones above the tonic, 0-11.
    pub semitones: usize,
    pub kind: Kind,
}

/// What a degree's numeral shows of the chord: its third, by the case, and a
/// diminished fifth by a sign after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Major,
    Minor,
    HalfDiminished,
    Diminished,
}

/// Where a chord leads: the kind of approach and the degree it approaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lead {
    pub approach: Approach,
    pub target: Degree,
    /// The target's root, the pitch class: how the strip names it.
    pub target_root: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Approach {
    /// A dominant seventh a fifth above its target.
    Dominant,
    /// A dominant seventh a semitone above its target.
    TritoneSub,
    /// The minor seventh, or m7♭5, before a dominant: the ii of its target.
    Two { half_diminished: bool },
}

/// One chord of a tune, analysed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Analysis {
    pub degree: Degree,
    /// `None` for a chord that leads nowhere but home, or nowhere at all.
    pub lead: Option<Lead>,
}

/// Every chord of `chords`, in the tune whose tonic is the pitch class `tonic`.
pub fn analyse(chords: &[Chord], tonic: usize) -> Vec<Analysis> {
    let tune = Tune { chords, tonic: tonic % 12 };
    (0..chords.len())
        .map(|i| Analysis { degree: tune.degree_of(&chords[i]), lead: tune.lead_of(i) })
        .collect()
}

struct Tune<'a> {
    chords: &'a [Chord],
    tonic: usize,
}

impl Tune<'_> {
    fn degree_of(&self, chord: &Chord) -> Degree {
        Degree { semitones: (chord.root as usize + 12 - self.tonic) % 12, kind: kind_of(&chord.quality) }
    }

    fn lead_of(&self, i: usize) -> Option<Lead> {
        let chord = &self.chords[i];
        let lead = match chord.quality {
            ChordQuality::Dominant7 if !self.is_home(chord) => self.dominant_lead(i),
            ChordQuality::Minor7 | ChordQuality::HalfDiminished => self.two_lead(i),
            _ => None,
        }?;
        // Going home is what the degree says already: G7 in C is V. The
        // tritone substitute is the exception - ♭II alone does not say it leads.
        let home = lead.target.semitones == 0;
        (!home || lead.approach == Approach::TritoneSub).then_some(lead)
    }

    /// A chord on the tonic is home unless it is a dominant in a tune whose home
    /// is not one: C7 in a C blues is I7, in C minor it leads to F.
    fn is_home(&self, chord: &Chord) -> bool {
        if chord.root as usize != self.tonic {
            return false;
        }
        let blues = self
            .chords
            .iter()
            .filter(|c| c.root as usize == self.tonic)
            .all(|c| c.quality == ChordQuality::Dominant7);
        blues || chord.quality != ChordQuality::Dominant7
    }

    /// A dominant seventh leads where the next chord resolves it, a fifth down
    /// or a semitone down. Unresolved, it still leads where its ii points: the
    /// ii-V names its target even when another chord comes instead - unless
    /// what comes is a whole tone up, the backdoor: Fm7 B♭7 to C is iv ♭VII I,
    /// and the degrees say it.
    fn dominant_lead(&self, i: usize) -> Option<Lead> {
        let root = self.chords[i].root as usize;
        if let Some(next) = self.next_other(i) {
            match (next.root as usize + 12 - root) % 12 {
                5 => return Some(self.lead_to(Approach::Dominant, next)),
                // D7 Db7 C: the D7 is not a substitute for the Db7's V.
                11 if next.quality != ChordQuality::Dominant7 => {
                    return Some(self.lead_to(Approach::TritoneSub, next))
                }
                2 => return None,
                _ => {}
            }
        }
        let before = self.previous_other(i)?;
        let two = (root + 12 - before.root as usize) % 12 == 5
            && matches!(before.quality, ChordQuality::Minor7 | ChordQuality::HalfDiminished);
        let target_root = (root + 5) % 12;
        two.then(|| Lead {
            approach: Approach::Dominant,
            target: self.implied_target(before, target_root),
            target_root,
        })
    }

    /// A minor seventh, or m7♭5, before a dominant a fourth up - or its tritone
    /// substitute a semitone down - is the ii of where that dominant leads, a
    /// whole tone below the ii. A dominant that leads nowhere, or is home,
    /// leaves the ii a plain degree.
    fn two_lead(&self, i: usize) -> Option<Lead> {
        let two = &self.chords[i];
        let root = two.root as usize;
        let next_index = self.next_other_index(i)?;
        let dominant = &self.chords[next_index];
        let step = (dominant.root as usize + 12 - root) % 12;
        if dominant.quality != ChordQuality::Dominant7 || !(step == 5 || step == 11) || self.is_home(dominant) {
            return None;
        }
        let lead = self.dominant_lead(next_index)?;
        let a_tone_below = (root + 10) % 12;
        (lead.target.semitones == (a_tone_below + 12 - self.tonic) % 12).then(|| Lead {
            approach: Approach::Two { half_diminished: two.quality == ChordQuality::HalfDiminished },
            ..lead
        })
    }

    fn lead_to(&self, approach: Approach, target: &Chord) -> Lead {
        Lead { approach, target: self.degree_of(target), target_root: target.root as usize }
    }

    fn implied_target(&self, two: &Chord, target: usize) -> Degree {
        let kind = if two.quality == ChordQuality::HalfDiminished { Kind::Minor } else { Kind::Major };
        Degree { semitones: (target + 12 - self.tonic) % 12, kind }
    }

    /// The first chord after `i` that is not the same chord again: C7 C7 F
    /// leads from both C7s to the F.
    fn next_other_index(&self, i: usize) -> Option<usize> {
        let n = self.chords.len();
        (1..n).map(|k| (i + k) % n).find(|&j| !same(&self.chords[j], &self.chords[i]))
    }

    fn next_other(&self, i: usize) -> Option<&Chord> {
        self.next_other_index(i).map(|j| &self.chords[j])
    }

    fn previous_other(&self, i: usize) -> Option<&Chord> {
        let n = self.chords.len();
        (1..n).map(|k| &self.chords[(i + n - k) % n]).find(|c| !same(c, &self.chords[i]))
    }
}

fn same(a: &Chord, b: &Chord) -> bool {
    a.root == b.root && a.quality == b.quality
}

fn kind_of(quality: &ChordQuality) -> Kind {
    match quality {
        ChordQuality::Major7 | ChordQuality::Dominant7 | ChordQuality::CustomScale(_) => Kind::Major,
        ChordQuality::Minor7 | ChordQuality::Minor6 | ChordQuality::MinorMajor7 => Kind::Minor,
        ChordQuality::HalfDiminished => Kind::HalfDiminished,
        ChordQuality::Diminished => Kind::Diminished,
    }
}

impl Degree {
    /// The numeral as a progression is written: counted on the major scale, so
    /// in G minor an F7 is `♭VII` and an F♯ dim `vii°`; upper case over a major
    /// third, lower over a minor one; `ø` and `°` for a diminished fifth.
    pub fn written(&self) -> String {
        let sign = match self.kind {
            Kind::HalfDiminished => "ø",
            Kind::Diminished => "°",
            _ => "",
        };
        format!("{}{sign}", self.numeral())
    }

    /// The numeral alone, without the sign of a diminished fifth: how a target
    /// is written after the slash.
    pub fn numeral(&self) -> String {
        const FLATS: [&str; 12] = ["I", "♭II", "II", "♭III", "III", "IV", "♭V", "V", "♭VI", "VI", "♭VII", "VII"];
        // A diminished chord between two degrees passes up from the lower one:
        // F♯ dim in F is ♯I°, F♯ m7♭5 in C ♯IVø.
        let numeral = match (self.kind, self.semitones) {
            (Kind::HalfDiminished | Kind::Diminished, 1) => "♯I",
            (Kind::HalfDiminished | Kind::Diminished, 6) => "♯IV",
            (Kind::Diminished, 8) => "♯V",
            (_, s) => FLATS[s],
        };
        match self.kind {
            Kind::Major => numeral.to_string(),
            _ => numeral.to_lowercase(),
        }
    }
}

impl Approach {
    pub fn written(self) -> &'static str {
        match self {
            Approach::Dominant => "V7",
            Approach::TritoneSub => "subV7",
            Approach::Two { half_diminished: true } => "iiø",
            Approach::Two { half_diminished: false } => "ii",
        }
    }
}

impl Lead {
    /// What the chord is and the degree it goes to, with that degree's chord
    /// named after it: `V7 → ♭III (B♭)`, `iiø → i (Gm)`. The degree alone,
    /// `♭VII` over `V7/♭III`, had to be worked out back into a chord; the name
    /// alone left the target's place in the tune out.
    pub fn written(&self) -> String {
        let root = crate::tab::spell(&crate::model::NoteName::from_index(self.target_root).to_string());
        let kind = match self.target.kind {
            Kind::Major => "",
            Kind::Minor => "m",
            Kind::HalfDiminished => "ø",
            Kind::Diminished => "°",
        };
        format!("{} → {} ({root}{kind})", self.approach.written(), self.target.numeral())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::parse_chords_line_pub;

    /// Degree and lead of each chord, `degree:lead`, the lead left out where
    /// there is none.
    fn read(line: &str, tonic: &str) -> String {
        let tonic = parse_chords_line_pub(&format!("{tonic}Maj7"))[0].root as usize;
        analyse(&parse_chords_line_pub(line), tonic)
            .iter()
            .map(|a| match a.lead {
                Some(lead) => format!("{}:{}", a.degree.written(), relative(&lead)),
                None => a.degree.written(),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// A lead as the analysis reads it, its target counted from the tonic:
    /// `V7/IV`, or `subV7` alone going home.
    fn relative(lead: &Lead) -> String {
        match lead.target.semitones {
            0 => lead.approach.written().to_string(),
            _ => format!("{}/{}", lead.approach.written(), lead.target.numeral()),
        }
    }

    /// The strip writes the target's degree and names its chord: in G minor
    /// the F7 is ♭VII going to ♭III, B♭.
    #[test]
    fn a_lead_names_its_target_chord() {
        let leads: Vec<String> = analyse(&parse_chords_line_pub("Cm7 F7 BbMaj7 Ab7 Gm7"), 7)
            .iter()
            .filter_map(|a| a.lead.map(|lead| lead.written()))
            .collect();
        assert_eq!(leads, ["ii → ♭III (B♭)", "V7 → ♭III (B♭)", "subV7 → i (Gm)"]);
    }

    /// Bars 3 to 15 of Solar: every ii-V goes somewhere else, and the C7
    /// on the tonic of C minor is the V of F, not a I.
    #[test]
    fn solar_leads_away_from_home() {
        assert_eq!(
            read("CmMaj7 Gm7 C7 FMaj7 Fm7 Bb7 EbMaj7 Ebm7 Ab7 DbMaj7 Dm7b5 G7", "C"),
            "i v:ii/IV I:V7/IV IV iv:ii/♭III ♭VII:V7/♭III ♭III ♭iii:ii/♭II ♭VI:V7/♭II ♭II iiø V"
        );
    }

    /// In a blues the dominant on the tonic is home: F7 going to B♭7 is I.
    #[test]
    fn a_blues_stays_home_on_its_seventh() {
        assert_eq!(read("F7 Bb7 F7 C7 Bb7 F7 C7", "F"), "I IV I V IV I V");
    }

    /// The tritone substitute leads a semitone down, and is named even going
    /// home, where ♭II alone would not say it.
    #[test]
    fn a_tritone_substitute_resolves_a_semitone_down() {
        assert_eq!(read("Dm7 Db7 CMaj7", "C"), "ii ♭II:subV7 I");
        assert_eq!(read("Ab7 GMaj7 CMaj7", "C"), "♭VI:subV7/V V I");
        // A semitone down to another dominant is not a substitute's resolution.
        assert_eq!(read("D7 Db7 CMaj7", "C"), "II ♭II:subV7 I");
    }

    /// Each dominant of a chain leads to the nearest chord, not further down.
    #[test]
    fn a_chain_of_dominants_names_the_nearest_target() {
        assert_eq!(read("A7 D7 G7 CMaj7", "C"), "VI:V7/II II:V7/V V I");
    }

    /// A dominant held for two bars leads from both: the next chord that is
    /// not itself again is the one it resolves to.
    #[test]
    fn a_repeated_dominant_leads_from_every_bar() {
        assert_eq!(read("CMaj7 C7 C7 FMaj7 FMaj7 G7", "C"), "I I:V7/IV I:V7/IV IV IV V");
    }

    /// A ii-V that does not resolve still names the target it implies, and
    /// a dominant with neither its resolution nor its ii names none.
    #[test]
    fn an_unresolved_ii_v_names_its_implied_target() {
        assert_eq!(read("Em7 A7 Em7 A7 D7 G7 CMaj7", "C"), "iii:ii/II VI:V7/II iii:ii/II VI:V7/II II:V7/V V I");
        assert_eq!(read("CMaj7 Bb7 CMaj7", "C"), "I ♭VII I");
    }

    /// The backdoor goes home a whole tone up: iv and ♭VII lead nowhere else.
    #[test]
    fn the_backdoor_is_its_degrees() {
        assert_eq!(read("CMaj7 Fm7 Bb7 CMaj7", "C"), "I iv ♭VII I");
    }

    /// The minor ii-V is iiø and its target lower case; in Autumn Leaves the
    /// relative major's ii-V keeps its degrees in G minor beside the lead.
    #[test]
    fn minor_and_relative_major_ii_v() {
        assert_eq!(read("Am7b5 D7 Gm7", "G"), "iiø V i");
        assert_eq!(read("Cm7 F7 BbMaj7 EbMaj7 Am7b5 D7 Gm7", "G"), "iv:ii/♭III ♭VII:V7/♭III ♭III ♭VI iiø V i");
        assert_eq!(read("Em7b5 A7 Dm7", "C"), "iiiø:iiø/ii VI:V7/ii ii");
    }

    /// The numerals tell apart what the degree alone used to merge: in F the
    /// B7 is ♭V, not V, and F♯ dim passes up from the I.
    #[test]
    fn every_degree_has_its_own_numeral() {
        assert_eq!(read("FMaj7 F#dim Gm7 C7", "F"), "I ♯i° ii V");
        assert_eq!(read("Gb7 FMaj7 B7 C7 FMaj7", "F"), "♭II:subV7 I ♭V V I");
        assert_eq!(read("F#m7b5 Fm7", "C"), "♯ivø iv");
    }
}

