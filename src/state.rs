use std::sync::{Arc, Mutex};
use std::collections::VecDeque; 
use crate::audio::AudioAnalysis;
use crate::arpeggio;
use crate::interval_levels::{self, Level};
use crate::rng::Rng;
use crate::fretboard::Region;
use crate::tab;
use crate::model::{Step, split_octave, Chord, NoteName, Song, load_songs, load_all_scale_definitions, load_arpeggio_patterns, ScaleDefinition, ChordQuality};

mod chords_mode;
mod credits;
mod formulas_mode;
mod fretboard_mode;
mod heard;
mod judge;
mod notes_mode;
pub use credits::Credits;
#[cfg(test)]
use chords_mode::{chord_reading, ChordReading};
#[cfg(test)]
use credits::Credit;
pub use heard::Heard;

/// Where one step of an exercise stands, as `note_state` reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteState {
    /// Later in the order; not asked for yet.
    Waiting,
    /// The one the exercise waits for now.
    Due,
    /// Free order: any step not yet played may come next.
    Open,
    /// Heard, and being held long enough to count.
    Answering,
    /// Credited.
    Done,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum AppMode { 
    Chords = 0, 
    Intervals = 1, 
    Scales = 2,
    Arpeggios = 3,
    /// Fretboard trainer: one fixed region of the neck, random notes inside it.
    /// Has no song and no chords - see `fretboard`.
    Fretboard = 4,
    /// Interval formulas: a drawn set of functions over a root, played in any
    /// order. No song and no chords either - see `formulas`.
    Formulas = 5,
}

impl From<i32> for AppMode {
    fn from(val: i32) -> Self {
        match val {
            4 => AppMode::Fretboard,
            5 => AppMode::Formulas,
            0 => AppMode::Chords,
            1 => AppMode::Intervals,
            2 => AppMode::Scales,
            3 => AppMode::Arpeggios,
            _ => AppMode::Intervals,
        }
    }
}

/// What the arpeggio mode practises. Saved as its index: 0 and 1.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum ArpExercise {
    /// A written phrase - a study - in a key of its own.
    #[default]
    Study,
    /// An arpeggio built for each chord of a standard, along its changes.
    OverChanges,
}

impl From<usize> for ArpExercise {
    fn from(index: usize) -> Self {
        match index {
            1 => ArpExercise::OverChanges,
            _ => ArpExercise::Study,
        }
    }
}

/// Which way an arpeggio over the changes runs. Saved as its index, 0 to 3.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum ArpDirection {
    #[default]
    Up,
    Down,
    /// Down over the first chord, up over the next, and so on.
    AlternateDownFirst,
    AlternateUpFirst,
}

impl From<usize> for ArpDirection {
    fn from(index: usize) -> Self {
        match index {
            0 => ArpDirection::Up,
            1 => ArpDirection::Down,
            2 => ArpDirection::AlternateDownFirst,
            _ => ArpDirection::AlternateUpFirst,
        }
    }
}

/// Which way a scale runs. Saved as its index, 0 to 2; the 3 an older
/// version saved for a scattered scale reads as up.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum ScaleDirection {
    #[default]
    Up,
    Down,
    /// Drawn again for every pass.
    Random,
}

impl From<usize> for ScaleDirection {
    fn from(index: usize) -> Self {
        match index {
            1 => ScaleDirection::Down,
            2 => ScaleDirection::Random,
            _ => ScaleDirection::Up,
        }
    }
}

/// Which degree a scale starts on: the root, the third, fifth or seventh,
/// or one drawn for every pass. Saved as its index, 0 to 4.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum ScaleFrom {
    #[default]
    Root,
    Third,
    Fifth,
    Seventh,
    Random,
}

impl From<usize> for ScaleFrom {
    fn from(index: usize) -> Self {
        match index {
            1 => ScaleFrom::Third,
            2 => ScaleFrom::Fifth,
            3 => ScaleFrom::Seventh,
            4 => ScaleFrom::Random,
            _ => ScaleFrom::Root,
        }
    }
}

/// Where the key goes after a finished scale, when it is not drawn at random.
/// Saved as its index, 0 to 4.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum KeyMove {
    #[default]
    Stays,
    SemitoneUp,
    SemitoneDown,
    /// C G D A ...
    CycleOfFifths,
    /// C F Bb Eb ...
    CycleOfFourths,
}

impl From<usize> for KeyMove {
    fn from(index: usize) -> Self {
        match index {
            1 => KeyMove::SemitoneUp,
            2 => KeyMove::SemitoneDown,
            3 => KeyMove::CycleOfFifths,
            4 => KeyMove::CycleOfFourths,
            _ => KeyMove::Stays,
        }
    }
}

impl KeyMove {
    /// Semitones up to the next key.
    fn step(self) -> usize {
        match self {
            KeyMove::Stays => 0,
            KeyMove::SemitoneUp => 1,
            KeyMove::SemitoneDown => 11,
            KeyMove::CycleOfFifths => 7,
            KeyMove::CycleOfFourths => 5,
        }
    }
}

/// What a formula is played against. Saved as its index, 0 to 2.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum FormulaExercise {
    /// A key, and nothing under it.
    #[default]
    InKey,
    /// A chord drawn at random, the formula placed on one of its degrees.
    OverChord,
    /// The chords of a standard, one after another.
    OverChanges,
}

impl FormulaExercise {
    /// Whether there is a chord under the formula at all.
    pub fn has_chord(self) -> bool {
        self != FormulaExercise::InKey
    }
}

impl From<usize> for FormulaExercise {
    fn from(index: usize) -> Self {
        match index {
            1 => FormulaExercise::OverChord,
            2 => FormulaExercise::OverChanges,
            _ => FormulaExercise::InKey,
        }
    }
}

/// How the formula is to sit against the chord under it. Saved as its index,
/// 0 to 3.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum FormulaPlacement {
    #[default]
    Any,
    Defines,
    Colours,
    Outside,
}

impl FormulaPlacement {
    /// The verdict a drawn placement has to come out with, if any.
    pub fn wanted(self) -> Option<crate::formulas::Verdict> {
        match self {
            FormulaPlacement::Any => None,
            FormulaPlacement::Defines => Some(crate::formulas::Verdict::Defines),
            FormulaPlacement::Colours => Some(crate::formulas::Verdict::Colours),
            FormulaPlacement::Outside => Some(crate::formulas::Verdict::Outside),
        }
    }
}

impl From<usize> for FormulaPlacement {
    fn from(index: usize) -> Self {
        match index {
            1 => FormulaPlacement::Defines,
            2 => FormulaPlacement::Colours,
            3 => FormulaPlacement::Outside,
            _ => FormulaPlacement::Any,
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum MatchStatus {
    None,       // white
    Exact,      // green - exact match
    Partial,    // yellow - triad or jazz substitution
    Flicker,    // red - detected but the signal is weak
}

/// Strings to suggest starting from, low to high. Only a hint: the model
/// reports 12 pitch classes with no position, so the app cannot check which
/// string was actually used.
///
/// Limited to the strings the chord shapes are drawn from. Suggesting the G
/// string sent the player looking for a shape the app does not have, which
/// reads as the diagrams being wrong rather than the hint. `diagrams.rs` has a
/// test tying the two together.
pub const START_STRINGS: [&str; 3] = ["E", "A", "D"];


/// How long a new lap ignores what it hears, so the last one's decay cannot
/// walk into it. A quarter of a second: past the moment the marks clear, short
/// enough to be over before anyone has played the next note.
const LAP_HOLD: f32 = 0.25;

/// Hops after the strike detector reports a class within which that class
/// counts as struck for "credit only what was struck": about 0.35 s.
///
/// On AtoA it took the false credits of the model's branches from 57 to none,
/// with no note missed; anything from 6 to 50 hops did the same. A strum pays
/// for it: the detector reports one class per strum. Measurements:
/// `docs/judge-measurements.md#struck_fresh`.
const STRUCK_FRESH: u32 = 22;

/// Hops after the strike detector reports a strike during which the ear may
/// credit only a class the detector reported struck: about 0.25 s.
///
/// Just after an attack the half-second FFT window holds the old note and the
/// new one together, and the estimate wanders. On AtoA this halved the ear's
/// false credits, 124 to 62, with no note missed. Measurements:
/// `docs/judge-measurements.md#ear_settle`.
const EAR_SETTLE: u32 = 16;

/// In Formulas, hops back the strike detector must have reported a class for
/// the ear to credit it at all: 3 s.
///
/// A dying note can make the estimate settle on another class, and a mark in
/// Formulas never expires. On AtoA: false credits 62 to 3. Kept out of the other
/// modes because there it loses quiet and legato notes. Measurements:
/// `docs/judge-measurements.md#ear_backed`.
const EAR_BACKED: u32 = 188;

/// Hops within which an attack still counts as happening now: about 0.2 s.
///
/// Long enough to cover the ear settling on the note that was struck, short
/// enough that the attack belonging to the previous chord has expired by the
/// time this one is shown - `INTERVAL_SHOW` alone is 0.35 s.
const FLUX_FRESH: u32 = 12;

/// Hops within which a strike of a class still counts as just now: 0.6 s.
///
/// Longer than `FLUX_FRESH` because the strike is reported at the attack and the
/// credit waits for the ear to settle on the note, which takes up to 0.4 s.
/// Asking for a flux attack as well was measured and dropped:
/// `docs/judge-measurements.md#class_fresh`.
const CLASS_FRESH: u32 = 38;

/// A class struck this recently when the exercise moves on was played for the
/// next part, not left over from the last: the 0.35 s the finished set is shown
/// for, in hops.
const JUST_STRUCK: u32 = 22;

/// Frames a class has to be absent from `voices` before it counts as having
/// stopped: about 0.3 s.
///
/// Longer than the gaps the list leaves in a note that is still ringing, and
/// shorter than the pause between one chord and the next.
const GONE_QUIET: u32 = 19;

/// How long after a credit its note cannot yet count as having left - see
/// `Credit::left`. Set when a late attack answer had to be kept with the credit
/// it belonged to, and kept because `left` is measured with it.
const CREDIT_SETTLE: f32 = 0.5;

/// How long the fretboard trainer keeps a note it has just credited on screen,
/// green, before drawing the next one.
///
/// Without it the trainer moved on in the same breath as it passed the note:
/// `next_fret_target` clears the verdict, so a step that passed inside one
/// answer from the model went green and back in a single call and nothing was
/// ever drawn green. The model answers every 40 ms at best, and `dt` is capped
/// at 250 ms, so at 0.12 s to pass the whole verdict could live and die between
/// two frames of the window.
///
/// A floor, not a fixed time: the hold setting may be longer, and 0 - which it
/// can be set to - would put the flash back.
const FRET_SHOW: f32 = 0.35;

/// How long a single note has to answer before it is credited. A note is right
/// the moment it is recognised, so this is much shorter than the chord hold.
const NOTE_HOLD: f32 = 0.12;

/// Leave the completed interval set visible, including its last mark.
const INTERVAL_SHOW: f32 = 0.35;
/// A stopped inference stream must not keep answering new audio frames.
const INTERVAL_MODEL_AGE: f32 = 0.25;

/// What the studies can be read over, in the order the settings list them.
pub const ARP_QUALITIES: [(&str, ChordQuality); 5] = [
    ("m7", ChordQuality::Minor7),
    ("Maj7", ChordQuality::Major7),
    ("7", ChordQuality::Dominant7),
    ("m7b5", ChordQuality::HalfDiminished),
    ("dim7", ChordQuality::Diminished),
];

/// How steady the single-frame estimate has to be: this many of the last
/// `EAR_WINDOW` ticks naming the class.
///
/// A vote, not a run. A run broke on any single stray frame and had to start
/// over, and at half a second to the note - which is not fast playing - the
/// estimate flickers between the note arriving and the one still ringing, so
/// the run was rarely reached before the next note began. Four of five
/// forgives one stray frame and still refuses a reading that cannot make up its
/// mind: an estimate alternating between two classes reaches two, never three.
const EAR_VOTES: usize = 4;
const EAR_WINDOW: usize = 5;

/// How long the single-frame estimate has to have been naming a note before it
/// is credited, in audio frames of 16 ms - and how long a DIFFERENT note has to
/// have been named before the one credited counts as having stopped sounding.
///
/// Two frames credit about five points more of what is played than three, at any
/// level, and still name almost nothing wrong; a vote over a window misnames one
/// frame in ten. Measurements: `docs/judge-measurements.md#credit_ticks`.
const CREDIT_TICKS: u32 = 2;
const LEFT_TICKS: u32 = 3;

/// The attack counters as they stood when the current chord became the target.
#[derive(Clone, Copy, Default, Debug)]
struct ChordHeard {
    /// The envelope's.
    onset_id: u64,
    /// The strike detector's, per class.
    class_strikes: [u32; 12],
}


/// Ticks the tests feed before expecting the vote to carry.
#[cfg(test)]
pub(crate) const CQT_STEADY_TICKS: u32 = EAR_WINDOW as u32;

/// How long a finished formula stays on screen before the next lap, whatever
/// the hold time is set to. Elsewhere that setting says how long a target has
/// to be held to pass; here the whole set has already been played, and the
/// screen owes the player the sight of the last function going green.
pub const FORMULA_LAP_PAUSE: f32 = 0.6;

pub struct MyApp {
    /// What the audio thread has reported, copied in once per frame.
    pub heard: Heard,
    pub analysis_state: Arc<Mutex<AudioAnalysis>>,
    
    pub song_library: Vec<Song>,
    pub scale_definitions: Vec<ScaleDefinition>,
    pub arpeggio_patterns: Vec<ScaleDefinition>,
    
    pub app_mode: AppMode,
    pub selected_library_idx: usize,
    pub secondary_index: usize, 
    /// Which arpeggio pattern was chosen, kept across mode switches. Leaving the
    /// mode used to throw the choice away and drop you back on the first pattern.
    saved_arpeggio_index: usize,
    /// And the key it stood in - see `set_mode`.
    saved_arpeggio_key: usize,
    /// Which tune was chosen, kept across mode switches and saved between
    /// sessions. A standard is worked on for weeks and read from four modes -
    /// the chords, the intervals in them, an arpeggio over each and a formula
    /// planted on each - and every one of those used to open on the first row
    /// of the library.
    saved_song_idx: usize,
    /// And the same for the scale being worked on: Scales is its own library,
    /// and the mode used to open on the first row of it however long you had
    /// been on one scale.
    saved_scale_idx: usize,
    
    pub song_title: String,
    pub chords: Vec<Chord>,
    pub current_chord_index: usize,
    
    pub current_note_step: usize,
    pub collected_notes: Vec<bool>,

    pub success_timer: f32,
    pub transition_delay: f32,
    /// Confidence threshold for the chord NAME (Chords mode). Confidence is the
    /// product of root and quality confidence, typically 0.85-0.95 when played well.
    pub chord_confidence: f32,
    /// Probability threshold for a SINGLE NOTE (Intervals/Scales/Arpeggios). A
    /// separate knob because it is a different quantity from chord-name confidence;
    /// they used to share one slider and half its range was dead.
    pub note_threshold: f32,
    
    pub match_status: MatchStatus,
    /// Judge a strum on its best reading instead of on how long it is held.
    ///
    /// A chord struck and released - a chick - is right or wrong within a few
    /// frames of the attack, and then it simply stops ringing. The hold timer
    /// cannot tell that from a wrong chord: both stop feeding it, and it drains.
    /// With this on, one reading of the target above the confidence threshold
    /// between two attacks is enough, and no later frame in the same strum can
    /// take it back. Wrong chords still fail, because their best reading after
    /// the attack is a different chord.
    pub short_verdict: bool,
    /// The panel is drawing shell voicings and nothing else. Shells leave out
    /// the fifth, so a chord told apart from another only by its fifth cannot
    /// be asked for in full - see the m7b5 arm of the match below.
    pub shells_only: bool,
    /// The chord shown before this one, so the strip along the bottom can show
    /// what was just played. `None` until something has been.
    pub prev_chord_index: Option<usize>,
    /// HOW that previous chord was passed, or `None` if it was stepped over.
    ///
    /// Feedback for a pass cannot go on the current chord: passing changes which
    /// chord that is, so lighting it marks the one not played yet - which is
    /// what it did, confusingly. It belongs on the chord that earned it, and it
    /// can stay there: the next pass moves it along, so there is nothing to time
    /// out. The status and not just a flag, because a chord passed on a triad or
    /// a substitution went yellow on the way through and saying green afterwards
    /// would be a nicer report than what happened.
    prev_status: MatchStatus,
    /// The same for every position of this pass through the order: played,
    /// the status it was passed on; stepped over with the arrows, or not
    /// reached yet, `None`. The strip colours the chords further behind from
    /// this - it used to paint them all green, so stepping forward while
    /// paused lit chords nobody had played.
    pass_results: Vec<MatchStatus>,
    
    pub bass_boost_enabled: bool,
    pub bass_boost_gain: f32,
    pub noise_gate: f32,
    
    /// Scales only: end the run on the root again, an octave up - 1 2 3 4 5 6 7 1.
    pub scale_repeat_root: bool,
    pub intervals_input: String,
    pub saved_intervals_input: String,
    /// The exercise of Intervals, an index into `interval_levels::LEVELS`.
    /// `None` reads `intervals_input` instead, as tests written against a
    /// typed-in set of degrees do.
    pub interval_level: Option<usize>,
    /// Where a rotated level starts over the chord now due, drawn per chord.
    pub level_rotation: usize,
    /// Intervals with the shuffle on: start each chord near the last note
    /// played - see `lead_from_last_note`.
    pub lead_line: bool,
    /// The pitch class of the step credited last in Intervals.
    pub last_played: Option<usize>,
    /// Chords: semitones up the tune is played in, 0 to 11. Back to 0 with
    /// every new tune.
    pub transpose: usize,
    
    /// Randomise the exercise order. What it means depends on the mode: in
    /// Chords it picks the next chord at random, in the note modes it also
    /// shuffles the order of the tones within the chord - which is where the
    /// melodies come from.
    /// Freezes progression. Colours keep reporting whether the chord is right,
    /// the app just stops moving on - for sitting on one shape and working it
    /// out. Deliberately not persisted: a paused app on next launch would look
    /// broken.
    pub paused: bool,
    pub random_mode: bool,
    /// Shuffle the CHORD ORDER as well, in the modes where the two are separable.
    /// In Intervals and Arpeggios the written progression is a tune, and hearing
    /// shuffled intervals walk through it is musical in a way that shuffling
    /// everything is not. In Chords there is nothing else for the switch to
    /// mean, so there it stays on.
    pub shuffle_chords: bool,
    /// Arpeggios: 0 the studies, played in a key of their own with no tune
    /// under them; 1 over the changes, an arpeggio per chord of the song.
    ///
    /// The two are different exercises and were one before: a long study
    /// walked over a progression restarts mid-phrase at every chord, which is
    /// neither the study nor the changes.
    pub arp_exercise: ArpExercise,
    /// Over the changes: which way each chord's arpeggio runs. 0 up, 1 down,
    /// 2 alternating from down, 3 alternating from up - the four the studies
    /// are written in.
    pub arp_direction: ArpDirection,
    /// The studies: which chord they are read over. An index into
    /// `ARP_QUALITIES`.
    pub arp_quality: usize,
    /// Permutation of `active_indices`, regenerated whenever the chord changes.
    /// Reshuffling every frame would make the target jump around; the order has
    /// to stay fixed for as long as the chord is being played.
    step_order: Vec<usize>,
    /// See `reroll`: the fret the first grip of an Intervals exercise is taken
    /// near.
    pub voicing_anchor: i32,
    /// And the string a scale is started from - an index into `START_STRINGS`,
    /// the three a scale is worth starting on.
    pub scale_start: usize,
    /// How many passes have gone by, which is what the scale's string is
    /// counted through. See `next_scale_pass`.
    pass: usize,
    /// The string an interval grip is taken from when the voices are not being
    /// led from the chord before. Drawn per chord - see `reroll`.
    pub grip_string: usize,
    /// Whether this pass runs downwards, where the direction is drawn - see
    /// `ScaleDirection::Random`. Drawn with each pass, not with each redraw.
    pub scale_descending: bool,
    /// Scales: which way, from which degree, whether back down again, and
    /// where the key goes next.
    pub scale_direction: ScaleDirection,
    pub scale_from: ScaleFrom,
    pub scale_there_and_back: bool,
    pub key_move: KeyMove,
    /// Scales: every tone from the low E string to the high one, two octaves
    /// in first position, in place of a run from `scale_from`. On by default.
    pub scale_e_to_e: bool,
    /// Scales: the tones in the order shown. Off, in any order.
    pub scale_in_order: bool,
    /// The degree a scale starts on this pass where `ScaleFrom::Random`.
    scale_from_drawn: usize,
    /// Which of its places on the neck a scale from E to e starts on this
    /// pass - see `scale_hand`.
    scale_place_drawn: usize,
    /// How many passes in a row the drawn direction has run the same way.
    same_way_passes: usize,
    /// Which string to suggest starting from (index into `START_STRINGS`), or
    /// None for no hint. The app CANNOT verify it - the model outputs 12 pitch
    /// classes with no position or octave - so this is a suggestion the player
    /// checks themselves.
    pub start_hint: Option<usize>,
    /// The order the chords are played in: straight through in sequence, a
    /// shuffled permutation while random order is on.
    ///
    /// Shuffled once when the toggle goes on rather than drawn afresh at every
    /// step, because the strip along the bottom shows what is coming - and with
    /// a coin toss at each step there is nothing to show. Also means stepping
    /// back while paused lands on the chord actually played before.
    play_order: Vec<usize>,
    /// Position within `play_order`; `current_chord_index` is what it points at.
    play_pos: usize,
    /// Fretboard trainer: the region stays fixed for the whole session, only
    /// the note changes. Settling into one hand position is the point.
    pub region: Region,
    /// Pitch class currently asked for, or None before the first draw.
    pub fret_target: Option<usize>,
    /// The notes asked for lately, so the next one is not one of them. See
    /// `next_fret_target`.
    recent_targets: Vec<usize>,
    /// Seconds the answered note stays on screen before the next is drawn. See
    /// `FRET_SHOW`.
    fret_hold: f32,
    rng: Rng,
    pub chord_history: VecDeque<(String, f32)>,
    /// Probabilities of the 12 pitch classes from the last window. The note modes
    /// rely on this rather than on the chord name - the pitch head is at F1 0.90,
    /// the chord name around 80%.
    pub last_pitches: [f32; 12],
    /// The pitch vector of the PREVIOUS prediction. A window holding two notes
    /// credits the older one on level; the newer one only stands out by how much
    /// it rose, which is what this is for.
    pub prev_pitches: [f32; 12],
    /// "Credit only what was struck": the model's branches may credit a class only
    /// while it counts as struck just now - see `struck_just_now`.
    ///
    /// The pitch head answers what SOUNDS, which includes the previous note still
    /// inside its 0.77 s window and strings ringing in sympathy. Not free on every
    /// recording - 4 missed notes in 49 once - hence an option, off by default. See
    /// `docs/judge-measurements.md#require_onset`.
    pub require_onset: bool,
    /// Judge formulas the strict way - the steady single-frame estimate alone -
    /// instead of the rule the note modes use.
    ///
    /// On by default: the loose rule credits whatever is sounding, which after
    /// a few notes is most of the formula. `SOLITITO_FORMULA_STRICT=0` gives
    /// the note modes' rule back for a comparison.
    pub strict_formulas: bool,
    judged_frame: u64,
    /// The model's last word, kept so the frame clock can judge a formula
    /// without waiting for the next one. The model is asked only while the
    /// context window is nine tenths full, and playing one note at a time never
    /// fills it - so a rule that needs nothing from the model was still waiting
    /// on it, which is what made the mode feel slow.
    last_ai_root: Option<NoteName>,
    last_ai_conf: f32,
    /// Play the set in the order it is written, lowest function first.
    pub formula_in_order: bool,
    /// Intervals: the set has to be taken in the order it is dealt, lowest
    /// step first, instead of in any order. Off by default - a grip is three
    /// notes under one hand and which finger lands first is not the exercise.
    pub interval_in_order: bool,
    /// Which step the timer below is counting for. Under free order the answer
    /// can be any step still wanted, and a timer that did not know which one it
    /// belonged to would carry a count from one note into another.
    answering_step: Option<usize>,
    interval_hold: f32,
    interval_model_age: f32,
    interval_audio_age: f32,
    interval_silence: f32,
    /// Print a line for every function credited. The panel's own switch; the
    /// environment can force it either way while something is being chased.
    pub log_credits: bool,
    /// What that estimate has been repeating, and for how many ticks. One frame
    /// is 16 ms and has no memory, so it names a neighbour now and then; where
    /// a mark never expires, that one frame is enough to light a function.
    cqt_run_pitch: Option<usize>,
    cqt_run: u32,
    /// What the estimate named over the last `EAR_WINDOW` ticks.
    ear_window: [Option<usize>; EAR_WINDOW],
    /// Seconds left of the hold a new lap begins with. What was ringing when
    /// the last one ended is still ringing now, and would walk into this one.
    ///
    /// Timed, not tied to the next attack. Tied to the attack it blocked
    /// EVERYTHING until one was detected - and an attack is detected from a
    /// transient, which soft playing does not always give, so the first
    /// function of a new lap could take seconds.
    lap_hold: f32,
    /// Require the notes one at a time. Off, the CQT only ever ADDS a way to
    /// pass, and a strummed chord still walks its intervals - the model's pitch
    /// head is polyphonic and reports every tone at once, which a monophonic
    /// estimate cannot. On, the CQT overrules the model and one pluck credits
    /// one step.
    pub single_notes: bool,
    /// Where the exercise writes down what it asked and what it credited, when
    /// SOLITITO_RECORD is set. None otherwise, which is every normal run.
    journal: Option<crate::journal::Journal>,
    /// Which of the four ways credited this, where the call site knew.
    credit_branch: Option<u8>,
    /// The (chord, step) pair last written down, to notice a new request.
    logged_ask: Option<(usize, usize)>,
    /// For each pitch class, the attack it was last credited on: the envelope's
    /// attack count and that class's own strike count. A step asking for a
    /// class already credited is refused until it is struck again, so a note
    /// left ringing cannot pass twice.
    ///
    /// One entry per class, not one for the last credit: a scale reading
    /// 1 2 3 4 5 6 7 1 has six notes between the two roots, and remembering
    /// only the note before would have forgotten the first root by the time the
    /// last one is due.
    ///
    /// Kept across a chord and across a lap - the string goes on ringing over
    /// both - and cleared when the exercise itself is reset.
    pub credited: Credits,
    /// What the strike counters and the envelope's attack count read when the
    /// current chord became the target. A chord that repeats in the
    /// progression is still ringing from the pass before and would pass again
    /// untouched; comparing against this asks for a fresh strum.
    chord_heard_at: ChordHeard,

    // --- Formulas mode ---
    /// The formula being practised, as a bitmask of functions.
    pub formula_mask: u16,
    /// Pitch class of its root, 0 = C.
    /// Which of the three formula exercises is running: 0 the formula in a key
    /// of its own, 1 the same formula planted on one chord, 2 planted on every
    /// chord of a tune in turn. See `place_over_chord`.
    pub formula_exercise: FormulaExercise,
    /// What kind of placement to draw: 0 any, 1 defines, 2 colours, 3 outside.
    pub formula_placement_want: FormulaPlacement,
    /// The chord the formula is being played over, in exercises 1 and 2.
    pub formula_chord: Option<Chord>,
    /// Semitones above that chord's root where the formula's `1` sits.
    pub formula_degree: usize,
    /// How many of the chord's own tones the placement covers, and what that
    /// makes it. Both for display: the count is the lesson, the word is a label
    /// on the count.
    pub formula_hits: u32,
    pub formula_verdict: Option<crate::formulas::Verdict>,
    pub formula_root: usize,
    /// How that root is spelled, for the screen.
    pub formula_key_name: String,
    /// Which of the formula's functions have been sounded, in ascending order
    /// of function - NOT in the order they have to be played. A formula is a
    /// set; the point is to move around inside it freely.
    pub formula_collected: Vec<bool>,
    /// Options, mirrored from the settings each tick.
    pub formula_notes: usize,
    pub formula_required: u16,
    pub formula_random_key: bool,
    pub formula_key_setting: String,
}

impl MyApp {
    /// The model is NOT held here. It lives in the thread that asks it, and
    /// its answers arrive through `AiResult` - a handle on this side was never
    /// read, and holding one would invite calling it from the UI thread.
    pub fn new(state: Arc<Mutex<AudioAnalysis>>) -> Self {
        let song_library = load_songs();
        let scale_definitions = load_all_scale_definitions();
        let arpeggio_patterns = load_arpeggio_patterns();
        
        let start_song = if !song_library.is_empty() { 
            song_library[0].clone() 
        } else { 
            Song { title: "No Songs".into(), chords: vec![] } 
        };

        Self {
            analysis_state: state,
            heard: Heard::default(),
            song_library,
            scale_definitions,
            arpeggio_patterns,
            app_mode: AppMode::Fretboard,
            selected_library_idx: 0,
            secondary_index: 0,
            saved_arpeggio_index: 0,
            saved_arpeggio_key: 0,
            saved_song_idx: 0,
            saved_scale_idx: 0,
            
            song_title: start_song.title,
            chords: start_song.chords,
            current_chord_index: 0,
            
            current_note_step: 0,
            collected_notes: vec![],
            
            success_timer: 0.0,
            transition_delay: 0.25,
            chord_confidence: 0.30,
            note_threshold: 0.60,
            
            match_status: MatchStatus::None,
            short_verdict: false,
            shells_only: false,
            prev_chord_index: None,
            prev_status: MatchStatus::None,
            pass_results: Vec::new(),
            
            bass_boost_enabled: true,
            bass_boost_gain: 5.0,
            noise_gate: 0.02,
            
            scale_repeat_root: false,
            intervals_input: "1 3 5".to_string(),
            saved_intervals_input: "1 3 5".to_string(),
            interval_level: None,
            level_rotation: 0,
            lead_line: false,
            last_played: None,
            transpose: 0,
            
            paused: false,
            random_mode: false,
            shuffle_chords: false,
            arp_exercise: ArpExercise::Study,
            arp_direction: ArpDirection::Up,
            arp_quality: 0,
            step_order: vec![],
            voicing_anchor: 5,
            scale_start: 0,
            scale_descending: false,
            scale_direction: ScaleDirection::Up,
            scale_from: ScaleFrom::Root,
            scale_there_and_back: false,
            key_move: KeyMove::Stays,
            scale_e_to_e: true,
            scale_in_order: true,
            scale_from_drawn: 0,
            scale_place_drawn: 0,
            same_way_passes: 0,
            pass: 0,
            grip_string: 0,
            start_hint: None,
            play_order: Vec::new(),
            play_pos: 0,
            region: Region::default(),
            fret_target: None,
            recent_targets: Vec::new(),
            fret_hold: 0.0,
            rng: Rng::default(),
            chord_history: VecDeque::with_capacity(20),
            last_pitches: [0.0; 12],
            prev_pitches: [0.0; 12],
            require_onset: false,
            strict_formulas: std::env::var("SOLITITO_FORMULA_STRICT")
                .map(|v| v != "0")
                .unwrap_or(true),
            judged_frame: u64::MAX,
            last_ai_root: None,
            last_ai_conf: 0.0,
            formula_in_order: false,
            // As the shipped settings have it. Left false here, every test
            // sat in the relaxed path by accident.
            interval_in_order: true,
            answering_step: None,
            interval_hold: 0.0,
            interval_model_age: INTERVAL_MODEL_AGE,
            interval_audio_age: 0.0,
            interval_silence: 0.0,
            log_credits: false,
            cqt_run_pitch: None,
            cqt_run: 0,
            ear_window: [None; EAR_WINDOW],
            lap_hold: 0.0,
            single_notes: false,
            journal: crate::journal::Journal::open(),
            credit_branch: None,
            logged_ask: None,
            credited: Credits::default(),
            chord_heard_at: ChordHeard::default(),
            formula_mask: 0,
            formula_exercise: FormulaExercise::InKey,
            formula_placement_want: FormulaPlacement::Any,
            formula_chord: None,
            formula_degree: 0,
            formula_hits: 0,
            formula_verdict: None,
            formula_root: 0,
            formula_key_name: String::new(),
            formula_collected: Vec::new(),
            formula_notes: 5,
            formula_required: 1,
            formula_random_key: true,
            formula_key_setting: "C".to_string(),
        }
    }
    
    /// Two octaves of a chord's own tones, running one way.
    ///
    /// Ascending is the k-th tone above the root; descending is the k-th tone
    /// BELOW it, which is the shape the studies use over the changes - `1 7, 5,
    /// 3, 1, 7,, 5,, 3,,` and not the ascending phrase read backwards. The
    /// first ends where the ear expects a line to end; the second lands on the
    /// root and stops the sentence dead.
    fn changes_run(&self, chord: &Chord, descending: bool) -> Vec<Step> {
        let tones = chord.quality.interval_names().len().max(1);
        (0..tones * 2)
            .map(|k| {
                if descending {
                    let degree = (tones - k % tones) % tones;
                    let octave = -(((k + tones - 1) / tones) as i8);
                    Step { degree, octave: if k % tones == 0 { -((k / tones) as i8) } else { octave } }
                } else {
                    Step { degree: k % tones, octave: (k / tones) as i8 }
                }
            })
            .collect()
    }

    /// Which way the arpeggio runs over the chord now due.
    fn changes_descending(&self) -> bool {
        match self.arp_direction {
            ArpDirection::Up => false,
            ArpDirection::Down => true,
            ArpDirection::AlternateDownFirst => self.play_pos % 2 == 0,
            ArpDirection::AlternateUpFirst => self.play_pos % 2 == 1,
        }
    }

    /// The steps of the exercise over `chord`, read from the interval text -
    /// or, over the changes, built for the chord.
    pub fn get_active_indices(&self, chord: &Chord) -> Vec<Step> {
        // Over the changes there is no written phrase to read: the arpeggio is
        // built for the chord and for where it falls in the progression.
        if self.app_mode == AppMode::Arpeggios && self.arp_exercise == ArpExercise::OverChanges {
            return self.changes_run(chord, self.changes_descending());
        }
        if let Some(level) = self.level() {
            if let Some(steps) = level.run_steps(&chord.quality, self.level_rotation) {
                return steps;
            }
            let names = chord.quality.scale_names();
            return level
                .cell(&chord.quality, self.level_rotation)
                .iter()
                .filter_map(|token| {
                    let (token, octave) = split_octave(token);
                    // A ninth is the second an octave up, and is shown so.
                    let octave = octave + compound_octave(token);
                    level_degree(token, &names).map(|degree| Step { degree, octave })
                })
                .collect();
        }
        let names = chord.quality.interval_names();
        let read_step = |token: &str| {
            let (token, octave) = split_octave(token);
            let degree = if self.app_mode == AppMode::Arpeggios {
                arpeggio_degree(token, &names)
            } else {
                scale_degree(token, &names)
            };
            degree.map(|degree| Step { degree, octave })
        };
        let mut steps: Vec<Step> = self.intervals_input.split_whitespace().filter_map(read_step).collect();

        let whole_neck = self.app_mode == AppMode::Scales && self.scale_e_to_e;
        if whole_neck {
            steps = self.e_to_e_from_its_start(chord);
        } else if self.app_mode == AppMode::Scales {
            steps = self.scale_from_its_start(steps, &names);
        }

        // A scale read as a scale ends where it started: the same note an octave
        // up. It is a step of its own, so it has to be played - and it is marked
        // an octave up, which is what puts the tick on the strip as `1'`. From E
        // to e the run already ends on the tone it started on.
        if self.app_mode == AppMode::Scales && self.scale_repeat_root && !whole_neck {
            if let Some(&first) = steps.first() {
                steps.push(Step { degree: first.degree, octave: first.octave + 1 });
            }
        }

        if self.app_mode == AppMode::Scales {
            if self.scale_runs_down() {
                steps.reverse();
            }
            // And back the way it came, without playing the turn twice.
            if self.scale_there_and_back && steps.len() > 1 {
                let back: Vec<Step> = steps[..steps.len() - 1].iter().rev().copied().collect();
                steps.extend(back);
            }
        }

        if steps.is_empty() && !names.is_empty() {
            steps.push(Step { degree: 0, octave: 0 });
        }
        steps
    }

    /// The level Intervals is playing, if it is playing one.
    pub fn level(&self) -> Option<&'static Level> {
        if self.app_mode != AppMode::Intervals {
            return None;
        }
        self.interval_level.and_then(|i| interval_levels::LEVELS.get(i))
    }

    /// Whether the steps are an order of their own - a scale, a study, a cell
    /// or a line - rather than a set the player may take in any order.
    fn steps_are_a_line(&self) -> bool {
        matches!(self.app_mode, AppMode::Arpeggios | AppMode::Scales)
            || self.level().is_some_and(|level| level.ordered)
    }

    /// Whether the shuffle may deal the steps in another order - see
    /// `Level::shuffles`.
    fn steps_shuffle(&self) -> bool {
        !matches!(self.app_mode, AppMode::Arpeggios | AppMode::Scales)
            && self.level().is_none_or(|level| level.shuffles)
    }

    /// The names a step's `degree` indexes: in a level of Intervals the
    /// chord's scale, everywhere else the chord's own tones.
    pub fn degree_names(&self, chord: &Chord) -> Vec<String> {
        if self.level().is_some() { chord.quality.scale_names() } else { chord.quality.interval_names() }
    }

    /// Semitones above the root, in the order of `degree_names`.
    pub fn degree_intervals(&self, chord: &Chord) -> Vec<u8> {
        if self.level().is_some() { chord.quality.scale_intervals() } else { chord.quality.intervals() }
    }

    /// Pitch classes, in the order of `degree_names`.
    pub fn degree_targets(&self, chord: &Chord) -> Vec<usize> {
        if self.level().is_some() { chord.scale_targets() } else { chord.get_target_indices() }
    }

    /// Voice leading: the chord now due starts on its tone nearest the last
    /// one played - a step away, never the same note, so the line moves - and
    /// the rest keeps its drawn order. A tone of the chord, not of its scale:
    /// the line lands on the harmony. Ties go to whichever was drawn first.
    fn lead_from_last_note(&mut self) {
        let (Some(last), Some(chord)) = (self.last_played, self.chords.get(self.current_chord_index)) else {
            return;
        };
        if !self.steps_shuffle() {
            return;
        }
        let steps = self.get_active_indices(chord);
        let targets = self.degree_targets(chord);
        if steps.len() != self.step_order.len() {
            return;
        }
        let distance = |pc: usize| {
            let up = (pc + 12 - last) % 12;
            up.min(12 - up)
        };
        let nearest = self
            .step_order
            .iter()
            .enumerate()
            .filter(|&(_, &step)| steps[step].degree < 4)
            .map(|(place, &step)| (place, distance(targets[steps[step].degree])))
            .filter(|&(_, d)| d > 0)
            .min_by_key(|&(_, d)| d);
        if let Some((place, _)) = nearest {
            self.step_order.swap(0, place);
        }
    }

    /// Another exercise: the lap starts again on it.
    pub fn set_interval_level(&mut self, index: usize) {
        if index < interval_levels::LEVELS.len() && self.interval_level != Some(index) {
            self.interval_level = Some(index);
            self.reset_logic_state();
            self.reroll();
        }
    }

    /// `get_active_indices` in play order. Without randomisation this is the
    /// identity; with it, the stored permutation. The UI renders the same order,
    /// so the highlight still runs left to right instead of jumping around.
    pub fn ordered_active_indices(&self, chord: &Chord) -> Vec<Step> {
        let active = self.get_active_indices(chord);
        // A phrase dealt in a random order is not that phrase any more - a
        // study written to climb in broken thirds comes back as a list of
        // notes, and a scale that is not walked in order is not a scale. In
        // both, what the shuffle means is the KEY and where on the neck to
        // take it; in Intervals the set really is a set, and there it scatters.
        if !self.steps_shuffle() {
            return active;
        }
        // The permutation goes stale when the user edits the interval list; the
        // length check catches that and falls back to the plain order.
        if !self.random_mode || self.step_order.len() != active.len() {
            return active;
        }
        self.step_order.iter().map(|&i| active[i]).collect()
    }

    /// Toggling the switch takes effect at once instead of waiting for the next
    /// chord: without the reroll the shuffle and the hint would appear only after
    /// the current exercise is finished, which reads as the switch being broken.
    /// Same treatment as the shuffle switch: it takes effect at once, because a
    /// change that waited for the end of the song would read as broken.
    pub fn set_shuffle_chords(&mut self, on: bool) {
        if self.shuffle_chords != on {
            self.shuffle_chords = on;
            self.rebuild_play_order();
            self.prev_chord_index = None;
            self.prev_status = MatchStatus::None;
            self.reroll();
        }
    }

    pub fn set_random_mode(&mut self, on: bool) {
        if self.random_mode != on {
            self.random_mode = on;
            // Both ways round this starts the song again: switching on shuffles
            // it up front, switching off returns to the written order from the
            // top. Carrying on mid-order would leave the strip showing
            // neighbours from an order no longer in force.
            self.rebuild_play_order();
            self.prev_chord_index = None;
            self.prev_status = MatchStatus::None;
            self.reroll();
            // In the arpeggio studies the shuffle has nothing to shuffle - a
            // phrase dealt in a random order is not that phrase - so what it
            // means there is the KEY: a new one after every pass, which
            // `advance_chord` draws. The study itself is the player's choice
            // and is left alone; the generator is a row in the list for anyone
            // who wants a new phrase as well.
            //
            // In Scales the switch is called "random key", and switched on it
            // draws one at once: waiting for the scale to be finished read as
            // the switch doing nothing.
            if on && self.app_mode == AppMode::Scales && !self.chords.is_empty() {
                let current = self.chords[0].root as usize;
                let next = self.rng.below_excluding(12, current);
                self.chords[0].root = NoteName::from_index(next);
                self.secondary_index = next;
                self.scale_start = self.rng.below(START_STRINGS.len());
                self.reset_logic_state();
            }
        }
    }

    /// Does the shuffle reach the chord order in the current mode?
    ///
    /// Split out because the toolbar switch means two different things at once:
    /// the order of the chords, and the order of the tones inside each. Only the
    /// note modes have both, and only there is the distinction worth a setting.
    fn shuffles_chord_order(&self) -> bool {
        if !self.random_mode {
            return false;
        }
        match self.app_mode {
            AppMode::Intervals | AppMode::Arpeggios => self.shuffle_chords,
            _ => true,
        }
    }

    /// Rebuilds the playing order and starts it from the beginning.
    fn rebuild_play_order(&mut self) {
        self.play_order = (0..self.chords.len()).collect();
        if self.shuffles_chord_order() {
            self.rng.shuffle(&mut self.play_order);
        }
        self.play_pos = 0;
        self.current_chord_index = self.play_order.first().copied().unwrap_or(0);
        self.pass_results = vec![MatchStatus::None; self.play_order.len()];
    }

    /// How the chord `offset` places from the current one in the order was
    /// left on this pass - see `pass_results`.
    pub fn pass_result(&self, offset: i32) -> MatchStatus {
        let pos = self.play_pos as i32 + offset;
        usize::try_from(pos)
            .ok()
            .and_then(|pos| self.pass_results.get(pos).copied())
            .unwrap_or(MatchStatus::None)
    }

    /// What the strip shows as coming next - the real next entry in the order,
    /// not the chord one further along the song.
    /// The strip of the progression: `2 * radius + 1` chords in the order
    /// they are played, each with its distance from the current one, negative
    /// behind it. It starts on the first chord of the lap and the current one
    /// walks along it to the middle; from there the strip moves under it, so
    /// there are always `radius` to come - round the end into the next lap,
    /// whose order is not redrawn. A progression shorter than the strip is
    /// shown once, whole.
    pub fn progress_around(&self, radius: usize) -> Vec<(usize, i32)> {
        let len = self.play_order.len();
        if len == 0 {
            return Vec::new();
        }
        let pos = self.play_pos.min(len - 1);
        let first = pos.saturating_sub(radius);
        let shown = (2 * radius + 1).min(len);
        (first..first + shown)
            .map(|at| (self.play_order[at % len], at as i32 - pos as i32))
            .collect()
    }

    pub fn next_chord_index(&self) -> usize {
        if self.play_order.is_empty() {
            return 0;
        }
        self.play_order[(self.play_pos + 1) % self.play_order.len()]
    }

    /// Swaps in a freshly built phrase when the generator entry is selected.
    ///
    /// Nothing happens for the hand-written patterns, so the two kinds live in
    /// the same list without a second code path.
    fn regenerate_arpeggio(&mut self) {
        // Only where a phrase is chosen at all. Over the changes it is built
        // for each chord, and there is no entry in a list to be the generator.
        if self.app_mode != AppMode::Arpeggios || self.arp_exercise != ArpExercise::Study {
            return;
        }
        // The phrase is the FIRST combo in the studies - the second one holds
        // the key. Reading the pattern list at the key's index instead drew a
        // fresh random phrase whenever the key happened to land on the
        // generator's row: the title said one study and the neck showed
        // another.
        let is_generator = self.arpeggio_patterns.get(self.selected_library_idx)
            .map(|p| p.name == crate::model::GENERATOR_NAME)
            .unwrap_or(false);
        if is_generator {
            self.intervals_input = arpeggio::random(&mut self.rng).join(" ");
        }
    }

    /// Which way this pass of the scale runs.
    pub fn scale_runs_down(&self) -> bool {
        match self.scale_direction {
            ScaleDirection::Up => false,
            ScaleDirection::Down => true,
            ScaleDirection::Random => self.scale_descending,
        }
    }

    /// The scale turned to start on the degree `scale_from` asks for, the
    /// tones below it carried an octave up. A scale without that degree - a
    /// pentatonic has no seventh - starts on its root.
    fn scale_from_its_start(&self, steps: Vec<Step>, names: &[String]) -> Vec<Step> {
        let token = match self.scale_from {
            ScaleFrom::Root => return steps,
            ScaleFrom::Third => "3",
            ScaleFrom::Fifth => "5",
            ScaleFrom::Seventh => "7",
            ScaleFrom::Random => {
                let start = self.scale_from_drawn % steps.len().max(1);
                return rotated(steps, start);
            }
        };
        let start = scale_degree(token, names)
            .and_then(|degree| steps.iter().position(|s| s.degree == degree && s.octave == 0))
            .unwrap_or(0);
        rotated(steps, start)
    }

    /// The degree a scale starts on: the one `scale_from` names, or the root
    /// where the scale has none - a pentatonic has no seventh.
    fn start_degree(&self, names: &[String]) -> usize {
        let token = match self.scale_from {
            ScaleFrom::Root => return 0,
            ScaleFrom::Third => "3",
            ScaleFrom::Fifth => "5",
            ScaleFrom::Seventh => "7",
            ScaleFrom::Random => return self.scale_from_drawn % names.len().max(1),
        };
        scale_degree(token, names).unwrap_or(0)
    }

    /// Where the hand stands for a scale from E to e this pass: the frets it
    /// covers, and the note the run starts on. The hand is round a fret where
    /// the start degree lies on some string, in the first twelve - drawn for
    /// every pass - as `tab::hand_from` has it. The run starts on
    /// that degree as low as the hand has it, the low E string first; run
    /// downwards, as high, the high e first. `None` outside a scale from E to e.
    pub fn scale_hand(&self, chord: &Chord) -> Option<((i32, i32), i32)> {
        if self.app_mode != AppMode::Scales || !self.scale_e_to_e {
            return None;
        }
        let frets = self.hand_frets(chord);
        let fret = *frets.get(self.scale_place_drawn % frets.len().max(1))?;
        let (lo, hi) = tab::hand_from(fret);
        let pc = self.start_pc(chord)?;
        let under_hand = (0..6)
            .flat_map(|string| (lo..=hi).map(move |fret| tab::TUNING[string] + fret))
            .filter(|pitch| pitch.rem_euclid(12) == pc);
        let start = if self.scale_runs_down() { under_hand.max() } else { under_hand.min() };
        Some(((lo, hi), start?))
    }

    /// The pitch class of the degree a scale starts on.
    fn start_pc(&self, chord: &Chord) -> Option<i32> {
        let intervals = chord.quality.intervals();
        let interval = *intervals.get(self.start_degree(&chord.quality.interval_names()))?;
        Some((chord.root as i32 + interval as i32) % 12)
    }

    /// Every fret in the first twelve where the start degree lies on some
    /// string: the places the hand can stand. Each once - two strings with it
    /// on one fret are the same hand.
    fn hand_frets(&self, chord: &Chord) -> Vec<i32> {
        let Some(pc) = self.start_pc(chord) else { return Vec::new() };
        (0..=12)
            .filter(|&fret| (0..6).any(|string| (tab::TUNING[string] + fret) % 12 == pc))
            .collect()
    }

    /// From E to e: the tones of the scale under the hand, from the note
    /// `scale_hand` starts on to the top - which, started on the low E string,
    /// is all of them but what lies below it there. Run downwards, from the
    /// start to the bottom: the steps up to it, turned round afterwards.
    fn e_to_e_from_its_start(&self, chord: &Chord) -> Vec<Step> {
        let Some(((lo, hi), start_pitch)) = self.scale_hand(chord) else {
            return Vec::new();
        };
        let root = tab::root_above_low_e(chord.root as usize);
        let intervals = chord.quality.intervals();
        let steps = scale_between(&intervals, root, tab::TUNING[0] + lo, tab::TUNING[5] + hi);
        let pitch = |s: &Step| root + intervals[s.degree] as i32 + 12 * s.octave as i32;
        let start = steps.iter().position(|s| pitch(s) == start_pitch).unwrap_or(0);
        if self.scale_runs_down() {
            steps[..=start].to_vec()
        } else {
            steps[start..].to_vec()
        }
    }

    /// A new pass of a scale: another string to start from - drawn, and never
    /// the one the last pass started on - and, where the direction is drawn, a
    /// new way and a new starting degree. Only a finished scale is a new pass: choosing
    /// another scale or key keeps the way it was going.
    fn next_scale_pass(&mut self) {
        self.pass = self.pass.wrapping_add(1);
        // Taken in turn, the place on the neck came round in an order the
        // hand learned; drawn from the other two, it is somewhere new.
        let others = START_STRINGS.len() - 1;
        self.scale_start = (self.scale_start + 1 + self.rng.below(others)) % START_STRINGS.len();
        // Drawn, but never three passes the same way: a fair coin gave five
        // descents in a row, which read as a scale that only goes down.
        let previous = self.scale_descending;
        let mut down = self.rng.below(2) == 1;
        if down == previous && self.same_way_passes >= 2 {
            down = !previous;
        }
        self.same_way_passes = if down == previous { self.same_way_passes + 1 } else { 1 };
        self.scale_descending = down;
        self.scale_from_drawn = self.rng.below(12);
    }

    /// The scale options, read from the UI. A change to the run's shape starts
    /// the lap again; "in order" only changes how it is judged.
    pub fn set_scale_way(&mut self, direction: ScaleDirection, from: ScaleFrom, there_and_back: bool, key_move: KeyMove, in_order: bool) {
        let changed = (self.scale_direction, self.scale_from, self.scale_there_and_back)
            != (direction, from, there_and_back);
        self.scale_direction = direction;
        self.scale_from = from;
        self.scale_there_and_back = there_and_back;
        self.key_move = key_move;
        self.scale_in_order = in_order;
        if changed && self.app_mode == AppMode::Scales {
            self.reset_logic_state();
        }
    }

    /// From E to e, switched: the run changes shape, so the lap starts again.
    pub fn set_scale_e_to_e(&mut self, on: bool) {
        if self.scale_e_to_e != on {
            self.scale_e_to_e = on;
            if self.app_mode == AppMode::Scales {
                    self.reset_logic_state();
            }
        }
    }

    /// Draws a new order of tones and a new string hint. Called on every chord
    /// change - never per frame, or the target would move while playing.
    fn reroll(&mut self) {
        // Where the hand stands from E to e: drawn for every pass and every
        // new exercise, and another place than the one before. Drawn only
        // between passes, a scale just chosen always stood at the nut.
        let places = self.chords.first().map_or(1, |chord| self.hand_frets(chord).len());
        if places > 1 {
            self.scale_place_drawn = (self.scale_place_drawn + 1 + self.rng.below(places - 1)) % places;
        }
        let n = if self.chords.is_empty() {
            0
        } else {
            self.get_active_indices(&self.chords[self.current_chord_index]).len()
        };
        // Where the hand takes the first grip in Intervals: drawn, so the
        // exercise does not always start from the same place on the neck. Only
        // the first one - after that the voices are led from the grip before.
        self.voicing_anchor = 1 + self.rng.below(9) as i32;
        // Which string a scale starts from and which way it runs: TAKEN IN
        // TURN, not drawn. The shape is the same everywhere and a scale known
        // upwards is half known, so what is wanted is coverage - and a fair
        // coin does not give it. Twelve drawn passes came out with five
        // descending in a row, which reads as "it only goes down"; three
        // strings and two directions are coprime, so counting through them
        // gives all six pairs in six passes and never twice the same in a row.
        // And which strings an interval grip is taken on when nothing is
        // leading it there: the bottom four can carry the lowest voice.
        self.grip_string = self.rng.below(4);
        self.level_rotation = self.rng.below(8);
        self.step_order = (0..n).collect();
        if self.random_mode {
            self.rng.shuffle(&mut self.step_order);
            if self.lead_line {
                self.lead_from_last_note();
            }
            self.start_hint = Some(self.rng.below(START_STRINGS.len()));
        } else {
            self.start_hint = None;
        }
    }

    pub fn set_mode(&mut self, mode_idx: i32) {
        let new_mode = AppMode::from(mode_idx);
        let is_chord_mode = |m| m == AppMode::Chords || m == AppMode::Intervals;
        
        if is_chord_mode(self.app_mode) && !is_chord_mode(new_mode) {
            self.saved_intervals_input = self.intervals_input.clone();
        } else if !is_chord_mode(self.app_mode) && is_chord_mode(new_mode) {
            self.intervals_input = self.saved_intervals_input.clone();
        }

        // Leaving Arpeggios: remember the study and the key it stood in, so
        // coming back lands where it was left. Only in the studies - over the
        // changes the first combo holds a tune, and a tune's row is not a
        // phrase's row.
        if self.app_mode == AppMode::Arpeggios && self.arp_exercise == ArpExercise::Study {
            self.saved_arpeggio_index = self.selected_library_idx;
            self.saved_arpeggio_key = self.secondary_index;
        }
        // And leaving a mode that reads the tunes: the standard follows.
        if self.reads_songs() {
            self.saved_song_idx = self.selected_library_idx;
        }
        if self.app_mode == AppMode::Scales {
            self.saved_scale_idx = self.selected_library_idx;
        }

        self.app_mode = new_mode;
        self.selected_library_idx = 0;
        self.secondary_index = 0;
        // The tune comes with. Four modes read the same library and a standard
        // is worked on across them, so dropping back to the first row on every
        // switch was work for the player and nothing else.
        if self.reads_songs() {
            self.selected_library_idx = self.saved_song();
        }
        if new_mode == AppMode::Scales {
            self.selected_library_idx = self.saved_scale();
        }
        // The arpeggio studies come back to the phrase and the key they were
        // left in - which for the generator means a freshly built phrase, not
        // the first fixed one. Scales start from C.
        if new_mode == AppMode::Arpeggios && self.arp_exercise == ArpExercise::Study {
            self.selected_library_idx =
                self.saved_arpeggio_index.min(self.arpeggio_patterns.len().saturating_sub(1));
            self.secondary_index = self.saved_arpeggio_key;
        }
        self.reload_library_content();
    }

    pub fn item_selected(&mut self, index: i32) {
        self.selected_library_idx = index as usize;
        // Another tune comes in its own key.
        self.transpose = 0;
        if self.reads_songs() {
            self.saved_song_idx = self.selected_library_idx;
        }
        if self.app_mode == AppMode::Scales {
            self.saved_scale_idx = self.selected_library_idx;
        }
        self.reload_library_content();
    }

    /// Whether the set may be taken in any order. Only Intervals ever can: a
    /// scale or an arpeggio is an order, and a chord has none to walk.
    pub fn free_order(&self) -> bool {
        match self.app_mode {
            AppMode::Intervals => !self.interval_in_order && !self.steps_are_a_line(),
            AppMode::Scales => !self.scale_in_order,
            _ => false,
        }
    }

    /// The step the exercise is waiting for, where it is waiting for a
    /// particular one. `None` under free order - there is no note "due" then,
    /// and the drawing must not ring one.
    pub fn note_due(&self) -> Option<usize> {
        (!self.free_order()).then_some(self.current_note_step)
    }

    /// Where one step of the exercise stands. Every drawing of the steps reads
    /// this, and it reads the judge's marks, including credits earned out of
    /// order. A cursor is not a record of what was played.
    pub fn note_state(&self, step: usize) -> NoteState {
        if self.collected_notes.get(step).copied().unwrap_or(false) {
            NoteState::Done
        } else if self.answering_step == Some(step) && self.success_timer > 0.05 {
            NoteState::Answering
        } else if self.free_order() {
            NoteState::Open
        } else if self.note_due() == Some(step) {
            NoteState::Due
        } else {
            NoteState::Waiting
        }
    }

    pub fn note_color(&self, step: usize) -> (u8, u8, u8) {
        match self.note_state(step) {
            NoteState::Done => (50, 255, 50),
            NoteState::Answering => (200, 255, 50),
            NoteState::Open | NoteState::Due => (180, 180, 180),
            NoteState::Waiting => (60, 60, 60),
        }
    }

    /// Whether the first combo holds tunes in the mode running now: the chords
    /// of a standard, the intervals inside them, an arpeggio over each chord,
    /// or a formula planted on each. The other exercises stand on their own and
    /// their first combo holds a scale or a study.
    pub fn reads_songs(&self) -> bool {
        match self.app_mode {
            AppMode::Chords | AppMode::Intervals => true,
            AppMode::Arpeggios => self.arp_exercise == ArpExercise::OverChanges,
            AppMode::Formulas => self.formula_exercise == FormulaExercise::OverChanges,
            _ => false,
        }
    }

    /// The remembered tune's row, kept inside the library as it stands now.
    pub fn saved_song(&self) -> usize {
        self.saved_song_idx.min(self.song_library.len().saturating_sub(1))
    }

    /// The same for the scale.
    pub fn saved_scale(&self) -> usize {
        self.saved_scale_idx.min(self.scale_definitions.len().saturating_sub(1))
    }

    /// Picks the tune of this name, if the library still holds it, and says
    /// whether it did. Called with what the settings remembered - the library
    /// is read from disk and can lose a tune between sessions, and a name that
    /// is gone has to leave the first row standing rather than nothing.
    pub fn select_song(&mut self, title: &str) -> bool {
        let Some(i) = self.song_library.iter().position(|s| s.title == title) else {
            return false;
        };
        self.saved_song_idx = i;
        if self.reads_songs() {
            self.selected_library_idx = i;
            self.reload_library_content();
        }
        true
    }

    /// The same for a scale, by the name the definitions carry - not the
    /// translated one on screen, which changes with the language setting.
    pub fn select_scale(&mut self, name: &str) -> bool {
        let Some(i) = self.scale_definitions.iter().position(|d| d.name == name) else {
            return false;
        };
        self.saved_scale_idx = i;
        if self.app_mode == AppMode::Scales {
            self.selected_library_idx = i;
            self.reload_library_content();
        }
        true
    }
    
    pub fn secondary_item_selected(&mut self, index: i32) {
        self.secondary_index = index as usize;
        if self.app_mode == AppMode::Scales || self.app_mode == AppMode::Arpeggios {
            self.reload_library_content();
        }
    }

    /// The key the tune now selected is written in, as a pitch class: the root
    /// of its last chord, which is the tonic in nearly every standard. `None`
    /// with no tune.
    pub fn song_tonic(&self) -> Option<usize> {
        let song = self.song_library.get(self.selected_library_idx)?;
        song.chords.last().map(|c| c.root as usize)
    }

    /// Chords: play the tune in `key`, a pitch class. The progression starts
    /// again in it.
    pub fn set_song_key(&mut self, key: usize) {
        let Some(tonic) = self.song_tonic() else { return };
        let transpose = (key % 12 + 12 - tonic) % 12;
        if transpose != self.transpose {
            self.transpose = transpose;
            self.reload_library_content();
            self.reset_logic_state();
        }
    }

    /// Reads the library again for whatever mode is running.
    ///
    /// Public for the one caller outside: switching a formula exercise to "over
    /// the changes" needs a tune under it before the next chord can be taken.
    pub fn reload_library(&mut self) {
        self.reload_library_content();
    }

    fn reload_library_content(&mut self) {
        if self.app_mode == AppMode::Formulas {
            // Over the changes the formula needs a tune under it, and that is
            // the same library Chords reads from.
            if self.formula_exercise == FormulaExercise::OverChanges && self.selected_library_idx < self.song_library.len() {
                let song = &self.song_library[self.selected_library_idx];
                self.song_title = song.title.clone();
                self.chords = song.chords.clone();
                self.current_chord_index = 0;
            }
            self.next_formula();
            return;
        }
        match self.app_mode {
            AppMode::Chords | AppMode::Intervals => {
                if self.selected_library_idx < self.song_library.len() {
                    let song = &self.song_library[self.selected_library_idx];
                    self.song_title = song.title.clone();
                    self.chords = song.chords.clone();
                    if self.app_mode == AppMode::Chords {
                        for chord in &mut self.chords {
                            chord.root = NoteName::from_index(chord.root as usize + self.transpose);
                        }
                    }
                }
            }
            AppMode::Arpeggios if self.arp_exercise == ArpExercise::Study => {
                // A study stands on its own: one chord, in the key chosen or
                // drawn, and the phrase is whichever study is selected.
                let quality = ARP_QUALITIES
                    .get(self.arp_quality)
                    .map(|(_, q)| q.clone())
                    .unwrap_or(ChordQuality::Minor7);
                let root = NoteName::from_index(self.secondary_index.min(11));
                self.chords = vec![Chord { root, quality }];
                if self.selected_library_idx < self.arpeggio_patterns.len() {
                    let pattern = &self.arpeggio_patterns[self.selected_library_idx];
                    self.song_title = pattern.name.clone();
                    self.intervals_input = pattern.names.join(" ");
                } else {
                    self.song_title = String::new();
                    self.intervals_input = "1 3 5 7".to_string();
                }
            }
            AppMode::Arpeggios => {
                // Over the changes the phrase is not chosen but built, one
                // arpeggio per chord, running the way `arp_direction` says.
                if self.selected_library_idx < self.song_library.len() {
                    let song = &self.song_library[self.selected_library_idx];
                    self.song_title = song.title.clone();
                    self.chords = song.chords.clone();
                }
                self.intervals_input = String::new();
            }
            AppMode::Scales => {
                if self.selected_library_idx < self.scale_definitions.len() {
                    let def = &self.scale_definitions[self.selected_library_idx];
                    self.song_title = def.name.clone();
                    self.intervals_input = def.names.join(" ");
                    let root = NoteName::from_index(self.secondary_index);
                    self.chords = vec![Chord { root: root, quality: ChordQuality::CustomScale(def.clone()) }];
                }
            }
            AppMode::Fretboard | AppMode::Formulas => {
                // No song, no chords - the exercise is the region itself, drawn
                // fresh on every entry into the mode.
                self.chords = vec![];
                self.song_title = String::new();
                self.randomize_region();
            }
        }
        self.regenerate_arpeggio();
        self.reset_logic_state();
    }

    /// Drops everything the app has heard so far, except which note was
    /// credited: see `credited`, whose whole purpose is to outlive the step it
    /// belongs to.
    ///
    /// The model answers about 0.77 s of audio, so at the moment the exercise
    /// moves on - a new chord, a new mode - its last answer is still about what
    /// came before. Left in place it credits the first target of the new chord
    /// from the ringing of the old one, which is why a mode could feel sharp on
    /// the first chord and loose on every one after it. Nothing is lost by
    /// forgetting: the next audio frame is 16 ms away and the next inference
    /// 40 ms.
    fn forget_what_was_heard(&mut self) {
        self.last_pitches = [0.0; 12];
        self.prev_pitches = [0.0; 12];
        self.ear_window = [None; EAR_WINDOW];
        self.cqt_run_pitch = None;
        self.cqt_run = 0;
        self.last_ai_root = None;
        self.last_ai_conf = 0.0;
        self.answering_step = None;
        self.interval_hold = 0.0;
        self.interval_model_age = INTERVAL_MODEL_AGE;
        self.interval_audio_age = 0.0;
        self.interval_silence = 0.0;
        self.judged_frame = self.heard.audio_frames;
        self.chord_heard_at = ChordHeard {
            onset_id: self.heard.onset_id,
            class_strikes: self.heard.class_strikes,
        };
    }

    /// What the ear hears, steadily - the state three audio frames of the same
    /// reading leave behind. Test seam.
    #[cfg(test)]
    fn hears(&mut self, pitch: Option<usize>) {
        self.heard.cqt_pitch = pitch;
        self.heard.steady_pitch = pitch;
        self.heard.steady_for = if pitch.is_some() { 3 } else { 0 };
    }

    /// One audio frame's reading, as the sync below takes it. Test seam: the
    /// real path reads it from the shared state under a lock.
    #[cfg(test)]
    fn feed_estimate(&mut self, frame: u64, pitch: Option<usize>) {
        if frame != self.heard.audio_frames {
            if pitch.is_some() && pitch == self.heard.steady_pitch {
                self.heard.steady_for = self.heard.steady_for.saturating_add(1);
            } else {
                self.heard.steady_pitch = pitch;
                self.heard.steady_for = 1;
            }
        }
        self.heard.cqt_pitch = pitch;
        self.heard.audio_frames = frame;
    }




    pub fn reset_logic_state(&mut self) {
        self.forget_what_was_heard();
        // What was credited outlives a chord or a lap - a run ending on the
        // note it starts from must not hand the next lap its first step off the
        // last one still ringing - but not a change of exercise, where nothing
        // is being continued.
        self.credited.forget_all();
        self.last_played = None;
        self.rebuild_play_order();
        self.current_note_step = 0;
        self.answering_step = None;
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
        self.chord_history.clear();
        self.reroll();
        self.update_collected_notes_size();
    }

    fn update_collected_notes_size(&mut self) {
        if !self.chords.is_empty() {
             let curr_chord = &self.chords[self.current_chord_index];
             let active_indices = self.get_active_indices(curr_chord);
             self.collected_notes = vec![false; active_indices.len()];
        } else {
             self.collected_notes = vec![];
        }
    }

    fn parse_ai_prediction(&self, pred: &str) -> (Option<NoteName>, String) {
        if pred == "Noise" || pred == "Unknown" || pred == "..." {
            return (None, "".to_string());
        }
        let parts: Vec<&str> = pred.split_whitespace().collect();
        if parts.is_empty() { return (None, "".to_string()); }

        if parts[0] == "Note" && parts.len() > 1 {
            return (self.str_to_note(parts[1]), "Note".to_string());
        }

        let root = self.str_to_note(parts[0]);
        let qual = if parts.len() > 1 { parts[1].to_string() } else { "".to_string() };
        (root, qual)
    }

    fn str_to_note(&self, s: &str) -> Option<NoteName> {
        match s {
            "C" => Some(NoteName::C), "C#" | "Db" => Some(NoteName::Df),
            "D" => Some(NoteName::D), "D#" | "Eb" => Some(NoteName::Ef),
            "E" => Some(NoteName::E), "F" => Some(NoteName::F),
            "F#" | "Gb" => Some(NoteName::Fsh), "G" => Some(NoteName::G),
            "G#" | "Ab" => Some(NoteName::Af), "A" => Some(NoteName::A),
            "A#" | "Bb" => Some(NoteName::Bf), "B" => Some(NoteName::B),
            _ => None
        }
    }

    /// UI clock: finished results remain visible even when audio stops.
    /// Intervals and Formulas also judge here, but only on a new audio frame;
    /// polling the same CQT sample twice must not count as two observations.
    pub fn tick(&mut self, dt: f32) {
        match self.app_mode {
            AppMode::Intervals => self.tick_intervals(dt),
            AppMode::Fretboard => self.tick_fretboard(dt),
            AppMode::Formulas => self.tick_formulas(dt),
            AppMode::Chords | AppMode::Scales | AppMode::Arpeggios => {}
        }
    }

    pub fn check_progress_with_ai(&mut self, dt: f32, ai_prediction: &str, confidence: f32) {
        let (ai_root, ai_qual) = self.parse_ai_prediction(ai_prediction);
        if self.app_mode == AppMode::Intervals {
            // Receiving an inference result is not an audio tick. Single
            // notes can be heard while the model has no window to answer.
            self.last_ai_root = ai_root;
            self.last_ai_conf = confidence;
            self.interval_model_age = 0.0;
            return;
        }
        self.check_progress(dt, ai_root, &ai_qual, confidence);
    }

    fn check_progress(&mut self, dt: f32, ai_root: Option<NoteName>, ai_qual: &str, confidence: f32) {
        if self.app_mode != AppMode::Intervals {
            self.credited.settle(dt, &self.heard);
        }
        match self.app_mode {
            AppMode::Formulas => {
                // Only remember what was said. The judging happens once per audio
                // frame in `tick`, because this mode credits on a per-frame reading
                // and the model is asked only while the window is full - which
                // playing one note at a time never manages, so waiting here meant
                // waiting for an answer that often never came.
                self.last_ai_root = ai_root;
                self.last_ai_conf = confidence;
            }
            AppMode::Fretboard => self.check_fretboard(dt, ai_root, confidence),
            AppMode::Chords => self.check_chord(dt, ai_root, ai_qual, confidence),
            AppMode::Intervals | AppMode::Scales | AppMode::Arpeggios => {
                self.check_notes(dt, ai_root, confidence)
            }
        }
    }

    /// Whether the success timer has run past `threshold` with play on, so the
    /// step can be passed. Paused, the timer is held at the threshold: it would
    /// otherwise grow for as long as the pause lasts and the next step would be
    /// passed the moment play resumes.
    fn held_past(&mut self, threshold: f32) -> bool {
        if self.success_timer <= threshold {
            return false;
        }
        if self.paused {
            self.success_timer = threshold;
            return false;
        }
        true
    }

    /// Name of the chord at `index`, as the strip along the bottom shows it.
    pub fn chord_label(&self, index: usize) -> String {
        self.chords
            .get(index)
            .map(|c| format!("{} {}", c.root.to_string(), c.quality.to_string()))
            .unwrap_or_default()
    }

    /// How the previous chord was passed - `None` means it was stepped over.
    pub fn prev_status(&self) -> MatchStatus {
        self.prev_status
    }

    /// Steps through the progression by hand, to go back to a chord that has
    /// already gone by. Only useful while paused - playing would move it on
    /// again at once - so the caller decides when to offer it.
    pub fn step_chord(&mut self, delta: i32) {
        // A formula stands until it is asked to move, and the arrows are the
        // asking. No progression here, so no direction either - both draw.
        if self.app_mode == AppMode::Formulas {
            self.next_formula();
            return;
        }
        if self.chords.is_empty() {
            return;
        }
        if self.play_order.len() != self.chords.len() {
            self.rebuild_play_order();
        }
        let len = self.play_order.len() as i32;
        self.prev_chord_index = Some(self.current_chord_index);
        // Along the order in force, so stepping back lands on the chord that was
        // actually played before - not the one written before it in the song.
        self.play_pos = (self.play_pos as i32 + delta).rem_euclid(len) as usize;
        self.current_chord_index = self.play_order[self.play_pos];
        // Stepping is not passing: nothing is lit, and nothing is banked
        // towards the chord stepped onto.
        self.success_timer = 0.0;
        self.current_note_step = 0;
        self.match_status = MatchStatus::None;
        self.prev_status = MatchStatus::None;
        self.forget_what_was_heard();
        // Landing on a chord draws it a string to start from and an order for
        // its notes, exactly as arriving there by playing would. Without this
        // the suggestion stayed on whatever it was when the arrows started.
        self.reroll();
        self.update_collected_notes_size();
    }





    /// Writes down one row, with the clock and the exercise's own context on
    /// it. Costs nothing when SOLITITO_RECORD is unset.
    fn journal_row(&mut self, event: &str, mut row: serde_json::Value) {
        if self.journal.is_none() {
            return;
        }
        let chord = self.chords.get(self.current_chord_index)
            .map(|c| format!("{} {}", c.root.to_string(), c.quality.to_string()));
        if let Some(map) = row.as_object_mut() {
            map.insert("event".into(), event.into());
            map.insert("hop".into(), self.heard.capture_hop.into());
            map.insert(
                "t".into(),
                serde_json::json!(
                    self.heard.capture_hop as f64 * crate::audio::HOP_LENGTH as f64
                        / crate::audio::TARGET_SR as f64
                ),
            );
            map.insert("mode".into(), format!("{:?}", self.app_mode).into());
            map.insert("chord".into(), serde_json::json!(chord));
            map.insert("step".into(), self.current_note_step.into());
        }
        if let Some(journal) = self.journal.as_mut() {
            journal.write(&row);
        }
    }

    /// A new request: which step is wanted now. Written whether or not anything
    /// answers it, because a request nobody answered is the other half of the
    /// measurement - the half this session kept forgetting to look at.
    fn journal_ask(&mut self) {
        if self.journal.is_none() {
            return;
        }
        let now = (self.current_chord_index, self.current_note_step);
        if self.logged_ask == Some(now) {
            return;
        }
        self.logged_ask = Some(now);
        self.journal_row("ask", serde_json::json!({}));
    }


    fn advance_chord(&mut self) {
        // Before anything is forgotten: the ear and the voices still describe
        // the chord being left, which is what has to be carried.
        self.carry_over_what_rings();
        // Captured before the reset below wipes it: this is how the chord being
        // left behind was actually matched, and the strip reports that.
        let earned = self.match_status;
        self.success_timer = 0.0;
        self.current_note_step = 0;
        self.match_status = MatchStatus::None;
        self.forget_what_was_heard();
        // Scales hold a single "chord" - the whole scale - so the list has one
        // entry and the index never moves. Advancing there means a new KEY:
        // finish the scale, get another one somewhere else on the neck.
        // The same in the arpeggio studies: they hold one chord and no
        // progression, so a finished pass means a new key, exactly as a
        // finished scale does.
        let key_exercise = self.app_mode == AppMode::Scales
            || (self.app_mode == AppMode::Arpeggios && self.arp_exercise == ArpExercise::Study);
        if key_exercise && !self.chords.is_empty() {
            let current = self.chords[0].root as usize;
            let next = if self.random_mode {
                self.rng.below_excluding(12, current)
            } else if self.app_mode == AppMode::Scales {
                (current + self.key_move.step()) % 12
            } else {
                current
            };
            self.chords[0].root = NoteName::from_index(next);
            // Keep the key combo honest; otherwise it would show the key the
            // player picked while the app asks for a different one.
            self.secondary_index = next;
        }
        if self.app_mode == AppMode::Scales {
            self.next_scale_pass();
        }

        self.prev_chord_index = Some(self.current_chord_index);
        self.prev_status = earned;
        if self.play_order.len() != self.chords.len() {
            self.rebuild_play_order();
        }
        // The order is not redrawn at the end of a lap. The strip promises the
        // next chord, and a reshuffle here would break that promise on the very
        // last one - it announced a chord from the order about to be replaced.
        // Toggling shuffle off and on deals a new one.
        self.pass_results.resize(self.play_order.len(), MatchStatus::None);
        if let Some(result) = self.pass_results.get_mut(self.play_pos) {
            *result = earned;
        }
        self.play_pos += 1;
        if self.play_pos >= self.play_order.len() {
            self.play_pos = 0;
            // A new pass: what the last one earned is not this one's.
            self.pass_results.fill(MatchStatus::None);
        }
        self.current_chord_index = self.play_order[self.play_pos];
        // A finished pass earns a new phrase when the generator is selected.
        self.regenerate_arpeggio();
        self.reroll();
        self.update_collected_notes_size();
    }

    /// Hands the audio thread the settings, and takes from it what it has
    /// heard since the last frame.
    pub fn sync_audio_settings(&mut self) {
        let analysis = Arc::clone(&self.analysis_state);
        let gate_open = match analysis.lock() {
            Ok(mut state) => {
                state.noise_gate = self.noise_gate;
                state.bass_boost_enabled = self.bass_boost_enabled;
                state.bass_boost_gain = self.bass_boost_gain;
                self.heard.read_audio_frame(&state);
                state.gate_open
            }
            Err(_) => false,
        };
        self.heard.audio_gate_open = gate_open;
        // Nothing refreshes the pitch vector while the gate is shut - the model
        // is not even asked - so without this it keeps its last value for as
        // long as the app runs, and a note that stopped sounding minutes ago
        // still counts as played.
        if !gate_open {
            self.last_pitches = [0.0; 12];
            self.prev_pitches = [0.0; 12];
        }
    }



}

/// The degree an arpeggio token asks for: the chord tones by number, in the
/// order the chord's names are written.
fn arpeggio_degree(token: &str, names: &[String]) -> Option<usize> {
    let degree = match token {
        "1" | "8" => 0,
        "3" => 1,
        "5" => 2,
        "7" => 3,
        "9" if names.len() > 4 => 4,
        "9" if names.len() > 1 => 1,
        "9" => 0,
        _ => return None,
    };
    (degree < names.len()).then_some(degree)
}

/// How far above the octave a token names its tone: the ninth, eleventh and
/// thirteenth are the second, fourth and sixth one octave up.
fn compound_octave(token: &str) -> i8 {
    let number = token.trim_start_matches(['b', '#']);
    matches!(number, "9" | "11" | "13") as i8
}

/// Every tone of a scale from pitch `low` to `high`, as steps from `root`.
fn scale_between(intervals: &[u8], root: i32, low: i32, high: i32) -> Vec<Step> {
    (low..=high)
        .filter_map(|note| {
            let above_root = note - root;
            let in_octave = above_root.rem_euclid(12) as u8;
            let degree = intervals.iter().position(|&i| i % 12 == in_octave)?;
            Some(Step { degree, octave: above_root.div_euclid(12) as i8 })
        })
        .collect()
}

/// `steps` started at `start`, those before it carried an octave up.
fn rotated(steps: Vec<Step>, start: usize) -> Vec<Step> {
    if start == 0 || start >= steps.len() {
        return steps;
    }
    let (before, from) = steps.split_at(start);
    from.iter()
        .copied()
        .chain(before.iter().map(|s| Step { degree: s.degree, octave: s.octave + 1 }))
        .collect()
}

/// The degree a token of an interval level asks for. 1, 3, 5 and 7 are the
/// chord's own, however it spells them - the diminished chord's seventh is
/// `bb7` - and anything else is looked for across its scale.
fn level_degree(token: &str, names: &[String]) -> Option<usize> {
    let own = &names[..names.len().min(4)];
    match token {
        // A sixth chord's fourth tone is its sixth: there "7" is that.
        "7" => scale_degree(token, own).or_else(|| own.iter().position(|n| n == "6")),
        "1" | "8" | "3" | "5" => scale_degree(token, own),
        _ => scale_degree(token, names),
    }
}

/// The degree a scale or interval token asks for.
///
/// An exact name wins wherever it sits, and is looked for across the whole set
/// before anything else. The loose rules answer "the third of this chord,
/// whatever it is" for a token the set spells differently - asked first, name
/// by name, they made the token `3` stop at `#2` in a scale holding both.
fn scale_degree(token: &str, names: &[String]) -> Option<usize> {
    names
        .iter()
        .position(|name| name == token)
        .or_else(|| names.iter().position(|name| loosely_names(token, name)))
}

/// Whether `name` is what `token` means, spelled another way.
fn loosely_names(token: &str, name: &str) -> bool {
    match token {
        "3" => name.contains('3') || name == "#2",
        "5" => name.contains('5') || name == "#4",
        "7" => name.contains('7') || name == "6" && name.contains("dim"),
        "2" | "9" => name.contains('2') || name.contains('9'),
        "4" | "11" => name.contains('4') || name.contains("11"),
        "6" | "13" => name.contains('6') || name.contains("13"),
        "b9" => name == "b9",
        "#9" => name == "#9",
        _ => false,
    }
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod fretboard_mode_tests;

#[cfg(test)]
mod generator_tests;

#[cfg(test)]
mod level_tests;

#[cfg(test)]
mod line_harness;

#[cfg(test)]
mod repeat_harness;

#[cfg(test)]
mod onset_sources;

