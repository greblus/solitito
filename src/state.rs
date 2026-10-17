use std::sync::{Arc, Mutex};
use std::collections::VecDeque; 
use crate::audio::AudioAnalysis;
use crate::arpeggio;
use crate::rng::Rng;
use crate::fretboard::Region;
use crate::model::{Step, split_octave, Chord, NoteName, Song, load_songs, load_all_scale_definitions, load_arpeggio_patterns, ScaleDefinition, ChordQuality};

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

/// How steady the single-frame estimate has to be: this many of the last
/// `EAR_WINDOW` ticks naming the class.
///
/// A vote, not a run. A run broke on any single stray frame and had to start
/// over, and at half a second to the note - which is not fast playing - the
/// estimate flickers between the note arriving and the one still ringing, so
/// the run was rarely reached before the next note began. Four of five
/// forgives one stray frame and still refuses a reading that cannot make up its
/// mind: an estimate alternating between two classes reaches two, never three.

/// Hops after the strike detector reports a class within which that class
/// counts as struck for "credit only what was struck": about 0.35 s.
///
/// Measured through the whole judge (`onset_sources`): on AtoA's 51 notes it
/// took the 57 false credits the onset head let through the model's branches
/// down to none, with no note missed and no later credit, and anywhere from 6
/// to 50 hops gave the same. A strummed chord pays for it: the detector
/// reports one class per strum, so 47 of 68 chord tones passed against 62 -
/// the rest wait for the ear.
const STRUCK_FRESH: u32 = 22;

/// Hops after the strike detector reports a strike during which the ear may
/// credit only a class the detector reported struck: about 0.25 s.
///
/// The FFT window is half a second wide, so just after an attack it holds the
/// note before and the new one together, and the estimate wanders - after C,
/// a D read as F for 13 hops before it settled. In Formulas that one reading
/// is a mark for the rest of the exercise. Measured on AtoA, the ear's false
/// credits went from 124 to 62 (82 to 36 one note at a time) with no note
/// missed or later; on four sessions of the user's playing, 388 credits, it
/// cost nothing; 16 hops did all that 22 and 30 did.
const EAR_SETTLE: u32 = 16;

/// In Formulas, hops back the strike detector must have reported a class for
/// the ear to credit it at all: 3 s.
///
/// As a note dies away the estimate can settle on another class - a fourth
/// above, the note whose third harmonic it is - and name it steadily from
/// about 1.8 s on. A mark in Formulas never expires, so there it takes a
/// strike. Measured with it everywhere: AtoA's false credits 62 -> 3, ringing
/// notes credited in a next chord with no strike 20 -> 12 of 113 (the rest
/// played, quietly). The cost kept it out of the other modes: 15 more of 388
/// session credits lost - quiet notes, legato ones, the low strings, which
/// the detector does not see - and a strummed chord's tones 46 -> 20 of 68.
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
///
/// Asking for a flux attack beside the strike as well was measured and dropped:
/// on AtoA's notes struck twice it took one false permission away and twelve
/// real re-strikes with it.
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

const EAR_VOTES: usize = 4;
const EAR_WINDOW: usize = 5;

/// How long the single-frame estimate has to have been naming a note before it
/// is credited, in audio frames of 16 ms - and how long a DIFFERENT note has to
/// have been named before the one credited counts as having stopped sounding.
///
/// Measured on a 20 s recording of single notes, at its own level and at 6, 12
/// and 18 dB below it, against what the estimate reads at full level. Per frame
/// of a note actually sounding, credited correctly / named as some other note:
///
/// | rule          | full   | -6 dB  | -12 dB | -18 dB |
/// |---------------|--------|--------|--------|--------|
/// | three frames  | 88/0 % | 86/0 % | 84/1 % | 80/1 % |
/// | two frames    | 93/0 % | 90/1 % | 87/2 % | 83/3 % |
/// | 2 of the last 5 | 91/8 % | 90/9 % | 89/9 % | 87/12 % |
///
/// Two frames is the whole of the gain: it credits five points more of what is
/// played at any level and still names almost nothing wrong. A vote over a
/// window - which is what Formulas uses, see `EAR_VOTES` - buys no more than
/// that and misnames one frame in ten, which is the "it credits notes I never
/// played" this mode had before.
///
/// What is NOT the constraint: the estimate's own score gate. On that recording
/// the lowest score of any reading was 0.52 against a `MONO_MIN_SCORE` of 0.50,
/// so lowering it would have changed nothing. What does bind, before any of
/// this, is the noise gate: at the recording's own level the default -34 dBFS
/// passes 78 % of frames to the analyser, 6 dB softer 36 %, and 12 dB softer
/// none at all.
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

/// What was true when a pitch class was last credited.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Credit {
    /// The envelope's attack count at that moment.
    pub onset: u64,
    /// Attacks the flux detector had counted, whatever their pitch.
    pub flux: u64,
    /// That class's own strike count from `strike::Strikes` at that moment.
    pub class_strike: u32,
    /// The octave marker of the step credited: 0 for the plain degree, 1 for
    /// one written `1'`. A step marked higher than the credit is asking for the
    /// same note in another octave, and that is answerable.
    pub octave: i8,
    /// What the single-frame estimate was reading, as an absolute semitone.
    pub semitone: Option<usize>,
    /// Seconds left before the note can count as having left - see
    /// `CREDIT_SETTLE`.
    pub settle: f32,
    /// Whether something else has been heard, steadily, since this note was
    /// credited. Then the note has stopped sounding and anything heard from it
    /// afterwards is a fresh pluck - whether or not a detector noticed.
    pub left: bool,
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
    pub arp_exercise: usize,
    /// Over the changes: which way each chord's arpeggio runs. 0 up, 1 down,
    /// 2 alternating from down, 3 alternating from up - the four the studies
    /// are written in.
    pub arp_direction: usize,
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
    /// How many passes have gone by, which is what the scale's string and
    /// direction are counted through. See `reroll`.
    pass: usize,
    /// The string an interval grip is taken from when the voices are not being
    /// led from the chord before. Drawn per chord - see `reroll`.
    pub grip_string: usize,
    /// Whether this pass of the scale runs downwards. Drawn with the string.
    pub scale_descending: bool,
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
    /// "Credit only what was struck": the model's branches may credit a class
    /// only while it counts as struck just now - see `struck_just_now`.
    ///
    /// The pitch head answers what SOUNDS: measured on a real recording it is
    /// right about that in 94% of frames and still leaves 78% of notes with
    /// some class lit that nobody played - an open string ringing in sympathy
    /// is sounding, and so is the note before. Off by default.
    ///
    /// Measured on a recording of single notes (`--probe`, 364 frames the app
    /// would ask about): of 83 credits the model would hand out for a class
    /// other than the one being played, two thirds carry no attack at all, and
    /// almost all of those are the PREVIOUS note still inside the model's
    /// 0.77 s window - which is what this tells apart. The notes themselves keep passing on the CQT branch, which this
    /// does not touch, so on that recording it costs nothing; an earlier
    /// measurement of the whole crediting rule put the cost at 4 missed notes
    /// in 49, so it is not free everywhere. Hence an option, not a default.
    pub require_onset: bool,
    /// Judge formulas the strict way - the steady single-frame estimate alone -
    /// instead of the rule the note modes use.
    ///
    /// On by default: the loose rule credits whatever is sounding, which after
    /// a few notes is most of the formula. `SOLITITO_FORMULA_STRICT=0` gives
    /// the note modes' rule back for a comparison.
    pub strict_formulas: bool,
    /// Audio frames pushed so far, and the last one this app has looked at.
    ///
    /// The judging is driven off THIS, not off the UI clock and not off the
    /// model's answers. Off the UI clock the same frame was sampled two or
    /// three times, which let one bad reading carry a vote; off the model's
    /// answers it stopped whenever the model did - and the model is asked only
    /// while the context window is nine tenths full, which playing one note at
    /// a time never manages. Once per new frame is once per new reading.
    audio_frames: u64,
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
    audio_gate_open: bool,
    /// Print a line for every function credited. The panel's own switch; the
    /// environment can force it either way while something is being chased.
    pub log_credits: bool,
    /// What the estimate has been saying, and for how many audio frames. A
    /// drawing that follows every frame flickers on noise; one that waits for a
    /// reading to hold does not. See `steady_note`.
    steady_pitch: Option<usize>,
    steady_for: u32,
    /// The same reading as an absolute semitone, for telling a note from itself
    /// an octave down. See `octave_is_answered`.
    pub cqt_semitone: Option<usize>,
    /// Pitch class sounding in the last CQT frame - see `audio::mono_pitch`.
    /// `None` while the gate is shut or the estimate is weak.
    pub cqt_pitch: Option<usize>,
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
    /// Attack counter, mirrored from the audio thread.
    pub onset_id: u64,
    /// Where the exercise writes down what it asked and what it credited, when
    /// SOLITITO_RECORD is set. None otherwise, which is every normal run.
    journal: Option<crate::journal::Journal>,
    /// The recording's hop clock, so a journal row is a position in the WAV.
    capture_hop: u64,
    /// Pitch classes genuinely sounding, one bit each, mirrored from the audio
    /// thread. See `voices::voices`.
    voices: u16,
    /// Whether `voices` has ever reported anything, so a build, a replay or a
    /// test without it falls back to the older reading of `left` instead of
    /// never setting it at all.
    voices_seen: bool,
    /// Frames each class has been absent from `voices` for.
    ///
    /// One frame's absence is not a note stopping. Measured on sesja-g1, only
    /// about half the credits had a second voice reported, so a note ringing
    /// under the one being played drops out of the list and comes back - and
    /// taking the first gap for silence is what let a note credited in one
    /// chord be credited again in the next.
    quiet_for: [u32; 12],
    /// Which of the four ways credited this, where the call site knew.
    credit_branch: Option<u8>,
    /// The (chord, step) pair last written down, to notice a new request.
    logged_ask: Option<(usize, usize)>,
    /// Attacks the flux detector has counted, mirrored from the audio thread.
    /// Pitch-blind: it says a string was hit, never which one.
    pub flux_id: u64,
    /// Hops since the last one.
    flux_age: u32,
    /// Strikes of each class, and hops since each - see `strike::Strikes`.
    class_strikes: [u32; 12],
    class_strike_age: [u32; 12],
    /// Whether that detector is running. Without its model file it is not, and
    /// the rules fall back to what they were before it existed.
    strikes_live: bool,
    /// Whether that detector has ever reported, so a build or a recording
    /// without it falls back to the older evidence instead of refusing
    /// everything.
    flux_seen: bool,
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
    pub credited: [Option<Credit>; 12],
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
    pub formula_exercise: u8,
    /// What kind of placement to draw: 0 any, 1 defines, 2 colours, 3 outside.
    pub formula_placement_want: u8,
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
            
            paused: false,
            random_mode: false,
            shuffle_chords: false,
            arp_exercise: 0,
            arp_direction: 0,
            arp_quality: 0,
            step_order: vec![],
            voicing_anchor: 5,
            scale_start: 0,
            scale_descending: false,
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
            audio_frames: 0,
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
            audio_gate_open: false,
            log_credits: false,
            steady_pitch: None,
            steady_for: 0,
            cqt_pitch: None,
            cqt_semitone: None,
            cqt_run_pitch: None,
            cqt_run: 0,
            ear_window: [None; EAR_WINDOW],
            lap_hold: 0.0,
            single_notes: false,
            onset_id: 0,
            journal: crate::journal::Journal::open(),
            capture_hop: 0,
            voices: 0,
            voices_seen: false,
            quiet_for: [0; 12],
            credit_branch: None,
            logged_ask: None,
            flux_id: 0,
            flux_age: u32::MAX,
            class_strikes: [0; 12],
            class_strike_age: [u32::MAX; 12],
            strikes_live: false,
            flux_seen: false,
            credited: [None; 12],
            chord_heard_at: ChordHeard::default(),
            formula_mask: 0,
            formula_exercise: 0,
            formula_placement_want: 0,
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
            0 => false,
            1 => true,
            2 => self.play_pos % 2 == 0,
            _ => self.play_pos % 2 == 1,
        }
    }

    pub fn get_active_indices(&self, chord: &Chord) -> Vec<Step> {
        // Over the changes there is no written phrase to read: the arpeggio is
        // built for the chord and for where it falls in the progression.
        if self.app_mode == AppMode::Arpeggios && self.arp_exercise == 1 {
            return self.changes_run(chord, self.changes_descending());
        }
        let all_names = chord.quality.interval_names(); 
        let user_tokens: Vec<&str> = self.intervals_input.split_whitespace().collect();
        let mut indices = Vec::new();
        
        if self.app_mode == AppMode::Arpeggios {
            for token in user_tokens {
                let (token, octave) = split_octave(token);
                let target_idx = match token {
                    "1" | "8" => 0,
                    "3" => 1,
                    "5" => 2,
                    "7" => 3,
                    "9" => if all_names.len() > 4 { 4 } else if all_names.len() > 1 { 1 } else { 0 },
                    _ => 999
                };
                
                if target_idx < all_names.len() {
                    indices.push(Step { degree: target_idx, octave });
                } else if token == "9" {
                    if let Some(pos) = all_names.iter().position(|n| n.contains("2") || n.contains("9")) {
                        indices.push(Step { degree: pos, octave });
                    }
                }
            }
        } else {
            for token in user_tokens {
                 let (token, octave) = split_octave(token);
                 // An exact name wins wherever it sits, and is looked for
                 // across the whole set before anything else. The loose rules
                 // below answer "the third of this chord, whatever it is" for a
                 // token the set spells differently - but they were being asked
                 // first, name by name, so in a scale holding both `#2` and `3`
                 // the token `3` stopped at `#2` and took it. Ten of the
                 // twenty-six built-in scales walked a step twice that way and
                 // never asked for the one it had swallowed: the altered scale
                 // read 1 b2 #2 #2 b5 #5 b7, and Bebop Dominant ended b7 b7.
                 if let Some(idx) = all_names.iter().position(|n| n == token) {
                     indices.push(Step { degree: idx, octave });
                     continue;
                 }
                 for (idx, name) in all_names.iter().enumerate() {
                    let is_match = if token == name { true } else {
                        match token {
                            "3" => name.contains("3") || name == "#2",
                            "5" => name.contains("5") || name == "#4",
                            "7" => name.contains("7") || name == "6" && name.contains("dim"), 
                            "2" | "9" => name.contains("2") || name.contains("9"),
                            "4" | "11" => name.contains("4") || name.contains("11"),
                            "6" | "13" => name.contains("6") || name.contains("13"),
                            "b9" => name == "b9", "#9" => name == "#9",
                            _ => false
                        }
                    };
                    if is_match { indices.push(Step { degree: idx, octave }); break; } 
                 }
            }
        }
        
        // A scale read as a scale ends where it started: the same note an octave
        // up. It is a step of its own, so it has to be played - and it is marked
        // an octave up, which is what puts the tick on the strip as `1'`.
        if self.app_mode == AppMode::Scales && self.scale_repeat_root && !indices.is_empty() {
            indices.push(Step { degree: indices[0].degree, octave: indices[0].octave + 1 });
        }

        // Down as often as up: a scale known in one direction is half known -
        // the fingers learn the climb and the ear never hears the descent. The
        // way is drawn with the string it starts from, once per pass.
        if self.app_mode == AppMode::Scales && self.scale_descending {
            indices.reverse();
        }

        if indices.is_empty() {
             if !all_names.is_empty() { vec![Step { degree: 0, octave: 0 }] } else { vec![] }
        } else {
            indices
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
        if matches!(self.app_mode, AppMode::Arpeggios | AppMode::Scales) {
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
    /// The chords around the current one in the order they are played, within
    /// this pass of the progression: up to `behind` already played, the
    /// current one, up to `ahead` still to come. Each with its distance from
    /// the current one, negative behind it.
    pub fn progress_window(&self, behind: usize, ahead: usize) -> Vec<(usize, i32)> {
        let pos = self.play_pos.min(self.play_order.len().saturating_sub(1));
        let from = pos.saturating_sub(behind);
        let to = (pos + ahead).min(self.play_order.len().saturating_sub(1));
        if self.play_order.is_empty() {
            return Vec::new();
        }
        (from..=to).map(|i| (self.play_order[i], i as i32 - pos as i32)).collect()
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
        if self.app_mode != AppMode::Arpeggios || self.arp_exercise != 0 {
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

    /// Draws a new order of tones and a new string hint. Called on every chord
    /// change - never per frame, or the target would move while playing.
    fn reroll(&mut self) {
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
        self.pass = self.pass.wrapping_add(1);
        self.scale_start = self.pass % START_STRINGS.len();
        self.scale_descending = self.pass % 2 == 1;
        // And which strings an interval grip is taken on when nothing is
        // leading it there: the bottom four can carry the lowest voice.
        self.grip_string = self.rng.below(4);
        self.step_order = (0..n).collect();
        if self.random_mode {
            self.rng.shuffle(&mut self.step_order);
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
        if self.app_mode == AppMode::Arpeggios && self.arp_exercise == 0 {
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
        if new_mode == AppMode::Arpeggios && self.arp_exercise == 0 {
            self.selected_library_idx =
                self.saved_arpeggio_index.min(self.arpeggio_patterns.len().saturating_sub(1));
            self.secondary_index = self.saved_arpeggio_key;
        }
        self.reload_library_content();
    }

    pub fn item_selected(&mut self, index: i32) {
        self.selected_library_idx = index as usize;
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
        self.app_mode == AppMode::Intervals && !self.interval_in_order
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
            AppMode::Arpeggios => self.arp_exercise == 1,
            AppMode::Formulas => self.formula_exercise == 2,
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
            if self.formula_exercise == 2 && self.selected_library_idx < self.song_library.len() {
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
                }
            }
            AppMode::Arpeggios if self.arp_exercise == 0 => {
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
        self.judged_frame = self.audio_frames;
        self.chord_heard_at = ChordHeard {
            onset_id: self.onset_id,
            class_strikes: self.class_strikes,
        };
    }

    /// What the ear hears, steadily - the state three audio frames of the same
    /// reading leave behind. Test seam.
    #[cfg(test)]
    fn hears(&mut self, pitch: Option<usize>) {
        self.cqt_pitch = pitch;
        self.steady_pitch = pitch;
        self.steady_for = if pitch.is_some() { 3 } else { 0 };
    }

    /// One audio frame's reading, as the sync below takes it. Test seam: the
    /// real path reads it from the shared state under a lock.
    #[cfg(test)]
    fn feed_estimate(&mut self, frame: u64, pitch: Option<usize>) {
        if frame != self.audio_frames {
            if pitch.is_some() && pitch == self.steady_pitch {
                self.steady_for = self.steady_for.saturating_add(1);
            } else {
                self.steady_pitch = pitch;
                self.steady_for = 1;
            }
        }
        self.cqt_pitch = pitch;
        self.audio_frames = frame;
    }

    /// What is sounding, once it has been saying so for long enough to believe.
    ///
    /// Three audio frames is 48 ms - short enough that an answer still feels
    /// immediate, long enough that a single stray reading from the room does
    /// not paint the neck.
    pub fn steady_note(&self) -> Option<usize> {
        self.held_for(LEFT_TICKS)
    }

    /// What is sounding, by the shorter measure a CREDIT is allowed to use.
    ///
    /// Two questions, two thresholds. "Has the target been sounding long enough
    /// to credit" costs one wrong credit when it is wrong; "has something ELSE
    /// taken over, so the note credited a moment ago has gone" is what stops a
    /// repeated note being credited twice off one pluck, and loosening that
    /// brings the repeats back. So only the first one is short.
    fn sounding_now(&self) -> Option<usize> {
        self.held_for(CREDIT_TICKS)
    }

    fn held_for(&self, ticks: u32) -> Option<usize> {
        (self.steady_for >= ticks).then_some(self.steady_pitch).flatten()
    }

    pub fn reset_logic_state(&mut self) {
        self.forget_what_was_heard();
        // What was credited outlives a chord or a lap - a run ending on the
        // note it starts from must not hand the next lap its first step off the
        // last one still ringing - but not a change of exercise, where nothing
        // is being continued.
        self.credited = [None; 12];
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

    /// Draws the next formula, and a key for it if the options ask for one.
    ///
    /// A filter that nothing satisfies leaves the previous formula standing
    /// rather than blanking the screen - the settings panel is where that gets
    /// explained, not here.
    pub fn next_formula(&mut self) {
        let key = if self.formula_random_key {
            None
        } else {
            crate::formulas::parse_key(&self.formula_key_setting)
        };
        if let Some(d) = crate::formulas::next(
            &mut self.rng,
            self.formula_notes,
            self.formula_required,
            key,
        ) {
            self.formula_mask = d.mask;
            self.formula_root = d.key.pitch() as usize;
            self.formula_key_name = d.key.name();
            self.formula_collected = vec![false; d.mask.count_ones() as usize];
            // Over a chord the drawn key is thrown away again: what the formula
            // is read against is the chord, and where its `1` sits is the
            // placement, not a key of its own.
            if self.formula_exercise != 0 {
                self.take_chord(true);
                self.place_over_chord();
            }
            // As after a finished lap: whatever is still ringing belongs to the
            // formula that has gone.
            self.lap_hold = LAP_HOLD;
            // Without this a log jumps key mid-way with nothing to say why.
            if std::env::var("SOLITITO_FORMULA").is_ok() {
                println!(
                    "--- nowa formula: {} w {}",
                    crate::formulas::to_text(self.formula_mask),
                    self.formula_key_name,
                );
            }
            self.success_timer = 0.0;
            self.match_status = MatchStatus::None;
        }
    }

    /// The chord under the formula, as a root and a set of pitch classes.
    fn chord_under(&self) -> Option<(usize, u16)> {
        let c = self.formula_chord.as_ref()?;
        let root = c.root as usize;
        let mut pcs = 0u16;
        for i in c.quality.intervals() {
            pcs |= 1 << ((root + i as usize) % 12);
        }
        Some((root, pcs))
    }

    /// A chord to practise over, drawn at random.
    ///
    /// The five qualities the app already knows, over any of the twelve roots.
    /// Drawn rather than chosen from a tune: this exercise is about hearing one
    /// formula land on one chord in twelve different ways, and a tune would
    /// keep taking the chord away before that had happened.
    fn draw_practice_chord(&mut self) -> Chord {
        const QUALITIES: [ChordQuality; 5] = [
            ChordQuality::Major7,
            ChordQuality::Dominant7,
            ChordQuality::Minor7,
            ChordQuality::HalfDiminished,
            ChordQuality::Diminished,
        ];
        Chord {
            root: NoteName::from_index(self.rng.below(12)),
            quality: QUALITIES[self.rng.below(QUALITIES.len())].clone(),
        }
    }

    /// Plants the standing formula on the standing chord.
    ///
    /// The formula does not change - only where its `1` is put, which is the
    /// whole of the exercise: the same set from the chord's own root spells the
    /// chord out, and from another degree it colours it or leaves it
    /// altogether. Everything downstream still works in
    /// terms of `formula_root`, so a placement is just a root arrived at a
    /// different way.
    pub fn place_over_chord(&mut self) {
        let Some((root, pcs)) = self.chord_under() else { return };
        let want = match self.formula_placement_want {
            1 => Some(crate::formulas::Verdict::Defines),
            2 => Some(crate::formulas::Verdict::Colours),
            3 => Some(crate::formulas::Verdict::Outside),
            _ => None,
        };
        let p = crate::formulas::draw_placement(
            &mut self.rng,
            self.formula_mask,
            root,
            pcs,
            want,
        );
        self.formula_degree = p.degree;
        self.formula_hits = p.hits;
        self.formula_verdict = Some(p.verdict);
        self.formula_root = (root + p.degree) % 12;
        self.formula_key_name = crate::formulas::KEY_POOL[self.formula_root].to_string();
        for done in self.formula_collected.iter_mut() {
            *done = false;
        }
        self.lap_hold = LAP_HOLD;
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
    }

    /// The chord the exercise is played over, drawn or taken from the tune.
    fn take_chord(&mut self, advance: bool) {
        match self.formula_exercise {
            1 => {
                if advance || self.formula_chord.is_none() {
                    self.formula_chord = Some(self.draw_practice_chord());
                }
            }
            2 => {
                if self.chords.is_empty() {
                    // No tune loaded yet: one drawn chord is better than none,
                    // and the tune takes over as soon as it arrives.
                    if self.formula_chord.is_none() {
                        self.formula_chord = Some(self.draw_practice_chord());
                    }
                    return;
                }
                if advance {
                    self.current_chord_index =
                        (self.current_chord_index + 1) % self.chords.len();
                }
                self.formula_chord = Some(self.chords[self.current_chord_index].clone());
            }
            _ => {
                self.formula_chord = None;
                self.formula_verdict = None;
            }
        }
    }

    /// The next chord of the tune, with a placement drawn on it.
    ///
    /// Asked for by hand, from the line under the pause: over the changes there
    /// are two things one might want next - the next chord, or a different
    /// formula to carry through the tune - and they are different gestures.
    pub fn next_change(&mut self) {
        if self.formula_exercise == 0 {
            return;
        }
        self.take_chord(true);
        self.place_over_chord();
    }

    /// Has the estimate been naming this class? See `EAR_VOTES`.
    fn ear_says(&self, pc: usize) -> bool {
        self.ear_window.iter().filter(|&&v| v == Some(pc % 12)).count() >= EAR_VOTES
    }

    /// UI clock: finished results remain visible even when audio stops.
    /// Intervals and Formulas also judge here, but only on a new audio frame;
    /// polling the same CQT sample twice must not count as two observations.
    pub fn tick(&mut self, dt: f32) {
        if self.app_mode == AppMode::Intervals {
            self.tick_intervals(dt);
            return;
        }
        // The fretboard trainer's show clock, for the same reason as the lap's
        // below: it must not stop when the player does.
        if self.app_mode == AppMode::Fretboard {
            if self.fret_hold > 0.0 && !self.paused {
                self.fret_hold = (self.fret_hold - dt).max(0.0);
                if self.fret_hold == 0.0 {
                    self.next_fret_target();
                }
            }
            return;
        }
        if self.app_mode != AppMode::Formulas {
            return;
        }
        if self.formula_collected.is_empty() {
            self.next_formula();
            return;
        }
        self.lap_hold = (self.lap_hold - dt).max(0.0);
        // One reading per new audio frame - no more, or the same frame votes
        // twice; no less, or the judging waits on the model.
        if self.audio_frames != self.judged_frame {
            self.judged_frame = self.audio_frames;
            self.ear_window.rotate_left(1);
            self.ear_window[EAR_WINDOW - 1] = self.cqt_pitch;
            if self.cqt_pitch.is_some() && self.cqt_pitch == self.cqt_run_pitch {
                self.cqt_run = self.cqt_run.saturating_add(1);
            } else {
                self.cqt_run_pitch = self.cqt_pitch;
                self.cqt_run = 1;
            }
            // Paused is "your turn": the whole set lit, nothing judged.
            if !self.paused {
                let done = self.collect_formula(self.last_ai_root, self.last_ai_conf);
                self.match_status = if done { MatchStatus::Exact } else { MatchStatus::None };
            }
        }
        if !self.formula_collected.iter().all(|&c| c) {
            return;
        }
        self.success_timer += dt;
        // Straight on after the pause. Waiting for the formula's own notes to
        // die away was tried, to stop a new lap filling itself in from their
        // decay - but that log predates the vote, and a decay wandering across
        // three classes cannot now hold any of them for four readings of five.
        // The wait was insurance against something already insured, and it was
        // paid for in the only currency this mode has: the seconds after a
        // formula, when the player is already playing the next one.
        let show = self.transition_delay.max(FORMULA_LAP_PAUSE);
        if self.success_timer > show && !self.paused {
            self.restart_formula();
        }
    }

    fn tick_intervals(&mut self, dt: f32) {
        self.interval_model_age += dt;
        if self.interval_model_age >= INTERVAL_MODEL_AGE {
                self.last_pitches = [0.0; 12];
            self.prev_pitches = [0.0; 12];
            self.last_ai_root = None;
            self.last_ai_conf = 0.0;
        }
        self.settle_credits(dt);

        self.interval_audio_age += dt;
        let fresh = self.audio_frames != self.judged_frame;
        let elapsed = self.interval_audio_age.min(0.032);
        if fresh {
            self.judged_frame = self.audio_frames;
            // A fresh sample after a stalled UI/audio stream does not prove
            // the note held throughout the gap.
            if self.interval_audio_age > 0.05 {
                self.answering_step = None;
                self.success_timer = 0.0;
                self.interval_silence = 0.0;
            }
            self.interval_audio_age = 0.0;
            if !self.audio_gate_open {
                self.interval_silence += elapsed;
                // Muting between plucks is evidence of a fresh note too. A
                // missing CQT estimate alone is NOT silence: require the gate.
                // Track this during the completed-set display as well.
                if self.interval_silence >= 0.2 {
                    for credit in self.credited.iter_mut().flatten() {
                        credit.left = true;
                        credit.settle = 0.0;
                    }
                }
            } else {
                self.interval_silence = 0.0;
            }
        }

        // The completed result has its own clock. It stays visible through
        // silence and pause, and cannot be overwritten by more predictions.
        if self.interval_hold > 0.0 {
            if !self.paused {
                self.interval_hold = (self.interval_hold - dt).max(0.0);
                if self.interval_hold == 0.0 {
                    self.advance_chord();
                }
            }
            return;
        }
        if !fresh {
            return;
        }
        if !self.audio_gate_open {
            self.answering_step = None;
            self.success_timer = 0.0;
            return;
        }
        self.check_progress(elapsed, self.last_ai_root, "", self.last_ai_conf);
    }

    /// Starts the standing formula again, exactly as it stands.
    ///
    /// Playing a formula through is not a reason to lose it, and not a reason to
    /// move it either: the exercise is the formula in that key, and the next lap
    /// is the same exercise. A different formula, or a different key for it,
    /// comes from the arrows or the settings.
    pub fn restart_formula(&mut self) {
        // Over a chord a finished lap is not a repeat: the exercise is the same
        // formula heard from somewhere else, so the placement moves. Over a
        // tune the chord moves too, and the formula is carried across it: the
        // constant is the set, and the harmony is what moves under it.
        if self.formula_exercise != 0 {
            self.take_chord(self.formula_exercise == 2);
            self.place_over_chord();
            return;
        }
        for done in self.formula_collected.iter_mut() {
            *done = false;
        }
        // The note that finished the lap is still ringing, and it would mark
        // itself off again the moment the marks cleared - the new lap would
        // start with one function given away. It counts from the next attack.
        self.lap_hold = LAP_HOLD;
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
    }

    /// Puts a kept formula back on screen, in whatever key is showing.
    ///
    /// No key travels with it: a formula is the same exercise in all twelve, and
    /// the key on screen is the one the options chose.
    pub fn load_formula(&mut self, mask: u16) {
        if mask == 0 {
            return;
        }
        self.formula_mask = mask;
        self.formula_collected = vec![false; mask.count_ones() as usize];
        self.lap_hold = LAP_HOLD;
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
    }

    /// Puts the standing formula in the key the options ask for.
    ///
    /// The functions do not move - only the notes they land on, so the marks
    /// start again. A key that cannot be read leaves the old one in place, as
    /// `next_formula` leaves the old formula when the filter matches nothing.
    pub fn rekey_formula(&mut self) {
        if self.formula_exercise != 0 {
            self.take_chord(false);
            self.place_over_chord();
            return;
        }
        let key = if self.formula_random_key {
            crate::formulas::draw_key(&mut self.rng)
        } else {
            crate::formulas::parse_key(&self.formula_key_setting)
        };
        if let Some(k) = key {
            self.formula_root = k.pitch() as usize;
            self.formula_key_name = k.name();
        }
        for done in self.formula_collected.iter_mut() {
            *done = false;
        }
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
    }

    /// Pitch class of each function in the current formula.
    pub fn formula_pitches(&self) -> Vec<usize> {
        crate::formulas::functions_of(self.formula_mask)
            .iter()
            .map(|&f| (self.formula_root + f) % 12)
            .collect()
    }

    /// Marks off whatever is sounding, and says whether the set is complete.
    ///
    /// Unordered on purpose: the exercise is to move around inside the set, so
    /// any function may be struck at any time. The formula is finished when
    /// every one of them has sounded.
    ///
    /// Three of the four ways in count here; the chord NAME does not. It says
    /// what the model made of 0.77 s and credits whatever function its root
    /// lands on, and a mark here never expires, so a name that named the wrong
    /// root would light a function for the rest of the exercise.
    ///
    /// The single-frame estimate was tried as the sole way in and read a plain
    /// E as Ab and Eb; the model's pitch head alone was tried next and took
    /// three or four seconds a note, its window being 0.77 s long. Both are
    /// needed, which is what `note_is_sounding` was built for.
    ///
    /// Two rules were tried and dropped, both recorded so they are not tried
    /// again: rationing credits to one per attack, which handed the credit to
    /// the note still ringing from before and left the note actually played
    /// nothing to claim; and crediting only near an attack, which lost the
    /// notes of anyone not picking hard enough to be heard as one.
    fn collect_formula(&mut self, ai_root: Option<NoteName>, confidence: f32) -> bool {
        let pitches = self.formula_pitches();
        let funcs = crate::formulas::functions_of(self.formula_mask);
        // What the ear reported and what was credited. On by default while the
        // mode is being tuned - it is the only way to tell a bad reading from a
        // bad rule, and it says one line per credit, not per frame.
        // SOLITITO_FORMULA=0 silences it.
        let loud = std::env::var("SOLITITO_FORMULA")
            .map(|v| v != "0")
            .unwrap_or(self.log_credits && !cfg!(test));
        // A new lap holds for a moment: what finished the last one is still
        // ringing, and the ear would hand it straight back.
        if self.lap_hold > 0.0 {
            return false;
        }
        // Measured on a real recording, over 49 notes (dist/latency_stats.py):
        //
        //   the three ways in together      110 false credits, 0 notes missed
        //   the steady estimate alone         33 false credits, 0 notes missed
        //   and with the take6 onset head     15 false credits, 4 notes missed
        //
        // So the model's pitch head is struck out here. It answers "what is
        // sounding", and a string ringing on - or one resonating in sympathy,
        // which is why the fourth and the fifth kept lighting up - is sounding
        // without having been played. It was 99 of those 110.
        //
        // The take6 onset head was tried as a gate here and dropped: live it
        // answered 0.09 for notes plainly being played, refusing far more than
        // it caught. The strike detector backs the ear instead - `EAR_BACKED`.
        // In order: only the lowest function not yet marked off may count. The
        // set is written low to high, so this is the same set read as a line -
        // and nothing else is refused loudly, it simply does not count yet.
        let next_due = self.formula_collected.iter().position(|&c| !c);
        for (i, &pc) in pitches.iter().enumerate() {
            if self.formula_collected[i] {
                continue;
            }
            if self.formula_in_order && next_due != Some(i) {
                continue;
            }
            let Some(branch) = self.sounding_by(pc, ai_root, confidence) else {
                continue;
            };
            // Strict: the single-frame estimate, and it has to have been saying
            // so.
            //
            // Loose (the default): the rule the note modes use, any of the four
            // ways in at once. It credited 110 things nobody played over 49
            // notes, against 33 for this one.
            if self.strict_formulas && (branch != 1 || !self.ear_says(pc)) {
                continue;
            }
            self.credit_class(pc, 0);
            self.formula_collected[i] = true;
            if loud {
                println!(
                    "atak#{} zaliczono {:<3} (sposob {branch})  ucho={:<3} model_prymy={:<3}",
                    self.onset_id,
                    funcs.get(i).map(|&f| crate::formulas::FUNCS[f]).unwrap_or("?"),
                    self.cqt_pitch
                        .map(|p| NoteName::from_index(p).to_string().to_string())
                        .unwrap_or_else(|| "-".into()),
                    ai_root
                        .map(|r| r.to_string().to_string())
                        .unwrap_or_else(|| "-".into()),
                );
            }
        }
        !self.formula_collected.is_empty() && self.formula_collected.iter().all(|&c| c)
    }

    /// Is the requested pitch class sounding right now?
    ///
    /// Shared by the note modes and the fretboard trainer so the two cannot
    /// drift apart. Four ways in, because no single one covers note practice:
    ///
    /// Measured on a scale at 0.6 s per note, the model's pitch head named the
    /// note actually being played in 7% of windows and the PREVIOUS one in 79% -
    /// not because it is wrong, but because it is asked about 0.77 s of audio and
    /// faithfully reports both notes it heard. The CQT estimate looks at one
    /// frame and got the current note 57% of the time, the previous one the rest,
    /// and never a note that was not played at all.
    fn note_is_sounding(&self, pc: usize, ai_root: Option<NoteName>, confidence: f32) -> bool {
        self.sounding_by(pc, ai_root, confidence).is_some()
    }

    /// The same question, answered with WHICH of the four ways in let the note
    /// through. Formulas need to know: their marks accumulate, so a branch that
    /// fires once too often is the difference between an exercise and a screen
    /// that fills itself in.
    fn sounding_by(&self, pc: usize, ai_root: Option<NoteName>, confidence: f32) -> Option<u8> {
        let target = pc % 12;

        // Played one at a time, the ear decides alone - and has to have been
        // saying so for three frames, not one.
        //
        // Measured over 49 notes of a real recording: the four ways below
        // together credited 110 things nobody played, the steady estimate alone
        // 33, and it missed nothing. Almost all of those 110 were the model's
        // pitch head answering about a note still ringing inside its 0.77 s
        // window - which is exactly what this option exists to refuse. Letting
        // the model through "unless the estimate names something else" was not
        // enough: it says nothing when the estimate is silent, and the root
        // head was not held back at all.
        // Scales and arpeggios are played one note at a time by definition -
        // nobody strums an arpeggio - so the rule is in force there whether the
        // option is ticked or not, and the checkbox is not offered.
        let one_at_a_time = self.one_at_a_time();
        if one_at_a_time {
            if self.sounding_now() == Some(target) && !self.ear_settling(target) && self.ear_backed(target) {
                return Some(1);
            }
            let ear = self.steady_note();
            // The ear naming ANOTHER note is a refusal; the ear saying nothing
            // is not. A note too quiet for the estimate's score gate - a low
            // string, the end of a phrase - would otherwise be uncreditable,
            // which is what made this harder to play than it had been. Where
            // the ear is silent the model may still answer, but only its pitch
            // head and only where an attack backs it: the chord NAME never
            // credits a note here, and that is where the notes nobody played
            // were coming from.
            if ear.is_some() || self.cqt_pitch.is_some() {
                return None;
            }
        }

        // 1. One frame, one note - no window to smear across.
        if self.cqt_pitch == Some(target) && !self.ear_settling(target) && self.ear_backed(target) {
            return Some(1);
        }

        let p_target = self.last_pitches[target];
        let p_max = self.last_pitches.iter().cloned().fold(0.0f32, f32::max);

        // 2. The model, where the target owns the window: a held note, or the
        //    only one in it.
        // Struck just now, or the class is only ringing - see `struck_just_now`.
        let struck = !self.require_onset || self.struck_just_now(target);
        if struck && p_target >= self.note_threshold && p_target >= p_max * 0.9 {
            return Some(2);
        }

        // 3. The model, where the target is the note that just ARRIVED in a
        //    window still holding the previous one. Being loudest is the wrong
        //    test there - the older note has had more of the window to itself,
        //    so it wins on level while the new one wins on rise. Only meaningful
        //    against a window that already held something: from an empty one
        //    every class has "risen", which is branch 2 wearing a disguise.
        let had_content = self.prev_pitches.iter().any(|&v| v > 0.2);
        let rise = p_target - self.prev_pitches[target];
        let best_rise = (0..12)
            .map(|i| self.last_pitches[i] - self.prev_pitches[i])
            .fold(f32::NEG_INFINITY, f32::max);
        if struck
            && had_content
            && p_target >= self.note_threshold
            && rise > 0.05
            && rise >= best_rise * 0.9
        {
            return Some(3);
        }

        // 4. The root head as independent confirmation - a single note is
        //    reported by the model as the root.
        //
        // Gated by the attack too. "Credit only what was struck" held back the
        // two pitch-head branches and left this one open, so with a microphone
        // in a room the model went on naming chords and this credited whatever
        // note their root happened to be - notes nobody had played.
        let by_root = struck
            && !one_at_a_time
            && matches!(ai_root, Some(r) if r == NoteName::from_index(target))
            && confidence >= self.chord_confidence;
        if by_root { Some(4) } else { None }
    }

    /// Whether the ear is still settling after a strike of some OTHER class -
    /// see `EAR_SETTLE`. Without the detector it never is.
    fn ear_settling(&self, pc: usize) -> bool {
        self.strikes_live
            && self.class_strike_age.iter().any(|&age| age <= EAR_SETTLE)
            && self.class_strike_age[pc] > EAR_SETTLE
    }

    /// Whether the ear may credit `pc` at all: outside Formulas, or without
    /// the detector, always; in Formulas only a class the detector reported
    /// struck - see `EAR_BACKED`.
    fn ear_backed(&self, pc: usize) -> bool {
        !self.strikes_live
            || self.app_mode != AppMode::Formulas
            || self.class_strike_age[pc] <= EAR_BACKED
    }

    /// Whether `pc` was struck just now, for "credit only what was struck".
    /// The strike detector where it runs - see `STRUCK_FRESH`. Without it only
    /// flux is left, which says that a string was hit but not which.
    fn struck_just_now(&self, pc: usize) -> bool {
        if self.strikes_live {
            return self.class_strike_age[pc] <= STRUCK_FRESH;
        }
        self.flux_seen && self.flux_age <= FLUX_FRESH
    }

    /// Draws a fresh practice region: a set of strings and a fret window.
    ///
    /// Called on entering the mode, then left alone - the region is meant to
    /// hold while you settle into one hand position. Only the note changes.
    /// A four-fret span is one finger per fret, index to little finger.
    pub fn randomize_region(&mut self) {
        const SPAN: u8 = 4;
        let strings = match self.rng.below(3) {
            0 => crate::fretboard::StringSet::All,
            1 => crate::fretboard::StringSet::LowThree,
            _ => crate::fretboard::StringSet::HighThree,
        };
        // Keep the whole window on the neck: last usable start is MAX_FRET-SPAN+1.
        let highest_start = crate::fretboard::MAX_FRET - SPAN + 1;
        self.region = Region {
            strings,
            fret_from: self.rng.below(highest_start as usize + 1) as u8,
            fret_span: SPAN,
        };
        self.fret_target = None;
        self.fret_hold = 0.0;
        self.next_fret_target();
    }

    /// Draws the next fretboard target. The region is untouched - it is fixed
    /// for the session and only changes when the player changes the settings.
    pub fn next_fret_target(&mut self) {
        if let Some(prev) = self.fret_target {
            self.recent_targets.push(prev);
        }
        // Three back. A region carries a handful of pitch classes, so avoiding
        // only the note just asked for brought the same few round in a circle -
        // and a note asked for again while it is still ringing has to be struck
        // afresh before it counts, which reads as the trainer sticking.
        while self.recent_targets.len() > 3 {
            self.recent_targets.remove(0);
        }
        self.fret_target = self.region.draw_avoiding(&mut self.rng, &self.recent_targets);
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
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
            self.settle_credits(dt);
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

    fn check_fretboard(&mut self, dt: f32, ai_root: Option<NoteName>, confidence: f32) {
        // The note just answered is still being shown. Nothing to judge
        // until the next one is drawn - and judging anyway would let the
        // string still ringing answer for it.
        if self.fret_hold > 0.0 {
            return;
        }
        let Some(target) = self.fret_target else {
            self.next_fret_target();
            return;
        };
        // The same rule as the note modes': a note drawn twice in a row has
        // to be struck twice. See `struck_since_credit`.
        let fresh = self.struck_since_credit(target);
        if fresh && self.note_is_sounding(target, ai_root, confidence) {
            self.success_timer += dt;
            self.match_status = MatchStatus::Exact;
        } else {
            self.success_timer = 0.0;
            self.match_status = MatchStatus::None;
        }
        // The same short hold the note modes use, not the chord one. A
        // chord is held while the ear settles on a NAME; a single note is
        // right the moment it is recognised, and asking it to hold for the
        // chord delay meant a plucked note decaying out of the estimate's
        // reach before the timer got there - it lit green, dropped back to
        // nothing, and the trainer never moved on.
        if !self.held_past(NOTE_HOLD) {
            return;
        }
        self.credit_class(target, 0);
        // Passed, but not moved on: the note stays where it is, green, for as
        // long as `FRET_SHOW` says. The clock for that runs in `tick`, every
        // frame, because this one is called only when the model answers - and
        // a hold timed on answers is a hold that stops when the player does.
        self.match_status = MatchStatus::Exact;
        self.success_timer = NOTE_HOLD;
        self.fret_hold = self.transition_delay.max(FRET_SHOW);
    }

    fn check_chord(&mut self, dt: f32, ai_root: Option<NoteName>, ai_qual: &str, confidence: f32) {
        let Some(target) = self.chords.get(self.current_chord_index) else {
            return;
        };
        if ai_qual == "Note" {
            self.match_status = MatchStatus::None;
            self.success_timer = (self.success_timer - dt * 2.0).max(0.0);
            return;
        }
        let reading = match ai_root {
            Some(root) => chord_reading(target, root, ai_qual, self.shells_only),
            None => ChordReading::Wrong,
        };
        // A chord asked for twice in a row has to be played twice. Without
        // this the ring of the pass before matches the moment the target is
        // set and the progression runs itself.
        let not_struck_again = reading != ChordReading::Wrong
            && self.chord_repeats(target)
            && !self.chord_struck_since();
        if not_struck_again {
            self.match_status = MatchStatus::None;
            self.success_timer = 0.0;
            return;
        }

        let is_weak_signal = confidence < self.chord_confidence;
        // Green means the target chord was heard, clearly enough to be
        // trusted - the same condition the bucket paints green on. With
        // this on, that is the whole test: no waiting for it to be held,
        // and no decay afterwards to undo it. Nothing runs away, because
        // advancing changes the target and the chord still ringing stops
        // matching it.
        if self.short_verdict && reading == ChordReading::Exact && !is_weak_signal {
            self.match_status = MatchStatus::Exact;
            if !self.paused {
                self.advance_chord();
            }
            return;
        }

        self.fill_chord_bucket(reading, is_weak_signal, dt);
        if self.held_past(self.transition_delay) {
            self.advance_chord();
        }
    }

    /// The leaky bucket: a clear reading of the chord fills the success timer
    /// at the rate of the clock, anything else drains it - a wrong chord the
    /// fastest.
    fn fill_chord_bucket(&mut self, reading: ChordReading, is_weak_signal: bool, dt: f32) {
        let (status, rate) = match (reading, is_weak_signal) {
            (ChordReading::Exact, false) => (MatchStatus::Exact, 1.0),
            (ChordReading::Exact, true) => (MatchStatus::Flicker, -1.0),
            (ChordReading::Partial, false) => (MatchStatus::Partial, 1.0),
            (ChordReading::Partial, true) => (MatchStatus::None, -2.0),
            (ChordReading::Wrong, _) => (MatchStatus::None, -4.0),
        };
        self.match_status = status;
        self.success_timer = (self.success_timer + dt * rate).max(0.0);
    }

    /// Intervals, Scales and Arpeggios: which step the sound answers, and
    /// whether it has answered long enough to be credited.
    ///
    /// These modes are NOT gated by chord confidence. `confidence` is the
    /// product of root and quality confidence, so an uncertain quality would
    /// block a note that is perfectly audible. Only the pitch head matters
    /// here - F1 0.90 against ~80% for the chord name.
    fn check_notes(&mut self, dt: f32, ai_root: Option<NoteName>, confidence: f32) {
        let Some(chord) = self.chords.get(self.current_chord_index) else {
            return;
        };
        let steps = self.ordered_active_indices(chord);
        let all_targets = chord.get_target_indices();
        let step_targets: Vec<usize> = steps.iter().map(|step| all_targets[step.degree]).collect();

        // The marks and the steps have to be the same length. They are
        // sized together when the exercise is set up, but a setting
        // that changes the STEPS - the closing root of a scale - can
        // land between two of those, and then the last step has no mark
        // to write. Resized rather than rebuilt: what has been played
        // this lap has been played.
        if self.collected_notes.len() != steps.len() {
            self.collected_notes.resize(steps.len(), false);
        }
        self.journal_ask();

        // Which step is being answered.
        //
        // A scale or an arpeggio IS an order - played out of it, it is
        // not that scale - so there the answer can only be the step due.
        // A grip is not: three notes under one hand, and which finger
        // lands first is not the exercise. So Intervals takes them in
        // any order unless the option asks otherwise, shuffled or not.
        //
        // One pluck still credits one step. Without that an exercise
        // asking for the same pitch class twice runs through both steps
        // on a single ringing string, with nothing played in between. A
        // step asking for the class just credited needs the string
        // struck again - see `struck_since_credit`.
        let answer = if self.free_order() {
            self.best_answer(&step_targets, ai_root, confidence)
        } else if self.current_note_step < steps.len() {
            let due = self.current_note_step;
            self.answer_for(step_targets[due], ai_root, confidence)
                .map(|by| (due, by))
        } else {
            return;
        };

        self.time_the_answer(answer.map(|(step, _)| step), dt);
        if !self.held_past(NOTE_HOLD) {
            return;
        }
        let Some((step, by)) = answer else {
            return;
        };
        self.credit_step(step, by, steps[step].octave, &step_targets);
    }

    /// Whether `target` answers for its step, and by which of the four ways -
    /// the lower the number the better the evidence, see `sounding_by`.
    fn answer_for(&self, target: usize, ai_root: Option<NoteName>, confidence: f32) -> Option<u8> {
        if self.relaxed_chords() {
            return self.sounding_by(target, ai_root, confidence);
        }
        if !self.struck_since_credit(target) {
            return None;
        }
        // The model's pitch head spreads a plucked note onto its
        // neighbours, so where the single-frame estimate names a
        // class, no OTHER class may be credited off the model that
        // frame - in both orders, since in order the root is
        // credited, the target moves to the third while the root
        // rings on, and a head trained on chord shapes offers it.
        //
        // HOW MANY voices, not which. Counting them is the part
        // of this witness that holds up; asking whether one
        // particular class is among them is the part with blind
        // spots - an octave cannot be resolved at all and a quiet
        // fifth is missed, so a strummed grip would come back
        // short and nothing but the note the ear names would
        // count. Two voices is already something one string cannot
        // be, and it is what the exception needs to know.
        //
        // The chord name used to answer this and must not again:
        // one plucked root is enough for the model to recognise
        // the shape, which is how a third nobody touched was
        // credited.
        let ear_names_another = self.cqt_pitch.is_some_and(|now| now != target % 12);
        let grip_is_sounding = self.several_sounding() && !self.single_notes;
        if ear_names_another && !grip_is_sounding {
            return None;
        }
        self.sounding_by(target, ai_root, confidence)
    }

    /// Under free order, the step that answers best among those still wanted.
    ///
    /// The BEST answer, not the first: taking the lowest-numbered step that
    /// answered credited whichever of them the model happened to name, in the
    /// order they were dealt. The estimate names one class only, so where it
    /// speaks it decides; below it, the class the pitch head is loudest on.
    fn best_answer(
        &self,
        step_targets: &[usize],
        ai_root: Option<NoteName>,
        confidence: f32,
    ) -> Option<(usize, u8)> {
        let louder_first = |a: usize, b: usize| {
            let level = |step: usize| self.last_pitches[step_targets[step] % 12];
            level(b).partial_cmp(&level(a)).unwrap_or(std::cmp::Ordering::Equal)
        };
        (0..step_targets.len())
            .filter(|&step| !self.is_collected(step))
            .filter_map(|step| {
                self.answer_for(step_targets[step], ai_root, confidence)
                    .map(|by| (step, by))
            })
            .min_by(|a, b| a.1.cmp(&b.1).then_with(|| louder_first(a.0, b.0)))
    }

    fn is_collected(&self, step: usize) -> bool {
        self.collected_notes.get(step).copied().unwrap_or(false)
    }

    /// Runs the success timer for the step answering. The count belongs to
    /// the step it started on: another step answering is a new answer, not a
    /// continuation of this one.
    fn time_the_answer(&mut self, answering: Option<usize>, dt: f32) {
        let Some(step) = answering else {
            self.answering_step = None;
            self.success_timer = 0.0;
            return;
        };
        if self.answering_step != Some(step) {
            self.answering_step = Some(step);
            self.success_timer = 0.0;
        }
        self.success_timer += dt;
    }

    /// Marks `step` played, and when it was the last one wanted, finishes the
    /// set: Intervals shows it complete, the other modes move to the next chord.
    fn credit_step(&mut self, step: usize, by: u8, octave: i8, step_targets: &[usize]) {
        if step < self.collected_notes.len() {
            self.collected_notes[step] = true;
        }
        self.credit_branch = Some(by);
        self.credit_class(step_targets[step], octave);
        self.answering_step = None;
        self.success_timer = 0.0;
        // Where the exercise stands: the first step still wanted. In order
        // that is the next one along, which is what it always was.
        let step_count = step_targets.len();
        self.current_note_step = (0..step_count)
            .find(|&i| !self.is_collected(i))
            .unwrap_or(step_count);
        if self.current_note_step < step_count {
            return;
        }
        if self.app_mode == AppMode::Intervals {
            self.match_status = MatchStatus::Exact;
            self.interval_hold = INTERVAL_SHOW;
        } else {
            self.advance_chord();
        }
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

    /// Where notes are played one at a time: Scales, Arpeggios and the
    /// Fretboard, and Intervals when the option says so. Everywhere else a
    /// strummed chord is allowed to walk its intervals off a single attack -
    /// see `a_strummed_chord_still_walks_its_intervals`.
    /// Several strings sounding at once, one of them the note being asked for.
    ///
    /// This is what the chord-name exception was reaching for and could not
    /// reach: a confidently recognised chord says the model knows the shape,
    /// not that its tones are in the air - and one plucked root is enough for
    /// it to recognise the shape. `voices` explains the spectrum instead of
    /// ranking it, so a class that is only the root's harmonic is not in it.
    ///
    /// Both halves. More than one voice alone would let any ringing pair carry
    /// a third nobody touched; the target being among them alone is branch 1
    /// wearing a disguise.
    /// Whether that class is sounding at all - not whether it is the loudest.
    fn voice_heard(&self, pc: usize) -> bool {
        self.voices & (1 << (pc % 12)) != 0
    }

    /// All three guards off: the exercise is being played for fun, with chords
    /// and whatever single tones of them come to hand.
    ///
    /// Then the model decides and nothing argues with it - no ear guard, no
    /// refusing a class because it was credited in the chord before. Those
    /// rules exist to make a TEST honest, and a player who has turned all of
    /// them off is not taking one. The strict path is the default and the
    /// measured one; this is the other thing the mode can be.
    fn relaxed_chords(&self) -> bool {
        self.app_mode == AppMode::Intervals
            && !self.single_notes
            && !self.require_onset
            && !self.interval_in_order
    }

    /// More than one string sounding - which one plucked note cannot be.
    ///
    /// Where nothing has ever been reported there is no witness, and the rule
    /// is as loose as it was before there was one.
    fn several_sounding(&self) -> bool {
        !self.voices_seen || self.voices.count_ones() >= 2
    }

    /// Whether notes are taken one at a time: by choice in Intervals and the
    /// fretboard trainer, always in Scales and Arpeggios - nobody strums one.
    ///
    /// The one definition. There used to be two: this one forced the rule on
    /// the fretboard trainer as well, while `sounding_by` kept a copy of its own
    /// that did not, so the trainer's repeat gate and its sounding check
    /// disagreed about which mode they were in.
    fn one_at_a_time(&self) -> bool {
        self.single_notes || matches!(self.app_mode, AppMode::Scales | AppMode::Arpeggios)
    }

    /// Whether a class asked for again has been played again.
    ///
    /// Two ways, and the measurements behind them:
    ///
    /// The estimate reads an absolute pitch, so a note sounding six semitones
    /// or more from where it was read when it was credited is a different
    /// string being played - the closing root of `1 2 3 4 5 6 7 1` against the
    /// opening one still ringing. That is proof on its own, and it needs no
    /// attack: on the test run the estimate read the opening root and the
    /// closing one an octave apart, 0.29 s after the string was hit. It cannot
    /// be a REQUIREMENT, though: a run closed in the octave it started in would
    /// never satisfy it, however many times it was played.
    ///
    /// Otherwise the class's own strike counter has to have moved - and, where
    /// notes are played one at a time, the estimate must not be reading some
    /// other note: a strike reported for the wrong class lands while the
    /// estimate reads the note actually being played, so it does not pass.
    ///
    /// Without the strike detector, flux - which says a string was hit, not
    /// which - stands in, and the envelope where flux has never reported.
    fn struck_since_credit(&self, pc: usize) -> bool {
        let pc = pc % 12;
        let Some(c) = self.credited[pc] else {
            return true;
        };
        // The note stopped sounding and CAME BACK: something else was heard
        // steadily since it was credited, and the single-frame estimate is
        // reading this note again now. Both halves matter - "something else was
        // heard" alone let a note credited a moment ago count a second time
        // while it merely rang on under the note being played, which is the
        // multiple-crediting this rule exists to stop. The estimate is
        // monophonic: it names what is loudest, so it names the ringing note
        // again only once the string is struck again.
        if c.left && self.steady_note() == Some(pc) {
            return true;
        }
        let elsewhere = self.cqt_pitch.is_some_and(|now| now != pc);
        // THIS class struck since its credit, and just now - the question the
        // rule always meant to ask, and the first evidence that can answer it.
        // A note played again an octave away is a strike of the same class and
        // counts through here; nothing else needs to look at octaves.
        //
        // Measured on every note of AtoA spliced with itself: of 58 permissions
        // given while the note was only ringing, 50 came from the octave branch
        // below - the estimate flips an octave on a decaying note and the rule
        // read the flip as a new pluck - and 8 from the pitch-blind attack.
        if self.strikes_live {
            return self.class_strikes[pc] != c.class_strike
                && self.class_strike_age[pc] <= CLASS_FRESH
                && !(self.one_at_a_time() && elsewhere);
        }
        // Without the detector: an octave jump is believed only with an attack
        // behind it, which is what made it a leak - see above.
        let fresh_attack = self.flux_seen && self.flux_id != c.flux && self.flux_age <= FLUX_FRESH;
        if let (Some(now), Some(then)) = (self.cqt_semitone, c.semitone) {
            if fresh_attack && now % 12 == pc && (now >= then + 6 || now + 6 <= then) {
                return true;
            }
        }
        // Pitch-blind, on purpose. The take6 onset head, asked WHICH class was
        // struck, answered 33 of AtoA's 51 notes right, 50 in the wrong class
        // and 18 twice.
        //
        // Whether a string was hit at all is a far easier question, and the
        // flux detector answers it without a class to get wrong: 47 of the same
        // 51 notes, 14 reports outside them, no pluck reported twice.
        //
        // WHICH note it was is then left to the monophonic estimate above,
        // which names the loudest class - a fundamental, not its harmonics.
        // JUST NOW, not merely since. The counter is pitch-blind by design, so
        // "an attack since the credit" is satisfied by any other step of the
        // exercise being played - and by the time a chord hands over, several
        // have been. That is how a note credited in one chord was credited
        // again in the next while it merely rang on: shuffle put a shared class
        // at the boundary, the counter had moved, and nothing else objected.
        //
        // A string struck again while it still rings is a different thing and
        // has to keep working, so this cannot ask for silence first - only for
        // the hit to be happening now.
        let struck = if self.flux_seen {
            self.flux_id != c.flux && self.flux_age <= FLUX_FRESH
        } else {
            self.onset_id != c.onset
        };
        // Not "the estimate names it" but "the estimate does not name something
        // else": a note too quiet for the estimate to score would otherwise
        // never be creditable a second time.
        struck && !(self.one_at_a_time() && elsewhere)
    }

    /// Counts down each credit's `settle`, and marks a note `left` once it has
    /// gone quiet - see `Credit::left`.
    fn settle_credits(&mut self, dt: f32) {
        // Steady, not any frame: the estimate flickers onto a ringing note for
        // a frame at a time, and a flicker is not the note stopping.
        let elsewhere = self.steady_note();
        // Read before the loop below borrows the credits.
        let quiet_for = self.quiet_for;
        let know_voices = self.voices_seen;
        for (c, credit) in self.credited.iter_mut().enumerate() {
            if let Some(cr) = credit {
                if cr.settle > 0.0 {
                    cr.settle = (cr.settle - dt).max(0.0);
                }
                // This note has gone quiet, and what is heard from it next is
                // a new pluck. Without it a note asked for again a few steps
                // later could not be credited at all when no detector caught
                // the pluck - the fretboard trainer stuck on a note that
                // was plainly right.
                //
                // "Something else is the loudest thing" is not that note going
                // quiet, and taking it for that is how a note credited in one
                // chord was credited again in the next: by the boundary the
                // other steps had been heard, so `left` was set, and the note
                // only had to still be ringing for the new chord to take it.
                // Shuffle made it show by putting a shared class at the
                // boundary, which a fixed progression may never do.
                //
                // So `voices` decides, which explains the spectrum rather than
                // ranking it: a class in it is sounding, loudest or not. Where
                // it says nothing at all - no audio yet - the older reading
                // stands, and the gate is no stricter than it was.
                if cr.settle <= 0.0
                    && elsewhere.is_some_and(|other| other != c)
                    && (!know_voices || quiet_for[c] >= GONE_QUIET)
                {
                    cr.left = true;
                }
            }
        }
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
            map.insert("hop".into(), self.capture_hop.into());
            map.insert(
                "t".into(),
                serde_json::json!(
                    self.capture_hop as f64 * crate::audio::HOP_LENGTH as f64
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

    fn credit_class(&mut self, pc: usize, octave: i8) {
        if self.journal.is_some() {
            // The three strongest pitch-head answers, so a rule can be tried
            // against this row later without playing anything again.
            let mut top: Vec<(usize, f32)> = (0..12).map(|i| (i, self.last_pitches[i])).collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let row = serde_json::json!({
                "pc": pc % 12,
                "octave": octave,
                "by": self.credit_branch,
                "ear": self.cqt_pitch,
                "semitone": self.cqt_semitone,
                "flux": self.flux_id,
                "flux_age": self.flux_age,
                "voices": (0..12).filter(|pc| self.voice_heard(*pc)).collect::<Vec<_>>(),
                "quiet_for": self.quiet_for[pc % 12],
                "class_strikes": self.class_strikes[pc % 12],
                "class_strike_age": self.class_strike_age[pc % 12],
                "strikes_live": self.strikes_live,
                "left": self.credited[pc % 12].map(|c| c.left),
                "onset": self.onset_id,
                "top": top.iter().take(3).map(|&(i, v)| serde_json::json!([i, v])).collect::<Vec<_>>(),
            });
            self.journal_row("credit", row);
        }
        self.credit_branch = None;
        self.credit_class_inner(pc, octave)
    }

    fn credit_class_inner(&mut self, pc: usize, octave: i8) {
        self.credited[pc % 12] = Some(Credit {
            onset: self.onset_id,
            flux: self.flux_id,
            class_strike: self.class_strikes[pc % 12],
            octave,
            semitone: self.cqt_semitone.filter(|n| n % 12 == pc % 12),
            settle: CREDIT_SETTLE,
            left: false,
        });
    }

    /// True when the chord now being asked for is the one just left behind -
    /// two of the same in a row in the song, or a progression of one chord.
    /// Its notes are still ringing, so matching proves nothing.
    fn chord_repeats(&self, target: &Chord) -> bool {
        match self.prev_chord_index {
            Some(p) => {
                p < self.chords.len()
                    && self.chords[p].root == target.root
                    && self.chords[p].quality == target.quality
            }
            None => false,
        }
    }

    /// Whether the strings have been hit since this chord became the target.
    ///
    /// The strike detector answers it with any strike at all: it reports a
    /// strum as one class, seldom two, and it does not fire on a chord that
    /// only rings. Measured on 17 strums of a real guitar (`onset_sources`):
    /// every strum seen and no ringing chord taken for one, where the take6
    /// onset head's two strikes on the chord's notes saw 9 strums and fired on
    /// all 17 chords while they rang. Without the detector, the envelope.
    fn chord_struck_since(&self) -> bool {
        let then = &self.chord_heard_at;
        if self.strikes_live {
            return self.class_strikes != then.class_strikes;
        }
        self.onset_id != then.onset_id
    }

    /// What is still sounding when the exercise moves on counts as already
    /// used in the next part of it: it needs a strike of its own there.
    ///
    /// The repeat rule only guarded classes credited before, so a note that
    /// rang out of the last chord WITHOUT being credited - a wrong note, an
    /// extra one - was free to answer the next chord the moment the ear named
    /// it. Requiring a strike for every first credit closes that too, but costs
    /// 7 to 11 of the 87 notes of the user's own session, which the detector
    /// does not see; this costs nothing where the note is played after it is
    /// asked for, which is every ordinary credit.
    ///
    /// A class struck in the last `JUST_STRUCK` hops is not marked: that is a
    /// player reaching the next chord early, while the finished one is shown.
    fn carry_over_what_rings(&mut self) {
        for pc in 0..12 {
            if self.credited[pc].is_some() {
                continue;
            }
            let sounding = self.voice_heard(pc) || self.steady_note() == Some(pc);
            let just_struck = self.strikes_live && self.class_strike_age[pc] <= JUST_STRUCK;
            if sounding && !just_struck {
                self.credit_class_inner(pc, 0);
            }
        }
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
            || (self.app_mode == AppMode::Arpeggios && self.arp_exercise == 0);
        if self.random_mode && key_exercise && !self.chords.is_empty() {
            let current = self.chords[0].root as usize;
            let next = self.rng.below_excluding(12, current);
            self.chords[0].root = NoteName::from_index(next);
            // Keep the key combo honest; otherwise it would show the key the
            // player picked while the app asks for a different one.
            self.secondary_index = next;
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

    pub fn sync_audio_settings(&mut self) {
        let mut gate_open = false;
        if let Ok(mut state) = self.analysis_state.lock() {
            state.noise_gate = self.noise_gate;
            state.bass_boost_enabled = self.bass_boost_enabled;
            state.bass_boost_gain = self.bass_boost_gain;
            // How long the single-frame estimate has been saying the same
            // thing. Counted for every mode, not only Formulas, because a
            // reading that has held for a few frames is the difference between
            // a note played and the room answering back - see the fretboard
            // drawing, which flickered on every stray reading before this.
            if state.frames_seen != self.audio_frames {
                if state.cqt_pitch.is_some() && state.cqt_pitch == self.steady_pitch {
                    self.steady_for = self.steady_for.saturating_add(1);
                } else {
                    self.steady_pitch = state.cqt_pitch;
                    self.steady_for = 1;
                }
            }
            self.cqt_pitch = state.cqt_pitch;
            self.cqt_semitone = state.cqt_semitone;
            self.onset_id = state.onset_id;
            if state.flux_id != self.flux_id {
                self.flux_seen = true;
            }
            self.flux_id = state.flux_id;
            self.flux_age = state.flux_age;
            self.class_strikes = state.class_strikes;
            self.class_strike_age = state.class_strike_age;
            self.strikes_live = state.strikes_live;
            self.capture_hop = state.hops;
            // On the audio clock, which is also the one the tests drive. A
            // closed noise gate stops it, and that case has its own rule: see
            // `interval_silence`.
            if state.voices != 0 {
                self.voices_seen = true;
            }
            if state.frames_seen != self.audio_frames {
                for pc in 0..12 {
                    if state.voices & (1 << pc) == 0 {
                        self.quiet_for[pc] = self.quiet_for[pc].saturating_add(1);
                    } else {
                        self.quiet_for[pc] = 0;
                    }
                }
            }
            self.voices = state.voices;
            self.audio_frames = state.frames_seen;
            gate_open = state.gate_open;
        }
        self.audio_gate_open = gate_open;
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

/// What the model's reading makes of the chord asked for.
#[derive(Clone, Copy, PartialEq, Debug)]
enum ChordReading {
    /// The chord itself, or a name that can only mean it.
    Exact,
    /// A substitution that leaves out or shares notes with it.
    Partial,
    Wrong,
}

/// How the model's chord - `root` and quality `heard` - reads against the
/// chord asked for.
fn chord_reading(target: &Chord, root: NoteName, heard: &str, shells_only: bool) -> ChordReading {
    let wanted = target.quality.to_string();
    if root == target.root {
        return same_root_reading(&wanted, heard, shells_only);
    }
    // Interval in semitones from the root asked for up to the root heard.
    let interval = (root as i32 - target.root as i32).rem_euclid(12);
    match (wanted.as_str(), heard) {
        // m7b5 read as m a minor third up, e.g. A m7b5 -> C m.
        ("m7b5", "m") if interval == 3 => ChordReading::Partial,
        // Maj7 read as m/m7 a major third up, e.g. C Maj7 -> E m.
        ("Maj7", "m") | ("Maj7", "m7") if interval == 4 => ChordReading::Partial,
        // A diminished seventh is four notes a minor third apart, so all four
        // are equally its root: C, Eb, Gb and A dim7 are one chord under four
        // names. Which name the model gives depends on the voicing, not on
        // what was played, so a reading three, six or nine semitones away is
        // the chord itself - green, not a substitution.
        ("dim", "dim") if interval % 3 == 0 => ChordReading::Exact,
        _ => ChordReading::Wrong,
    }
}

fn same_root_reading(wanted: &str, heard: &str, shells_only: bool) -> ChordReading {
    if heard == wanted {
        return ChordReading::Exact;
    }
    match (wanted, heard) {
        ("Maj7", "") | ("Maj7", "Maj") | ("m7", "m") | ("7", "") | ("m7b5", "dim") => {
            ChordReading::Partial
        }
        // A shell voicing of a m7b5 is root, third and seventh - the m7
        // shell, note for note, since the fifth is the only place the two
        // differ, and the shell leaves the fifth out. With shells the only
        // shapes on screen, that grip is what the app asked for, so playing
        // it is a pass and not a substitution: green. Yellow otherwise, where
        // the full shape is drawn with its flat fifth and a m7 means the fifth
        // was missed. Heard as plain m when the seventh dies away, exactly as
        // an ordinary m7 is.
        ("m7b5", "m7") | ("m7b5", "m") if shells_only => ChordReading::Exact,
        ("m7b5", "m7") | ("m7b5", "m") => ChordReading::Partial,
        _ => ChordReading::Wrong,
    }
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod fretboard_mode_tests;

#[cfg(test)]
mod generator_tests;

#[cfg(test)]
mod repeat_harness;

#[cfg(test)]
mod onset_sources;

