use std::sync::{Arc, Mutex};
use std::collections::VecDeque; 
use crate::audio::AudioAnalysis;
use crate::arpeggio;
use crate::rng::Rng;
use crate::fretboard::Region;
use crate::model::{Step, split_octave, Chord, NoteName, Song, load_songs, load_all_scale_definitions, load_arpeggio_patterns, ScaleDefinition, ChordQuality};

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
/// How much attack the onset head has to report for a class to count as struck,
/// when the option asking for one is on. Measured: at 0.02 two thirds of the
/// model's false credits go and the played notes keep passing.
const ONSET_MIN: f32 = 0.02;

/// A class counts as struck again when the head's answer for it rises past
/// `ONSET_AGAIN`, having been under `ONSET_LOW` since the last one counted.
///
/// Both numbers are measured, on the six re-plucks of `dist/latency_material.py`.
/// The head's answer does not settle after the spike - it wobbles - so at 0.40
/// one pluck was counted as more than one: seven strikes for six plucks. At
/// 0.60 it is exactly six for six. Higher costs recall: 0.80 sees only two.
///
/// Re-arming is relative, not absolute. On a single plucked string the answer
/// does fall to nothing between plucks, but under a strummed chord left ringing
/// it hovers - measured between 0.11 and 0.29 for a whole second - and an
/// absolute floor of 0.10 would never re-arm, so the next strum could not be
/// seen at all. A class is armed again once its answer drops below three tenths
/// of the peak that counted the last strike, and never above `ONSET_LOW`.
const ONSET_AGAIN: f32 = 0.60;
const ONSET_LOW: f32 = 0.10;
const ONSET_REARM: f32 = 0.3;

/// How many strikes on a chord's own notes count as strumming it again. One is
/// not enough: on the measured material a single class fired by itself while
/// the chord merely rang, and the repeat passed untouched. Two never did, and
/// a strum supplies two or more anyway - it caught all six re-strums, 0.20 to
/// 0.33 s after the strings were hit.
const CHORD_STRIKES: u32 = 2;

/// How long the attack head may still be answering about a pluck already
/// credited. Measured: its answer arrives 0.2 to 0.5 s after the string is hit,
/// which is after the estimate has named the note and the step has been
/// credited on it. Left alone, that late strike answers the NEXT step asking
/// for the same note - the closing root of a run handing the next lap its first
/// step, off nothing but its own ringing. While it lasts, and while the same
/// note is still the one the estimate reads, the credit keeps up with the
/// counter instead.
const STRIKE_SETTLE: f32 = 0.5;

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

/// What was true when a pitch class was last credited.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Credit {
    /// The envelope's attack count at that moment.
    pub onset: u64,
    /// That class's own strike count, from the attack head.
    pub strike: u32,
    /// The octave marker of the step credited: 0 for the plain degree, 1 for
    /// one written `1'`. A step marked higher than the credit is asking for the
    /// same note in another octave, and that is answerable.
    pub octave: i8,
    /// What the single-frame estimate was reading, as an absolute semitone.
    pub semitone: Option<usize>,
    /// Seconds left in which the head may still be answering about the pluck
    /// this credit was earned on - see `STRIKE_SETTLE`.
    pub settle: f32,
    /// Whether something else has been heard, steadily, since this note was
    /// credited. Then the note has stopped sounding and anything heard from it
    /// afterwards is a fresh pluck - whether or not the attack head noticed.
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
    /// Probability that each pitch class was STRUCK in the last few frames, from
    /// the model's onset head. All zeros with a model that has none.
    ///
    /// The distinction is the whole reason the head exists: measured on a real
    /// recording, the pitch head is right about what SOUNDS in 94% of frames and
    /// still leaves 78% of notes with some class lit that nobody played - an
    /// open string ringing in sympathy is sounding, and so is the note before.
    /// The model's two branches may only credit a class the onset head says was
    /// STRUCK. Off by default; the option reads "credit only what was struck".
    ///
    /// Measured on a recording of single notes (`--probe`, 364 frames the app
    /// would ask about): of 83 credits the model would hand out for a class
    /// other than the one being played, two thirds carry no attack at all, and
    /// almost all of those are the PREVIOUS note still inside the model's
    /// 0.77 s window - which is the very thing the head was trained to tell
    /// apart. The notes themselves keep passing on the CQT branch, which this
    /// does not touch, so on that recording it costs nothing; an earlier
    /// measurement of the whole crediting rule put the cost at 4 missed notes
    /// in 49, so it is not free everywhere. Hence an option, not a default.
    pub require_onset: bool,
    /// How many times each class has been STRUCK, by the model's onset head: a
    /// counter per class, stepped when the head's answer for it crosses upward.
    ///
    /// The envelope detector cannot do this job. Its level is the RMS of a
    /// 512 ms window, so a second pluck of a string already ringing barely
    /// moves it: measured on the test material it caught 2 of 6 re-plucks,
    /// where the head caught 6 of 6. What makes the head usable here and not as
    /// a credit gate is that the class is already known - the question is only
    /// "again?", not "which one".
    pub strike_id: [u32; 12],
    /// Whether a class has been quiet enough since its last counted strike for
    /// the next rise to count as a new one.
    onset_armed: [bool; 12],
    /// Whether the model in use answers about attacks at all. An older
    /// three-head model reports nothing but zeros, and everything that asks
    /// "was this struck again" then falls back to the envelope detector.
    onset_head_seen: bool,
    /// The answer that counted each class's last strike. Re-arming is measured
    /// against it - see `ONSET_REARM`.
    onset_peak: [f32; 12],
    pub last_onsets: [f32; 12],
    /// Ticks since the model last answered. The app asks it only once the
    /// context window is nine tenths full, which after a pause is 688 ms of
    /// playing - so between phrases this runs away and the head's answer is
    /// about a note that has already gone.
    onset_age: u32,
    /// Judge formulas the strict way - the steady single-frame estimate with
    /// the onset head vouching for it - instead of the rule the note modes use.
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
    chord_heard_at: ([u32; 12], u64),

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
            rng: Rng::default(),
            chord_history: VecDeque::with_capacity(20),
            last_pitches: [0.0; 12],
            prev_pitches: [0.0; 12],
            require_onset: false,
            strike_id: [0; 12],
            onset_armed: [true; 12],
            onset_head_seen: false,
            onset_peak: [1.0; 12],
            last_onsets: [0.0; 12],
            onset_age: u32::MAX,
            strict_formulas: std::env::var("SOLITITO_FORMULA_STRICT")
                .map(|v| v != "0")
                .unwrap_or(true),
            audio_frames: 0,
            judged_frame: u64::MAX,
            last_ai_root: None,
            last_ai_conf: 0.0,
            formula_in_order: false,
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
            credited: [None; 12],
            chord_heard_at: ([0; 12], 0),
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
    }

    /// What the strip shows as coming next - the real next entry in the order,
    /// not the chord one further along the song.
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
        self.last_onsets = [0.0; 12];
        self.onset_age = u32::MAX;
        self.ear_window = [None; EAR_WINDOW];
        self.cqt_run_pitch = None;
        self.cqt_run = 0;
        self.chord_heard_at = (self.strike_id, self.onset_id);
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

    /// The finished lap's clock, run every frame whether the model answers or
    /// not.
    ///
    /// Only this. Judging was moved here too and had to come back: the reading
    /// is per audio frame, the UI runs on its own clock, and sampling one from
    /// the other counted the same frame two or three times over - so a single
    /// bad reading could fill the vote by itself. Measured on the guitar it was
    /// both slower and less accurate, which settles it.
    ///
    /// The lap's clock is the exception, because it must not stop when the
    /// player does: the judge is called only when the model answers, and the
    /// model is asked only while the context window is nine tenths full, so a
    /// finished set stood there all green with no next one coming.
    pub fn tick(&mut self, dt: f32) {
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

    /// A fresh answer from the model's onset head.
    ///
    /// Set through this rather than by hand: the age is what tells a gate that
    /// knows something from one repeating an answer about a note long gone.
    pub fn set_onsets(&mut self, v: [f32; 12]) {
        // A strike is a crossing, not a level: the head's answer lingers for a
        // few hundred milliseconds after the pluck, so counting "above the
        // threshold" would call one strike several. It is latched rather than
        // compared with the previous frame, because the rise is gradual - 15,
        // 25, 36, 41 over four frames on the measured material - and by the time
        // it passes the threshold the frame before it is no longer low.
        if v.iter().any(|&x| x > 0.0) {
            self.onset_head_seen = true;
        }
        for c in 0..12 {
            if v[c] < (ONSET_REARM * self.onset_peak[c]).max(ONSET_LOW) {
                self.onset_armed[c] = true;
            } else if self.onset_armed[c] && v[c] >= ONSET_AGAIN {
                self.onset_armed[c] = false;
                self.onset_peak[c] = v[c];
                self.strike_id[c] = self.strike_id[c].wrapping_add(1);
            }
        }
        self.last_onsets = v;
        self.onset_age = 0;
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
        //   and with the onset head           15 false credits, 4 notes missed
        //
        // So the model's pitch head is struck out here. It answers "what is
        // sounding", and a string ringing on - or one resonating in sympathy,
        // which is why the fourth and the fifth kept lighting up - is sounding
        // without having been played. It was 99 of those 110.
        //
        // The onset head is not consulted. On a recording it looked the answer:
        // sure about WHEN at 202 ms against the other paths' 676. Live it
        // answered 0.09 for notes plainly being played, so as a gate it refused
        // far more than it caught. That was measured while the model was asked
        // only from a nine tenths full context window, which playing one note at
        // a time never filled; it is asked from half a window now, so the
        // reading is worth taking again before the head is written off. It is
        // still read and printed, against a recording at playing speed.
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
            // so. The onset head is NOT asked - live it answered 0.09 for notes
            // plainly being played, so as a gate it refused more than it caught.
            // It is still read and logged, to be measured against a recording
            // at playing speed rather than argued about.
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
        let one_at_a_time = self.single_notes
            || matches!(self.app_mode, AppMode::Scales | AppMode::Arpeggios);
        if one_at_a_time {
            if self.sounding_now() == Some(target) {
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
        if self.cqt_pitch == Some(target) {
            return Some(1);
        }

        let p_target = self.last_pitches[target];
        let p_max = self.last_pitches.iter().cloned().fold(0.0f32, f32::max);

        // 2. The model, where the target owns the window: a held note, or the
        //    only one in it.
        let stale = false;
        // The head is believed only while it has something to say: an answer
        // older than one frame is about a note that has already gone. The
        // threshold is low on purpose - what is separated here is "struck" from
        // "no attack at all", not loud from quiet.
        let struck = !self.require_onset
            || (self.onset_age <= 1 && self.last_onsets[target] >= ONSET_MIN);
        if !stale && struck && p_target >= self.note_threshold && p_target >= p_max * 0.9 {
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
        if !stale
            && struck
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
        self.onset_age = self.onset_age.saturating_add(1);
        self.settle_credits(dt);

        // Formulas have no song either: a drawn set of functions over a drawn
        // root, played in any order.
        if self.app_mode == AppMode::Formulas {
            // Only remember what was said. The judging happens once per audio
            // frame in `tick`, because this mode credits on a per-frame reading
            // and the model is asked only while the window is full - which
            // playing one note at a time never manages, so waiting here meant
            // waiting for an answer that often never came.
            let (ai_root, _) = self.parse_ai_prediction(ai_prediction);
            self.last_ai_root = ai_root;
            self.last_ai_conf = confidence;
            let _ = dt;
            return;
        }

        // The fretboard trainer has no song, so it runs before the chord guard.
        if self.app_mode == AppMode::Fretboard {
            let (ai_root, _) = self.parse_ai_prediction(ai_prediction);
            let Some(target) = self.fret_target else { self.next_fret_target(); return; };
            // The same rule as the note modes': a note drawn twice in a row has
            // to be struck twice. See `strike_id`.
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
            let hold = 0.12;
            if self.success_timer > hold {
                if self.paused {
                    self.success_timer = hold;
                } else {
                    self.credit_class(target, 0);
                    self.next_fret_target();
                }
            }
            return;
        }

        if self.chords.is_empty() { return; }
        
        let (ai_root, ai_qual) = self.parse_ai_prediction(ai_prediction);
        let target_chord = &self.chords[self.current_chord_index];
        let target_root = target_chord.root;

        let active_indices = self.ordered_active_indices(target_chord);
        let all_targets = target_chord.get_target_indices(); 

        match self.app_mode {
            // Both returned above, before the chord guard - neither has chords.
            AppMode::Fretboard | AppMode::Formulas => {}
            AppMode::Chords => {
                let target_qual_str = target_chord.quality.to_string();
                let mut exact_match = false;
                let mut partial_match = false;
                
                if ai_qual == "Note" {
                    self.match_status = MatchStatus::None;
                    self.success_timer = (self.success_timer - dt * 2.0).max(0.0);
                    return; 
                }

                let is_weak_signal = confidence < self.chord_confidence;

                if let Some(r) = ai_root {
                    // Interval in semitones between detected and target root
                    let root_diff = (r as i32 - target_root as i32).rem_euclid(12);

                    if r == target_root {
                        if ai_qual == target_qual_str { 
                            exact_match = true; 
                        } else {
                            match (target_qual_str.as_str(), ai_qual.as_str()) {
                                ("Maj7", "") | ("Maj7", "Maj") => partial_match = true, 
                                ("m7", "m") => partial_match = true,     
                                ("7", "") => partial_match = true,       
                                ("m7b5", "dim") => partial_match = true, 
                                // A shell voicing of a m7b5 is root, third and
                                // seventh - the m7 shell, note for note, since
                                // the fifth is the only place the two differ,
                                // and the shell leaves the fifth out. With
                                // shells the only shapes on screen, that grip is
                                // what the app asked for, so playing it is a
                                // pass and not a substitution: green. Yellow
                                // otherwise, where the full shape is drawn with
                                // its flat fifth and a m7 means the fifth was
                                // missed. Heard as plain m when the seventh dies
                                // away, exactly as an ordinary m7 is.
                                ("m7b5", "m7") | ("m7b5", "m") if self.shells_only => {
                                    exact_match = true
                                }
                                ("m7b5", "m7") | ("m7b5", "m") => partial_match = true,
                                _ => {}
                            }
                        }
                    } else {
                        // --- JAZZ LOGIC ---
                        // m7b5 read as m a minor third up, e.g. A m7b5 -> C m
                        if target_qual_str == "m7b5" && ai_qual == "m" && root_diff == 3 {
                            partial_match = true;
                        }
                        // Maj7 read as m/m7 a major third up, e.g. C Maj7 -> E m
                        if target_qual_str == "Maj7" && (ai_qual == "m" || ai_qual == "m7") && root_diff == 4 {
                            partial_match = true;
                        }
                        // A diminished seventh is four notes a minor third
                        // apart, so all four are equally its root: C, Eb, Gb and
                        // A dim7 are one chord under four names. Which name the
                        // model gives depends on the voicing, not on what was
                        // played, so a reading three, six or nine semitones away
                        // is the chord itself - green, not a substitution.
                        if target_qual_str == "dim" && ai_qual == "dim" && root_diff % 3 == 0 {
                            exact_match = true;
                        }
                    }
                }
                
                // Green means the target chord was heard, clearly enough to be
                // trusted - the same condition the bucket paints green on. With
                // this on, that is the whole test: no waiting for it to be held,
                // and no decay afterwards to undo it. Nothing runs away, because
                // advancing changes the target and the chord still ringing stops
                // matching it.
                // A chord asked for twice in a row has to be played twice.
                // Without this the ring of the pass before matches the moment
                // the target is set and the progression runs itself.
                if (exact_match || partial_match)
                    && self.chord_repeats(target_chord)
                    && !self.chord_struck_since(&all_targets)
                {
                    self.match_status = MatchStatus::None;
                    self.success_timer = 0.0;
                    return;
                }

                if self.short_verdict && exact_match && !is_weak_signal {
                    self.match_status = MatchStatus::Exact;
                    if !self.paused {
                        self.advance_chord();
                    }
                    return;
                }

                // LEAKY BUCKET LOGIC
                if exact_match {
                    if is_weak_signal {
                        self.match_status = MatchStatus::Flicker;
                        self.success_timer = (self.success_timer - dt).max(0.0);
                    } else {
                        self.match_status = MatchStatus::Exact;
                        self.success_timer += dt; 
                    }
                } else if partial_match {
                    if is_weak_signal {
                        self.match_status = MatchStatus::None;
                        self.success_timer = (self.success_timer - dt * 2.0).max(0.0);
                    } else {
                        self.match_status = MatchStatus::Partial;
                        self.success_timer += dt;
                    }
                } else {
                    self.match_status = MatchStatus::None;
                    self.success_timer = (self.success_timer - dt * 4.0).max(0.0);
                }
                
                if self.success_timer > self.transition_delay {
                    if self.paused {
                        // Hold the timer at the threshold: it would otherwise
                        // grow for as long as the pause lasts and the next chord
                        // would jump the moment play resumes.
                        self.success_timer = self.transition_delay;
                    } else {
                        self.advance_chord();
                    }
                }
            },
            
            AppMode::Intervals | AppMode::Scales | AppMode::Arpeggios => {
                // These modes are NOT gated by chord confidence. `confidence` is now
                // the product of root and quality confidence, so an uncertain quality
                // would block a note that is perfectly audible. Only the pitch head
                // matters here - F1 0.90 against ~80% for the chord name.
                if self.current_note_step >= active_indices.len() { return; }

                let internal_idx = active_indices[self.current_note_step].degree;
                let target_note_idx = all_targets[internal_idx];

                // Target pitch class (0..11); NoteName ordering matches the
                // pitch_logits output (C, Db, D, ...).
                // One pluck credits one step. Without this an arpeggio asking
                // for the same pitch class twice runs through both steps on a
                // single ringing string, with nothing played in between.
                // A step asking for the class just credited needs the string
                // struck again. NOT gated on "play the notes one at a time":
                // in a strummed chord the successive steps are DIFFERENT
                // classes, so this never fires there - it only ever refuses a
                // repeat of the same note off one pluck.
                let step_octave = active_indices[self.current_note_step].octave;
                let fresh = self.struck_since_credit(target_note_idx);
                let note_match =
                    fresh && self.note_is_sounding(target_note_idx, ai_root, confidence);

                if note_match { self.success_timer += dt; } else { self.success_timer = 0.0; }

                let note_delay = 0.12;
                if self.paused && self.success_timer > note_delay {
                    self.success_timer = note_delay;
                }
                if !self.paused && self.success_timer > note_delay {
                    if self.current_note_step < self.collected_notes.len() {
                        self.collected_notes[self.current_note_step] = true;
                    }
                    self.credit_class(target_note_idx, step_octave);
                    self.current_note_step += 1;
                    self.success_timer = 0.0; 

                    if self.current_note_step >= active_indices.len() {
                        self.advance_chord();
                    }
                }
            }
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
    fn one_at_a_time(&self) -> bool {
        self.single_notes
            || matches!(
                self.app_mode,
                AppMode::Scales | AppMode::Arpeggios | AppMode::Fretboard
            )
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
    /// other note. The second half is what the strike counter cannot supply: the
    /// attack head spreads an attack over notes nobody played, so the root
    /// collects strikes of its own while the degrees above it are played, two
    /// of them on the test run. Those strays land while the estimate is reading
    /// the note actually being played, so they no longer pass.
    ///
    /// The envelope's attack counter stands in for the strike counter only for
    /// a model with no attack head - the older three-head one.
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
        if let (Some(now), Some(then)) = (self.cqt_semitone, c.semitone) {
            if now % 12 == pc && (now >= then + 6 || now + 6 <= then) {
                return true;
            }
        }
        let struck = if self.onset_head_seen {
            self.strike_id[pc] != c.strike
        } else {
            self.onset_id != c.onset
        };
        // Not "the estimate names it" but "the estimate does not name something
        // else": a note too quiet for the estimate to score would otherwise
        // never be creditable a second time. This is the same test `sounding_by`
        // calls stale.
        let elsewhere = self.cqt_pitch.is_some_and(|now| now != pc);
        struck && !(self.one_at_a_time() && elsewhere)
    }

    /// See `STRIKE_SETTLE`: a strike still arriving from the pluck a note was
    /// credited on belongs to that credit, not to the next step asking for the
    /// same note.
    fn settle_credits(&mut self, dt: f32) {
        let (now, strikes) = (self.cqt_pitch, self.strike_id);
        // Steady, not any frame: the estimate flickers onto a ringing note for
        // a frame at a time, and a flicker is not the note stopping.
        let elsewhere = self.steady_note();
        for (c, credit) in self.credited.iter_mut().enumerate() {
            if let Some(cr) = credit {
                if cr.settle > 0.0 {
                    cr.settle = (cr.settle - dt).max(0.0);
                    if now == Some(c) {
                        cr.strike = strikes[c];
                    }
                }
                // Something else has been sounding since: this note has gone
                // quiet, and what is heard from it next is a new pluck. Without
                // this a note asked for again a few steps later could not be
                // credited at all when the attack head missed the pluck - the
                // fretboard trainer stuck on a note that was plainly right.
                if cr.settle <= 0.0 && elsewhere.is_some_and(|other| other != c) {
                    cr.left = true;
                }
            }
        }
    }

    fn credit_class(&mut self, pc: usize, octave: i8) {
        self.credited[pc % 12] = Some(Credit {
            onset: self.onset_id,
            strike: self.strike_id[pc % 12],
            octave,
            semitone: self.cqt_semitone.filter(|n| n % 12 == pc % 12),
            settle: STRIKE_SETTLE,
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

    /// Whether the strings have been hit since this chord became the target:
    /// either the envelope heard an attack, or the attack head counted
    /// `CHORD_STRIKES` of them on the chord's own notes. See `strike_id`.
    fn chord_struck_since(&self, classes: &[usize]) -> bool {
        let (strikes, onset) = self.chord_heard_at;
        if !self.onset_head_seen {
            return self.onset_id != onset;
        }
        let struck: u32 = classes
            .iter()
            .map(|c| self.strike_id[c % 12].wrapping_sub(strikes[c % 12]))
            .sum();
        struck >= CHORD_STRIKES
    }

    fn advance_chord(&mut self) {
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
        self.play_pos += 1;
        if self.play_pos >= self.play_order.len() {
            self.play_pos = 0;
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
            self.audio_frames = state.frames_seen;
            gate_open = state.gate_open;
        }
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::audio::{CTX_FRAMES, TOTAL_FEATURES};

    /// One note played and heard: the single-frame estimate holding still on it,
    /// which is the only way into a formula's marks.
    pub(crate) fn ear_hears(a: &mut MyApp, pc: usize) {
        ready(a);
        for _ in 0..CQT_STEADY_TICKS {
            ear_frame(a, Some(pc % 12));
        }
    }

    /// One reading, as one new audio frame carrying it.
    pub(crate) fn ear_frame(a: &mut MyApp, pc: Option<usize>) {
        a.cqt_pitch = pc;
        a.audio_frames += 1;
        a.tick(0.016);
    }

    /// Past the hold a new lap begins with - which real playing walks through
    /// without noticing, and a test would otherwise have to wait out.
    pub(crate) fn ready(a: &mut MyApp) {
        a.lap_hold = 0.0;
    }

    /// The ear letting go: answers arriving with nothing heard in them.
    pub(crate) fn ear_silent(a: &mut MyApp) {
        for _ in 0..EAR_WINDOW {
            ear_frame(a, None);
        }
    }

    /// Frames passing with nothing new from the model - which is what happens
    /// the moment the player stops playing.
    pub(crate) fn frames_pass(a: &mut MyApp, n: usize, dt: f32) {
        for _ in 0..n {
            a.tick(dt);
        }
    }

    /// MyApp needs the shared audio state; nothing here touches it.
    /// A credit as the tests write one: the two counters, nothing heard.
    fn credit(onset: u64, strike: u32) -> Credit {
        Credit { onset, strike, octave: 0, semitone: None, settle: 0.0, left: false }
    }

    pub(crate) fn app() -> MyApp {
        let analysis = Arc::new(Mutex::new(AudioAnalysis {
            cqt_semitone: None,
            input_history: [[0.0; TOTAL_FEATURES]; CTX_FRAMES],
            frame_live: [false; CTX_FRAMES],
            spectrum_visual: [0.0; 48],
            chroma_sum: [0.0; 12],
            bass_boost_enabled: false,
            bass_boost_gain: 1.0,
            noise_gate: 0.0,
            input_level: 0.0,
            onset_id: 0,
            frames_since_onset: 0,
            cqt_pitch: None,
            gate_open: false,
            frames_seen: 0,
        }));
        let mut a = MyApp::new(analysis);
        a.chords = vec![
            Chord { root: NoteName::C, quality: ChordQuality::Major7 },
            Chord { root: NoteName::D, quality: ChordQuality::Minor7 },
            Chord { root: NoteName::G, quality: ChordQuality::Dominant7 },
            Chord { root: NoteName::A, quality: ChordQuality::Minor7 },
        ];
        a.reset_logic_state();
        a
    }

    /// A standard is worked on for weeks and read from four modes. Losing it
    /// on the way to Scales and back was the app forgetting what the session
    /// was about.
    #[test]
    fn the_tune_follows_from_mode_to_mode() {
        let mut a = app();
        if a.song_library.len() < 2 {
            return;                       // a build with one tune has nothing to lose
        }
        a.set_mode(AppMode::Chords as i32);
        a.item_selected(1);
        let title = a.song_title.clone();
        a.set_mode(AppMode::Scales as i32);
        a.set_mode(AppMode::Intervals as i32);
        assert_eq!(a.selected_library_idx, 1, "the tune snapped back to the first row");
        assert_eq!(a.song_title, title);
        // Arpeggios over the changes reads the same library; the studies do not,
        // and their own row is remembered separately.
        a.arp_exercise = 1;
        a.set_mode(AppMode::Arpeggios as i32);
        assert_eq!(a.song_title, title, "over the changes it is the same tune");
    }

    /// One scale is worked on for days, and Scales opened on the first row of
    /// its library however long that had been going on.
    #[test]
    fn the_scale_is_kept_too() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        assert!(a.scale_definitions.len() > 2, "a library to choose from");
        a.item_selected(2);
        let name = a.song_title.clone();
        a.set_mode(AppMode::Chords as i32);
        a.set_mode(AppMode::Scales as i32);
        assert_eq!(a.selected_library_idx, 2, "back on the first row");
        assert_eq!(a.song_title, name);
        // And the name is what the settings hand back, not the row.
        let mut b = app();
        assert!(!b.select_scale("No Such Scale"));
        assert!(b.select_scale(&name));
        b.set_mode(AppMode::Scales as i32);
        assert_eq!(b.song_title, name);
    }

    /// What the settings hand back is a NAME, and the library is read from disk.
    #[test]
    fn a_tune_that_is_gone_leaves_the_choice_alone() {
        let mut a = app();
        a.set_mode(AppMode::Chords as i32);
        let before = a.song_title.clone();
        assert!(!a.select_song("Nothing Of The Sort"), "it cannot have been found");
        assert_eq!(a.song_title, before);
        assert_eq!(a.selected_library_idx, 0);
        let known = a.song_library.last().expect("a library").title.clone();
        assert!(a.select_song(&known));
        assert_eq!(a.song_title, known);
    }

    #[test]
    fn sequential_order_without_randomisation() {
        let mut a = app();
        a.set_random_mode(false);
        let seen: Vec<usize> = (0..4).map(|_| { a.advance_chord(); a.current_chord_index }).collect();
        assert_eq!(seen, vec![1, 2, 3, 0]);
    }

    /// The same chord twice in a row reads as the app having missed the first one.
    #[test]
    fn randomised_order_never_repeats_immediately() {
        let mut a = app();
        a.set_random_mode(true);
        let mut prev = a.current_chord_index;
        for _ in 0..300 {
            a.advance_chord();
            assert_ne!(a.current_chord_index, prev, "the same chord came up twice in a row");
            prev = a.current_chord_index;
        }
    }

    /// A single-chord list has no alternative - it must not loop forever or panic.
    #[test]
    fn randomisation_survives_a_one_chord_list() {
        let mut a = app();
        a.chords.truncate(1);
        a.reset_logic_state();
        a.set_random_mode(true);
        a.advance_chord();
        assert_eq!(a.current_chord_index, 0);
    }

    #[test]
    fn order_is_untouched_when_randomisation_is_off() {
        let mut a = app();
        a.set_random_mode(false);
        let chord = a.chords[a.current_chord_index].clone();
        assert_eq!(a.ordered_active_indices(&chord), a.get_active_indices(&chord));
    }

    /// Shuffling must not drop or duplicate a tone - every step still gets played.
    #[test]
    fn shuffled_steps_are_a_permutation() {
        let mut a = app();
        a.intervals_input = "1 3 5 7".to_string();
        a.set_random_mode(true);
        for _ in 0..50 {
            let chord = a.chords[a.current_chord_index].clone();
            let plain = a.get_active_indices(&chord);
            let mut ordered = a.ordered_active_indices(&chord);
            let mut expect = plain.clone();
            ordered.sort();
            expect.sort();
            assert_eq!(ordered, expect, "the shuffle changed the set of tones");
            a.advance_chord();
        }
    }

    /// Editing the interval list leaves a stale permutation behind; the length
    /// check must fall back to the plain order instead of indexing out of bounds.
    #[test]
    fn stale_permutation_falls_back_instead_of_panicking() {
        let mut a = app();
        a.intervals_input = "1 3 5 7".to_string();
        a.set_random_mode(true);
        a.intervals_input = "1 3".to_string();      // shorter, no reroll yet
        let chord = a.chords[a.current_chord_index].clone();
        assert_eq!(a.ordered_active_indices(&chord), a.get_active_indices(&chord));
    }

    /// Builds a Scales-mode app the way reload_library_content does.
    fn scales_app() -> MyApp {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        a
    }

    /// A scale is one "chord", so finishing it must move the KEY, not the index.
    #[test]
    fn scales_draw_a_new_key_on_every_pass() {
        let mut a = scales_app();
        assert!(!a.chords.is_empty(), "no scale loaded - the test would pass vacuously");
        a.set_random_mode(true);
        let mut keys = std::collections::HashSet::new();
        let mut prev = a.chords[0].root as usize;
        for _ in 0..200 {
            a.advance_chord();
            let now = a.chords[0].root as usize;
            assert_ne!(now, prev, "the same key came up twice in a row");
            keys.insert(now);
            prev = now;
        }
        assert!(keys.len() >= 10, "keys barely varied: {} of 12", keys.len());
    }

    /// The combo has to follow, or it would name a key the app is not asking for.
    #[test]
    fn key_combo_index_tracks_the_drawn_key() {
        let mut a = scales_app();
        assert!(!a.chords.is_empty(), "no scale loaded - the test would pass vacuously");
        a.set_random_mode(true);
        for _ in 0..50 {
            a.advance_chord();
            assert_eq!(a.secondary_index, a.chords[0].root as usize);
        }
    }

    #[test]
    fn scales_keep_their_key_without_randomisation() {
        let mut a = scales_app();
        assert!(!a.chords.is_empty(), "no scale loaded - the test would pass vacuously");
        a.set_random_mode(false);
        let key = a.chords[0].root as usize;
        for _ in 0..20 { a.advance_chord(); }
        assert_eq!(a.chords[0].root as usize, key, "the key moved with randomisation off");
    }

    /// Over the changes the second combo is not a key and must not be moved:
    /// the progression already says which chord comes next.
    #[test]
    fn the_changes_exercise_keeps_its_second_combo() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.arp_exercise = 1;
        a.set_random_mode(true);
        let sec = a.secondary_index;
        for _ in 0..20 { a.advance_chord(); }
        assert_eq!(a.secondary_index, sec, "something moved the combo behind our back");
    }

    /// A study has one chord and no progression, so a finished pass means a new
    /// key - the same as a finished scale, and the same switch decides it.
    #[test]
    fn a_study_draws_a_new_key_when_a_pass_ends() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.arp_exercise = 0;
        a.reload_library();
        assert!(!a.chords.is_empty(), "the study has no chord to stand on");
        a.set_random_mode(true);
        let key = a.chords[0].root as usize;
        let mut moved = false;
        for _ in 0..20 {
            a.advance_chord();
            if a.chords[0].root as usize != key {
                moved = true;
                break;
            }
        }
        assert!(moved, "the key never moved between passes");

        a.set_random_mode(false);
        let held = a.chords[0].root as usize;
        for _ in 0..20 { a.advance_chord(); }
        assert_eq!(a.chords[0].root as usize, held, "the key moved with randomisation off");
    }

    /// Pause must stop progression WITHOUT stopping the feedback - the whole
    /// point is to sit on one chord and keep seeing whether it is right.
    #[test]
    fn pause_holds_the_chord_but_keeps_scoring() {
        let mut a = app();
        // Straight to the field: set_mode would reload the library and replace
        // the test chords. The default mode is Fretboard, which never reaches
        // the chord branch at all.
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.paused = true;
        a.transition_delay = 0.05;
        let start = a.current_chord_index;
        for _ in 0..40 {
            a.check_progress_with_ai(0.1, "C Maj7", 0.99);
        }
        assert_eq!(a.current_chord_index, start, "paused, yet it moved on");
        assert_eq!(a.match_status, MatchStatus::Exact, "paused stopped the colours too");
    }

    /// A chord struck and released is right or wrong within a few frames of the
    /// attack. The hold timer cannot see that: the decay stops feeding it and it
    /// drains, so a correct chick never reaches the threshold.
    #[test]
    fn short_verdict_passes_a_chord_that_stops_ringing() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = true;
        a.transition_delay = 0.25;          // far longer than the strum lasts
        let start = a.current_chord_index;
        // One frame of a clean reading, then the sound dies away.
        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        assert_ne!(a.current_chord_index, start, "a clean strum did not count");
    }

    /// Nothing runs away when green advances at once: moving on changes the
    /// target, and the chord still ringing no longer matches it. So a chord held
    /// down does not walk through the whole song.
    #[test]
    fn a_ringing_chord_does_not_walk_through_the_song() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = true;
        a.transition_delay = 0.25;
        let start = a.current_chord_index;
        for _ in 0..40 {
            a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        }
        assert_eq!(a.current_chord_index, (start + 1) % a.chords.len(),
                   "one chord ringing moved the song on more than once");
    }

    /// Two of the same chord in a row: the first pass is earned, the second has
    /// to be strummed again. The ring of the first would otherwise match the
    /// target the instant it is set.
    #[test]
    fn the_same_chord_twice_has_to_be_played_twice() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = true;
        a.transition_delay = 0.25;
        a.chords = vec![
            Chord { root: NoteName::C, quality: ChordQuality::Major7 },
            Chord { root: NoteName::C, quality: ChordQuality::Major7 },
        ];
        a.reset_logic_state();

        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        assert_eq!(a.current_chord_index, 1, "the first strum did not pass");

        // The same chord keeps ringing, unplayed.
        for _ in 0..40 {
            a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        }
        assert_eq!(a.current_chord_index, 1, "the ring passed the repeat by itself");

        // Struck again: the attack head answers for the chord's own notes.
        let mut v = [0.0; 12];
        for c in [0, 4, 7, 11] { v[c] = 0.9; }     // C E G B
        a.set_onsets(v);
        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        assert_ne!(a.current_chord_index, 1, "a real second strum was refused");
    }

    /// One note of the chord speaking up is not a strum. Measured on the
    /// generated strums, a single class fired by itself while a chord merely
    /// rang on; two never did.
    #[test]
    fn one_note_speaking_up_is_not_a_strum() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = true;
        a.transition_delay = 0.25;
        a.chords = vec![
            Chord { root: NoteName::C, quality: ChordQuality::Major7 },
            Chord { root: NoteName::C, quality: ChordQuality::Major7 },
        ];
        a.reset_logic_state();
        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        let after_first = a.current_chord_index;
        assert_eq!(after_first, 1, "the first strum did not pass");

        let mut one = [0.0; 12];
        one[4] = 0.9;                              // E alone
        a.set_onsets(one);
        for _ in 0..20 { a.check_progress_with_ai(0.02, "C Maj7", 0.99); }
        assert_eq!(a.current_chord_index, after_first, "one note passed as a strum");

        let mut two = [0.0; 12];
        two[0] = 0.05; two[4] = 0.05; two[7] = 0.05;
        a.set_onsets(two);                         // everything falls back
        two[0] = 0.9; two[7] = 0.9;
        a.set_onsets(two);
        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        assert_ne!(a.current_chord_index, after_first, "two notes struck did not pass");
    }

    /// The answer for a class under a ringing chord hovers instead of falling
    /// to nothing, so re-arming is measured against the strike before it.
    #[test]
    fn a_strike_counts_over_a_ringing_chord() {
        let mut a = app();
        let pc = 0;
        let mut v = [0.0; 12];
        v[pc] = 0.9;
        a.set_onsets(v);                           // struck
        let first = a.strike_id[pc];
        // It never falls below 0.10 again - the chord is still ringing.
        for level in [0.40, 0.25, 0.20, 0.25] {
            v[pc] = level;
            a.set_onsets(v);
        }
        assert_eq!(a.strike_id[pc], first, "the hover counted as a strike");
        v[pc] = 0.85;                              // struck again
        a.set_onsets(v);
        assert_eq!(a.strike_id[pc], first + 1, "the second strike was missed");
    }

    /// A m7b5 has no shell of its own: the shell is root, seventh and third,
    /// and the flat fifth is the one note it leaves out. With shells the only
    /// shapes on screen, the m7 shell IS the shape asked for.
    #[test]
    fn the_shell_of_a_half_diminished_passes_as_itself() {
        for (shells_only, want) in [(true, MatchStatus::Exact), (false, MatchStatus::Partial)] {
            let mut a = app();
            a.app_mode = AppMode::Chords;
            a.set_random_mode(false);
            a.short_verdict = false;
            a.shells_only = shells_only;
            a.chords = vec![Chord { root: NoteName::A, quality: ChordQuality::HalfDiminished }];
            a.reset_logic_state();
            for _ in 0..40 { a.check_progress_with_ai(0.02, "A m7", 0.99); }
            assert_eq!(a.prev_status(), want, "shells_only = {shells_only}");
        }
    }

    /// The strike has to be one of the chord's own notes. Something else being
    /// plucked nearby does not pass the repeat.
    #[test]
    fn a_strike_elsewhere_does_not_pass_the_repeat() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = true;
        a.transition_delay = 0.25;
        a.chords = vec![Chord { root: NoteName::C, quality: ChordQuality::Major7 }];
        a.reset_logic_state();

        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        let after_first = a.current_chord_index;

        let mut v = [0.0; 12];
        v[1] = 0.9;                                 // Db, in no C Maj7
        a.set_onsets(v);
        let onset = a.onset_id;                     // and the envelope heard nothing
        a.onset_id = onset;
        for _ in 0..40 {
            a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        }
        assert_eq!(a.current_chord_index, after_first, "a stray note passed the chord");
    }

    /// Passing has to be visible, and on the right chord. Advancing changes
    /// which chord is current, so the green belongs on the previous one - the
    /// one that earned it. Lighting the current chord marked the one that had
    /// not been played yet.
    #[test]
    fn a_pass_lights_the_chord_that_earned_it() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = true;
        a.transition_delay = 0.25;
        let played = a.current_chord_index;

        a.check_progress_with_ai(0.02, "C Maj7", 0.99);

        assert_eq!(a.prev_status(), MatchStatus::Exact, "the pass was not shown at all");
        assert_eq!(a.prev_chord_index, Some(played), "the wrong chord was lit");
        assert_ne!(a.current_chord_index, played, "it did not move on");
    }

    /// The green stays put. It marked a pass for a fifth of a second before,
    /// which was too brief to catch; there is nothing to time out, because the
    /// next pass moves the mark along by itself.
    #[test]
    fn the_green_stays_until_the_next_pass_moves_it() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = true;
        a.transition_delay = 0.25;

        let first = a.current_chord_index;
        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        assert_eq!(a.prev_chord_index, Some(first));

        // Whatever comes next - silence, a wrong chord, a long wait - the mark
        // stays on the chord that earned it.
        for _ in 0..40 {
            a.check_progress_with_ai(0.05, "F# m", 0.10);
        }
        assert_eq!(a.prev_status(), MatchStatus::Exact, "the mark disappeared on its own");
        assert_eq!(a.prev_chord_index, Some(first), "the mark wandered off");

        // Passing the chord now current moves it along, and only then.
        let second = a.current_chord_index;
        let name = a.chord_label(second);
        let (root, qual) = name.split_at(name.find(' ').unwrap_or(name.len()));
        a.check_progress_with_ai(0.02, &format!("{root}{qual}"), 0.99);
        assert_eq!(a.prev_chord_index, Some(second), "the next pass did not move the mark");
        assert_eq!(a.prev_status(), MatchStatus::Exact);
    }

    /// A chord passed on a triad or a substitution went yellow on the way
    /// through. Reporting it green afterwards would be a kinder account than
    /// what actually happened.
    #[test]
    fn the_mark_says_how_the_chord_was_passed() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = false;          // the hold timer, so a partial can pass
        a.transition_delay = 0.05;
        let played = a.current_chord_index;

        // A plain triad against a Maj7 target: yellow, and enough to pass.
        for _ in 0..10 {
            a.check_progress_with_ai(0.02, "C", 0.99);
        }

        assert_ne!(a.current_chord_index, played, "the triad never passed");
        assert_eq!(a.prev_chord_index, Some(played));
        assert_eq!(a.prev_status(), MatchStatus::Partial,
                   "a chord passed on a triad was reported as an exact match");
    }

    /// The same four notes have four names. Which one comes back depends on
    /// the voicing, so the app cannot insist on one of them.
    #[test]
    fn a_diminished_seventh_is_the_same_chord_from_all_four_roots() {
        for named in ["Eb dim", "Gb dim", "A dim"] {
            let mut a = app();
            a.app_mode = AppMode::Chords;
            a.set_random_mode(false);
            a.short_verdict = false;
            a.transition_delay = 0.05;
            a.chords = vec![Chord { root: NoteName::C, quality: ChordQuality::Diminished }];
            a.reset_logic_state();

            for _ in 0..10 {
                a.check_progress_with_ai(0.02, named, 0.99);
            }
            assert_eq!(
                a.prev_status(),
                MatchStatus::Exact,
                "{named} is C dim7 played from another of its notes"
            );
        }

        // A minor third is the interval that makes them one chord; a semitone
        // away is a different chord and stays wrong.
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = false;
        a.transition_delay = 0.05;
        a.chords = vec![Chord { root: NoteName::C, quality: ChordQuality::Diminished }];
        a.reset_logic_state();
        for _ in 0..10 {
            a.check_progress_with_ai(0.02, "Db dim", 0.99);
        }
        assert_ne!(a.prev_status(), MatchStatus::Exact, "Db dim7 is not C dim7");
    }

    /// A shell voicing of a m7b5 is the m7 shell, note for note - the fifth is
    /// the only place the two chords differ, and a shell has no fifth. The app
    /// draws that shape in shell mode, so it has to accept hearing it.
    #[test]
    fn a_half_diminished_passes_on_its_shell() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = false;
        a.transition_delay = 0.05;
        a.chords = vec![Chord { root: NoteName::D, quality: ChordQuality::HalfDiminished }];
        a.reset_logic_state();

        for _ in 0..10 {
            a.check_progress_with_ai(0.02, "D m7", 0.99);
        }
        assert_eq!(
            a.prev_status(),
            MatchStatus::Partial,
            "the shell the app itself draws was not accepted"
        );

        // And it is not mistaken for the chord: green is still only the chord.
        let mut b = app();
        b.app_mode = AppMode::Chords;
        b.set_random_mode(false);
        b.short_verdict = false;
        b.transition_delay = 0.05;
        b.chords = vec![Chord { root: NoteName::D, quality: ChordQuality::HalfDiminished }];
        b.reset_logic_state();
        for _ in 0..10 {
            b.check_progress_with_ai(0.02, "D m7b5", 0.99);
        }
        assert_eq!(b.prev_status(), MatchStatus::Exact);
    }

    /// Shuffle used to draw a chord at each step, so there was no "next" to
    /// show and stepping back landed anywhere. The order is now laid out when
    /// the toggle goes on, and a lap visits every chord exactly once.
    #[test]
    fn shuffle_lays_out_the_whole_song_up_front() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(true);
        let len = a.chords.len();

        let mut seen = vec![a.current_chord_index];
        for _ in 1..len {
            a.advance_chord();
            seen.push(a.current_chord_index);
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), len, "a lap did not cover every chord exactly once");
    }

    /// Turning it off goes back to the written order, from the top.
    #[test]
    fn turning_shuffle_off_restarts_the_song() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(true);
        a.advance_chord();
        a.advance_chord();

        a.set_random_mode(false);

        assert_eq!(a.current_chord_index, 0, "it did not start the song again");
        a.advance_chord();
        assert_eq!(a.current_chord_index, 1, "it is not walking the written order");
    }

    /// What the strip promises is what actually arrives - in shuffled order the
    /// chord after this one is not the next one in the song.
    #[test]
    fn the_strip_promises_the_chord_that_arrives() {
        for random in [false, true] {
            let mut a = app();
            a.app_mode = AppMode::Chords;
            a.set_random_mode(random);
            for _ in 0..(a.chords.len() + 2) {
                let promised = a.next_chord_index();
                a.advance_chord();
                assert_eq!(a.current_chord_index, promised,
                           "random={random}: the strip showed a chord that did not come");
            }
        }
    }

    /// And stepping back while paused returns to the chord actually played
    /// before, not the one written before it in the song.
    #[test]
    fn stepping_back_follows_the_order_in_force() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(true);
        let first = a.current_chord_index;
        a.advance_chord();
        assert_ne!(a.current_chord_index, first);

        a.step_chord(-1);

        assert_eq!(a.current_chord_index, first, "stepping back left the shuffled order");
    }

    /// Stepping back to a chord that has gone by is for practising it again -
    /// it must not look or count as though it had just been played.
    #[test]
    fn stepping_back_is_not_a_pass() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = true;
        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        let after_pass = a.current_chord_index;

        a.step_chord(-1);

        assert_ne!(a.current_chord_index, after_pass, "stepping back did nothing");
        assert_eq!(a.prev_status(), MatchStatus::None, "stepping back lit a chord as passed");
        assert_eq!(a.match_status, MatchStatus::None);
        assert_eq!(a.success_timer, 0.0, "stepping banked progress");
    }

    /// Stepping by hand has to draw a new string to start from, like arriving
    /// by playing does. It did not, so shuffle plus pause plus the arrows left
    /// the suggestion frozen on whichever string it happened to be.
    #[test]
    fn stepping_draws_a_new_string_hint() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(true);

        let mut seen = std::collections::HashSet::new();
        for _ in 0..12 {
            a.step_chord(1);
            seen.insert(a.start_hint);
        }
        assert!(a.start_hint.is_some(), "stepping left no suggestion at all");
        assert!(seen.len() > 1, "the suggestion never changed while stepping: {seen:?}");
    }

    /// And it wraps, so the strip can be walked round a short progression.
    #[test]
    fn stepping_wraps_both_ways() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        let len = a.chords.len();
        a.current_chord_index = 0;
        a.step_chord(-1);
        assert_eq!(a.current_chord_index, len - 1, "stepping back off the start did not wrap");
        a.step_chord(1);
        assert_eq!(a.current_chord_index, 0, "stepping forward did not wrap back");
    }

    /// The point of the option: it must not turn a wrong chord into a pass.
    #[test]
    fn short_verdict_still_rejects_the_wrong_chord() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = true;
        a.transition_delay = 0.25;
        let start = a.current_chord_index;
        for _ in 0..20 {
            a.check_progress_with_ai(0.02, "F# m", 0.99);
        }
        assert_eq!(a.current_chord_index, start, "a wrong chord was accepted");
    }

    /// A correct chord read too quietly to be trusted is not a verdict either -
    /// that is what the confidence threshold is for.
    #[test]
    fn short_verdict_needs_the_confidence_threshold() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = true;
        a.chord_confidence = 0.80;
        a.transition_delay = 0.25;
        let start = a.current_chord_index;
        for _ in 0..20 {
            a.check_progress_with_ai(0.02, "C Maj7", 0.50);
        }
        assert_eq!(a.current_chord_index, start, "an unsure reading counted");
    }

    /// With the option off nothing changes: holding is still what advances.
    #[test]
    fn without_the_option_a_single_frame_is_not_enough() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.short_verdict = false;
        a.transition_delay = 0.25;
        let start = a.current_chord_index;
        a.check_progress_with_ai(0.02, "C Maj7", 0.99);
        assert_eq!(a.current_chord_index, start, "one frame advanced without the option");
    }

    /// The timer must not keep running while paused, or the next chord would
    /// jump the instant play resumes.
    #[test]
    fn pause_does_not_bank_up_progress() {
        let mut a = app();
        a.app_mode = AppMode::Chords;
        a.set_random_mode(false);
        a.transition_delay = 0.05;
        a.paused = true;
        for _ in 0..100 { a.check_progress_with_ai(0.1, "C Maj7", 0.99); }
        assert!(a.success_timer <= a.transition_delay + 1e-6,
                "timer ran up to {} while paused", a.success_timer);
        let start = a.current_chord_index;
        a.paused = false;
        a.check_progress_with_ai(0.1, "C Maj7", 0.99);
        assert_ne!(a.current_chord_index, start, "did not resume after unpausing");
    }

    #[test]
    fn pause_freezes_the_fretboard_target() {
        let mut a = app();
        a.set_mode(AppMode::Fretboard as i32);
        a.paused = true;
        a.transition_delay = 0.05;
        a.note_threshold = 0.5;
        let target = a.fret_target.unwrap();
        a.last_pitches = [0.0; 12];
        a.last_pitches[target] = 1.0;
        for _ in 0..40 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.fret_target, Some(target), "paused, yet it drew a new note");
    }

    /// The behaviour the option exists to protect: a strummed chord lights every
    /// one of its tones in the pitch head at once and the interval modes walk
    /// through them - something a monophonic estimate cannot do, so it must not
    /// be allowed to overrule it while the option is off.
    #[test]
    fn a_strummed_chord_still_walks_its_intervals() {
        let mut a = app();
        a.note_threshold = 0.6;
        a.single_notes = false;              // the default
        a.last_pitches = [0.0; 12];
        a.last_pitches[0] = 0.95;            // C
        a.last_pitches[4] = 0.92;            // E
        a.last_pitches[7] = 0.90;            // G
        a.cqt_pitch = Some(0);               // the estimate can only name one
        for pc in [0usize, 4, 7] {
            assert!(a.note_is_sounding(pc, None, 0.0), "chord tone {pc} stopped passing");
        }
        a.single_notes = true;
        assert!(
            !a.note_is_sounding(4, None, 0.0),
            "with the option on, E has to be played on its own"
        );
    }

    /// The reported case: not the first chord of a run but every one after it.
    #[test]
    fn a_new_chord_does_not_inherit_the_old_ones_ringing() {
        let mut a = app();
        a.set_mode(AppMode::Intervals as i32);
        a.last_pitches = [0.9; 12];
        a.prev_pitches = [0.8; 12];
        a.set_onsets([0.5; 12]);
        a.credited[3] = Some(credit(7, 0));

        a.advance_chord();

        assert_eq!(a.last_pitches, [0.0; 12], "the previous chord came along");
        assert_eq!(a.last_onsets, [0.0; 12]);
        // The credit is the exception: it outlives the chord on purpose, so
        // that the same note asked for on both sides of the boundary needs the
        // string struck again. It can only refuse, never credit.
        assert_eq!(a.credited[3], Some(credit(7, 0)), "the credit was dropped at the boundary");
    }

    /// Entering a mode is a fresh start for the ear as well as for the exercise.
    #[test]
    fn a_mode_starts_without_what_was_heard_before_it() {
        let mut a = app();
        a.last_pitches = [0.9; 12];
        a.prev_pitches = [0.8; 12];
        a.set_onsets([0.5; 12]);
        a.credited[3] = Some(credit(7, 0));
        a.cqt_pitch = Some(3);

        a.set_mode(AppMode::Intervals as i32);

        assert_eq!(a.last_pitches, [0.0; 12], "the model's last answer came along");
        assert_eq!(a.prev_pitches, [0.0; 12]);
        assert_eq!(a.last_onsets, [0.0; 12], "an attack from before the switch");
        assert_eq!(a.credited[3], None, "a note credited in another mode still counted");
        // And nothing passes on it: the pitch head has nothing to say until the
        // next audio frame arrives.
        a.cqt_pitch = None;
        assert!(!a.note_is_sounding(3, None, 0.0));
    }

    /// The note before, still ringing inside the model's window, is what the
    /// option is aimed at: the model reports it as sounding and it is, but it
    /// was not struck.
    #[test]
    fn what_was_struck_can_be_asked_for() {
        let mut a = app();
        a.note_threshold = 0.6;
        a.single_notes = false;
        a.last_pitches = [0.0; 12];
        a.last_pitches[0] = 0.95;            // C, being played
        a.last_pitches[10] = 0.93;           // Bb, the note before, still ringing
        a.cqt_pitch = Some(0);

        // Without the option both pass, which is the complaint.
        assert!(a.note_is_sounding(0, None, 0.0));
        assert!(a.note_is_sounding(10, None, 0.0));

        a.require_onset = true;
        let mut onsets = [0.0; 12];
        onsets[0] = 0.4;                     // only C was struck
        a.set_onsets(onsets);
        assert!(a.note_is_sounding(0, None, 0.0), "the struck note stopped passing");
        assert!(!a.note_is_sounding(10, None, 0.0), "the ringing note still counted");

        // An answer from before this frame says nothing about now - checked
        // with the CQT silent, since that branch credits on its own and this
        // gate deliberately does not touch it.
        a.cqt_pitch = None;
        assert!(a.note_is_sounding(0, None, 0.0), "the model lost its own branch");
        a.check_progress_with_ai(0.016, "", 0.0);
        a.check_progress_with_ai(0.016, "", 0.0);
        assert!(
            !a.note_is_sounding(0, None, 0.0),
            "a stale answer from the head was taken for a fresh attack"
        );
    }

    /// Dwa różne losowania pod jednym przyciskiem: kolejność akordów i kolejność
    /// dźwięków w akordzie. Rozdzielone tylko tam, gdzie oba istnieją.
    #[test]
    fn the_shuffle_reaches_the_chord_order_only_where_asked() {
        let mut a = app();

        // Chords: nothing else for the switch to mean, so the option is ignored.
        a.set_mode(AppMode::Chords as i32);
        a.set_random_mode(true);
        assert!(a.shuffles_chord_order(), "shuffle stopped working in Chords");
        a.set_shuffle_chords(true);
        assert!(a.shuffles_chord_order());

        // Intervals: the progression is a tune, so it stays put unless asked.
        a.set_mode(AppMode::Intervals as i32);
        a.set_shuffle_chords(false);
        assert!(
            !a.shuffles_chord_order(),
            "the progression was reordered without being asked"
        );
        a.set_shuffle_chords(true);
        assert!(a.shuffles_chord_order(), "the option did not reach the chord order");

        // And nothing is shuffled at all with the switch off.
        a.set_random_mode(false);
        assert!(!a.shuffles_chord_order());
    }

    /// Zachowanie, o które chodzi: interwały lecą losowo, progresja zostaje
    /// zapisana - z tego robi się melodia.
    #[test]
    fn shuffled_tones_walk_the_written_progression() {
        let mut a = app();
        a.set_mode(AppMode::Intervals as i32);
        a.set_random_mode(true);
        a.set_shuffle_chords(false);
        let n = a.chords.len();
        assert!(n > 1, "test potrzebuje progresji dłuższej niż jeden akord");
        assert_eq!(
            a.play_order,
            (0..n).collect::<Vec<_>>(),
            "kolejność akordów miała zostać taka, jak zapisana"
        );
    }

    /// One CQT frame carries no memory, so it answers "what is sounding NOW"
    /// where the model answers "what sounded in the last 0.77 s".
    #[test]
    fn the_cqt_alone_credits_the_note_being_played() {
        let mut a = app();
        a.note_threshold = 0.6;
        a.last_pitches = [0.0; 12];          // model still says nothing
        a.cqt_pitch = Some(4);               // E is sounding
        assert!(a.note_is_sounding(4, None, 0.0), "the CQT knew and was ignored");
        assert!(!a.note_is_sounding(7, None, 0.0), "credited a note nobody played");
    }

    /// The failure this whole path exists for: the window still holds the note
    /// before the one being played, and it is the LOUDER of the two.
    #[test]
    fn a_note_that_has_been_left_behind_stops_counting() {
        let mut a = app();
        a.single_notes = true;
        a.note_threshold = 0.6;
        a.last_pitches = [0.0; 12];
        a.last_pitches[2] = 0.95;            // D, played a moment ago, still loud
        a.last_pitches[4] = 0.70;            // E, the one under the fingers now
        a.hears(Some(4));
        assert!(!a.note_is_sounding(2, None, 0.0), "credited the previous note");
        assert!(a.note_is_sounding(4, None, 0.0), "did not credit the current one");
    }

    /// Without the CQT to arbitrate, the new note is the one that ROSE - being
    /// loudest belongs to whichever has had more of the window to itself.
    #[test]
    fn a_new_note_under_a_louder_old_one_counts() {
        let mut a = app();
        a.note_threshold = 0.6;
        a.cqt_pitch = None;                  // gate open, estimate too weak to rule
        a.prev_pitches = [0.05; 12];
        a.prev_pitches[2] = 0.90;            // D was already there
        a.last_pitches = [0.0; 12];
        a.last_pitches[2] = 0.95;            // and barely moved
        a.last_pitches[4] = 0.70;            // E arrived
        assert!(a.note_is_sounding(4, None, 0.0), "the note that arrived did not count");
    }

    /// Nothing refreshes the pitch vector while the gate is shut, so it has to be
    /// cleared - otherwise a note that stopped sounding still reads as played.
    #[test]
    fn the_gate_closing_clears_the_pitch_memory() {
        let mut a = app();
        a.last_pitches = [0.5; 12];
        a.prev_pitches = [0.5; 12];
        a.analysis_state.lock().unwrap().gate_open = false;
        a.sync_audio_settings();
        assert_eq!(a.last_pitches, [0.0; 12], "stale pitches survived the gate closing");
        assert_eq!(a.prev_pitches, [0.0; 12]);
    }

    /// Every study is written out exactly as it is played in its source.
    ///
    /// The phrases below were read out of the Guitar Pro files note by note and
    /// turned into degrees against each one's own chord; this test is what
    /// keeps the table honest. The reported case was a study picked by name and
    /// drawn short: the minor one runs two octaves AND a third, so its last
    /// three notes sit on the top string - 5 8 5 in A - and a table entry that
    /// turned round at the seventh had none there at all.
    #[test]
    fn a_book_study_is_written_out_as_it_is_played() {
        let studies = [
            (
                "Minor (Two Octaves and a Third)",
                "1 3 5 7 1' 3' 5' 7' 1'' 3'' 1'' 7' 5' 3' 1' 7 5 3 1",
            ),
            (
                "Major (Leading Tone)",
                "1 3 5 7 1' 3' 5' 7' 1'' 7' 5' 3' 1' 7 5 3 1 7, 1",
            ),
            (
                "Dominant (Approach from Below)",
                "1 3 5 7 1' 3' 5' 7' 5' 3' 1' 7 5 3 1 7, 5, 7, 1",
            ),
            (
                "Skipping Notes (Fifths and Fourths)",
                "1 5 3 7 5 1' 7 3' 1' 5' 3' 7' 5' 1'' 7' 3'' 1'' 5' 7' 3' 5' 1' 3' 7 1' 5 7 3 5 1",
            ),
            (
                "Triplets (Up-Down)",
                "1 3 5 3 5 7 5 7 1' 7 1' 3' 1' 3' 5' 3' 5' 3' 5' 7' 5' 7' 1'' 3'' 1'' 7' 5' 7' \
                 5' 3' 5' 3' 1' 3' 1' 7 1' 7 5 7 5 7 5 3 5 3 1",
            ),
        ];
        let patterns = crate::model::load_arpeggio_patterns();
        for (name, written) in studies {
            let want: Vec<&str> = written.split_whitespace().collect();
            let found = patterns
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("{name} is not in the list at all"));
            assert_eq!(
                found.names, want,
                "{name} is not what is played in the source"
            );
        }
    }

    /// And the minor study reaches the top string, which is the whole of what
    /// tells it from a two-octave run: its last climb is the third above the
    /// second root.
    #[test]
    fn the_minor_study_reaches_the_top_string() {
        let patterns = crate::model::load_arpeggio_patterns();
        let study = patterns
            .iter()
            .find(|p| p.name == "Minor (Two Octaves and a Third)")
            .expect("the minor study is missing");
        let steps = crate::model::steps_of(&study.names);
        let names: Vec<String> = ["1", "3", "5", "7"].iter().map(|s| s.to_string()).collect();
        let spots = crate::tab::place(9, &[0, 3, 7, 10], &steps, &names);   // A m7
        let top: Vec<i32> = spots.iter().filter(|s| s.string == 5).map(|s| s.fret).collect();
        assert_eq!(top, vec![5, 8, 5], "the top string does not read 5 8 5");
    }

    /// A scale is drawn with the degrees IT writes, not with the nearest
    /// interval name: the altered scale spells its second `#2`, and reading
    /// `♭3` on the neck would be a different degree of the same pitch.
    #[test]
    fn the_neck_spells_a_scale_the_way_the_scale_does() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        let altered = a
            .scale_definitions
            .iter()
            .position(|d| d.name.starts_with("Altered"))
            .expect("the altered scale is missing");
        a.item_selected(altered as i32);
        a.scale_start = 0;
        let chord = a.chords[0].clone();
        let steps = a.get_active_indices(&chord);
        let spots = crate::tab::place(
            chord.root as usize,
            &chord.quality.intervals(),
            &steps,
            &chord.quality.interval_names(),
        );
        let labels: Vec<String> = spots.iter().map(|s| s.label()).collect();
        assert!(labels.contains(&"♯2".to_string()), "the neck reads {labels:?}");
        assert!(!labels.contains(&"♭3".to_string()), "a degree was respelled: {labels:?}");
    }

    /// What the shuffle does in Scales: it draws the key and where on the neck
    /// to take it, and leaves the degrees in order. A scale not walked in
    /// order is not a scale.
    #[test]
    fn the_shuffle_moves_a_scale_without_scattering_it() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        let chord = a.chords[0].clone();
        let written = a.get_active_indices(&chord);
        assert!(written.len() >= 5, "the scale is too short to tell");

        a.set_random_mode(true);
        // No permutation: whatever the pass drew, the strip is the written run
        // - the same list `get_active_indices` gives, forwards or backwards.
        assert_eq!(
            a.ordered_active_indices(&chord),
            a.get_active_indices(&chord),
            "the shuffle dealt the scale out of order"
        );
        let _ = &written;

        // The key moves between passes, and so does the string it starts from.
        let key = a.chords[0].root as usize;
        let mut keys = false;
        let mut strings = std::collections::HashSet::new();
        for _ in 0..6 {
            a.advance_chord();
            keys |= a.chords[0].root as usize != key;
            strings.insert(a.scale_start);
        }
        assert!(keys, "the key never moved");
        assert_eq!(strings.len(), 3, "the scale did not go through the strings");
    }

    /// Up and down in turn, and through the strings with them: what is wanted
    /// is coverage, and a fair coin does not give it - twelve drawn passes came
    /// out with five descending in a row.
    #[test]
    fn a_scale_runs_both_ways() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        let chord = a.chords[0].clone();
        let mut ways = Vec::new();
        let mut pairs = std::collections::HashSet::new();
        for _ in 0..6 {
            a.advance_chord();
            let steps = a.get_active_indices(&chord);
            ways.push(steps.first().unwrap().degree < steps.last().unwrap().degree);
            pairs.insert((a.scale_start, a.scale_descending));
        }
        assert!(ways.windows(2).all(|w| w[0] != w[1]), "two passes ran the same way");
        assert_eq!(pairs.len(), 6, "six passes did not cover the six ways to take it");

        // Whichever way, it is the written run and not a shuffled one.
        a.scale_descending = true;
        let down = a.get_active_indices(&chord);
        a.scale_descending = false;
        let up = a.get_active_indices(&chord);
        let mut reversed = down.clone();
        reversed.reverse();
        assert_eq!(reversed, up, "the descent is not the climb turned round");
    }

    /// And without the shuffle the string still moves, while the key stays: the
    /// shape is the same everywhere, so where it is taken is the exercise.
    #[test]
    fn a_scale_starts_from_a_drawn_string_even_unshuffled() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        a.set_random_mode(false);
        let key = a.chords[0].root as usize;
        let mut strings = std::collections::HashSet::new();
        for _ in 0..6 {
            a.advance_chord();
            strings.insert(a.scale_start);
            assert_eq!(a.chords[0].root as usize, key, "the key moved unasked");
        }
        assert_eq!(strings.len(), 3, "the scale did not go through the strings");
    }

    /// What twenty passes of a scale actually draw.
    /// `cargo test --release -- --ignored --nocapture the_scale_passes`
    #[test]
    #[ignore]
    fn the_scale_passes() {
        for shuffled in [false, true] {
            let mut a = app();
            a.set_mode(AppMode::Scales as i32);
            a.set_random_mode(shuffled);
            println!("--- losowanie {}", if shuffled { "wlaczone" } else { "wylaczone" });
            for _ in 0..12 {
                a.advance_chord();
                let chord = a.chords[0].clone();
                let steps = a.get_active_indices(&chord);
                println!(
                    "  tonacja {:<3} struna {} kierunek {:<5} pierwszy stopien {}",
                    chord.root.to_string(),
                    a.scale_start,
                    if a.scale_descending { "dol" } else { "gora" },
                    steps.first().map(|s| s.degree).unwrap_or(0),
                );
            }
        }
    }

    /// Renders a scale on the neck to stdout, for looking at it. Ignored in
    /// normal runs; `cargo test -- --ignored --nocapture the_scale_picture`.
    /// Where each step of a study lands, for reading a screenshot back.
    /// `cargo test --release -- --ignored --nocapture the_study_places`
    #[test]
    #[ignore]
    fn the_study_places() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.arp_exercise = 0;
        a.item_selected(0);
        a.secondary_item_selected(0);             // C
        let chord = a.chords[0].clone();
        let steps = a.get_active_indices(&chord);
        let spots = crate::tab::place_near(
            chord.root as usize,
            &chord.quality.intervals(),
            &steps,
            &chord.quality.interval_names(),
            None,
        );
        for (i, spot) in spots.iter().enumerate() {
            println!("krok {i:2}: struna {} próg {:2}  {}", spot.string, spot.fret, spot.label());
        }
    }

    #[test]
    #[ignore]
    fn the_scale_picture() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        a.set_random_mode(true);
        let chord = a.chords[0].clone();
        let steps = a.ordered_active_indices(&chord);
        let spots = crate::tab::place_near(
            chord.root as usize,
            &chord.quality.intervals(),
            &steps,
            &chord.quality.interval_names(),
            Some(a.voicing_anchor),
        );
        let done: Vec<bool> = (0..spots.len()).map(|i| i < 2).collect();
        println!("{}", crate::tab::neck(&spots, &done, 2, false));
    }

    /// The reported stall, with the attack head silent throughout: the study
    /// asks for the flat seventh at step 7, three other notes are played, and
    /// step 11 asks for it again. Without something to say the note had gone
    /// quiet in between, the phrase could not get past that step.
    #[test]
    fn a_phrase_gets_past_a_note_it_has_already_used() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.arp_exercise = 0;
        a.item_selected(0);
        a.note_threshold = 0.5;
        let seventh = 10usize;                    // over the default m7

        a.hears(Some(seventh));
        a.credit_class(seventh, 0);
        assert!(!a.struck_since_credit(seventh), "the same pluck counted twice");

        // Three other notes, each held steadily, and no attack reported at all.
        let mut frame = 0u64;
        for pc in [0usize, 3, 0] {
            for _ in 0..4 {
                frame += 1;
                a.audio_frames = frame - 1;
                a.feed_estimate(frame, Some(pc));
            }
            a.check_progress_with_ai(0.3, "Noise", 0.0);
        }
        // The phrase comes back to it and the player plays it.
        for _ in 0..4 {
            frame += 1;
            a.audio_frames = frame - 1;
            a.feed_estimate(frame, Some(seventh));
        }
        assert!(
            a.struck_since_credit(seventh),
            "the phrase could not come back to a note it had used"
        );
    }

    /// An arpeggio played through, note by note, finishes and starts again.
    #[test]
    fn an_arpeggio_played_through_comes_round() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.arp_exercise = 0;
        a.set_random_mode(false);
        a.item_selected(0);                       // the minor study
        a.note_threshold = 0.5;
        let chord = a.chords[0].clone();
        let steps = a.get_active_indices(&chord);
        let all = chord.get_target_indices();
        assert!(steps.len() > 8, "the phrase is too short to tell");

        let mut frame = 0u64;
        for (n, step) in steps.iter().enumerate() {
            let pc = all[step.degree] % 12;
            // Played: the string struck, the estimate on it, the model too.
            let mut v = [0.0; 12];
            v[pc] = 0.9;
            a.set_onsets(v);
            a.set_onsets([0.0; 12]);
            a.last_pitches = [0.0; 12];
            a.last_pitches[pc] = 1.0;
            for _ in 0..4 {
                frame += 1;
                a.audio_frames = frame - 1;
                a.feed_estimate(frame, Some(pc));
            }
            for _ in 0..10 {
                a.check_progress_with_ai(0.1, "Noise", 0.0);
            }
            let expected = if n + 1 == steps.len() { 0 } else { n + 1 };
            assert_eq!(
                a.current_note_step, expected,
                "step {n} of {} did not count", steps.len()
            );
        }
    }

    /// Played one at a time, nothing but the ear counts - and it has to have
    /// held. The model answers about 0.77 s of audio, which is where the notes
    /// nobody played come from.
    #[test]
    fn one_at_a_time_believes_the_ear_alone() {
        let mut a = app();
        a.single_notes = true;
        a.note_threshold = 0.5;
        // The model is sure, loudly, and the ear says nothing: no credit.
        a.last_pitches = [0.0; 12];
        a.last_pitches[7] = 1.0;
        a.last_onsets[7] = 1.0;
        a.onset_age = 0;
        // The ear says nothing at all, so the model's pitch head may answer -
        // a note too quiet for the estimate has to stay playable.
        assert!(a.note_is_sounding(7, None, 0.0), "a quiet note became uncreditable");
        // But the ear naming something ELSE is a refusal.
        a.hears(Some(2));
        assert!(!a.note_is_sounding(7, None, 0.0), "the model overruled the ear");
        // And the chord name never credits a note here.
        a.hears(None);
        a.last_pitches = [0.0; 12];
        assert!(
            !a.note_is_sounding(7, Some(NoteName::G), 1.0),
            "the root head credited a note nobody played"
        );
        // The ear, steadily: that is the whole test.
        a.hears(Some(7));
        assert!(a.note_is_sounding(7, None, 0.0), "the ear was not believed");
        // One frame of it is not enough.
        a.steady_for = 1;
        assert!(!a.note_is_sounding(7, None, 0.0), "a single frame counted");
    }

    /// Arpeggios are one note at a time by definition, ticked or not.
    #[test]
    fn an_arpeggio_is_always_one_note_at_a_time() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.single_notes = false;                  // and it makes no difference
        a.note_threshold = 0.5;
        a.last_pitches = [0.0; 12];
        a.last_pitches[7] = 1.0;
        a.last_onsets[7] = 1.0;
        a.onset_age = 0;
        a.hears(Some(2));                        // the ear on another note
        assert!(!a.note_is_sounding(7, None, 0.0), "the model overruled the ear");
        a.hears(Some(7));
        assert!(a.note_is_sounding(7, None, 0.0), "the ear was not believed");
    }

    /// And "credit only what was struck" holds back every way in, the chord
    /// name included: with a microphone in a room the model goes on naming
    /// chords, and their roots were being credited as notes.
    #[test]
    fn the_root_head_needs_an_attack_too() {
        let mut a = app();
        a.single_notes = false;
        a.require_onset = true;
        a.note_threshold = 0.5;
        a.onset_age = 99;                        // nothing struck for a while
        assert!(
            !a.note_is_sounding(7, Some(NoteName::G), 1.0),
            "a chord name credited a note with no attack behind it"
        );
        a.last_onsets[7] = 1.0;
        a.onset_age = 0;
        assert!(
            a.note_is_sounding(7, Some(NoteName::G), 1.0),
            "a struck note went uncredited"
        );
    }

    /// One pluck credits one step. A note credited a moment ago and still
    /// ringing must not count again for the next step that asks for it - the
    /// reported case being `3` played and `1` credited along with it.
    #[test]
    fn a_ringing_note_does_not_count_for_the_next_step() {
        let mut a = app();
        a.set_mode(AppMode::Intervals as i32);
        let (root, third) = (0usize, 4usize);

        a.hears(Some(root));
        a.credit_class(root, 0);

        // The third is played on top; the root rings on, and the model - which
        // is polyphonic - still reports it.
        let mut frame = 0u64;
        for _ in 0..6 {
            frame += 1;
            a.audio_frames = frame - 1;
            a.feed_estimate(frame, Some(third));
        }
        a.check_progress_with_ai(0.6, "Noise", 0.0);
        assert!(
            !a.struck_since_credit(root),
            "the ringing root counted again while the third was being played"
        );

        // Struck again - the estimate names it, because it is loudest now.
        for _ in 0..4 {
            frame += 1;
            a.audio_frames = frame - 1;
            a.feed_estimate(frame, Some(root));
        }
        assert!(a.struck_since_credit(root), "a real second pluck did not count");
    }

    /// A note asked for again, after something else has been played, counts
    /// when it is played - even where the attack head missed the pluck.
    ///
    /// The reported case: the fretboard trainer lighting green on a note that
    /// was plainly right and never moving on. The note had been credited a few
    /// steps earlier, the head did not report the new attack, and the rule that
    /// stops one pluck counting twice refused it for ever.
    #[test]
    fn a_note_that_went_quiet_counts_when_it_comes_back() {
        let mut a = app();
        a.set_mode(AppMode::Fretboard as i32);
        let pc = 4usize;
        a.hears(Some(pc));
        a.credit_class(pc, 0);
        assert!(!a.struck_since_credit(pc), "an unstruck repeat counted at once");

        // Something else, steadily, and then silence: the note has gone.
        for frame in 1..=4u64 {
            a.audio_frames = frame - 1;
            a.feed_estimate(frame, Some(9));
        }
        a.check_progress_with_ai(0.6, "Noise", 0.0);   // past the settling window
        // And now it is played again: the estimate names it, because a struck
        // string is what is loudest.
        for frame in 5..=8u64 {
            a.audio_frames = frame - 1;
            a.feed_estimate(frame, Some(pc));
        }
        assert!(
            a.struck_since_credit(pc),
            "the note stayed uncreditable after it stopped sounding"
        );
    }

    /// A drawing follows a reading that has held, not every frame: at a
    /// microphone, with something playing in the room, the neck flickered.
    #[test]
    fn a_stray_reading_does_not_reach_the_drawing() {
        let mut a = app();
        assert_eq!(a.steady_note(), None, "nothing has been heard yet");

        // One frame of a note is not a note.
        for (frame, pitch) in [(1u64, Some(3usize)), (2, Some(7)), (3, Some(3))] {
            a.audio_frames = frame - 1;
            a.feed_estimate(frame, pitch);
            assert_eq!(a.steady_note(), None, "a single frame reached the drawing");
        }
        // Three frames of the same one is.
        for frame in 4..=6u64 {
            a.audio_frames = frame - 1;
            a.feed_estimate(frame, Some(9));
        }
        assert_eq!(a.steady_note(), Some(9), "a reading that held did not count");

        // And it lets go as soon as something else is heard.
        a.audio_frames = 6;
        a.feed_estimate(7, Some(2));
        assert_eq!(a.steady_note(), None, "the old reading outlived what replaced it");
    }

    /// A note quiet enough that the estimate loses it every other frame was
    /// uncreditable: the run of three was rarely reached, and the answer was to
    /// play louder. See `CREDIT_TICKS` for what was measured.
    #[test]
    fn two_frames_are_enough_to_credit_but_not_to_take_over() {
        let mut a = app();
        a.app_mode = AppMode::Scales;
        assert_eq!(a.sounding_now(), None, "nothing heard yet");

        a.audio_frames = 0;
        a.feed_estimate(1, Some(9));
        assert_eq!(a.sounding_now(), None, "one frame is still not a note");

        a.audio_frames = 1;
        a.feed_estimate(2, Some(9));
        assert_eq!(a.sounding_now(), Some(9), "two frames of the same reading");
        // But the longer measure - the one that says a credited note has gone
        // quiet - has not been reached, so a repeat cannot ride on this.
        assert_eq!(a.steady_note(), None, "two frames must not count as taking over");

        a.audio_frames = 2;
        a.feed_estimate(3, Some(9));
        assert_eq!(a.steady_note(), Some(9), "three frames still means what it did");
    }

    /// A phrase keeps its order, whichever phrase is showing: the notes of a
    /// study dealt at random are no longer that study.
    #[test]
    fn the_shuffle_does_not_reorder_an_arpeggio() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.arp_exercise = 0;
        a.item_selected(0);
        a.set_random_mode(true);
        let chord = a.chords[0].clone();
        assert_eq!(
            a.ordered_active_indices(&chord),
            a.get_active_indices(&chord),
            "the shuffle dealt the phrase a new order"
        );
    }

    /// What the shuffle means there: the KEY. The study is the player's choice
    /// and the switch does not touch it.
    #[test]
    fn the_shuffle_moves_the_key_and_leaves_the_study() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.arp_exercise = 0;
        a.item_selected(2);
        let study = a.arpeggio_patterns[2].name.clone();
        let key = a.chords[0].root as usize;

        a.set_random_mode(true);
        assert_eq!(
            a.arpeggio_patterns[a.selected_library_idx].name, study,
            "the shuffle changed the study"
        );

        let mut moved = false;
        for _ in 0..20 {
            a.advance_chord();
            if a.chords[0].root as usize != key {
                moved = true;
                break;
            }
        }
        assert!(moved, "the key never moved between passes");
        assert_eq!(
            a.arpeggio_patterns[a.selected_library_idx].name, study,
            "a pass changed the study"
        );
    }

    /// And the studies named for a chord know which one: the phrase is the
    /// book's only when it is read over the chord it was written for.
    #[test]
    fn a_study_named_for_a_chord_names_the_right_one() {
        use crate::model::study_quality;
        assert_eq!(study_quality("Minor (Two Octaves and a Third)"), Some(0));
        assert_eq!(study_quality("Major (Leading Tone)"), Some(1));
        assert_eq!(study_quality("Dominant (Approach from Below)"), Some(2));
        // The rest name no quality, and must not set one.
        assert_eq!(study_quality("Skipping Notes (Fifths and Fourths)"), None);
        assert_eq!(study_quality("Triplets (Up-Down)"), None);
        assert_eq!(study_quality("Two Octaves Up-Down"), None);
        // The index has to mean what the settings list means by it.
        for (i, want) in [(0, "m7"), (1, "Maj7"), (2, "7")] {
            assert_eq!(ARP_QUALITIES[i].0, want, "the quality list moved under it");
        }
    }

    /// The study on screen is the study that was chosen, in every key.
    ///
    /// The reported case: with the key on A - the tenth row of the key list,
    /// and the generator sits in the tenth row of the PATTERN list - the phrase
    /// was quietly replaced by a freshly generated one, so the name above said
    /// "Up-Down, Approach from Below" while the neck showed something else
    /// entirely, two octaves shorter and never reaching the top string.
    #[test]
    fn a_study_plays_the_phrase_it_names() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.arp_exercise = 0;
        let patterns = a.arpeggio_patterns.clone();
        for (i, pattern) in patterns.iter().enumerate() {
            if pattern.name == crate::model::GENERATOR_NAME {
                continue;                       // that one is meant to change
            }
            for key in 0..12 {
                a.item_selected(i as i32);
                a.secondary_item_selected(key);
                assert_eq!(
                    a.intervals_input,
                    pattern.names.join(" "),
                    "{} in key {key} was swapped for something else",
                    pattern.name
                );
                assert_eq!(a.song_title, pattern.name, "the name above does not match");
                assert_eq!(a.chords[0].root as usize, key as usize, "the key was not taken");
            }
        }
    }

    /// Over the changes the phrase is built, and the descending one is the
    /// shape the studies use: from the root DOWNWARDS through the chord's own
    /// tones, not the ascending phrase read backwards.
    #[test]
    fn the_changes_run_matches_the_studies() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.arp_exercise = 1;
        let chord = Chord { root: NoteName::D, quality: ChordQuality::Dominant7 };

        a.arp_direction = 0;
        let up: Vec<(usize, i8)> =
            a.get_active_indices(&chord).iter().map(|s| (s.degree, s.octave)).collect();
        assert_eq!(
            up,
            vec![(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
            "1 3 5 7 1' 3' 5' 7'"
        );

        a.arp_direction = 1;
        let down: Vec<(usize, i8)> =
            a.get_active_indices(&chord).iter().map(|s| (s.degree, s.octave)).collect();
        assert_eq!(
            down,
            vec![(0, 0), (3, -1), (2, -1), (1, -1), (0, -1), (3, -2), (2, -2), (1, -2)],
            "1 7, 5, 3, 1, 7,, 5,, 3,,"
        );
    }

    /// Alternating: one chord one way, the next the other, from whichever side
    /// the option starts on.
    #[test]
    fn alternating_turns_the_run_round_at_every_chord() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.arp_exercise = 1;
        a.set_random_mode(false);
        let chord = a.chords[a.current_chord_index].clone();
        let is_down = |a: &MyApp, c: &Chord| a.get_active_indices(c)[1].octave < 0;

        a.arp_direction = 2;                    // from the descending one
        assert!(is_down(&a, &chord), "the first chord did not start downwards");
        a.play_pos = 1;
        assert!(!is_down(&a, &chord), "the second chord did not turn round");

        a.arp_direction = 3;                    // and the other way about
        assert!(is_down(&a, &chord), "starting up, the second chord runs down");
        a.play_pos = 0;
        assert!(!is_down(&a, &chord), "the first chord did not start upwards");
    }

    /// Every token of every arpeggio pattern is a chord tone the mode can ask
    /// for. A typo drops the step silently - the phrase just comes out shorter
    /// than it was written - so the count is what is checked. Some patterns run
    /// one way only, so where they end is not.
    #[test]
    fn the_arpeggio_patterns_are_playable_as_written() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        let chord = Chord { root: NoteName::C, quality: ChordQuality::Minor7 };
        for pattern in crate::model::load_arpeggio_patterns() {
            if pattern.name == crate::model::GENERATOR_NAME {
                continue;                       // built fresh every pass
            }
            let written = pattern.names.join(" ");
            a.intervals_input = written.clone();
            let steps = a.get_active_indices(&chord);
            assert_eq!(
                steps.len(),
                pattern.names.len(),
                "{}: {} of {} tokens are playable - {written}",
                pattern.name,
                steps.len(),
                pattern.names.len()
            );
        }
    }

    /// Every built-in scale asks for its own notes, in its own order.
    ///
    /// The reported case: the altered scale showed `1 b2 #2 #2 b5 #5 b7` - the
    /// major third asked for as `#2`, and the note itself never asked for at
    /// all. Ten of the twenty-six scales were walking a step twice like that,
    /// wherever they spell both an altered degree and its natural neighbour.
    #[test]
    fn a_scale_walks_the_degrees_it_is_written_with() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        a.scale_descending = false;              // the written order, not a pass
        let scales = crate::model::load_all_scale_definitions();
        assert!(scales.len() >= 20, "the built-in scales did not load");
        for def in scales {
            let written: Vec<String> = def.names.clone();
            a.intervals_input = written.join(" ");
            let chord = Chord {
                root: NoteName::C,
                quality: ChordQuality::CustomScale(def.clone()),
            };
            let walked: Vec<usize> =
                a.get_active_indices(&chord).iter().map(|s| s.degree).collect();
            assert_eq!(
                walked,
                (0..written.len()).collect::<Vec<_>>(),
                "{} walks {:?} of {:?}",
                def.name,
                walked.iter().map(|&i| &written[i]).collect::<Vec<_>>(),
                written
            );
        }
    }

    /// And the loose rules still answer for a token the set spells otherwise:
    /// `3` over a minor seventh is its flat third, not nothing at all.
    #[test]
    fn a_plain_degree_still_finds_the_one_the_chord_has() {
        let mut a = app();
        a.set_mode(AppMode::Intervals as i32);
        a.intervals_input = "1 3 5 7".to_string();
        let chord = Chord { root: NoteName::C, quality: ChordQuality::Minor7 };
        let walked: Vec<usize> =
            a.get_active_indices(&chord).iter().map(|s| s.degree).collect();
        assert_eq!(walked, vec![0, 1, 2, 3], "b3 and b7 stopped answering for 3 and 7");
    }

    /// The scale run ends where it started, when the option asks for it.
    #[test]
    fn a_scale_can_end_on_its_root_again() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        a.scale_descending = false;              // the closing root is the top one
        let chord = a.chords[0].clone();
        let plain = a.get_active_indices(&chord);
        a.scale_repeat_root = true;
        let looped = a.get_active_indices(&chord);
        assert_eq!(looped.len(), plain.len() + 1, "no extra step");
        assert_eq!(
            looped.last().unwrap().degree,
            plain.first().unwrap().degree,
            "the run does not end on the note it started from"
        );
        assert_eq!(looped.last().unwrap().octave, plain.first().unwrap().octave + 1,
                   "the closing root is not marked an octave up");
        // And it is the scales' own: the other modes are untouched.
        a.set_mode(AppMode::Intervals as i32);
        let chord = a.chords[a.current_chord_index].clone();
        assert_eq!(a.get_active_indices(&chord).len(),
                   { a.scale_repeat_root = false; a.get_active_indices(&chord).len() });
    }

    /// An arpeggio may ask for the same pitch class twice in a row. One pluck
    /// must credit one step, or a single ringing string walks through both.
    ///
    /// No longer gated on "play the notes one at a time": a repeat off one
    /// pluck was wrong in every mode, and the test never fires on a strummed
    /// chord because there the successive steps are different classes.
    #[test]
    fn one_pluck_credits_one_step() {
        let mut a = app();
        a.single_notes = false;                    // the default, and it still holds
        a.set_mode(AppMode::Fretboard as i32);
        a.transition_delay = 0.05;
        a.note_threshold = 0.5;
        let target = a.fret_target.unwrap();
        a.last_pitches = [0.0; 12];
        a.last_pitches[target % 12] = 1.0;
        a.credit_class(target, 0);                // this pluck already counted
        a.onset_id = 0;
        for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.fret_target, Some(target), "the same pluck counted twice");

        a.onset_id = 1;                            // strings struck again
        for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(
            a.credited[target % 12], Some(credit(1, 0)),
            "a fresh attack on the right note did not count"
        );
    }

    /// A scale that ends on the note it starts from asks for that note twice
    /// over the turn of the lap. The string is still ringing, so the first step
    /// of the next pass has to wait for it to be played again.
    #[test]
    fn the_closing_root_does_not_open_the_next_pass() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        a.set_random_mode(false);
        a.scale_repeat_root = true;
        a.note_threshold = 0.5;
        let chord = a.chords[0].clone();
        let steps = a.get_active_indices(&chord);
        let root_pc = chord.get_target_indices()[steps[0].degree] % 12;

        // Standing on the closing root and playing it: the lap ends and the
        // string goes on ringing, which is where the run leaves the player.
        a.current_note_step = steps.len() - 1;
        a.last_pitches = [0.0; 12];
        a.last_pitches[root_pc] = 1.0;

        for _ in 0..20 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.current_note_step, 0, "the closing root did not end the lap");

        // The string is still ringing, so the model goes on reporting it.
        a.last_pitches[root_pc] = 1.0;
        for _ in 0..20 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.current_note_step, 0, "the ringing root opened the next pass");

        let mut v = [0.0; 12];
        v[root_pc] = 0.9;
        a.set_onsets(v);                            // played again
        for _ in 0..20 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.current_note_step, 1, "a real second pluck did not count");
    }

    /// The whole run, played with every note left ringing. The closing root is
    /// the note the run started from, six notes back and still sounding, and
    /// the six plucks in between move the envelope's attack counter - which is
    /// why the envelope cannot be what answers "was this struck again".
    #[test]
    fn the_closing_root_is_not_credited_to_the_opening_one() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        a.set_random_mode(false);
        a.scale_repeat_root = true;
        a.note_threshold = 0.5;
        let chord = a.chords[0].clone();
        let steps = a.get_active_indices(&chord);
        let all = chord.get_target_indices();
        let pcs: Vec<usize> = steps.iter().map(|st| all[st.degree] % 12).collect();
        assert_eq!(pcs.first(), pcs.last(), "the run does not close on its root");

        // The run played note by note, each one left ringing. The estimate
        // follows the newest string, and the head spreads every attack onto the
        // root as well - which is what it does on the measured material.
        let low = 24 + pcs[0];                      // where the run starts
        let mut onsets;
        for (i, pc) in pcs.iter().take(steps.len() - 1).enumerate() {
            a.last_pitches[*pc] = 1.0;
            a.hears(Some(*pc));
            a.cqt_semitone = Some(low + [0, 2, 4, 5, 7, 9, 11][i]);
            onsets = [0.0; 12];
            onsets[*pc] = 0.9;
            onsets[pcs[0]] = 0.7;                   // the stray
            a.set_onsets(onsets);
            a.set_onsets([0.0; 12]);                // and it falls back
            a.onset_id += 1;
            for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        }
        assert!(a.strike_id[pcs[0]] > 1, "the strays did not reach the root's counter");
        assert_eq!(a.current_note_step, steps.len() - 1, "the run did not reach its last step");

        // Nothing is played now. The opening root goes on ringing, its own
        // counter has moved from the strays, and the estimate is reading the
        // seventh degree - the string that was hit last.
        for _ in 0..20 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.current_note_step, steps.len() - 1, "the opening root closed the run");

        // Played where the run asks for it, an octave up: the estimate says so,
        // and that is enough on its own.
        a.hears(Some(pcs[0]));
        a.cqt_semitone = Some(low + 12);
        for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.current_note_step, 0, "playing the closing root did not end the run");
    }

    /// The reported case, and the one the head's own latency causes: the strike
    /// for the pluck that closed the run arrives after the step was credited on
    /// it, and would then answer the next lap's first step off the same ringing
    /// string.
    #[test]
    fn the_late_strike_belongs_to_the_pluck_that_earned_it() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        a.set_random_mode(false);
        a.scale_repeat_root = true;
        a.note_threshold = 0.5;
        let chord = a.chords[0].clone();
        let steps = a.get_active_indices(&chord);
        let pc = chord.get_target_indices()[steps[0].degree] % 12;
        let high = 24 + pc + 12;

        // Standing on the closing root, just credited on the pluck that is
        // still ringing and still what the estimate reads.
        a.current_note_step = 0;
        a.last_pitches[pc] = 1.0;
        a.hears(Some(pc));
        a.cqt_semitone = Some(high);
        a.credit_class(pc, 1);

        // The head answers about that pluck a third of a second later.
        let mut v = [0.0; 12];
        v[pc] = 0.9;
        for _ in 0..3 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        a.set_onsets(v);
        for _ in 0..20 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.current_note_step, 0, "the pluck's own strike opened the next lap");

        // Struck again, well after: that is a second pluck and it counts.
        a.set_onsets([0.0; 12]);
        a.set_onsets(v);
        for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.current_note_step, 1, "a real second pluck did not count");
    }

    /// And the run that closes where it started, in the same octave - a shape
    /// that does not reach up. The estimate cannot prove anything there, so the
    /// note has to be struck again, which is the whole of what is asked.
    #[test]
    fn a_root_repeated_in_its_own_octave_still_counts() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        a.set_random_mode(false);
        a.scale_repeat_root = true;
        a.note_threshold = 0.5;
        let chord = a.chords[0].clone();
        let steps = a.get_active_indices(&chord);
        let all = chord.get_target_indices();
        let pc = all[steps[0].degree] % 12;
        let low = 24 + pc;

        a.current_note_step = steps.len() - 1;
        a.last_pitches[pc] = 1.0;
        a.hears(Some(pc));
        a.cqt_semitone = Some(low);
        a.credit_class(pc, 0);                      // credited on the opening root

        // Ringing on, nothing struck: it must not close the run.
        for _ in 0..20 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.current_note_step, steps.len() - 1, "the ring closed the run");

        // Struck again, in the same octave, with the estimate reading it.
        let mut v = [0.0; 12];
        v[pc] = 0.9;
        a.set_onsets(v);
        for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.current_note_step, 0, "a note played again in its own octave never counted");
    }

    /// The onset head is the other way a repeat is recognised, and the one that
    /// works when the string is still ringing: the envelope detector reads the
    /// RMS of a 512 ms window, where a second pluck barely shows.
    #[test]
    fn a_second_strike_is_seen_by_the_onset_head() {
        let mut a = app();
        a.single_notes = false;
        a.set_mode(AppMode::Fretboard as i32);
        a.transition_delay = 0.05;
        a.note_threshold = 0.5;
        let target = a.fret_target.unwrap();
        let pc = target % 12;
        a.last_pitches = [0.0; 12];
        a.last_pitches[pc] = 1.0;

        // A strike: the head's answer for that class rises past the threshold.
        let mut v = [0.0; 12];
        v[pc] = 0.8;
        a.set_onsets(v);
        let first = a.strike_id[pc];
        a.credit_class(pc, 0);                       // and it has been credited

        // The answer lingering does not count as another strike.
        a.set_onsets(v);
        for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.fret_target, Some(target), "a lingering answer counted as a new pluck");

        // Decayed and struck again: that is a second strike, with no help from
        // the envelope detector, whose counter has not moved.
        a.set_onsets([0.0; 12]);
        a.set_onsets(v);
        assert_ne!(a.strike_id[pc], first, "the crossing was not counted");
        for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_ne!(a.fret_target, Some(target), "the second pluck did not count");
    }

    /// Feeds the note the exercise is currently asking for, once.
    fn play_current_note(a: &mut MyApp, dt: f32) {
        let chord = a.chords[a.current_chord_index].clone();
        let steps = a.ordered_active_indices(&chord);
        let targets = chord.get_target_indices();
        if let Some(step) = steps.get(a.current_note_step) {
            let pc = targets[step.degree] % 12;
            a.last_pitches = [0.0; 12];
            a.last_pitches[pc] = 1.0;
        }
        a.check_progress_with_ai(dt, "Noise", 0.0);
    }

    /// Scales: the key is drawn afresh once the whole scale has been played.
    #[test]
    fn scales_change_key_after_the_scale_is_finished() {
        let mut a = scales_app();
        a.set_random_mode(true);
        a.note_threshold = 0.5;
        let first = a.chords[0].root as usize;
        let steps = a.ordered_active_indices(&a.chords[0].clone()).len();
        for _ in 0..steps { play_current_note(&mut a, 0.5); }
        assert_ne!(a.chords[0].root as usize, first, "the key did not move after a full pass");
    }

    /// Paused, the step never completes, so the scale never finishes and the key
    /// stays put. This is the chain the pause button relies on.
    #[test]
    fn paused_scales_keep_their_key_and_their_step() {
        let mut a = scales_app();
        a.set_random_mode(true);
        a.note_threshold = 0.5;
        a.paused = true;
        let key = a.chords[0].root as usize;
        let step = a.current_note_step;
        for _ in 0..200 { play_current_note(&mut a, 0.5); }
        assert_eq!(a.current_note_step, step, "paused, yet it stepped through the scale");
        assert_eq!(a.chords[0].root as usize, key, "paused, yet the key changed");
    }

    /// ...and unpausing lets it run again.
    #[test]
    fn unpausing_lets_the_scale_finish() {
        let mut a = scales_app();
        a.set_random_mode(true);
        a.note_threshold = 0.5;
        a.paused = true;
        for _ in 0..50 { play_current_note(&mut a, 0.5); }
        let key = a.chords[0].root as usize;
        a.paused = false;
        let steps = a.ordered_active_indices(&a.chords[0].clone()).len();
        for _ in 0..steps { play_current_note(&mut a, 0.5); }
        assert_ne!(a.chords[0].root as usize, key, "still stuck after unpausing");
    }

    #[test]
    fn hint_appears_only_with_randomisation() {
        let mut a = app();
        a.set_random_mode(false);
        assert!(a.start_hint.is_none());
        a.set_random_mode(true);
        assert!(a.start_hint.is_some());
        assert!(a.start_hint.unwrap() < START_STRINGS.len());
    }
}

#[cfg(test)]
mod fretboard_mode_tests {
    use super::tests::*;
    use super::*;

    /// The region is drawn once per entry into the mode, not per note - the
    /// whole point is to stay in one hand position.
    #[test]
    fn region_holds_while_notes_change() {
        let mut a = app();
        a.set_mode(AppMode::Fretboard as i32);
        let region = a.region;
        let mut notes = std::collections::HashSet::new();
        for _ in 0..40 {
            notes.insert(a.fret_target.expect("a target must be drawn"));
            a.next_fret_target();
            assert_eq!(a.region.strings, region.strings, "the region moved mid-exercise");
            assert_eq!(a.region.fret_from, region.fret_from, "the region moved mid-exercise");
        }
        assert!(notes.len() > 1, "the note never changed");
    }

    #[test]
    fn every_target_is_playable_in_the_region() {
        let mut a = app();
        for _ in 0..30 {
            a.set_mode(AppMode::Fretboard as i32);
            for _ in 0..30 {
                let pc = a.fret_target.expect("a target must be drawn");
                assert!(
                    !a.region.positions_of(pc).is_empty(),
                    "{:?}: asked for {pc}, not reachable there", a.region
                );
                a.next_fret_target();
            }
        }
    }

    /// A window running off the end of the neck would ask for notes that cannot
    /// be played on a 15-fret range.
    #[test]
    fn random_region_stays_on_the_neck() {
        let mut a = app();
        for _ in 0..300 {
            a.randomize_region();
            assert!(
                a.region.fret_to() <= crate::fretboard::MAX_FRET,
                "region {:?} runs past fret {}", a.region, crate::fretboard::MAX_FRET
            );
            assert_eq!(a.region.fret_span, 4, "span should be one finger per fret");
        }
    }

    /// Entering the mode repeatedly must give different regions, otherwise the
    /// randomisation is decorative.
    #[test]
    fn re_entering_the_mode_gives_a_new_region() {
        let mut a = app();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..40 {
            a.set_mode(AppMode::Fretboard as i32);
            seen.insert((a.region.strings as i32, a.region.fret_from));
        }
        assert!(seen.len() > 3, "regions barely varied: {seen:?}");
    }

    /// The fretboard branch runs before the chord guard; with no chords loaded
    /// the old code path would have returned early and the mode would be dead.
    #[test]
    fn progress_runs_without_any_chords() {
        let mut a = app();
        a.set_mode(AppMode::Fretboard as i32);
        assert!(a.chords.is_empty(), "the trainer should not load a song");
        let target = a.fret_target.unwrap();
        a.last_pitches = [0.0; 12];
        a.last_pitches[target] = 1.0;
        a.note_threshold = 0.5;
        // The trainer holds for a fixed 0.12 s, as the note modes do - the
        // chord delay is not its measure.
        a.check_progress_with_ai(0.1, "Noise", 0.0);
        assert_eq!(a.fret_target, Some(target), "credited before the hold was up");
        a.check_progress_with_ai(0.1, "Noise", 0.0);
        assert_ne!(a.fret_target, Some(target), "a played note did not advance");
    }
}

#[cfg(test)]
mod generator_tests {
    use super::tests::*;
    use super::*;

    fn pick_generator(a: &mut MyApp) {
        a.set_mode(AppMode::Arpeggios as i32);
        a.arp_exercise = 0;
        let idx = a.arpeggio_patterns.iter()
            .position(|p| p.name == crate::model::GENERATOR_NAME)
            .expect("generator entry missing from the arpeggio list");
        // The phrase is the first combo in the studies; the second holds the key.
        a.item_selected(idx as i32);
    }

    /// The generator sits last, after the hand-written phrases.
    #[test]
    fn generator_is_the_last_entry() {
        let a = app();
        let last = a.arpeggio_patterns.last().expect("no arpeggio patterns");
        assert_eq!(last.name, crate::model::GENERATOR_NAME);
    }

    /// Selecting it must replace the placeholder with a real phrase.
    #[test]
    fn selecting_the_generator_builds_a_phrase() {
        let mut a = app();
        pick_generator(&mut a);
        let n = a.intervals_input.split_whitespace().count();
        assert!(n >= 8, "phrase of only {n} steps: {:?}", a.intervals_input);
    }

    /// A new phrase after every pass - that is the whole point of the entry.
    #[test]
    fn each_pass_brings_a_different_phrase() {
        let mut a = app();
        pick_generator(&mut a);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..30 {
            seen.insert(a.intervals_input.clone());
            a.advance_chord();
        }
        assert!(seen.len() > 3, "only {} distinct phrases in 30 passes", seen.len());
    }

    /// The fixed patterns must be left alone.
    #[test]
    fn hand_written_patterns_are_not_regenerated() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.secondary_item_selected(0);
        let first = a.intervals_input.clone();
        for _ in 0..10 { a.advance_chord(); }
        assert_eq!(a.intervals_input, first, "a fixed pattern was overwritten");
    }

    /// Leaving the mode and coming back keeps the generator selected AND brings
    /// a new phrase - both halves matter, and the selection half was broken:
    /// set_mode zeroed the pattern index.
    #[test]
    fn leaving_and_returning_regenerates() {
        let mut a = app();
        pick_generator(&mut a);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..30 {
            seen.insert(a.intervals_input.clone());
            a.set_mode(AppMode::Chords as i32);
            a.set_mode(AppMode::Arpeggios as i32);
            assert_eq!(
                a.arpeggio_patterns[a.selected_library_idx].name,
                crate::model::GENERATOR_NAME,
                "the generator selection was lost on the way back"
            );
        }
        assert!(seen.len() > 3, "only {} distinct phrases across 30 round trips", seen.len());
    }

    /// A fixed pattern must survive the round trip too, unchanged.
    #[test]
    fn a_fixed_pattern_survives_the_round_trip() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.item_selected(1);
        let name = a.arpeggio_patterns[1].name.clone();
        let phrase = a.intervals_input.clone();
        a.set_mode(AppMode::Scales as i32);
        a.set_mode(AppMode::Arpeggios as i32);
        assert_eq!(a.arpeggio_patterns[a.selected_library_idx].name, name);
        assert_eq!(a.intervals_input, phrase, "a fixed pattern changed on the way back");
    }

    /// Whatever it builds has to be playable by the exercise logic.
    #[test]
    fn generated_phrases_drive_the_exercise() {
        let mut a = app();
        pick_generator(&mut a);
        for _ in 0..20 {
            let chord = a.chords[a.current_chord_index].clone();
            let steps = a.ordered_active_indices(&chord);
            assert!(!steps.is_empty(), "phrase {:?} yielded no steps", a.intervals_input);
            let names = chord.quality.interval_names();
            for s in &steps {
                assert!(s.degree < names.len(), "step points past the chord");
            }
            a.advance_chord();
        }
    }

    /// A formula played through STAYS. It is the exercise itself, not a step
    /// inside one, so finishing it starts another lap rather than drawing a
    /// replacement; a different formula comes from the settings.
    #[test]
    fn finishing_a_formula_keeps_it() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        let mask = a.formula_mask;
        assert!(mask != 0, "entering the mode should draw a formula");

        for (n, pc) in a.formula_pitches().into_iter().enumerate() {
            a.onset_id = n as u64 + 1;
            ear_hears(&mut a, pc);
        }
        assert!(a.formula_collected.iter().all(|&c| c), "the formula was not collected");

        // Past the hold time, and once the last note has died away, the lap
        // ends - see `a_lap_waits_for_its_own_notes_to_die`.
        ear_silent(&mut a);
        frames_pass(&mut a, 1, FORMULA_LAP_PAUSE + 0.01);
        assert_eq!(a.formula_mask, mask, "a new formula was drawn");
        assert!(
            a.formula_collected.iter().all(|&c| !c),
            "the marks did not start again"
        );
    }

    /// A single frame of the fast estimate credits nothing.
    ///
    /// It names a neighbour now and then - the root crediting the semitone
    /// beside it was the shape this took on the guitar - and a mark here never
    /// expires. Held for four ticks, the same reading counts.
    #[test]
    fn one_frame_of_the_fast_estimate_is_not_enough() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.strict_formulas = true;
        a.onset_id = 2;
        let pitches = a.formula_pitches();

        // Flickering between two of them: neither ever settles.
        ready(&mut a);
        for _ in 0..8 {
            for &pc in pitches.iter().take(2) {
                ear_frame(&mut a, Some(pc));
            }
        }
        assert!(
            a.formula_collected.iter().all(|&c| !c),
            "a flickering estimate marked functions off"
        );

        // Held still, it counts.
        ear_hears(&mut a, pitches[0]);
        assert!(a.formula_collected[0], "a steady reading did not count");
    }

    /// The strict rule is what the mode runs on, and the loose one - what the
    /// note modes use - is still there to compare against. It credits whatever
    /// is sounding, which after a few notes is most of the formula: 110 things
    /// nobody played over 49 notes, against 33.
    #[test]
    fn strict_is_the_default_and_loose_still_works() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        assert!(a.strict_formulas, "the strict rule is what the mode runs on");

        a.strict_formulas = false;
        a.onset_id = 2;
        let pc = a.formula_pitches()[0];
        // The model's pitch head alone, on the first frame after it is seen.
        ready(&mut a);
        a.last_pitches = [0.0; 12];
        a.last_pitches[pc % 12] = 1.0;
        ear_frame(&mut a, None);
        assert!(a.formula_collected[0], "the note modes would have counted this");
    }

    /// A finished lap moves on even when the player has stopped playing.
    ///
    /// The judge is only called when the model answers, and the model is only
    /// asked while the context window is nine tenths full - so when the hands
    /// come off the strings it stops being called, and the lap that timed itself
    /// there stopped with it: every function green and no next formula, until
    /// something was played again.
    #[test]
    fn a_finished_lap_moves_on_in_silence() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.strict_formulas = true;
        for (n, &pc) in a.formula_pitches().iter().enumerate() {
            a.onset_id = n as u64 + 1;
            ear_hears(&mut a, pc);
        }
        assert!(a.formula_collected.iter().all(|&c| c), "the formula was not collected");

        // Silence: the ear lets go, then nothing is asked of the model at all -
        // only frames go by, which is what happens when the hands come off.
        ear_silent(&mut a);
        frames_pass(&mut a, 200, 0.016);
        assert!(
            a.formula_collected.iter().all(|&c| !c),
            "the lap never moved on with nobody playing"
        );
    }

    /// Paused in Formulas is "your turn": nothing is judged until it is let go.
    #[test]
    fn a_paused_formula_judges_nothing() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.strict_formulas = true;
        a.onset_id = 2;
        let pc = a.formula_pitches()[0];
        ready(&mut a);
        a.paused = true;
        for _ in 0..40 {
            ear_frame(&mut a, Some(pc));
        }
        assert!(
            a.formula_collected.iter().all(|&c| !c),
            "a paused formula was still being judged"
        );

        a.paused = false;
        ear_hears(&mut a, pc);
        assert!(a.formula_collected[0], "letting go did not start it again");
    }

    /// A finished lap moves on after its pause, and holds for a moment so that
    /// what was ringing when it ended cannot walk into it.
    ///
    /// The hold is timed rather than tied to the next attack. Tied to the
    /// attack it blocked everything until one was detected, and an attack comes
    /// from a transient, which soft playing does not always give - so the first
    /// function of a new lap could take seconds.
    #[test]
    fn a_lap_moves_on_and_holds_for_a_moment() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.strict_formulas = true;
        let pitches = a.formula_pitches();
        for (n, &pc) in pitches.iter().enumerate() {
            a.onset_id = n as u64 + 1;
            ear_hears(&mut a, pc);
        }
        assert!(a.formula_collected.iter().all(|&c| c), "the formula was not collected");

        frames_pass(&mut a, 1, FORMULA_LAP_PAUSE + 0.01);
        assert!(a.formula_collected.iter().all(|&c| !c), "the lap did not move on");

        // The last note still ringing, straight away: held.
        for _ in 0..EAR_WINDOW {
            ear_frame(&mut a, Some(pitches[0]));
        }
        assert!(
            a.formula_collected.iter().all(|&c| !c),
            "the decay of the last lap walked into the new one"
        );

        // A moment later the same note, played, counts like any other.
        frames_pass(&mut a, 1, LAP_HOLD);
        for _ in 0..EAR_WINDOW {
            ear_frame(&mut a, Some(pitches[0]));
        }
        assert!(a.formula_collected[0], "the hold never let go");
    }

    /// One reading per audio frame: no more, no less.
    ///
    /// Sampled off the UI clock the same frame was counted two or three times,
    /// and one bad reading could carry the vote by itself. Sampled off the
    /// model's answers it stopped whenever the model did - and the model is
    /// asked only while the context window is nine tenths full, which playing
    /// one note at a time never manages. Both were measured on the guitar, and
    /// both were worse.
    #[test]
    fn a_frame_votes_once_however_often_it_is_looked_at() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.strict_formulas = true;
        a.onset_id = 2;
        let stray = a.formula_pitches()[1];

        // One frame, looked at twenty times: one vote, so nothing counts.
        ready(&mut a);
        a.cqt_pitch = Some(stray);
        a.audio_frames += 1;
        for _ in 0..20 {
            a.tick(0.0);
        }
        assert!(
            a.formula_collected.iter().all(|&c| !c),
            "one frame filled the vote by being looked at repeatedly"
        );

        // Four more frames of it, and it is evidence.
        for _ in 0..(EAR_VOTES - 1) {
            ear_frame(&mut a, Some(stray));
        }
        assert!(a.formula_collected[1], "four readings of five did not count");
    }

    /// A kept formula comes back as a set, with nothing marked off - and in
    /// whatever key is on screen, because a formula has none of its own.
    #[test]
    fn a_favourite_comes_back_whole() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        let mask = crate::formulas::parse("1 b3 5 b7").unwrap();
        let key = a.formula_key_name.clone();
        a.load_formula(mask);
        assert_eq!(a.formula_mask, mask, "the set did not come back");
        assert_eq!(a.formula_key_name, key, "it brought a key of its own");
        assert_eq!(a.formula_collected.len(), 4, "the marks do not fit the set");
        assert!(a.formula_collected.iter().all(|&c| !c), "it came back half played");
    }

    /// A settled reading counts. Judging runs where the answers arrive, so each
    /// reading in the vote is a new one - sampling it on the UI clock instead
    /// counted the same audio frame twice and let one bad reading through.
    #[test]
    fn a_settled_reading_counts() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.strict_formulas = true;
        a.onset_id = 2;
        let pc = a.formula_pitches()[0];

        ear_hears(&mut a, pc);
        assert!(a.formula_collected[0], "a settled reading did not count");
    }

    /// In order: the set is played low to high, and a function out of turn
    /// waits. Nothing is refused loudly - it simply does not count yet.
    #[test]
    fn in_order_takes_the_lowest_function_first() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.strict_formulas = true;
        a.formula_in_order = true;
        a.onset_id = 2;
        let pitches = a.formula_pitches();

        // The second function first: it waits.
        ear_hears(&mut a, pitches[1]);
        assert!(
            a.formula_collected.iter().all(|&c| !c),
            "a function out of turn was counted"
        );

        // The first, then the second: both count, in that order.
        a.onset_id += 1;
        ear_hears(&mut a, pitches[0]);
        assert!(a.formula_collected[0], "the function due did not count");
        a.onset_id += 1;
        ear_hears(&mut a, pitches[1]);
        assert!(a.formula_collected[1], "the next one due did not count");
    }

    /// Off, which is the default, the set is a set: any of them, any order.
    #[test]
    fn out_of_order_is_the_default() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.strict_formulas = true;
        assert!(!a.formula_in_order, "a formula is unordered unless asked");
        a.onset_id = 2;
        let pitches = a.formula_pitches();
        ear_hears(&mut a, pitches[2]);
        assert!(a.formula_collected[2], "the set refused a function it should take");
    }

    /// One stray frame does not undo a steady reading. A run had to start over
    /// on any flicker, which at half a second to the note was rarely finished
    /// before the next note began - the vote survives it.
    #[test]
    fn a_stray_frame_does_not_reset_the_evidence() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.strict_formulas = true;
        a.onset_id = 2;
        let pitches = a.formula_pitches();
        let (played, stray) = (pitches[0], pitches[1]);

        // The note, one stray reading, then the note again: four of five.
        ready(&mut a);
        let frames = [played, stray, played, played, played];
        for (n, pc) in frames.iter().enumerate() {
            ear_frame(&mut a, Some(*pc));
            if n + 1 < frames.len() {
                assert!(!a.formula_collected[0], "counted before the evidence was in");
            }
        }
        assert!(a.formula_collected[0], "a stray frame threw away four good ones");
    }

    /// A chord name credits nothing here, however sure the model is of it. The
    /// marks never expire, so a function lit by a name nobody played would stay
    /// lit for the rest of the exercise.
    #[test]
    fn a_chord_name_alone_credits_nothing() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.strict_formulas = true;
        a.formula_random_key = false;
        a.formula_key_setting = "C".to_string();
        a.rekey_formula();
        // Nothing heard: no single frame, no pitch vector. Only the name.
        a.cqt_pitch = None;
        a.last_pitches = [0.0; 12];
        a.prev_pitches = [0.0; 12];
        for n in 0..20u64 {
            a.onset_id = n;
            a.check_progress_with_ai(0.05, "C Maj7", 0.99);
        }
        assert!(
            a.formula_collected.iter().all(|&c| !c),
            "the chord name credited a function on its own"
        );
    }

    /// And it stays in its key. A lap ending is not a reason to move the
    /// exercise; the key is drawn for a new formula, not for a new lap of the
    /// one being played.
    #[test]
    fn a_lap_does_not_move_the_key() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.formula_random_key = true;
        let key = a.formula_key_name.clone();
        let mask = a.formula_mask;

        for lap in 0..5u64 {
            for (n, pc) in a.formula_pitches().into_iter().enumerate() {
                a.onset_id = lap * 12 + n as u64 + 1;
                ear_hears(&mut a, pc);
            }
            a.cqt_pitch = None;
            a.check_progress_with_ai(FORMULA_LAP_PAUSE + 0.01, "...", 0.0);
            assert_eq!(a.formula_key_name, key, "the lap moved the key");
            assert_eq!(a.formula_mask, mask, "the lap moved the formula");
        }
    }

    #[test]
    fn over_a_chord_the_lap_moves_the_placement_and_keeps_the_chord() {
        let mut a = app();
        a.formula_exercise = 1;
        a.set_mode(AppMode::Formulas as i32);
        let chord = a.formula_chord.clone().expect("no chord to play over");
        let mask = a.formula_mask;
        let mut degrees = std::collections::HashSet::new();
        for _ in 0..40 {
            degrees.insert(a.formula_degree);
            a.restart_formula();
            assert_eq!(a.formula_mask, mask, "the formula moved with the placement");
            let now = a.formula_chord.clone().unwrap();
            assert_eq!(now.root, chord.root, "the chord moved");
        }
        assert!(degrees.len() > 1, "the placement never moved");
        // And the root the exercise is read from is the chord's root plus the
        // degree, which is what makes a placement a placement.
        let root = a.formula_chord.as_ref().unwrap().root as usize;
        assert_eq!(a.formula_root, (root + a.formula_degree) % 12);
    }

    #[test]
    fn over_the_changes_the_lap_moves_the_chord() {
        let mut a = app();
        a.formula_exercise = 2;
        a.set_mode(AppMode::Formulas as i32);
        // Whatever the library gave it, the tune has to be walked in order.
        a.chords = vec![
            Chord { root: NoteName::C, quality: ChordQuality::Minor7 },
            Chord { root: NoteName::F, quality: ChordQuality::Dominant7 },
            Chord { root: NoteName::Bf, quality: ChordQuality::Major7 },
        ];
        a.current_chord_index = 0;
        let mask = a.formula_mask;
        let mut seen = vec![];
        for _ in 0..6 {
            a.restart_formula();
            seen.push(a.formula_chord.as_ref().unwrap().root);
            assert_eq!(a.formula_mask, mask, "the formula did not survive the chord");
        }
        assert_eq!(
            seen,
            vec![NoteName::F, NoteName::Bf, NoteName::C,
                 NoteName::F, NoteName::Bf, NoteName::C],
            "the changes are walked in order and round again",
        );
    }

    #[test]
    fn asking_for_a_kind_of_placement_gets_that_kind() {
        let mut a = app();
        a.formula_exercise = 1;
        a.formula_placement_want = 1; // spells the chord out
        a.set_mode(AppMode::Formulas as i32);
        // A chord and a formula chosen rather than drawn: over some pairs no
        // placement spells the chord out at all, and the draw then answers with
        // what exists rather than refusing - see `draw_placement`. That
        // fallback is the subject of its own test; this one is about getting
        // the kind that was asked for.
        a.formula_mask = crate::formulas::parse("1 b3 5 b7").unwrap();
        a.formula_collected = vec![false; 4];
        a.formula_chord = Some(Chord { root: NoteName::C, quality: ChordQuality::Minor7 });
        a.place_over_chord();
        for _ in 0..30 {
            assert_eq!(
                a.formula_verdict,
                Some(crate::formulas::Verdict::Defines),
                "degree {}",
                a.formula_degree,
            );
            a.restart_formula();
        }
    }

    /// The key does move when the settings say so - random draws a fresh one.
    #[test]
    fn a_random_key_is_drawn_when_asked_for() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.formula_random_key = true;
        let mask = a.formula_mask;
        let mut keys = std::collections::HashSet::new();
        for _ in 0..40 {
            keys.insert(a.formula_key_name.clone());
            a.rekey_formula();
            assert_eq!(a.formula_mask, mask, "the formula changed along with the key");
        }
        assert!(keys.len() > 1, "the key never moved");
    }

    /// A key typed into the settings is a key asked for.
    #[test]
    fn a_chosen_key_stays_put() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        a.formula_random_key = false;
        a.formula_key_setting = "Ab".to_string();
        for _ in 0..10 {
            a.rekey_formula();
            assert_eq!(a.formula_key_name, "Ab", "the key wandered off the chosen one");
        }
    }

    /// The arrows are the only way to ask for a different formula by hand, so
    /// they have to draw one whichever way they point.
    #[test]
    fn the_arrows_draw_another_formula() {
        let mut a = app();
        a.set_mode(AppMode::Formulas as i32);
        let mut masks = std::collections::HashSet::new();
        for i in 0..40 {
            masks.insert(a.formula_mask);
            a.step_chord(if i % 2 == 0 { 1 } else { -1 });
            assert_eq!(
                a.formula_collected.len(),
                a.formula_mask.count_ones() as usize,
                "the marks do not match the formula drawn"
            );
        }
        assert!(masks.len() > 1, "the arrows never changed the formula");
    }
}
