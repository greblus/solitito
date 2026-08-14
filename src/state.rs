use std::sync::{Arc, Mutex};
use std::collections::VecDeque; 
use crate::audio::AudioAnalysis;
use crate::brain::ChordBrain;
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
}

impl From<i32> for AppMode {
    fn from(val: i32) -> Self {
        match val {
            4 => AppMode::Fretboard,
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

pub struct MyApp {
    pub analysis_state: Arc<Mutex<AudioAnalysis>>,
    pub brain: Option<Arc<Mutex<ChordBrain>>>,
    
    pub song_library: Vec<Song>,
    pub scale_definitions: Vec<ScaleDefinition>,
    pub arpeggio_patterns: Vec<ScaleDefinition>,
    
    pub app_mode: AppMode,
    pub selected_library_idx: usize,
    pub secondary_index: usize, 
    /// Which arpeggio pattern was chosen, kept across mode switches. Leaving the
    /// mode used to throw the choice away and drop you back on the first pattern.
    saved_arpeggio_index: usize,
    
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
    /// Permutation of `active_indices`, regenerated whenever the chord changes.
    /// Reshuffling every frame would make the target jump around; the order has
    /// to stay fixed for as long as the chord is being played.
    step_order: Vec<usize>,
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
    /// Pitch class sounding in the last CQT frame - see `audio::mono_pitch`.
    /// `None` while the gate is shut or the estimate is weak.
    pub cqt_pitch: Option<usize>,
    /// Require the notes one at a time. Off, the CQT only ever ADDS a way to
    /// pass, and a strummed chord still walks its intervals - the model's pitch
    /// head is polyphonic and reports every tone at once, which a monophonic
    /// estimate cannot. On, the CQT overrules the model and one pluck credits
    /// one step.
    pub single_notes: bool,
    /// Attack counter, mirrored from the audio thread.
    pub onset_id: u64,
    /// Which pitch class was credited last, and on which attack. A step is not
    /// credited twice on one pluck: an arpeggio that asks for the same note
    /// twice in a row would otherwise run through both on a single ringing
    /// string, with nothing played in between.
    pub credited: Option<(usize, u64)>,
}

impl MyApp {
    pub fn new(state: Arc<Mutex<AudioAnalysis>>, brain: Option<Arc<Mutex<ChordBrain>>>) -> Self {
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
            brain,
            song_library,
            scale_definitions,
            arpeggio_patterns,
            app_mode: AppMode::Fretboard,
            selected_library_idx: 0,
            secondary_index: 0,
            saved_arpeggio_index: 0,
            
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
            prev_chord_index: None,
            prev_status: MatchStatus::None,
            
            bass_boost_enabled: true,
            bass_boost_gain: 5.0,
            noise_gate: 0.02,
            
            intervals_input: "1 3 5".to_string(),
            saved_intervals_input: "1 3 5".to_string(),
            
            paused: false,
            random_mode: false,
            step_order: vec![],
            start_hint: None,
            play_order: Vec::new(),
            play_pos: 0,
            region: Region::default(),
            fret_target: None,
            rng: Rng::default(),
            chord_history: VecDeque::with_capacity(20),
            last_pitches: [0.0; 12],
            prev_pitches: [0.0; 12],
            cqt_pitch: None,
            single_notes: false,
            onset_id: 0,
            credited: None,
        }
    }
    
    pub fn get_active_indices(&self, chord: &Chord) -> Vec<Step> {
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
        }
    }

    /// Rebuilds the playing order and starts it from the beginning.
    fn rebuild_play_order(&mut self) {
        self.play_order = (0..self.chords.len()).collect();
        if self.random_mode {
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
        if self.app_mode != AppMode::Arpeggios {
            return;
        }
        let is_generator = self.arpeggio_patterns.get(self.secondary_index)
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

        if self.app_mode == AppMode::Arpeggios {
            self.saved_arpeggio_index = self.secondary_index;
        }

        self.app_mode = new_mode;
        self.selected_library_idx = 0;
        // Scales use this for the key and start from C; Arpeggios use it for the
        // pattern, and coming back should land on the one you picked - which for
        // the generator means a freshly built phrase, not the first fixed one.
        self.secondary_index = if new_mode == AppMode::Arpeggios {
            self.saved_arpeggio_index
        } else {
            0
        };
        self.reload_library_content();
    }

    pub fn item_selected(&mut self, index: i32) {
        self.selected_library_idx = index as usize;
        self.reload_library_content();
    }
    
    pub fn secondary_item_selected(&mut self, index: i32) {
        self.secondary_index = index as usize;
        if self.app_mode == AppMode::Scales || self.app_mode == AppMode::Arpeggios {
            self.reload_library_content();
        }
    }

    fn reload_library_content(&mut self) {
        match self.app_mode {
            AppMode::Chords | AppMode::Intervals => {
                if self.selected_library_idx < self.song_library.len() {
                    let song = &self.song_library[self.selected_library_idx];
                    self.song_title = song.title.clone();
                    self.chords = song.chords.clone();
                }
            }
            AppMode::Arpeggios => {
                if self.selected_library_idx < self.song_library.len() {
                    let song = &self.song_library[self.selected_library_idx];
                    self.song_title = song.title.clone();
                    self.chords = song.chords.clone();
                }
                if self.secondary_index < self.arpeggio_patterns.len() {
                    let pattern = &self.arpeggio_patterns[self.secondary_index];
                    self.intervals_input = pattern.names.join(" ");
                } else {
                    self.intervals_input = "1 3 5 7".to_string(); 
                }
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
            AppMode::Fretboard => {
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

    pub fn reset_logic_state(&mut self) {
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
        let target = pc % 12;

        // 1. One frame, one note - no window to smear across.
        if self.cqt_pitch == Some(target) {
            return true;
        }

        let p_target = self.last_pitches[target];
        let p_max = self.last_pitches.iter().cloned().fold(0.0f32, f32::max);

        // 2. The model, where the target owns the window: a held note, or the
        //    only one in it. Both model branches are suppressed when the CQT
        //    names a DIFFERENT note: everything the model can say rests on
        //    0.77 s of audio, and against one frame that is a claim about the
        //    past. The root head below is left as a second opinion, so a note
        //    the CQT reads badly still has a way through.
        let stale = self.single_notes && self.cqt_pitch.is_some_and(|now| now != target);
        if !stale && p_target >= self.note_threshold && p_target >= p_max * 0.9 {
            return true;
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
            && had_content
            && p_target >= self.note_threshold
            && rise > 0.05
            && rise >= best_rise * 0.9
        {
            return true;
        }

        // 4. The root head as independent confirmation - a single note is
        //    reported by the model as the root.
        matches!(ai_root, Some(r) if r == NoteName::from_index(target))
            && confidence >= self.chord_confidence
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
        let prev = self.fret_target;
        self.fret_target = self.region.draw(&mut self.rng, prev);
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
    }

    pub fn check_progress_with_ai(&mut self, dt: f32, ai_prediction: &str, confidence: f32) {

        // The fretboard trainer has no song, so it runs before the chord guard.
        if self.app_mode == AppMode::Fretboard {
            let (ai_root, _) = self.parse_ai_prediction(ai_prediction);
            let Some(target) = self.fret_target else { self.next_fret_target(); return; };
            let fresh = !self.single_notes
                || match self.credited {
                    Some((pc, onset)) => pc != target % 12 || onset != self.onset_id,
                    None => true,
                };
            if fresh && self.note_is_sounding(target, ai_root, confidence) {
                self.success_timer += dt;
                self.match_status = MatchStatus::Exact;
            } else {
                self.success_timer = 0.0;
                self.match_status = MatchStatus::None;
            }
            if self.success_timer > self.transition_delay {
                if self.paused {
                    self.success_timer = self.transition_delay;
                } else {
                    self.credited = Some((target % 12, self.onset_id));
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
            // Returned above, before the chord guard - it has no chords to check.
            AppMode::Fretboard => {}
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
                    }
                }
                
                // Green means the target chord was heard, clearly enough to be
                // trusted - the same condition the bucket paints green on. With
                // this on, that is the whole test: no waiting for it to be held,
                // and no decay afterwards to undo it. Nothing runs away, because
                // advancing changes the target and the chord still ringing stops
                // matching it.
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
                let fresh = !self.single_notes
                    || match self.credited {
                        Some((pc, onset)) => pc != target_note_idx % 12 || onset != self.onset_id,
                        None => true,
                    };
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
                    self.credited = Some((target_note_idx % 12, self.onset_id));
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
        // Landing on a chord draws it a string to start from and an order for
        // its notes, exactly as arriving there by playing would. Without this
        // the suggestion stayed on whatever it was when the arrows started.
        self.reroll();
        self.update_collected_notes_size();
    }

    fn advance_chord(&mut self) {
        // Captured before the reset below wipes it: this is how the chord being
        // left behind was actually matched, and the strip reports that.
        let earned = self.match_status;
        self.success_timer = 0.0;
        self.current_note_step = 0;
        self.match_status = MatchStatus::None;
        // Scales hold a single "chord" - the whole scale - so the list has one
        // entry and the index never moves. Advancing there means a new KEY:
        // finish the scale, get another one somewhere else on the neck.
        if self.random_mode && self.app_mode == AppMode::Scales && !self.chords.is_empty() {
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
            self.cqt_pitch = state.cqt_pitch;
            self.onset_id = state.onset_id;
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

    /// MyApp needs the shared audio state; nothing here touches it.
    pub(crate) fn app() -> MyApp {
        let analysis = Arc::new(Mutex::new(AudioAnalysis {
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
        }));
        let mut a = MyApp::new(analysis, None);
        a.chords = vec![
            Chord { root: NoteName::C, quality: ChordQuality::Major7 },
            Chord { root: NoteName::D, quality: ChordQuality::Minor7 },
            Chord { root: NoteName::G, quality: ChordQuality::Dominant7 },
            Chord { root: NoteName::A, quality: ChordQuality::Minor7 },
        ];
        a.reset_logic_state();
        a
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

    /// Only Scales redraw the key - in Arpeggios secondary_index selects the
    /// PATTERN, and moving it would silently switch the exercise.
    #[test]
    fn arpeggios_do_not_have_their_secondary_index_hijacked() {
        let mut a = app();
        a.set_mode(AppMode::Arpeggios as i32);
        a.set_random_mode(true);
        let sec = a.secondary_index;
        for _ in 0..20 { a.advance_chord(); }
        assert_eq!(a.secondary_index, sec, "the arpeggio pattern was changed behind our back");
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
        a.cqt_pitch = Some(4);
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

    /// An arpeggio may ask for the same pitch class twice in a row. One pluck
    /// must credit one step, or a single ringing string walks through both.
    #[test]
    fn one_pluck_credits_one_step() {
        let mut a = app();
        a.single_notes = true;
        a.set_mode(AppMode::Fretboard as i32);
        a.transition_delay = 0.05;
        a.note_threshold = 0.5;
        let target = a.fret_target.unwrap();
        a.last_pitches = [0.0; 12];
        a.last_pitches[target % 12] = 1.0;
        a.credited = Some((target % 12, 0));      // this pluck already counted
        a.onset_id = 0;
        for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(a.fret_target, Some(target), "the same pluck counted twice");

        a.onset_id = 1;                            // strings struck again
        for _ in 0..10 { a.check_progress_with_ai(0.1, "Noise", 0.0); }
        assert_eq!(
            a.credited, Some((target % 12, 1)),
            "a fresh attack on the right note did not count"
        );
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
        a.transition_delay = 0.05;
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
        let idx = a.arpeggio_patterns.iter()
            .position(|p| p.name == crate::model::GENERATOR_NAME)
            .expect("generator entry missing from the arpeggio list");
        a.secondary_item_selected(idx as i32);
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
                a.arpeggio_patterns[a.secondary_index].name,
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
        a.secondary_item_selected(1);
        let name = a.arpeggio_patterns[1].name.clone();
        let phrase = a.intervals_input.clone();
        a.set_mode(AppMode::Scales as i32);
        a.set_mode(AppMode::Arpeggios as i32);
        assert_eq!(a.arpeggio_patterns[a.secondary_index].name, name);
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
}
