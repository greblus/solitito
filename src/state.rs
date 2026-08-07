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
pub const START_STRINGS: [&str; 4] = ["E", "A", "D", "G"];

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
            
            bass_boost_enabled: true,
            bass_boost_gain: 5.0,
            noise_gate: 0.02,
            
            intervals_input: "1 3 5".to_string(),
            saved_intervals_input: "1 3 5".to_string(),
            
            paused: false,
            random_mode: false,
            step_order: vec![],
            start_hint: None,
            region: Region::default(),
            fret_target: None,
            rng: Rng::default(),
            chord_history: VecDeque::with_capacity(20),
            last_pitches: [0.0; 12],
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
            self.reroll();
        }
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
        self.current_chord_index = 0;
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
    /// drift apart. Gated on the pitch head rather than the chord name: F1 0.90
    /// against roughly 80% for the name.
    fn note_is_sounding(&self, pc: usize, ai_root: Option<NoteName>, confidence: f32) -> bool {
        let p_target = self.last_pitches[pc % 12];
        let p_max = self.last_pitches.iter().cloned().fold(0.0f32, f32::max);
        // Loudest class in the window as well as confident: without the second
        // condition a note still ringing from the previous step would score.
        if p_target >= self.note_threshold && p_target >= p_max * 0.9 {
            return true;
        }
        // The root head is independent confirmation - a single note is reported
        // by the model as the root.
        matches!(ai_root, Some(r) if r == NoteName::from_index(pc % 12))
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
            if self.note_is_sounding(target, ai_root, confidence) {
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
                let note_match = self.note_is_sounding(target_note_idx, ai_root, confidence);

                if note_match { self.success_timer += dt; } else { self.success_timer = 0.0; }

                let note_delay = 0.12;
                if self.paused && self.success_timer > note_delay {
                    self.success_timer = note_delay;
                }
                if !self.paused && self.success_timer > note_delay {
                    if self.current_note_step < self.collected_notes.len() {
                        self.collected_notes[self.current_note_step] = true;
                    }
                    self.current_note_step += 1;
                    self.success_timer = 0.0; 

                    if self.current_note_step >= active_indices.len() {
                        self.advance_chord();
                    }
                }
            }
        }
    }

    fn advance_chord(&mut self) {
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

        self.current_chord_index = if self.random_mode {
            // Never the same chord twice in a row - repeating it reads as the app
            // having failed to notice the previous one.
            self.rng.below_excluding(self.chords.len(), self.current_chord_index)
        } else {
            (self.current_chord_index + 1) % self.chords.len()
        };
        // A finished pass earns a new phrase when the generator is selected.
        self.regenerate_arpeggio();
        self.reroll();
        self.update_collected_notes_size();
    }

    pub fn sync_audio_settings(&self) {
        if let Ok(mut state) = self.analysis_state.lock() {
            state.noise_gate = self.noise_gate;
            state.bass_boost_enabled = self.bass_boost_enabled;
            state.bass_boost_gain = self.bass_boost_gain;
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
