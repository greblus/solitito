use std::sync::{Arc, Mutex};
use std::collections::VecDeque; 
use crate::audio::AudioAnalysis;
use crate::brain::ChordBrain;
use crate::rng::Rng;
use crate::model::{Chord, NoteName, Song, load_songs, load_all_scale_definitions, load_arpeggio_patterns, ScaleDefinition, ChordQuality};

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum AppMode { 
    Chords = 0, 
    Intervals = 1, 
    Scales = 2,
    Arpeggios = 3 
}

impl From<i32> for AppMode {
    fn from(val: i32) -> Self {
        match val {
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
            app_mode: AppMode::Intervals,
            selected_library_idx: 0,
            secondary_index: 0,
            
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
            
            random_mode: false,
            step_order: vec![],
            start_hint: None,
            rng: Rng::default(),
            chord_history: VecDeque::with_capacity(20),
            last_pitches: [0.0; 12],
        }
    }
    
    pub fn get_active_indices(&self, chord: &Chord) -> Vec<usize> {
        let all_names = chord.quality.interval_names(); 
        let user_tokens: Vec<&str> = self.intervals_input.split_whitespace().collect();
        let mut indices = Vec::new();
        
        if self.app_mode == AppMode::Arpeggios {
            for token in user_tokens {
                let target_idx = match token {
                    "1" | "8" => 0,
                    "3" => 1,
                    "5" => 2,
                    "7" => 3,
                    "9" => if all_names.len() > 4 { 4 } else if all_names.len() > 1 { 1 } else { 0 },
                    _ => 999
                };
                
                if target_idx < all_names.len() {
                    indices.push(target_idx);
                } else if token == "9" {
                    if let Some(pos) = all_names.iter().position(|n| n.contains("2") || n.contains("9")) {
                        indices.push(pos);
                    }
                }
            }
        } else {
            for token in user_tokens {
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
                    if is_match { indices.push(idx); break; } 
                 }
            }
        }
        
        if indices.is_empty() {
             if !all_names.is_empty() { vec![0] } else { vec![] }
        } else {
            indices
        }
    }

    /// `get_active_indices` in play order. Without randomisation this is the
    /// identity; with it, the stored permutation. The UI renders the same order,
    /// so the highlight still runs left to right instead of jumping around.
    pub fn ordered_active_indices(&self, chord: &Chord) -> Vec<usize> {
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

        self.app_mode = new_mode;
        self.selected_library_idx = 0;
        self.secondary_index = 0; 
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
        }
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

    pub fn check_progress_with_ai(&mut self, dt: f32, ai_prediction: &str, confidence: f32) {
        if self.chords.is_empty() { return; }
        
        let (ai_root, ai_qual) = self.parse_ai_prediction(ai_prediction);
        let target_chord = &self.chords[self.current_chord_index];
        let target_root = target_chord.root;

        let active_indices = self.ordered_active_indices(target_chord);
        let all_targets = target_chord.get_target_indices(); 

        match self.app_mode {
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
                    self.advance_chord(); 
                }
            },
            
            AppMode::Intervals | AppMode::Scales | AppMode::Arpeggios => {
                // These modes are NOT gated by chord confidence. `confidence` is now
                // the product of root and quality confidence, so an uncertain quality
                // would block a note that is perfectly audible. Only the pitch head
                // matters here - F1 0.90 against ~80% for the chord name.
                if self.current_note_step >= active_indices.len() { return; }

                let internal_idx = active_indices[self.current_note_step];
                let target_note_idx = all_targets[internal_idx];
                let target_note_enum = NoteName::from_index(target_note_idx);

                // Target pitch class (0..11); NoteName ordering matches the
                // pitch_logits output (C, Db, D, ...).
                let target_pc = target_note_idx % 12;
                let p_target = self.last_pitches[target_pc];
                let p_max = self.last_pitches.iter().cloned().fold(0.0f32, f32::max);

                // A note counts as played when the pitch head is confident AND it
                // is the loudest class in the window. The second condition prevents
                // crediting a note still ringing from the previous step.
                let mut note_match =
                    p_target >= self.note_threshold && p_target >= p_max * 0.9;

                // Agreement from the root head is independent confirmation - for a
                // single note the model describes it as the root.
                if let Some(r) = ai_root {
                    if r == target_note_enum && confidence >= self.chord_confidence {
                        note_match = true;
                    }
                }

                if note_match { self.success_timer += dt; } else { self.success_timer = 0.0; }

                let note_delay = 0.12; 
                if self.success_timer > note_delay {
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
mod tests {
    use super::*;
    use crate::audio::{CTX_FRAMES, TOTAL_FEATURES};

    /// MyApp needs the shared audio state; nothing here touches it.
    fn app() -> MyApp {
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
