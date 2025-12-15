use std::sync::{Arc, Mutex};
use std::collections::VecDeque; 
use crate::audio::AudioAnalysis;
use crate::brain::ChordBrain;
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
    pub sensitivity: f32,
    pub tail_threshold: f32,
    
    pub bass_boost_enabled: bool,
    pub bass_boost_gain: f32,
    pub input_gain: f32,
    pub noise_gate: f32,
    
    pub intervals_input: String,
    pub saved_intervals_input: String,
    
    pub random_mode: bool,
    pub chord_history: VecDeque<(String, f32)>, 
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
            transition_delay: 0.6,
            sensitivity: 0.3,
            tail_threshold: 0.3,
            
            bass_boost_enabled: true,
            bass_boost_gain: 5.0,
            input_gain: 2.0,
            noise_gate: 0.02,
            
            intervals_input: "1 3 5".to_string(),
            saved_intervals_input: "1 3 5".to_string(),
            
            random_mode: false,
            chord_history: VecDeque::with_capacity(20),
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
            // FIX: Usunięto gwiazdki (*token, *name)
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
        self.chord_history.clear();
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
        if confidence < self.sensitivity { 
            self.success_timer = 0.0;
            return; 
        }

        let (ai_root, ai_qual) = self.parse_ai_prediction(ai_prediction);
        let target_chord = &self.chords[self.current_chord_index];
        let target_root = target_chord.root;

        let active_indices = self.get_active_indices(target_chord);
        let all_targets = target_chord.get_target_indices(); 

        match self.app_mode {
            AppMode::Chords => {
                let target_qual_str = target_chord.quality.to_string();
                let mut match_found = false;

                if let Some(r) = ai_root {
                    if r == target_root {
                        if ai_qual == target_qual_str { match_found = true; }
                        if target_qual_str == "Maj7" && ai_qual == "" { match_found = true; }
                    }
                }
                if match_found { self.success_timer += dt; } else { self.success_timer = 0.0; }
                if self.success_timer > self.transition_delay { self.advance_chord(); }
            },
            
            AppMode::Intervals | AppMode::Scales | AppMode::Arpeggios => {
                if self.current_note_step >= active_indices.len() { return; }

                let internal_idx = active_indices[self.current_note_step];
                let target_note_idx = all_targets[internal_idx];
                let target_note_enum = NoteName::from_index(target_note_idx);

                let mut note_match = false;
                if let Some(r) = ai_root {
                    if r == target_note_enum {
                        note_match = true;
                    }
                }

                if note_match { self.success_timer += dt; } else { self.success_timer = 0.0; }

                if self.success_timer > self.transition_delay {
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
        self.current_chord_index = (self.current_chord_index + 1) % self.chords.len();
        self.update_collected_notes_size();
    }

    pub fn sync_audio_settings(&self) {
        if let Ok(mut state) = self.analysis_state.lock() {
            state.input_gain = self.input_gain;
            state.noise_gate = self.noise_gate;
            state.bass_boost_enabled = self.bass_boost_enabled;
            state.bass_boost_gain = self.bass_boost_gain;
        }
    }
}
