use std::sync::{Arc, Mutex};
use std::collections::VecDeque; 
use crate::audio::AudioAnalysis; 
use crate::brain::ChordBrain;
use crate::model::{Chord, NoteName, Song, load_songs, load_all_scale_definitions, ScaleDefinition, ChordQuality};

#[derive(PartialEq, Clone, Copy)]
pub enum AppMode { Songs, Scales }

pub struct MyApp {
    pub analysis_state: Arc<Mutex<AudioAnalysis>>,
    
    // Brain w Mutex, bo posiada stan wewnętrzny (bufor klatek)
    pub brain: Option<Arc<Mutex<ChordBrain>>>,
    
    pub song_library: Vec<Song>,
    pub scale_definitions: Vec<ScaleDefinition>,
    
    pub app_mode: AppMode,
    pub selected_song_idx: usize,
    pub scale_root: NoteName,
    pub selected_scale_def_idx: usize,
    
    pub song_title: String,
    pub chords: Vec<Chord>,
    pub current_chord_index: usize,
    
    pub success_timer: f32,
    pub sensitivity: f32,
    pub tail_threshold: f32,
    pub transition_delay: f32, 
    
    pub bass_boost_enabled: bool,
    pub bass_boost_gain: f32,
    pub input_gain: f32,
    pub noise_gate: f32,
    
    pub intervals_input: String,
    
    pub random_mode: bool,
    pub current_random_target: Option<usize>,
    pub current_sequence_index: usize,
    pub random_sequence: Vec<usize>,
    
    pub collected_notes: [bool; 12],
    pub stale_notes: [f32; 12],
    pub time_since_change: f32,
    pub total_time: f64,
    
    pub chord_history: VecDeque<(String, f32)>, 
}

impl MyApp {
    pub fn new(state: Arc<Mutex<AudioAnalysis>>, brain: Option<Arc<Mutex<ChordBrain>>>) -> Self {
        let song_library = load_songs();
        let scale_definitions = load_all_scale_definitions();
        
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
            app_mode: AppMode::Songs,
            selected_song_idx: 0,
            scale_root: NoteName::C,
            selected_scale_def_idx: 0,
            song_title: start_song.title,
            chords: start_song.chords,
            current_chord_index: 0,
            success_timer: 0.0,
            sensitivity: 0.15, 
            tail_threshold: 0.3,
            transition_delay: 0.6, 
            bass_boost_enabled: true,
            bass_boost_gain: 5.0,
            input_gain: 3.0,
            noise_gate: 0.0002,
            intervals_input: "1 3 5".to_string(),
            random_mode: false,
            current_random_target: None,
            current_sequence_index: 0,
            random_sequence: Vec::new(),
            collected_notes: [false; 12],
            stale_notes: [0.0; 12],
            time_since_change: 0.0,
            total_time: 0.0,
            chord_history: VecDeque::with_capacity(20),
        }
    }
    
    pub fn load_selected_song(&mut self) {
        if self.selected_song_idx < self.song_library.len() {
            let song = &self.song_library[self.selected_song_idx];
            self.song_title = song.title.clone();
            self.chords = song.chords.clone();
            self.current_chord_index = 0;
            self.reset_logic_state();
        }
    }

    pub fn build_scale_chord(&mut self) {
        if self.selected_scale_def_idx < self.scale_definitions.len() {
            let def = self.scale_definitions[self.selected_scale_def_idx].clone();
            self.song_title = def.name.clone();
            self.chords = vec![Chord { 
                root: self.scale_root, 
                quality: ChordQuality::CustomScale(def) 
            }];
            self.current_chord_index = 0;
            self.reset_logic_state();
        }
    }

    pub fn reset_logic_state(&mut self) {
        self.success_timer = 0.0;
        self.collected_notes = [false; 12];
        self.stale_notes = [0.0; 12];
        self.time_since_change = 0.0;
        self.current_sequence_index = 0;
        self.current_random_target = None;
        self.random_sequence.clear();
        self.chord_history.clear();
    }

    pub fn get_target_config_indices(&self) -> Vec<usize> {
        let parts: Vec<&str> = self.intervals_input.split_whitespace().collect();
        let mut indices = Vec::new();
        
        // POPRAWKA: Iteracja przez referencję &parts
        for p in &parts {
            match *p { 
                "1" => indices.push(0), 
                "2" | "b2" => indices.push(1),
                "3" | "b3" => indices.push(1),
                "4" => indices.push(3),
                "5" | "b5" => indices.push(2), 
                "6" => indices.push(3),
                "7" | "b7" => indices.push(3), 
                _ => {} 
            }
        }
        
        let mut simple_indices = Vec::new();
        // POPRAWKA: Iteracja przez referencję &parts
        for p in &parts {
            match *p {
                "1" => simple_indices.push(0),
                "3" => simple_indices.push(1),
                "5" => simple_indices.push(2),
                "7" => simple_indices.push(3),
                _ => {}
            }
        }
        simple_indices
    }

    pub fn is_note_active(&self, note_idx: usize, chroma: &[f32; 12]) -> bool {
        if self.stale_notes[note_idx] > 0.0 { return false; }
        let energy = chroma[note_idx];
        if energy < self.sensitivity { return false; }
        let max_energy = chroma.iter().fold(0.0f32, |a, &b| a.max(b));
        if energy < max_energy * 0.3 { return false; }
        true
    }

    pub fn update_stale_notes(&mut self, chroma: &[f32; 12]) {
        for i in 0..12 {
            let lock = self.stale_notes[i];
            if lock > 0.0 && chroma[i] < lock * self.tail_threshold { 
                self.stale_notes[i] = 0.0; 
            }
        }
    }

    pub fn check_progress(&mut self, dt: f32, chroma: &[f32; 12]) {
        self.total_time += dt as f64;
        self.time_since_change += dt;
        if self.time_since_change < self.transition_delay { return; }
        if self.chords.is_empty() { return; }
        
        let current_chord = &self.chords[self.current_chord_index];
        let all_targets = current_chord.get_target_indices();
        let config_indices = self.get_target_config_indices();
        
        let valid_indices: Vec<usize> = config_indices.into_iter()
            .filter(|&idx| idx < all_targets.len())
            .collect();

        if valid_indices.is_empty() { return; }

        let mut collected_count = 0;
        let mut required_hits = 0;

        for &internal_idx in &valid_indices {
            required_hits += 1;
            let note_idx = all_targets[internal_idx];
            if self.is_note_active(note_idx, chroma) { 
                self.collected_notes[note_idx] = true; 
            }
            if self.collected_notes[note_idx] { collected_count += 1; }
        }

        if required_hits > 0 && collected_count >= required_hits { 
            self.success_timer += dt; 
        } else { 
            self.success_timer = 0.0; 
        }

        if self.success_timer > 0.4 {
            self.success_timer = 0.0;
            self.collected_notes = [false; 12];
            self.stale_notes = *chroma;
            self.time_since_change = 0.0;
            self.current_chord_index = (self.current_chord_index + 1) % self.chords.len();
        }
    }

    pub fn sync_audio_settings(&self) {
        if let Ok(mut state) = self.analysis_state.lock() {
            state.bass_boost_enabled = self.bass_boost_enabled;
            state.bass_boost_gain = self.bass_boost_gain;
            state.input_gain = self.input_gain;
            state.noise_gate = self.noise_gate;
        }
    }
}
