// src/state.rs
use std::sync::{Arc, Mutex};
use std::collections::VecDeque; 
use crate::audio::{AudioAnalysis, LOG_BINS}; // Import stałej rozmiaru (128)
use crate::brain::ChordBrain;
use crate::model::{Chord, NoteName, Song, load_songs, load_all_scale_definitions, ScaleDefinition, ChordQuality};

#[derive(PartialEq, Clone, Copy)]
pub enum AppMode { Songs, Scales }

pub struct MyApp {
    pub analysis_state: Arc<Mutex<AudioAnalysis>>,
    pub brain: Option<Arc<ChordBrain>>,
    
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
    pub intervals_input: String,
    
    pub random_mode: bool,
    pub current_random_target: Option<usize>,
    pub current_sequence_index: usize,
    pub random_sequence: Vec<usize>,
    
    pub collected_notes: [bool; 12],
    pub stale_notes: [f32; 12],
    pub time_since_change: f32,
    pub total_time: f64,
    
    // Bufor dla AI (Log Bins)
    pub raw_input_for_ai: [f32; LOG_BINS], 
    
    pub prediction_buffer: VecDeque<(String, f32)>, 
    pub ai_prediction: String,
}

impl MyApp {
    pub fn new(state: Arc<Mutex<AudioAnalysis>>, brain: Option<Arc<ChordBrain>>) -> Self {
        let song_library = load_songs();
        let scale_definitions = load_all_scale_definitions();
        let start_song = if !song_library.is_empty() { song_library[0].clone() } else { Song { title: "No Songs".into(), chords: vec![] } };

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
            intervals_input: "1 3 5".to_string(),
            random_mode: false,
            current_random_target: None,
            current_sequence_index: 0,
            random_sequence: Vec::new(),
            collected_notes: [false; 12],
            stale_notes: [0.0; 12],
            time_since_change: 0.0,
            total_time: 0.0,
            raw_input_for_ai: [0.0; LOG_BINS], // Inicjalizacja zerami
            ai_prediction: String::from("AI: ..."),
            prediction_buffer: VecDeque::with_capacity(20),
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
            self.chords = vec![Chord { root: self.scale_root, quality: ChordQuality::CustomScale(def) }];
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
        self.prediction_buffer.clear();
    }

    pub fn get_target_config_indices(&self) -> Vec<usize> {
        let parts: Vec<&str> = self.intervals_input.split_whitespace().collect();
        let mut indices = Vec::new();
        for p in parts {
            if self.app_mode == AppMode::Scales {
                match p {
                    "1" => indices.push(0), "2" | "b2" | "9" | "b9" => indices.push(1), "3" | "b3" => indices.push(2),
                    "4" | "#4" | "11" | "#11" => indices.push(3), "5" | "b5" => indices.push(4), "6" | "b6" | "13" | "b13" => indices.push(5),
                    "7" | "b7" | "maj7" => indices.push(6), _ => {}
                }
            } else {
                match p {
                    "1" => indices.push(0), "3" | "b3" | "sus4" => indices.push(1), "5" | "b5" => indices.push(2),
                    "7" | "b7" | "6" => indices.push(3), _ => {}
                }
            }
        }
        indices
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
    
    fn pick_random_target(&mut self, options: &[usize]) {
        if options.is_empty() { return; }
        let seed = (self.total_time * 1000.0) as usize;
        let idx = seed % options.len();
        if options.len() > 1 {
            if let Some(current) = self.current_random_target {
                if options[idx] == options[current] {
                    let new_idx = (idx + 1) % options.len();
                    self.current_random_target = Some(new_idx);
                    return;
                }
            }
        }
        self.current_random_target = Some(idx);
    }
    
    fn generate_shuffled_sequence(&self, len: usize) -> Vec<usize> {
        let mut seq: Vec<usize> = (0..len).collect();
        if len < 2 { return seq; }
        let mut pseudo_seed = (self.total_time * 12345.0) as u64;
        for i in (1..len).rev() {
            pseudo_seed = pseudo_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (pseudo_seed as usize) % (i + 1);
            seq.swap(i, j);
        }
        seq
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

        if self.random_mode {
            if self.current_random_target.is_none() { self.pick_random_target(&valid_indices); }
            if let Some(target_logic_idx) = self.current_random_target {
                if target_logic_idx < valid_indices.len() {
                    let internal_idx = valid_indices[target_logic_idx];
                    let note_idx = all_targets[internal_idx];
                    if self.is_note_active(note_idx, chroma) { self.success_timer += dt; } else { self.success_timer = 0.0; }
                }
            }
            if self.success_timer > 0.4 {
                self.success_timer = 0.0;
                self.stale_notes = *chroma; 
                self.pick_random_target(&valid_indices);
            }
        } else if self.app_mode == AppMode::Scales {
            if self.current_sequence_index < valid_indices.len() {
                let internal_idx = valid_indices[self.current_sequence_index];
                let note_idx = all_targets[internal_idx];
                if self.is_note_active(note_idx, chroma) { self.current_sequence_index += 1; }
            } else { self.success_timer += dt; }
            if self.success_timer > 0.4 {
                self.success_timer = 0.0;
                self.collected_notes = [false; 12];
                self.stale_notes = *chroma; 
                self.current_sequence_index = 0;
                if self.random_mode { self.random_sequence = self.generate_shuffled_sequence(valid_indices.len()); }
            }
        } else {
            let mut collected_count = 0;
            let mut required_hits = 0;
            for &internal_idx in &valid_indices {
                required_hits += 1;
                let note_idx = all_targets[internal_idx];
                if self.is_note_active(note_idx, chroma) { self.collected_notes[note_idx] = true; }
                if self.collected_notes[note_idx] { collected_count += 1; }
            }
            let pass_threshold = if required_hits <= 3 { required_hits } else { required_hits - 1 };
            if required_hits > 0 && collected_count >= pass_threshold { self.success_timer += dt; } else { self.success_timer = 0.0; }
            if self.success_timer > 0.4 {
                self.success_timer = 0.0;
                self.collected_notes = [false; 12];
                self.stale_notes = *chroma;
                self.time_since_change = 0.0;
                self.current_chord_index = (self.current_chord_index + 1) % self.chords.len();
            }
        }
    }

    pub fn sync_audio_settings(&self) {
        if let Ok(mut state) = self.analysis_state.lock() {
            state.bass_boost_enabled = self.bass_boost_enabled;
            state.bass_boost_gain = self.bass_boost_gain;
        }
    }
}
