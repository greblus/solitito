// src/main.rs
mod model;
mod audio; 

use eframe::egui;
use std::sync::{Arc, Mutex};
use audio::{AudioAnalysis, start_audio_stream};
use model::{Chord, NoteName, Song, load_songs, load_all_scale_definitions, ScaleDefinition, ChordQuality}; 

fn main() -> eframe::Result<()> {
    let analysis_state = Arc::new(Mutex::new(AudioAnalysis {
        chroma_energy: [0.0; 12],
        bass_boost_enabled: true,
        bass_boost_gain: 10.0,
    }));

    let _stream = start_audio_stream(analysis_state.clone()).ok();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Jazz Assistant",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::new(analysis_state)))),
    )
}

// Helper do jasnych tekstów
fn lbl(text: &str) -> egui::RichText {
    egui::RichText::new(text).color(egui::Color32::WHITE).strong()
}

#[derive(PartialEq)]
enum AppMode {
    Songs,
    Scales,
}

struct MyApp {
    analysis_state: Arc<Mutex<AudioAnalysis>>,
    
    // DANE
    song_library: Vec<Song>,
    scale_definitions: Vec<ScaleDefinition>,
    
    // STAN APLIKACJI
    app_mode: AppMode,
    selected_song_idx: usize,
    
    // SCALE BUILDER
    scale_root: NoteName,
    selected_scale_def_idx: usize,
    
    // BIEŻĄCY ELEMENT
    song_title: String,
    chords: Vec<Chord>,
    current_chord_index: usize,
    
    // LOGIKA AUDIO / GRY
    success_timer: f32,
    sensitivity: f32,
    tail_threshold: f32,
    transition_delay: f32, 
    
    show_settings: bool,
    bass_boost_enabled: bool,
    bass_boost_gain: f32,
    intervals_input: String,
    
    // LOGIKA TRENINGU (To są pola, których brakowało!)
    random_mode: bool,
    current_random_target: Option<usize>,
    current_sequence_index: usize,
    random_sequence: Vec<usize>,
    
    collected_notes: [bool; 12],
    stale_notes: [f32; 12],
    time_since_change: f32,
}

impl MyApp {
    fn new(state: Arc<Mutex<AudioAnalysis>>) -> Self {
        let song_library = load_songs();
        let scale_definitions = load_all_scale_definitions();
        
        let start_song = if !song_library.is_empty() { 
            song_library[0].clone() 
        } else { 
            Song { title: "No Songs".into(), chords: vec![] } 
        };

        Self {
            analysis_state: state,
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
            show_settings: false, 
            bass_boost_enabled: true,
            bass_boost_gain: 10.0,
            intervals_input: "1 3 5".to_string(),
            
            // Inicjalizacja brakujących pól
            random_mode: false,
            current_random_target: None,
            current_sequence_index: 0,
            random_sequence: Vec::new(),
            
            collected_notes: [false; 12],
            stale_notes: [0.0; 12],
            time_since_change: 0.0,
        }
    }

    fn load_selected_song(&mut self) {
        if self.selected_song_idx < self.song_library.len() {
            let song = &self.song_library[self.selected_song_idx];
            self.song_title = song.title.clone();
            self.chords = song.chords.clone();
            self.current_chord_index = 0;
            self.reset_logic_state();
        }
    }
    
    fn build_scale_chord(&mut self) {
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
    
    fn reset_logic_state(&mut self) {
        self.success_timer = 0.0;
        self.collected_notes = [false; 12];
        self.stale_notes = [0.0; 12];
        self.time_since_change = 0.0;
        self.current_sequence_index = 0;
        self.current_random_target = None;
        self.random_sequence.clear();
    }

    fn get_visible_indices(&self) -> Vec<usize> {
        let parts: Vec<&str> = self.intervals_input.split_whitespace().collect();
        let mut indices = Vec::new();
        
        for p in parts {
            if self.app_mode == AppMode::Scales {
                match p {
                    "1" => indices.push(0),
                    "2" | "b2" | "9" => indices.push(1),
                    "3" | "b3" => indices.push(2),
                    "4" | "#4" | "11" => indices.push(3),
                    "5" | "b5" => indices.push(4),
                    "6" | "b6" | "13" => indices.push(5),
                    "7" | "b7" | "maj7" => indices.push(6),
                    _ => {}
                }
            } else {
                match p {
                    "1" => indices.push(0),
                    "3" | "b3" | "sus4" => indices.push(1),
                    "5" | "b5" => indices.push(2),
                    "7" | "b7" | "6" => indices.push(3),
                    _ => {}
                }
            }
        }
        indices
    }
    
    fn get_target_config_indices(&self) -> Vec<usize> {
        self.get_visible_indices()
    }

    fn is_note_active(&self, note_idx: usize, chroma: &[f32; 12]) -> bool {
        if self.stale_notes[note_idx] > 0.0 { return false; }
        let energy = chroma[note_idx];
        if energy < self.sensitivity { return false; }
        let max_energy = chroma.iter().fold(0.0f32, |a, &b| a.max(b));
        if energy < max_energy * 0.3 { return false; }
        true
    }

    fn update_stale_notes(&mut self, chroma: &[f32; 12]) {
        for i in 0..12 {
            let lock = self.stale_notes[i];
            if lock > 0.0 && chroma[i] < lock * self.tail_threshold {
                self.stale_notes[i] = 0.0;
            }
        }
    }
    
    fn pick_random_target(&mut self, options: &[usize], time_seed: f64) {
        if options.is_empty() { return; }
        let seed = (time_seed * 1000.0) as usize;
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
    
    // Generator permutacji (np. [0,1,2] -> [2,0,1])
    fn generate_shuffled_sequence(&self, len: usize, seed: f64) -> Vec<usize> {
        let mut seq: Vec<usize> = (0..len).collect();
        if len < 2 { return seq; }
        let mut pseudo_seed = (seed * 12345.0) as u64;
        for i in (1..len).rev() {
            pseudo_seed = pseudo_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (pseudo_seed as usize) % (i + 1);
            seq.swap(i, j);
        }
        seq
    }

    fn check_progress(&mut self, ctx: &egui::Context, chroma: &[f32; 12]) {
        self.time_since_change += ctx.input(|i| i.stable_dt);
        if self.time_since_change < self.transition_delay { return; }
        if self.chords.is_empty() { return; }

        let current_chord = &self.chords[self.current_chord_index];
        let all_targets = current_chord.get_target_indices();
        let config_indices = self.get_target_config_indices();

        let valid_indices: Vec<usize> = config_indices.into_iter()
            .filter(|&idx| idx < all_targets.len())
            .collect();
            
        if valid_indices.is_empty() { return; }

        // --- RANDOM MODE ---
        if self.random_mode {
            if self.current_random_target.is_none() {
                self.pick_random_target(&valid_indices, ctx.input(|i| i.time));
            }
            
            if let Some(target_logic_idx) = self.current_random_target {
                if target_logic_idx < valid_indices.len() {
                    let internal_idx = valid_indices[target_logic_idx];
                    let note_idx = all_targets[internal_idx];
                    if self.is_note_active(note_idx, chroma) {
                        self.success_timer += ctx.input(|i| i.stable_dt);
                    } else {
                        self.success_timer = 0.0;
                    }
                }
            }
            
            // SUKCES RANDOM
            if self.success_timer > 0.4 {
                self.success_timer = 0.0;
                self.stale_notes = *chroma; 
                self.pick_random_target(&valid_indices, ctx.input(|i| i.time));
            }
        } 
        // --- SCALES MODE (SEQUENTIAL) ---
        else if self.app_mode == AppMode::Scales {
            if self.current_sequence_index < valid_indices.len() {
                let internal_idx = valid_indices[self.current_sequence_index];
                let note_idx = all_targets[internal_idx];

                if self.is_note_active(note_idx, chroma) {
                    self.current_sequence_index += 1;
                }
            } else {
                // Koniec sekwencji
                self.success_timer += ctx.input(|i| i.stable_dt);
            }
            
            // SUKCES SEKOWENCJI
            if self.success_timer > 0.4 {
                self.success_timer = 0.0;
                self.collected_notes = [false; 12];
                self.stale_notes = *chroma; 
                self.current_sequence_index = 0; // Reset pętli
            }
        }
        // --- SONGS MODE (COLLECT) ---
        else {
            let mut collected_count = 0;
            let mut required_hits = 0;
            for &internal_idx in &valid_indices {
                required_hits += 1;
                let note_idx = all_targets[internal_idx];
                if self.is_note_active(note_idx, chroma) {
                    self.collected_notes[note_idx] = true;
                }
                if self.collected_notes[note_idx] {
                    collected_count += 1;
                }
            }
            let pass_threshold = if required_hits <= 3 { required_hits } else { required_hits - 1 };
            if required_hits > 0 && collected_count >= pass_threshold {
                self.success_timer += ctx.input(|i| i.stable_dt);
            } else {
                self.success_timer = 0.0;
            }
            
            // SUKCES SONGS
            if self.success_timer > 0.4 {
                self.success_timer = 0.0;
                self.collected_notes = [false; 12];
                self.stale_notes = *chroma;
                self.time_since_change = 0.0;
                self.current_chord_index = (self.current_chord_index + 1) % self.chords.len();
            }
        }
    }

    fn sync_audio_settings(&self) {
        if let Ok(mut state) = self.analysis_state.lock() {
            state.bass_boost_enabled = self.bass_boost_enabled;
            state.bass_boost_gain = self.bass_boost_gain;
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let chroma = {
            let state = self.analysis_state.lock().unwrap();
            state.chroma_energy
        };

        self.update_stale_notes(&chroma);
        self.check_progress(ctx, &chroma);
        self.sync_audio_settings();

        if self.chords.is_empty() {
            egui::CentralPanel::default().show(ctx, |ui| { ui.label("No data loaded"); });
            return;
        }

        let next_idx = (self.current_chord_index + 1) % self.chords.len();
        let next_chord = self.chords[next_idx].clone(); 
        let current_chord = self.chords[self.current_chord_index].clone();

        if self.app_mode == AppMode::Songs {
            egui::TopBottomPanel::bottom("footer_panel")
                .frame(egui::Frame::default().fill(egui::Color32::from_black_alpha(255)).inner_margin(20.0))
                .show_separator_line(false)
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                         let next_name = format!("{} {}", next_chord.root.to_string(), next_chord.quality.to_string());
                         ui.label(egui::RichText::new("Next Chord").size(14.0).color(egui::Color32::GRAY));
                         ui.label(egui::RichText::new(next_name).size(32.0).color(egui::Color32::LIGHT_GRAY));
                    });
                });
        }

        let panel_frame = egui::Frame::default().fill(egui::Color32::from_black_alpha(255));
        
        egui::CentralPanel::default().frame(panel_frame).show(ctx, |ui| {
            
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    let btn_text = if self.show_settings { "Close" } else { "⚙ Settings" };
                    if ui.button(btn_text).clicked() {
                        self.show_settings = !self.show_settings;
                    }
                });
            });

            if self.show_settings {
                ui.add_space(10.0);
                ui.vertical_centered(|ui| {
                    ui.set_max_width(380.0); 
                    egui::Frame::group(ui.style())
                        .fill(egui::Color32::from_black_alpha(180))
                        .stroke(egui::Stroke::new(1.0, egui::Color32::GRAY))
                        .inner_margin(15.0)
                        .show(ui, |ui| {
                            ui.heading(lbl("Settings"));
                            ui.add_space(10.0);

                            egui::Grid::new("settings_grid").num_columns(2).spacing([10.0, 10.0]).show(ui, |ui| {
                                ui.label(lbl("Mode:"));
                                ui.horizontal(|ui| {
                                    if ui.radio_value(&mut self.app_mode, AppMode::Songs, lbl("Songs")).clicked() {
                                        self.selected_song_idx = 0;
                                        self.intervals_input = "1 3 5".to_string(); 
                                        self.load_selected_song();
                                    }
                                    if ui.radio_value(&mut self.app_mode, AppMode::Scales, lbl("Scales")).clicked() {
                                        self.selected_scale_def_idx = 0;
                                        self.intervals_input = "1 2 3 4 5 6 7".to_string(); 
                                        self.build_scale_chord(); 
                                    }
                                });
                                ui.end_row();

                                if self.app_mode == AppMode::Songs {
                                    ui.label(lbl("Song:"));
                                    let titles: Vec<String> = self.song_library.iter().map(|s| s.title.clone()).collect();
                                    let mut trigger_load = false;
                                    egui::ComboBox::from_id_salt("song_select")
                                        .selected_text(if !titles.is_empty() { &titles[self.selected_song_idx] } else { "" })
                                        .show_ui(ui, |ui| {
                                            for (i, title) in titles.iter().enumerate() {
                                                if ui.selectable_value(&mut self.selected_song_idx, i, title).clicked() {
                                                    trigger_load = true;
                                                }
                                            }
                                        });
                                    if trigger_load { self.load_selected_song(); }
                                } else {
                                    ui.label(lbl("Root:"));
                                    let mut trigger_build = false;
                                    egui::ComboBox::from_id_salt("root_select")
                                        .selected_text(self.scale_root.to_string())
                                        .show_ui(ui, |ui| {
                                            for note in model::ALL_NOTES {
                                                if ui.selectable_value(&mut self.scale_root, note, note.to_string()).clicked() {
                                                    trigger_build = true;
                                                }
                                            }
                                        });
                                    ui.end_row();
                                    
                                    ui.label(lbl("Type:"));
                                    let scale_names: Vec<String> = self.scale_definitions.iter().map(|d| d.name.clone()).collect();
                                    egui::ComboBox::from_id_salt("type_select")
                                        .selected_text(if !scale_names.is_empty() { &scale_names[self.selected_scale_def_idx] } else { "" })
                                        .show_ui(ui, |ui| {
                                            for (i, name) in scale_names.iter().enumerate() {
                                                if ui.selectable_value(&mut self.selected_scale_def_idx, i, name).clicked() {
                                                    trigger_build = true;
                                                }
                                            }
                                        });
                                    if trigger_build { self.build_scale_chord(); }
                                }
                                ui.end_row();

                                ui.label(lbl("Random Trainer:"));
                                if ui.checkbox(&mut self.random_mode, lbl("Enable")).changed() {
                                    self.reset_logic_state();
                                }
                                ui.end_row();

                                ui.label(lbl("Threshold:"));
                                ui.horizontal(|ui| { ui.add(egui::Slider::new(&mut self.sensitivity, 0.0..=2.0).step_by(0.001).max_decimals(3).text("")); });
                                ui.end_row();

                                ui.label(lbl("Tail Release:"));
                                ui.horizontal(|ui| { ui.add(egui::Slider::new(&mut self.tail_threshold, 0.1..=0.9).step_by(0.05).text("")); });
                                ui.end_row();

                                ui.label(lbl("Input Delay:"));
                                ui.horizontal(|ui| { ui.add(egui::Slider::new(&mut self.transition_delay, 0.0..=1.0).step_by(0.1).text("s")); });
                                ui.end_row();

                                ui.label(lbl("Bass Boost:"));
                                ui.checkbox(&mut self.bass_boost_enabled, lbl("Enable"));
                                ui.end_row();

                                if self.bass_boost_enabled {
                                    ui.label(lbl("Boost Gain:"));
                                    ui.horizontal(|ui| { ui.add(egui::Slider::new(&mut self.bass_boost_gain, 1.0..=20.0).text("x")); });
                                    ui.end_row();
                                }

                                ui.label(lbl("Intervals:"));
                                ui.horizontal(|ui| {
                                    ui.add_sized([200.0, 20.0], egui::TextEdit::singleline(&mut self.intervals_input));
                                });
                                ui.end_row();
                            });

                            ui.add_space(15.0);
                            ui.separator();
                            ui.add_space(10.0);

                            // Debugger
                            let count = 12;
                            let slot_width = 22.0;
                            let total_width = count as f32 * slot_width;
                            ui.allocate_ui_with_layout(egui::vec2(total_width, 60.0), egui::Layout::top_down(egui::Align::Center), |ui| {
                                ui.columns(count, |cols| {
                                    for i in 0..12 {
                                        let val = chroma[i];
                                        let active = self.is_note_active(i, &chroma); 
                                        let note = NoteName::from_index(i);
                                        cols[i].vertical_centered(|ui| {
                                            let height = (val * 20.0).clamp(0.0, 40.0);
                                            let color = if active { egui::Color32::GREEN } else { egui::Color32::from_gray(60) };
                                            let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 40.0), egui::Sense::hover());
                                            ui.painter().rect_filled(rect, 1.0, egui::Color32::BLACK);
                                            ui.painter().rect_filled(egui::Rect::from_min_max(egui::pos2(rect.min.x, rect.max.y - height), rect.max), 1.0, color);
                                            
                                            if self.stale_notes[i] > 0.0 {
                                                let lock_h = (self.stale_notes[i] * 20.0).clamp(0.0, 40.0);
                                                let ly = rect.max.y - lock_h;
                                                ui.painter().line_segment([egui::pos2(rect.min.x, ly), egui::pos2(rect.max.x, ly)], egui::Stroke::new(2.0, egui::Color32::BLUE));
                                            }
                                            let th_h = (self.sensitivity * 20.0).clamp(0.0, 40.0);
                                            let ty = rect.max.y - th_h;
                                            ui.painter().line_segment([egui::pos2(rect.min.x, ty), egui::pos2(rect.max.x, ty)], egui::Stroke::new(1.0, egui::Color32::RED));
                                            ui.label(egui::RichText::new(note.to_string()).size(9.0));
                                        });
                                    }
                                });
                            });
                        });
                });
            }

            ui.vertical_centered(|ui| {
                ui.add_space(60.0); 
                ui.label(egui::RichText::new(&self.song_title).size(24.0).color(egui::Color32::WHITE));
                
                ui.add_space(30.0);
                
                let full_text = if self.app_mode == AppMode::Scales {
                    current_chord.root.to_string().to_string()
                } else {
                    format!("{} {}", current_chord.root.to_string(), current_chord.quality.to_string())
                };
                
                ui.label(egui::RichText::new(full_text).size(80.0).strong().color(egui::Color32::WHITE));
            });

            ui.add_space(50.0);

            ui.vertical_centered(|ui| {
                let all_interval_names = current_chord.quality.interval_names();
                let all_target_indices = current_chord.get_target_indices();
                
                let config_indices = self.get_target_config_indices();
                let valid_indices: Vec<(usize, usize)> = config_indices.into_iter()
                    .enumerate()
                    .filter(|(_, internal_idx)| *internal_idx < all_interval_names.len())
                    .collect();

                let count = valid_indices.len();
                let slot_width = 55.0;
                let total_width = count as f32 * slot_width;

                ui.allocate_ui_with_layout(
                    egui::vec2(total_width, 60.0),
                    egui::Layout::top_down(egui::Align::Center),
                    |ui| {
                        ui.columns(count, |cols| {
                            for (col_idx, &(random_logic_idx, internal_idx)) in valid_indices.iter().enumerate() {
                                
                                let name = all_interval_names[internal_idx].clone();
                                let note_idx = all_target_indices[internal_idx];
                                
                                let was_collected = self.collected_notes[note_idx];
                                let in_delay = self.time_since_change < self.transition_delay;
                                let active_now = self.is_note_active(note_idx, &chroma) && !in_delay;
                                
                                let mut is_active = false;
                                let mut is_target = false;

                                if self.app_mode == AppMode::Songs {
                                    // Songs
                                    is_active = was_collected || active_now;
                                } else {
                                    // Scales
                                    if self.random_mode {
                                        if let Some(target) = self.current_random_target {
                                            if col_idx == target { is_target = true; }
                                            if is_target && active_now { is_active = true; }
                                        }
                                    } else {
                                        // Sequential
                                        if col_idx < self.current_sequence_index {
                                            is_active = true;
                                        } else if col_idx == self.current_sequence_index {
                                            is_target = true;
                                            if active_now { is_active = true; }
                                        }
                                    }
                                }

                                let color = if is_active {
                                    egui::Color32::from_rgb(100, 200, 255)
                                } else if is_target {
                                    egui::Color32::YELLOW 
                                } else if in_delay {
                                    egui::Color32::from_rgb(60, 60, 60)
                                } else {
                                    egui::Color32::from_rgb(50, 80, 120)
                                };

                                cols[col_idx].vertical_centered(|ui| {
                                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                                    ui.add(egui::Label::new(
                                        egui::RichText::new(name).size(32.0).color(color)
                                    ));
                                });
                            }
                        });
                    }
                );
            });
        });

        ctx.request_repaint();
    }
}
