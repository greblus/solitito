// src/main.rs
#![cfg_attr(windows, windows_subsystem = "windows")]

mod model;
mod audio;
mod brain;
mod state;

use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::collections::HashMap;
use audio::{AudioAnalysis, start_audio_stream};
use brain::ChordBrain;
use state::{MyApp, AppMode};
use slint::{Timer, TimerMode, ModelRc, VecModel, Color, SharedString};

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    if let Err(e) = ort::init().with_name("Solitito").commit() { eprintln!("ORT Error: {}", e); }
    
    let analysis_state = Arc::new(Mutex::new(AudioAnalysis {
        chroma_sum: [0.0; 12],
        spectrum_48: [0.0; 48],
        bass_boost_enabled: true,
        bass_boost_gain: 5.0,
    }));
    let _stream = start_audio_stream(analysis_state.clone()).unwrap();
    let brain = ChordBrain::new().ok().map(Arc::new);

    let my_app = Arc::new(Mutex::new(MyApp::new(analysis_state.clone(), brain)));

    let ui = AppWindow::new()?;
    let ui_weak = ui.as_weak();

    // --- INICJALIZACJA LISTY W SLINT (NA START) ---
    {
        let app = my_app.lock().unwrap();
        let titles: Vec<SharedString> = app.song_library.iter()
            .map(|s| SharedString::from(&s.title))
            .collect();
        ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(titles))));
    }

    let timer = Timer::default();
    let app_clone = my_app.clone();
    
    timer.start(TimerMode::Repeated, std::time::Duration::from_millis(16), move || {
        let ui = ui_weak.unwrap();
        let mut app = app_clone.lock().unwrap();

        // A. Audio
        let (chroma, spectrum) = {
            let s = app.analysis_state.lock().unwrap();
            (s.chroma_sum, s.spectrum_48)
        };

        // B. Sync Settings
        app.sensitivity = ui.get_threshold();
        app.tail_threshold = ui.get_tail();
        app.transition_delay = ui.get_delay();
        app.bass_boost_enabled = ui.get_boost_enabled();
        app.bass_boost_gain = ui.get_boost_gain();
        app.random_mode = ui.get_random_enabled();
        
        let input_txt = ui.get_interval_input_text().to_string();
        if input_txt != app.intervals_input {
            app.intervals_input = input_txt;
        }

        // C. Logic
        let dt = 0.016; 
        app.update_stale_notes(&chroma);
        app.check_progress(dt, &chroma); 
        app.sync_audio_settings();

        // D. AI
        if let Some(brain) = &app.brain {
            if let Ok((chord, score)) = brain.predict(&spectrum) {
                if score > -10.0 { app.prediction_buffer.push_back((chord, score)); }
                while app.prediction_buffer.len() > 20 { app.prediction_buffer.pop_front(); }
                
                let mut votes: HashMap<String, f32> = HashMap::new();
                for (c, s) in &app.prediction_buffer { *votes.entry(c.clone()).or_insert(0.0) += *s; }
                
                let mut best_chord = String::from("...");
                let mut best_total_score = -1000.0;
                for (c, total) in votes {
                    if total > best_total_score { best_total_score = total; best_chord = c; }
                }
                let avg_score = best_total_score / app.prediction_buffer.len() as f32;

                if avg_score > 1.5 {
                     ui.set_ai_text(format!("AI: {} ({:.1})", best_chord, avg_score).into());
                } else {
                     ui.set_ai_text("AI: ...".into());
                }
            }
        }

        // E. Update UI
        ui.set_song_title(app.song_title.clone().into());
        
        if app.chords.is_empty() {
             ui.set_chord_name("No Data".into());
        } else {
            let curr_chord = &app.chords[app.current_chord_index];
            
            if app.app_mode == AppMode::Scales {
                 ui.set_chord_name(curr_chord.root.to_string().into());
                 ui.set_next_chord("".into());
            } else {
                 ui.set_chord_name(format!("{} {}", curr_chord.root.to_string(), curr_chord.quality.to_string()).into());
                 let next = &app.chords[(app.current_chord_index + 1) % app.chords.len()];
                 ui.set_next_chord(format!("{} {}", next.root.to_string(), next.quality.to_string()).into());
            }

            // --- FILTROWANIE INTERWAŁÓW DLA UI ---
            // To odpowiada na Twoje pytanie o "3 5 7".
            // Generujemy UI TYLKO dla tych interwałów, które są w configu.
            
            let all_interval_names = curr_chord.quality.interval_names();
            let all_target_indices = curr_chord.get_target_indices();
            let config_indices = app.get_target_config_indices();
            
            let mut ui_names = Vec::new();
            let mut ui_colors = Vec::new();

            // Iterujemy tylko po valid_indices (czyli przecięciu akordu i konfiguracji)
            let valid_indices: Vec<(usize, usize)> = config_indices.into_iter()
                .enumerate()
                .filter(|(_, internal_idx)| *internal_idx < all_interval_names.len())
                .collect();

            for (col_idx, &(_random_logic_idx, internal_idx)) in valid_indices.iter().enumerate() {
                 let name = &all_interval_names[internal_idx];
                 let note_idx = all_target_indices[internal_idx];
                 
                 let was_collected = app.collected_notes[note_idx];
                 let in_delay = app.time_since_change < app.transition_delay;
                 let active_now = app.is_note_active(note_idx, &chroma) && !in_delay;
                 let mut is_active = was_collected || active_now;
                 let mut is_target = false;

                 if app.random_mode {
                    if let Some(target) = app.current_random_target {
                        if col_idx == target { is_target = true; }
                        if is_target && active_now { is_active = true; }
                    }
                 } else if app.app_mode == AppMode::Scales {
                     if col_idx < app.current_sequence_index { is_active = true; }
                     else if col_idx == app.current_sequence_index { is_target = true; if active_now { is_active = true; } }
                 }

                 ui_names.push(SharedString::from(name));
                 
                 if is_active {
                     ui_colors.push(Color::from_rgb_u8(100, 255, 100)); 
                 } else if is_target {
                     ui_colors.push(Color::from_rgb_u8(255, 255, 0)); 
                 } else if in_delay {
                     ui_colors.push(Color::from_rgb_u8(60, 60, 60));
                 } else {
                     ui_colors.push(Color::from_rgb_u8(80, 80, 80));
                 }
            }
            
            ui.set_interval_names(ModelRc::from(Rc::new(VecModel::from(ui_names))));
            ui.set_interval_colors(ModelRc::from(Rc::new(VecModel::from(ui_colors))));
            
            let spec_vec: Vec<f32> = spectrum.to_vec();
            ui.set_spectrum_data(ModelRc::from(Rc::new(VecModel::from(spec_vec))));
        }
    });

    // --- OBSŁUGA ZDARZEŃ (CALLBACKI) ---
    let app_weak = my_app.clone();
    let ui_weak_cb = ui.as_weak();
    
    // 1. Zmiana Trybu
    ui.on_toggle_mode(move |mode_idx| {
        let mut app = app_weak.lock().unwrap();
        let ui = ui_weak_cb.unwrap();
        
        if mode_idx == 0 { 
            app.app_mode = AppMode::Songs; 
            app.selected_song_idx = 0;
            app.intervals_input = "1 3 5".to_string(); 
            
            // Ładujemy listę Piosenek do UI
            let titles: Vec<SharedString> = app.song_library.iter().map(|s| SharedString::from(&s.title)).collect();
            ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(titles))));
            
            app.load_selected_song();
        } else { 
            app.app_mode = AppMode::Scales; 
            app.selected_scale_def_idx = 0;
            app.intervals_input = "1 2 3 4 5 6 7".to_string(); 
            
            // Ładujemy listę Skal do UI (Zamiast piosenek)
            let titles: Vec<SharedString> = app.scale_definitions.iter().map(|s| SharedString::from(&s.name)).collect();
            ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(titles))));
            
            app.build_scale_chord();
        }
        app.reset_logic_state();
    });
    
    // 2. Wybór z listy (Piosenka lub Skala)
    let app_weak_2 = my_app.clone();
    ui.on_item_selected(move |index| {
        let mut app = app_weak_2.lock().unwrap();
        
        if app.app_mode == AppMode::Songs {
            app.selected_song_idx = index as usize;
            app.load_selected_song();
        } else {
            // W trybie Skal lista zawiera TYPY skal.
            // Root (C, D..) nadal wybieramy w logice (tutaj uprościliśmy: lista = typy).
            // Ale zaraz... w Slint mamy tylko jeden ComboBox. 
            // Umówmy się, że ComboBox wybiera TYP SKALI (Dorian, Major), a Root jest stały (np. C) 
            // lub dodamy drugi ComboBox w przyszłości.
            // Na razie: wybierasz typ skali z listy.
            app.selected_scale_def_idx = index as usize;
            app.build_scale_chord();
        }
        app.reset_logic_state();
    });

    ui.run()
}
