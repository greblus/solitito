// src/main.rs
#![cfg_attr(windows, windows_subsystem = "windows")]

mod model;
mod audio;
mod brain;
mod state;

use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::collections::HashMap;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use audio::{AudioAnalysis, start_audio_stream, start_file_playback, LOG_BINS, CTX_FRAMES};
use brain::ChordBrain;
use state::{MyApp, AppMode};
use slint::{Timer, TimerMode, ModelRc, VecModel, Color, SharedString};

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    // Inicjalizacja ORT
    if let Err(e) = ort::init().with_name("Solitito").commit() { eprintln!("ORT Error: {}", e); }
    
    // Inicjalizacja stanu
    let analysis_state = Arc::new(Mutex::new(AudioAnalysis {
        chroma_sum: [0.0; 12],
        spectrum_visual: [0.0; 48],
        // Bufor dla AI (Wektor)
        raw_input_for_ai: vec![0.0; LOG_BINS * CTX_FRAMES],
        
        bass_boost_enabled: true,
        bass_boost_gain: 5.0,
        input_gain: 3.0,
        noise_gate: 0.0002,
    }));
    
    let args: Vec<String> = env::args().collect();
    let mut file_mode = false;
    let mut _mic_stream = None;

    // Obsługa trybu plikowego (Benchmark)
    if args.len() > 2 && args[1] == "--file" {
        // Resetujemy plik logów
        let _ = std::fs::File::create("benchmark_results.txt");
        
        let path = args[2].clone();
        if let Err(e) = start_file_playback(path, analysis_state.clone()) {
            eprintln!("BŁĄD PLIKU: {}", e);
            return Ok(());
        }
        file_mode = true;
    } else {
        // Start mikrofonu
        match start_audio_stream(analysis_state.clone()) {
            Ok(s) => _mic_stream = Some(s),
            Err(e) => eprintln!("BŁĄD MIKROFONU: {}", e),
        }
    }
    
    let brain = ChordBrain::new().ok().map(Arc::new);
    let my_app = Arc::new(Mutex::new(MyApp::new(analysis_state.clone(), brain)));

    let ui = AppWindow::new()?;
    let ui_weak = ui.as_weak();

    // Inicjalizacja listy piosenek
    {
        let app = my_app.lock().unwrap();
        let titles: Vec<SharedString> = app.song_library.iter()
            .map(|s| SharedString::from(&s.title))
            .collect();
        ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(titles))));
    }

    let timer = Timer::default();
    let app_clone = my_app.clone();
    let mut frame_counter = 0;
    
    // Pętla główna
    timer.start(TimerMode::Repeated, std::time::Duration::from_millis(16), move || {
        let ui = ui_weak.unwrap();
        let mut app = app_clone.lock().unwrap();
        frame_counter += 1;

        // A. Pobranie danych (klonujemy wektor dla AI)
        let (chroma, spectrum_vis, raw_ai) = {
            let s = app.analysis_state.lock().unwrap();
            (s.chroma_sum, s.spectrum_visual, s.raw_input_for_ai.clone())
        };

        // B. Ustawienia (tylko w trybie live pobieramy gain z suwaków)
        if !file_mode {
            app.input_gain = ui.get_input_gain();
            app.noise_gate = ui.get_noise_gate();
        }
        
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

        // C. Logika gry
        let dt = 0.016; 
        app.update_stale_notes(&chroma);
        app.check_progress(dt, &chroma); 
        app.sync_audio_settings();

        // D. AI PREDICTION (CRNN)
        if let Some(brain) = &app.brain {
            let expected = if file_mode { None } else {
                 if !app.chords.is_empty() {
                     let c = &app.chords[app.current_chord_index];
                     let q = c.quality.to_string();
                     if q.is_empty() { Some(c.root.to_string().to_string()) } 
                     else { Some(format!("{} {}", c.root.to_string(), q)) }
                 } else { None }
            };

            // Predykcja (bez argumentu chroma, bo CRNN go nie używa)
            if let Ok((chord, score)) = brain.predict(&raw_ai, expected.as_deref()) {
                
                // Logowanie do pliku (Benchmark)
                if file_mode && frame_counter % 6 == 0 {
                     if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("benchmark_results.txt") {
                         let _ = writeln!(f, "{:.2}s | {} | {:.2}", app.total_time, chord, score);
                    }
                }

                // Historia i wygładzanie UI
                app.chord_history.push_back((chord, score));
                if app.chord_history.len() > 6 { app.chord_history.pop_front(); }
                
                let mut votes: HashMap<String, f32> = HashMap::new();
                for (c, s) in &app.chord_history {
                    // Sumujemy głosy (ignorujemy tylko kompletne błędy < -5.0)
                    if *s > -5.0 { 
                        *votes.entry(c.clone()).or_insert(0.0) += *s; 
                    }
                }
                
                let mut best_c = String::from("...");
                let mut max_v = -1000.0;
                for (c, v) in votes {
                    if v > max_v { max_v = v; best_c = c; }
                }
                
                // Normalizacja pewności
                let confidence = if !app.chord_history.is_empty() { max_v / 6.0 } else { 0.0 };

                if confidence > 2.0 { 
                     ui.set_ai_text(format!("AI: {} ({:.1})", best_c, confidence).into());
                } else {
                     ui.set_ai_text("AI: ...".into());
                }
            }
        }

        // E. Aktualizacja UI
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

            // Wizualizacja Interwałów (Note Bars)
            let all_interval_names = curr_chord.quality.interval_names();
            let all_target_indices = curr_chord.get_target_indices();
            let config_indices = app.get_target_config_indices();
            
            let mut ui_names = Vec::new();
            let mut ui_colors = Vec::new();

            let valid_indices: Vec<(usize, usize)> = config_indices.into_iter()
                .enumerate()
                .filter(|(_, idx)| *idx < all_interval_names.len())
                .collect();

            for (_col_idx, &(_, internal_idx)) in valid_indices.iter().enumerate() {
                 let name = &all_interval_names[internal_idx];
                 let note_idx = all_target_indices[internal_idx];
                 
                 let was_collected = app.collected_notes[note_idx];
                 let in_delay = app.time_since_change < app.transition_delay;
                 let active_now = app.is_note_active(note_idx, &chroma) && !in_delay;
                 let mut is_active = was_collected || active_now;
                 
                 // POPRAWKA: Dodano mut
                 let mut is_target = false; 

                 if app.random_mode {
                    if let Some(target) = app.current_random_target {
                        if _col_idx == target { is_target = true; }
                        if is_target && active_now { is_active = true; }
                    }
                 } else if app.app_mode == AppMode::Scales {
                     if _col_idx < app.current_sequence_index { is_active = true; }
                     else if _col_idx == app.current_sequence_index { is_target = true; if active_now { is_active = true; } }
                 }

                 ui_names.push(SharedString::from(name));
                 
                 if is_active { ui_colors.push(Color::from_rgb_u8(100, 255, 100)); } 
                 else if is_target { ui_colors.push(Color::from_rgb_u8(255, 255, 0)); } 
                 else if in_delay { ui_colors.push(Color::from_rgb_u8(60, 60, 60)); } 
                 else { ui_colors.push(Color::from_rgb_u8(80, 80, 80)); }
            }
            
            ui.set_interval_names(ModelRc::from(Rc::new(VecModel::from(ui_names))));
            ui.set_interval_colors(ModelRc::from(Rc::new(VecModel::from(ui_colors))));
            
            // Wizualizacja Spektrum (Debug Bar)
            let spec_vec: Vec<f32> = spectrum_vis.to_vec();
            let mut spec_colors = Vec::new();
            let targets = curr_chord.get_target_indices();

            for i in 0..48 {
                let note_idx = (i + 40) % 12; 
                let val = spectrum_vis[i];
                
                let color = if val > 0.1 { 
                    if targets.contains(&note_idx) { Color::from_rgb_u8(50, 255, 100) } else { Color::from_rgb_u8(255, 50, 50) }
                } else {
                    if (i + 40) % 12 == 0 { Color::from_rgb_u8(60, 60, 80) } else { Color::from_rgb_u8(30, 30, 30) }
                };
                spec_colors.push(color);
            }
            
            ui.set_spectrum_data(ModelRc::from(Rc::new(VecModel::from(spec_vec))));
            ui.set_spectrum_colors(ModelRc::from(Rc::new(VecModel::from(spec_colors))));
        }
    });

    // Callbacki UI
    let app_weak = my_app.clone();
    let ui_weak_cb = ui.as_weak();
    ui.on_toggle_mode(move |mode_idx| {
        let mut app = app_weak.lock().unwrap();
        let ui = ui_weak_cb.unwrap();
        if mode_idx == 0 { 
            app.app_mode = AppMode::Songs; app.selected_song_idx = 0; app.intervals_input = "1 3 5".to_string(); 
            let titles: Vec<SharedString> = app.song_library.iter().map(|s| SharedString::from(&s.title)).collect();
            ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(titles))));
            app.load_selected_song();
        } else { 
            app.app_mode = AppMode::Scales; app.selected_scale_def_idx = 0; app.intervals_input = "1 2 3 4 5 6 7".to_string(); 
            let titles: Vec<SharedString> = app.scale_definitions.iter().map(|s| SharedString::from(&s.name)).collect();
            ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(titles))));
            app.build_scale_chord();
        }
        app.reset_logic_state();
    });
    
    let app_weak_2 = my_app.clone();
    ui.on_item_selected(move |index| {
        let mut app = app_weak_2.lock().unwrap();
        if app.app_mode == AppMode::Songs { app.selected_song_idx = index as usize; app.load_selected_song(); } 
        else { app.selected_scale_def_idx = index as usize; app.build_scale_chord(); }
        app.reset_logic_state();
    });

    ui.run()
}
