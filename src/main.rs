//#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

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
use std::time::Duration;

// TOTAL_FEATURES = 204
use audio::{AudioAnalysis, start_audio_stream, start_file_playback, TOTAL_FEATURES};
use brain::ChordBrain;
use state::{MyApp, AppMode};

use slint::{Timer, TimerMode, ModelRc, VecModel, Color, SharedString};

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    if let Err(e) = ort::init().with_name("Solitito").commit() {
        eprintln!("CRITICAL: Failed to initialize ONNX Runtime: {}", e);
    }

    // Bufor ustawiony na 204 (192 CQT + 12 Chroma)
    let analysis_state = Arc::new(Mutex::new(AudioAnalysis {
        chroma_sum: [0.0; 12],
        spectrum_visual: [0.0; 48],
        raw_input_for_ai: [0.0; TOTAL_FEATURES],
        bass_boost_enabled: true,
        bass_boost_gain: 5.0,
        input_gain: 3.0,
        noise_gate: 0.015, // Trochę wyższa bramka dla CQT
    }));
    
    let args: Vec<String> = env::args().collect();
    let mut file_mode = false;
    let mut _mic_stream = None;

    if args.len() > 2 && args[1] == "--file" {
        let _ = std::fs::File::create("benchmark_results.txt");
        let path = args[2].clone();
        println!("Starting FILE mode: {}", path);
        if let Err(e) = start_file_playback(path, analysis_state.clone()) {
            eprintln!("ERR FILE: {}", e);
            return Ok(());
        }
        file_mode = true;
    } else {
        println!("Starting LIVE mode (Microphone)...");
        match start_audio_stream(analysis_state.clone()) {
            Ok(s) => _mic_stream = Some(s),
            Err(e) => eprintln!("ERR MIC: {}", e),
        }
    }
    
    // MODEL V10
    let model_filename = "chord_model_v10_final.onnx"; 
    
    let brain: Option<Arc<Mutex<ChordBrain>>> = match ChordBrain::new(model_filename) {
        Ok(b) => Some(Arc::new(Mutex::new(b))),
        Err(e) => {
            eprintln!("WARNING: Could not load AI Model '{}': {}", model_filename, e);
            eprintln!("AI Chord recognition will be DISABLED.");
            None
        }
    };

    let my_app = Arc::new(Mutex::new(MyApp::new(analysis_state.clone(), brain)));

    let ui = AppWindow::new()?;
    let ui_weak = ui.as_weak();

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
    
    timer.start(TimerMode::Repeated, Duration::from_millis(16), move || {
        let ui = ui_weak.unwrap();
        let mut app = app_clone.lock().unwrap();
        frame_counter += 1;

        let (chroma, spectrum_vis, raw_ai) = {
            let s = app.analysis_state.lock().unwrap();
            (s.chroma_sum, s.spectrum_visual, s.raw_input_for_ai)
        };

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

        let dt = 0.016; 
        app.update_stale_notes(&chroma);
        app.check_progress(dt, &chroma); 
        app.sync_audio_settings();

        // --- AI INFERENCE ---
        let brain_arc = app.brain.clone();

        if let Some(brain_mutex) = brain_arc {
            if let Ok(mut b) = brain_mutex.lock() {
                // Predict używa teraz bufora o rozmiarze 204
                if let Ok((chord, score)) = b.predict(&raw_ai) {
                    
                    if file_mode && frame_counter % 6 == 0 { 
                         if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("benchmark_results.txt") {
                             let _ = writeln!(f, "{:.2}s | {} | {:.2}", app.total_time, chord, score);
                        }
                    }

                    app.chord_history.push_back((chord, score));
                    if app.chord_history.len() > 8 { 
                        app.chord_history.pop_front(); 
                    }
                    
                    let mut votes: HashMap<String, f32> = HashMap::new();
                    for (c, s) in &app.chord_history {
                        *votes.entry(c.clone()).or_insert(0.0) += *s; 
                    }
                    
                    let mut best_c = String::from("...");
                    let mut max_v = 0.0;
                    for (c, v) in votes {
                        if v > max_v { max_v = v; best_c = c; }
                    }
                    
                    let confidence = if !app.chord_history.is_empty() { max_v / app.chord_history.len() as f32 } else { 0.0 };

                    // Wyświetl wynik, jeśli pewność jest wystarczająca
                    if confidence > 0.4 {  // Zmniejszony próg dla V10
                         ui.set_ai_text(format!("AI: {} ({:.1})", best_c, confidence).into());
                    } else {
                         ui.set_ai_text("AI: ...".into());
                    }
                }
            }
        }

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
                 let is_active = was_collected || active_now;
                 
                 ui_names.push(SharedString::from(name));
                 if is_active { 
                     ui_colors.push(Color::from_rgb_u8(100, 255, 100)); 
                 } else { 
                     ui_colors.push(Color::from_rgb_u8(80, 80, 80)); 
                 }
            }
            
            ui.set_interval_names(ModelRc::from(Rc::new(VecModel::from(ui_names))));
            ui.set_interval_colors(ModelRc::from(Rc::new(VecModel::from(ui_colors))));
            
            let spec_vec: Vec<f32> = spectrum_vis.to_vec();
            let mut spec_colors = Vec::new();
            let targets = curr_chord.get_target_indices();
            
            for i in 0..48 {
                let note_idx = (i + 40) % 12; // +40 dla offsetu wizualnego
                let val = spectrum_vis[i];
                let is_target = targets.contains(&note_idx);
                let color = if val > 0.1 { 
                    if is_target { Color::from_rgb_u8(50, 255, 100) } 
                    else { Color::from_rgb_u8(255, 50, 50) }
                } else {
                    if (i + 40) % 12 == 0 { Color::from_rgb_u8(60, 60, 80) } 
                    else { Color::from_rgb_u8(30, 30, 30) }
                };
                spec_colors.push(color);
            }
            ui.set_spectrum_data(ModelRc::from(Rc::new(VecModel::from(spec_vec))));
            ui.set_spectrum_colors(ModelRc::from(Rc::new(VecModel::from(spec_colors))));
        }
    });

    let app_weak = my_app.clone();
    let ui_weak_cb = ui.as_weak();
    
    ui.on_toggle_mode(move |mode_idx| {
        let mut app = app_weak.lock().unwrap();
        let ui = ui_weak_cb.unwrap();
        
        if mode_idx == 0 { 
            app.app_mode = AppMode::Songs; 
            app.selected_song_idx = 0; 
            app.intervals_input = "1 3 5".to_string(); 
            let titles: Vec<SharedString> = app.song_library.iter().map(|s| SharedString::from(&s.title)).collect();
            ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(titles))));
            app.load_selected_song();
        } else { 
            app.app_mode = AppMode::Scales; 
            app.selected_scale_def_idx = 0; 
            app.intervals_input = "1 2 3 4 5 6 7".to_string(); 
            let titles: Vec<SharedString> = app.scale_definitions.iter().map(|s| SharedString::from(&s.name)).collect();
            ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(titles))));
            app.build_scale_chord();
        }
        app.reset_logic_state();
    });

    let app_weak_2 = my_app.clone();
    ui.on_item_selected(move |index| {
        let mut app = app_weak_2.lock().unwrap();
        if app.app_mode == AppMode::Songs { 
            app.selected_song_idx = index as usize; 
            app.load_selected_song(); 
        } else { 
            app.selected_scale_def_idx = index as usize; 
            app.build_scale_chord(); 
        }
        app.reset_logic_state();
    });

    ui.run()
}
