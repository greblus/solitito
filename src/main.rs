#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod model;
mod audio;
mod brain;
mod state;

use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::collections::HashMap;
use std::env;
use std::time::Duration;
use std::thread;

use audio::{AudioAnalysis, start_audio_stream, start_file_playback};
use brain::{ChordBrain, Prediction};
use state::{MyApp, AppMode, MatchStatus};

use slint::{Timer, TimerMode, ModelRc, VecModel, Color, SharedString, PhysicalPosition};

slint::include_modules!();

#[derive(Clone, Default)]
struct AiResult {
    pred: Prediction,
    updated: bool,
}

fn main() -> Result<(), slint::PlatformError> {

    let _keep_awake = keepawake::Builder::new()
        .display(true)
        .idle(true)
        .sleep(true)
        .create();

    if let Err(e) = &_keep_awake {
        eprintln!("Warning: Could not enable keep-awake: {}", e);
    }
    
    if let Err(e) = ort::init().with_name("Solitito").commit() {
        eprintln!("CRITICAL: Failed to initialize ONNX Runtime: {}", e);
    }

    let default_gain = 2.0;
    let default_gate = 0.02;
    let default_boost_enabled = true;
    let default_boost_gain = 5.0;

    let analysis_state = Arc::new(Mutex::new(AudioAnalysis {
        // FIX: Poprawna inicjalizacja pola history zamiast raw_input
        input_history: [[0.0; 168]; 48],
        frame_live: [false; 48],
        spectrum_visual: [0.0; 48],
        chroma_sum: [0.0; 12],
        bass_boost_enabled: default_boost_enabled,
        bass_boost_gain: default_boost_gain, 
        input_gain: default_gain,      
        noise_gate: default_gate,    
    }));
    
    let ai_result_state = Arc::new(Mutex::new(AiResult::default()));

    let args: Vec<String> = env::args().collect();
    let mut file_mode = false;
    let mut _mic_stream = None;

    if args.len() > 2 && args[1] == "--file" {
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
    
    // WĄTEK AI
    let analysis_for_ai = analysis_state.clone();
    let result_for_ai = ai_result_state.clone();
    
    thread::spawn(move || {
        let model_filename = "best_model_v2_take6.onnx";
        let mut brain = match ChordBrain::new(model_filename) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("WARNING: Could not load AI Model: {}", e);
                return;
            }
        };
        
        // Ile okna kontekstowego musi nieść realny sygnał, żeby w ogóle pytać model.
        // Trener budował okna wyłącznie WEWNĄTRZ wybrzmiewającego akordu, więc model
        // nigdy nie widział wejścia "połowa cisza, połowa akord". Po uderzeniu w struny
        // aplikacja podaje mu dokładnie takie okno przez 0.77 s i dostaje w odpowiedzi
        // zgadywanie — stąd wrażenie, że akord "dochodzi" dopiero w ogonku.
        const MIN_FILL: f32 = 0.9;

        // SOLITITO_DEBUG=1 — wypisuje, CO model słyszy i MIĘDZY CZYM się waha.
        // Bez tego przy pomyłce widać tylko końcową nazwę, a to za mało, żeby
        // odróżnić "nie słyszy septymy" od "słyszy, ale jej nie używa".
        let debug_ai = std::env::var("SOLITITO_DEBUG").is_ok();
        if debug_ai {
            println!("🔎 Tryb diagnostyczny: pryma | jakości | interwały względem prymy");
        }

        loop {
            // FIX: Pobieramy historię
            let (history, fill) = {
                let state = analysis_for_ai.lock().unwrap();
                (state.input_history, state.history_fill())
            };

            if fill < MIN_FILL {
                thread::sleep(Duration::from_millis(40));
                continue;
            }

            if let Ok(pred) = brain.predict(&history) {
                if debug_ai {
                    print_debug(&pred);
                }
                if let Ok(mut res) = result_for_ai.lock() {
                    res.pred = pred;
                    res.updated = true;
                }
            }
            thread::sleep(Duration::from_millis(40));
        }
    });

    let my_app = Arc::new(Mutex::new(MyApp::new(analysis_state.clone(), None)));
    let ui = AppWindow::new()?;
    let ui_weak = ui.as_weak();

    {
        let app = my_app.lock().unwrap();
        let titles: Vec<SharedString> = app.song_library.iter()
            .map(|s| SharedString::from(&s.title))
            .collect();
        ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(titles))));
        ui.set_input_gain(default_gain);
        ui.set_noise_gate(default_gate);
        ui.set_boost_enabled(default_boost_enabled);
        ui.set_boost_gain(default_boost_gain);
        ui.set_current_mode(app.app_mode as i32); 
        ui.set_interval_input_text(app.intervals_input.clone().into()); 
    }

    ui.window().set_position(PhysicalPosition::new(450, 10));

    let timer = Timer::default();
    let app_clone = my_app.clone();
    let result_for_ui = ai_result_state.clone();
    
    let keys_list: Vec<SharedString> = vec!["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        .into_iter().map(SharedString::from).collect();

    timer.start(TimerMode::Repeated, Duration::from_millis(16), move || {
        let ui = ui_weak.unwrap();
        let mut app = app_clone.lock().unwrap();

        let spectrum_vis = {
            let s = app.analysis_state.lock().unwrap();
            s.spectrum_visual
        };

        if !file_mode {
            app.input_gain = ui.get_input_gain();
            app.noise_gate = ui.get_noise_gate();
        }
        app.bass_boost_enabled = ui.get_boost_enabled();
        app.bass_boost_gain = ui.get_boost_gain();
        app.sensitivity = ui.get_threshold();     
        app.transition_delay = ui.get_delay();    
        app.tail_threshold = ui.get_tail();       
        app.random_mode = ui.get_random_enabled();
        
        let ui_txt = ui.get_interval_input_text().to_string();
        if ui_txt != app.intervals_input {
            app.intervals_input = ui_txt;
            app.reset_logic_state();
        }
        app.sync_audio_settings();

        if let Ok(mut res) = result_for_ui.lock() {
            if res.updated {
                let chord = res.pred.chord.clone();
                let score = res.pred.confidence;
                app.last_pitches = res.pred.pitches;

                // FIX: Reset flagi po odczytaniu
                res.updated = false;

                // Głosowanie nad 5 oknami (~0.2 s). Sonda probe_quality zmierzyła,
                // że głosowanie większościowe wypada lepiej niż uśrednianie
                // rozkładów — model bywa pewny i błędny, a średnia tę pewność
                // przenosi dalej, podczas gdy głosowanie ją tłumi.
                app.chord_history.push_back((chord.clone(), score));
                if app.chord_history.len() > 5 { app.chord_history.pop_front(); }
                
                let mut votes: HashMap<String, f32> = HashMap::new();
                for (c, s) in &app.chord_history {
                    *votes.entry(c.clone()).or_insert(0.0) += *s; 
                }

                let fallback_str = String::from("...");
                let fallback_val = 0.0;
                let (best_c, max_v) = votes.iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap_or((&fallback_str, &fallback_val));
                
                let current_confidence = if !app.chord_history.is_empty() { 
                    *max_v / app.chord_history.len() as f32 
                } else { 0.0 };
                
                ui.set_ai_text(format!("{} ({:.0}%)", best_c, current_confidence * 100.0).into());
                
                let dt = 0.040; 
                // FIX: Używamy uśrednionego wyniku (best_c) do logiki gry
                app.check_progress_with_ai(dt, best_c, current_confidence);
            }
        }

        ui.set_song_title(app.song_title.clone().into());
        
        if app.chords.is_empty() { 
            ui.set_chord_name("No Data".into()); 
        } else {
            let curr_chord = &app.chords[app.current_chord_index];
            
            if app.app_mode == AppMode::Scales {
                 ui.set_chord_name(curr_chord.root.to_string().into());
            } else {
                 let q_str = curr_chord.quality.to_string();
                 let quality_display = if q_str.is_empty() { "Maj" } else { q_str.as_str() };
                 ui.set_chord_name(format!("{} {}", curr_chord.root.to_string(), quality_display).into());
            }
            
            if app.app_mode == AppMode::Chords {
                match app.match_status {
                    MatchStatus::Exact => ui.set_chord_text_color(slint::Brush::SolidColor(Color::from_rgb_u8(50, 255, 50))), 
                    MatchStatus::Partial => ui.set_chord_text_color(slint::Brush::SolidColor(Color::from_rgb_u8(255, 220, 50))), 
                    MatchStatus::Flicker => ui.set_chord_text_color(slint::Brush::SolidColor(Color::from_rgb_u8(255, 50, 50))), 
                    MatchStatus::None => ui.set_chord_text_color(slint::Brush::SolidColor(Color::from_rgb_u8(255, 255, 255))), 
                }
            } else {
                ui.set_chord_text_color(slint::Brush::SolidColor(Color::from_rgb_u8(255, 255, 255)));
            }
            
            let next_idx = (app.current_chord_index + 1) % app.chords.len();
            let next_c = &app.chords[next_idx];
            ui.set_next_chord(format!("{} {}", next_c.root.to_string(), next_c.quality.to_string()).into());

            if app.app_mode != AppMode::Chords {
                let all_names = curr_chord.quality.interval_names();
                let active_indices = app.get_active_indices(curr_chord);
                let mut ui_names = Vec::new();
                let mut ui_colors = Vec::new();
                for (step_idx, &internal_idx) in active_indices.iter().enumerate() {
                    if internal_idx < all_names.len() {
                        let name = &all_names[internal_idx];
                        ui_names.push(SharedString::from(name));
                        if step_idx < app.current_note_step {
                            ui_colors.push(Color::from_rgb_u8(50, 255, 50));
                        } else if step_idx == app.current_note_step {
                            if app.success_timer > 0.05 {
                                    ui_colors.push(Color::from_rgb_u8(200, 255, 50));
                            } else {
                                    ui_colors.push(Color::from_rgb_u8(180, 180, 180));
                            }
                        } else {
                            ui_colors.push(Color::from_rgb_u8(60, 60, 60));
                        }
                    }
                }
                ui.set_interval_names(ModelRc::from(Rc::new(VecModel::from(ui_names))));
                ui.set_interval_colors(ModelRc::from(Rc::new(VecModel::from(ui_colors))));
            }
            
            let spec_vec: Vec<f32> = spectrum_vis.to_vec();
            let mut spec_colors = Vec::new();
            
            let targets = curr_chord.get_target_indices();
            for i in 0..48 {
                let note_idx = (i + 40) % 12; 
                let val = spectrum_vis[i];
                let is_target = targets.contains(&note_idx);
                let color = if val > 0.05 { 
                    if is_target { Color::from_rgb_u8(50, 255, 100) } else { Color::from_rgb_u8(255, 50, 50) }
                } else {
                    if (i + 40) % 12 == 0 { Color::from_rgb_u8(60, 60, 80) } else { Color::from_rgb_u8(30, 30, 30) }
                };
                spec_colors.push(color);
            }
            ui.set_spectrum_data(ModelRc::from(Rc::new(VecModel::from(spec_vec))));
            ui.set_spectrum_colors(ModelRc::from(Rc::new(VecModel::from(spec_colors))));
        }
    });

    let app_weak = my_app.clone();
    let ui_weak_cb = ui.as_weak();
    let keys_list_clone = keys_list.clone();
    
    ui.on_mode_changed(move |mode_idx| {
        let mut app = app_weak.lock().unwrap();
        let ui = ui_weak_cb.unwrap();
        app.set_mode(mode_idx);
        ui.set_current_mode(mode_idx);
        ui.set_interval_input_text(app.intervals_input.clone().into());
        let (label, items, sec_label, sec_items) = match app.app_mode {
            AppMode::Scales => (
                "Select Scale:", app.scale_definitions.iter().map(|s| SharedString::from(&s.name)).collect::<Vec<SharedString>>(),
                "Key (Root):", keys_list_clone.clone()
            ),
            AppMode::Arpeggios => (
                "Select Song:", app.song_library.iter().map(|s| SharedString::from(&s.title)).collect::<Vec<SharedString>>(),
                "Pattern:", app.arpeggio_patterns.iter().map(|s| SharedString::from(&s.name)).collect::<Vec<SharedString>>()
            ),
            _ => ("Select Song:", app.song_library.iter().map(|s| SharedString::from(&s.title)).collect::<Vec<SharedString>>(), "", vec![]),
        };
        ui.set_library_label(label.into());
        ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(items))));
        ui.set_current_item_index(0);
        ui.set_secondary_label(sec_label.into());
        ui.set_secondary_items(ModelRc::from(Rc::new(VecModel::from(sec_items))));
        ui.set_current_secondary_index(0);
    });

    let app_weak_2 = my_app.clone();
    let ui_weak_2 = ui.as_weak();
    ui.on_item_selected(move |index| {
        let mut app = app_weak_2.lock().unwrap();
        let ui = ui_weak_2.unwrap();
        app.item_selected(index);
        if app.app_mode == AppMode::Scales || app.app_mode == AppMode::Arpeggios {
             ui.set_interval_input_text(app.intervals_input.clone().into());
        }
    });

    let app_weak_3 = my_app.clone();
    let ui_weak_3 = ui.as_weak();
    ui.on_secondary_item_selected(move |index| {
        let mut app = app_weak_3.lock().unwrap();
        let ui = ui_weak_3.unwrap();
        app.secondary_item_selected(index);
        if app.app_mode == AppMode::Scales || app.app_mode == AppMode::Arpeggios {
             ui.set_interval_input_text(app.intervals_input.clone().into());
        }
    });

    ui.run()
}

/// Podgląd diagnostyczny (SOLITITO_DEBUG=1).
///
/// Interwały wypisujemy WZGLĘDEM ROZPOZNANEJ PRYMY, bo tak myśli się o akordzie:
/// "Gm zamiast Gm7" znaczy, że pod b7 powinna być wysoka liczba. Jeśli jest niska,
/// wąskim gardłem jest słyszenie; jeśli wysoka, a jakość i tak wychodzi "m",
/// to głowica jakości nie korzysta z informacji, którą ma pod ręką.
fn print_debug(p: &brain::Prediction) {
    const IV: [&str; 12] = ["R", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"];
    const QN: [&str; 11] = [
        "maj", "min", "maj7", "dom7", "min7", "m7b5", "dim7", "aug", "sus", "note", "N",
    ];

    if p.root_idx >= 12 {
        return;
    }

    let iv: String = (0..12)
        .map(|i| {
            let v = p.pitches[(p.root_idx + i) % 12];
            let bar = if v > 0.75 { "#" } else if v > 0.5 { "+" } else if v > 0.25 { "." } else { " " };
            format!("{}{:.0}{} ", IV[i], v * 100.0, bar)
        })
        .collect();

    let mut top: Vec<(usize, f32)> = p.qual_probs.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let quals: String = top
        .iter()
        .take(3)
        .map(|(i, v)| format!("{}={:.0}% ", QN[*i], v * 100.0))
        .collect();

    println!("{:<10} | {} | {}", p.chord, quals.trim_end(), iv.trim_end());
}
