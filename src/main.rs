#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod model;
mod audio;
mod brain;
mod fretboard;
mod i18n;
mod latch;
mod rng;
mod settings;
mod state;

use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::collections::HashMap;
use std::env;
use std::time::Duration;
use std::thread;

use audio::{AudioAnalysis, start_audio_stream, start_file_playback};
use brain::{ChordBrain, Prediction};
use latch::ChordLatch;
use i18n::Lang;
use settings::Settings;
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

    let default_gate_db: f32 = -34.0;            // ~0.02 in linear RMS
    let default_boost_enabled = true;
    let default_boost_gain = 5.0;

    let analysis_state = Arc::new(Mutex::new(AudioAnalysis {
        input_history: [[0.0; 168]; 48],
        frame_live: [false; 48],
        onset_id: 0,
        frames_since_onset: 0,
        spectrum_visual: [0.0; 48],
        chroma_sum: [0.0; 12],
        bass_boost_enabled: default_boost_enabled,
        bass_boost_gain: default_boost_gain, 
        noise_gate: db_to_lin(default_gate_db),
        input_level: 0.0,
    }));
    
    let ai_result_state = Arc::new(Mutex::new(AiResult::default()));

    let args: Vec<String> = env::args().collect();

    // --check: load the DSP weights and the model, report, exit. Without it a
    // package cannot be verified on a machine with no sound card and no display:
    // the weights are only read when the audio stream opens.
    if args.iter().any(|a| a == "--check") {
        let mut ok = true;
        match audio::CqtAnalyzer::new("dsp_weights.json") {
            Ok(_) => println!("✅ dsp_weights.json"),
            Err(e) => { eprintln!("❌ dsp_weights.json: {e}"); ok = false; }
        }
        match ChordBrain::new("best_model_v2_take6.onnx") {
            Ok(_) => println!("✅ best_model_v2_take6.onnx"),
            Err(e) => { eprintln!("❌ model: {e}"); ok = false; }
        }
        std::process::exit(if ok { 0 } else { 1 });
    }

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
    
    // AI THREAD
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
        
        // How much of the context window must carry real signal before we ask the
        // model at all. The trainer only built windows INSIDE a sustained chord, so
        // "half silence, half chord" is out of distribution - which is why chords
        // seemed to resolve only in the decay.
        const MIN_FILL: f32 = 0.9;

        // SOLITITO_DEBUG=1 prints WHAT the model hears and WHAT it is torn between.
        // Without it a mistake shows only as a chord name, which cannot separate
        // "it does not hear the seventh" from "it hears it and ignores it".
        let debug_ai = std::env::var("SOLITITO_DEBUG").is_ok();
        if debug_ai {
            println!("🔎 Diagnostics: root | qualities | intervals relative to the root");
        }

        loop {
            let (history, fill) = {
                let state = analysis_for_ai.lock().unwrap();
                (state.input_history, state.history_fill())
            };

            if fill < MIN_FILL {
                thread::sleep(Duration::from_millis(40));
                continue;
            }

            // The period is timed FROM THE START of inference. It used to be
            // "compute, then sleep 40 ms", so the real period was inference PLUS
            // 40 ms and varied with machine load.
            let started = std::time::Instant::now();
            if let Ok(pred) = brain.predict(&history) {
                if debug_ai {
                    print_debug(&pred);
                }
                if let Ok(mut res) = result_for_ai.lock() {
                    res.pred = pred;
                    res.updated = true;
                }
            }
            let spent = started.elapsed();
            if let Some(left) = Duration::from_millis(40).checked_sub(spent) {
                thread::sleep(left);
            }
        }
    });

    // Persistent settings, read once before the window is built so the app is in
    // the chosen mode from the first frame instead of jumping.
    let cfg = Settings::load();

    // Language resolved BEFORE building the window, so no English flashes first.
    let lang = Lang::from_setting(cfg.language);
    println!("🌍 UI language: {:?}", lang);

    let my_app = Arc::new(Mutex::new(MyApp::new(analysis_state.clone(), None)));
    my_app.lock().unwrap().set_mode(cfg.startup_mode);
    let ui = AppWindow::new()?;
    let ui_weak = ui.as_weak();

    {
        let app = my_app.lock().unwrap();
        let titles: Vec<SharedString> = app.song_library.iter()
            .map(|s| SharedString::from(&s.title))
            .collect();
        ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(titles))));
        ui.set_gate_db(default_gate_db);
        ui.set_boost_enabled(default_boost_enabled);
        ui.set_boost_gain(default_boost_gain);
        ui.set_current_mode(app.app_mode as i32); 
        ui.set_startup_mode(cfg.startup_mode);
        ui.set_language_idx(cfg.language);
        apply_language(&ui, lang);
        ui.set_interval_input_text(app.intervals_input.clone().into()); 
    }

    ui.window().set_position(PhysicalPosition::new(450, 10));

    let timer = Timer::default();
    let app_clone = my_app.clone();
    let result_for_ui = ai_result_state.clone();
    
    let keys_list: Vec<SharedString> = vec!["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        .into_iter().map(SharedString::from).collect();

    // Chord quality latch - see the `latch` module docs.
    let mut chord_latch = ChordLatch::default();
    // When the last prediction was consumed. The progress timer MUST be given
    // real elapsed time: the AI thread does inference AND sleeps 40 ms, so the
    // real interval is 55-90 ms and varies. A hard-coded 0.040 made the counter
    // run slower than the clock, turning a 0.6 s threshold into about a second.
    let mut last_ai_at: Option<std::time::Instant> = None;

    timer.start(TimerMode::Repeated, Duration::from_millis(16), move || {
        let ui = ui_weak.unwrap();
        let mut app = app_clone.lock().unwrap();

        let (spectrum_vis, in_level) = {
            let s = app.analysis_state.lock().unwrap();
            (s.spectrum_visual, s.input_level)
        };
        ui.set_input_level_db(lin_to_db(in_level));

        if !file_mode {
            app.noise_gate = db_to_lin(ui.get_gate_db());
        }
        app.bass_boost_enabled = ui.get_boost_enabled();
        app.bass_boost_gain = ui.get_boost_gain();
        app.chord_confidence = ui.get_chord_confidence();
        app.note_threshold = ui.get_note_threshold();
        app.transition_delay = ui.get_delay();
        app.set_random_mode(ui.get_random_enabled());
        
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

                // clear the flag once consumed
                res.updated = false;

                // Majority voting over 3 windows. probe_quality measured majority
                // voting as better than averaging the distributions - the model is
                // sometimes confident and wrong, and averaging carries that
                // confidence forward while voting damps it. Three rather than five:
                // with five, a new chord needed three predictions to outweigh the
                // old one, adding ~0.2 s to every transition.
                app.chord_history.push_back((chord.clone(), score));
                if app.chord_history.len() > 3 { app.chord_history.pop_front(); }
                
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

                // Latch: a chord does not change identity while it rings out. It
                // engages at high confidence but holds regardless - during the decay
                // the model reports the poorer quality at 94-96%.
                let (onset_id, since_onset) = {
                    let s = app.analysis_state.lock().unwrap();
                    (s.onset_id, s.frames_since_onset)
                };
                let shown = chord_latch.update(
                    ui.get_lock_quality(), onset_id, since_onset, best_c, current_confidence,
                );

                ui.set_ai_text(
                    format!("{} ({:.0}%)", shown, current_confidence * 100.0).into()
                );

                // Real elapsed time, not assumed. Capped so a thread stall does not
                // jump the whole threshold at once.
                let now = std::time::Instant::now();
                let dt = last_ai_at
                    .map(|t| now.duration_since(t).as_secs_f32())
                    .unwrap_or(0.040)
                    .min(0.25);
                last_ai_at = Some(now);
                app.check_progress_with_ai(dt, &shown, current_confidence);
            }
        }

        ui.set_song_title(app.song_title.clone().into());
        
        // Scales in random mode redraw the key on their own; push it to the
        // combo, but only on a real change so it does not fight the user
        // mid-selection (the binding is two-way).
        if app.app_mode == AppMode::Scales
            && ui.get_current_secondary_index() != app.secondary_index as i32
        {
            ui.set_current_secondary_index(app.secondary_index as i32);
        }

        if app.app_mode == AppMode::Fretboard {
            // Minimal by design: the note, and under it where to play it.
            let name = match app.fret_target {
                Some(pc) => model::NoteName::from_index(pc).to_string().to_string(),
                None => "...".to_string(),
            };
            ui.set_chord_name(name.into());
            ui.set_start_hint(app.region.describe().into());
            ui.set_chord_text_color(slint::Brush::SolidColor(match app.match_status {
                MatchStatus::Exact => Color::from_rgb_u8(50, 255, 50),
                _ => Color::from_rgb_u8(255, 255, 255),
            }));
            ui.set_interval_names(ModelRc::from(Rc::new(VecModel::from(Vec::<SharedString>::new()))));
        } else if app.chords.is_empty() { 
            ui.set_chord_name(ui.global::<Tr>().get_no_data()); 
        } else {
            let curr_chord = &app.chords[app.current_chord_index];
            
            if app.app_mode == AppMode::Scales {
                 ui.set_chord_name(curr_chord.root.to_string().into());
            } else {
                 let q_str = curr_chord.quality.to_string();
                 let quality_display = if q_str.is_empty() { "Maj" } else { q_str.as_str() };
                 ui.set_chord_name(format!("{} {}", curr_chord.root.to_string(), quality_display).into());
            }
            
            // Suggestion only: the model gives pitch classes with no position,
            // so nothing here is verified. Empty string hides the line.
            let hint = match app.start_hint {
                Some(i) if i < state::START_STRINGS.len() => {
                    i18n::strings(lang).start_from.replace("{}", state::START_STRINGS[i])
                }
                _ => String::new(),
            };
            ui.set_start_hint(hint.into());

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
                let active_indices = app.ordered_active_indices(curr_chord);
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
    
    // Saved immediately - there is no "Save" button, so the setting has to survive
    // closing the window without an extra step.
    {
        let cur = cfg.clone();
        ui.on_startup_mode_changed(move |mode_idx| {
            Settings { startup_mode: mode_idx, ..cur.clone() }.save();
        });
    }
    {
        let cur = cfg.clone();
        let uw = ui.as_weak();
        ui.on_language_changed(move |idx| {
            Settings { language: idx, ..cur.clone() }.save();
            // Strings are swapped immediately; no restart needed.
            if let Some(ui) = uw.upgrade() {
                apply_language(&ui, Lang::from_setting(idx));
            }
        });
    }

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

/// Diagnostic output (SOLITITO_DEBUG=1).
///
/// Intervals are printed RELATIVE TO THE DETECTED ROOT, which is how one thinks
/// about a chord: "Gm instead of Gm7" means b7 should show a high number. If it
/// is low, hearing is the bottleneck; if it is high and the quality still comes
/// out "m", the quality head is not using information it already has.
/// Fills the UI `Tr` global with the chosen language's strings.
fn apply_language(ui: &AppWindow, lang: Lang) {
    let t = i18n::strings(lang);
    let g = ui.global::<Tr>();
    g.set_chords(t.chords.into());
    g.set_intervals(t.intervals.into());
    g.set_scales(t.scales.into());
    g.set_arpeggios(t.arpeggios.into());
    g.set_settings(t.settings.into());
    g.set_close(t.close.into());
    g.set_settings_title(t.settings_title.into());
    g.set_audio_calibration(t.audio_calibration.into());
    g.set_noise_gate(t.noise_gate.into());
    g.set_gate_hint(t.gate_hint.into());
    g.set_bass_boost(t.bass_boost.into());
    g.set_lock_quality(t.lock_quality.into());
    g.set_random_order(t.random_order.into());
    g.set_random_hint(t.random_hint.into());
    g.set_fretboard(t.fretboard.into());
    g.set_startup_mode(t.startup_mode.into());
    g.set_chord_confidence(t.chord_confidence.into());
    g.set_note_threshold(t.note_threshold.into());
    g.set_hold_time(t.hold_time.into());
    g.set_show_debug(t.show_debug.into());
    g.set_ai_prediction(t.ai_prediction.into());
    g.set_intervals_label(t.intervals_label.into());
    g.set_intervals_hint(t.intervals_hint.into());
    g.set_intervals_placeholder(t.intervals_placeholder.into());
    g.set_next(t.next.into());
    g.set_no_data(t.no_data.into());
    g.set_language(t.language.into());
    g.set_lang_auto(t.lang_auto.into());
}

/// Linear level (RMS) -> dBFS. Floored at -72 dB so silence is not -inf, capped
/// at 0 dB where the scale ends.
fn lin_to_db(v: f32) -> f32 {
    if v <= 2.5e-4 { -72.0 } else { (20.0 * v.log10()).clamp(-72.0, 0.0) }
}

fn db_to_lin(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

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

#[cfg(test)]
mod db_tests {
    use super::*;

    #[test]
    fn db_conversion_round_trips() {
        for db in [-72.0f32, -48.0, -34.0, -20.0, -6.0, 0.0] {
            let back = lin_to_db(db_to_lin(db));
            assert!((back - db).abs() < 0.01, "{db} dB -> {back} dB");
        }
    }

    #[test]
    fn the_old_default_threshold_is_minus_34_db() {
        // The previous default gate was 0.02 in linear RMS.
        assert!((db_to_lin(-34.0) - 0.02).abs() < 0.001);
    }

    #[test]
    fn the_slider_range_reaches_laptop_mic_noise() {
        // A laptop microphone with AGC can sit at 0.3-0.5 RMS of noise. The old
        // slider ended at 0.1 linear, so the gate COULD NOT be set above it.
        assert!(db_to_lin(0.0) >= 1.0, "the top end is full scale");
        assert!(db_to_lin(-72.0) < 0.0003, "the bottom end must be practically silence");
    }

    #[test]
    fn silence_does_not_produce_infinity() {
        assert_eq!(lin_to_db(0.0), -72.0);
        assert!(lin_to_db(1e-9).is_finite());
    }
}
