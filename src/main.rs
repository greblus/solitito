#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod model;
mod arpeggio;
mod audio;
mod brain;
mod cli;
mod controls;
mod devices;
mod diagrams;
mod flux;
mod journal;
mod formulas;
mod fretboard;
mod i18n;
mod interval_levels;
mod latch;
mod rng;
mod settings;
mod state;
mod strike;
mod tab;
mod ui_frame;
mod voices;

use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::cell::RefCell;
use std::env;
use std::time::Duration;
use std::thread;

use audio::{db_to_lin, AudioAnalysis, InputInfo, start_file_playback};
use devices::{channel_choices, fill_device_list, open_input};
use brain::{ChordBrain, Prediction};
use i18n::Lang;
use settings::Settings;
use state::{MyApp, AppMode, ArpExercise};

use slint::{Timer, TimerMode, Model, ModelRc, VecModel, SharedString, PhysicalPosition, PhysicalSize};

slint::include_modules!();

// Compiled in for the same reason as the chord shapes: no loose asset files to
// lose in a release package. Black on transparent, recoloured by Slint.
const ICON_SHUFFLE: &str = include_str!("assets/icons/shuffle.svg");
const ICON_GEAR: &str = include_str!("assets/icons/gear.svg");
const ICON_STAR: &str = include_str!("assets/icons/star.svg");
/// The same star with its inside filled. Two files rather than one plus the
/// inverted plate the other toggles use: a whole square lighting up said
/// "pressed", where what is meant is "kept".
const ICON_STAR_ON: &str = include_str!("assets/icons/star_on.svg");
const ICON_PAUSE: &str = include_str!("assets/icons/pause.svg");
const ICON_PLAY: &str = include_str!("assets/icons/play.svg");

/// How far the UI may be scaled from its design size. The floor is where the
/// smallest labels stop being readable, and below it the window stops shrinking
/// too, because `min-width`/`min-height` in the .slint file are in logical
/// pixels and the backend turns them into a physical minimum using this factor.
/// The ceiling only guards against a nonsense window size; nothing needs it.
const SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.5..=4.0;

/// Sets the window's scale factor so the fixed-size UI fills as much of the
/// window as it can without distortion.
///
/// This is the whole of the resizing behaviour. Nothing in the UI is written in
/// relative units: the layout is drawn at `design_w` x `design_h` and the scale
/// factor - the same knob a HiDPI screen turns - zooms it. So the proportions
/// are fixed by construction, text and the SVG shapes are re-rendered sharp at
/// the new size rather than stretched, and the leftover slack on one axis
/// becomes a black margin.
///
/// The factor is derived from the PHYSICAL size, which the scale factor does not
/// affect, so setting it cannot feed back into another resize.
fn fit_ui_to_window(ui: &AppWindow) {
    let size = ui.window().size();
    // Before the window is mapped, and when minimised on some platforms.
    if size.width == 0 || size.height == 0 {
        return;
    }
    // `design_*` are lengths in logical pixels, which at this point means the
    // design units themselves - that is what makes them the right divisor.
    let scale = (size.width as f32 / ui.get_design_w())
        .min(size.height as f32 / ui.get_design_h())
        .clamp(*SCALE_RANGE.start(), *SCALE_RANGE.end());
    // Nothing to correct when the factor is the one already in force - and the
    // pair of events below is not free: it restates the window's logical size,
    // which is the last thing to do to a window a window manager is in the
    // middle of resizing.
    if (ui.window().scale_factor() - scale).abs() < f32::EPSILON {
        return;
    }
    ui.window()
        .dispatch_event(slint::platform::WindowEvent::ScaleFactorChanged { scale_factor: scale });
    // Required, and not obvious: changing the scale factor does not restate the
    // window's LOGICAL size, so the root item keeps the size it had under the
    // old factor. Slint says as much - `event_loop.rs` carries a TODO to send
    // this event itself. Without it `root.width` is stale, the centring below
    // the layout is computed from the wrong number, and the UI sits offset in
    // the window at the wrong size.
    ui.window().dispatch_event(slint::platform::WindowEvent::Resized {
        size: slint::LogicalSize::new(size.width as f32 / scale, size.height as f32 / scale),
    });
}

/// Assigns a property only when the value actually differs.
///
/// Slint marks the scene dirty on every assignment, changed or not, and the tick
/// runs sixty times a second - so pushing unchanged values kept the window
/// redrawing continuously with nothing happening. Confirmed with
/// SLINT_DEBUG_PERFORMANCE=refresh_lazy: a steady 60 fps at rest, in the mode
/// where Slint is supposed to draw only on demand.
fn set_if_changed<T: PartialEq>(current: T, new: T, set: impl FnOnce(T)) {
    if current != new {
        set(new);
    }
}

/// Pitch-class names in CQT order, so a probe row reads as notes and not indices.
const NOTE_NAMES: [&str; 12] =
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/// Which model file to load.
///
/// `SOLITITO_MODEL` overrides it, so two models can be measured against the same
/// recording without moving files around - which is how a comparison ends up
/// being run twice on the same one.
fn model_path() -> String {
    if let Ok(p) = std::env::var("SOLITITO_MODEL") {
        return p;
    }
    // The chord outputs alone first. The file with the old onset head carries
    // the same three unchanged, so a package from before still works; the
    // plain take6 file is an older network with other answers, the last resort.
    for name in [
        "best_model_v2_take6_chords.onnx",
        "best_model_v2_take6_onset.onnx",
        "best_model_v2_take6.onnx",
    ] {
        if std::path::Path::new(name).exists() {
            return name.to_string();
        }
    }
    "best_model_v2_take6_chords.onnx".to_string()
}

fn svg_icon(src: &str) -> slint::Image {
    slint::Image::load_from_svg_data(src.as_bytes()).unwrap_or_default()
}

/// Everything the settings panel can change, for telling "touched" from
/// "untouched". The gear lights up only when the current values differ from the
/// ones the app started with - lighting it merely because the panel is open said
/// nothing the open panel did not already say.
#[derive(Clone, PartialEq)]
struct SettingsSnapshot {
    gate_db: f32,
    chord_confidence: f32,
    note_threshold: f32,
    delay: f32,
    boost_gain: f32,
    boost_enabled: bool,
    lock_quality: bool,
    short_verdict: bool,
    single_notes: bool,
    require_onset: bool,
    shuffle_chords: bool,
    random_enabled: bool,
    show_diagrams: bool,
    preview: i32,
    voice_leading: bool,
    lead_line: bool,
    tab_frets: bool,
    arp_exercise: i32,
    arp_direction: i32,
    arp_quality: i32,
    scale_repeat_root: bool,
    scale_direction: i32,
    scale_from: i32,
    scale_there_and_back: bool,
    key_move: i32,
    scale_in_order: bool,
    scale_e_to_e: bool,
    show_full_shapes: bool,
    show_shell_shapes: bool,
    ai_debug: bool,
    startup_mode: i32,
    language: i32,
    intervals: String,
    show_spectrum: bool,
    audio_device: i32,
    audio_channel: i32,
    jazz_notation: bool,
    formula_exercise: i32,
    formula_placement: i32,
    formula_notes: i32,
    formula_key: String,
    formula_random_key: bool,
    formula_required: String,
    formula_show_names: bool,
    formula_show_similar: bool,
    formula_show_chords: bool,
    in_order: bool,
    interval_in_order: bool,
    interval_level: i32,
    debug_console: bool,
    show_input_meter: bool,
    show_heard: bool,
}

impl SettingsSnapshot {
    fn read(ui: &AppWindow) -> Self {
        Self {
            gate_db: ui.get_gate_db(),
            chord_confidence: ui.get_chord_confidence(),
            note_threshold: ui.get_note_threshold(),
            delay: ui.get_delay(),
            boost_gain: ui.get_boost_gain(),
            boost_enabled: ui.get_boost_enabled(),
            lock_quality: ui.get_lock_quality(),
            short_verdict: ui.get_short_verdict(),
            single_notes: ui.get_single_notes(),
            require_onset: ui.get_require_onset(),
            shuffle_chords: ui.get_shuffle_chords(),
            random_enabled: ui.get_random_enabled(),
            show_diagrams: ui.get_show_diagrams(),
            preview: ui.get_preview(),
            voice_leading: ui.get_voice_leading(),
            lead_line: ui.get_lead_line(),
            tab_frets: ui.get_tab_frets(),
            arp_exercise: ui.get_arp_exercise(),
            arp_direction: ui.get_arp_direction(),
            arp_quality: ui.get_arp_quality(),
            scale_repeat_root: ui.get_scale_repeat_root(),
            scale_direction: ui.get_scale_direction(),
            scale_from: ui.get_scale_from(),
            scale_there_and_back: ui.get_scale_there_and_back(),
            key_move: ui.get_key_move(),
            scale_in_order: ui.get_scale_in_order(),
            scale_e_to_e: ui.get_scale_e_to_e(),
            show_full_shapes: ui.get_show_full_shapes(),
            show_shell_shapes: ui.get_show_shell_shapes(),
            ai_debug: ui.get_ai_debug_visible(),
            startup_mode: ui.get_startup_mode(),
            language: ui.get_language_idx(),
            intervals: ui.get_interval_input_text().to_string(),
            show_spectrum: ui.get_show_spectrum(),
            audio_device: ui.get_audio_device_index(),
            audio_channel: ui.get_audio_channel_index(),
            jazz_notation: ui.get_jazz_notation(),
            formula_exercise: ui.get_formula_exercise(),
            formula_placement: ui.get_formula_placement(),
            formula_notes: ui.get_formula_notes(),
            formula_key: ui.get_formula_key_text().to_string(),
            formula_random_key: ui.get_formula_random_key(),
            formula_required: ui.get_formula_required_text().to_string(),
            formula_show_names: ui.get_formula_show_names(),
            formula_show_similar: ui.get_formula_show_similar(),
            formula_show_chords: ui.get_formula_show_chords(),
            in_order: ui.get_in_order(),
            interval_in_order: ui.get_interval_in_order(),
            interval_level: ui.get_interval_level(),
            debug_console: ui.get_debug_console(),
            show_input_meter: ui.get_show_input_meter(),
            show_heard: ui.get_show_heard(),
        }
    }
}

#[derive(Clone, Default)]
struct AiResult {
    pred: Prediction,
    updated: bool,
    /// How full the context window was when this was asked. The chord NAME is
    /// only believed on a full one - that is what the model was trained on -
    /// while the pitch head is read whatever it says.
    fill: f32,
}

// How much of the context window must carry real signal before the model is
// asked at all, and how much before its CHORD NAME is believed.
//
// They were one number, 0.9, and that number is right for the name: the
// trainer only built windows INSIDE a sustained chord, so "half silence,
// half chord" is out of distribution. But it also meant the model was not
// asked at all until 43 frames - 688 ms - of unbroken sound, and playing one
// note at a time never gets there. The screen froze on the last chord it
// had, and everything downstream froze with it.
//
// Measured on notes with known onsets: at 50-70% fill the pitch head names
// an isolated note correctly in every frame of the measurement. So the model
// is asked from half a window, and the name waits for the full one.
//
// Both live here rather than inside the thread that asks: the gate on the
// name is on the other side of the channel, and written out as 0.9 there it
// was the same number in two places, free to drift apart.
const MIN_FILL: f32 = 0.5;
const MIN_FILL_CHORD: f32 = 0.9;

fn main() -> Result<(), slint::PlatformError> {
    let _keep_awake = keep_screen_awake();
    audio::hush_alsa();
    if let Err(e) = ort::init().with_name("Solitito").commit() {
        eprintln!("CRITICAL: Failed to initialize ONNX Runtime: {}", e);
    }
    let analysis_state = new_analysis_state();
    let ai_result_state = Arc::new(Mutex::new(AiResult::default()));

    let args: Vec<String> = env::args().collect();
    if let Some(exit_code) = cli::run_report(&args) {
        std::process::exit(exit_code);
    }
    let file_mode = match args.get(2).filter(|_| args[1] == "--file") {
        Some(path) => {
            println!("Starting FILE mode: {}", path);
            if let Err(e) = start_file_playback(path.clone(), analysis_state.clone()) {
                eprintln!("ERR FILE: {}", e);
                return Ok(());
            }
            true
        }
        None => {
            println!("Starting LIVE mode...");
            false
        }
    };
    let _ai_thread = spawn_model_thread(analysis_state.clone(), ai_result_state.clone());

    // Persistent settings, read once before the window is built so the app is in
    // the chosen mode from the first frame instead of jumping.
    let cfg = Settings::load();
    // Language resolved BEFORE building the window, so no English flashes first.
    let lang = Lang::from_setting(cfg.language);
    println!("🌍 UI language: {:?}", lang);
    let my_app = Arc::new(Mutex::new(restore_app(&cfg, analysis_state.clone())));

    let ui = AppWindow::new()?;
    show_settings(&ui, &my_app.lock().unwrap(), &cfg, lang);
    ui.window().set_position(PhysicalPosition::new(450, 10));

    let shared = controls::Shared {
        app: my_app.clone(),
        // The single live copy of the settings. Each closure used to write a
        // clone taken at startup, so saving one setting rewrote the others with
        // their values from launch time - changing the mode and then the
        // language silently reverted the mode. They now all edit this.
        cfg: Rc::new(RefCell::new(cfg.clone())),
        analysis: analysis_state.clone(),
        // Held so it can be swapped when the device or channel changes.
        audio_in: Rc::new(RefCell::new(None)),
        opened: Rc::new(RefCell::new(InputInfo::default())),
        device_names: Rc::new(RefCell::new(Vec::new())),
    };
    if !file_mode {
        open_saved_input(&ui, &shared, &cfg, lang);
    }

    let _restore_window = restore_window(&ui, &cfg);
    fit_ui_to_window(&ui);
    let frame_shared = ui_frame::Shared {
        app: my_app.clone(),
        ai_result: ai_result_state.clone(),
        cfg: shared.cfg.clone(),
        device_names: shared.device_names.clone(),
        opened: shared.opened.clone(),
    };
    let mut frame = ui_frame::UiFrame::new(&ui, frame_shared, file_mode, lang);
    let timer = Timer::default();
    timer.start(TimerMode::Repeated, Duration::from_millis(16), move || frame.run());

    controls::connect(&ui, &shared);
    ui.run()
}

/// Keeps the screen on while the app runs: a trainer is watched, not typed at.
fn keep_screen_awake() -> Option<keepawake::AwakeHandle> {
    let awake = keepawake::Builder::new()
        .display(true)
        .idle(true)
        .sleep(true)
        .create();
    if let Err(e) = &awake {
        eprintln!("Warning: Could not enable keep-awake: {}", e);
    }
    awake.ok()
}

const DEFAULT_GATE_DB: f32 = -34.0; // ~0.02 in linear RMS
const DEFAULT_BOOST_ENABLED: bool = true;
const DEFAULT_BOOST_GAIN: f32 = 5.0;

/// What the audio thread and the rest share, before anything has been heard.
fn new_analysis_state() -> Arc<Mutex<AudioAnalysis>> {
    Arc::new(Mutex::new(AudioAnalysis {
        input_history: [[0.0; 168]; 48],
        frame_live: [false; 48],
        onset_id: 0,
        flux_id: 0,
        flux_age: u32::MAX,
        hops: 0,
        voices: 0,
        class_strikes: [0; 12],
        class_strike_age: [u32::MAX; 12],
        strikes_live: false,
        frames_since_onset: 0,
        spectrum_visual: [0.0; 48],
        chroma_sum: [0.0; 12],
        bass_boost_enabled: DEFAULT_BOOST_ENABLED,
        bass_boost_gain: DEFAULT_BOOST_GAIN,
        noise_gate: db_to_lin(DEFAULT_GATE_DB),
        input_level: 0.0,
        cqt_pitch: None,
        cqt_semitone: None,
        gate_open: false,
        frames_seen: 0,
    }))
}

/// The thread that asks the model, every 40 ms, about the window of features
/// the audio thread keeps, and leaves the answer in `result`.
///
/// Named so `top -H` can tell inference apart from the UI thread - both were
/// just "solitito", which made "is it the model or the drawing?" unanswerable
/// without a debugger.
fn spawn_model_thread(
    analysis: Arc<Mutex<AudioAnalysis>>,
    result: Arc<Mutex<AiResult>>,
) -> std::io::Result<thread::JoinHandle<()>> {
    thread::Builder::new()
        .name("solitito-ai".into())
        .spawn(move || run_model(analysis, result))
}

fn run_model(analysis: Arc<Mutex<AudioAnalysis>>, result: Arc<Mutex<AiResult>>) {
    let model_filename = model_path();
    let mut brain = match ChordBrain::new(&model_filename) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("WARNING: Could not load AI Model: {}", e);
            return;
        }
    };

    // SOLITITO_DEBUG=1 prints WHAT the model hears and WHAT it is torn between.
    // Without it a mistake shows only as a chord name, which cannot separate
    // "it does not hear the seventh" from "it hears it and ignores it".
    let debug_ai = std::env::var("SOLITITO_DEBUG").is_ok();
    if debug_ai {
        println!("🔎 Diagnostics: root | qualities | intervals relative to the root");
    }

    loop {
        let (history, fill) = {
            let state = analysis.lock().unwrap();
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
            if let Ok(mut res) = result.lock() {
                res.pred = pred;
                res.fill = fill;
                res.updated = true;
            }
        }
        let spent = started.elapsed();
        if let Some(left) = Duration::from_millis(40).checked_sub(spent) {
            thread::sleep(left);
        }
    }
}

/// The app as it was left: the mode, the exercises, the standard and the scale.
fn restore_app(cfg: &Settings, analysis: Arc<Mutex<AudioAnalysis>>) -> MyApp {
    let mut app = MyApp::new(analysis);
    // Before the first library is read: which arpeggio exercise is running
    // decides what the library IS - the studies or the tunes.
    app.arp_exercise = cfg.arp_exercise.into();
    app.arp_direction = cfg.arp_direction.into();
    app.arp_quality = cfg.arp_quality;
    // Same for Formulas: whether the exercise is planted on a tune decides
    // whether the first combo holds one, and the tune below is picked from
    // that list.
    app.formula_exercise = (cfg.formula_exercise as usize).into();
    app.set_mode(cfg.startup_mode);
    app.require_onset = cfg.require_onset_for(cfg.startup_mode);
    app.lead_line = cfg.lead_line;
    app.interval_level = Some(cfg.interval_level.min(interval_levels::LEVELS.len() - 1));
    // The standard and the scale that were being worked on. Silently
    // ignored if the library no longer holds them - see `select_song`.
    app.select_song(&cfg.song);
    app.select_scale(&cfg.scale);
    app
}

/// Every control in the window set to the saved settings.
fn show_settings(ui: &AppWindow, app: &MyApp, cfg: &Settings, lang: Lang) {
    let (items, _) = library_lists(app, lang);
    ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(items))));
    // Per device: the same room and the same fingers give levels tens of
    // decibels apart through an interface and through a laptop microphone.
    ui.set_gate_db(
        cfg.gate_for(cfg.audio_device.as_deref())
            .unwrap_or(DEFAULT_GATE_DB),
    );
    ui.set_boost_enabled(DEFAULT_BOOST_ENABLED);
    ui.set_boost_gain(DEFAULT_BOOST_GAIN);
    ui.set_current_mode(app.app_mode as i32); 
    ui.set_current_item_index(app.selected_library_idx as i32);
    ui.set_startup_mode(cfg.startup_mode);
    ui.set_language_idx(cfg.language);
    ui.set_short_verdict(cfg.short_verdict);
    ui.set_single_notes(cfg.single_notes);
    ui.set_require_onset(cfg.require_onset_for(cfg.startup_mode));
    ui.set_shuffle_chords(cfg.shuffle_chords);
    ui.set_show_diagrams(cfg.show_diagrams);
    // Per mode, so the app opens showing the mode it opens in the way that
    // mode was left.
    ui.set_preview(cfg.preview_for(cfg.startup_mode) as i32);
    ui.set_voice_leading(cfg.voice_leading);
    ui.set_lead_line(cfg.lead_line);
    ui.set_tab_frets(cfg.tab_frets);
    ui.set_arp_exercise(cfg.arp_exercise as i32);
    ui.set_arp_direction(cfg.arp_direction as i32);
    ui.set_arp_quality(cfg.arp_quality as i32);
    ui.set_scale_repeat_root(cfg.scale_repeat_root);
    ui.set_scale_direction(state::ScaleDirection::from(cfg.scale_direction) as i32);
    ui.set_scale_in_order(cfg.scale_in_order);
    ui.set_scale_from(cfg.scale_from.min(4) as i32);
    ui.set_scale_e_to_e(cfg.scale_e_to_e);
    ui.set_fret_hint(cfg.fret_hint);
    ui.set_scale_there_and_back(cfg.scale_there_and_back);
    ui.set_key_move(cfg.key_move.min(4) as i32);
    ui.set_show_full_shapes(cfg.show_full_shapes);
    ui.set_show_shell_shapes(cfg.show_shell_shapes);
    ui.set_jazz_notation(cfg.jazz_notation);
    ui.set_formula_exercise(cfg.formula_exercise as i32);
    ui.set_formula_placement(cfg.formula_placement as i32);
    ui.set_formula_notes(cfg.formula_notes as i32);
    ui.set_formula_key_text(cfg.formula_key.clone().into());
    ui.set_formula_random_key(cfg.formula_random_key);
    ui.set_formula_required_text(cfg.formula_required.clone().into());
    ui.set_formula_show_names(cfg.formula_note_names);
    ui.set_formula_show_similar(cfg.formula_show_similar);
    ui.set_formula_show_chords(cfg.formula_show_chords);
    ui.set_in_order(cfg.formula_in_order);
    ui.set_interval_in_order(cfg.interval_in_order);
    ui.set_interval_level(cfg.interval_level.min(interval_levels::LEVELS.len() - 1) as i32);
    ui.set_debug_console(cfg.debug_console);
    ui.set_show_input_meter(cfg.show_input_meter);
    ui.set_show_heard(cfg.show_heard);
    ui.set_quick_open(cfg.quick_options_open);
    ui.set_app_version(env!("CARGO_PKG_VERSION").into());
    ui.set_show_spectrum(cfg.show_spectrum);
    ui.set_ai_debug_visible(cfg.ai_debug);
    ui.set_icon_shuffle(svg_icon(ICON_SHUFFLE));
    ui.set_icon_gear(svg_icon(ICON_GEAR));
    ui.set_icon_star(svg_icon(ICON_STAR));
    ui.set_icon_star_on(svg_icon(ICON_STAR_ON));
    ui.set_icon_pause(svg_icon(ICON_PAUSE));
    ui.set_icon_play(svg_icon(ICON_PLAY));
    apply_language(ui, lang, Some(app));
    ui.set_interval_input_text(model::shown_phrase(&app.intervals_input).into()); 
}

/// Lists the inputs and opens the one saved, or the default.
///
/// Listed BEFORE the stream opens, always. Enumerating means opening every
/// device in turn, and a card that is already taken cannot be opened - so a
/// list built after this app has a card of its own is missing that card, and
/// one built while a sound server holds the hardware is missing all of them.
/// ALSA's own complaints while it probes are silenced by `hush_alsa`.
fn open_saved_input(ui: &AppWindow, shared: &controls::Shared, cfg: &Settings, lang: Lang) {
    let t = i18n::strings(lang);
    let present = audio::list_input_devices();
    let mut names = present.clone();
    // The saved device stays in the picker even when the scan missed it, so
    // the panel shows what the app is trying to use - marked, so it does not
    // read as an interface that is plugged in when it is not.
    if let Some(want) = cfg.audio_device.as_deref() {
        if !names.iter().any(|n| n == want) {
            names.push(want.to_string());
        }
    }
    let chosen = cfg
        .audio_device
        .as_deref()
        .and_then(|want| names.iter().position(|n| n == want))
        .map(|i| i as i32 + 1)
        .unwrap_or(0);
    fill_device_list(ui, &names, t.audio_default, &present, None);
    *shared.device_names.borrow_mut() = names;
    ui.set_audio_device_index(chosen);
    ui.set_audio_channel_index(cfg.audio_channel.max(1) as i32 - 1);

    // A saved device that is no longer plugged in falls back to the default
    // rather than refusing to open anything.
    let want = (chosen > 0)
        .then(|| shared.device_names.borrow()[chosen as usize - 1].clone());
    open_input(&shared.analysis, &shared.audio_in, &shared.opened, ui, want.as_deref(), cfg.audio_channel);
    ui.set_audio_channels(ModelRc::from(Rc::new(VecModel::from(channel_choices(
        ui.get_audio_channel_count(),
        t.audio_one,
    )))));
}

/// How the window opens: as it was left - on full screen, or at the size it
/// had - and on a first launch, maximised, so it has the screen's proportions
/// rather than the narrow ones of the layout's minimum.
enum Opening {
    FullScreen,
    Size(u32, u32),
    Maximised,
}

/// Restoring the window has to wait for the event loop. The window does not
/// exist until then - `show()` only queues its creation - and a size set
/// before it exists is dropped, leaving the window at its preferred size.
/// Verified on this machine: setting it any earlier had no effect at all.
///
/// The timer has to be held for as long as the window lives, or it is dropped
/// and never fires. The closure only gets a weak handle, so it can stop itself.
fn restore_window(ui: &AppWindow, cfg: &Settings) -> Rc<Timer> {
    let opening = match (cfg.fullscreen, cfg.window_w, cfg.window_h) {
        (true, _, _) => Opening::FullScreen,
        (false, Some(w), Some(h)) => Opening::Size(w, h),
        _ => Opening::Maximised,
    };
    let restore = Rc::new(Timer::default());
    let uw = ui.as_weak();
    let timer = Rc::downgrade(&restore);
    let give_up_at = std::time::Instant::now() + Duration::from_secs(3);
    // Asking once is not enough. Slint sizes the window from the layout's
    // preferred size during its first pass, which lands after the event loop
    // has already started and overwrites anything set before it - measured
    // here at somewhere between 300ms and 600ms after launch. So ask
    // repeatedly and stop as soon as it holds, rather than sleeping for a
    // guessed interval that a slower machine would miss.
    restore.start(TimerMode::Repeated, Duration::from_millis(50), move || {
        let Some(ui) = uw.upgrade() else { return };
        let window = ui.window();
        let held = match opening {
            Opening::FullScreen => window.is_fullscreen(),
            Opening::Size(w, h) => (window.size().width, window.size().height) == (w, h),
            Opening::Maximised => window.is_maximized(),
        };
        // Two ways out, because a window manager is entitled to refuse: it
        // took, or it has had long enough that it never will. Without the
        // deadline a tiling WM would leave this firing forever.
        if held || std::time::Instant::now() > give_up_at {
            if let Some(timer) = timer.upgrade() {
                timer.stop();
            }
            return;
        }
        match opening {
            Opening::FullScreen => window.set_fullscreen(true),
            Opening::Size(w, h) => window.set_size(PhysicalSize::new(w, h)),
            Opening::Maximised => window.set_maximized(true),
        }
        fit_ui_to_window(&ui);
    });
    restore
}

/// The two combo labels a mode shows over its pickers.
///
/// Wanted in two places - when the mode changes and when the language does -
/// and it used to be built only in the first, from a language captured at
/// startup. Switching to English left "Utwór" standing over an English panel.
fn mode_labels(t: &i18n::Strings, mode: AppMode, arp_study: bool) -> (&'static str, &'static str) {
    match mode {
        AppMode::Scales => (t.select_scale, t.key_root),
        AppMode::Arpeggios if arp_study => (t.select_study, t.key_root),
        AppMode::Arpeggios => (t.select_song, ""),
        _ => (t.select_song, ""),
    }
}

/// The twelve keys, as the second combo lists them in Scales.
fn keys_list() -> Vec<SharedString> {
    ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        .into_iter()
        .map(SharedString::from)
        .collect()
}

/// What the two combos hold in the current mode, in the current language.
///
/// Scale names are translated here rather than in `MyApp`: the state keeps the
/// name the definition file gave, which is what a user's own scale is matched
/// and saved by. Only the screen changes language.
fn library_lists(app: &MyApp, lang: Lang) -> (Vec<SharedString>, Vec<SharedString>) {
    let scales = || -> Vec<SharedString> {
        app.scale_definitions
            .iter()
            .map(|s| SharedString::from(i18n::scale_name(lang, &s.name)))
            .collect()
    };
    let songs = || -> Vec<SharedString> {
        app.song_library.iter().map(|s| SharedString::from(&s.title)).collect()
    };
    match app.app_mode {
        AppMode::Scales => (scales(), keys_list()),
        // A study picks the phrase and the key it stands in; over the changes
        // the tune brings its own chords and the direction is a setting.
        AppMode::Arpeggios if app.arp_exercise == ArpExercise::Study => (
            app.arpeggio_patterns
                .iter()
                .map(|s| SharedString::from(i18n::arpeggio_name(lang, &s.name)))
                .collect(),
            keys_list(),
        ),
        AppMode::Arpeggios => (songs(), vec![]),
        _ => (songs(), vec![]),
    }
}

/// Fills the UI `Tr` global with the chosen language's strings.
fn apply_language(ui: &AppWindow, lang: Lang, app: Option<&MyApp>) {
    let t = i18n::strings(lang);
    // Not everything on screen goes through the Tr global: these are built in
    // Rust and stayed in whatever language they were made in.
    let (label, sec_label) = mode_labels(&t, AppMode::from(ui.get_current_mode()), ui.get_arp_exercise() == 0);
    ui.set_library_label(label.into());
    ui.set_secondary_label(sec_label.into());
    // The scale names among them: they are the library list in Scales, so the
    // list is dealt again in the new language, and the choice standing in it
    // kept - rebuilding a model resets the index.
    if let Some(app) = app {
        let (items, sec_items) = library_lists(app, lang);
        let (chosen, sec_chosen) = (ui.get_current_item_index(), ui.get_current_secondary_index());
        ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(items))));
        ui.set_secondary_items(ModelRc::from(Rc::new(VecModel::from(sec_items))));
        ui.set_current_item_index(chosen);
        ui.set_current_secondary_index(sec_chosen);
    }
    // The exercise list is named in the language too; the choice standing in
    // it is kept, as above.
    // The scale's two lists too, their choices kept.
    let (from, moves) = (ui.get_scale_from(), ui.get_key_move());
    let starts: Vec<slint::SharedString> =
        ["1", "3", "5", "7", t.scale_from_drawn].iter().map(|&s| s.into()).collect();
    ui.set_scale_from_items(ModelRc::from(Rc::new(VecModel::from(starts))));
    let next_keys: Vec<slint::SharedString> = t.key_moves.iter().map(|&s| s.into()).collect();
    ui.set_key_move_items(ModelRc::from(Rc::new(VecModel::from(next_keys))));
    ui.set_scale_from(from);
    ui.set_key_move(moves);
    let level = ui.get_interval_level();
    let levels: Vec<slint::SharedString> =
        interval_levels::labels(&interval_levels::Names {
            groups: &t.level_groups,
            any_chord_tone: t.level_any_chord_tone,
            any_scale_tone: t.level_any_scale_tone,
        })
        .into_iter()
        .map(Into::into)
        .collect();
    ui.set_interval_levels(ModelRc::from(Rc::new(VecModel::from(levels))));
    ui.set_interval_level(level);
    let channels = ui.get_audio_channels().row_count() as i32;
    let chosen = ui.get_audio_channel_index();
    ui.set_audio_channels(ModelRc::from(Rc::new(VecModel::from(channel_choices(
        channels, t.audio_one,
    )))));
    ui.set_audio_channel_index(chosen);
    let g = ui.global::<Tr>();
    g.set_chords(t.chords.into());
    g.set_intervals(t.intervals.into());
    g.set_scales(t.scales.into());
    g.set_arpeggios(t.arpeggios.into());
    g.set_settings(t.settings.into());
    g.set_close(t.close.into());
    g.set_settings_title(t.settings_title.into());
    g.set_heard(t.heard.into());
    g.set_input_card(t.input_card.into());
    g.set_input_hearing(t.input_hearing.into());
    g.set_input_silent(t.input_silent.into());
    g.set_gate_mark(t.gate_mark.into());
    g.set_show_input_meter(t.show_input_meter.into());
    g.set_show_heard(t.show_heard.into());
    g.set_tile_done(t.tile_done.into());
    g.set_tile_now(t.tile_now.into());
    g.set_tile_next(t.tile_next.into());
    g.set_tile_open(t.tile_open.into());
    g.set_chip_in_order(t.chip_in_order.into());
    g.set_chip_single(t.chip_single.into());
    g.set_chip_struck(t.chip_struck.into());
    g.set_chip_lock(t.chip_lock.into());
    g.set_chip_attack(t.chip_attack.into());
    g.set_quick_full(t.quick_full.into());
    g.set_quick_shells(t.quick_shells.into());
    g.set_nav_settings(t.nav_settings.into());
    g.set_tab_audio(t.tab_audio.into());
    g.set_tab_practice(t.tab_practice.into());
    g.set_tab_app(t.tab_app.into());
    g.set_audio_calibration(t.audio_calibration.into());
    g.set_audio_device(t.audio_device.into());
    g.set_audio_channel(t.audio_channel.into());
    g.set_audio_default(t.audio_default.into());
    g.set_audio_one(t.audio_one.into());
    g.set_noise_gate(t.noise_gate.into());
    g.set_gate_hint(t.gate_hint.into());
    g.set_bass_boost(t.bass_boost.into());
    g.set_arp_exercise(t.arp_exercise.into());
    g.set_arp_study(t.arp_study.into());
    g.set_arp_changes(t.arp_changes.into());
    g.set_arp_quality(t.arp_quality.into());
    g.set_arp_direction(t.arp_direction.into());
    g.set_arp_up(t.arp_up.into());
    g.set_arp_down(t.arp_down.into());
    g.set_arp_alt_down(t.arp_alt_down.into());
    g.set_arp_alt_up(t.arp_alt_up.into());
    g.set_tab_frets(t.tab_frets.into());
    g.set_tab_frets_hint(t.tab_frets_hint.into());
    g.set_voice_leading(t.voice_leading.into());
    g.set_voice_leading_hint(t.voice_leading_hint.into());
    g.set_lead_line(t.lead_line.into());
    g.set_lead_line_hint(t.lead_line_hint.into());
    g.set_preview(t.preview.into());
    g.set_preview_hint(t.preview_hint.into());
    g.set_preview_names(t.preview_names.into());
    g.set_preview_tab(t.preview_tab.into());
    g.set_preview_neck(t.preview_neck.into());
    g.set_preview_note_names(t.preview_note_names.into());
    g.set_preview_functions(t.preview_functions.into());
    g.set_preview_note_names_neck(t.preview_note_names_neck.into());
    g.set_preview_functions_neck(t.preview_functions_neck.into());
    g.set_preview_tiles(t.preview_tiles.into());
    g.set_debug_console(t.debug_console.into());
    g.set_shuffle_chords(t.shuffle_chords.into());
    g.set_shuffle_chords_hint(t.shuffle_chords_hint.into());
    g.set_random_order(t.random_order.into());
    g.set_show_diagrams(t.show_diagrams.into());
    g.set_repeat_root(t.repeat_root.into());
    g.set_chip_up(t.chip_up.into());
    g.set_chip_down(t.chip_down.into());
    g.set_chip_drawn(t.chip_drawn.into());
    g.set_chip_drawn_hint(t.chip_drawn_hint.into());
    g.set_note_hint_hover(t.note_hint_hover.into());
    g.set_chip_hint(t.chip_hint.into());
    g.set_chip_e_to_e(t.scale_from_e_to_e.into());
    g.set_random_key(t.random_key.into());
    g.set_there_and_back(t.there_and_back.into());
    g.set_scale_from(t.scale_from.into());
    g.set_key_move(t.key_move.into());
    g.set_repeat_root_hint(t.repeat_root_hint.into());
    g.set_shapes_kind(t.shapes_kind.into());
    g.set_shapes_no_shell(t.shapes_no_shell.into());
    g.set_shapes_no_shell_end(t.shapes_no_shell_end.into());
    g.set_shapes_substitute(t.shapes_substitute.into());
    g.set_shapes_full(t.shapes_full.into());
    g.set_shapes_shell(t.shapes_shell.into());
    g.set_random_hint(t.random_hint.into());
    g.set_fretboard(t.fretboard.into());
    g.set_formulas(t.formulas.into());
    g.set_formula_key_line(t.formula_key_line.into());
    g.set_formula_similar(t.formula_similar.into());
    g.set_formula_notes(t.formula_notes.into());
    g.set_formula_key(t.formula_key.into());
    g.set_formula_random(t.formula_random.into());
    g.set_formula_required(t.formula_required.into());
    g.set_formula_required_hint(t.formula_required_hint.into());
    g.set_formula_note_names(t.formula_note_names.into());
    g.set_formula_similar_opt(t.formula_similar_opt.into());
    g.set_formula_another(t.formula_another.into());
    g.set_formula_free(t.formula_free.into());
    g.set_fav_name(t.fav_name.into());
    g.set_fav_pick(t.fav_pick.into());
    g.set_fav_search(t.fav_search.into());
    g.set_exercise(t.exercise.into());
    g.set_exercise_key(t.exercise_key.into());
    g.set_exercise_chord(t.exercise_chord.into());
    g.set_exercise_changes(t.exercise_changes.into());
    g.set_placement(t.placement.into());
    g.set_placement_from(t.placement_from.into());
    g.set_next_chord(t.next_chord.into());
    g.set_placement_any(t.placement_any.into());
    g.set_placement_defines(t.placement_defines.into());
    g.set_placement_colours(t.placement_colours.into());
    g.set_placement_outside(t.placement_outside.into());
    g.set_against_chord(t.against_chord.into());
    g.set_jazz_names(t.jazz_names.into());
    g.set_jazz_names_hint(t.jazz_names_hint.into());
    g.set_chord_tones(t.chord_tones.into());
    g.set_fav_hint(t.fav_hint.into());
    g.set_fav_add(t.fav_add.into());
    g.set_fav_remove(t.fav_remove.into());
    g.set_formula_chords(t.formula_chords.into());
    g.set_formula_chords_opt(t.formula_chords_opt.into());
    g.set_startup_mode(t.startup_mode.into());
    g.set_chord_confidence(t.chord_confidence.into());
    g.set_note_threshold(t.note_threshold.into());
    g.set_hold_time(t.hold_time.into());
    g.set_show_debug(t.show_debug.into());
    g.set_show_spectrum(t.show_spectrum.into());
    g.set_ai_prediction(t.ai_prediction.into());
    g.set_intervals_label(t.intervals_label.into());
    g.set_interval_level(t.interval_level.into());
    g.set_intervals_hint(t.intervals_hint.into());
    g.set_intervals_placeholder(t.intervals_placeholder.into());
    g.set_next(t.next.into());
    g.set_previous(t.previous.into());
    g.set_no_data(t.no_data.into());
    g.set_language(t.language.into());
    g.set_lang_auto(t.lang_auto.into());
}

/// Diagnostic output (SOLITITO_DEBUG=1).
///
/// Intervals are printed RELATIVE TO THE DETECTED ROOT, which is how one thinks
/// about a chord: "Gm instead of Gm7" means b7 should show a high number. If it
/// is low, hearing is the bottleneck; if it is high and the quality still comes
/// out "m", the quality head is not using information it already has.
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
