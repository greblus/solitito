#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod model;
mod arpeggio;
mod audio;
mod brain;
mod diagrams;
mod fretboard;
mod i18n;
mod latch;
mod rng;
mod settings;
mod state;

use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::time::Duration;
use std::thread;

use audio::{AudioAnalysis, InputInfo, start_audio_stream, start_file_playback};
use brain::{ChordBrain, Prediction};
use latch::ChordLatch;
use i18n::Lang;
use settings::Settings;
use state::{MyApp, AppMode, MatchStatus};

use slint::{Timer, TimerMode, Model, ModelRc, VecModel, Color, SharedString, PhysicalPosition, PhysicalSize};

slint::include_modules!();

// Compiled in for the same reason as the chord shapes: no loose asset files to
// lose in a release package. Black on transparent, recoloured by Slint.
const ICON_SHUFFLE: &str = include_str!("icons/shuffle.svg");
const ICON_GEAR: &str = include_str!("icons/gear.svg");
const ICON_PAUSE: &str = include_str!("icons/pause.svg");
const ICON_PLAY: &str = include_str!("icons/play.svg");

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

/// Opens the input and reports what actually came up.
///
/// The report matters as much as the stream: on Windows the console is hidden,
/// so a stream that failed to open was indistinguishable from one that opened
/// and heard nothing - the interface's own meters kept showing signal either
/// way. Now the settings panel says which device, rate, channel count and
/// sample format are in use, or why there is none.
fn open_input(
    state: &Arc<Mutex<AudioAnalysis>>,
    holder: &Rc<RefCell<Option<cpal::Stream>>>,
    opened: &Rc<RefCell<InputInfo>>,
    ui: &AppWindow,
    device: Option<&str>,
    channel: usize,
) {
    // Dropped before opening the next: some backends refuse a second stream on
    // a device that already has one.
    holder.borrow_mut().take();
    match start_audio_stream(state.clone(), device, channel) {
        Ok((stream, info)) => {
            let line = format!(
                "{} · {} Hz · {} ch · {}",
                info.name, info.sample_rate, info.channels, info.format
            );
            println!("🎧 {line}");
            ui.set_audio_info(line.into());
            ui.set_audio_channel_count(info.channels as i32);
            *opened.borrow_mut() = info;
            *holder.borrow_mut() = Some(stream);
        }
        Err(e) => {
            eprintln!("ERR AUDIO IN: {e}");
            ui.set_audio_info(format!("✖ {e}").into());
            ui.set_audio_channel_count(0);
            *opened.borrow_mut() = InputInfo::default();
        }
    }
}

/// Puts "system default" at the head of the device list, so index 0 always
/// means "whatever the OS picks" whether or not any devices were found.
fn fill_device_list(ui: &AppWindow, names: &[String], default_label: &str) {
    let mut items: Vec<SharedString> = vec![SharedString::from(default_label)];
    // Shortened for display only. The full names stay in `device_names`, which
    // is what gets matched and saved - a truncated one would find nothing on
    // the next launch.
    items.extend(names.iter().map(|n| SharedString::from(short_device_name(n))));
    ui.set_audio_devices(ModelRc::from(Rc::new(VecModel::from(items))));
}

/// The channel picker's entries for a device with `count` inputs, numbered from
/// one so they match what is printed on the box.
fn channel_choices(count: i32, one_label: &str) -> Vec<SharedString> {
    (1..=count.max(1))
        .map(|c| SharedString::from(format!("{one_label} {c}")))
        .collect()
}

/// A device name short enough for the picker, without losing what tells two of
/// them apart.
///
/// Backend names are not written for people - ALSA reports the likes of
/// `alsa_input.usb-BEHRINGER_UMC204HD_192k-00.analog-stereo`. The prefix says
/// nothing, and what separates two sockets of the same interface sits at the
/// END, so the middle is what goes.
fn short_device_name(name: &str) -> String {
    const MAX: usize = 36;
    let n = name
        .trim_start_matches("alsa_input.")
        .trim_start_matches("alsa_output.");
    let chars: Vec<char> = n.chars().collect();
    if chars.len() <= MAX {
        return n.to_string();
    }
    let head: String = chars[..MAX / 2 - 1].iter().collect();
    let tail: String = chars[chars.len() - (MAX / 2 - 1)..].iter().collect();
    format!("{head}…{tail}")
}

#[cfg(test)]
mod ui_tests {
    use super::*;

    /// A name has to stay recognisable and, more to the point, has to stay
    /// DIFFERENT from the other sockets on the same interface - which is what a
    /// plain truncation would destroy, since they differ only at the end.
    #[test]
    fn shortening_keeps_devices_apart() {
        let a = "alsa_input.usb-BEHRINGER_UMC204HD_192k-00.analog-stereo-input-1";
        let b = "alsa_input.usb-BEHRINGER_UMC204HD_192k-00.analog-stereo-input-2";
        let (sa, sb) = (short_device_name(a), short_device_name(b));
        assert_ne!(sa, sb, "two inputs of one interface collapsed to the same label");
        assert!(sa.chars().count() <= 36, "still too long: {sa}");
        assert!(!sa.starts_with("alsa_input."), "the prefix that says nothing survived");
    }

    /// Short names are left exactly as they are - Windows reports readable ones.
    #[test]
    fn a_short_name_is_left_alone() {
        let n = "Line In (BEHRINGER UMC204HD)";
        assert_eq!(short_device_name(n), n);
    }

    /// One entry per input, numbered from one, and never an empty picker.
    #[test]
    fn channels_are_numbered_from_one() {
        let c = channel_choices(2, "Channel");
        assert_eq!(c, vec!["Channel 1", "Channel 2"]);
        assert_eq!(channel_choices(0, "Channel").len(), 1, "a picker with nothing in it");
    }
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
    random_enabled: bool,
    show_diagrams: bool,
    ai_debug: bool,
    startup_mode: i32,
    language: i32,
    intervals: String,
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
            random_enabled: ui.get_random_enabled(),
            show_diagrams: ui.get_show_diagrams(),
            ai_debug: ui.get_ai_debug_visible(),
            startup_mode: ui.get_startup_mode(),
            language: ui.get_language_idx(),
            intervals: ui.get_interval_input_text().to_string(),
        }
    }
}

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
    
    audio::hush_alsa();

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
    // --devices: what the backend can see, and what each one reports. On a
    // machine with a sound server, `default`, `pulse` and `pipewire` are three
    // names for the same path - the server's mix, not the interface's sockets -
    // which is why picking a channel on them changes nothing.
    if args.iter().any(|a| a == "--devices") {
        audio::hush_alsa();
        for name in audio::list_input_devices() {
            match audio::probe_input(&name) {
                Ok(i) => println!("  {name}\n      {} Hz · {} ch · {}", i.sample_rate, i.channels, i.format),
                Err(e) => println!("  {name}\n      unavailable: {e}"),
            }
        }
        std::process::exit(0);
    }

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
    // Held so it can be swapped when the device or channel changes. Dropping the
    // old stream first is deliberate: some backends refuse a second stream on a
    // device that already has one open.
    let audio_in: Rc<RefCell<Option<cpal::Stream>>> = Rc::new(RefCell::new(None));
    let opened: Rc<RefCell<InputInfo>> = Rc::new(RefCell::new(InputInfo::default()));

    if args.len() > 2 && args[1] == "--file" {
        let path = args[2].clone();
        println!("Starting FILE mode: {}", path);
        if let Err(e) = start_file_playback(path, analysis_state.clone()) {
            eprintln!("ERR FILE: {}", e);
            return Ok(());
        }
        file_mode = true;
    } else {
        println!("Starting LIVE mode...");
    }
    
    // AI THREAD
    let analysis_for_ai = analysis_state.clone();
    let result_for_ai = ai_result_state.clone();
    
    // Named so `top -H` can tell inference apart from the UI thread - both were
    // just "solitito", which made "is it the model or the drawing?" unanswerable
    // without a debugger.
    let _ai_thread = thread::Builder::new().name("solitito-ai".into()).spawn(move || {
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
        ui.set_short_verdict(cfg.short_verdict);
        ui.set_show_spectrum(cfg.show_spectrum);
        ui.set_icon_shuffle(svg_icon(ICON_SHUFFLE));
        ui.set_icon_gear(svg_icon(ICON_GEAR));
        ui.set_icon_pause(svg_icon(ICON_PAUSE));
        ui.set_icon_play(svg_icon(ICON_PLAY));
        apply_language(&ui, lang);
        ui.set_interval_input_text(app.intervals_input.clone().into()); 
    }

    ui.window().set_position(PhysicalPosition::new(450, 10));

    // --- AUDIO INPUT ---
    // The device list is NOT built here unless a saved device has to be found in
    // it. Enumerating makes ALSA probe every plugin it knows - OSS emulation,
    // dsnoop, route - and each failure prints to stderr, so a list nobody asked
    // for buries the app's own output. It is filled when the settings panel is
    // first opened instead; see `devices_listed` below.
    let device_names: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let devices_listed = Rc::new(std::cell::Cell::new(false));
    if !file_mode {
        let t = i18n::strings(lang);
        let mut chosen = 0;
        if cfg.audio_device.is_some() {
            let names = audio::list_input_devices();
            chosen = cfg
                .audio_device
                .as_deref()
                .and_then(|want| names.iter().position(|n| n == want))
                .map(|i| i as i32 + 1)
                .unwrap_or(0);
            fill_device_list(&ui, &names, t.audio_default);
            *device_names.borrow_mut() = names;
            devices_listed.set(true);
        } else {
            fill_device_list(&ui, &[], t.audio_default);
        }
        ui.set_audio_device_index(chosen);
        ui.set_audio_channel_index(cfg.audio_channel.max(1) as i32 - 1);

        // A saved device that is no longer plugged in falls back to the default
        // rather than refusing to open anything.
        let want = (chosen > 0)
            .then(|| device_names.borrow()[chosen as usize - 1].clone());
        open_input(&analysis_state, &audio_in, &opened, &ui, want.as_deref(), cfg.audio_channel);
        ui.set_audio_channels(ModelRc::from(Rc::new(VecModel::from(channel_choices(
            ui.get_audio_channel_count(),
            t.audio_one,
        )))));
    }

    // The single live copy of the settings. Each closure below used to write a
    // clone taken at startup, so saving one setting rewrote the others with
    // their values from launch time - changing the mode and then the language
    // silently reverted the mode. They now all edit this.
    let live_cfg = Rc::new(RefCell::new(cfg.clone()));

    // Restoring the size has to wait for the event loop. The window does not
    // exist until then - `show()` only queues its creation - and a size set
    // before it exists is dropped, leaving the window at its preferred size.
    // Verified on this machine: setting it any earlier had no effect at all.
    // Held for as long as the window lives, or the timer is dropped and never
    // fires. The closure only gets a weak handle, so it can stop itself.
    let _restore_size = Rc::new(Timer::default());
    if let (Some(w), Some(h)) = (cfg.window_w, cfg.window_h) {
        let uw = ui.as_weak();
        let timer = Rc::downgrade(&_restore_size);
        let give_up_at = std::time::Instant::now() + Duration::from_secs(3);
        // Asking once is not enough. Slint sizes the window from the layout's
        // preferred size during its first pass, which lands after the event loop
        // has already started and overwrites anything set before it - measured
        // here at somewhere between 300ms and 600ms after launch. So ask
        // repeatedly and stop as soon as it holds, rather than sleeping for a
        // guessed interval that a slower machine would miss.
        _restore_size.start(TimerMode::Repeated, Duration::from_millis(50), move || {
            let Some(ui) = uw.upgrade() else { return };
            let now = ui.window().size();
            // Two ways out, because a window manager is entitled to refuse: the
            // size took, or it has had long enough that it never will. Without
            // the deadline a tiling WM would leave this firing forever.
            if (now.width, now.height) == (w, h) || std::time::Instant::now() > give_up_at {
                if let Some(timer) = timer.upgrade() {
                    timer.stop();
                }
                return;
            }
            ui.window().set_size(PhysicalSize::new(w, h));
            fit_ui_to_window(&ui);
        });
    }
    fit_ui_to_window(&ui);
    {
        let uw = ui.as_weak();
        ui.on_window_resized(move || {
            if let Some(ui) = uw.upgrade() {
                fit_ui_to_window(&ui);
            }
        });
    }

    // On close rather than on every resize: dragging an edge produces a stream
    // of sizes, and only the last one is the answer.
    {
        let uw = ui.as_weak();
        let cfg = live_cfg.clone();
        ui.window().on_close_requested(move || {
            if let Some(ui) = uw.upgrade() {
                let size = ui.window().size();
                let mut cfg = cfg.borrow_mut();
                cfg.window_w = Some(size.width);
                cfg.window_h = Some(size.height);
                cfg.save();
            }
            slint::CloseRequestResponse::HideWindow
        });
    }

    // Taken after the initial set_* calls, so persisted settings count as the
    // baseline rather than as a change. Retaken when the panel closes: the mark
    // is feedback that an edit registered, not a permanent badge.
    let mut baseline = SettingsSnapshot::read(&ui);
    let mut settings_were_open = false;
    let cfg_tick = live_cfg.clone();
    let names_tick = device_names.clone();
    let listed_tick = devices_listed.clone();

    // Built once and then written into, NOT replaced every frame. Assigning a
    // fresh model makes Slint throw away all 48 bars and build them again, and
    // at 60 frames a second that was most of a core - visible only with the
    // settings panel open, because the spectrum lives inside it.
    let spectrum_data: Rc<VecModel<f32>> = Rc::new(VecModel::from(vec![0.0_f32; 48]));
    let spectrum_colors: Rc<VecModel<Color>> =
        Rc::new(VecModel::from(vec![Color::from_rgb_u8(30, 30, 30); 48]));
    ui.set_spectrum_data(ModelRc::from(spectrum_data.clone()));
    ui.set_spectrum_colors(ModelRc::from(spectrum_colors.clone()));

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
    // Where the interval strip stood last frame. A step BACKWARDS means the
    // exercise restarted - new chord, new key - so the strip snaps back to the
    // first page instead of turning over at the normal pace.
    let mut last_interval_step: i32 = 0;
    // Pattern length as well as position: entering a mode or changing chord
    // swaps the whole strip, and that is a restart too - without this the first
    // slide into place would crawl at the page-turn speed.
    let mut last_interval_len: i32 = 0;
    // Rasterising the SVGs is not free and the shapes only change when the chord
    // QUALITY does, which is far less often than every frame.
    let mut last_diagram_key = String::new();
    // Live readouts are what drive the redraw rate, and the panel is expensive to
    // redraw. Neither of these carries information at sixty frames a second: the
    // confidence figure jitters by a point between frames, and the meter is
    // smoothed so it never settles exactly. Both are therefore held back unless
    // they have something new to say.
    let mut last_ai_text = String::new();
    let mut last_ai_name = String::new();
    let mut last_ai_push = std::time::Instant::now();
    let mut last_level_db = f32::NEG_INFINITY;
    let mut last_spectrum_push = std::time::Instant::now();

    timer.start(TimerMode::Repeated, Duration::from_millis(16), move || {
        let ui = ui_weak.unwrap();
        let mut app = app_clone.lock().unwrap();

        let (spectrum_vis, in_level) = {
            let s = app.analysis_state.lock().unwrap();
            (s.spectrum_visual, s.input_level)
        };
        // Half a decibel is below what the bar can show anyway.
        let level_db = lin_to_db(in_level);
        if (level_db - last_level_db).abs() > 0.5 {
            last_level_db = level_db;
            ui.set_input_level_db(level_db);
        }

        if !file_mode {
            app.noise_gate = db_to_lin(ui.get_gate_db());
        }
        app.bass_boost_enabled = ui.get_boost_enabled();
        app.bass_boost_gain = ui.get_boost_gain();
        app.chord_confidence = ui.get_chord_confidence();
        app.note_threshold = ui.get_note_threshold();
        app.transition_delay = ui.get_delay();
        app.set_random_mode(ui.get_random_enabled());
        app.short_verdict = ui.get_short_verdict();
        // The fretboard trainer hides the pause button, so a pause carried over
        // from another mode would freeze it with nothing on screen to explain why.
        if app.app_mode == AppMode::Fretboard && ui.get_paused() {
            ui.set_paused(false);
        }
        app.paused = ui.get_paused();
        let settings_open = ui.get_show_settings();
        // The list is built the first time the panel is opened, not at startup:
        // enumerating sets ALSA probing every plugin it knows and printing a
        // failure for each, which is a poor greeting for someone who never
        // changes the input.
        if settings_open && !listed_tick.get() && !file_mode {
            let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
            let names = audio::list_input_devices();
            fill_device_list(&ui, &names, t.audio_default);
            let saved = cfg_tick.borrow().audio_device.clone();
            ui.set_audio_device_index(
                saved
                    .and_then(|want| names.iter().position(|n| *n == want))
                    .map(|i| i as i32 + 1)
                    .unwrap_or(0),
            );
            *names_tick.borrow_mut() = names;
            listed_tick.set(true);
        }
        if settings_were_open && !settings_open {
            baseline = SettingsSnapshot::read(&ui);   // leaving the panel clears the mark
        }
        settings_were_open = settings_open;
        set_if_changed(
            ui.get_settings_touched(),
            settings_open && SettingsSnapshot::read(&ui) != baseline,
            |v| ui.set_settings_touched(v),
        );
        
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

                // The chord name appears at once - that is what is being read.
                // A changed percentage on the same chord waits its turn, at five
                // updates a second rather than sixty.
                let text = format!("{} ({:.0}%)", shown, current_confidence * 100.0);
                let name_changed = shown != last_ai_name;
                if (name_changed || last_ai_push.elapsed() >= Duration::from_millis(200))
                    && text != last_ai_text
                {
                    ui.set_ai_text(text.clone().into());
                    last_ai_text = text;
                    last_ai_name = shown.clone();
                    last_ai_push = std::time::Instant::now();
                }

                // Real elapsed time, not assumed. Capped so a thread stall does not
                // jump the whole threshold at once.
                let now = std::time::Instant::now();
                let dt = last_ai_at
                    .map(|t| now.duration_since(t).as_secs_f32())
                    .unwrap_or(0.040)
                    .min(0.25);
                last_ai_at = Some(now);
                let before = app.current_chord_index;
                app.check_progress_with_ai(dt, &shown, current_confidence);
                // SOLITITO_STRUM=1: one line per AI frame, to see whether attacks
                // are being detected at all during real playing. The per-strum
                // verdict can only re-arm on a new onset id.
                if std::env::var("SOLITITO_STRUM").is_ok() {
                    println!(
                        "atak#{onset_id} klatek_od_ataku={since_onset:<3} slyszy={shown:<8}                          pewnosc={:.2} cel={:<8} struna={:<3} licznik={:.2}/{:.2} {}",
                        current_confidence,
                        app.chords.get(app.current_chord_index)
                            .map(|c| format!("{}{}", c.root.to_string(), c.quality.to_string()))
                            .unwrap_or_default(),
                        app.start_hint
                            .and_then(|i| state::START_STRINGS.get(i))
                            .copied()
                            .unwrap_or("-"),
                        app.success_timer,
                        app.transition_delay,
                        if app.current_chord_index != before { "-> PRZESZLO" } else { "" },
                    );
                }
            }
        }

        set_if_changed(ui.get_song_title(), app.song_title.clone().into(), |v| ui.set_song_title(v));
        
        // Scales in random mode redraw the key on their own; push it to the
        // combo, but only on a real change so it does not fight the user
        // mid-selection (the binding is two-way).
        if app.app_mode == AppMode::Scales
            && ui.get_current_secondary_index() != app.secondary_index as i32
        {
            set_if_changed(ui.get_current_secondary_index(), app.secondary_index as i32, |v| ui.set_current_secondary_index(v));
        }

        if app.app_mode == AppMode::Fretboard {
            // Minimal by design: the note, and under it where to play it.
            let name = match app.fret_target {
                Some(pc) => model::NoteName::from_index(pc).to_string().to_string(),
                None => "...".to_string(),
            };
            set_if_changed(ui.get_chord_name(), name.into(), |v| ui.set_chord_name(v));
            set_if_changed(ui.get_start_hint(), app.region.describe().into(), |v| ui.set_start_hint(v));
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
            set_if_changed(ui.get_start_hint(), hint.into(), |v| ui.set_start_hint(v));

            // Shapes depend on the quality alone - the diagram is movable, the
            // root only decides which fret to put it on.
            let q_key = curr_chord.quality.to_string();
            if q_key != last_diagram_key {
                last_diagram_key = q_key;
                let imgs: Vec<slint::Image> = diagrams::for_quality(&curr_chord.quality)
                    .iter()
                    .filter_map(|d| slint::Image::load_from_svg_data(d.svg.as_bytes()).ok())
                    .collect();
                ui.set_chord_diagrams(ModelRc::from(Rc::new(VecModel::from(imgs))));
                // A shape left open would belong to the previous chord.
                ui.set_diagram_zoom(-1);
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
            
            let next_idx = app.next_chord_index();
            set_if_changed(ui.get_next_chord(), app.chord_label(next_idx).into(), |v| ui.set_next_chord(v));
            set_if_changed(
                ui.get_prev_chord(),
                app.prev_chord_index.map(|i| app.chord_label(i)).unwrap_or_default().into(),
                |v| ui.set_prev_chord(v),
            );
            // Same palette as the big name, so one colour means one thing
            // wherever it appears. Dim grey for a chord stepped over rather
            // than played.
            let prev_col = slint::Brush::SolidColor(match app.prev_status() {
                MatchStatus::Exact => Color::from_rgb_u8(50, 255, 50),
                MatchStatus::Partial => Color::from_rgb_u8(255, 220, 50),
                MatchStatus::Flicker => Color::from_rgb_u8(255, 50, 50),
                // Stepped over, not played: as quiet as the chord ahead.
                MatchStatus::None => Color::from_rgb_u8(138, 138, 138),
            });
            set_if_changed(ui.get_prev_color(), prev_col, |v| ui.set_prev_color(v));

            if app.app_mode != AppMode::Chords {
                let all_names = curr_chord.quality.interval_names();
                let active_indices = app.ordered_active_indices(curr_chord);
                let mut ui_names = Vec::new();
                let mut ui_colors = Vec::new();
                for (step_idx, step) in active_indices.iter().enumerate() {
                    if step.degree < all_names.len() {
                        // The octave marker is display only - the model has no
                        // octave, so 1 and 1' are checked identically.
                        let name = model::with_octave(&all_names[step.degree], step.octave);
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
                // Order matters: the duration binding reads interval_jump when x
                // is recomputed, so the flag has to be in place first.
                let step = app.current_note_step as i32;
                let len = ui_names.len() as i32;
                let restarted = step < last_interval_step || len != last_interval_len;
                set_if_changed(ui.get_interval_jump(), restarted, |v| ui.set_interval_jump(v));
                last_interval_step = step;
                last_interval_len = len;
                set_if_changed(ui.get_interval_step(), step, |v| ui.set_interval_step(v));
                ui.set_interval_names(ModelRc::from(Rc::new(VecModel::from(ui_names))));
                ui.set_interval_colors(ModelRc::from(Rc::new(VecModel::from(ui_colors))));
            }
            
        }

        // Outside the branch above on purpose. It used to sit inside the "chord
        // mode with a song loaded" arm, so the fretboard trainer - which carries
        // no chords at all - never pushed a frame and the spectrum froze while
        // the model kept predicting. That was the "sometimes it does not draw".
        let targets: Vec<usize> = match app.app_mode {
            AppMode::Fretboard => app.fret_target.into_iter().collect(),
            _ => app.chords.get(app.current_chord_index)
                    .map(|c| c.get_target_indices())
                    .unwrap_or_default(),
        };
        let spec_vec: Vec<f32> = spectrum_vis.to_vec();
        let mut spec_colors = Vec::new();
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
        // Skipped entirely when nothing shows it: the bars are the panel's most
        // expensive content, and off by default.
        if !ui.get_show_spectrum() && !ui.get_ai_debug_visible() {
            return;
        }
        // And rationed even when shown. With signal every one of the 48 bars
        // moves on every frame, so this alone would keep the whole panel
        // redrawing at sixty frames a second; twenty is more than the eye reads
        // off a level display.
        if last_spectrum_push.elapsed() < Duration::from_millis(50) {
            return;
        }
        last_spectrum_push = std::time::Instant::now();
        // Only rows that actually moved: setting a row notifies whether or not
        // the value changed, and in silence most of them do not.
        for (i, v) in spec_vec.iter().enumerate() {
            if spectrum_data.row_data(i) != Some(*v) {
                spectrum_data.set_row_data(i, *v);
            }
        }
        for (i, c) in spec_colors.iter().enumerate() {
            if spectrum_colors.row_data(i) != Some(*c) {
                spectrum_colors.set_row_data(i, *c);
            }
        }
    });

    let app_weak = my_app.clone();
    let ui_weak_cb = ui.as_weak();
    let keys_list_clone = keys_list.clone();
    
    // Saved immediately - there is no "Save" button, so the setting has to survive
    // closing the window without an extra step.
    {
        let cur = live_cfg.clone();
        let state = analysis_state.clone();
        let holder = audio_in.clone();
        let info = opened.clone();
        let names_cb = device_names.clone();
        let uw = ui.as_weak();
        ui.on_audio_device_changed(move |idx| {
            let Some(ui) = uw.upgrade() else { return };
            // From the list already on screen: enumerating again would set ALSA
            // probing every plugin a second time, for names the user is looking
            // at right now.
            let name = if idx > 0 { names_cb.borrow().get(idx as usize - 1).cloned() } else { None };
            let channel = {
                let mut c = cur.borrow_mut();
                c.audio_device = name.clone();
                c.save();
                c.audio_channel
            };
            open_input(&state, &holder, &info, &ui, name.as_deref(), channel);

            // A different device may have a different number of channels, and a
            // choice that no longer exists has to fall back to mixing.
            let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
            let count = ui.get_audio_channel_count();
            ui.set_audio_channels(ModelRc::from(Rc::new(VecModel::from(channel_choices(
                count, t.audio_one,
            )))));
            // A device with fewer inputs cannot honour a channel picked on a
            // bigger one; fall back to the first rather than to silence.
            if ui.get_audio_channel_index() >= count.max(1) {
                ui.set_audio_channel_index(0);
                let mut c = cur.borrow_mut();
                c.audio_channel = 1;
                c.save();
            }
        });
    }
    {
        let cur = live_cfg.clone();
        let state = analysis_state.clone();
        let holder = audio_in.clone();
        let info = opened.clone();
        let uw = ui.as_weak();
        ui.on_audio_channel_changed(move |idx| {
            let Some(ui) = uw.upgrade() else { return };
            let channel = idx.max(0) as usize + 1;
            let device = {
                let mut c = cur.borrow_mut();
                c.audio_channel = channel;
                c.save();
                c.audio_device.clone()
            };
            open_input(&state, &holder, &info, &ui, device.as_deref(), channel);
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_show_spectrum_changed(move |on| {
            let mut cur = cur.borrow_mut();
            cur.show_spectrum = on;
            cur.save();
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_short_verdict_changed(move |on| {
            let mut cur = cur.borrow_mut();
            cur.short_verdict = on;
            cur.save();
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_startup_mode_changed(move |mode_idx| {
            let mut cur = cur.borrow_mut();
            cur.startup_mode = mode_idx;
            cur.save();
        });
    }
    {
        let cur = live_cfg.clone();
        let uw = ui.as_weak();
        ui.on_language_changed(move |idx| {
            {
                let mut cur = cur.borrow_mut();
                cur.language = idx;
                cur.save();
            }
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
        let t = i18n::strings(lang);
        let (label, items, sec_label, sec_items) = match app.app_mode {
            AppMode::Scales => (
                t.select_scale, app.scale_definitions.iter().map(|s| SharedString::from(&s.name)).collect::<Vec<SharedString>>(),
                t.key_root, keys_list_clone.clone()
            ),
            AppMode::Arpeggios => (
                t.select_song, app.song_library.iter().map(|s| SharedString::from(&s.title)).collect::<Vec<SharedString>>(),
                t.pattern, app.arpeggio_patterns.iter().map(|s| SharedString::from(&s.name)).collect::<Vec<SharedString>>()
            ),
            _ => (t.select_song, app.song_library.iter().map(|s| SharedString::from(&s.title)).collect::<Vec<SharedString>>(), "", vec![]),
        };
        ui.set_library_label(label.into());
        ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(items))));
        ui.set_current_item_index(0);
        ui.set_secondary_label(sec_label.into());
        ui.set_secondary_items(ModelRc::from(Rc::new(VecModel::from(sec_items))));
        // Not a hard 0: returning to Arpeggios restores the pattern that was
        // chosen, and the combo has to say so.
        ui.set_current_secondary_index(app.secondary_index as i32);
    });

    {
        let app_weak = my_app.clone();
        ui.on_step_chord(move |delta| {
            app_weak.lock().unwrap().step_chord(delta);
        });
    }

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
    g.set_lock_quality(t.lock_quality.into());
    g.set_short_verdict(t.short_verdict.into());
    g.set_short_verdict_hint(t.short_verdict_hint.into());
    g.set_random_order(t.random_order.into());
    g.set_show_diagrams(t.show_diagrams.into());
    g.set_random_hint(t.random_hint.into());
    g.set_fretboard(t.fretboard.into());
    g.set_startup_mode(t.startup_mode.into());
    g.set_chord_confidence(t.chord_confidence.into());
    g.set_note_threshold(t.note_threshold.into());
    g.set_hold_time(t.hold_time.into());
    g.set_show_debug(t.show_debug.into());
    g.set_show_spectrum(t.show_spectrum.into());
    g.set_ai_prediction(t.ai_prediction.into());
    g.set_intervals_label(t.intervals_label.into());
    g.set_intervals_hint(t.intervals_hint.into());
    g.set_intervals_placeholder(t.intervals_placeholder.into());
    g.set_next(t.next.into());
    g.set_previous(t.previous.into());
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



