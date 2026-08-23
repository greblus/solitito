#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod model;
mod arpeggio;
mod audio;
mod brain;
mod diagrams;
mod formulas;
mod fretboard;
mod i18n;
mod latch;
mod rng;
mod settings;
mod state;

use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::cell::{Cell, RefCell};
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
            let mut line = format!(
                "{} · {} Hz · {} ch · {}",
                info.name, info.sample_rate, info.channels, info.format
            );
            if info.fell_back {
                // Said in words, not just marked: the app still hears something,
                // so nothing looks wrong - except that the channel picker now
                // belongs to a device the user did not choose.
                let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
                line = format!("⚠ {} - {line}", t.audio_busy);
            }
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

/// Adds newly seen devices to the list already on screen, and says whether
/// anything was added.
///
/// The list only ever grows. ALSA hands out a card once, so anything holding
/// one - PipeWire, another app, or this app's own stream - drops it from the
/// enumeration entirely; a plain re-scan would leave the picker showing
/// `default`, `pulse`, `pipewire` and `jack` and nothing else, and would take
/// the user's own choice with it.
fn merge_devices(known: &mut Vec<String>, fresh: Vec<String>) -> bool {
    let before = known.len();
    for name in fresh {
        if !known.contains(&name) {
            known.push(name);
        }
    }
    known.len() != before
}

/// Puts "system default" at the head of the device list, so index 0 always
/// means "whatever the OS picks" whether or not any devices were found.
///
/// `present` is the latest scan. Anything in `names` that it does not contain is
/// remembered rather than seen - the saved device while the interface is still
/// in its bag, most often - and is marked, because an entry that reads like any
/// other is a promise the list cannot keep.
/// Is this entry a device the backend can currently see?
///
/// The device we are recording from counts as present even though no scan
/// reports it: a card can be opened once, and we are the ones holding it. Left
/// out, the picker marks the interface you are playing through as unavailable
/// the moment it starts working.
fn device_is_present(name: &str, scan: &[String], open_now: Option<&str>) -> bool {
    scan.iter().any(|n| n == name) || open_now == Some(name)
}

fn fill_device_list(
    ui: &AppWindow,
    names: &[String],
    default_label: &str,
    present: &[String],
    open_now: Option<&str>,
) {
    let mut items: Vec<SharedString> = vec![SharedString::from(default_label)];
    // Shortened for display only. The full names stay in `device_names`, which
    // is what gets matched and saved - a truncated one would find nothing on
    // the next launch.
    items.extend(names.iter().map(|n| {
        let short = short_device_name(n);
        if device_is_present(n, present, open_now) {
            SharedString::from(short)
        } else {
            SharedString::from(format!("⚠ {short}"))
        }
    }));
    ui.set_audio_devices(ModelRc::from(Rc::new(VecModel::from(items))));
}

/// Re-scans the inputs and refreshes the picker, keeping the current choice.
///
/// Cheap enough to do whenever the list is about to be read, which is the point:
/// an interface plugged in after the app started has to appear without a
/// restart, and the one that was unplugged has to stop looking available.
fn rescan_devices(
    ui: &AppWindow,
    known: &Rc<RefCell<Vec<String>>>,
    default_label: &str,
    open_now: Option<&str>,
) {
    let present = audio::list_input_devices();
    let mut names = known.borrow_mut();
    merge_devices(&mut names, present.clone());
    // Merging only appends, so no index shifts under the user - but replacing
    // the model does reset the picker, hence putting it back.
    let chosen = ui.get_audio_device_index();
    fill_device_list(ui, &names, default_label, &present, open_now);
    ui.set_audio_device_index(chosen);
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

    /// The interface you are playing through is missing from every scan for the
    /// same reason it is working: we hold the card. Marking it unavailable was
    /// the bug - the label appeared the moment the device started being used.
    #[test]
    fn the_device_we_are_recording_from_counts_as_present() {
        let scan = vec!["default".to_string(), "pipewire".to_string()];
        assert!(device_is_present("default", &scan, None));
        assert!(
            !device_is_present("sysdefault:CARD=U192k", &scan, None),
            "a card nobody can see is not present"
        );
        assert!(
            device_is_present("sysdefault:CARD=U192k", &scan, Some("sysdefault:CARD=U192k")),
            "the card we are recording from was marked unavailable"
        );
        // Holding one card says nothing about another.
        assert!(!device_is_present("hw:CARD=X", &scan, Some("sysdefault:CARD=U192k")));
    }

    /// A re-scan must never take a device off the list. ALSA leaves out
    /// everything that is currently held - including the card this app is
    /// recording from - so a scan that replaced the list would drop the user's
    /// own device the moment it started working.
    #[test]
    fn a_rescan_only_ever_adds() {
        let mut known = vec!["default".to_string(), "sysdefault:CARD=U192k".to_string()];
        // The card is busy now, so the scan cannot see it.
        assert!(!merge_devices(&mut known, vec!["default".into()]), "nothing new to report");
        assert_eq!(known.len(), 2, "the busy card was dropped from the picker");
        // A newly plugged-in interface shows up without a restart.
        assert!(merge_devices(&mut known, vec!["default".into(), "hw:CARD=X".into()]));
        assert_eq!(known, ["default", "sysdefault:CARD=U192k", "hw:CARD=X"]);
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

/// Gives the diagnostic flags somewhere to print on Windows.
///
/// A release build declares `windows_subsystem = "windows"`, so it starts with
/// no console and no standard handles - `println!` goes nowhere. That is right
/// for the app and wrong for `--devices`, `--check` and `--bench`, which exist
/// to answer questions about a Windows machine and had nothing to answer with.
///
/// Attaches to the console that launched it; failing that (a double-click) it
/// opens one of its own. Standard handles are only replaced when they are not
/// already valid, so `solitito.exe --bench > out.txt` still redirects.
#[cfg(windows)]
fn attach_console() -> bool {
    use std::os::raw::c_void;

    const ATTACH_PARENT_PROCESS: u32 = 0xFFFF_FFFF;
    const STD_INPUT_HANDLE: u32 = 0xFFFF_FFF6;   // -10
    const STD_OUTPUT_HANDLE: u32 = 0xFFFF_FFF5;  // -11
    const STD_ERROR_HANDLE: u32 = 0xFFFF_FFF4;   // -12
    const GENERIC_READ: u32 = 0x8000_0000;
    const GENERIC_WRITE: u32 = 0x4000_0000;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;
    const OPEN_EXISTING: u32 = 3;

    #[allow(non_snake_case)]
    #[link(name = "kernel32")]
    extern "system" {
        fn AttachConsole(process_id: u32) -> i32;
        fn AllocConsole() -> i32;
        fn GetStdHandle(which: u32) -> *mut c_void;
        fn SetStdHandle(which: u32, handle: *mut c_void) -> i32;
        fn CreateFileA(
            name: *const u8,
            access: u32,
            share: u32,
            security: *mut c_void,
            disposition: u32,
            flags: u32,
            template: *mut c_void,
        ) -> *mut c_void;
    }

    let invalid = |h: *mut c_void| h.is_null() || h as isize == -1;

    unsafe {
        // Its own window only when there is no console to borrow.
        let allocated = if AttachConsole(ATTACH_PARENT_PROCESS) != 0 {
            false
        } else {
            AllocConsole() != 0
        };

        // CONOUT$/CONIN$ name the attached console whatever it is called.
        let wanted: [(&[u8], u32); 3] = [
            (b"CONOUT$\0", STD_OUTPUT_HANDLE),
            (b"CONOUT$\0", STD_ERROR_HANDLE),
            (b"CONIN$\0", STD_INPUT_HANDLE),
        ];
        for (device, which) in wanted {
            // A valid handle here means the shell redirected it, which is not
            // ours to undo.
            if !invalid(GetStdHandle(which)) {
                continue;
            }
            let h = CreateFileA(
                device.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null_mut(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            );
            if !invalid(h) {
                SetStdHandle(which, h);
            }
        }

        allocated
    }
}

#[cfg(not(windows))]
fn attach_console() -> bool {
    false
}

/// Holds a console that this app opened itself open long enough to be read -
/// otherwise a double-clicked `--bench` reports into a window that closes with
/// it. A borrowed console needs none of this; the shell prompt comes back.
fn keep_console_open(allocated: bool) {
    if !allocated {
        return;
    }
    println!("\n[Enter]");
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
}

/// Runs a recording through the live feature path and reports, window by
/// window, what the model makes of it.
///
/// Deliberately ungated: the app only asks the model once 90% of the 48-frame
/// window carries signal, so "the model cannot hear it" and "the app never
/// asked" look identical from the outside. Here the level, the window fill and
/// the twelve pitch probabilities are printed side by side, which separates the
/// two.
/// Everything the binary can be asked to do, in one place.
///
/// Worth having as a flag rather than only in the README: the reporting modes
/// exist to answer questions about a machine that is not the developer's, and
/// whoever is sitting at it has the binary to hand and the README somewhere else.
fn help_text() -> String {
    format!(
        "Solitito {} - real-time polyphonic guitar trainer\n\
         \n\
         Usage: solitito [OPTION]...\n\
         With no options the trainer window opens.\n\
         \n\
         Reporting - each prints and exits:\n\
         \x20 -h, --help            this text\n\
         \x20     --devices         every input the backend can see, and what each reports\n\
         \x20     --check           load the model and the DSP weights, say whether they are there\n\
         \x20     --bench           time one inference; the model is asked every 40 ms, so that\n\
         \x20                       figure is essentially the whole CPU cost of the app\n\
         \x20     --probe FILE.wav  what the model hears in a recording, window by window: level,\n\
         \x20                       how full its context window was, the twelve pitch\n\
         \x20                       probabilities, and the note the CQT alone reports\n\
         \x20       --gate DB       noise gate for the probe, in dBFS (default -34)\n\
         \x20       --boost GAIN    apply the bass boost, as the settings panel would\n\
         \n\
         Running:\n\
         \x20     --file FILE.wav   drive the trainer from a recording instead of the input\n\
         \n\
         Environment:\n\
         \x20 SOLITITO_DEBUG=1      print the model's reading of every window while playing\n\
         \x20 SOLITITO_STRUM=1      print the strum trace: attack, what was heard, the verdict\n\
         \n\
         Settings live in $XDG_CONFIG_HOME/solitito/settings.json, falling back to\n\
         ~/.config and %APPDATA%. The model and dsp_weights.json are read from the\n\
         working directory.\n",
        env!("CARGO_PKG_VERSION")
    )
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
    // The onset model first: it carries the three older outputs unchanged under
    // their old names, so nothing that reads them notices, and it adds the head
    // that says what was STRUCK. Falls back to the plain file where that one
    // has not been fetched.
    for name in ["best_model_v2_take6_onset.onnx", "best_model_v2_take6.onnx"] {
        if std::path::Path::new(name).exists() {
            return name.to_string();
        }
    }
    "best_model_v2_take6.onnx".to_string()
}

fn probe_file(path: &str, gate_db: f32, boost: Option<f32>, step: usize) -> anyhow::Result<()> {
    use audio::{CTX_FRAMES, FFT_SIZE, HOP_LENGTH, INPUT_GAIN, TARGET_SR, TOTAL_FEATURES};

    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let ch = spec.channels as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int if spec.bits_per_sample == 16 => reader
            .into_samples::<i16>()
            .map(|s| s.unwrap_or(0) as f32 / 32768.0)
            .collect(),
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
        }
        _ => anyhow::bail!("only 16-bit integer or 32-bit float WAV is read here"),
    };

    // First channel and a linear resample to 16 kHz - the same two steps the
    // stream callback does, so the features are the ones the app would see.
    let mono: Vec<f32> = raw.chunks(ch).map(|f| f[0]).collect();
    let ratio = spec.sample_rate as f32 / TARGET_SR as f32;
    let mut sig: Vec<f32> = Vec::with_capacity((mono.len() as f32 / ratio) as usize + 8);
    let mut read = 0.0f32;
    while read + 1.0 < mono.len() as f32 {
        let i = read as usize;
        let f = read - i as f32;
        sig.push(mono[i] + f * (mono[i + 1] - mono[i]));
        read += ratio;
    }

    let mut analyzer = audio::CqtAnalyzer::new("dsp_weights.json")?;
    let mut brain = ChordBrain::new(&model_path())?;
    let gate = db_to_lin(gate_db);

    let mut hist = [[0.0f32; TOTAL_FEATURES]; CTX_FRAMES];
    let mut live = [false; CTX_FRAMES];
    let mut last_cqt: Option<Vec<f32>>;

    println!(
        "\n{path} · {} Hz · {ch} ch · gate {gate_db:.0} dB · bass boost {}",
        spec.sample_rate,
        match boost { Some(g) => format!("x{g:.0}"), None => "off".into() }
    );
    // The onset block is printed whether the model has the head or not - a
    // format that changes shape with the file is a format nothing can parse.
    println!(
        "     t   dBFS  fill    C  C#   D  D#   E   F  F#   G  G#   A  A#   B \
| struck:  C  C#   D  D#   E   F  F#   G  G#   A  A#   B     CQT      model"
    );

    let mut frame = 0usize;
    let mut at = FFT_SIZE;
    while at <= sig.len() {
        let chunk = &sig[at - FFT_SIZE..at];
        let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / FFT_SIZE as f32).sqrt();

        hist.rotate_left(1);
        live.rotate_left(1);
        if rms > gate {
            let amplified: Vec<f32> = chunk.iter().map(|&x| x * INPUT_GAIN).collect();
            let (cqt, chroma, bass, _) =
                analyzer.compute_cqt_chroma(&amplified, boost.is_some(), boost.unwrap_or(1.0));
            let mut f = Vec::with_capacity(TOTAL_FEATURES);
            f.extend_from_slice(&cqt);
            f.extend_from_slice(&chroma);
            f.extend_from_slice(&bass);
            hist[CTX_FRAMES - 1].copy_from_slice(&f);
            live[CTX_FRAMES - 1] = true;
            last_cqt = Some(cqt);
        } else {
            hist[CTX_FRAMES - 1] = [0.0; TOTAL_FEATURES];
            live[CTX_FRAMES - 1] = false;
            last_cqt = None;
        }

        frame += 1;
        if frame >= CTX_FRAMES && frame % step == 0 {
            let fill = live.iter().filter(|&&b| b).count() as f32 / CTX_FRAMES as f32;
            let p = brain.predict(&hist)?;
            let db = if rms > 0.0 { 20.0 * rms.log10() } else { -99.0 };
            let cells: String =
                p.pitches.iter().map(|v| format!("{:4.0}", v * 100.0)).collect();
            let struck: String =
                p.onsets.iter().map(|v| format!("{:4.0}", v * 100.0)).collect();
            let cqt = match last_cqt.as_ref().and_then(|c| audio::mono_pitch(c)) {
                Some((pc, s)) => format!("{:>3} {:.2}", NOTE_NAMES[pc], s),
                None => "  -     ".to_string(),
            };
            println!(
                "{:6.2} {:6.1} {:4.0}% {} |{}  {}  {} {:.2}{}",
                at as f32 / TARGET_SR as f32,
                db,
                fill * 100.0,
                cells,
                struck,
                cqt,
                p.chord,
                p.confidence,
                if fill < 0.9 { "   << app would not ask" } else { "" }
            );
        }
        at += HOP_LENGTH;
    }
    Ok(())
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
    show_full_shapes: bool,
    show_shell_shapes: bool,
    ai_debug: bool,
    startup_mode: i32,
    language: i32,
    intervals: String,
    show_spectrum: bool,
    audio_device: i32,
    audio_channel: i32,
    formula_jazz_names: bool,
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
    debug_console: bool,
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
            show_full_shapes: ui.get_show_full_shapes(),
            show_shell_shapes: ui.get_show_shell_shapes(),
            ai_debug: ui.get_ai_debug_visible(),
            startup_mode: ui.get_startup_mode(),
            language: ui.get_language_idx(),
            intervals: ui.get_interval_input_text().to_string(),
            show_spectrum: ui.get_show_spectrum(),
            audio_device: ui.get_audio_device_index(),
            audio_channel: ui.get_audio_channel_index(),
            formula_jazz_names: ui.get_formula_jazz_names(),
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
            debug_console: ui.get_debug_console(),
        }
    }
}

#[derive(Clone, Default)]
struct AiResult {
    pred: Prediction,
    updated: bool,
    /// How full the context window was when this was asked. The chord NAME is
    /// only believed on a full one - that is what the model was trained on -
    /// while the pitch and onset heads are read whatever it says.
    fill: f32,
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
        cqt_pitch: None,
        gate_open: false,
        frames_seen: 0,
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
    // One of the reporting flags: borrow (or open) a console first, or the
    // report goes nowhere on a Windows release build.
    let flagged = |f: &str| args.iter().any(|a| a == f);
    let reporting = ["--help", "-h", "--devices", "--check", "--bench", "--probe"];
    let console = if reporting.iter().any(|f| flagged(f)) {
        attach_console()
    } else {
        false
    };

    if flagged("--help") || flagged("-h") {
        print!("{}", help_text());
        keep_console_open(console);
        std::process::exit(0);
    }

    if args.iter().any(|a| a == "--devices") {
        audio::hush_alsa();
        for name in audio::list_input_devices() {
            match audio::probe_input(&name) {
                Ok(i) => println!("  {name}\n      {} Hz · {} ch · {}", i.sample_rate, i.channels, i.format),
                Err(e) => println!("  {name}\n      unavailable: {e}"),
            }
        }
        keep_console_open(console);
        std::process::exit(0);
    }

    // --probe FILE.wav: what the model hears in a recording, window by window.
    //
    // The same feature path as the live stream, but with nothing gated away and
    // nothing thrown at the screen - so a question like "is it the model that
    // cannot hear single notes, or does the app never ask?" can be answered from
    // a recording instead of from memory. Prints the level and the window fill
    // beside the twelve pitch probabilities, because those three together are
    // what decides whether a note counts.
    if let Some(i) = args.iter().position(|a| a == "--probe") {
        let Some(path) = args.get(i + 1).cloned() else {
            eprintln!("usage: solitito --probe FILE.wav [--gate DB] [--step N]");
            keep_console_open(console);
            std::process::exit(2);
        };
        let gate_db: f32 = args
            .iter()
            .position(|a| a == "--gate")
            .and_then(|g| args.get(g + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(-34.0);
        let boost = args
            .iter()
            .position(|a| a == "--boost")
            .and_then(|b| args.get(b + 1))
            .and_then(|v| v.parse::<f32>().ok());
        // Every eighth frame is enough to read a chord off the screen; a
        // latency measurement needs every one of them - a frame is 16 ms.
        let step: usize = args
            .iter()
            .position(|a| a == "--step")
            .and_then(|g| args.get(g + 1))
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(8);
        match probe_file(&path, gate_db, boost, step) {
            Ok(()) => {}
            Err(e) => eprintln!("❌ {e}"),
        }
        keep_console_open(console);
        std::process::exit(0);
    }

    // --bench: how long ONE inference takes, which is the whole of the app's
    // load while a chord rings - the model is asked every 40 ms, so 4 ms and
    // 38 ms are the difference between 10% of a core and all of it. Worth a flag
    // rather than a guess: the same machine gave wildly different figures under
    // two operating systems, and only a number says which build is at fault.
    if args.iter().any(|a| a == "--bench") {
        let mut brain = match ChordBrain::new(&model_path()) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("❌ model: {e}");
                keep_console_open(console);
                std::process::exit(1);
            }
        };
        let frames = [[0.05f32; audio::TOTAL_FEATURES]; audio::CTX_FRAMES];
        for _ in 0..5 {
            let _ = brain.predict(&frames);          // warm-up, not measured
        }
        let mut times: Vec<f64> = Vec::new();
        for _ in 0..60 {
            let t0 = std::time::Instant::now();
            let _ = brain.predict(&frames);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        println!(
            "⏱  inference: min {:.1} ms · median {:.1} ms · mean {:.1} ms · max {:.1} ms",
            times[0], times[times.len() / 2], mean, times[times.len() - 1]
        );
        println!("   at one every 40 ms that is {:.0}% of a core", mean / 40.0 * 100.0);
        keep_console_open(console);
        std::process::exit(0);
    }

    if args.iter().any(|a| a == "--check") {
        let mut ok = true;
        match audio::CqtAnalyzer::new("dsp_weights.json") {
            Ok(_) => println!("✅ dsp_weights.json"),
            Err(e) => { eprintln!("❌ dsp_weights.json: {e}"); ok = false; }
        }
        match ChordBrain::new(&model_path()) {
            Ok(_) => println!("✅ best_model_v2_take6.onnx"),
            Err(e) => { eprintln!("❌ model: {e}"); ok = false; }
        }
        keep_console_open(console);
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

    // AI THREAD
    let analysis_for_ai = analysis_state.clone();
    let result_for_ai = ai_result_state.clone();
    
    // Named so `top -H` can tell inference apart from the UI thread - both were
    // just "solitito", which made "is it the model or the drawing?" unanswerable
    // without a debugger.
    let _ai_thread = thread::Builder::new().name("solitito-ai".into()).spawn(move || {
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
                    res.fill = fill;
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

    let my_app = Arc::new(Mutex::new(MyApp::new(analysis_state.clone())));
    my_app.lock().unwrap().set_mode(cfg.startup_mode);
    let ui = AppWindow::new()?;
    let ui_weak = ui.as_weak();

    {
        let app = my_app.lock().unwrap();
        let titles: Vec<SharedString> = app.song_library.iter()
            .map(|s| SharedString::from(&s.title))
            .collect();
        ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(titles))));
        // Per device: the same room and the same fingers give levels tens of
        // decibels apart through an interface and through a laptop microphone.
        ui.set_gate_db(
            cfg.gate_for(cfg.audio_device.as_deref())
                .unwrap_or(default_gate_db),
        );
        ui.set_boost_enabled(default_boost_enabled);
        ui.set_boost_gain(default_boost_gain);
        ui.set_current_mode(app.app_mode as i32); 
        ui.set_startup_mode(cfg.startup_mode);
        ui.set_language_idx(cfg.language);
        ui.set_short_verdict(cfg.short_verdict);
        ui.set_single_notes(cfg.single_notes);
        ui.set_require_onset(cfg.require_onset);
        ui.set_shuffle_chords(cfg.shuffle_chords);
        ui.set_show_diagrams(cfg.show_diagrams);
        ui.set_show_full_shapes(cfg.show_full_shapes);
        ui.set_show_shell_shapes(cfg.show_shell_shapes);
        ui.set_formula_jazz_names(cfg.formula_jazz_names);
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
        ui.set_debug_console(cfg.debug_console);
        ui.set_show_spectrum(cfg.show_spectrum);
        ui.set_ai_debug_visible(cfg.ai_debug);
        ui.set_icon_shuffle(svg_icon(ICON_SHUFFLE));
        ui.set_icon_gear(svg_icon(ICON_GEAR));
        ui.set_icon_star(svg_icon(ICON_STAR));
        ui.set_icon_star_on(svg_icon(ICON_STAR_ON));
        ui.set_icon_pause(svg_icon(ICON_PAUSE));
        ui.set_icon_play(svg_icon(ICON_PLAY));
        apply_language(&ui, lang);
        ui.set_interval_input_text(app.intervals_input.clone().into()); 
    }

    ui.window().set_position(PhysicalPosition::new(450, 10));

    // --- AUDIO INPUT ---
    // Listed BEFORE the stream opens, always. Enumerating means opening every
    // device in turn, and a card that is already taken cannot be opened - so a
    // list built after this app has a card of its own is missing that card, and
    // one built while a sound server holds the hardware is missing all of them.
    // ALSA's own complaints while it probes are silenced by `hush_alsa`.
    let device_names: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    if !file_mode {
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
        fill_device_list(&ui, &names, t.audio_default, &present, None);
        *device_names.borrow_mut() = names;
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
    // The size to write back on the way out. Maximising is not resizing: the
    // window then has the screen's size, and saving that turned a maximise
    // followed by a close into a window that opens the size of the desktop -
    // and lost the size that had been set by hand. Only sizes seen while the
    // window is its own are remembered.
    let last_own_size: Rc<Cell<Option<(u32, u32)>>> = Rc::new(Cell::new(None));
    {
        let uw = ui.as_weak();
        let own = last_own_size.clone();
        ui.on_window_resized(move || {
            if let Some(ui) = uw.upgrade() {
                fit_ui_to_window(&ui);
                let size = ui.window().size();
                if !ui.window().is_maximized() && size.width > 0 && size.height > 0 {
                    own.set(Some((size.width, size.height)));
                }
            }
        });
    }

    // On close rather than on every resize: dragging an edge produces a stream
    // of sizes, and only the last one is the answer.
    {
        let uw = ui.as_weak();
        let cfg = live_cfg.clone();
        let own = last_own_size.clone();
        ui.window().on_close_requested(move || {
            if let Some(ui) = uw.upgrade() {
                // Closed while maximised, with no unmaximised size seen this
                // run: the file keeps what it had rather than learning the
                // screen's size.
                let size = ui.window().size();
                let size = match own.get() {
                    Some(s) => Some(s),
                    None if ui.window().is_maximized() => None,
                    None => Some((size.width, size.height)),
                };
                let mut cfg = cfg.borrow_mut();
                if let Some((w, h)) = size {
                    cfg.window_w = Some(w);
                    cfg.window_h = Some(h);
                }
                let dev = cfg.audio_device.clone();
                cfg.set_gate(dev.as_deref(), ui.get_gate_db());
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
    let names_tick = device_names.clone();
    let cfg_gate = live_cfg.clone();
    let cfg_formulas = live_cfg.clone();
    let opened_tick = opened.clone();

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
    let mut last_ui_tick: Option<std::time::Instant> = None;
    // The chord list only changes with the formula or its key, and rebuilding it
    // every tick would drop whichever chord the pointer is on sixty times a
    // second. The key is in the guard because the chords are also spelled out in
    // letters, and those follow it.
    // Mask, root and placement: a chord change can land the formula on the same
    // root from a different degree, and then only the degree tells this row that
    // its numerals are stale.
    let mut last_formula_chords: (u16, String, usize) = (0, String::new(), usize::MAX);
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
        app.single_notes = ui.get_single_notes();
        app.require_onset = ui.get_require_onset();
        app.set_shuffle_chords(ui.get_shuffle_chords());
        {
            // The two text fields are read from the UI, not pushed into it -
            // the same arrangement the Intervals field uses, and the only one
            // that leaves them editable.
            let key_txt = ui.get_formula_key_text().to_string();
            let req_txt = ui.get_formula_required_text().to_string();
            let mut cfg = cfg_formulas.borrow_mut();
            if cfg.formula_key != key_txt || cfg.formula_required != req_txt {
                cfg.formula_key = key_txt;
                cfg.formula_required = req_txt;
                cfg.save();
            }
        }
        {
            // Formulas options. A formula now stands until something here moves,
            // so a change has to act at once - waiting for "the next formula"
            // would mean waiting for one that is never drawn.
            let cfg = cfg_formulas.borrow();
            let required = formulas::parse(&cfg.formula_required).unwrap_or(1);
            // Which of the two: the size and the filter describe the formula
            // itself, the key only says where to read it. Redrawing on a key
            // change would throw away the formula being practised, and the key
            // field is read letter by letter as it is typed.
            // Changing the exercise changes what the formula is read against,
            // so it redraws: a placement over a chord that is no longer there
            // would be read as a key.
            let redraw = app.formula_notes != cfg.formula_notes
                || app.formula_required != required
                || app.formula_exercise != cfg.formula_exercise;
            // A different kind of placement is the same formula seen from
            // somewhere else - the same gesture as asking for another key.
            let rekey = app.formula_random_key != cfg.formula_random_key
                || app.formula_key_setting != cfg.formula_key
                || app.formula_placement_want != cfg.formula_placement;
            let exercise_changed = app.formula_exercise != cfg.formula_exercise;
            app.formula_exercise = cfg.formula_exercise;
            app.formula_placement_want = cfg.formula_placement;
            app.formula_notes = cfg.formula_notes;
            app.formula_required = required;
            app.formula_random_key = cfg.formula_random_key;
            app.formula_key_setting = cfg.formula_key.clone();
            app.formula_in_order = cfg.formula_in_order;
            app.log_credits = cfg.debug_console;
            if app.app_mode == state::AppMode::Formulas {
                // Over the changes the tune has to be under it before a chord
                // can be taken from it.
                if exercise_changed && cfg.formula_exercise == 2 {
                    app.reload_library();
                }
                if redraw {
                    app.next_formula();
                } else if rekey {
                    app.rekey_formula();
                }
            }
        }
        // Neither the fretboard trainer nor formulas show the pause button, so a
        // pause carried in from another mode would freeze them with nothing on
        // screen to explain why - in Formulas it came out as a set that stayed
        // green while the next one never came.
        if app.app_mode == AppMode::Fretboard && ui.get_paused() {
            ui.set_paused(false);
        }
        app.paused = ui.get_paused();
        let settings_open = ui.get_show_settings();
        // Re-scanned every time the panel opens - and again when the picker
        // itself is opened, see `on_audio_devices_rescan`.
        if settings_open && !settings_were_open && !file_mode {
            let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
            let open_now = (!opened_tick.borrow().fell_back)
                .then(|| cfg_gate.borrow().audio_device.clone())
                .flatten();
            rescan_devices(&ui, &names_tick, t.audio_default, open_now.as_deref());
        }
        if settings_were_open && !settings_open {
            baseline = SettingsSnapshot::read(&ui);   // leaving the panel clears the mark
            // Written when the panel closes rather than on every drag: the
            // slider moves in dozens of steps and each one would be a file.
            let mut cfg = cfg_gate.borrow_mut();
            let dev = cfg.audio_device.clone();
            if cfg.set_gate(dev.as_deref(), ui.get_gate_db()) {
                cfg.save();
            }
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
        // Every frame, model answering or not: the ear is a per-frame estimate,
        // and a finished lap is timed in wall clock rather than in answers.
        let now = std::time::Instant::now();
        let ui_dt = last_ui_tick
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(0.016)
            .min(0.25);
        last_ui_tick = Some(now);
        app.tick(ui_dt);

        if let Ok(mut res) = result_for_ui.lock() {
            if res.updated {
                // A half-full window is answerable for what is SOUNDING, not for
                // what chord it is.
                let named = res.fill >= MIN_FILL_CHORD;
                let chord = res.pred.chord.clone();
                let score = res.pred.confidence;
                app.prev_pitches = app.last_pitches;
                app.last_pitches = res.pred.pitches;
                // What was STRUCK, as against what is sounding. Zeros with a
                // model that has no onset head, and the modes fall back.
                app.set_onsets(res.pred.onsets);

                // clear the flag once consumed
                res.updated = false;

                // Majority voting over 3 windows. probe_quality measured majority
                // voting as better than averaging the distributions - the model is
                // sometimes confident and wrong, and averaging carries that
                // confidence forward while voting damps it. Three rather than five:
                // with five, a new chord needed three predictions to outweigh the
                // old one, adding ~0.2 s to every transition.
                // Only full windows vote. A half one is not evidence against the
                // chord standing, and letting it in would dilute the vote of
                // whoever is strumming shortly.
                if named {
                    app.chord_history.push_back((chord.clone(), score));
                    if app.chord_history.len() > 3 { app.chord_history.pop_front(); }
                }
                
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
                // "..." is what the app already uses for "nothing to say". It is
                // better on screen than the chord from a minute ago: the display
                // used to hold the last name it had for as long as nobody played
                // a full window, which with one note at a time is for ever.
                let shown = if named {
                    chord_latch.update(
                        ui.get_lock_quality(), onset_id, since_onset, best_c, current_confidence,
                    )
                } else {
                    "...".to_string()
                };
                let current_confidence = if named { current_confidence } else { 0.0 };

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

        if app.app_mode == AppMode::Formulas {
            let funcs = formulas::functions_of(app.formula_mask);
            let names: Vec<SharedString> = funcs
                .iter()
                .map(|&f| SharedString::from(formulas::FUNCS[f]))
                .collect();
            ui.set_formula_functions(ModelRc::from(Rc::new(VecModel::from(names))));
            // Paused here means "your turn": the whole set lit, nothing judged,
            // and the screen says so. It is the one mode where a pause is an
            // exercise rather than a halt.
            let lit = if app.paused {
                vec![true; app.formula_collected.len()]
            } else {
                app.formula_collected.clone()
            };
            ui.set_formula_collected(ModelRc::from(Rc::new(VecModel::from(lit))));

            // What the formula is being played over, when it is being played
            // over anything: the chord, the degree of it the formula stands on,
            // the same functions read from the chord's root, and which of them
            // the chord owns. The count is the lesson - the word beside it is
            // only a label on the count.
            {
                let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
                let jazz_names = cfg_formulas.borrow().formula_jazz_names;
                let (chord, degree, against, own, verdict) = match &app.formula_chord {
                    Some(c) if app.formula_exercise != 0 => {
                        let root = c.root as usize;
                        let mut pcs = 0u16;
                        for i in c.quality.intervals() {
                            pcs |= 1 << ((root + i as usize) % 12);
                        }
                        let word = match app.formula_verdict {
                            Some(formulas::Verdict::Defines) => t.placement_defines,
                            Some(formulas::Verdict::Colours) => t.placement_colours,
                            Some(formulas::Verdict::Outside) => t.placement_outside,
                            None => "",
                        };
                        let tones = pcs.count_ones();
                        (
                            format!("{}{}", c.root.to_string(), c.quality.to_string()),
                            formulas::ROMAN[app.formula_degree].to_string(),
                            {
                                // What belongs to the chord is named by the
                                // chord - its own third is "3" or "b3", not a
                                // ninth of anything. What does not is a plain
                                // function, or a tension if that is asked for.
                                let mut own: [Option<&str>; 12] = [None; 12];
                                let names = c.quality.interval_names();
                                for (k, i) in c.quality.intervals().iter().enumerate() {
                                    if let Some(n) = names.get(k) {
                                        own[*i as usize % 12] = Some(n.as_str());
                                    }
                                }
                                formulas::functions_of(app.formula_mask)
                                    .iter()
                                    .map(|&f| {
                                        let semi = (f + app.formula_degree) % 12;
                                        if let Some(n) = own[semi] {
                                            n.to_string()
                                        } else if jazz_names {
                                            formulas::TENSIONS[semi].to_string()
                                        } else {
                                            formulas::FUNCS[semi].to_string()
                                        }
                                    })
                                    .collect()
                            },
                            formulas::chord_tones_in(
                                app.formula_mask,
                                root,
                                pcs,
                                app.formula_degree,
                            ),
                            format!(
                                "{} · {} {} {} {}",
                                word, app.formula_hits, t.of_count, tones, t.chord_tones
                            ),
                        )
                    }
                    _ => (String::new(), String::new(), vec![], vec![], String::new()),
                };
                set_if_changed(ui.get_formula_chord_name(), chord.into(), |v| {
                    ui.set_formula_chord_name(v)
                });
                set_if_changed(ui.get_formula_degree_name(), degree.into(), |v| {
                    ui.set_formula_degree_name(v)
                });
                let label = match &app.formula_chord {
                    Some(c) if app.formula_exercise != 0 => {
                        format!("{}{}", t.against_chord, c.root.to_string())
                    }
                    _ => String::new(),
                };
                set_if_changed(ui.get_formula_against_label(), label.into(), |v| {
                    ui.set_formula_against_label(v)
                });
                set_if_changed(ui.get_formula_verdict(), verdict.into(), |v| {
                    ui.set_formula_verdict(v)
                });
                let against: Vec<SharedString> =
                    against.into_iter().map(SharedString::from).collect();
                ui.set_formula_against(ModelRc::from(Rc::new(VecModel::from(against))));
                ui.set_formula_is_chord_tone(ModelRc::from(Rc::new(VecModel::from(own))));
            }
            set_if_changed(ui.get_formula_key(), app.formula_key_name.clone().into(), |v| {
                ui.set_formula_key(v)
            });

            // Note names are a crutch, so they are only built when asked for.
            let notes: Vec<SharedString> = if cfg_formulas.borrow().formula_note_names {
                match formulas::parse_key(&app.formula_key_name) {
                    Some(k) => formulas::note_names(app.formula_mask, &k)
                        .into_iter()
                        .map(SharedString::from)
                        .collect(),
                    None => vec![],
                }
            } else {
                vec![]
            };
            ui.set_formula_note_names(ModelRc::from(Rc::new(VecModel::from(notes))));

            // The favourites, narrowed by whatever is typed beside them, and
            // whether what is on screen is one of them. Rebuilt every tick: the
            // list is short and the filter has to answer as it is typed.
            {
                let cfg = cfg_formulas.borrow();
                let needle = ui.get_favourite_filter().to_string().to_lowercase();
                let shown: Vec<SharedString> = cfg
                    .favourites
                    .iter()
                    .filter(|f| needle.is_empty() || f.name.to_lowercase().contains(&needle))
                    .map(|f| SharedString::from(f.name.as_str()))
                    .collect();
                if ui.get_favourites().row_count() != shown.len()
                    || (0..shown.len()).any(|i| ui.get_favourites().row_data(i) != Some(shown[i].clone()))
                {
                    ui.set_favourites(ModelRc::from(Rc::new(VecModel::from(shown))));
                }
                // What the combo shows: the formula on screen, if it is one of
                // them. Left at -1 the box was blank even right after a search
                // had put something there, and the only way to see the result
                // was to open the list.
                let here = cfg
                    .favourites
                    .iter()
                    .filter(|f| needle.is_empty() || f.name.to_lowercase().contains(&needle))
                    .position(|f| f.mask == app.formula_mask);
                set_if_changed(ui.get_formula_starred(), here.is_some(), |v| {
                    ui.set_formula_starred(v)
                });
                // The formula on screen when it is one of them; otherwise
                // whatever was last picked, and failing that the first row.
                // Falling back to -1 left the box blank whenever the drawn
                // formula happened not to be a favourite, which is most of the
                // time, and a blank box reads as an empty list.
                let shown_len = ui.get_favourites().row_count() as i32;
                let idx = match here {
                    Some(i) => i as i32,
                    None if shown_len == 0 => -1,
                    None => ui.get_favourite_index().clamp(0, shown_len - 1),
                };
                set_if_changed(ui.get_favourite_index(), idx, |v| ui.set_favourite_index(v));
            }

            // Chords that fit inside the formula, with the functions each is
            // built from flagged against the row of functions above - pointing
            // at one lights them up. Six is what a line holds; past that the
            // list stops saying anything.
            if last_formula_chords.0 != app.formula_mask
                || last_formula_chords.1 != app.formula_key_name
                || last_formula_chords.2 != app.formula_degree
            {
                last_formula_chords = (
                    app.formula_mask,
                    app.formula_key_name.clone(),
                    app.formula_degree,
                );
                let fits = formulas::chords_inside(app.formula_mask, 7);
                // Counted from the chord under the formula when there is one,
                // so this row agrees with the two above it.
                let names: Vec<SharedString> = match app.formula_exercise {
                    0 => fits.iter().map(|f| SharedString::from(f.name())).collect(),
                    _ => fits
                        .iter()
                        .map(|f| SharedString::from(f.name_from(app.formula_degree)))
                        .collect(),
                };
                // The same chords in letters, under the degrees - the note names
                // under the functions, one row down.
                let spelled: Vec<SharedString> = match formulas::parse_key(&app.formula_key_name) {
                    Some(k) => fits.iter().map(|f| SharedString::from(f.named_in(&k))).collect(),
                    None => vec![],
                };
                ui.set_formula_chord_names(ModelRc::from(Rc::new(VecModel::from(spelled))));
                let mut flags: Vec<bool> = Vec::with_capacity(fits.len() * funcs.len());
                for fit in &fits {
                    let used = fit.functions();
                    flags.extend(funcs.iter().map(|f| used.contains(f)));
                }
                ui.set_formula_chords(ModelRc::from(Rc::new(VecModel::from(names))));
                ui.set_formula_chord_in(ModelRc::from(Rc::new(VecModel::from(flags))));
                // The chord pointed at belonged to the formula that has gone.
                ui.set_formula_hover(-1);
            }

            // What this formula is nearly. The scale is handed over spelled
            // out, with the formula's own functions flagged, so the screen can
            // show the difference instead of describing it.
            let near = formulas::nearest_scales(app.formula_mask, &app.scale_definitions, 1);
            let lang_now = Lang::from_setting(ui.get_language_idx());
            let (name, degrees, inside) = match near.first() {
                Some(n) => {
                    let sc = &app.scale_definitions[n.scale];
                    // Spelled from the semitone rather than from the scale file:
                    // scale definitions write #4 where a formula writes b5, and
                    // the two rows sit one above the other. There is no harmonic
                    // argument for either here - a drawn set has no key of its
                    // own - so they may as well agree, and flats are what the
                    // functions above already use.
                    let degs: Vec<SharedString> = sc
                        .intervals
                        .iter()
                        .map(|&semi| SharedString::from(formulas::FUNCS[semi as usize % 12]))
                        .collect();
                    let flags: Vec<bool> = sc
                        .intervals
                        .iter()
                        .map(|&semi| app.formula_mask & (1 << (semi as usize % 12)) != 0)
                        .collect();
                    (i18n::scale_name(lang_now, &sc.name).to_string(), degs, flags)
                }
                None => (String::new(), vec![], vec![]),
            };
            set_if_changed(ui.get_formula_scale_name(), name.into(), |v| {
                ui.set_formula_scale_name(v)
            });
            ui.set_formula_scale_degrees(ModelRc::from(Rc::new(VecModel::from(degrees))));
            ui.set_formula_scale_in(ModelRc::from(Rc::new(VecModel::from(inside))));
            ui.set_interval_names(ModelRc::from(Rc::new(VecModel::from(Vec::<SharedString>::new()))));
        } else if app.app_mode == AppMode::Fretboard {
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
            // The kind belongs in the key as much as the quality does: the
            // same chord has a different row of shapes under each.
            // Neither ticked draws nothing, which is a legitimate answer and
            // not the same as the master switch being off: that one is about
            // whether the row exists at all.
            let shapes = match (ui.get_show_full_shapes(), ui.get_show_shell_shapes()) {
                (true, true) => Some(diagrams::Shapes::Both),
                (false, true) => Some(diagrams::Shapes::Shell),
                (true, false) => Some(diagrams::Shapes::Full),
                (false, false) => None,
            };
            let q_key = format!("{}/{:?}", curr_chord.quality.to_string(), shapes);
            if q_key != last_diagram_key {
                last_diagram_key = q_key;
                let imgs: Vec<slint::Image> = match shapes {
                    Some(kind) => diagrams::for_quality(&curr_chord.quality, kind)
                        .iter()
                        .filter_map(|d| slint::Image::load_from_svg_data(d.svg.as_bytes()).ok())
                        .collect(),
                    None => vec![],
                };
                // Past four, half of them go to a second row: six side by side
                // came to 60px each at the smallest window. The split is here
                // rather than in the UI because a Slint model cannot be sliced.
                let per_row = if imgs.len() > 4 { imgs.len().div_ceil(2) } else { imgs.len() };
                let rest = imgs[per_row..].to_vec();
                let first = imgs[..per_row].to_vec();
                ui.set_chord_diagrams(ModelRc::from(Rc::new(VecModel::from(first))));
                ui.set_chord_diagrams_2(ModelRc::from(Rc::new(VecModel::from(rest))));
                // Captioned whenever the row is not the chord's own shell: a
                // substitute taken for the chord itself is worse than no
                // substitute at all.
                let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
                let note = match (shapes, diagrams::shell_note(&curr_chord.quality)) {
                    (None, _) | (Some(diagrams::Shapes::Full), _) => "",
                    (_, diagrams::ShellNote::Own) => "",
                    (_, diagrams::ShellNote::MinorForHalfDim) => t.shapes_substitute,
                    (_, diagrams::ShellNote::None_) => t.shapes_no_shell,
                };
                // Naming the four dominants beats stating the rule that finds
                // them. A diminished seventh is the third, fifth, seventh and
                // flat ninth of a dominant whose root sits a semitone below any
                // of its notes - which is four dominants, and reading that
                // sentence is harder than reading the names.
                let note = if note == t.shapes_no_shell {
                    let r = curr_chord.root as usize;
                    let names: Vec<&str> = [11usize, 2, 5, 8]
                        .iter()
                        .map(|d| formulas::KEY_POOL[(r + d) % 12])
                        .collect();
                    format!("{}{}7b9{}", note, names.join("7b9, "), t.shapes_no_shell_end)
                } else {
                    note.to_string()
                };
                ui.set_shapes_note(note.into());
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
        // Fired as the picker opens, before its list is drawn: an interface
        // plugged in while the app was running is on the list by the time it
        // appears, without a restart and without a Refresh button to find.
        let names_cb = device_names.clone();
        let info_rescan = opened.clone();
        let cfg_rescan = live_cfg.clone();
        let uw = ui.as_weak();
        ui.on_audio_devices_rescan(move || {
            let Some(ui) = uw.upgrade() else { return };
            let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
            let open_now = (!info_rescan.borrow().fell_back)
                .then(|| cfg_rescan.borrow().audio_device.clone())
                .flatten();
            rescan_devices(&ui, &names_cb, t.audio_default, open_now.as_deref());
        });
    }

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
                // The gate belongs to the device being left, not to the one
                // arriving - save it before the name changes under it.
                let leaving = c.audio_device.clone();
                c.set_gate(leaving.as_deref(), ui.get_gate_db());
                c.audio_device = name.clone();
                c.save();
                c.audio_channel
            };
            // Whatever this input was last set to, or the app default if it is
            // new. Set before the stream opens so the meter and the threshold
            // agree from the first frame.
            if let Some(db) = cur.borrow().gate_for(name.as_deref()) {
                ui.set_gate_db(db);
            }
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
        ui.on_ai_debug_changed(move |on| {
            let mut cur = cur.borrow_mut();
            cur.ai_debug = on;
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
        ui.on_single_notes_changed({
            let cur = cur.clone();
            move |on| {
                let mut cur = cur.borrow_mut();
                cur.single_notes = on;
                cur.save();
            }
        });
        ui.on_show_diagrams_changed({
            let cur = cur.clone();
            move |on| {
                let mut cur = cur.borrow_mut();
                cur.show_diagrams = on;
                cur.save();
            }
        });
        ui.on_shapes_choice_changed({
            let cur = cur.clone();
            move |full: bool, shell: bool| {
                let mut cur = cur.borrow_mut();
                cur.show_full_shapes = full;
                cur.show_shell_shapes = shell;
                cur.save();
            }
        });
        ui.on_require_onset_changed(move |on| {
            let mut cur = cur.borrow_mut();
            cur.require_onset = on;
            cur.save();
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_formula_jazz_names_changed({
            let cur = cur.clone();
            move |on| {
                let mut cur = cur.borrow_mut();
                cur.formula_jazz_names = on;
                cur.save();
            }
        });
        ui.on_formula_exercise_changed({
            let cur = cur.clone();
            move |v| {
                let mut cur = cur.borrow_mut();
                cur.formula_exercise = v.clamp(0, 2) as u8;
                cur.save();
            }
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_formula_placement_changed({
            let cur = cur.clone();
            move |v| {
                let mut cur = cur.borrow_mut();
                cur.formula_placement = v.clamp(0, 3) as u8;
                cur.save();
            }
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_formula_notes_changed(move |n| {
            let mut cur = cur.borrow_mut();
            cur.formula_notes = n.clamp(1, 12) as usize;
            cur.save();
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_formula_random_key_changed(move |on| {
            let mut cur = cur.borrow_mut();
            cur.formula_random_key = on;
            cur.save();
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_formula_show_names_changed(move |on| {
            let mut cur = cur.borrow_mut();
            cur.formula_note_names = on;
            cur.save();
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_formula_show_similar_changed(move |on| {
            let mut cur = cur.borrow_mut();
            cur.formula_show_similar = on;
            cur.save();
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_in_order_changed(move |on| {
            let mut cur = cur.borrow_mut();
            cur.formula_in_order = on;
            cur.save();
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_debug_console_changed(move |on| {
            let mut cur = cur.borrow_mut();
            cur.debug_console = on;
            cur.save();
        });
    }
    {
        // The star: keep what is on screen, or let go of it. Kept formulas are
        // matched on the set AND the key, because the same functions read from
        // another root are a different exercise.
        let cur = live_cfg.clone();
        let app_star = my_app.clone();
        let uw = ui.as_weak();
        ui.on_star_clicked(move || {
            let Some(ui) = uw.upgrade() else { return };
            let mask = app_star.lock().unwrap().formula_mask;
            let mut cur = cur.borrow_mut();
            let before = cur.favourites.len();
            cur.favourites.retain(|f| f.mask != mask);
            if cur.favourites.len() != before {
                cur.save();
                return;
            }
            // Not kept yet: ask what to call it. The field opens empty so the
            // word in it can be read, and Enter on an empty field still names
            // it after the formula - see `on_favourite_named`.
            ui.set_favourite_name(SharedString::new());
            ui.set_naming(true);
        });
    }
    {
        let cur = live_cfg.clone();
        let app_named = my_app.clone();
        ui.on_favourite_named(move |name| {
            let mask = app_named.lock().unwrap().formula_mask;
            let name = name.to_string();
            let name = if name.trim().is_empty() {
                formulas::to_text(mask)
            } else {
                name.trim().to_string()
            };
            let mut cur = cur.borrow_mut();
            cur.favourites.retain(|f| f.mask != mask);
            cur.favourites.push(settings::Favourite { name, mask });
            cur.save();
        });
    }
    {
        // Enter in the search field, or the button beside it: take the first
        // one left in the narrowed list. Typing three letters and reaching for
        // the mouse to finish the job is the thing this avoids.
        let cur = live_cfg.clone();
        let app_search = my_app.clone();
        let uw = ui.as_weak();
        ui.on_favourite_search(move || {
            let Some(ui) = uw.upgrade() else { return };
            let cur = cur.borrow();
            let needle = ui.get_favourite_filter().to_string().to_lowercase();
            if let Some(f) = cur
                .favourites
                .iter()
                .find(|f| needle.is_empty() || f.name.to_lowercase().contains(&needle))
            {
                app_search.lock().unwrap().load_formula(f.mask);
            }
        });
    }
    {
        // Picking one draws it. The index is into the FILTERED list, so the
        // filter is applied again here rather than trusted to have stood still.
        let cur = live_cfg.clone();
        let app_pick = my_app.clone();
        let uw = ui.as_weak();
        ui.on_favourite_picked(move |idx| {
            let Some(ui) = uw.upgrade() else { return };
            let cur = cur.borrow();
            let needle = ui.get_favourite_filter().to_string().to_lowercase();
            let Some(f) = cur
                .favourites
                .iter()
                .filter(|f| needle.is_empty() || f.name.to_lowercase().contains(&needle))
                .nth(idx.max(0) as usize)
            else {
                return;
            };
            app_pick.lock().unwrap().load_formula(f.mask);
        });
    }
    {
        // The cross on a row. Same walk as picking one - the index is into the
        // filtered list - and then it is gone from the file as well as the list.
        let cur = live_cfg.clone();
        let uw = ui.as_weak();
        ui.on_favourite_deleted(move |idx| {
            let Some(ui) = uw.upgrade() else { return };
            let mut cur = cur.borrow_mut();
            let needle = ui.get_favourite_filter().to_string().to_lowercase();
            let Some(mask) = cur
                .favourites
                .iter()
                .filter(|f| needle.is_empty() || f.name.to_lowercase().contains(&needle))
                .nth(idx.max(0) as usize)
                .map(|f| f.mask)
            else {
                return;
            };
            cur.favourites.retain(|f| f.mask != mask);
            cur.save();
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_formula_show_chords_changed(move |on| {
            let mut cur = cur.borrow_mut();
            cur.formula_show_chords = on;
            cur.save();
        });
    }
    {
        let cur = live_cfg.clone();
        ui.on_shuffle_chords_changed(move |on| {
            let mut cur = cur.borrow_mut();
            cur.shuffle_chords = on;
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
        // The language as it is now, not as it was at startup.
        let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
        let (label, sec_label) = mode_labels(&t, app.app_mode);
        let (items, sec_items): (Vec<SharedString>, Vec<SharedString>) = match app.app_mode {
            AppMode::Scales => (
                app.scale_definitions.iter().map(|s| SharedString::from(&s.name)).collect(),
                keys_list_clone.clone(),
            ),
            AppMode::Arpeggios => (
                app.song_library.iter().map(|s| SharedString::from(&s.title)).collect(),
                app.arpeggio_patterns.iter().map(|s| SharedString::from(&s.name)).collect(),
            ),
            _ => (
                app.song_library.iter().map(|s| SharedString::from(&s.title)).collect(),
                vec![],
            ),
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
    {
        let app_weak = my_app.clone();
        ui.on_next_change(move || {
            app_weak.lock().unwrap().next_change();
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
/// The two combo labels a mode shows over its pickers.
///
/// Wanted in two places - when the mode changes and when the language does -
/// and it used to be built only in the first, from a language captured at
/// startup. Switching to English left "Utwór" standing over an English panel.
fn mode_labels(t: &i18n::Strings, mode: AppMode) -> (&'static str, &'static str) {
    match mode {
        AppMode::Scales => (t.select_scale, t.key_root),
        AppMode::Arpeggios => (t.select_song, t.pattern),
        _ => (t.select_song, ""),
    }
}

fn apply_language(ui: &AppWindow, lang: Lang) {
    let t = i18n::strings(lang);
    // Not everything on screen goes through the Tr global: these three are
    // built in Rust and stayed in whatever language they were made in.
    let (label, sec_label) = mode_labels(&t, AppMode::from(ui.get_current_mode()));
    ui.set_library_label(label.into());
    ui.set_secondary_label(sec_label.into());
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
    g.set_tab_audio(t.tab_audio.into());
    g.set_tab_practice(t.tab_practice.into());
    g.set_tab_general(t.tab_general.into());
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
    g.set_single_notes(t.single_notes.into());
    g.set_single_notes_hint(t.single_notes_hint.into());
    g.set_require_onset(t.require_onset.into());
    g.set_require_onset_hint(t.require_onset_hint.into());
    g.set_in_order(t.in_order.into());
    g.set_in_order_hint(t.in_order_hint.into());
    g.set_debug_console(t.debug_console.into());
    g.set_shuffle_chords(t.shuffle_chords.into());
    g.set_shuffle_chords_hint(t.shuffle_chords_hint.into());
    g.set_random_order(t.random_order.into());
    g.set_show_diagrams(t.show_diagrams.into());
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



