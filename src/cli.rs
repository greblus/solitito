//! The reporting modes: each prints what it was asked for and exits without
//! opening the trainer window. `--help` lists them.

use crate::brain::{ChordBrain, Prediction};
use crate::audio::db_to_lin;
use crate::{audio, model, model_path, strike, tab, NOTE_NAMES};

/// One reporting mode, with where its flag stands among the arguments when it
/// reads values after it.
enum Report {
    Help,
    Devices,
    Tab { at: usize },
    Probe { at: usize },
    Bench,
    Check,
}

/// Runs the report asked for on the command line and returns its exit code,
/// or `None` when no report was asked for and the trainer should start.
pub fn run_report(args: &[String]) -> Option<i32> {
    let report = report_asked_for(args)?;

    // Borrow (or open) a console first, or the report goes nowhere on a
    // Windows release build.
    let console = attach_console();
    let exit_code = match report {
        Report::Help => {
            print!("{}", help_text());
            0
        }
        Report::Devices => {
            print_devices();
            0
        }
        Report::Tab { at } => {
            print_tabs(args, at);
            0
        }
        Report::Probe { at } => probe(args, at),
        Report::Bench => bench(),
        Report::Check => check(),
    };
    keep_console_open(console);
    Some(exit_code)
}

/// The first reporting flag found, in this order of precedence.
fn report_asked_for(args: &[String]) -> Option<Report> {
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let position = |flag: &str| args.iter().position(|a| a == flag);

    if has("--help") || has("-h") {
        Some(Report::Help)
    } else if has("--devices") {
        Some(Report::Devices)
    } else if let Some(at) = position("--tab") {
        Some(Report::Tab { at })
    } else if let Some(at) = position("--probe") {
        Some(Report::Probe { at })
    } else if has("--bench") {
        Some(Report::Bench)
    } else if has("--check") {
        Some(Report::Check)
    } else {
        None
    }
}

/// The value written after `flag`, parsed, if both are there.
fn value_after<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    let at = args.iter().position(|a| a == flag)?;
    args.get(at + 1)?.parse().ok()
}

/// --devices: what the backend can see, and what each one reports. On a
/// machine with a sound server, `default`, `pulse` and `pipewire` are three
/// names for the same path - the server's mix, not the interface's sockets -
/// which is why picking a channel on them changes nothing.
fn print_devices() {
    audio::hush_alsa();
    for name in audio::list_input_devices() {
        match audio::probe_input(&name) {
            Ok(i) => println!("  {name}\n      {} Hz · {} ch · {}", i.sample_rate, i.channels, i.format),
            Err(e) => println!("  {name}\n      unavailable: {e}"),
        }
    }
}

/// --tab: the arpeggio phrases as tablature, without starting the window. The
/// shapes are a table of degrees and the drawing is what says whether the
/// table is right, so it has to be readable from a terminal like the rest.
fn print_tabs(args: &[String], at: usize) {
    let wanted = args.get(at + 1).cloned().unwrap_or_default();
    let root = args.get(at + 2).map(|s| s.as_str()).unwrap_or("A").to_string();
    let quality = args.get(at + 3).map(|s| s.as_str()).unwrap_or("m7").to_string();
    let request = TabRequest {
        args,
        root_pc: NOTE_NAMES.iter().position(|n| *n == root).unwrap_or(9),
        root: root.clone(),
        quality: quality_named(&quality),
        // How many of the phrase to light green, for looking at the marks
        // without playing the thing.
        played: args.get(at + 4).and_then(|v| v.parse().ok()).unwrap_or(0),
    };
    for pattern in phrases_named(&wanted) {
        eprintln!("{} · {root} {quality}", pattern.name);
        println!("{}", request.draw(&model::steps_of(&pattern.names)));
    }
}

/// The chord quality as `--tab` spells it; anything unknown is m7.
fn quality_named(name: &str) -> model::ChordQuality {
    match name {
        "maj7" => model::ChordQuality::Major7,
        "7" => model::ChordQuality::Dominant7,
        "m7b5" => model::ChordQuality::HalfDiminished,
        "dim" => model::ChordQuality::Diminished,
        _ => model::ChordQuality::Minor7,
    }
}

/// The phrases `--tab` was asked for: every bundled pattern whose name holds
/// `wanted`, or - when it starts with a digit - the phrase typed out itself,
/// `--tab "1 3 7"`, which is what an interval set is and the only way to look
/// at a grip without opening the window.
fn phrases_named(wanted: &str) -> Vec<model::ScaleDefinition> {
    if wanted.starts_with(|c: char| c.is_ascii_digit()) {
        return vec![model::ScaleDefinition {
            name: wanted.to_string(),
            intervals: vec![],
            names: wanted.split_whitespace().map(String::from).collect(),
        }];
    }
    let needle = wanted.to_lowercase();
    model::load_arpeggio_patterns()
        .into_iter()
        .filter(|pattern| pattern.name.to_lowercase().contains(&needle))
        .collect()
}

/// What `--tab` draws every phrase over, and which drawing the flags ask for.
struct TabRequest<'a> {
    args: &'a [String],
    root: String,
    root_pc: usize,
    quality: model::ChordQuality,
    played: usize,
}

impl TabRequest<'_> {
    fn flagged(&self, flag: &str) -> bool {
        self.args.iter().any(|a| a == flag)
    }

    /// One phrase as SVG: the strip by default, or the neck, a chord box, or
    /// the fretboard trainer's region.
    fn draw(&self, steps: &[model::Step]) -> String {
        let intervals = self.quality.intervals();
        let names = self.quality.interval_names();
        let frets = self.flagged("--frets");
        let spots = tab::place(self.root_pc, &intervals, steps, &names);
        let done: Vec<bool> = (0..spots.len()).map(|i| i < self.played).collect();
        if self.flagged("--region") {
            // The fretboard trainer's picture: strings in play bright, the rest
            // dim, and the note revealed only when asked for.
            let strings: Vec<usize> = (0..3).collect();
            let marks: Vec<(usize, i32)> =
                if self.played > 0 { vec![(0, 8), (1, 8)] } else { vec![] };
            return tab::region(&strings, 7, 10, &marks, &self.root, self.played > 1);
        }
        if self.flagged("--neck") {
            return tab::neck(&spots, &done, self.played, frets);
        }
        if self.flagged("--grip") {
            let close = self.flagged("--close");
            let voiced =
                tab::place_voiced(self.root_pc, &intervals, steps, &names, None, 5, 0, close);
            return tab::grip(&voiced, &done, self.played, frets);
        }
        tab::svg(&spots, &done, self.played, frets)
    }
}

/// --probe FILE.wav: the same feature path as the live stream, but with
/// nothing gated away and nothing thrown at the screen - so a question like "is
/// it the model that cannot hear single notes, or does the app never ask?" can
/// be answered from a recording instead of from memory.
fn probe(args: &[String], at: usize) -> i32 {
    let Some(path) = args.get(at + 1) else {
        eprintln!("usage: solitito --probe FILE.wav [--gate DB] [--step N]");
        return 2;
    };
    let gate_db: f32 = value_after(args, "--gate").unwrap_or(-34.0);
    let boost: Option<f32> = value_after(args, "--boost");
    // Every eighth frame is enough to read a chord off the screen; a
    // latency measurement needs every one of them - a frame is 16 ms.
    let step: usize = value_after(args, "--step").filter(|&n| n >= 1).unwrap_or(8);
    if let Err(e) = probe_file(path, gate_db, boost, step) {
        eprintln!("❌ {e}");
    }
    0
}

/// --bench: how long ONE inference takes, which is the whole of the app's
/// load while a chord rings - the model is asked every 40 ms, so 4 ms and
/// 38 ms are the difference between 10% of a core and all of it. Worth a flag
/// rather than a guess: the same machine gave wildly different figures under
/// two operating systems, and only a number says which build is at fault.
fn bench() -> i32 {
    let mut brain = match ChordBrain::new(&model_path()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("❌ model: {e}");
            return 1;
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
    times.sort_by(|a, b| a.total_cmp(b));
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "⏱  inference: min {:.1} ms · median {:.1} ms · mean {:.1} ms · max {:.1} ms",
        times[0], times[times.len() / 2], mean, times[times.len() - 1]
    );
    println!("   at one every 40 ms that is {:.0}% of a core", mean / 40.0 * 100.0);
    0
}

/// --check: load the DSP weights and the model, report, exit. Without it a
/// package cannot be verified on a machine with no sound card and no display:
/// the weights are only read when the audio stream opens.
fn check() -> i32 {
    let mut ok = true;
    match audio::CqtAnalyzer::new("dsp_weights.json") {
        Ok(_) => println!("✅ dsp_weights.json"),
        Err(e) => { eprintln!("❌ dsp_weights.json: {e}"); ok = false; }
    }
    let model = model_path();
    match ChordBrain::new(&model) {
        Ok(_) => println!("✅ {model}"),
        Err(e) => { eprintln!("❌ model: {e}"); ok = false; }
    }
    // Optional: without it the app runs, judging repeats on older evidence.
    match strike::Strikes::load_any() {
        Ok((s, path)) => println!("✅ {path} (próg {:.2})", s.latch.threshold),
        Err(e) => println!("⚠️  model uderzeń: {e} - powtórzenia bez niego"),
    }
    if ok { 0 } else { 1 }
}

/// Runs a recording through the live feature path and reports, window by
/// window, what the model makes of it.
///
/// Deliberately ungated: the app only asks the model once 90% of the 48-frame
/// window carries signal, so "the model cannot hear it" and "the app never
/// asked" look identical from the outside. Here the level, the window fill and
/// the twelve pitch probabilities are printed side by side, which separates the
/// two.
fn probe_file(path: &str, gate_db: f32, boost: Option<f32>, step: usize) -> anyhow::Result<()> {
    use audio::{CTX_FRAMES, FFT_SIZE, HOP_LENGTH};

    let (sig, spec) = read_at_16k(path)?;
    let mut analyzer = audio::CqtAnalyzer::new("dsp_weights.json")?;
    let mut brain = ChordBrain::new(&model_path())?;
    let gate = db_to_lin(gate_db);

    println!(
        "\n{path} · {} Hz · {} ch · gate {gate_db:.0} dB · bass boost {}",
        spec.sample_rate,
        spec.channels,
        match boost { Some(g) => format!("x{g:.0}"), None => "off".into() }
    );
    println!(
        "     t   dBFS  fill    C  C#   D  D#   E   F  F#   G  G#   A  A#   B     CQT      model"
    );

    let mut window = ProbeWindow::new();
    let mut frame = 0usize;
    let mut at = FFT_SIZE;
    while at <= sig.len() {
        let chunk = &sig[at - FFT_SIZE..at];
        let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / FFT_SIZE as f32).sqrt();
        window.push(chunk, rms > gate, &mut analyzer, boost);
        frame += 1;
        if frame >= CTX_FRAMES && frame % step == 0 {
            let prediction = brain.predict(&window.history)?;
            print_probe_row(at, rms, window.fill(), &prediction, window.last_cqt.as_deref());
        }
        at += HOP_LENGTH;
    }
    Ok(())
}

/// A WAV file's first channel, resampled linearly to 16 kHz - the same two
/// steps the stream callback does, so the features are the ones the app would
/// see.
fn read_at_16k(path: &str) -> anyhow::Result<(Vec<f32>, hound::WavSpec)> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
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
    let mono: Vec<f32> = raw.chunks(spec.channels as usize).map(|f| f[0]).collect();
    let ratio = spec.sample_rate as f32 / audio::TARGET_SR as f32;
    let mut sig: Vec<f32> = Vec::with_capacity((mono.len() as f32 / ratio) as usize + 8);
    let mut read = 0.0f32;
    while read + 1.0 < mono.len() as f32 {
        let i = read as usize;
        let f = read - i as f32;
        sig.push(mono[i] + f * (mono[i + 1] - mono[i]));
        read += ratio;
    }
    Ok((sig, spec))
}

/// The model's context window as the probe fills it: the features of the last
/// `CTX_FRAMES` windows, which of them carried signal, and the newest CQT.
struct ProbeWindow {
    history: [[f32; audio::TOTAL_FEATURES]; audio::CTX_FRAMES],
    live: [bool; audio::CTX_FRAMES],
    last_cqt: Option<Vec<f32>>,
}

impl ProbeWindow {
    fn new() -> Self {
        ProbeWindow {
            history: [[0.0; audio::TOTAL_FEATURES]; audio::CTX_FRAMES],
            live: [false; audio::CTX_FRAMES],
            last_cqt: None,
        }
    }

    /// One window of audio in, features if the gate is `open`, silence if not.
    fn push(
        &mut self,
        chunk: &[f32],
        open: bool,
        analyzer: &mut audio::CqtAnalyzer,
        boost: Option<f32>,
    ) {
        let newest = audio::CTX_FRAMES - 1;
        self.history.rotate_left(1);
        self.live.rotate_left(1);
        if !open {
            self.history[newest] = [0.0; audio::TOTAL_FEATURES];
            self.live[newest] = false;
            self.last_cqt = None;
            return;
        }
        let amplified: Vec<f32> = chunk.iter().map(|&x| x * audio::INPUT_GAIN).collect();
        let (cqt, chroma, bass, _) =
            analyzer.compute_cqt_chroma(&amplified, boost.is_some(), boost.unwrap_or(1.0));
        let mut features = Vec::with_capacity(audio::TOTAL_FEATURES);
        features.extend_from_slice(&cqt);
        features.extend_from_slice(&chroma);
        features.extend_from_slice(&bass);
        self.history[newest].copy_from_slice(&features);
        self.live[newest] = true;
        self.last_cqt = Some(cqt);
    }

    /// How much of the window carries signal, 0 to 1.
    fn fill(&self) -> f32 {
        self.live.iter().filter(|&&b| b).count() as f32 / audio::CTX_FRAMES as f32
    }
}

/// One line of the probe: time, level, fill, the twelve pitch probabilities,
/// the note the CQT alone reports, and the model's chord.
fn print_probe_row(at: usize, rms: f32, fill: f32, p: &Prediction, last_cqt: Option<&[f32]>) {
    let db = if rms > 0.0 { 20.0 * rms.log10() } else { -99.0 };
    let cells: String = p.pitches.iter().map(|v| format!("{:4.0}", v * 100.0)).collect();
    let cqt = match last_cqt.and_then(audio::mono_pitch) {
        // The octave too: what tells a note struck again from the same note
        // still ringing below it is which one the estimate reads.
        Some((n, s)) => format!("{:>3}{} {:.2}", NOTE_NAMES[n % 12], n / 12, s),
        None => "  -      ".to_string(),
    };
    println!(
        "{:6.2} {:6.1} {:4.0}% {}  {}  {} {:.2}{}",
        at as f32 / audio::TARGET_SR as f32,
        db,
        fill * 100.0,
        cells,
        cqt,
        p.chord,
        p.confidence,
        if fill < 0.9 { "   << app would not ask" } else { "" }
    );
}

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
         \x20     --tab [NAME] [ROOT] [QUALITY]\n\
         \x20                       an arpeggio phrase as tablature, written to stdout as SVG:\n\
         \x20                       degrees in the dots, one line per string. NAME matches a\n\
         \x20                       pattern by part of its name (default: all of them), ROOT and\n\
         \x20                       QUALITY default to A and m7. A fourth argument lights\n\
         \x20                       that many notes green, as playing them would; --frets\n\
         \x20                       writes fret numbers in the dots instead of degrees;\n\
         \x20                       --grip draws it as a chord box, --neck draws it on the\n\
         \x20                       neck with the note due ringed, and NAME may be a\n\
         \x20                       phrase typed out, e.g. --tab \"1 3 7\" C m7 --grip\n\
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
