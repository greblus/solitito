//! Settings that survive a restart.
//!
//! Deliberately a small separate file: this is the only state that outlives the
//! window, and a read failure must NEVER block startup. Every error - missing
//! file, corrupt JSON, no write permission - falls back to defaults.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Startup mode. The indices match `AppMode` and the UI buttons - reordering
/// them there requires a change here.
/// Index = the `AppMode` discriminant, NOT the on-screen button order (Fretboard
/// sits first in the UI but keeps index 4, so a saved setting keeps its meaning).
/// The range check below depends on this being complete - a missing entry would
/// silently reset the user's choice to Chords.
pub const MODE_NAMES: [&str; 6] =
    ["Chords", "Intervals", "Scales", "Arpeggios", "Fretboard", "Formulas"];

/// A formula worth coming back to: the set, and what the player calls it.
///
/// No key. A formula IS key-independent - that is the whole of the notation -
/// so keeping one per key would put the same exercise on the list twelve times
/// and leave the star dark on eleven of them. It arrives in whatever key is on
/// screen, which is what the options are for.
///
/// The mask is stored rather than the text, so a rename cannot break it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Favourite {
    pub name: String,
    pub mask: u16,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Settings {
    /// Which mode the app opens in. Fretboard by default.
    pub startup_mode: i32,
    /// UI language: 0 = from the system locale, 1 = Polish, 2 = English.
    #[serde(default)]
    pub language: i32,
    /// Advance on the first clear reading of the target instead of waiting for
    /// it to be held. Saved because it describes how someone plays, which does
    /// not change between sessions the way a noise gate might.
    #[serde(default)]
    pub short_verdict: bool,
    /// Note modes: require the notes one at a time. Off by default, because
    /// playing the whole chord and watching the intervals go past one by one is
    /// something the app can do and a monophonic tuner cannot.
    #[serde(default)]
    pub single_notes: bool,
    /// Let the model credit only what the onset head heard being struck. Off by
    /// default: it removes most of the credits that belong to the note before,
    /// but a note whose attack the head misses then has only the CQT branch.
    #[serde(default)]
    pub require_onset: bool,
    /// Whether the shuffle also reorders the chords in Intervals and Arpeggios.
    /// Off by default: shuffled intervals over the written progression is the
    /// musical half of the idea, shuffling both is the experiment.
    #[serde(default)]
    pub shuffle_chords: bool,
    /// The spectrum in the settings panel. Off by default - it is a calibration
    /// aid, and 48 bars redrawn with every frame are what the panel spends most
    /// of its time on.
    #[serde(default)]
    pub show_spectrum: bool,
    /// The model's reading of the chord in the main window. Off by default, and
    /// remembered like the spectrum beside it - it was not, and a restart put it
    /// out with nothing said.
    #[serde(default)]
    pub ai_debug: bool,
    /// Input device to open, by name. `None` means the system default. Saved
    /// because on Windows the right interface is not always the default one,
    /// and picking it again on every launch is the sort of thing that makes an
    /// app feel broken.
    #[serde(default)]
    pub audio_device: Option<String>,
    /// Which input of that device to listen on, 1-based. Saved alongside the
    /// device because the right socket is a property of the setup, not of the
    /// session.
    #[serde(default = "first_channel")]
    pub audio_channel: usize,
    /// Noise gate in dBFS, per input. Keyed by what the picker shows, because
    /// that is what the user chose: an interface with a hot output and a laptop
    /// microphone need thresholds tens of decibels apart, and having to find the
    /// setting again after every switch is what a saved value is for.
    #[serde(default)]
    pub gates: HashMap<String, f32>,
    /// Which formula exercise: 0 the formula in a key of its own (what the mode
    /// has always been), 1 the same formula planted on one chord, 2 planted on
    /// each chord of a tune in turn.
    #[serde(default)]
    pub formula_exercise: u8,
    /// What kind of placement to draw over a chord: 0 any, 1 one that spells the
    /// chord out, 2 one that colours it, 3 one outside it.
    #[serde(default)]
    pub formula_placement: u8,
    /// Name what falls outside the chord as a tension - b9, #11, b13 - instead
    /// of by its plain function. Off by default: the plain functions are the
    /// language the whole mode is written in, and the jazz names are a second
    /// vocabulary to learn.
    #[serde(default)]
    pub formula_jazz_names: bool,
    /// Draw the shape thumbnails under the chord name in Chords mode. Was a
    /// UI property alone and reset to on at every launch.
    #[serde(default = "yes")]
    pub show_diagrams: bool,
    /// Arpeggios: 0 the studies in a key of their own, 1 over the changes of a
    /// tune. Two exercises, not one setting of the same: a long study walked
    /// over a progression restarts mid-phrase at every chord.
    #[serde(default)]
    pub arp_exercise: usize,
    /// Over the changes: 0 up, 1 down, 2 alternating from down, 3 alternating
    /// from up.
    #[serde(default)]
    pub arp_direction: usize,
    /// The studies: which chord to read them over, indexing `ARP_QUALITIES`.
    #[serde(default)]
    pub arp_quality: usize,
    /// How the exercise is shown: 0 the line of degree names, 1 tablature, 2
    /// the neck itself. A new key rather than the old `tab_view`, which was a
    /// switch and is now a choice of three.
    ///
    /// Tablature by default: the line of names is the smaller thing on screen,
    /// but it says nothing about where the fingers go.
    #[serde(default = "tablature")]
    pub preview: usize,
    /// Intervals: take each grip where the fingers have least to move from the
    /// one before. Off, every chord is taken where the neck is drawn to offer
    /// it, which is how a shape is learned all over the neck rather than in one
    /// corner of it.
    #[serde(default = "yes")]
    pub voice_leading: bool,
    /// Write the fret number in each dot of the tablature instead of the
    /// degree. Off by default: the degrees are what the app teaches, and the
    /// frets are the crutch for reading a shape onto the neck.
    #[serde(default)]
    pub tab_frets: bool,
    /// Scales: finish the run on the root again, an octave up.
    #[serde(default)]
    pub scale_repeat_root: bool,
    /// Which shapes to draw. Both together is a legitimate answer - the row
    /// then holds ten thumbnails where four fit comfortably, which is for
    /// comparing them rather than playing from - and so is neither, which
    /// draws no shapes at all.
    #[serde(default = "yes")]
    pub show_full_shapes: bool,
    #[serde(default)]
    pub show_shell_shapes: bool,
    /// Formulas mode: how many notes each drawn formula has, the root included.
    #[serde(default = "five_notes")]
    pub formula_notes: usize,
    /// The key to read formulas in, e.g. "A", "Ab", "F#". Remembered even while
    /// `formula_random_key` is on, so unticking the box brings back the key that
    /// was typed rather than an empty field.
    #[serde(default = "key_of_c")]
    pub formula_key: String,
    /// Draw a fresh key for every formula instead of using `formula_key`.
    #[serde(default = "yes")]
    pub formula_random_key: bool,
    /// Functions every drawn formula has to contain, e.g. "b3 b7". Empty means
    /// no filter.
    #[serde(default)]
    pub formula_required: String,
    /// Show note names under the functions, and under the chords that fit. On
    /// by default: the functions are the exercise and the names are a crutch,
    /// but a crutch nobody asked for is worse than one they can put down.
    #[serde(default = "yes")]
    pub formula_note_names: bool,
    /// Show the nearest known scale under the formula, with the formula's own
    /// degrees picked out. On by default: it is what turns a drawn set into
    /// something a player can place.
    #[serde(default = "yes")]
    pub formula_show_similar: bool,
    /// Formulas: the set has to be played in the order it is written, lowest
    /// function first, instead of moving around inside it freely.
    #[serde(default)]
    pub formula_in_order: bool,
    /// Print a line for every function credited, and what was heard. Off by
    /// default: it is a developer's window on the judging, and on Windows a
    /// release build has no console to print it to.
    #[serde(default)]
    pub debug_console: bool,
    /// Show the chords that fit inside the formula, pointing at one lighting up
    /// the functions it is built from. On by default, for the same reason as the
    /// scale: it says what the set can be played over.
    #[serde(default = "yes")]
    pub formula_show_chords: bool,
    /// Formulas kept by name, newest last.
    #[serde(default)]
    pub favourites: Vec<Favourite>,
    /// Window size in PHYSICAL pixels, saved when the window closes. Physical
    /// rather than logical because the logical size depends on the scale factor
    /// the app itself sets from this - storing logical would make the size drift
    /// a little on every restart. `None` means never saved: open at the design
    /// size and let the window manager place it.
    #[serde(default)]
    pub window_w: Option<u32>,
    #[serde(default)]
    pub window_h: Option<u32>,
}

/// Anything outside this is a mistake, not a window: a saved size of 0 would
/// open an invisible window with no way to grab it, and a huge one could land
/// entirely off-screen. Both are unrecoverable without editing the file by hand.
const SANE_WINDOW_PX: std::ops::RangeInclusive<u32> = 200..=20_000;

/// Key for "system default" in `gates`. Empty because no backend names a device
/// with an empty string, so it cannot collide with a real one.
const DEFAULT_DEVICE_KEY: &str = "";

/// The slider's range. A value outside it comes from a hand-edited file and
/// would put the handle off the end of the track.
const SANE_GATE_DB: std::ops::RangeInclusive<f32> = -72.0..=0.0;

fn five_notes() -> usize {
    5
}

fn key_of_c() -> String {
    "C".to_string()
}

fn tablature() -> usize {
    1
}

fn yes() -> bool {
    true
}

/// Serde default for the channel. Files written while there was still a "mix
/// all channels" option hold 0, which is not a channel.
fn first_channel() -> usize {
    1
}

impl Default for Settings {
    fn default() -> Self {
        // Fretboard, language from the system, window at its design size
        Self {
            startup_mode: 4,
            language: 0,
            short_verdict: false,
            single_notes: false,
            require_onset: false,
            shuffle_chords: false,
            show_spectrum: false,
            ai_debug: false,
            audio_device: None,
            audio_channel: 1,
            gates: HashMap::new(),
            formula_jazz_names: false,
            scale_repeat_root: false,
            show_diagrams: true,
            arp_exercise: 0,
            arp_direction: 0,
            arp_quality: 0,
            preview: 1,
            voice_leading: true,
            tab_frets: false,
            show_full_shapes: true,
            show_shell_shapes: false,
            formula_exercise: 0,
            formula_placement: 0,
            formula_notes: 5,
            formula_key: "C".to_string(),
            formula_random_key: true,
            formula_required: String::new(),
            formula_note_names: true,
            formula_show_similar: true,
            formula_in_order: false,
            debug_console: false,
            formula_show_chords: true,
            favourites: Vec::new(),
            window_w: None,
            window_h: None,
        }
    }
}

impl Settings {
    /// Saved gate for a device, if there is one. `None` means "never set here",
    /// and the caller keeps its own default rather than guessing.
    pub fn gate_for(&self, device: Option<&str>) -> Option<f32> {
        self.gates
            .get(device.unwrap_or(DEFAULT_DEVICE_KEY))
            .copied()
            .filter(|db| SANE_GATE_DB.contains(db))
    }

    /// Remembers the gate for a device. Returns whether anything changed, so the
    /// caller can skip writing the file when nothing moved.
    pub fn set_gate(&mut self, device: Option<&str>, db: f32) -> bool {
        if !SANE_GATE_DB.contains(&db) {
            return false;
        }
        let key = device.unwrap_or(DEFAULT_DEVICE_KEY).to_string();
        if self.gates.get(&key).is_some_and(|old| (old - db).abs() < 0.01) {
            return false;
        }
        self.gates.insert(key, db);
        true
    }

    pub fn load() -> Self {
        let Some(path) = config_path() else {
            return Self::default();
        };
        match std::fs::read_to_string(&path) {
            Ok(txt) => match serde_json::from_str::<Settings>(&txt) {
                Ok(mut s) => {
                    // The file may come from another version or have been edited
                    // by hand; an out-of-range index would break the UI.
                    if !(0..MODE_NAMES.len() as i32).contains(&s.startup_mode) {
                        s.startup_mode = 0;
                    }
                    if !(0..3).contains(&s.language) {
                        s.language = 0;
                    }
                    s.clamp_window();
                    // 0 meant "mix all" before that option was dropped.
                    if s.audio_channel == 0 {
                        s.audio_channel = 1;
                    }
                    s.clamp_formulas();
                    println!("⚙️  Settings from {}", path.display());
                    s
                }
                Err(e) => {
                    eprintln!("⚠️  {} is corrupt ({e}) - using defaults.", path.display());
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    /// Drops a saved window size that could not be honoured. Both axes go
    /// together: half a size is not a size, and keeping one would open a window
    /// with the saved height and a default width, which looks like a bug.
    fn clamp_window(&mut self) {
        let ok = |v: Option<u32>| v.is_some_and(|v| SANE_WINDOW_PX.contains(&v));
        if !(ok(self.window_w) && ok(self.window_h)) {
            self.window_w = None;
            self.window_h = None;
        }
    }

    /// Brings the Formulas settings back into range. A hand-edited file could
    /// otherwise ask for a nine-note formula containing eleven functions, and
    /// the draw would come back empty every time with nothing to explain it.
    fn clamp_formulas(&mut self) {
        self.formula_notes = self.formula_notes.clamp(1, 12);
        // A file written by a newer build, or by hand: an exercise nobody
        // implements would leave the mode drawing nothing at all.
        if self.formula_exercise > 2 {
            self.formula_exercise = 0;
        }
        if self.formula_placement > 3 {
            self.formula_placement = 0;
        }
        if crate::formulas::parse_key(&self.formula_key).is_none() {
            self.formula_key = key_of_c();
        }
        match crate::formulas::parse(&self.formula_required) {
            // The filter cannot ask for more functions than the formula holds.
            Some(m) if m.count_ones() as usize <= self.formula_notes => {}
            _ if self.formula_required.trim().is_empty() => {}
            _ => self.formula_required = String::new(),
        }
    }

    pub fn save(&self) {
        let Some(path) = config_path() else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match serde_json::to_string_pretty(self) {
            Ok(txt) => {
                if let Err(e) = std::fs::write(&path, txt) {
                    eprintln!("⚠️  Could not write settings to {}: {e}", path.display());
                }
            }
            Err(e) => eprintln!("⚠️  Could not serialise settings: {e}"),
        }
    }
}

/// `$XDG_CONFIG_HOME/solitito/settings.json`, falling back to `$HOME/.config`
/// and `%APPDATA%` on Windows.
fn config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .or_else(|| std::env::var_os("APPDATA").map(PathBuf::from))?;
    Some(base.join("solitito").join("settings.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_chords() {
        assert_eq!(Settings::default().startup_mode, 4, "Fretboard is the default mode");
        assert_eq!(MODE_NAMES[0], "Chords");
    }

    #[test]
    fn mode_names_match_appmode() {
        // The order is a contract with `AppMode` in state.rs and the buttons in
        // appwindow.slint. A mismatch means starting in the wrong mode.
        assert_eq!(MODE_NAMES.len(), 6, "every AppMode needs an entry or its setting is dropped");
        assert_eq!(MODE_NAMES[1], "Intervals");
        assert_eq!(MODE_NAMES[2], "Scales");
        assert_eq!(MODE_NAMES[3], "Arpeggios");
        assert_eq!(MODE_NAMES[4], "Fretboard");
        assert_eq!(MODE_NAMES[5], "Formulas");
    }

    #[test]
    fn corrupt_json_does_not_break_startup() {
        // Serde must reject garbage; the caller then gets defaults.
        assert!(serde_json::from_str::<Settings>("{ not json at all").is_err());
        assert!(serde_json::from_str::<Settings>("{}").is_err(), "a missing field is an error");
        let ok: Settings = serde_json::from_str(r#"{"startup_mode":2}"#).unwrap();
        assert_eq!(ok.startup_mode, 2);
    }

    #[test]
    fn short_verdict_survives_a_round_trip() {
        let s = Settings { short_verdict: true, ..Settings::default() };
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert!(back.short_verdict, "the option did not survive being saved");
        // And a file written before it existed must still load, with it off.
        let old: Settings = serde_json::from_str(r#"{"startup_mode":4,"language":1}"#).unwrap();
        assert!(!old.short_verdict);
    }

    #[test]
    fn single_notes_survives_a_round_trip_and_defaults_off() {
        assert!(!Settings::default().single_notes, "the strict mode must not be the default");
        let s = Settings { single_notes: true, ..Settings::default() };
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert!(back.single_notes, "the option did not survive being saved");
        // A file written before the option existed still loads, with it off.
        let old: Settings = serde_json::from_str(r#"{"startup_mode":4,"language":1}"#).unwrap();
        assert!(!old.single_notes);
    }

    #[test]
    fn the_gate_is_remembered_per_device() {
        let mut s = Settings::default();
        assert_eq!(s.gate_for(None), None, "nothing saved yet is not a threshold of 0 dB");

        assert!(s.set_gate(None, -34.0), "the system default should be storable");
        assert!(s.set_gate(Some("sysdefault:CARD=U192k"), -52.0));
        assert_eq!(s.gate_for(None), Some(-34.0));
        assert_eq!(s.gate_for(Some("sysdefault:CARD=U192k")), Some(-52.0));
        assert_eq!(s.gate_for(Some("some other card")), None, "one device answered for another");

        // Writing the same value again is not a change, so the file is left alone.
        assert!(!s.set_gate(None, -34.0));

        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back.gate_for(Some("sysdefault:CARD=U192k")), Some(-52.0));

        // A hand-edited file cannot put the slider handle off the end of its track.
        assert!(!s.set_gate(None, -900.0));
        let mut mad = Settings::default();
        mad.gates.insert(String::new(), 40.0);
        assert_eq!(mad.gate_for(None), None, "an impossible value was handed back");
    }

    #[test]
    fn settings_without_gates_still_load() {
        let old: Settings = serde_json::from_str(r#"{"startup_mode":4,"language":1}"#).unwrap();
        assert!(old.gates.is_empty());
        assert_eq!(old.gate_for(None), None);
    }

    #[test]
    fn shuffle_chords_survives_a_round_trip_and_defaults_off() {
        assert!(!Settings::default().shuffle_chords, "the progression should stay written by default");
        let s = Settings { shuffle_chords: true, ..Settings::default() };
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert!(back.shuffle_chords);
        let old: Settings = serde_json::from_str(r#"{"startup_mode":4,"language":1}"#).unwrap();
        assert!(!old.shuffle_chords);
    }

    #[test]
    fn formula_settings_default_and_survive_a_round_trip() {
        let d = Settings::default();
        assert_eq!(d.formula_notes, 5);
        assert_eq!(d.formula_key, "C");
        assert!(d.formula_random_key, "a drawn key needs no typing to get going");
        assert!(d.formula_note_names, "the names are there until someone puts them down");

        let s = Settings {
            formula_notes: 7,
            formula_key: "Eb".into(),
            formula_random_key: false,
            formula_required: "b3 b7".into(),
            formula_note_names: true,
            ..Settings::default()
        };
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back.formula_key, "Eb");
        assert_eq!(back.formula_required, "b3 b7");
        assert!(!back.formula_random_key);

        // A file written before the mode existed still loads.
        let old: Settings = serde_json::from_str(r#"{"startup_mode":4,"language":1}"#).unwrap();
        assert_eq!(old.formula_notes, 5);
        assert_eq!(old.formula_key, "C");
    }

    #[test]
    fn a_hand_edited_formula_setting_is_brought_back_into_range() {
        let mut s = Settings {
            formula_notes: 99,
            formula_key: "H".into(),
            formula_required: "1 b3 5 b7".into(),
            ..Settings::default()
        };
        s.clamp_formulas();
        assert_eq!(s.formula_notes, 12);
        assert_eq!(s.formula_key, "C", "H is not a key here");
        assert_eq!(s.formula_required, "1 b3 5 b7", "four functions fit in twelve notes");

        // A filter wider than the formula would never draw anything.
        let mut tight = Settings { formula_notes: 3, formula_required: "1 b3 5 b7".into(), ..Settings::default() };
        tight.clamp_formulas();
        assert_eq!(tight.formula_required, "", "an unsatisfiable filter is dropped");
    }

    #[test]
    fn window_size_survives_a_round_trip() {
        let s = Settings { window_w: Some(900), window_h: Some(1200), ..Settings::default() };
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!((back.window_w, back.window_h), (Some(900), Some(1200)));
    }

    #[test]
    fn settings_from_before_this_feature_still_load() {
        // Files written by 0.3.0 have no window keys at all. They must open at
        // the design size, not fail to parse and drop the user's mode as well.
        let old: Settings = serde_json::from_str(r#"{"startup_mode":4,"language":1}"#).unwrap();
        assert_eq!(old.startup_mode, 4);
        assert_eq!((old.window_w, old.window_h), (None, None));
    }

    #[test]
    fn an_unusable_window_size_is_dropped_whole() {
        // A zero width would open a window with nothing to grab. Half a saved
        // size is no better than none, so both axes go together.
        for (w, h) in [(Some(0), Some(800)), (Some(900), None), (Some(90_000), Some(800))] {
            let mut s = Settings { window_w: w, window_h: h, ..Settings::default() };
            s.clamp_window();
            assert_eq!((s.window_w, s.window_h), (None, None), "{w:?}x{h:?} should be dropped");
        }
        let mut ok = Settings { window_w: Some(900), window_h: Some(1200), ..Settings::default() };
        ok.clamp_window();
        assert_eq!((ok.window_w, ok.window_h), (Some(900), Some(1200)), "a sane size is kept");
    }

    #[test]
    fn out_of_range_index_falls_back_to_chords() {
        let mut s: Settings = serde_json::from_str(r#"{"startup_mode":99}"#).unwrap();
        if !(0..MODE_NAMES.len() as i32).contains(&s.startup_mode) {
            s.startup_mode = 0;
        }
        assert_eq!(s.startup_mode, 0);
    }

    #[test]
    fn favourites_survive_a_round_trip_and_an_old_file_has_none() {
        let s = Settings {
            favourites: vec![Favourite {
                name: "kranciasta".into(),
                mask: 0b1000_1001_0001,
            }],
            ..Settings::default()
        };
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back.favourites, s.favourites, "a kept formula did not survive");

        let old: Settings = serde_json::from_str(r#"{"startup_mode":4,"language":1}"#).unwrap();
        assert!(old.favourites.is_empty(), "a file from before them must read clean");
    }
}
