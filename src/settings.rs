//! Settings that survive a restart.
//!
//! Deliberately a small separate file: this is the only state that outlives the
//! window, and a read failure must NEVER block startup. Every error - missing
//! file, corrupt JSON, no write permission - falls back to defaults.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Startup mode. The indices match `AppMode` and the UI buttons - reordering
/// them there requires a change here.
/// Index = the `AppMode` discriminant, NOT the on-screen button order (Fretboard
/// sits first in the UI but keeps index 4, so a saved setting keeps its meaning).
/// The range check below depends on this being complete - a missing entry would
/// silently reset the user's choice to Chords.
pub const MODE_NAMES: [&str; 5] = ["Chords", "Intervals", "Scales", "Arpeggios", "Fretboard"];

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

impl Default for Settings {
    fn default() -> Self {
        // Fretboard, language from the system, window at its design size
        Self {
            startup_mode: 4,
            language: 0,
            short_verdict: false,
            window_w: None,
            window_h: None,
        }
    }
}

impl Settings {
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
        assert_eq!(MODE_NAMES.len(), 5, "every AppMode needs an entry or its setting is dropped");
        assert_eq!(MODE_NAMES[1], "Intervals");
        assert_eq!(MODE_NAMES[2], "Scales");
        assert_eq!(MODE_NAMES[3], "Arpeggios");
        assert_eq!(MODE_NAMES[4], "Fretboard");
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
}
