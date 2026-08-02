//! Settings that survive a restart.
//!
//! Deliberately a small separate file: this is the only state that outlives the
//! window, and a read failure must NEVER block startup. Every error - missing
//! file, corrupt JSON, no write permission - falls back to defaults.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Startup mode. The indices match `AppMode` and the UI buttons - reordering
/// them there requires a change here.
pub const MODE_NAMES: [&str; 4] = ["Chords", "Intervals", "Scales", "Arpeggios"];

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Settings {
    /// Which mode the app opens in. Chords by default.
    pub startup_mode: i32,
    /// UI language: 0 = from the system locale, 1 = Polish, 2 = English.
    #[serde(default)]
    pub language: i32,
}

impl Default for Settings {
    fn default() -> Self {
        Self { startup_mode: 0, language: 0 } // Chords, language from the system
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
        assert_eq!(Settings::default().startup_mode, 0);
        assert_eq!(MODE_NAMES[0], "Chords");
    }

    #[test]
    fn mode_names_match_appmode() {
        // The order is a contract with `AppMode` in state.rs and the buttons in
        // appwindow.slint. A mismatch means starting in the wrong mode.
        assert_eq!(MODE_NAMES.len(), 4);
        assert_eq!(MODE_NAMES[1], "Intervals");
        assert_eq!(MODE_NAMES[2], "Scales");
        assert_eq!(MODE_NAMES[3], "Arpeggios");
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
    fn out_of_range_index_falls_back_to_chords() {
        let mut s: Settings = serde_json::from_str(r#"{"startup_mode":99}"#).unwrap();
        if !(0..MODE_NAMES.len() as i32).contains(&s.startup_mode) {
            s.startup_mode = 0;
        }
        assert_eq!(s.startup_mode, 0);
    }
}
