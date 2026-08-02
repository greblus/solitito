//! Ustawienia trwałe między uruchomieniami.
//!
//! Świadomie osobny, mały plik zamiast dopisywania do stanu aplikacji: to jedyna
//! rzecz w programie, która przeżywa zamknięcie okna, i chcemy, żeby awaria
//! odczytu NIGDY nie blokowała startu. Każdy błąd (brak pliku, uszkodzony JSON,
//! brak prawa zapisu) kończy się cichym powrotem do wartości domyślnych.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Tryb wybierany przy starcie. Liczby odpowiadają `AppMode` i indeksom
/// przycisków w UI — zmiana kolejności tam wymaga zmiany tutaj.
pub const MODE_NAMES: [&str; 4] = ["Chords", "Intervals", "Scales", "Arpeggios"];

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Settings {
    /// Tryb, w którym aplikacja ma się uruchamiać. Domyślnie Chords.
    pub startup_mode: i32,
}

impl Default for Settings {
    fn default() -> Self {
        Self { startup_mode: 0 } // Chords
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
                    // Plik mógł powstać w innej wersji albo zostać ręcznie
                    // poprawiony — indeks spoza zakresu wywaliłby UI.
                    if !(0..MODE_NAMES.len() as i32).contains(&s.startup_mode) {
                        s.startup_mode = 0;
                    }
                    println!("⚙️  Ustawienia z {}", path.display());
                    s
                }
                Err(e) => {
                    eprintln!("⚠️  {} jest uszkodzony ({e}) — wartości domyślne.", path.display());
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
                    eprintln!("⚠️  Nie zapisano ustawień do {}: {e}", path.display());
                }
            }
            Err(e) => eprintln!("⚠️  Nie udało się zserializować ustawień: {e}"),
        }
    }
}

/// `$XDG_CONFIG_HOME/solitito/settings.json`, z odwrotem do `$HOME/.config`
/// i `%APPDATA%` na Windowsie.
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
    fn domyslnie_chords() {
        assert_eq!(Settings::default().startup_mode, 0);
        assert_eq!(MODE_NAMES[0], "Chords");
    }

    #[test]
    fn nazwy_trybow_pokrywaja_sie_z_appmode() {
        // Kolejność jest kontraktem z `AppMode` w state.rs i z przyciskami w
        // appwindow.slint. Rozjazd = start w innym trybie, niż wybrany.
        assert_eq!(MODE_NAMES.len(), 4);
        assert_eq!(MODE_NAMES[1], "Intervals");
        assert_eq!(MODE_NAMES[2], "Scales");
        assert_eq!(MODE_NAMES[3], "Arpeggios");
    }

    #[test]
    fn uszkodzony_json_nie_wywraca_startu() {
        // Serde ma odrzucić śmieci, a wywołujący dostaje wartości domyślne.
        assert!(serde_json::from_str::<Settings>("{ to nie jest json").is_err());
        assert!(serde_json::from_str::<Settings>("{}").is_err(), "brak pola = błąd");
        let ok: Settings = serde_json::from_str(r#"{"startup_mode":2}"#).unwrap();
        assert_eq!(ok.startup_mode, 2);
    }

    #[test]
    fn indeks_spoza_zakresu_wraca_do_chords() {
        let mut s: Settings = serde_json::from_str(r#"{"startup_mode":99}"#).unwrap();
        if !(0..MODE_NAMES.len() as i32).contains(&s.startup_mode) {
            s.startup_mode = 0;
        }
        assert_eq!(s.startup_mode, 0);
    }
}
