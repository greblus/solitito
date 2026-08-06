//! UI translations.
//!
//! No gettext on purpose: with two dozen strings, a system dependency (libintl),
//! `.mo` catalogues and installing them alongside the binary cost more than they
//! give. The strings are compiled in, so the app stays a single file and there is
//! no way to lose a translation.
//!
//! Slint receives them through the `Tr` global, filled once at startup.

/// Detected or forced UI language.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Lang {
    En,
    Pl,
}

impl Lang {
    /// 0 = auto (from the system), 1 = Polish, 2 = English.
    pub fn from_setting(v: i32) -> Self {
        match v {
            1 => Lang::Pl,
            2 => Lang::En,
            _ => Lang::detect(),
        }
    }

    /// From the system locale.
    ///
    /// POSIX order: `LC_ALL` beats `LC_MESSAGES` beats `LANG`. Windows usually
    /// has none of those, hence `LANGUAGE` as well, which some environments set;
    /// English is the last resort.
    pub fn detect() -> Self {
        for var in ["LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"] {
            if let Ok(v) = std::env::var(var) {
                if let Some(l) = Self::from_locale(&v) {
                    return l;
                }
            }
        }
        Lang::En
    }

    /// Extracts the language code from `pl_PL.UTF-8` or `pl`.
    pub fn from_locale(s: &str) -> Option<Self> {
        let code = s.trim().to_ascii_lowercase();
        if code.is_empty() || code == "c" || code == "posix" {
            return None;
        }
        // only the part before '_' / '-' / '.' matters
        let base = code
            .split(|c| c == '_' || c == '-' || c == '.' || c == ':')
            .next()
            .unwrap_or("");
        match base {
            "pl" => Some(Lang::Pl),
            "" => None,
            _ => Some(Lang::En),
        }
    }
}

/// Every UI string. The fields match the properties of the `Tr` global in
/// appwindow.slint - adding a string means changing both places.
pub struct Strings {
    pub chords: &'static str,
    pub intervals: &'static str,
    pub scales: &'static str,
    pub arpeggios: &'static str,
    pub settings: &'static str,
    pub close: &'static str,
    pub settings_title: &'static str,
    pub audio_calibration: &'static str,
    pub noise_gate: &'static str,
    pub gate_hint: &'static str,
    pub bass_boost: &'static str,
    pub lock_quality: &'static str,
    pub random_order: &'static str,
    pub random_hint: &'static str,
    /// `{}` is replaced with the string name (E/A/D/G).
    pub start_from: &'static str,
    pub startup_mode: &'static str,
    pub chord_confidence: &'static str,
    pub note_threshold: &'static str,
    pub hold_time: &'static str,
    pub show_debug: &'static str,
    pub ai_prediction: &'static str,
    pub intervals_label: &'static str,
    pub intervals_hint: &'static str,
    pub intervals_placeholder: &'static str,
    pub next: &'static str,
    pub no_data: &'static str,
    pub language: &'static str,
    pub lang_auto: &'static str,
}

pub const EN: Strings = Strings {
    chords: "Chords",
    intervals: "Intervals",
    scales: "Scales",
    arpeggios: "Arpeggios",
    settings: "⚙ Settings",
    close: "Close",
    settings_title: "Settings & Calibration",
    audio_calibration: "Audio Calibration",
    noise_gate: "Noise gate: ",
    gate_hint: "Bar = current level, red line = threshold. Set it just above the noise with strings untouched.",
    bass_boost: "Bass Boost",
    lock_quality: "Lock chord quality until new attack",
    random_order: "Random order",
    random_hint: "Also shuffles the tones inside each chord.",
    start_from: "start from the {} string",
    startup_mode: "Startup mode: ",
    chord_confidence: "Chord confidence: ",
    note_threshold: "Note threshold: ",
    hold_time: "Hold time: ",
    show_debug: "Show AI Debug in Main Window",
    ai_prediction: "AI Prediction:  ",
    intervals_label: "Intervals: ",
    intervals_hint: "(Input specific intervals to practice, e.g. '1 3' for shell chords)",
    intervals_placeholder: "e.g. 1 3 5",
    next: "Next: ",
    no_data: "No Data",
    language: "Language: ",
    lang_auto: "Auto",
};

pub const PL: Strings = Strings {
    chords: "Akordy",
    intervals: "Interwały",
    scales: "Skale",
    arpeggios: "Arpeggia",
    settings: "⚙ Ustawienia",
    close: "Zamknij",
    settings_title: "Ustawienia i kalibracja",
    audio_calibration: "Kalibracja dźwięku",
    noise_gate: "Bramka szumu: ",
    gate_hint: "Pasek = bieżący poziom, czerwona kreska = próg. Ustaw go tuż nad szumem przy nieruszanych strunach.",
    bass_boost: "Podbicie basu",
    lock_quality: "Trzymaj jakość akordu do następnego szarpnięcia",
    random_order: "Losowa kolejność",
    random_hint: "W trybach dźwiękowych miesza też składniki akordu.",
    start_from: "zagraj od struny {}",
    startup_mode: "Tryb po uruchomieniu: ",
    chord_confidence: "Pewność akordu: ",
    note_threshold: "Próg dźwięku: ",
    hold_time: "Czas przytrzymania: ",
    show_debug: "Pokaż podgląd modelu w oknie głównym",
    ai_prediction: "Rozpoznano:  ",
    intervals_label: "Interwały: ",
    intervals_hint: "(Podaj interwały do ćwiczenia, np. '1 3' dla chwytów szkieletowych)",
    intervals_placeholder: "np. 1 3 5",
    next: "Następny: ",
    no_data: "Brak danych",
    language: "Język: ",
    lang_auto: "Automatycznie",
};

pub fn strings(lang: Lang) -> &'static Strings {
    match lang {
        Lang::Pl => &PL,
        Lang::En => &EN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_common_locales() {
        assert_eq!(Lang::from_locale("pl_PL.UTF-8"), Some(Lang::Pl));
        assert_eq!(Lang::from_locale("pl"), Some(Lang::Pl));
        assert_eq!(Lang::from_locale("pl-PL"), Some(Lang::Pl));
        assert_eq!(Lang::from_locale("en_US.UTF-8"), Some(Lang::En));
        assert_eq!(Lang::from_locale("de_DE"), Some(Lang::En), "unknown language -> English");
    }

    #[test]
    fn uninformative_locales_do_not_decide() {
        // "C" and "POSIX" say nothing about user preference; treating them as
        // English would mask the next variable in the order.
        assert_eq!(Lang::from_locale("C"), None);
        assert_eq!(Lang::from_locale("POSIX"), None);
        assert_eq!(Lang::from_locale(""), None);
        assert_eq!(Lang::from_locale("   "), None);
    }

    #[test]
    fn forcing_beats_the_system() {
        assert_eq!(Lang::from_setting(1), Lang::Pl);
        assert_eq!(Lang::from_setting(2), Lang::En);
    }

    #[test]
    fn both_sets_are_complete() {
        // An empty string means someone added a field and forgot to translate.
        for (name, en, pl) in [
            ("chords", EN.chords, PL.chords),
            ("settings_title", EN.settings_title, PL.settings_title),
            ("gate_hint", EN.gate_hint, PL.gate_hint),
            ("lock_quality", EN.lock_quality, PL.lock_quality),
            ("random_order", EN.random_order, PL.random_order),
            ("random_hint", EN.random_hint, PL.random_hint),
            ("start_from", EN.start_from, PL.start_from),
            ("intervals_hint", EN.intervals_hint, PL.intervals_hint),
            ("no_data", EN.no_data, PL.no_data),
        ] {
            assert!(!en.is_empty(), "{name}: English missing");
            assert!(!pl.is_empty(), "{name}: Polish missing");
            assert_ne!(en, pl, "{name}: Polish identical to English");
        }
    }
}
