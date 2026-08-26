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
    pub tab_audio: &'static str,
    pub tab_practice: &'static str,
    pub tab_general: &'static str,
    pub tab_app: &'static str,
    pub audio_calibration: &'static str,
    pub audio_device: &'static str,
    pub audio_channel: &'static str,
    pub audio_default: &'static str,
    pub audio_one: &'static str,
    /// Shown when the chosen device could not be opened.
    pub audio_busy: &'static str,
    pub noise_gate: &'static str,
    pub gate_hint: &'static str,
    pub bass_boost: &'static str,
    pub lock_quality: &'static str,
    pub short_verdict: &'static str,
    pub short_verdict_hint: &'static str,
    pub single_notes: &'static str,
    pub single_notes_hint: &'static str,
    pub require_onset: &'static str,
    pub require_onset_hint: &'static str,
    pub in_order: &'static str,
    pub in_order_hint: &'static str,
    pub debug_console: &'static str,
    pub shuffle_chords: &'static str,
    pub shuffle_chords_hint: &'static str,
    pub random_order: &'static str,
    pub show_diagrams: &'static str,
    pub repeat_root: &'static str,
    pub repeat_root_hint: &'static str,
    pub shapes_kind: &'static str,
    pub shapes_full: &'static str,
    pub shapes_shell: &'static str,
    pub shapes_no_shell: &'static str,
    pub shapes_no_shell_end: &'static str,
    pub shapes_substitute: &'static str,
    // Combo labels. These used to be hard-coded English in main.rs, bypassing
    // the whole translation table.
    pub select_song: &'static str,
    pub select_scale: &'static str,
    pub pattern: &'static str,
    pub key_root: &'static str,
    pub random_hint: &'static str,
    /// `{}` is replaced with the string name (E/A/D/G).
    pub start_from: &'static str,
    // --- fretboard trainer ---
    pub fretboard: &'static str,
    pub formulas: &'static str,
    pub formula_key_line: &'static str,
    pub formula_similar: &'static str,
    pub formula_notes: &'static str,
    pub formula_key: &'static str,
    pub formula_random: &'static str,
    pub formula_required: &'static str,
    pub formula_required_hint: &'static str,
    pub formula_note_names: &'static str,
    pub formula_similar_opt: &'static str,
    pub formula_another: &'static str,
    pub formula_free: &'static str,
    pub exercise: &'static str,
    pub exercise_key: &'static str,
    pub exercise_chord: &'static str,
    pub exercise_changes: &'static str,
    pub placement: &'static str,
    /// The same word for the view, where it reads as a sentence rather than as
    /// a settings label.
    pub placement_from: &'static str,
    pub next_chord: &'static str,
    /// "1 OF 4 chord tones".
    pub of_count: &'static str,
    pub placement_any: &'static str,
    pub placement_defines: &'static str,
    pub placement_colours: &'static str,
    pub placement_outside: &'static str,
    /// The line that reads the formula against the chord under it.
    pub against_chord: &'static str,
    pub jazz_names: &'static str,
    pub chord_tones: &'static str,
    pub fav_name: &'static str,
    pub fav_pick: &'static str,
    pub fav_search: &'static str,
    pub fav_hint: &'static str,
    pub fav_add: &'static str,
    pub fav_remove: &'static str,
    pub formula_chords: &'static str,
    pub formula_chords_opt: &'static str,
    pub startup_mode: &'static str,
    pub chord_confidence: &'static str,
    pub note_threshold: &'static str,
    pub hold_time: &'static str,
    pub show_debug: &'static str,
    pub show_spectrum: &'static str,
    pub ai_prediction: &'static str,
    pub intervals_label: &'static str,
    pub intervals_hint: &'static str,
    pub intervals_placeholder: &'static str,
    pub next: &'static str,
    pub previous: &'static str,
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
    tab_audio: "Audio",
    tab_practice: "Practice",
    tab_general: "General",
    tab_app: "App",
    audio_calibration: "Audio Calibration",
    audio_device: "Input: ",
    audio_channel: "Channel: ",
    audio_default: "System default",
    audio_one: "Channel",
    audio_busy: "chosen input unavailable - busy or unplugged; listening on the default instead",
    noise_gate: "Noise gate: ",
    gate_hint: "Just above the noise, strings untouched.",
    bass_boost: "Bass Boost",
    lock_quality: "Lock chord quality until new attack",
    short_verdict: "Judge short strums on the attack",
    short_verdict_hint: "One clear reading after the attack counts; the decay cannot undo it.",
    single_notes: "Play the notes one at a time",
    single_notes_hint: "On: each note played on its own, a repeat needs a fresh attack.",
    require_onset: "Credit only what was struck",
    require_onset_hint: "The model may credit a note only where it also heard an attack. Removes most of the credits that belong to the previous note, still ringing inside the model's window.",
    in_order: "Play the notes in order",
    in_order_hint: "Lowest function first. Off: any of them, in any order.",
    debug_console: "Console debug",
    shuffle_chords: "Shuffle the chords as well",
    shuffle_chords_hint: "With the shuffle on. Off: the progression stays as written.",
    random_order: "Random order",
    show_diagrams: "Show chord shapes",
    repeat_root: "End on the root again",
    repeat_root_hint: "Scales only: 1 2 3 4 5 6 7 1, the last one an octave up. It is a step of its own and has to be played.",
    shapes_kind: "Shapes: ",
    shapes_full: "full chords",
    shapes_shell: "shell voicings",
    shapes_no_shell: "dim7 has no shell — the same grip is also ",
    shapes_no_shell_end: ", each without its root",
    shapes_substitute: "substitute: the m7 shell",
    select_song: "Song:",
    select_scale: "Scale:",
    pattern: "Pattern:",
    key_root: "Key:",
    random_hint: "In the note modes: the tones inside each chord.",
    start_from: "start from the {} string",
    fretboard: "Fretboard",
    formulas: "Formulas",
    formula_key_line: "Key",
    formula_similar: "Similar to:",
    formula_notes: "Notes in a formula: ",
    formula_key: "Key: ",
    formula_random: "random",
    formula_required: "Must contain: ",
    formula_required_hint: "Functions every formula has to have, e.g. b3 b7. Empty draws from all of them.",
    formula_note_names: "Show note and chord names",
    formula_similar_opt: "Show the nearest scale",
    formula_another: "another formula",
    formula_free: "Your turn! Improvise!",
    exercise: "Exercise: ",
    exercise_key: "Formula in a key",
    exercise_chord: "Over a chord",
    exercise_changes: "Over the changes",
    placement: "Placement: ",
    placement_from: "Placed from ",
    next_chord: "next chord",
    of_count: "of",
    placement_any: "any",
    placement_defines: "spells the chord out",
    placement_colours: "colours it",
    placement_outside: "outside it",
    against_chord: "Relative to ",
    jazz_names: "Name tensions the jazz way (b9, #11, b13)",
    chord_tones: "chord tones",
    fav_name: "name it",
    fav_pick: "Favourites: ",
    fav_search: "search",
    fav_hint: "The star keeps the formula on screen. Picking one here draws it.",
    fav_add: "Keep this formula",
    fav_remove: "Drop it from the favourites",
    formula_chords: "Chords inside the formula:",
    formula_chords_opt: "Show the chords that fit",
    startup_mode: "Startup mode: ",
    chord_confidence: "Chord confidence: ",
    note_threshold: "Note threshold: ",
    hold_time: "Hold time: ",
    show_debug: "Show AI Debug in Main Window",
    show_spectrum: "Show spectrum",
    ai_prediction: "AI Prediction:  ",
    intervals_label: "Intervals: ",
    intervals_hint: "Degrees, e.g. 1 3 5. Octave marks: 5' up, 5, down.",
    intervals_placeholder: "e.g. 1 3 5",
    next: "Next",
    previous: "Previous",
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
    tab_audio: "Dźwięk",
    tab_practice: "Ćwiczenia",
    tab_general: "Ogólne",
    tab_app: "Program",
    audio_calibration: "Kalibracja dźwięku",
    audio_device: "Wejście: ",
    audio_channel: "Kanał: ",
    audio_default: "Domyślne systemowe",
    audio_one: "Kanał",
    audio_busy: "wybrane wejście niedostępne - zajęte lub odłączone; słucham domyślnego",
    noise_gate: "Bramka szumu: ",
    gate_hint: "Tuż nad szumem, przy nieruszanych strunach.",
    bass_boost: "Podbicie basu",
    lock_quality: "Trzymaj jakość akordu do następnego szarpnięcia",
    short_verdict: "Oceniaj krótkie szarpnięcia po ataku",
    short_verdict_hint: "Liczy się czysty odczyt tuż po ataku; wybrzmiewanie go nie cofnie.",
    single_notes: "Graj dźwięki pojedynczo",
    single_notes_hint: "Włączone: każdy dźwięk osobno, powtórzony wymaga nowego szarpnięcia.",
    require_onset: "Zaliczaj tylko to, co uderzone",
    require_onset_hint: "Model może zaliczyć dźwięk tylko tam, gdzie usłyszał też atak. Odsiewa większość zaliczeń należących do poprzedniego dźwięku, wciąż brzmiącego w oknie modelu.",
    in_order: "Graj dźwięki po kolei",
    in_order_hint: "Od najniższej funkcji. Wyłączone: dowolna, w dowolnej kolejności.",
    debug_console: "Debug na konsoli",
    shuffle_chords: "Losuj także kolejność akordów",
    shuffle_chords_hint: "Przy włączonym losowaniu. Wyłączone: progresja jak zapisana.",
    random_order: "Losowa kolejność",
    show_diagrams: "Wyświetlaj schematy akordów",
    repeat_root: "Kończ powtórzoną prymą",
    repeat_root_hint: "Tylko Skale: 1 2 3 4 5 6 7 1, ostatnia oktawę wyżej. To osobny krok i trzeba go zagrać.",
    shapes_kind: "Schematy: ",
    shapes_full: "pełne akordy",
    shapes_shell: "shell voicings",
    shapes_no_shell: "dim7 nie ma shella — ten sam chwyt to także ",
    shapes_no_shell_end: ", każdy bez prymy",
    shapes_substitute: "substytut: shell m7",
    select_song: "Utwór:",
    select_scale: "Skala:",
    pattern: "Wzorzec:",
    key_root: "Tonacja:",
    random_hint: "W trybach nutowych: składniki akordu.",
    start_from: "zagraj od struny {}",
    fretboard: "Gryf",
    formulas: "Formuły",
    formula_key_line: "Tonacja",
    formula_similar: "Podobieństwo:",
    formula_notes: "Dźwięków w formule: ",
    formula_key: "Tonacja: ",
    formula_random: "losowa",
    formula_required: "Musi zawierać: ",
    formula_required_hint: "Funkcje obecne w każdej formule, np. b3 b7. Puste losuje ze wszystkich.",
    formula_note_names: "Pokazuj nazwy dźwięków i akordów",
    formula_similar_opt: "Pokazuj najbliższą skalę",
    formula_another: "nowa formuła",
    formula_free: "Twoja kolej! Improwizuj!",
    exercise: "Ćwiczenie: ",
    exercise_key: "Formuła w tonacji",
    exercise_chord: "Nad akordem",
    exercise_changes: "Nad standardem",
    placement: "Nałożenie: ",
    placement_from: "Nałożenie od ",
    next_chord: "następny akord",
    of_count: "z",
    placement_any: "dowolne",
    placement_defines: "pasuje do akordu",
    placement_colours: "barwi akord",
    placement_outside: "na zewnątrz",
    against_chord: "Względem ",
    jazz_names: "Nazwy napięć po jazzowemu (b9, #11, b13)",
    chord_tones: "dźwięków akordu",
    fav_name: "nazwij",
    fav_pick: "Ulubione: ",
    fav_search: "szukaj",
    fav_hint: "Gwiazdka zapisuje formułę z ekranu. Wybór z listy ją przywraca.",
    fav_add: "Dodaj do ulubionych",
    fav_remove: "Usuń z ulubionych",
    formula_chords: "Akordy pasujące do formuły:",
    formula_chords_opt: "Pokazuj pasujące akordy",
    startup_mode: "Tryb po uruchomieniu: ",
    chord_confidence: "Pewność akordu: ",
    note_threshold: "Próg dźwięku: ",
    hold_time: "Czas przytrzymania: ",
    show_debug: "Pokaż podgląd modelu w oknie głównym",
    show_spectrum: "Pokaż widmo",
    ai_prediction: "Rozpoznano:  ",
    intervals_label: "Interwały: ",
    intervals_hint: "Stopnie, np. 1 3 5. Oktawa: 5' w górę, 5, w dół.",
    intervals_placeholder: "np. 1 3 5",
    next: "Następny",
    previous: "Poprzedni",
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
            ("show_diagrams", EN.show_diagrams, PL.show_diagrams),
            ("select_song", EN.select_song, PL.select_song),
            ("pattern", EN.pattern, PL.pattern),
            ("key_root", EN.key_root, PL.key_root),
            ("random_hint", EN.random_hint, PL.random_hint),
            ("start_from", EN.start_from, PL.start_from),
            ("fretboard", EN.fretboard, PL.fretboard),
            ("intervals_hint", EN.intervals_hint, PL.intervals_hint),
            ("no_data", EN.no_data, PL.no_data),
        ] {
            assert!(!en.is_empty(), "{name}: English missing");
            assert!(!pl.is_empty(), "{name}: Polish missing");
            assert_ne!(en, pl, "{name}: Polish identical to English");
        }
    }
}

/// Scale names in the chosen language.
///
/// The definitions carry English names because that is what the files and the
/// literature use, and translating them at the point of display keeps the file
/// format and the indices into it untouched. A name not in the table is passed
/// through - a user's own scale in `user_scales_def.txt` stays as written.
pub fn scale_name(lang: Lang, name: &str) -> &str {
    if lang != Lang::Pl {
        return name;
    }
    match name {
        "Major Scale (Ionian)" => "Durowa (jońska)",
        "Natural Minor (Aeolian)" => "Molowa naturalna (eolska)",
        "Harmonic Minor" => "Molowa harmoniczna",
        "Melodic Minor (Jazz)" => "Molowa melodyczna (jazzowa)",
        "Pentatonic Minor" => "Pentatonika molowa",
        "Pentatonic Major" => "Pentatonika durowa",
        "Blues Scale" => "Bluesowa",
        "Whole-Half Diminished" => "Zmniejszona (cały-pół)",
        "Half-Whole Diminished (Dominant)" => "Zmniejszona (pół-cały, dominantowa)",
        "Whole Tone" => "Całotonowa",
        "Altered Scale (Super Locrian)" => "Alterowana (superlokrycka)",
        "Lydian Dominant" => "Lidyjska dominantowa",
        "Lydian Augmented" => "Lidyjska zwiększona",
        "Phrygian Dominant" => "Frygijska dominantowa",
        "Locrian #2" => "Lokrycka #2",
        "Dorian" => "Dorycka",
        "Mixolydian" => "Miksolidyjska",
        "Lydian" => "Lidyjska",
        "Phrygian" => "Frygijska",
        "Locrian" => "Lokrycka",
        "Bebop Dominant" => "Bebopowa dominantowa",
        "Bebop Major" => "Bebopowa durowa",
        "Bebop Dorian" => "Bebopowa dorycka",
        "Major 6 Diminished" => "Durowa 6 zmniejszona",
        "Minor 6 Diminished" => "Molowa 6 zmniejszona",
        "Dominant 7th Diminished" => "Dominantowa 7 zmniejszona",
        other => other,
    }
}
