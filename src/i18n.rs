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
    /// The settings entry of the side navigation, which draws its own icon.
    pub nav_settings: &'static str,
    /// The label of the bar that says what the app hears.
    pub heard: &'static str,
    /// The input card in the side navigation: its label, its state above
    /// and below the silence threshold, and what the yellow line is.
    pub input_card: &'static str,
    pub input_hearing: &'static str,
    pub input_silent: &'static str,
    pub gate_mark: &'static str,
    pub show_input_meter: &'static str,
    pub show_heard: &'static str,
    /// The badge under an interval tile: credited, asked for now, later in the
    /// order, or - in free order - any of them.
    pub tile_done: &'static str,
    pub tile_now: &'static str,
    pub tile_next: &'static str,
    pub tile_open: &'static str,
    /// Quick switches on the Intervals screen; the full wording is in settings.
    pub chip_in_order: &'static str,
    pub chip_single: &'static str,
    pub chip_struck: &'static str,
    pub chip_lock: &'static str,
    pub chip_attack: &'static str,
    /// Scales: the way it runs, among the quick options.
    pub chip_up: &'static str,
    pub chip_down: &'static str,
    pub chip_drawn: &'static str,
    /// What the drawn direction draws between, in a bubble beside it.
    pub chip_drawn_hint: &'static str,
    /// Said on hovering the fretboard trainer's note.
    pub note_hint_hover: &'static str,
    /// The fretboard trainer's quick option: the hint with every note.
    pub chip_hint: &'static str,
    /// After the key a tune is written in, in the key list of Chords.
    pub key_original: &'static str,
    /// The shuffle's name in Scales, where it draws the key.
    pub random_key: &'static str,
    pub there_and_back: &'static str,
    pub scale_from: &'static str,
    /// The degrees a scale may start on after 1 3 5 7: drawn.
    pub scale_from_drawn: &'static str,
    /// The whole of first position, low E to high e.
    pub scale_from_e_to_e: &'static str,
    pub key_move: &'static str,
    /// Where the key goes - see `state::KeyMove`, in its order.
    pub key_moves: [&'static str; 5],
    /// Quick options of Chords: which shapes to draw.
    pub quick_full: &'static str,
    pub quick_shells: &'static str,
    pub tab_audio: &'static str,
    pub tab_practice: &'static str,
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
    pub debug_console: &'static str,
    pub shuffle_chords: &'static str,
    pub shuffle_chords_hint: &'static str,
    pub random_order: &'static str,
    pub show_diagrams: &'static str,
    pub repeat_root: &'static str,
    pub repeat_root_hint: &'static str,
    pub voice_leading: &'static str,
    pub voice_leading_hint: &'static str,
    pub lead_line: &'static str,
    pub lead_line_hint: &'static str,
    pub preview: &'static str,
    pub preview_hint: &'static str,
    pub preview_names: &'static str,
    pub preview_tab: &'static str,
    pub preview_neck: &'static str,
    pub preview_note_names: &'static str,
    pub preview_functions: &'static str,
    pub preview_note_names_neck: &'static str,
    pub preview_functions_neck: &'static str,
    pub preview_tiles: &'static str,
    pub tab_frets: &'static str,
    pub tab_frets_hint: &'static str,
    pub arp_exercise: &'static str,
    pub arp_study: &'static str,
    pub arp_changes: &'static str,
    pub arp_quality: &'static str,
    pub arp_direction: &'static str,
    pub arp_up: &'static str,
    pub arp_down: &'static str,
    pub arp_alt_down: &'static str,
    pub arp_alt_up: &'static str,
    pub select_study: &'static str,
    pub shapes_kind: &'static str,
    pub shapes_full: &'static str,
    pub shapes_shell: &'static str,
    pub shapes_no_shell: &'static str,
    pub shapes_no_shell_end: &'static str,
    pub shapes_substitute: &'static str,
    /// m6 and mMaj7, which the chord model cannot name: what it accepts.
    pub shapes_heard_as: &'static str,
    pub shapes_or: &'static str,
    pub shapes_same_notes: &'static str,
    // Combo labels. These used to be hard-coded English in main.rs, bypassing
    // the whole translation table.
    pub select_song: &'static str,
    pub select_scale: &'static str,
    /// Nothing in the UI shows this any more; the translation-coverage test
    /// still checks it, which a release build cannot see.
    #[allow(dead_code)]
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
    /// Intervals: the exercise, and the headings its list is grouped under -
    /// see `interval_levels::Group`.
    pub interval_level: &'static str,
    pub level_groups: [&'static str; crate::interval_levels::GROUPS],
    /// A scale run started on a tone drawn for every chord.
    pub level_any_chord_tone: &'static str,
    pub level_any_scale_tone: &'static str,
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
    nav_settings: "Settings",
    heard: "Hearing",
    input_card: "Input",
    input_hearing: "hearing",
    input_silent: "silence",
    gate_mark: "yellow line = silence threshold",
    show_input_meter: "Input meter in the side bar",
    show_heard: "The Hearing bar under the exercise",
    tile_done: "done",
    tile_now: "now",
    tile_next: "later",
    tile_open: "play",
    chip_in_order: "In order",
    chip_single: "Single notes",
    chip_struck: "Struck only",
    chip_lock: "Hold the quality",
    chip_attack: "Short strums",
    chip_up: "Up",
    chip_down: "Down",
    chip_drawn: "Random",
    chip_drawn_hint: "Up or down",
    note_hint_hover: "Click to see a hint",
    chip_hint: "Hint",
    key_original: " (original)",
    random_key: "Random key",
    there_and_back: "There and back",
    scale_from: "Start on: ",
    scale_from_drawn: "random",
    scale_from_e_to_e: "From E to e",
    key_move: "Next key: ",
    key_moves: ["the same", "a semitone up", "a semitone down", "round the fifths", "round the fourths"],
    quick_full: "Full chords",
    quick_shells: "Shell voicings",
    tab_audio: "Audio",
    tab_practice: "Practice",
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
    debug_console: "Console debug",
    shuffle_chords: "Shuffle the chords as well",
    shuffle_chords_hint: "With the shuffle on. Off: the progression stays as written.",
    random_order: "Random order",
    show_diagrams: "Show chord shapes",
    repeat_root: "End on the root again",
    repeat_root_hint: "Ends on the root an octave up, a step of its own.",
    arp_exercise: "Exercise",
    arp_study: "Study in a key",
    arp_changes: "Over the changes",
    arp_quality: "Chord",
    arp_direction: "Direction",
    arp_up: "Ascending",
    arp_down: "Descending",
    arp_alt_down: "Alternating, from descending",
    arp_alt_up: "Alternating, from ascending",
    select_study: "Study",
    tab_frets: "Fret numbers instead of degrees",
    tab_frets_hint: "In the drawing only; the root keeps its colour.",
    voice_leading: "Nearest grip",
    voice_leading_hint: "Each grip where the hand moves least. Off: anywhere on the neck.",
    lead_line: "Voice leading",
    lead_line_hint: "With the shuffle on: each chord starts on its tone nearest the last one played, never the same note.",
    preview: "View",
    preview_hint: "Names say which degrees; drawings say where the fingers go.",
    preview_names: "Line of intervals",
    preview_tab: "Tablature",
    preview_neck: "On the neck",
    preview_note_names: "Note names",
    preview_functions: "Interval functions",
    preview_note_names_neck: "Note names on the neck",
    preview_functions_neck: "Interval functions on the neck",
    preview_tiles: "Interval tiles",
    shapes_kind: "Shapes: ",
    shapes_full: "full chords",
    shapes_shell: "shell voicings",
    shapes_no_shell: "dim7 has no shell — the same grip is also ",
    shapes_no_shell_end: ", each without its root",
    shapes_substitute: "substitute: the m7 shell",
    shapes_heard_as: "the app accepts a substitute: ",
    shapes_or: " or ",
    shapes_same_notes: " (the same notes)",
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
    interval_level: "Exercise: ",
    level_groups: [
        "One tone", "Two tones", "Three tones", "Four tones", "Four tones to the octave",
        "Structure 1–5", "Structure 5–9", "Voice leading", "Suspension resolved",
        "Chord scale", "Passing-note scale",
    ],
    level_any_chord_tone: "random chord tone",
    level_any_scale_tone: "random scale tone",
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
    nav_settings: "Ustawienia",
    heard: "Słyszę",
    input_card: "Wejście",
    input_hearing: "słyszę",
    input_silent: "cisza",
    gate_mark: "żółta kreska = próg ciszy",
    show_input_meter: "Miernik wejścia w pasku bocznym",
    show_heard: "Pasek „Słyszę” pod ćwiczeniem",
    tile_done: "zaliczone",
    tile_now: "teraz",
    tile_next: "dalej",
    tile_open: "zagraj",
    chip_in_order: "Po kolei",
    chip_single: "Pojedynczo",
    chip_struck: "Tylko uderzone",
    chip_lock: "Trzymaj jakość",
    chip_attack: "Krótkie uderzenia",
    chip_up: "W górę",
    chip_down: "W dół",
    chip_drawn: "Losowo",
    chip_drawn_hint: "W górę lub w dół",
    note_hint_hover: "Kliknij, aby zobaczyć podpowiedź",
    chip_hint: "Podpowiedź",
    key_original: " (oryginał)",
    random_key: "Losowa tonacja",
    there_and_back: "Tam i z powrotem",
    scale_from: "Początek: ",
    scale_from_drawn: "losowo",
    scale_from_e_to_e: "Od E do e",
    key_move: "Ruch tonacji: ",
    key_moves: ["ta sama", "o półton w górę", "o półton w dół", "kołem kwintowym", "kołem kwartowym"],
    quick_full: "Pełne chwyty",
    quick_shells: "Shelle",
    tab_audio: "Dźwięk",
    tab_practice: "Ćwiczenia",
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
    debug_console: "Debug na konsoli",
    shuffle_chords: "Losuj także kolejność akordów",
    shuffle_chords_hint: "Przy włączonym losowaniu. Wyłączone: progresja jak zapisana.",
    random_order: "Losowa kolejność",
    show_diagrams: "Wyświetlaj schematy akordów",
    repeat_root: "Kończ powtórzoną prymą",
    repeat_root_hint: "Kończy na prymie oktawę wyżej, jako osobny krok.",
    arp_exercise: "Ćwiczenie",
    arp_study: "Studium w tonacji",
    arp_changes: "Ogrywanie zmian",
    arp_quality: "Akord",
    arp_direction: "Kierunek",
    arp_up: "Wznoszące",
    arp_down: "Opadające",
    arp_alt_down: "Naprzemiennie, od opadającego",
    arp_alt_up: "Naprzemiennie, od wznoszącego",
    select_study: "Studium",
    tab_frets: "Numery progów zamiast stopni",
    tab_frets_hint: "Tylko na rysunku; pryma zachowuje kolor.",
    voice_leading: "Najbliższy chwyt",
    voice_leading_hint: "Chwyt tam, gdzie ręka przesuwa się najmniej. Wyłączone: gdziekolwiek.",
    lead_line: "Prowadzenie głosów",
    lead_line_hint: "Przy losowaniu: akord zaczyna się od składnika najbliższego ostatnio zagranemu dźwiękowi, nigdy od tego samego.",
    preview: "Widok",
    preview_hint: "Nazwy mówią, które stopnie; rysunki – gdzie idą palce.",
    preview_names: "Linia interwałów",
    preview_tab: "Tabulatura",
    preview_neck: "Na podstrunnicy",
    preview_note_names: "Tekstowe nazwy dźwięków",
    preview_functions: "Funkcje interwałowe",
    preview_note_names_neck: "Nazwy dźwięków na gryfie",
    preview_functions_neck: "Funkcje interwałowe na gryfie",
    preview_tiles: "Kafelki interwałów",
    shapes_kind: "Schematy: ",
    shapes_full: "pełne akordy",
    shapes_shell: "shell voicings",
    shapes_no_shell: "dim7 nie ma shella — ten sam chwyt to także ",
    shapes_no_shell_end: ", każdy bez prymy",
    shapes_substitute: "substytut: shell m7",
    shapes_heard_as: "program przyjmuje substytut: ",
    shapes_or: " lub ",
    shapes_same_notes: " (te same dźwięki)",
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
    interval_level: "Ćwiczenie: ",
    level_groups: [
        "Jeden dźwięk", "Dwa dźwięki", "Trzy dźwięki", "Cztery dźwięki", "Cztery dźwięki z oktawą",
        "Struktura 1–5", "Struktura 5–9", "Prowadzenie głosów", "Rozwiązanie zawieszenia",
        "Skala akordu", "Skala z dźwiękiem przejściowym",
    ],
    level_any_chord_tone: "losowy składnik",
    level_any_scale_tone: "losowy stopień",
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
            ("nav_settings", EN.nav_settings, PL.nav_settings),
            ("heard", EN.heard, PL.heard),
            ("input_card", EN.input_card, PL.input_card),
            ("input_hearing", EN.input_hearing, PL.input_hearing),
            ("input_silent", EN.input_silent, PL.input_silent),
            ("gate_mark", EN.gate_mark, PL.gate_mark),
            ("show_input_meter", EN.show_input_meter, PL.show_input_meter),
            ("show_heard", EN.show_heard, PL.show_heard),
            ("tile_done", EN.tile_done, PL.tile_done),
            ("tile_now", EN.tile_now, PL.tile_now),
            ("tile_next", EN.tile_next, PL.tile_next),
            ("tile_open", EN.tile_open, PL.tile_open),
            ("chip_in_order", EN.chip_in_order, PL.chip_in_order),
            ("chip_single", EN.chip_single, PL.chip_single),
            ("chip_struck", EN.chip_struck, PL.chip_struck),
            ("chip_lock", EN.chip_lock, PL.chip_lock),
            ("chip_attack", EN.chip_attack, PL.chip_attack),
            ("chip_up", EN.chip_up, PL.chip_up),
            ("chip_down", EN.chip_down, PL.chip_down),
            ("chip_drawn", EN.chip_drawn, PL.chip_drawn),
            ("chip_drawn_hint", EN.chip_drawn_hint, PL.chip_drawn_hint),
            ("note_hint_hover", EN.note_hint_hover, PL.note_hint_hover),
            ("chip_hint", EN.chip_hint, PL.chip_hint),
            ("key_original", EN.key_original, PL.key_original),
            ("random_key", EN.random_key, PL.random_key),
            ("there_and_back", EN.there_and_back, PL.there_and_back),
            ("scale_from", EN.scale_from, PL.scale_from),
            ("key_move", EN.key_move, PL.key_move),
            ("quick_full", EN.quick_full, PL.quick_full),
            ("quick_shells", EN.quick_shells, PL.quick_shells),
            ("preview_tiles", EN.preview_tiles, PL.preview_tiles),
            ("gate_hint", EN.gate_hint, PL.gate_hint),
            ("random_order", EN.random_order, PL.random_order),
            ("show_diagrams", EN.show_diagrams, PL.show_diagrams),
            ("select_song", EN.select_song, PL.select_song),
            ("pattern", EN.pattern, PL.pattern),
            ("key_root", EN.key_root, PL.key_root),
            ("random_hint", EN.random_hint, PL.random_hint),
            ("start_from", EN.start_from, PL.start_from),
            ("fretboard", EN.fretboard, PL.fretboard),
            ("intervals_hint", EN.intervals_hint, PL.intervals_hint),
            ("interval_level", EN.interval_level, PL.interval_level),
            ("level_any_chord_tone", EN.level_any_chord_tone, PL.level_any_chord_tone),
            ("level_any_scale_tone", EN.level_any_scale_tone, PL.level_any_scale_tone),
            ("lead_line", EN.lead_line, PL.lead_line),
            ("lead_line_hint", EN.lead_line_hint, PL.lead_line_hint),
            ("no_data", EN.no_data, PL.no_data),
        ] {
            assert!(!en.is_empty(), "{name}: English missing");
            assert!(!pl.is_empty(), "{name}: Polish missing");
            assert_ne!(en, pl, "{name}: Polish identical to English");
        }
    }

    /// Every scale the app ships has a Polish name, and no two the same one.
    #[test]
    fn every_built_in_scale_has_its_own_polish_name() {
        let names: Vec<String> = crate::model::load_all_scale_definitions().into_iter().map(|s| s.name).collect();
        let mut polish = std::collections::HashSet::new();
        for name in &names {
            let pl = scale_name(Lang::Pl, name);
            assert_ne!(pl, name, "{name}: no Polish name");
            assert!(polish.insert(pl), "{name}: Polish name used twice");
        }
    }
}

/// Scale names in the chosen language.
///
/// The definitions carry English names because that is what the files and the
/// literature use, and translating them at the point of display keeps the file
/// format and the indices into it untouched. A name not in the table is passed
/// through - a user's own scale in `user_scales_def.txt` stays as written.
/// Arpeggio patterns by name, the same way scales are translated: the name in
/// the table stays English, because that is what a saved setting and a user's
/// own pattern are matched by, and only the screen changes language.
pub fn arpeggio_name(lang: Lang, name: &str) -> &str {
    if lang != Lang::Pl {
        return name;
    }
    match name {
        "Two Octaves Up-Down" => "Dwie oktawy w górę i w dół",
        "Two Octaves Up" => "Dwie oktawy w górę",
        "Two Octaves Down" => "Dwie oktawy w dół",
        "Two Octaves Down from the Root" => "Dwie oktawy w dół od prymy",
        "Minor (Two Octaves and a Third)" => "Molowe (dwie oktawy i tercja)",
        "Major (Leading Tone)" => "Durowe (dźwięk prowadzący)",
        "Dominant (Approach from Below)" => "Dominantowe (dojście od dołu)",
        "Skipping Notes (Fifths and Fourths)" => "Z pominięciem dźwięków (kwinty i kwarty)",
        "Triplets (Up-Down)" => "Triolowe (w górę i w dół)",
        "Generator (new phrase each pass)" => "Generator (nowa fraza za każdym razem)",
        other => other,
    }
}

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
        "Bebop Dorian" => "Bebopowa dorycka (z tercją wielką)",
        "Bebop Dorian (major 7th)" => "Bebopowa dorycka (z septymą wielką)",
        "Bebop Dominant b9 b13" => "Bebopowa dominantowa b9 b13",
        "Bebop Tonic Minor (Dorian)" => "Bebopowa molowa toniczna (dorycka)",
        "Pentatonic Dominant" => "Pentatonika dominantowa",
        "Pentatonic Minor 6" => "Pentatonika molowa z sekstą",
        "Major Blues Scale" => "Bluesowa durowa",
        "Augmented Scale" => "Zwiększona (symetryczna)",
        "Dorian b2" => "Dorycka b2",
        "Mixolydian b6" => "Miksolidyjska b6",
        "Locrian Natural 6" => "Lokrycka z sekstą wielką",
        "Ionian #5" => "Jońska #5",
        "Dorian #4" => "Dorycka #4",
        "Lydian #9" => "Lidyjska #9",
        "Superlocrian bb7" => "Superlokrycka bb7",
        "Harmonic Major" => "Durowa harmoniczna",
        "Dorian b5" => "Dorycka b5",
        "Phrygian b4" => "Frygijska b4",
        "Lydian b3" => "Lidyjska b3",
        "Mixolydian b2" => "Miksolidyjska b2",
        "Lydian Augmented #2" => "Lidyjska zwiększona #2",
        "Locrian bb7" => "Lokrycka bb7",
        "Major 6 Diminished" => "Durowa 6 zmniejszona",
        "Minor 6 Diminished" => "Molowa 6 zmniejszona",
        "Dominant 7th Diminished" => "Dominantowa 7 zmniejszona",
        other => other,
    }
}
