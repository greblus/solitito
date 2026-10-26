//! One frame of the window, every 16 ms: read what the controls say into the
//! app, take the model's latest answer, and show the app's state.
//!
//! Most of what is kept between frames is there to push a thing to Slint only
//! when it has changed - a new model every frame rebuilds a list, re-parses an
//! SVG, or redraws the settings panel for nothing.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use slint::{Color, ComponentHandle, Model, ModelRc, SharedString, VecModel};

use crate::i18n::{self, Lang};
use crate::latch::ChordLatch;
use crate::model::{self, Chord, ChordQuality, NoteName};
use crate::settings::{Preview, Settings};
use crate::state::{self, AppMode, ArpExercise, FormulaExercise, FormulaPlacement, MatchStatus, MyApp, NoteState};
use crate::{diagrams, formulas, tab};
use crate::{AiResult, AppWindow, Chip, HeardNote, IntervalTile, SettingsSnapshot, Tr};
use crate::audio::{db_to_lin, lin_to_db};
use crate::devices::rescan_devices;
use crate::{set_if_changed, svg_icon, MIN_FILL_CHORD};

/// What the frame shares with the rest of the app.
pub struct Shared {
    pub app: Arc<Mutex<MyApp>>,
    pub ai_result: Arc<Mutex<AiResult>>,
    pub cfg: Rc<RefCell<Settings>>,
    pub device_names: Rc<RefCell<Vec<String>>>,
    pub opened: Rc<RefCell<crate::audio::InputInfo>>,
}

pub struct UiFrame {
    ui: slint::Weak<AppWindow>,
    shared: Shared,
    file_mode: bool,
    /// The language the window started in.
    lang: Lang,

    // Built once and then written into, NOT replaced every frame. Assigning a
    // fresh model makes Slint throw away all 48 bars and build them again, and
    // at 60 frames a second that was most of a core - visible only with the
    // settings panel open, because the spectrum lives inside it.
    spectrum_data: Rc<VecModel<f32>>,
    spectrum_colors: Rc<VecModel<Color>>,

    // Taken after the initial set_* calls, so persisted settings count as the
    // baseline rather than as a change. Retaken when the panel closes: the mark
    // is feedback that an edit registered, not a permanent badge.
    baseline: SettingsSnapshot,
    settings_were_open: bool,

    /// Chord quality latch - see the `latch` module docs.
    chord_latch: ChordLatch,
    /// When the last prediction was consumed. The progress timer MUST be given
    /// real elapsed time: the AI thread does inference AND sleeps 40 ms, so the
    /// real interval is 55-90 ms and varies. A hard-coded 0.040 made the counter
    /// run slower than the clock, turning a 0.6 s threshold into about a second.
    last_ai_at: Option<Instant>,
    /// Where the interval strip stood last frame. A step BACKWARDS means the
    /// exercise restarted - new chord, new key - so the strip snaps back to the
    /// first page instead of turning over at the normal pace.
    last_interval_step: i32,
    /// Pattern length as well as position: entering a mode or changing chord
    /// swaps the whole strip, and that is a restart too - without this the first
    /// slide into place would crawl at the page-turn speed.
    last_interval_len: i32,
    /// What the tablature strip was last drawn for. Rebuilding it means parsing
    /// an SVG, which is not a thing to do sixty times a second for a picture
    /// that changes when a note is played.
    last_tab: String,
    /// The grip drawn last, so the next one can lead its voices from it.
    last_voicing: Vec<tab::Spot>,
    /// The fretboard trainer's last answer: the note it credited, kept on the
    /// neck after it moves on to the next one.
    last_answer: Option<(usize, bool)>,
    /// The note the fretboard hint was opened for: a new one closes it.
    hint_target: Option<usize>,
    /// Rasterising the SVGs is not free and the shapes only change when the chord
    /// QUALITY does, which is far less often than every frame.
    last_diagram_key: String,
    /// The progression strip and what is heard, pushed only when they change:
    /// both are lists, and a new model every frame would redraw them for nothing.
    last_progress: Vec<(String, i32, bool, Color)>,
    last_song_keys: Vec<String>,
    last_heard: (String, Vec<(String, bool)>),
    last_heard_push: Instant,
    last_ui_tick: Option<Instant>,
    /// The chord list only changes with the formula or its key, and rebuilding it
    /// every tick would drop whichever chord the pointer is on sixty times a
    /// second. The key is in the guard because the chords are also spelled out in
    /// letters, and those follow it.
    /// Mask, root and placement: a chord change can land the formula on the same
    /// root from a different degree, and then only the degree tells this row that
    /// its numerals are stale.
    last_formula_chords: (u16, String, usize),
    // Live readouts are what drive the redraw rate, and the panel is expensive to
    // redraw. Neither of these carries information at sixty frames a second: the
    // confidence figure jitters by a point between frames, and the meter is
    // smoothed so it never settles exactly. Both are therefore held back unless
    // they have something new to say.
    last_ai_text: String,
    last_ai_name: String,
    last_ai_push: Instant,
    last_level_db: f32,
    last_level_push: Instant,
    last_tiles: Vec<IntervalTile>,
    last_spectrum_push: Instant,
}

impl UiFrame {
    pub fn new(ui: &AppWindow, shared: Shared, file_mode: bool, lang: Lang) -> Self {
        let spectrum_data = Rc::new(VecModel::from(vec![0.0_f32; 48]));
        let spectrum_colors = Rc::new(VecModel::from(vec![Color::from_rgb_u8(30, 30, 30); 48]));
        ui.set_spectrum_data(ModelRc::from(spectrum_data.clone()));
        ui.set_spectrum_colors(ModelRc::from(spectrum_colors.clone()));
        let now = Instant::now();
        UiFrame {
            ui: ui.as_weak(),
            shared,
            file_mode,
            lang,
            spectrum_data,
            spectrum_colors,
            baseline: SettingsSnapshot::read(ui),
            settings_were_open: false,
            chord_latch: ChordLatch::default(),
            last_ai_at: None,
            last_interval_step: 0,
            last_interval_len: 0,
            last_tab: String::new(),
            last_voicing: Vec::new(),
            last_answer: None,
            hint_target: None,
            last_diagram_key: String::new(),
            last_progress: Vec::new(),
            last_song_keys: Vec::new(),
            last_heard: (String::new(), Vec::new()),
            last_heard_push: now,
            last_ui_tick: None,
            last_formula_chords: (0, String::new(), usize::MAX),
            last_ai_text: String::new(),
            last_ai_name: String::new(),
            last_ai_push: now,
            last_level_db: f32::NEG_INFINITY,
            last_level_push: now,
            last_tiles: Vec::new(),
            last_spectrum_push: now,
        }
    }

    /// One frame: controls into the app, the model's answer, the app onto the screen.
    pub fn run(&mut self) {
        let Some(ui) = self.ui.upgrade() else { return };
        let app_handle = Arc::clone(&self.shared.app);
        let mut app = app_handle.lock().unwrap();

        let (spectrum_vis, in_level) = {
            let s = app.analysis_state.lock().unwrap();
            (s.spectrum_visual, s.input_level)
        };
        self.show_input_level(&ui, in_level);

        self.read_practice_options(&ui, &mut app);
        self.save_formula_fields(&ui);
        self.read_formula_options(&ui, &mut app);
        Self::follow_pause(&ui, &mut app);
        self.follow_settings_panel(&ui);
        Self::read_interval_text(&ui, &mut app);
        app.sync_audio_settings();
        self.advance_clock(&mut app);
        self.take_model_answer(&ui, &mut app);

        Self::show_title(&ui, &app);
        if app.app_mode == AppMode::Formulas {
            self.show_formula(&ui, &app);
        } else if app.app_mode == AppMode::Fretboard {
            self.show_fretboard(&ui, &app);
        } else if app.chords.is_empty() {
            ui.set_chord_name(ui.global::<Tr>().get_no_data());
        } else {
            self.show_song_step(&ui, &app);
        }
        self.drop_drawing_of_other_modes(&ui, &app);

        self.show_spectrum(&ui, &app, &spectrum_vis);
    }

    fn show_input_level(&mut self, ui: &AppWindow, in_level: f32) {
        // Half a decibel is below what the bar can show anyway. At most 20
        // times a second, like the spectrum: the meter in the side navigation
        // is always on screen, and every new level redraws the window.
        let level_db = lin_to_db(in_level);
        if (level_db - self.last_level_db).abs() > 0.5
            && self.last_level_push.elapsed() >= Duration::from_millis(50)
        {
            self.last_level_db = level_db;
            self.last_level_push = Instant::now();
            ui.set_input_level_db(level_db);
        }
    }

    /// The practice controls, read into the app every frame.
    fn read_practice_options(&mut self, ui: &AppWindow, app: &mut MyApp) {
        if !self.file_mode {
            app.noise_gate = db_to_lin(ui.get_gate_db());
        }
        app.bass_boost_enabled = ui.get_boost_enabled();
        app.bass_boost_gain = ui.get_boost_gain();
        app.chord_confidence = ui.get_chord_confidence();
        app.note_threshold = ui.get_note_threshold();
        app.transition_delay = ui.get_delay();
        // The shuffle can change WHICH phrase is showing in the arpeggio
        // studies, so the combo and the interval box follow it.
        let study_before = app.selected_library_idx;
        app.set_random_mode(ui.get_random_enabled());
        if app.selected_library_idx != study_before {
            set_if_changed(ui.get_current_item_index(), app.selected_library_idx as i32, |v| {
                ui.set_current_item_index(v)
            });
            ui.set_interval_input_text(model::shown_phrase(&app.intervals_input).into());
        }
        app.short_verdict = ui.get_short_verdict();
        app.lead_line = ui.get_lead_line();
        app.arp_exercise = (ui.get_arp_exercise().max(0) as usize).into();
        app.shells_only = ui.get_show_shell_shapes() && !ui.get_show_full_shapes();
        if app.app_mode == state::AppMode::Arpeggios {
            let quality = ui.get_arp_quality().max(0) as usize;
            app.arp_direction = (ui.get_arp_direction().max(0) as usize).into();
            if app.arp_quality != quality {
                app.arp_quality = quality;
                app.reload_library();
            }
        }
        app.single_notes = ui.get_single_notes();
        app.require_onset = ui.get_require_onset();
        // The extra step changes the length of the run, so the strip and the
        // marks have to be rebuilt when it is switched.
        if app.scale_repeat_root != ui.get_scale_repeat_root() {
            app.scale_repeat_root = ui.get_scale_repeat_root();
            app.reset_logic_state();
        }
        app.set_shuffle_chords(ui.get_shuffle_chords());
        app.set_scale_way(
            (ui.get_scale_direction().max(0) as usize).into(),
            (ui.get_scale_from().max(0) as usize).into(),
            ui.get_scale_there_and_back(),
            (ui.get_key_move().max(0) as usize).into(),
            ui.get_scale_in_order(),
        );
        app.set_scale_e_to_e(ui.get_scale_e_to_e());
    }

    /// The two formula text fields, saved as they are typed. They are read
    /// from the UI, not pushed into it - the same arrangement the Intervals
    /// field uses, and the only one that leaves them editable.
    fn save_formula_fields(&self, ui: &AppWindow) {
        let key_txt = ui.get_formula_key_text().to_string();
        let req_txt = ui.get_formula_required_text().to_string();
        let mut cfg = self.shared.cfg.borrow_mut();
        if cfg.formula_key != key_txt || cfg.formula_required != req_txt {
            cfg.formula_key = key_txt;
            cfg.formula_required = req_txt;
            cfg.save();
        }
    }

    fn read_formula_options(&self, ui: &AppWindow, app: &mut MyApp) {
        // Formulas options. A formula now stands until something here moves,
        // so a change has to act at once - waiting for "the next formula"
        // would mean waiting for one that is never drawn.
        let cfg = self.shared.cfg.borrow();
        let required = formulas::parse(&cfg.formula_required).unwrap_or(1);
        // Which of the two: the size and the filter describe the formula
        // itself, the key only says where to read it. Redrawing on a key
        // change would throw away the formula being practised, and the key
        // field is read letter by letter as it is typed.
        // Changing the exercise changes what the formula is read against,
        // so it redraws: a placement over a chord that is no longer there
        // would be read as a key.
        let exercise = FormulaExercise::from(cfg.formula_exercise as usize);
        let placement = FormulaPlacement::from(cfg.formula_placement as usize);
        let redraw = app.formula_notes != cfg.formula_notes
            || app.formula_required != required
            || app.formula_exercise != exercise;
        // A different kind of placement is the same formula seen from
        // somewhere else - the same gesture as asking for another key.
        let rekey = app.formula_random_key != cfg.formula_random_key
            || app.formula_key_setting != cfg.formula_key
            || app.formula_placement_want != placement;
        let exercise_changed = app.formula_exercise != exercise;
        app.formula_exercise = exercise;
        app.formula_placement_want = placement;
        app.formula_notes = cfg.formula_notes;
        app.formula_required = required;
        app.formula_random_key = cfg.formula_random_key;
        app.formula_key_setting = cfg.formula_key.clone();
        app.formula_in_order = cfg.formula_in_order;
        app.interval_in_order = cfg.interval_in_order;
        app.log_credits = cfg.debug_console;
        if app.app_mode == state::AppMode::Formulas {
            // Over the changes the tune has to be under it before a chord
            // can be taken from it.
            if exercise_changed && exercise == FormulaExercise::OverChanges {
                app.selected_library_idx = app.saved_song();
                app.reload_library();
                ui.set_current_item_index(app.selected_library_idx as i32);
            }
            if redraw {
                app.next_formula();
            } else if rekey {
                app.rekey_formula();
            }
        }
    }

    fn follow_pause(ui: &AppWindow, app: &mut MyApp) {
        // Neither the fretboard trainer nor formulas show the pause button, so a
        // pause carried in from another mode would freeze them with nothing on
        // screen to explain why - in Formulas it came out as a set that stayed
        // green while the next one never came.
        if app.app_mode == AppMode::Fretboard && ui.get_paused() {
            ui.set_paused(false);
        }
        app.paused = ui.get_paused();
    }

    /// Device rescan when the settings panel opens; when it closes, the gate
    /// saved and the "changed" mark cleared.
    fn follow_settings_panel(&mut self, ui: &AppWindow) {
        let settings_open = ui.get_show_settings();
        // Re-scanned every time the panel opens - and again when the picker
        // itself is opened, see `on_audio_devices_rescan`.
        if settings_open && !self.settings_were_open && !self.file_mode {
            let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
            let open_now = (!self.shared.opened.borrow().fell_back)
                .then(|| self.shared.cfg.borrow().audio_device.clone())
                .flatten();
            rescan_devices(ui, &self.shared.device_names, t.audio_default, open_now.as_deref());
        }
        if self.settings_were_open && !settings_open {
            self.baseline = SettingsSnapshot::read(ui);   // leaving the panel clears the mark
            // Written when the panel closes rather than on every drag: the
            // slider moves in dozens of steps and each one would be a file.
            let mut cfg = self.shared.cfg.borrow_mut();
            let dev = cfg.audio_device.clone();
            if cfg.set_gate(dev.as_deref(), ui.get_gate_db()) {
                cfg.save();
            }
        }
        self.settings_were_open = settings_open;
        set_if_changed(
            ui.get_settings_touched(),
            settings_open && SettingsSnapshot::read(ui) != self.baseline,
            |v| ui.set_settings_touched(v),
        );
    }

    fn read_interval_text(ui: &AppWindow, app: &mut MyApp) {
        let ui_txt = model::written_phrase(&ui.get_interval_input_text());
        if ui_txt != app.intervals_input {
            app.intervals_input = ui_txt;
            app.reset_logic_state();
        }
    }

    fn advance_clock(&mut self, app: &mut MyApp) {
        // Every frame, model answering or not: the ear is a per-frame estimate,
        // and a finished lap is timed in wall clock rather than in answers.
        let now = Instant::now();
        let ui_dt = self.last_ui_tick
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(0.016)
            .min(0.25);
        self.last_ui_tick = Some(now);
        app.tick(ui_dt);
    }

    /// The model's latest answer, if there is a new one: voted, latched, shown,
    /// and handed to the judge.
    fn take_model_answer(&mut self, ui: &AppWindow, app: &mut MyApp) {
        let ai_result = Arc::clone(&self.shared.ai_result);
        let Ok(mut res) = ai_result.lock() else { return };
        if !res.updated {
            return;
        }
        // A half-full window is answerable for what is SOUNDING, not for
        // what chord it is.
        let named = res.fill >= MIN_FILL_CHORD;
        let chord = res.pred.chord.clone();
        let score = res.pred.confidence;
        app.prev_pitches = app.last_pitches;
        app.last_pitches = res.pred.pitches;

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

        let (best_c, current_confidence) = vote_on_chord(&app.chord_history);

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
            self.chord_latch.update(
                ui.get_lock_quality(), onset_id, since_onset, &best_c, current_confidence,
            )
        } else {
            "...".to_string()
        };
        let current_confidence = if named { current_confidence } else { 0.0 };

        self.show_model_answer(ui, &shown, current_confidence);

        // Real elapsed time, not assumed. Capped so a thread stall does not
        // jump the whole threshold at once.
        let now = Instant::now();
        let dt = self.last_ai_at
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(0.040)
            .min(0.25);
        self.last_ai_at = Some(now);
        let before = app.current_chord_index;
        app.check_progress_with_ai(dt, &shown, current_confidence);
        // SOLITITO_STRUM=1: one line per AI frame, to see whether attacks
        // are being detected at all during real playing. The per-strum
        // verdict can only re-arm on a new onset id.
        if std::env::var("SOLITITO_STRUM").is_ok() {
            print_strum_trace(app, onset_id, since_onset, &shown, current_confidence, before);
        }
    }

    /// The chord name appears at once - that is what is being read. A changed
    /// percentage on the same chord waits its turn, at five updates a second
    /// rather than sixty.
    fn show_model_answer(&mut self, ui: &AppWindow, shown: &str, confidence: f32) {
        let text = format!("{} ({:.0}%)", shown, confidence * 100.0);
        let name_changed = shown != self.last_ai_name;
        if (name_changed || self.last_ai_push.elapsed() >= Duration::from_millis(200))
            && text != self.last_ai_text
        {
            ui.set_ai_text(text.clone().into());
            self.last_ai_text = text;
            self.last_ai_name = shown.to_string();
            self.last_ai_push = Instant::now();
        }
    }

    fn show_title(ui: &AppWindow, app: &MyApp) {
        // In Scales the title IS a scale name, and it is kept in the language
        // the definition file wrote it in - see `library_lists`.
        let lang_here = Lang::from_setting(ui.get_language_idx());
        let shown_title = match app.app_mode {
            AppMode::Scales => i18n::scale_name(lang_here, &app.song_title).to_string(),
            // In a study the title IS the phrase's name; over the changes it is
            // the tune's, which is not ours to translate.
            AppMode::Arpeggios if app.arp_exercise == ArpExercise::Study => {
                i18n::arpeggio_name(lang_here, &app.song_title).to_string()
            }
            _ => app.song_title.clone(),
        };
        set_if_changed(ui.get_song_title(), shown_title.into(), |v| ui.set_song_title(v));

        // Scales in random mode redraw the key on their own; push it to the
        // combo, but only on a real change so it does not fight the user
        // mid-selection (the binding is two-way).
        if app.app_mode == AppMode::Scales
            && ui.get_current_secondary_index() != app.secondary_index as i32
        {
            set_if_changed(ui.get_current_secondary_index(), app.secondary_index as i32, |v| ui.set_current_secondary_index(v));
        }
    }

    fn show_formula(&mut self, ui: &AppWindow, app: &MyApp) {
        Self::show_formula_functions(ui, app);
        self.show_formula_placement(ui, app);
        self.show_formula_key(ui, app);
        self.show_favourites(ui, app);
        self.show_formula_chords(ui, app);
        Self::show_nearest_scale(ui, app);
    }

    fn show_formula_functions(ui: &AppWindow, app: &MyApp) {
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
    }

    /// What the formula is being played over, when it is being played over
    /// anything: the chord, the degree of it the formula stands on, the same
    /// functions read from the chord's root, and which of them the chord owns.
    /// The count is the lesson - the word beside it is only a label on the count.
    fn show_formula_placement(&self, ui: &AppWindow, app: &MyApp) {
        let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
        let jazz_names = self.shared.cfg.borrow().formula_jazz_names;
        let shown = match &app.formula_chord {
            Some(chord) if app.formula_exercise.has_chord() => placement_over(app, chord, &t, jazz_names),
            _ => Placement::default(),
        };
        set_if_changed(ui.get_formula_chord_name(), shown.chord.into(), |v| {
            ui.set_formula_chord_name(v)
        });
        set_if_changed(ui.get_formula_degree_name(), shown.degree.into(), |v| {
            ui.set_formula_degree_name(v)
        });
        set_if_changed(ui.get_formula_against_label(), shown.label.into(), |v| {
            ui.set_formula_against_label(v)
        });
        set_if_changed(ui.get_formula_verdict(), shown.verdict.into(), |v| {
            ui.set_formula_verdict(v)
        });
        let against: Vec<SharedString> = shown.against.into_iter().map(SharedString::from).collect();
        ui.set_formula_against(ModelRc::from(Rc::new(VecModel::from(against))));
        ui.set_formula_is_chord_tone(ModelRc::from(Rc::new(VecModel::from(shown.own))));
    }

    fn show_formula_key(&self, ui: &AppWindow, app: &MyApp) {
        set_if_changed(ui.get_formula_key(), app.formula_key_name.clone().into(), |v| {
            ui.set_formula_key(v)
        });

        // Note names are a crutch, so they are only built when asked for.
        let wanted = self.shared.cfg.borrow().formula_note_names;
        let notes: Vec<SharedString> = wanted
            .then(|| formulas::parse_key(&app.formula_key_name))
            .flatten()
            .map(|key| formulas::note_names(app.formula_mask, &key))
            .unwrap_or_default()
            .into_iter()
            .map(SharedString::from)
            .collect();
        ui.set_formula_note_names(ModelRc::from(Rc::new(VecModel::from(notes))));
    }

    /// The favourites, narrowed by whatever is typed beside them, and whether
    /// what is on screen is one of them. Rebuilt every tick: the list is short
    /// and the filter has to answer as it is typed.
    fn show_favourites(&self, ui: &AppWindow, app: &MyApp) {
        let cfg = self.shared.cfg.borrow();
        let filter = ui.get_favourite_filter();
        let shown: Vec<SharedString> = cfg
            .favourites_matching(&filter)
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
        let here = cfg.favourites_matching(&filter).position(|f| f.mask == app.formula_mask);
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

    fn show_formula_chords(&mut self, ui: &AppWindow, app: &MyApp) {
        let funcs = formulas::functions_of(app.formula_mask);
        // Chords that fit inside the formula, with the functions each is
        // built from flagged against the row of functions above - pointing
        // at one lights them up. Six is what a line holds; past that the
        // list stops saying anything.
        if self.last_formula_chords.0 != app.formula_mask
            || self.last_formula_chords.1 != app.formula_key_name
            || self.last_formula_chords.2 != app.formula_degree
        {
            self.last_formula_chords = (
                app.formula_mask,
                app.formula_key_name.clone(),
                app.formula_degree,
            );
            let fits = formulas::chords_inside(app.formula_mask, 7);
            // Counted from the chord under the formula when there is one,
            // so this row agrees with the two above it.
            let names: Vec<SharedString> = match app.formula_exercise {
                FormulaExercise::InKey => fits.iter().map(|f| SharedString::from(f.name())).collect(),
                FormulaExercise::OverChord | FormulaExercise::OverChanges => fits
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
    }

    fn show_nearest_scale(ui: &AppWindow, app: &MyApp) {
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
    }

    fn show_fretboard(&mut self, ui: &AppWindow, app: &MyApp) {
        // Minimal by design: the note, and under it where to play it.
        let name = match app.fret_target {
            Some(pc) => model::NoteName::from_index(pc).to_string().to_string(),
            None => "...".to_string(),
        };
        set_if_changed(ui.get_chord_name(), name.clone().into(), |v| ui.set_chord_name(v));
        set_if_changed(ui.get_start_hint(), app.region.describe().into(), |v| ui.set_start_hint(v));
        self.draw_fretboard_region(ui, app);
        self.show_fretboard_hint(ui, app);
        ui.set_chord_text_color(slint::Brush::SolidColor(match app.match_status {
            MatchStatus::Exact => Color::from_rgb_u8(50, 255, 50),
            _ => Color::from_rgb_u8(255, 255, 255),
        }));
        ui.set_interval_names(ModelRc::from(Rc::new(VecModel::from(Vec::<SharedString>::new()))));
    }

    /// The region drawn on the neck, and the ANSWER in it: where the note that
    /// is sounding lies, green if it is the one asked for. Nothing before that
    /// - drawing the target first would turn the mode into copying a picture.
    fn draw_fretboard_region(&mut self, ui: &AppWindow, app: &MyApp) {
        if !Preview::from(ui.get_preview()).draws_on_neck() {
            self.clear_drawing(ui);
            return;
        }
        let strings: Vec<usize> = app.region.strings.strings().to_vec();
        let (from, to) = (app.region.fret_from as i32, app.region.fret_to() as i32);
        // What the JUDGE credited, not a reading of its own. Marking the neck
        // from `steady_note` - three frames of the estimate - while the judge
        // passes a note on one frame of it, or on the model's pitch head, meant
        // the trainer moved on from notes the drawing had never marked: right,
        // and never lit.
        //
        // Only a right answer is drawn. A wrong one leaves the box empty: the
        // note asked for is on the screen already, and marking every stray note
        // on the neck was more to read than to learn from. And it is kept after
        // the trainer moves on, so there is something to look at while the next
        // note is read.
        if let (MatchStatus::Exact, Some(target)) = (app.match_status, app.fret_target) {
            self.last_answer = Some((target % 12, true));
        }
        let (heard, right) = match self.last_answer {
            Some((pc, was_right)) => (Some(pc), was_right),
            None => (None, false),
        };
        let marks = heard.map_or_else(Vec::new, |pc| places_of(pc, &strings, from, to));
        let label = heard
            .map(|pc| model::NoteName::from_index(pc).to_string().to_string())
            .unwrap_or_default();
        let signature = format!("{strings:?}|{from}|{to}|{marks:?}|{label}|{right}");
        if signature == self.last_tab {
            return;
        }
        self.last_tab = signature;
        ui.set_tab_aspect(tab::region_aspect(from, to));
        ui.set_tab_image(svg_icon(&tab::region(&strings, from, to, &marks, &label, right)));
    }

    /// The hint the note's tile opens: where the note lies on the neck as far
    /// up as a region is drawn. Closed again by the next note, so it is asked for
    /// each time and not left standing as the answer to every one - unless the
    /// quick option keeps it on.
    fn show_fretboard_hint(&mut self, ui: &AppWindow, app: &MyApp) {
        if self.hint_target != app.fret_target {
            self.hint_target = app.fret_target;
            ui.set_note_hint_open(false);
            ui.set_note_hint_note(-1);
        }
        let Some(pc) = app.fret_target.map(|target| target % 12) else { return };
        let shown = ui.get_note_hint_open() || ui.get_fret_hint();
        if !shown || ui.get_note_hint_note() == pc as i32 {
            return;
        }
        let strings: Vec<usize> = app.region.strings.strings().to_vec();
        let (from, to) = tab::HINT_FRETS;
        let marks = places_of(pc, &[0, 1, 2, 3, 4, 5], from, to);
        let label = model::NoteName::from_index(pc).to_string().to_string();
        ui.set_note_hint_aspect(tab::region_aspect(from, to));
        ui.set_note_hint_image(svg_icon(&tab::neck_hint(&strings, &marks, &label)));
        ui.set_note_hint_note(pc as i32);
    }

    /// Takes down the drawing under the exercise, if there is one.
    fn clear_drawing(&mut self, ui: &AppWindow) {
        if !self.last_tab.is_empty() {
            self.last_tab.clear();
            ui.set_tab_image(slint::Image::default());
        }
    }

    /// Chords, Intervals, Scales and Arpeggios: the chord or the step the song
    /// stands on.
    fn show_song_step(&mut self, ui: &AppWindow, app: &MyApp) {
        let chord = &app.chords[app.current_chord_index];
        Self::show_chord_name(ui, app, chord);
        self.show_start_hint(ui, app);
        self.show_shapes(ui, chord);
        Self::show_chord_colour(ui, app);
        self.show_progress(ui, app);
        if app.app_mode == AppMode::Chords {
            self.show_song_keys(ui, app);
        }
        self.show_heard(ui, app, chord);
        if app.app_mode != AppMode::Chords {
            self.show_steps(ui, app, chord);
        }
    }

    fn show_chord_name(ui: &AppWindow, app: &MyApp, chord: &Chord) {
        if app.app_mode == AppMode::Scales {
             ui.set_chord_name(chord.root.to_string().into());
        } else {
             let q_str = chord.quality.to_string();
             let quality_display = if q_str.is_empty() { "Maj" } else { q_str.as_str() };
             ui.set_chord_name(format!("{} {}", chord.root.to_string(), quality_display).into());
        }
    }

    fn show_start_hint(&self, ui: &AppWindow, app: &MyApp) {
        // Suggestion only: the model gives pitch classes with no position,
        // so nothing here is verified. Empty string hides the line.
        // Nothing to suggest where the grip is drawn: the picture already
        // says which strings to take, and a line saying "start from the A"
        // under a diagram that shows it is noise.
        // Nor in Scales or Arpeggios: the drawing says which string every
        // note is on, so a line naming one of them says less than the
        // picture does.
        let drawn_grip = Preview::from(ui.get_preview()).draws_on_neck()
            && matches!(
                app.app_mode,
                AppMode::Intervals | AppMode::Scales | AppMode::Arpeggios
            );
        let hint = match app.start_hint {
            Some(i) if i < state::START_STRINGS.len() && !drawn_grip => {
                i18n::strings(self.lang).start_from.replace("{}", state::START_STRINGS[i])
            }
            _ => String::new(),
        };
        set_if_changed(ui.get_start_hint(), hint.into(), |v| ui.set_start_hint(v));
    }

    fn show_shapes(&mut self, ui: &AppWindow, chord: &Chord) {
        // Shapes depend on the quality alone - the diagram is movable, the
        // root only decides which fret to put it on.
        // The kind belongs in the key as much as the quality does: the
        // same chord has a different row of shapes under each.
        // Neither ticked draws nothing, which is a legitimate answer and
        // not the same as the master switch being off: that one is about
        // whether the row exists at all.
        let shapes = shapes_asked_for(ui.get_show_full_shapes(), ui.get_show_shell_shapes());
        // More than three shapes always take two rows. Chosen by the
        // window's width, a wide screen put six or ten in one row, and
        // they came out too small to read.
        let count = match shapes {
            Some(kind) => diagrams::for_quality(&chord.quality, kind).len(),
            None => 0,
        };
        let one_row = count <= 3;
        let q_key = format!(
            "{}/{:?}/{}",
            chord.quality.to_string(),
            shapes,
            one_row
        );
        if q_key != self.last_diagram_key {
            self.last_diagram_key = q_key;
            let labelled: Vec<(&'static str, slint::Image)> = match shapes {
                Some(kind) => diagrams::for_quality(&chord.quality, kind)
                    .iter()
                    .filter_map(|d| {
                        slint::Image::load_from_svg_data(d.on_card().as_bytes())
                            .ok()
                            .map(|img| (d.label, img))
                    })
                    .collect(),
                None => vec![],
            };
            let imgs: Vec<slint::Image> = labelled.iter().map(|(_, img)| img.clone()).collect();
            // Half of them go to a second row. The split is here rather
            // than in the UI because a Slint model cannot be sliced.
            let per_row = if !one_row {
                imgs.len().div_ceil(2)
            } else {
                imgs.len()
            };
            let grouped = (!one_row).then(|| diagrams::by_root_string(&labelled)).flatten();
            let (first, rest) = match grouped {
                Some((top, bottom)) => (top, bottom),
                None => (imgs[..per_row].to_vec(), imgs[per_row..].to_vec()),
            };
            ui.set_chord_diagrams(ModelRc::from(Rc::new(VecModel::from(first))));
            ui.set_chord_diagrams_2(ModelRc::from(Rc::new(VecModel::from(rest))));
            let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
            let note = shapes_caption(&t, shapes, chord);
            ui.set_shapes_note(note.into());
            // A shape left open would belong to the previous chord.
            ui.set_diagram_zoom(-1);
        }
    }

    /// The big chord name in the colour of the verdict - in Chords only; the
    /// note modes have their own marks.
    fn show_chord_colour(ui: &AppWindow, app: &MyApp) {
        let status = match app.app_mode {
            AppMode::Chords => app.match_status,
            _ => MatchStatus::None,
        };
        let colour = match status {
            MatchStatus::Exact => Color::from_rgb_u8(50, 255, 50),
            MatchStatus::Partial => Color::from_rgb_u8(255, 220, 50),
            MatchStatus::Flicker => Color::from_rgb_u8(255, 50, 50),
            MatchStatus::None => Color::from_rgb_u8(255, 255, 255),
        };
        ui.set_chord_text_color(slint::Brush::SolidColor(colour));
    }

    /// Chords: the keys the tune may be played in, its own marked.
    fn show_song_keys(&mut self, ui: &AppWindow, app: &MyApp) {
        let Some(tonic) = app.song_tonic() else { return };
        let marked = i18n::strings(Lang::from_setting(ui.get_language_idx())).key_original;
        let keys: Vec<String> = (0..12)
            .map(|pc| {
                let name = NoteName::from_index(pc).to_string().to_string();
                if pc == tonic { format!("{name}{marked}") } else { name }
            })
            .collect();
        if keys != self.last_song_keys {
            let model: Vec<SharedString> = keys.iter().map(|k| k.into()).collect();
            ui.set_song_keys(ModelRc::new(VecModel::from(model)));
            self.last_song_keys = keys;
        }
        set_if_changed(ui.get_song_key(), ((tonic + app.transpose) % 12) as i32, |v| ui.set_song_key(v));
    }

    fn show_progress(&mut self, ui: &AppWindow, app: &MyApp) {
        let next_idx = app.next_chord_index();
        set_if_changed(ui.get_next_chord(), app.chord_label(next_idx).into(), |v| ui.set_next_chord(v));
        // The progression strip. A chord behind the current one is written
        // in the colour it was passed on - the big name's palette, so one
        // colour means one thing wherever it appears - and one stepped
        // over, or still to come, in grey.
        let progress: Vec<(String, i32, bool, Color)> = app
            .progress_around(3)
            .into_iter()
            .map(|(i, d)| {
                let result = match d {
                    -1 => app.prev_status(),
                    d if d < 0 => app.pass_result(d),
                    _ => MatchStatus::None,
                };
                let fg = match result {
                    MatchStatus::Exact => Color::from_rgb_u8(110, 231, 160),
                    MatchStatus::Partial => Color::from_rgb_u8(255, 220, 50),
                    MatchStatus::Flicker => Color::from_rgb_u8(255, 90, 90),
                    MatchStatus::None => Color::from_rgb_u8(138, 146, 163),
                };
                (app.chord_label(i), d, result != MatchStatus::None, fg)
            })
            .collect();
        if progress != self.last_progress {
            let chips: Vec<Chip> = progress
                .iter()
                .map(|(name, d, played, fg)| Chip {
                    name: name.into(),
                    current: *d == 0,
                    played: *played,
                    fg: *fg,
                    offset: *d,
                })
                .collect();
            ui.set_progress(slint::ModelRc::new(slint::VecModel::from(chips)));
            self.last_progress = progress;
        }
    }

    fn show_heard(&mut self, ui: &AppWindow, app: &MyApp, chord: &Chord) {
        // What is heard. In Chords the name the model gives the chord; in
        // the note modes the note the ear holds. Beside it the notes the
        // pitch head hears sounding, green where the chord asked for has
        // them - in pitch order, not by loudness: ranked by level they
        // traded places every frame.
        let heard_name = if app.app_mode == AppMode::Chords {
            if self.last_ai_name == "..." { String::new() } else { self.last_ai_name.clone() }
        } else {
            app.heard.steady_note().map(|pc| NoteName::from_index(pc).to_string().to_string()).unwrap_or_default()
        };
        let tones = chord.get_target_indices();
        let mut sounding: Vec<(usize, f32)> = app
            .last_pitches
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p >= app.note_threshold)
            .map(|(pc, &p)| (pc, p))
            .collect();
        // The five loudest, then put back in pitch order.
        sounding.sort_by(|a, b| b.1.total_cmp(&a.1));
        sounding.truncate(5);
        sounding.sort_by_key(|&(pc, _)| pc);
        let notes: Vec<(String, bool)> = sounding
            .iter()
            .map(|&(pc, _)| (NoteName::from_index(pc).to_string().to_string(), tones.contains(&pc)))
            .collect();
        // At most four times a second: the model answers twenty-five times
        // and a note on the edge of the threshold blinked in and out.
        if (heard_name.as_str(), &notes) != (self.last_heard.0.as_str(), &self.last_heard.1)
            && self.last_heard_push.elapsed() >= Duration::from_millis(250)
        {
            self.last_heard_push = Instant::now();
            ui.set_heard_name(heard_name.clone().into());
            let model: Vec<HeardNote> = notes
                .iter()
                .map(|(name, in_chord)| HeardNote { name: name.into(), in_chord: *in_chord })
                .collect();
            ui.set_heard_notes(slint::ModelRc::new(slint::VecModel::from(model)));
            self.last_heard = (heard_name, notes);
        }
    }

    /// The note modes: the steps as a strip of names, as tiles, and on the neck.
    fn show_steps(&mut self, ui: &AppWindow, app: &MyApp, chord: &Chord) {
        let all_names = app.degree_names(chord);
        let active_indices = app.ordered_active_indices(chord);
        let mut ui_names = Vec::new();
        let mut ui_colors = Vec::new();
        for (step_idx, step) in active_indices.iter().enumerate() {
            if step.degree < all_names.len() {
                // The octave marker is display only - the model has no
                // octave, so 1 and 1' are checked identically.
                let name = shown_step(app, &all_names[step.degree], step.octave);
                ui_names.push(SharedString::from(name));
                let (r, g, b) = app.note_color(step_idx);
                ui_colors.push(Color::from_rgb_u8(r, g, b));
            }
        }
        // Order matters: the duration binding reads interval_jump when x
        // is recomputed, so the flag has to be in place first.
        // A completed set has its cursor one past the end. Keep its
        // last page visible during the green-result hold.
        let step = app.current_note_step.min(ui_names.len().saturating_sub(1)) as i32;
        let len = ui_names.len() as i32;
        let restarted = step < self.last_interval_step || len != self.last_interval_len;
        set_if_changed(ui.get_interval_jump(), restarted, |v| ui.set_interval_jump(v));
        self.last_interval_step = step;
        self.last_interval_len = len;
        set_if_changed(ui.get_interval_step(), step, |v| ui.set_interval_step(v));
        ui.set_interval_names(ModelRc::from(Rc::new(VecModel::from(ui_names))));
        ui.set_interval_colors(ModelRc::from(Rc::new(VecModel::from(ui_colors))));
        self.show_tiles(ui, app, chord, &active_indices, &all_names);
        self.draw_steps_on_neck(ui, app, chord, &active_indices);
    }

    fn show_tiles(
        &mut self,
        ui: &AppWindow,
        app: &MyApp,
        chord: &Chord,
        active_indices: &[model::Step],
        all_names: &[String],
    ) {
        // Intervals draws the steps as tiles, with the note each degree
        // falls on in this chord. Pushed only when something changed:
        // a new model every frame would rebuild every tile.
        if app.app_mode == AppMode::Intervals
            && Preview::from(ui.get_preview()) == Preview::Tiles
        {
            let key = formulas::parse_key(chord.root.to_string());
            let intervals = app.degree_intervals(chord);
            let tiles: Vec<IntervalTile> = active_indices
                .iter()
                .enumerate()
                .filter(|(_, step)| step.degree < all_names.len())
                .map(|(step_idx, step)| {
                    // Spelled by the degree's own name, so a Lydian #4 reads F#
                    // and not the Gb its semitones alone would give.
                    let note = match (&key, intervals.get(step.degree)) {
                        (Some(key), Some(&semis)) => formulas::degree_note_name(key, &all_names[step.degree])
                            .unwrap_or_else(|| formulas::note_name(key, semis as usize % 12)),
                        _ => String::new(),
                    };
                    let state = match app.note_state(step_idx) {
                        NoteState::Waiting => 0,
                        NoteState::Due => 1,
                        NoteState::Open => 2,
                        NoteState::Answering => 3,
                        NoteState::Done => 4,
                    };
                    IntervalTile {
                        degree: shown_step(app, &all_names[step.degree], step.octave).into(),
                        note: note.into(),
                        state,
                    }
                })
                .collect();
            if tiles != self.last_tiles {
                ui.set_interval_tiles(ModelRc::from(Rc::new(VecModel::from(tiles.clone()))));
                self.last_tiles = tiles;
            }
        }
    }

    /// The same phrase again, drawn on the neck: which string each degree
    /// falls on, and the ones already played lit green. Not in the fretboard
    /// trainer, which draws its own region, and not in Chords, which has its
    /// shapes.
    fn draw_steps_on_neck(&mut self, ui: &AppWindow, app: &MyApp, chord: &Chord, steps: &[model::Step]) {
        let drawable = matches!(
            app.app_mode,
            AppMode::Intervals | AppMode::Scales | AppMode::Arpeggios
        );
        if !drawable || !Preview::from(ui.get_preview()).draws_on_neck() {
            self.clear_drawing(ui);
            return;
        }
        // Built from the steps ACTUALLY drawn, not from the text they were read
        // out of: anything that reorders them - the shuffle, once - left the
        // picture standing while the strip of names above it had already moved,
        // and the change only showed up later when something else forced a
        // redraw.
        let signature = format!(
            "{}|{}|{:?}|{:?}|{}|{}|{}|{}|{}|{}|{}|{}|{:?}",
            chord.root as usize,
            chord.quality.to_string(),
            steps,
            app.collected_notes,
            ui.get_tab_frets(),
            app.voicing_anchor,
            app.random_mode,
            app.current_note_step,
            ui.get_preview(),
            app.grip_string,
            ui.get_voice_leading(),
            app.free_order(),
            app.scale_hand(chord)
        );
        if signature == self.last_tab {
            return;
        }
        self.last_tab = signature;

        // Intervals are a GRIP, not a phrase: three or four notes taken at
        // once, a string for each - laid out as a voicing, and drawn as a chord
        // box. A phrase gets the neck it is played on, or the tablature strip
        // where the setting asks for it.
        let is_grip = app.app_mode == AppMode::Intervals && (1..=4).contains(&steps.len());
        let spots = if is_grip {
            self.place_grip(ui, app, chord, steps)
        } else {
            place_phrase(app, chord, steps)
        };
        self.last_voicing = if is_grip { spots.clone() } else { Vec::new() };
        let done = &this_way_only(app, chord, steps, &app.collected_notes);
        let frets = ui.get_tab_frets();
        // No ring under free order: nothing is "due", and a ring would say the
        // opposite - for a grip, and for a scale let out of its order.
        let due = app.note_due().unwrap_or(spots.len());
        let (svg, aspect) = if Preview::from(ui.get_preview()) == Preview::Tab {
            (tab::svg(&spots, done, due, frets), tab::aspect(spots.len()))
        } else {
            (tab::neck(&spots, done, due, frets), tab::neck_aspect(&spots))
        };
        ui.set_tab_aspect(aspect);
        ui.set_tab_image(svg_icon(&svg));
    }

    /// A grip, with the voices led from the grip before - unless the option is
    /// off or the shuffle is on: then the chords come in a drawn order, there
    /// is no line to lead, and each grip is taken where the neck is drawn to
    /// offer it, so the grips move about instead of settling on the bottom
    /// strings.
    fn place_grip(&self, ui: &AppWindow, app: &MyApp, chord: &Chord, steps: &[model::Step]) -> Vec<tab::Spot> {
        let lead = ui.get_voice_leading() && !app.random_mode;
        let prev = (lead && !self.last_voicing.is_empty()).then_some(self.last_voicing.as_slice());
        // The first note is the bass where the exercise is an inversion: the
        // chord's own tones, in the order written. In a structure, a line or a
        // suspension it is a melody, and the grip is just the closest one.
        let inversion = !app.random_mode
            && app.level().is_none_or(|level| {
                use crate::interval_levels::Group::*;
                matches!(level.group, OneTone | TwoTones | ThreeTones | FourTones)
            });
        tab::place_grip_voiced(
            chord.root as usize,
            &app.degree_intervals(chord),
            steps,
            &app.degree_names(chord),
            prev,
            // Led, the hand keeps to one place for the exercise; taken one by
            // one, each grip is drawn its own.
            if lead { app.home_anchor } else { app.voicing_anchor },
            app.grip_string,
            // Shuffled, the grips are close forms only - packed inside an
            // octave on neighbouring strings. With no line to lead there is
            // nothing holding them together, and spread forms drawn at random
            // read as a different chord each time rather than as one shape
            // moving.
            app.random_mode,
            inversion,
        )
    }

    fn drop_drawing_of_other_modes(&mut self, ui: &AppWindow, app: &MyApp) {
        // A drawing belongs to the mode that made it. Chords has its shape
        // diagrams and the fretboard trainer its region, and the picture left
        // over from the note modes was turning up under them.
        let draws = matches!(
            app.app_mode,
            AppMode::Intervals | AppMode::Scales | AppMode::Arpeggios | AppMode::Fretboard
        );
        if !draws {
            self.clear_drawing(ui);
        }
    }

    fn show_spectrum(&mut self, ui: &AppWindow, app: &MyApp, spectrum_vis: &[f32; 48]) {
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
        // Skipped entirely when nothing shows it: the bars are the panel's most
        // expensive content, and off by default.
        if !ui.get_show_spectrum() && !ui.get_ai_debug_visible() {
            return;
        }
        // And rationed even when shown. With signal every one of the 48 bars
        // moves on every frame, so this alone would keep the whole panel
        // redrawing at sixty frames a second; twenty is more than the eye reads
        // off a level display.
        if self.last_spectrum_push.elapsed() < Duration::from_millis(50) {
            return;
        }
        self.last_spectrum_push = Instant::now();
        let spec_colors: Vec<Color> = spectrum_vis
            .iter()
            .enumerate()
            .map(|(i, &level)| bar_colour(i, level, &targets))
            .collect();
        // Only rows that actually moved: setting a row notifies whether or not
        // the value changed, and in silence most of them do not.
        for (i, v) in spectrum_vis.iter().enumerate() {
            if self.spectrum_data.row_data(i) != Some(*v) {
                self.spectrum_data.set_row_data(i, *v);
            }
        }
        for (i, c) in spec_colors.iter().enumerate() {
            if self.spectrum_colors.row_data(i) != Some(*c) {
                self.spectrum_colors.set_row_data(i, *c);
            }
        }
    }
}

/// The model's chord over the last answers: the name with the most summed
/// confidence, and that sum per answer.
///
/// Majority voting over 3 windows. probe_quality measured majority voting as
/// better than averaging the distributions - the model is sometimes confident
/// and wrong, and averaging carries that confidence forward while voting
/// damps it.
fn vote_on_chord(history: &VecDeque<(String, f32)>) -> (String, f32) {
    let mut votes: HashMap<&str, f32> = HashMap::new();
    for (chord, score) in history {
        *votes.entry(chord.as_str()).or_insert(0.0) += *score;
    }
    match votes.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)) {
        Some((chord, sum)) => (chord.to_string(), sum / history.len() as f32),
        None => ("...".to_string(), 0.0),
    }
}

/// SOLITITO_STRUM=1: one line per answer of the model - the attack, what was
/// heard, the chord asked for and the success timer.
fn print_strum_trace(
    app: &MyApp,
    onset_id: u64,
    since_onset: u32,
    shown: &str,
    confidence: f32,
    chord_before: usize,
) {
    println!(
        "atak#{onset_id} klatek_od_ataku={since_onset:<3} slyszy={shown:<8}                          pewnosc={:.2} cel={:<8} struna={:<3} licznik={:.2}/{:.2} {}",
        confidence,
        app.chords.get(app.current_chord_index)
            .map(|c| format!("{}{}", c.root.to_string(), c.quality.to_string()))
            .unwrap_or_default(),
        app.start_hint
            .and_then(|i| state::START_STRINGS.get(i))
            .copied()
            .unwrap_or("-"),
        app.success_timer,
        app.transition_delay,
        if app.current_chord_index != chord_before { "-> PRZESZLO" } else { "" },
    );
}

/// What Formulas writes about the chord a formula is placed over.
#[derive(Default)]
struct Placement {
    chord: String,
    degree: String,
    label: String,
    against: Vec<String>,
    own: Vec<bool>,
    verdict: String,
}

fn placement_over(app: &MyApp, chord: &Chord, t: &i18n::Strings, jazz_names: bool) -> Placement {
    let root = chord.root as usize;
    let pcs = pitch_classes_of(chord);
    let word = match app.formula_verdict {
        Some(formulas::Verdict::Defines) => t.placement_defines,
        Some(formulas::Verdict::Colours) => t.placement_colours,
        Some(formulas::Verdict::Outside) => t.placement_outside,
        None => "",
    };
    Placement {
        chord: format!("{}{}", chord.root.to_string(), chord.quality.to_string()),
        degree: formulas::ROMAN[app.formula_degree].to_string(),
        label: format!("{}{}", t.against_chord, chord.root.to_string()),
        against: functions_over(app, chord, jazz_names),
        own: formulas::chord_tones_in(app.formula_mask, root, pcs, app.formula_degree),
        verdict: format!(
            "{} · {} {} {} {}",
            word, app.formula_hits, t.of_count, pcs.count_ones(), t.chord_tones
        ),
    }
}

/// The chord's pitch classes as a 12-bit mask.
fn pitch_classes_of(chord: &Chord) -> u16 {
    let root = chord.root as usize;
    chord
        .quality
        .intervals()
        .iter()
        .fold(0u16, |pcs, &i| pcs | 1 << ((root + i as usize) % 12))
}

/// The formula's functions read from the chord's root. What belongs to the
/// chord is named by the chord - its own third is "3" or "b3", not a ninth of
/// anything. What does not is a plain function, or a tension if that is asked
/// for.
fn functions_over(app: &MyApp, chord: &Chord, jazz_names: bool) -> Vec<String> {
    let mut own: [Option<&str>; 12] = [None; 12];
    let names = chord.quality.interval_names();
    for (name, &i) in names.iter().zip(chord.quality.intervals().iter()) {
        own[i as usize % 12] = Some(name.as_str());
    }
    formulas::functions_of(app.formula_mask)
        .iter()
        .map(|&f| {
            let semi = (f + app.formula_degree) % 12;
            match own[semi] {
                Some(name) => name.to_string(),
                None if jazz_names => formulas::TENSIONS[semi].to_string(),
                None => formulas::FUNCS[semi].to_string(),
            }
        })
        .collect()
}

/// A step's degree as the screen writes it. In Intervals the plain degree, as
/// the list of exercises has it - the octave is where the hand takes the note;
/// in the phrases of Scales and Arpeggios, where the run through the octave is
/// the exercise, the compound number above it.
fn shown_step(app: &MyApp, name: &str, octave: i8) -> String {
    if app.app_mode == AppMode::Intervals {
        crate::interval_levels::plain_degree(name)
    } else {
        model::shown_degree(name, octave)
    }
}

/// What a scale has played since it last turned round: on the way back the
/// notes ahead were green from the way up, and the player walked on green with
/// nothing left to show where the run was going. Only the drawing forgets -
/// the judge keeps every credit.
fn this_way_only(app: &MyApp, chord: &Chord, steps: &[model::Step], done: &[bool]) -> Vec<bool> {
    if app.app_mode != AppMode::Scales {
        return done.to_vec();
    }
    let intervals = chord.quality.intervals();
    let pitch = |s: &model::Step| intervals.get(s.degree).map_or(0, |&i| i as i32) + 12 * s.octave as i32;
    let due = app.note_due().unwrap_or(steps.len()).min(steps.len());
    // The last turn at or before the step due: where the way changed sign.
    let turn = (1..due)
        .rev()
        .find(|&i| {
            let (before, after) = (pitch(&steps[i]) - pitch(&steps[i - 1]), steps.get(i + 1).map(|n| pitch(n) - pitch(&steps[i])));
            after.is_some_and(|after| before.signum() != after.signum())
        })
        .unwrap_or(0);
    done.iter().enumerate().map(|(i, &d)| d && i >= turn).collect()
}

/// A phrase laid on the neck. A scale starts from a drawn string, after every
/// pass: the shape is the same everywhere, and one place on the neck practised
/// over and over is not the exercise. An arpeggio study keeps the position its
/// source is written in.
fn place_phrase(app: &MyApp, chord: &Chord, steps: &[model::Step]) -> Vec<tab::Spot> {
    // From E to e the hand stands where the pass drew it, and does not move.
    if let Some((hand, _)) = app.scale_hand(chord) {
        return tab::place_in_hand(
            chord.root as usize,
            &chord.quality.intervals(),
            steps,
            &chord.quality.interval_names(),
            hand,
        );
    }
    let from = (app.app_mode == AppMode::Scales).then_some(app.scale_start);
    // Read the way the steps were written: a level of Intervals counts its
    // degrees through the chord's scale, and against the chord's four tones
    // alone its tensions had nowhere to go - a chord scale drew an empty neck.
    tab::place_from(
        chord.root as usize,
        &app.degree_intervals(chord),
        steps,
        &app.degree_names(chord),
        None,
        from,
    )
}

/// Every place inside the region where pitch class `pc` sounds.
fn places_of(pc: usize, strings: &[usize], from: i32, to: i32) -> Vec<(usize, i32)> {
    strings
        .iter()
        .flat_map(|&string| {
            (from..=to).filter_map(move |fret| {
                let sounding = (tab::TUNING[string] + fret).rem_euclid(12) as usize;
                (sounding == pc % 12).then_some((string, fret))
            })
        })
        .collect()
}

/// Which rows of shapes the two switches ask for. Neither ticked draws
/// nothing, which is a legitimate answer and not the same as the master switch
/// being off: that one is about whether the row exists at all.
fn shapes_asked_for(full: bool, shell: bool) -> Option<diagrams::Shapes> {
    match (full, shell) {
        (true, true) => Some(diagrams::Shapes::Both),
        (false, true) => Some(diagrams::Shapes::Shell),
        (true, false) => Some(diagrams::Shapes::Full),
        (false, false) => None,
    }
}

/// The line under the shapes. Captioned whenever the row is not the chord's
/// own shell: a substitute taken for the chord itself is worse than no
/// substitute at all.
fn shapes_caption(t: &i18n::Strings, shapes: Option<diagrams::Shapes>, chord: &Chord) -> String {
    let r = chord.root as usize;
    match (shapes, diagrams::shell_note(&chord.quality)) {
        (None, _) => String::new(),
        // A m6 is the m7b5 a minor third below: A m6 is F# m7b5.
        (_, diagrams::ShellNote::HeardAsMinor) => {
            let minor = format!("{}m", formulas::KEY_POOL[r]);
            match chord.quality {
                ChordQuality::Minor6 => format!(
                    "{}{minor}{}{}m7b5{}",
                    t.shapes_heard_as, t.shapes_or, formulas::KEY_POOL[(r + 9) % 12], t.shapes_same_notes
                ),
                _ => format!("{}{minor}", t.shapes_heard_as),
            }
        }
        (Some(diagrams::Shapes::Full), _) => String::new(),
        (_, diagrams::ShellNote::Own) => String::new(),
        (_, diagrams::ShellNote::MinorForHalfDim) => t.shapes_substitute.to_string(),
        // Naming the four dominants beats stating the rule that finds them. A
        // diminished seventh is the third, fifth, seventh and flat ninth of a
        // dominant whose root sits a semitone below any of its notes - which
        // is four dominants, and reading that sentence is harder than reading
        // the names.
        (_, diagrams::ShellNote::None_) => {
            let names: Vec<&str> = [11usize, 2, 5, 8]
                .iter()
                .map(|d| formulas::KEY_POOL[(r + d) % 12])
                .collect();
            format!("{}{}7b9{}", t.shapes_no_shell, names.join("7b9, "), t.shapes_no_shell_end)
        }
    }
}

/// A bar of the spectrum: green where it sounds and is asked for, red where it
/// sounds and is not, and in silence dark - a shade lighter on every C.
fn bar_colour(bar: usize, level: f32, targets: &[usize]) -> Color {
    let pitch_class = (bar + 40) % 12;
    match (level > 0.05, targets.contains(&pitch_class)) {
        (true, true) => Color::from_rgb_u8(50, 255, 100),
        (true, false) => Color::from_rgb_u8(255, 50, 50),
        (false, _) if pitch_class == 0 => Color::from_rgb_u8(60, 60, 80),
        (false, _) => Color::from_rgb_u8(30, 30, 30),
    }
}

#[cfg(test)]
mod this_way_tests {
    use super::*;
    use crate::state::tests::app;
    use crate::state::{KeyMove, ScaleDirection, ScaleFrom};

    /// There and back: on the way down only what was played since the turn is
    /// green; the judge's own record is left as it was.
    #[test]
    fn the_way_back_starts_without_green() {
        let mut a = app();
        a.set_mode(AppMode::Scales as i32);
        a.set_scale_e_to_e(false);
        a.set_scale_way(ScaleDirection::Up, ScaleFrom::Root, true, KeyMove::Stays, true);
        let chord = a.chords[0].clone();
        let steps = a.ordered_active_indices(&chord);
        let top = steps.len() / 2;
        // Up to the top and two steps back down.
        for k in 0..top + 3 {
            a.collected_notes[k] = true;
        }
        a.current_note_step = top + 3;
        let shown = this_way_only(&a, &chord, &steps, &a.collected_notes);
        assert!(shown[..top].iter().all(|&d| !d), "the way up still green: {shown:?}");
        assert!(shown[top..top + 3].iter().all(|&d| d), "the way down not green: {shown:?}");
        assert!(a.collected_notes[..top].iter().all(|&d| d), "the judge forgot");
    }

    /// Every level of Intervals longer than a grip lies on the neck in full,
    /// over every chord of the tune: a chord scale once drew an empty neck.
    #[test]
    fn every_line_of_intervals_is_drawn_on_the_neck() {
        let mut a = app();
        a.set_mode(AppMode::Intervals as i32);
        for level in 0..crate::interval_levels::LEVELS.len() {
            a.set_interval_level(level);
            for chord in a.chords.clone() {
                let steps = a.ordered_active_indices(&chord);
                if steps.len() <= 4 {
                    continue;
                }
                let spots = place_phrase(&a, &chord, &steps);
                assert_eq!(spots.len(), steps.len(), "level {level} over {chord:?}");
            }
        }
    }
}
