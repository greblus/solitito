//! What the window's controls do: every `on_*` callback, grouped by the part of
//! the window it belongs to.
//!
//! A setting is saved the moment it changes - there is no "Save" button, so it
//! has to survive closing the window without an extra step.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use slint::{ComponentHandle, ModelRc, PhysicalSize, SharedString, VecModel};

use crate::audio::{AudioAnalysis, InputInfo};
use crate::i18n::{self, Lang};
use crate::settings::{self, Settings};
use crate::state::{AppMode, ArpExercise, MyApp};
use crate::{formulas, model, AppWindow};
use crate::devices::{channel_choices, open_input, rescan_devices};
use crate::{apply_language, fit_ui_to_window, library_lists, mode_labels};

/// Everything the callbacks reach besides the window itself.
pub struct Shared {
    pub app: Arc<Mutex<MyApp>>,
    pub cfg: Rc<RefCell<Settings>>,
    pub analysis: Arc<Mutex<AudioAnalysis>>,
    /// The open input stream, held so it can be swapped for another device.
    pub audio_in: Rc<RefCell<Option<cpal::Stream>>>,
    pub opened: Rc<RefCell<InputInfo>>,
    pub device_names: Rc<RefCell<Vec<String>>>,
}

pub fn connect(ui: &AppWindow, shared: &Shared) {
    connect_window(ui, &shared.cfg);
    connect_audio(ui, shared);
    connect_switches(ui, &shared.cfg);
    connect_practice(ui, shared);
    connect_favourites(ui, shared);
    connect_language(ui, shared);
    connect_navigation(ui, shared);
}

/// A callback that writes one setting and saves the file.
fn saving<T>(
    cfg: &Rc<RefCell<Settings>>,
    write: impl Fn(&mut Settings, T) + 'static,
) -> impl FnMut(T) + 'static {
    let cfg = Rc::clone(cfg);
    move |value| {
        let mut cfg = cfg.borrow_mut();
        write(&mut cfg, value);
        cfg.save();
    }
}

/// The options that only need to be remembered: the frame reads them from the
/// window, or the window reads them itself.
fn connect_switches(ui: &AppWindow, cfg: &Rc<RefCell<Settings>>) {
    ui.on_show_spectrum_changed(saving(cfg, |s, on| s.show_spectrum = on));
    ui.on_ai_debug_changed(saving(cfg, |s, on| s.ai_debug = on));
    ui.on_short_verdict_changed(saving(cfg, |s, on| s.short_verdict = on));
    ui.on_single_notes_changed(saving(cfg, |s, on| s.single_notes = on));
    ui.on_scale_repeat_root_changed(saving(cfg, |s, on| s.scale_repeat_root = on));
    {
        let cur = cfg.clone();
        let uw = ui.as_weak();
        ui.on_scale_way_changed(move || {
            let Some(ui) = uw.upgrade() else { return };
            let mut cur = cur.borrow_mut();
            cur.scale_direction = ui.get_scale_direction().max(0) as usize;
            cur.scale_from = ui.get_scale_from().max(0) as usize;
            cur.scale_there_and_back = ui.get_scale_there_and_back();
            cur.key_move = ui.get_key_move().max(0) as usize;
            cur.scale_in_order = ui.get_scale_in_order();
            cur.scale_e_to_e = ui.get_scale_e_to_e();
            cur.save();
        });
    }
    ui.on_arp_direction_changed(saving(cfg, |s, idx: i32| s.arp_direction = idx.max(0) as usize));
    ui.on_arp_quality_changed(saving(cfg, |s, idx: i32| s.arp_quality = idx.max(0) as usize));
    ui.on_tab_frets_changed(saving(cfg, |s, on| s.tab_frets = on));
    ui.on_voice_leading_changed(saving(cfg, |s, on| s.voice_leading = on));
    ui.on_lead_line_changed(saving(cfg, |s, on| s.lead_line = on));
    ui.on_show_diagrams_changed(saving(cfg, |s, on| s.show_diagrams = on));
    ui.on_formula_jazz_names_changed(saving(cfg, |s, on| s.formula_jazz_names = on));
    ui.on_formula_exercise_changed(saving(cfg, |s, v: i32| s.formula_exercise = v.clamp(0, 2) as u8));
    ui.on_formula_placement_changed(saving(cfg, |s, v: i32| s.formula_placement = v.clamp(0, 3) as u8));
    ui.on_formula_notes_changed(saving(cfg, |s, n: i32| s.formula_notes = n.clamp(1, 12) as usize));
    ui.on_formula_random_key_changed(saving(cfg, |s, on| s.formula_random_key = on));
    ui.on_formula_show_names_changed(saving(cfg, |s, on| s.formula_note_names = on));
    ui.on_formula_show_similar_changed(saving(cfg, |s, on| s.formula_show_similar = on));
    ui.on_formula_show_chords_changed(saving(cfg, |s, on| s.formula_show_chords = on));
    ui.on_in_order_changed(saving(cfg, |s, on| s.formula_in_order = on));
    ui.on_interval_in_order_changed(saving(cfg, |s, on| s.interval_in_order = on));
    ui.on_debug_console_changed(saving(cfg, |s, on| s.debug_console = on));
    ui.on_show_input_meter_changed(saving(cfg, |s, on| s.show_input_meter = on));
    ui.on_show_heard_changed(saving(cfg, |s, on| s.show_heard = on));
    ui.on_quick_open_changed(saving(cfg, |s, on| s.quick_options_open = on));
    ui.on_shuffle_chords_changed(saving(cfg, |s, on| s.shuffle_chords = on));
    ui.on_startup_mode_changed(saving(cfg, |s, mode: i32| s.startup_mode = mode));
}

/// The window's size and full screen, remembered for the next start.
fn connect_window(ui: &AppWindow, cfg: &Rc<RefCell<Settings>>) {
    // The size to write back on the way out. Maximising is not resizing: the
    // window then has the screen's size, and saving that turned a maximise
    // followed by a close into a window that opens the size of the desktop -
    // and lost the size that had been set by hand. Only sizes seen while the
    // window is its own are remembered.
    let last_own_size: Rc<Cell<Option<(u32, u32)>>> = Rc::new(Cell::new(None));
    // The size the window had maximised. A first launch opens maximised, and
    // has no size of its own to go back to - see `own_size_after_maximise`.
    let maximised_size: Rc<Cell<Option<(u32, u32)>>> = Rc::new(Cell::new(None));
    let first_launch = cfg.borrow().window_w.is_none();
    {
        let uw = ui.as_weak();
        let own = last_own_size.clone();
        let maximised = maximised_size.clone();
        ui.on_window_resized(move || {
            let Some(ui) = uw.upgrade() else { return };
            fit_ui_to_window(&ui);
            let window = ui.window();
            let size = window.size();
            if size.width == 0 || size.height == 0 || window.is_fullscreen() {
                return;
            }
            if window.is_maximized() {
                maximised.set(Some((size.width, size.height)));
                return;
            }
            if own.get().is_none() {
                if let Some(screen) = maximised.get() {
                    let (w, h) = own_size_after_maximise(screen);
                    window.set_size(PhysicalSize::new(w, h));
                    own.set(Some((w, h)));
                    return;
                }
                // Before the start-up maximise has taken: the size the window
                // was created at, not one anybody chose.
                if first_launch {
                    return;
                }
            }
            own.set(Some((size.width, size.height)));
        });
    }
    {
        let uw = ui.as_weak();
        let cfg = cfg.clone();
        ui.on_toggle_full_screen(move || {
            let Some(ui) = uw.upgrade() else { return };
            let on = !ui.window().is_fullscreen();
            ui.window().set_fullscreen(on);
            let mut cfg = cfg.borrow_mut();
            cfg.fullscreen = on;
            cfg.save();
        });
    }
    {
        let uw = ui.as_weak();
        let cfg = cfg.clone();
        ui.on_leave_full_screen(move || {
            let Some(ui) = uw.upgrade() else { return false };
            if !ui.window().is_fullscreen() {
                return false;
            }
            ui.window().set_fullscreen(false);
            let mut cfg = cfg.borrow_mut();
            cfg.fullscreen = false;
            cfg.save();
            true
        });
    }

    // On close rather than on every resize: dragging an edge produces a stream
    // of sizes, and only the last one is the answer.
    let uw = ui.as_weak();
    let cfg = cfg.clone();
    ui.window().on_close_requested(move || {
        if let Some(ui) = uw.upgrade() {
            // Closed maximised or on full screen, with no size of its own seen
            // this run: the file keeps what it had rather than learning the
            // screen's size.
            let window = ui.window();
            let size = window.size();
            let size = match last_own_size.get() {
                Some(s) => Some(s),
                None if window.is_maximized() || window.is_fullscreen() => None,
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

/// The size a window that has only ever been maximised takes when it is not:
/// three quarters of the screen it filled, so it keeps the screen's proportions
/// instead of falling back to the narrow size it was created at.
fn own_size_after_maximise((w, h): (u32, u32)) -> (u32, u32) {
    (w * 3 / 4, h * 3 / 4)
}

/// The input device and channel.
fn connect_audio(ui: &AppWindow, shared: &Shared) {
    {
        // Fired as the picker opens, before its list is drawn: an interface
        // plugged in while the app was running is on the list by the time it
        // appears, without a restart and without a Refresh button to find.
        let names = shared.device_names.clone();
        let opened = shared.opened.clone();
        let cfg = shared.cfg.clone();
        let uw = ui.as_weak();
        ui.on_audio_devices_rescan(move || {
            let Some(ui) = uw.upgrade() else { return };
            let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
            let open_now = (!opened.borrow().fell_back)
                .then(|| cfg.borrow().audio_device.clone())
                .flatten();
            rescan_devices(&ui, &names, t.audio_default, open_now.as_deref());
        });
    }
    {
        let cur = shared.cfg.clone();
        let state = shared.analysis.clone();
        let holder = shared.audio_in.clone();
        let info = shared.opened.clone();
        let names = shared.device_names.clone();
        let uw = ui.as_weak();
        ui.on_audio_device_changed(move |idx| {
            let Some(ui) = uw.upgrade() else { return };
            // From the list already on screen: enumerating again would set ALSA
            // probing every plugin a second time, for names the user is looking
            // at right now.
            let name = if idx > 0 { names.borrow().get(idx as usize - 1).cloned() } else { None };
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
    let cur = shared.cfg.clone();
    let state = shared.analysis.clone();
    let holder = shared.audio_in.clone();
    let info = shared.opened.clone();
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

/// Practice options that do more than remember themselves.
fn connect_practice(ui: &AppWindow, shared: &Shared) {
    {
        let cur = shared.cfg.clone();
        let app = shared.app.clone();
        ui.on_interval_level_changed(move |idx| {
            let index = idx.max(0) as usize;
            app.lock().unwrap().set_interval_level(index);
            let mut cur = cur.borrow_mut();
            cur.interval_level = index;
            cur.save();
        });
    }
    {
        let cur = shared.cfg.clone();
        let app = shared.app.clone();
        let uw = ui.as_weak();
        // The exercise decides what the two combos hold, so they are dealt
        // again the moment it changes - and the exercise itself has to
        // reach the state before the lists are built from it.
        ui.on_arp_exercise_changed(move |idx| {
            {
                let mut cur = cur.borrow_mut();
                cur.arp_exercise = idx.max(0) as usize;
                cur.save();
            }
            let mut app = app.lock().unwrap();
            app.arp_exercise = (idx.max(0) as usize).into();
            // Over the changes the first combo turns into the library of
            // tunes, and it opens on the standard being worked on.
            app.selected_library_idx = if app.reads_songs() { app.saved_song() } else { 0 };
            app.reload_library();
            if let Some(ui) = uw.upgrade() {
                show_library(&ui, &app);
                ui.set_interval_input_text(model::shown_phrase(&app.intervals_input).into());
            }
        });
    }
    {
        let cur = shared.cfg.clone();
        let uw = ui.as_weak();
        // Against the mode showing now: the dropdown sits in that mode's
        // own section of the panel, and what it sets belongs to it alone.
        ui.on_preview_changed(move |idx| {
            let Some(ui) = uw.upgrade() else { return };
            let mut cur = cur.borrow_mut();
            if cur.set_preview_for(ui.get_current_mode(), idx.max(0) as usize) {
                cur.save();
            }
        });
    }
    {
        let cur = shared.cfg.clone();
        ui.on_shapes_choice_changed(move |full: bool, shell: bool| {
            let mut cur = cur.borrow_mut();
            cur.show_full_shapes = full;
            cur.show_shell_shapes = shell;
            cur.save();
        });
    }
    let cur = shared.cfg.clone();
    let uw = ui.as_weak();
    ui.on_require_onset_changed(move |on| {
        let Some(ui) = uw.upgrade() else { return };
        let mut cur = cur.borrow_mut();
        cur.set_require_onset_for(ui.get_current_mode(), on);
        cur.save();
    });
}

/// The star under a formula and the list of kept formulas.
fn connect_favourites(ui: &AppWindow, shared: &Shared) {
    {
        // The star: keep what is on screen, or let go of it. Kept formulas are
        // matched on the set AND the key, because the same functions read from
        // another root are a different exercise.
        let cur = shared.cfg.clone();
        let app = shared.app.clone();
        let uw = ui.as_weak();
        ui.on_star_clicked(move || {
            let Some(ui) = uw.upgrade() else { return };
            let mask = app.lock().unwrap().formula_mask;
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
        let cur = shared.cfg.clone();
        let app = shared.app.clone();
        ui.on_favourite_named(move |name| {
            let mask = app.lock().unwrap().formula_mask;
            let name = match name.trim() {
                "" => formulas::to_text(mask),
                named => named.to_string(),
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
        let cur = shared.cfg.clone();
        let app = shared.app.clone();
        let uw = ui.as_weak();
        ui.on_favourite_search(move || {
            let Some(ui) = uw.upgrade() else { return };
            let filter = ui.get_favourite_filter();
            let first = cur.borrow().favourites_matching(&filter).next().map(|f| f.mask);
            if let Some(mask) = first {
                app.lock().unwrap().load_formula(mask);
            }
        });
    }
    {
        // Picking one draws it. The index is into the FILTERED list, so the
        // filter is applied again here rather than trusted to have stood still.
        let cur = shared.cfg.clone();
        let app = shared.app.clone();
        let uw = ui.as_weak();
        ui.on_favourite_picked(move |idx| {
            let Some(ui) = uw.upgrade() else { return };
            let filter = ui.get_favourite_filter();
            let picked = cur.borrow().favourites_matching(&filter).nth(idx.max(0) as usize).map(|f| f.mask);
            if let Some(mask) = picked {
                app.lock().unwrap().load_formula(mask);
            }
        });
    }
    // The cross on a row. Same walk as picking one - the index is into the
    // filtered list - and then it is gone from the file as well as the list.
    let cur = shared.cfg.clone();
    let uw = ui.as_weak();
    ui.on_favourite_deleted(move |idx| {
        let Some(ui) = uw.upgrade() else { return };
        let mut cur = cur.borrow_mut();
        let filter = ui.get_favourite_filter();
        let Some(mask) = cur.favourites_matching(&filter).nth(idx.max(0) as usize).map(|f| f.mask)
        else {
            return;
        };
        cur.favourites.retain(|f| f.mask != mask);
        cur.save();
    });
}

fn connect_language(ui: &AppWindow, shared: &Shared) {
    let cur = shared.cfg.clone();
    let app = shared.app.clone();
    let uw = ui.as_weak();
    ui.on_language_changed(move |idx| {
        {
            let mut cur = cur.borrow_mut();
            cur.language = idx;
            cur.save();
        }
        // Strings are swapped immediately; no restart needed.
        if let Some(ui) = uw.upgrade() {
            let app = app.lock().unwrap();
            apply_language(&ui, Lang::from_setting(idx), Some(&app));
        }
    });
}

/// The side bar's modes, the arrows beside the pause, and the two combos.
fn connect_navigation(ui: &AppWindow, shared: &Shared) {
    {
        let app = shared.app.clone();
        let cfg = shared.cfg.clone();
        let uw = ui.as_weak();
        ui.on_mode_changed(move |mode_idx| {
            let mut app = app.lock().unwrap();
            let Some(ui) = uw.upgrade() else { return };
            app.set_mode(mode_idx);
            ui.set_current_mode(mode_idx);
            // Each mode is shown the way it was last shown. Read after the mode is
            // set, or the dropdown in the panel would still be answering for the
            // one being left.
            ui.set_preview(cfg.borrow().preview_for(mode_idx) as i32);
            app.require_onset = cfg.borrow().require_onset_for(mode_idx);
            ui.set_require_onset(app.require_onset);
            ui.set_interval_input_text(model::shown_phrase(&app.intervals_input).into());
            show_library(&ui, &app);
        });
    }
    {
        let app = shared.app.clone();
        ui.on_step_chord(move |delta| app.lock().unwrap().step_chord(delta));
    }
    {
        let app = shared.app.clone();
        let uw = ui.as_weak();
        // Paused first: a chord jumped to is a chord to look at and play again,
        // and the exercise must not run on from it on its own.
        ui.on_go_to_chord(move |offset| {
            if let Some(ui) = uw.upgrade() {
                ui.set_paused(true);
            }
            let mut app = app.lock().unwrap();
            app.paused = true;
            app.step_chord(offset);
        });
    }
    {
        let app = shared.app.clone();
        ui.on_next_change(move || app.lock().unwrap().next_change());
    }
    {
        let app = shared.app.clone();
        let cfg = shared.cfg.clone();
        let uw = ui.as_weak();
        ui.on_item_selected(move |index| {
            let mut app = app.lock().unwrap();
            let Some(ui) = uw.upgrade() else { return };
            take_study_quality(&ui, &mut app, &cfg, index);
            app.item_selected(index);
            remember_song_or_scale(&app, &cfg);
            if app.app_mode == AppMode::Scales || app.app_mode == AppMode::Arpeggios {
                ui.set_interval_input_text(model::shown_phrase(&app.intervals_input).into());
            }
        });
    }
    let app = shared.app.clone();
    let uw = ui.as_weak();
    ui.on_secondary_item_selected(move |index| {
        let mut app = app.lock().unwrap();
        let Some(ui) = uw.upgrade() else { return };
        app.secondary_item_selected(index);
        if app.app_mode == AppMode::Scales || app.app_mode == AppMode::Arpeggios {
            ui.set_interval_input_text(model::shown_phrase(&app.intervals_input).into());
        }
    });
}

/// The two combos under the title, filled for the mode on screen and opened on
/// what is chosen. The language as it is now, not as it was at startup.
fn show_library(ui: &AppWindow, app: &MyApp) {
    let lang = Lang::from_setting(ui.get_language_idx());
    let t = i18n::strings(lang);
    let (label, sec_label) = mode_labels(&t, app.app_mode, app.arp_exercise == ArpExercise::Study);
    let (items, sec_items) = library_lists(app, lang);
    ui.set_library_label(label.into());
    ui.set_library_items(ModelRc::from(Rc::new(VecModel::from(items))));
    // Not a hard 0: a mode that reads the tunes opens on the standard being
    // worked on, and returning to Arpeggios restores the pattern chosen.
    ui.set_current_item_index(app.selected_library_idx as i32);
    ui.set_secondary_label(sec_label.into());
    ui.set_secondary_items(ModelRc::from(Rc::new(VecModel::from(sec_items))));
    ui.set_current_secondary_index(app.secondary_index as i32);
}

/// A study named for a chord is written over that chord - see
/// `model::study_quality`. Picking it and then having to set the chord by hand
/// was a trap: the name promised a quality the app did not set, and the shape
/// on the neck was the phrase read over something else.
fn take_study_quality(ui: &AppWindow, app: &mut MyApp, cfg: &RefCell<Settings>, index: i32) {
    if app.app_mode != AppMode::Arpeggios || app.arp_exercise != ArpExercise::Study {
        return;
    }
    let Some(quality) = app
        .arpeggio_patterns
        .get(index.max(0) as usize)
        .and_then(|p| model::study_quality(&p.name))
    else {
        return;
    };
    app.arp_quality = quality;
    ui.set_arp_quality(quality as i32);
    let mut cfg = cfg.borrow_mut();
    cfg.arp_quality = quality;
    cfg.save();
}

/// A standard - and a scale - is worked on over sessions, not over a sitting.
/// `song_title` carries whichever the mode is reading.
fn remember_song_or_scale(app: &MyApp, cfg: &RefCell<Settings>) {
    let mut cfg = cfg.borrow_mut();
    let slot = if app.reads_songs() {
        &mut cfg.song
    } else if app.app_mode == AppMode::Scales {
        &mut cfg.scale
    } else {
        return;
    };
    if *slot != app.song_title {
        *slot = app.song_title.clone();
        cfg.save();
    }
}
