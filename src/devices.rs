//! The audio input in the settings panel: opening a device, and the list of
//! devices and channels the pickers show.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use slint::{ModelRc, SharedString, VecModel};

use crate::audio::{self, start_audio_stream, AudioAnalysis, InputInfo};
use crate::i18n::{self, Lang};
use crate::AppWindow;

/// Opens the input and reports what actually came up.
///
/// The report matters as much as the stream: on Windows the console is hidden,
/// so a stream that failed to open was indistinguishable from one that opened
/// and heard nothing - the interface's own meters kept showing signal either
/// way. Now the settings panel says which device, rate, channel count and
/// sample format are in use, or why there is none.
pub(crate) fn open_input(
    state: &Arc<Mutex<AudioAnalysis>>,
    holder: &Rc<RefCell<Option<cpal::Stream>>>,
    opened: &Rc<RefCell<InputInfo>>,
    ui: &AppWindow,
    device: Option<&str>,
    channel: usize,
) {
    // Dropped before opening the next: some backends refuse a second stream on
    // a device that already has one.
    holder.borrow_mut().take();
    match start_audio_stream(state.clone(), device, channel) {
        Ok((stream, info)) => {
            let mut line = format!(
                "{} · {} Hz · {} ch · {}",
                info.name, info.sample_rate, info.channels, info.format
            );
            if info.fell_back {
                // Said in words, not just marked: the app still hears something,
                // so nothing looks wrong - except that the channel picker now
                // belongs to a device the user did not choose.
                let t = i18n::strings(Lang::from_setting(ui.get_language_idx()));
                line = format!("⚠ {} - {line}", t.audio_busy);
            }
            println!("🎧 {line}");
            ui.set_audio_info(line.into());
            ui.set_audio_channel_count(info.channels as i32);
            *opened.borrow_mut() = info;
            *holder.borrow_mut() = Some(stream);
        }
        Err(e) => {
            eprintln!("ERR AUDIO IN: {e}");
            ui.set_audio_info(format!("✖ {e}").into());
            ui.set_audio_channel_count(0);
            *opened.borrow_mut() = InputInfo::default();
        }
    }
}

/// Adds newly seen devices to the list already on screen, and says whether
/// anything was added.
///
/// The list only ever grows. ALSA hands out a card once, so anything holding
/// one - PipeWire, another app, or this app's own stream - drops it from the
/// enumeration entirely; a plain re-scan would leave the picker showing
/// `default`, `pulse`, `pipewire` and `jack` and nothing else, and would take
/// the user's own choice with it.
fn merge_devices(known: &mut Vec<String>, fresh: Vec<String>) -> bool {
    let before = known.len();
    for name in fresh {
        if !known.contains(&name) {
            known.push(name);
        }
    }
    known.len() != before
}

/// Is this entry a device the backend can currently see?
///
/// The device we are recording from counts as present even though no scan
/// reports it: a card can be opened once, and we are the ones holding it. Left
/// out, the picker marks the interface you are playing through as unavailable
/// the moment it starts working.
fn device_is_present(name: &str, scan: &[String], open_now: Option<&str>) -> bool {
    scan.iter().any(|n| n == name) || open_now == Some(name)
}

/// Puts "system default" at the head of the device list, so index 0 always
/// means "whatever the OS picks" whether or not any devices were found.
///
/// `present` is the latest scan. Anything in `names` that it does not contain is
/// remembered rather than seen - the saved device while the interface is still
/// in its bag, most often - and is marked, because an entry that reads like any
/// other is a promise the list cannot keep.
pub(crate) fn fill_device_list(
    ui: &AppWindow,
    names: &[String],
    default_label: &str,
    present: &[String],
    open_now: Option<&str>,
) {
    let mut items: Vec<SharedString> = vec![SharedString::from(default_label)];
    // Shortened for display only. The full names stay in `device_names`, which
    // is what gets matched and saved - a truncated one would find nothing on
    // the next launch.
    items.extend(names.iter().map(|n| {
        let short = short_device_name(n);
        if device_is_present(n, present, open_now) {
            SharedString::from(short)
        } else {
            SharedString::from(format!("⚠ {short}"))
        }
    }));
    ui.set_audio_devices(ModelRc::from(Rc::new(VecModel::from(items))));
}

/// Re-scans the inputs and refreshes the picker, keeping the current choice.
///
/// Cheap enough to do whenever the list is about to be read, which is the point:
/// an interface plugged in after the app started has to appear without a
/// restart, and the one that was unplugged has to stop looking available.
pub(crate) fn rescan_devices(
    ui: &AppWindow,
    known: &Rc<RefCell<Vec<String>>>,
    default_label: &str,
    open_now: Option<&str>,
) {
    let present = audio::list_input_devices();
    let mut names = known.borrow_mut();
    merge_devices(&mut names, present.clone());
    // Merging only appends, so no index shifts under the user - but replacing
    // the model does reset the picker, hence putting it back.
    let chosen = ui.get_audio_device_index();
    fill_device_list(ui, &names, default_label, &present, open_now);
    ui.set_audio_device_index(chosen);
}

/// The channel picker's entries for a device with `count` inputs, numbered from
/// one so they match what is printed on the box.
pub(crate) fn channel_choices(count: i32, one_label: &str) -> Vec<SharedString> {
    (1..=count.max(1))
        .map(|c| SharedString::from(format!("{one_label} {c}")))
        .collect()
}

/// A device name short enough for the picker, without losing what tells two of
/// them apart.
///
/// Backend names are not written for people - ALSA reports the likes of
/// `alsa_input.usb-BEHRINGER_UMC204HD_192k-00.analog-stereo`. The prefix says
/// nothing, and what separates two sockets of the same interface sits at the
/// END, so the middle is what goes.
fn short_device_name(name: &str) -> String {
    const MAX: usize = 36;
    let n = name
        .trim_start_matches("alsa_input.")
        .trim_start_matches("alsa_output.");
    let chars: Vec<char> = n.chars().collect();
    if chars.len() <= MAX {
        return n.to_string();
    }
    let head: String = chars[..MAX / 2 - 1].iter().collect();
    let tail: String = chars[chars.len() - (MAX / 2 - 1)..].iter().collect();
    format!("{head}…{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A name has to stay recognisable and, more to the point, has to stay
    /// DIFFERENT from the other sockets on the same interface - which is what a
    /// plain truncation would destroy, since they differ only at the end.
    #[test]
    fn shortening_keeps_devices_apart() {
        let a = "alsa_input.usb-BEHRINGER_UMC204HD_192k-00.analog-stereo-input-1";
        let b = "alsa_input.usb-BEHRINGER_UMC204HD_192k-00.analog-stereo-input-2";
        let (sa, sb) = (short_device_name(a), short_device_name(b));
        assert_ne!(sa, sb, "two inputs of one interface collapsed to the same label");
        assert!(sa.chars().count() <= 36, "still too long: {sa}");
        assert!(!sa.starts_with("alsa_input."), "the prefix that says nothing survived");
    }

    /// The interface you are playing through is missing from every scan for the
    /// same reason it is working: we hold the card. Marking it unavailable was
    /// the bug - the label appeared the moment the device started being used.
    #[test]
    fn the_device_we_are_recording_from_counts_as_present() {
        let scan = vec!["default".to_string(), "pipewire".to_string()];
        assert!(device_is_present("default", &scan, None));
        assert!(
            !device_is_present("sysdefault:CARD=U192k", &scan, None),
            "a card nobody can see is not present"
        );
        assert!(
            device_is_present("sysdefault:CARD=U192k", &scan, Some("sysdefault:CARD=U192k")),
            "the card we are recording from was marked unavailable"
        );
        // Holding one card says nothing about another.
        assert!(!device_is_present("hw:CARD=X", &scan, Some("sysdefault:CARD=U192k")));
    }

    /// A re-scan must never take a device off the list. ALSA leaves out
    /// everything that is currently held - including the card this app is
    /// recording from - so a scan that replaced the list would drop the user's
    /// own device the moment it started working.
    #[test]
    fn a_rescan_only_ever_adds() {
        let mut known = vec!["default".to_string(), "sysdefault:CARD=U192k".to_string()];
        // The card is busy now, so the scan cannot see it.
        assert!(!merge_devices(&mut known, vec!["default".into()]), "nothing new to report");
        assert_eq!(known.len(), 2, "the busy card was dropped from the picker");
        // A newly plugged-in interface shows up without a restart.
        assert!(merge_devices(&mut known, vec!["default".into(), "hw:CARD=X".into()]));
        assert_eq!(known, ["default", "sysdefault:CARD=U192k", "hw:CARD=X"]);
    }

    /// Short names are left exactly as they are - Windows reports readable ones.
    #[test]
    fn a_short_name_is_left_alone() {
        let n = "Line In (BEHRINGER UMC204HD)";
        assert_eq!(short_device_name(n), n);
    }

    /// One entry per input, numbered from one, and never an empty picker.
    #[test]
    fn channels_are_numbered_from_one() {
        let c = channel_choices(2, "Channel");
        assert_eq!(c, vec!["Channel 1", "Channel 2"]);
        assert_eq!(channel_choices(0, "Channel").len(), 1, "a picker with nothing in it");
    }
}
