//! Integration regressions for independent onset events and the real judge.
use super::*;
use std::time::Duration;

fn intervals() -> MyApp {
    let mut a = tests::app();
    a.app_mode = AppMode::Intervals;
    a.intervals_input = "1 3 5".into();
    a.interval_in_order = true;
    a.require_onset = true;
    a.rise_active = true;
    a.reset_logic_state();
    a
}

fn attack(a: &mut MyApp, pcs: &[usize]) {
    let at = Instant::now();
    for &pc in pcs {
        a.receive_rise(crate::rise::Event {
            frame: a.rise_frame + 1,
            pc,
            at,
        });
    }
}

fn frames(a: &mut MyApp, n: usize) {
    for _ in 0..n {
        a.rise_frame += 1;
        a.tick(0.016);
    }
}

#[test]
fn short_attack_credits_before_the_long_cqt_or_chord_window_exists() {
    let mut a = intervals();
    assert!(!a.audio_gate_open);
    attack(&mut a, &[0]);
    frames(&mut a, 10);
    assert_eq!(a.collected_notes, [true, false, false]);
    assert!(!a.rise_pending(0));
}

#[test]
fn old_cqt_pitch_and_octave_do_not_replace_an_attack() {
    let mut a = intervals();
    attack(&mut a, &[0]);
    frames(&mut a, 10);
    a.hears(Some(7));
    a.cqt_semitone = Some(43);
    a.last_pitches = [0.99; 12];
    a.last_onsets = [0.99; 12];
    a.hears(Some(0));
    assert!(!a.struck_since_credit(0));
    assert_eq!(a.sounding_by(4, Some(NoteName::E), 0.99), None);
    frames(&mut a, 30);
    assert_eq!(a.collected_notes, [true, false, false]);
}

#[test]
fn polyphonic_attack_walks_the_intervals_but_not_the_next_chord() {
    let mut a = intervals();
    a.chords = vec![a.chords[0].clone(), a.chords[0].clone()];
    a.reset_logic_state();
    attack(&mut a, &[0, 4, 7]);
    frames(&mut a, 27);
    assert!(a.collected_notes.iter().all(|&done| done));
    frames(&mut a, 100);
    assert_eq!(a.current_chord_index, 1);
    assert!(a.collected_notes.iter().all(|&done| !done));
    attack(&mut a, &[0]);
    frames(&mut a, 10);
    assert_eq!(a.collected_notes, [true, false, false]);
}

#[test]
fn unordered_polyphony_does_not_wait_for_a_chord_name_or_monophonic_cqt() {
    let mut a = intervals();
    a.interval_in_order = false;
    a.hears(Some(0));
    attack(&mut a, &[0, 4, 7]);
    frames(&mut a, 27);
    assert!(a.collected_notes.iter().all(|&done| done));
}

#[test]
fn single_notes_consume_the_whole_attack_but_accept_a_new_pluck() {
    let mut a = intervals();
    a.single_notes = true;
    attack(&mut a, &[0, 4, 7]);
    frames(&mut a, 30);
    assert_eq!(a.collected_notes, [true, false, false]);
    attack(&mut a, &[4]);
    frames(&mut a, 10);
    assert_eq!(a.collected_notes, [true, true, false]);
}

#[test]
fn duplicate_delayed_and_expired_events_cannot_cross_a_boundary() {
    let mut a = intervals();
    let event = crate::rise::Event {
        frame: 1,
        pc: 0,
        at: Instant::now(),
    };
    a.receive_rise(event);
    frames(&mut a, 10);
    a.receive_rise(event);
    assert!(!a.rise_pending(0));
    let late = crate::rise::Event {
        frame: 2,
        pc: 4,
        at: Instant::now(),
    };
    a.forget_what_was_heard();
    a.receive_rise(late);
    assert!(!a.rise_pending(4));
    assert_eq!(
        a.strike_id[4], 0,
        "a late event must not release the repeated-chord latch"
    );
    a.rise_after = Instant::now() - Duration::from_secs(2);
    a.receive_rise(crate::rise::Event {
        frame: 3,
        pc: 7,
        at: Instant::now() - Duration::from_secs(1),
    });
    assert!(!a.rise_pending(7));
}

#[test]
fn scales_and_arpeggios_advance_without_waiting_for_the_chord_model() {
    for mode in [AppMode::Scales, AppMode::Arpeggios] {
        let mut a = intervals();
        a.app_mode = mode;
        attack(&mut a, &[0]);
        frames(&mut a, 10);
        assert!(a.collected_notes[0], "{mode:?}");
        // Old AI results cannot count a second time on a different clock.
        for _ in 0..10 {
            a.check_progress_with_ai(0.04, "C Maj7", 0.99);
        }
        assert_eq!(a.collected_notes, [true, false, false]);
    }
}

#[test]
fn formula_accepts_polyphony_only_when_the_onset_option_is_enabled() {
    let mut a = intervals();
    a.app_mode = AppMode::Formulas;
    a.formula_root = 0;
    a.formula_mask = (1 << 0) | (1 << 4) | (1 << 7);
    a.formula_collected = vec![false; 3];
    a.strict_formulas = true;
    a.lap_hold = 0.0;
    attack(&mut a, &[0, 4, 7]);
    frames(&mut a, 1);
    assert!(a.formula_collected.iter().all(|&done| done));
    a.restart_formula();
    a.lap_hold = 0.0;
    frames(&mut a, 1);
    assert!(a.formula_collected.iter().all(|&done| !done));
}

#[test]
fn stopping_audio_cannot_complete_a_partial_credit() {
    let mut a = intervals();
    attack(&mut a, &[0]);
    frames(&mut a, 3);
    for _ in 0..100 {
        a.tick(0.016);
    }
    assert!(a.collected_notes.iter().all(|&done| !done));
}

#[test]
fn input_restart_or_discontinuity_discards_pending_attacks() {
    let mut a = intervals();
    {
        let mut shared = a.analysis_state.lock().unwrap();
        shared.rise.enabled = true;
        shared.rise.invalidate();
    }
    a.sync_audio_settings();
    attack(&mut a, &[0]);
    assert!(a.rise_pending(0));
    {
        let mut shared = a.analysis_state.lock().unwrap();
        shared.rise.invalidated_at = Some(Instant::now());
    }
    a.sync_audio_settings();
    assert!(!a.rise_pending(0));
    attack(&mut a, &[4]);
    assert!(a.rise_pending(4));
    a.analysis_state.lock().unwrap().rise.invalidate();
    a.sync_audio_settings();
    assert!(!a.rise_pending(4));
}

#[test]
fn changing_note_policy_does_not_reuse_a_previous_attack() {
    let mut a = intervals();
    attack(&mut a, &[0]);
    assert!(a.rise_pending(0));
    a.set_note_policy(true, true);
    assert!(!a.rise_pending(0));
}

#[test]
#[ignore = "live worker integration: needs Rise model and SOLITITO_RISE_WAV (AtoA)"]
fn real_worker_delivers_a_pluck_to_the_judge_without_the_chord_model() -> anyhow::Result<()> {
    real_worker_credits_root(NoteName::A)
}

#[test]
#[ignore = "live rescue integration: needs SOLITITO_RISE_WAV with a weak E pluck and SOLITITO_ONSET_RESCUE=1"]
fn real_worker_credits_a_weak_e_with_rescue() -> anyhow::Result<()> {
    anyhow::ensure!(std::env::var("SOLITITO_ONSET_RESCUE").as_deref() == Ok("1"), "Enable rescue for this test");
    real_worker_credits_root(NoteName::E)
}

fn real_worker_credits_root(root: NoteName) -> anyhow::Result<()> {
    let (audio, rate) = crate::rise::read_wav(&std::env::var("SOLITITO_RISE_WAV")?)?;
    let mut a = intervals();
    a.chords[0].root = root;
    a.reset_logic_state();
    a.noise_gate = 0.0;
    let mut input = crate::rise::start(a.analysis_state.clone(), rate)?
        .ok_or_else(|| anyhow::anyhow!("Rise is disabled"))?;
    a.sync_audio_settings();
    let count = (rate as usize * 2).min(audio.len());
    for chunk in audio[..count].chunks(rate as usize * 16 / 1000) {
        input.push(chunk);
        std::thread::sleep(Duration::from_millis(16));
        a.sync_audio_settings();
        a.tick(0.016);
    }
    assert!(
        a.rise_frame >= 115,
        "worker stalled at frame {}",
        a.rise_frame
    );
    assert_eq!(a.collected_notes, [true, false, false]);
    assert_eq!(
        a.audio_frames, 0,
        "test must not obtain evidence from the CQT path"
    );
    Ok(())
}
