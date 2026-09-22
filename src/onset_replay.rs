//! Deterministic diagnostic inputs to the real interval judge. These tests
//! isolate policy from DSP: CQT and onset probabilities are supplied explicitly.
//! Audio advances every 16 ms; model delivery is sampled every 48 ms (three
//! hops), with all three possible phases tested. This is not a wall-clock
//! simulation of the live worker's nominal 40 ms period.

use super::{tests::app, *};

const DT: f32 = 0.016;
const ROOT: usize = 11; // B
const THIRD: usize = 2; // D
const FIFTH: usize = 6; // F#

struct Replay {
    app: MyApp,
    frame: usize,
    phase: usize,
    deliveries: Vec<(usize, usize)>,
    credits: Vec<(usize, usize, usize)>, // frame, chord index, step
}

impl Replay {
    fn new(intervals: &str, ordered: bool, phase: usize) -> Self {
        let mut app = app();
        app.set_mode(AppMode::Intervals as i32);
        app.random_mode = false;
        app.shuffle_chords = false;
        app.interval_in_order = ordered;
        app.require_onset = true;
        app.intervals_input = intervals.into();
        app.chords = vec![
            Chord {
                root: NoteName::B,
                quality: ChordQuality::Minor7,
            },
            Chord {
                root: NoteName::B,
                quality: ChordQuality::Minor7,
            },
        ];
        app.reset_logic_state();
        Self {
            app,
            frame: 0,
            phase,
            deliveries: vec![],
            credits: vec![],
        }
    }

    fn onset_after(&mut self, frames: usize, pc: usize) {
        self.deliveries.push((self.frame + frames, pc));
    }

    fn hear(&mut self, pc: usize, frames: usize) {
        for _ in 0..frames {
            let before = self.app.collected_notes.clone();
            let chord = self.app.current_chord_index;
            if self.frame % 3 == self.phase {
                let mut onsets = [0.0; 12];
                for &(arrival, class) in &self.deliveries {
                    if (arrival..arrival + 6).contains(&self.frame) {
                        onsets[class] = 0.9;
                    }
                }
                self.app.set_onsets(onsets);
                self.app.check_progress_with_ai(3.0 * DT, "Noise", 0.0);
            }
            {
                let mut audio = self.app.analysis_state.lock().unwrap();
                audio.frames_seen += 1;
                audio.cqt_pitch = Some(pc);
                audio.cqt_semitone = Some(36 + pc);
                audio.gate_open = true;
            }
            self.app.sync_audio_settings();
            self.app.tick(DT);
            if self.app.current_chord_index == chord {
                for (step, &collected) in self.app.collected_notes.iter().enumerate() {
                    if collected && !before[step] {
                        self.credits.push((self.frame + 1, chord, step));
                    }
                }
            }
            self.frame += 1;
        }
    }

    fn first_chord(&mut self) {
        for pc in [ROOT, THIRD, FIFTH] {
            self.onset_after(0, pc);
            self.hear(pc, 25);
        }
        self.hear(FIFTH, 20);
        assert_eq!(
            self.app.current_chord_index, 1,
            "the first Bm7 did not finish"
        );
        assert!(self.app.collected_notes.iter().all(|&v| !v));
    }
}

#[test]
fn an_identical_bm7_does_not_reuse_a_continuously_held_fifth() {
    for phase in 0..3 {
        let mut replay = Replay::new("1 3 5", false, phase);
        replay.first_chord();
        replay.hear(FIFTH, 50);
        assert!(replay.app.collected_notes.iter().all(|&v| !v));
    }
}

#[test]
fn a_replucked_fifth_in_an_identical_bm7_keeps_the_normal_credit_delay() {
    for phase in 0..3 {
        let mut replay = Replay::new("1 3 5", false, phase);
        replay.first_chord();
        let pluck = replay.frame;
        replay.onset_after(0, FIFTH);
        replay.hear(FIFTH, 12);
        let &(credit, chord, step) = replay.credits.last().unwrap();
        assert_eq!((chord, step), (1, 2));
        assert!(
            credit - pluck <= 10,
            "credit exceeded 160 ms: {} frames",
            credit - pluck
        );
    }
}

#[test]
#[ignore = "known CQT return bypass with require_onset in repeated Bm7"]
fn an_old_root_returning_in_the_next_bm7_is_not_a_new_pluck() {
    for phase in 0..3 {
        let mut replay = Replay::new("1 3 5", false, phase);
        replay.first_chord();
        replay.hear(ROOT, 12);
        assert!(replay.app.collected_notes.iter().all(|&v| !v));
    }
}

#[test]
#[ignore = "known CQT return bypass when an older fifth becomes loudest again"]
fn an_old_fifth_returning_after_the_root_is_not_a_new_pluck() {
    for phase in 0..3 {
        let mut replay = Replay::new("1 3 5", false, phase);
        replay.first_chord();
        replay.onset_after(0, ROOT);
        replay.hear(ROOT, 25);
        assert!(replay.app.collected_notes[0]);
        replay.hear(FIFTH, 12);
        assert!(!replay.app.collected_notes[2]);
    }
}

#[test]
fn a_late_first_onset_is_absorbed_during_the_existing_settle_window() {
    for phase in 0..3 {
        let mut replay = Replay::new("1 1", true, phase);
        replay.onset_after(20, ROOT); // first pluck's answer, 320 ms late
        replay.hear(ROOT, 65);
        assert_eq!(&replay.app.collected_notes[..], &[true, false]);
    }
}

#[test]
#[ignore = "known settle window consumes a real second attack 320 ms after the first"]
fn a_fast_repluck_with_an_ideal_onset_is_not_discarded() {
    for phase in 0..3 {
        let mut replay = Replay::new("1 1", true, phase);
        replay.onset_after(0, ROOT);
        replay.hear(ROOT, 20);
        assert_eq!(&replay.app.collected_notes[..], &[true, false]);
        replay.onset_after(0, ROOT);
        replay.hear(ROOT, 12);
        assert_eq!(&replay.app.collected_notes[..], &[true, true]);
    }
}

#[test]
#[ignore = "known first-onset ambiguity after the fixed settle window expires"]
fn an_old_onset_delivered_after_settle_is_not_a_second_pluck() {
    for phase in 0..3 {
        let mut replay = Replay::new("1 1", true, phase);
        replay.onset_after(42, ROOT); // 672 ms; may also represent slow delivery
        replay.hear(ROOT, 54);
        assert_eq!(&replay.app.collected_notes[..], &[true, false]);
    }
}

#[test]
#[ignore = "diagnostic timing table; invoke with --ignored --nocapture"]
fn print_interval_onset_timing_matrix() {
    println!("ONSET_REPLAY,case,phase,arrival_ms,second_credit_ms");
    for phase in 0..3 {
        for delay in [10, 15, 20, 25, 30, 35, 40, 45, 50] {
            for new_pluck in [false, true] {
                let mut replay = Replay::new("1 1", true, phase);
                if new_pluck {
                    replay.onset_after(0, ROOT);
                }
                replay.onset_after(delay, ROOT);
                // Stop before the completed-set display advances the chord.
                replay.hear(ROOT, delay + 12);
                let credit = replay
                    .credits
                    .iter()
                    .find(|&&(_, chord, step)| chord == 0 && step == 1);
                let second_ms = credit
                    .map(|&(frame, _, _)| (frame * 16).to_string())
                    .unwrap_or_default();
                println!(
                    "ONSET_REPLAY,{},{},{},{}",
                    if new_pluck {
                        "real_repluck"
                    } else {
                        "late_first_onset"
                    },
                    phase,
                    delay * 16,
                    second_ms
                );
            }
        }
    }
}
