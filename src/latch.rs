//! Chord quality latch ("lock chord quality until new attack").
//!
//! A seventh decays faster than the root, third and fifth. Holding a `Gm7`:
//!
//! ```text
//! G m7 | min7=96% | b7=96      <- right after the attack
//! G m7 | min7=52% | b7=52
//! G m  | min=49%  | b7=45      <- the seventh died, the model changes its mind
//! ```
//!
//! The model is right about the current window, but a chord does not change
//! identity while it rings out. The app knows that; the model cannot.
//!
//! The latch ENGAGES at high confidence but HOLDS regardless of it - during the
//! decay the model reports the poorer quality at 94-96%, so a confidence
//! threshold alone would save nothing. Only a new attack or a root change
//! releases it.
//!
//! It engages only once the context window has filled with sound after the
//! attack (`CTX_FRAMES` frames); earlier the window still holds the tail of the
//! PREVIOUS chord and the latch would capture the wrong name.

/// Minimum confidence required to ENGAGE the latch.
pub const LOCK_MIN_CONF: f32 = 0.60;

/// Frames to wait after an attack before the context window is entirely filled
/// with the new chord. Equal to `audio::CTX_FRAMES`.
pub const SETTLE_FRAMES: u32 = 48;

#[derive(Default)]
pub struct ChordLatch {
    locked: Option<String>,
    last_onset: u64,
}

impl ChordLatch {
    /// The chord to display and feed to the exercise logic.
    ///
    /// * `enabled` - UI switch; when `false` this is a pass-through
    /// * `onset_id` - attack counter from `AudioAnalysis`; a change releases it
    /// * `frames_since_onset` - frames elapsed since the attack
    /// * `chord` / `conf` - current prediction, already voted on
    pub fn update(
        &mut self,
        enabled: bool,
        onset_id: u64,
        frames_since_onset: u32,
        chord: &str,
        conf: f32,
    ) -> String {
        if !enabled {
            self.locked = None;
            self.last_onset = onset_id;
            return chord.to_string();
        }

        // New attack = new chord. Release unconditionally.
        if onset_id != self.last_onset {
            self.last_onset = onset_id;
            self.locked = None;
        }

        if let Some(held) = &self.locked {
            // The one thing the latch is for: the top note has died and the
            // four-note chord now reads as its triad. Hold, however sure the
            // model is - during the decay it reports the poorer quality at 94-96%.
            if is_decay_of(held, chord) {
                return held.clone();
            }
            // Anything else is a genuine change, not decay: a different root, or
            // a different four-note chord on the same root. Take it once the
            // model is sure.
            //
            // Only checking the root here was the bug: m7b5 differs from m7 by
            // one note, so an early misread on the same root could never be
            // corrected and the chord stayed stuck until the strings were struck
            // again.
            if conf >= LOCK_MIN_CONF && is_real(chord) {
                self.locked = Some(chord.to_string());
                return chord.to_string();
            }
            return held.clone();
        }

        // Engage only once the window lies entirely after the attack.
        if frames_since_onset >= SETTLE_FRAMES && conf >= LOCK_MIN_CONF && is_real(chord) {
            self.locked = Some(chord.to_string());
        }
        chord.to_string()
    }

    /// What is currently held. Used by the tests; the UI no longer shows a latch
    /// marker - it carried nothing useful while playing.
    #[allow(dead_code)]
    pub fn held(&self) -> Option<&str> {
        self.locked.as_deref()
    }
}

/// Root from a prediction string ("G m7" -> "G", "Note F" -> "F").
fn root_of(chord: &str) -> &str {
    let mut it = chord.split_whitespace();
    match it.next() {
        Some("Note") => it.next().unwrap_or(""),
        Some(r) => r,
        None => "",
    }
}

/// Quality part of a prediction: "A m7b5" -> "m7b5", a bare "A" -> "" (major).
fn quality_of(chord: &str) -> &str {
    let mut it = chord.split_whitespace();
    match it.next() {
        Some("Note") | None => "",
        Some(_) => it.next().unwrap_or(""),
    }
}

/// Four-note qualities, spelled as `brain::quality_suffix` writes them.
fn is_four_note(q: &str) -> bool {
    matches!(q, "Maj7" | "7" | "m7" | "m7b5" | "dim")
}

/// Is `now` simply what `held` sounds like once its top note has gone?
///
/// That is the only thing worth holding through. A four-note chord thinning to
/// its triad on the same root is decay; a DIFFERENT four-note chord on that root
/// is the model correcting itself, and blocking it leaves the player strumming
/// at an app that will not budge.
fn is_decay_of(held: &str, now: &str) -> bool {
    root_of(held) == root_of(now)
        && is_four_note(quality_of(held))
        && !is_four_note(quality_of(now))
}

/// Is this a chord worth latching onto at all.
fn is_real(chord: &str) -> bool {
    !chord.is_empty() && chord != "Noise" && chord != "..." && !chord.starts_with("Note")
}

#[cfg(test)]
mod tests {
    use super::*;

    const AFTER: u32 = SETTLE_FRAMES;

    #[test]
    fn holds_the_seventh_through_decay() {
        // Replayed from the diagnostic output: the model loses the seventh but
        // stays highly confident about the poorer quality.
        let mut l = ChordLatch::default();
        assert_eq!(l.update(true, 1, AFTER, "G m7", 0.95), "G m7");
        for conf in [0.94, 0.95, 0.96] {
            assert_eq!(
                l.update(true, 1, AFTER + 10, "G m", conf), "G m7",
                "the latch must hold despite high confidence in the poorer quality"
            );
        }
    }

    /// Reported from play: an Am7b5 that the model recognises well refuses to
    /// show up, and only the latch causes it.
    ///
    /// m7b5 differs from m7 by one note, so early in the ring the model can read
    /// the poorer m7 and the latch takes it. From then on the root never changes,
    /// so the ONLY release left is a fresh attack - and the correct reading,
    /// however confident, cannot get through.
    #[test]
    fn a_confident_correction_on_the_same_root_gets_through() {
        let mut l = ChordLatch::default();
        // The model briefly reads the neighbouring quality and the latch takes it.
        assert_eq!(l.update(true, 1, AFTER, "A m7", 0.72), "A m7");
        // Now it settles on the right one, and stays sure of it.
        for _ in 0..30 {
            let shown = l.update(true, 1, AFTER + 20, "A m7b5", 0.93);
            assert_eq!(
                shown, "A m7b5",
                "a different seventh on the same root is a correction, not decay"
            );
        }
    }

    /// The correction has to be CONVINCING. Letting any reading through would
    /// be the same as switching the latch off.
    #[test]
    fn an_unconvincing_reading_does_not_dislodge_the_latch() {
        let mut l = ChordLatch::default();
        l.update(true, 1, AFTER, "A m7", 0.85);
        for conf in [0.10, 0.35, 0.59] {
            assert_eq!(l.update(true, 1, AFTER + 5, "A m7b5", conf), "A m7");
        }
    }

    /// Decay stays held no matter how sure the model is - that is the whole
    /// point, and it is what the diagnostic output showed at 94-96%.
    #[test]
    fn decay_is_held_even_at_full_confidence() {
        for (rich, thin) in [("G m7", "G m"), ("C Maj7", "C"), ("D 7", "D"),
                             ("B m7b5", "B m"), ("F dim", "F m")] {
            let mut l = ChordLatch::default();
            l.update(true, 1, AFTER, rich, 0.90);
            assert_eq!(
                l.update(true, 1, AFTER + 10, thin, 0.99), rich,
                "{rich} thinning to {thin} is decay and must be held"
            );
        }
    }

    /// ...but a different four-note chord on the same root is a correction.
    #[test]
    fn a_different_seventh_is_a_correction_not_decay() {
        for (held, corrected) in [("A m7", "A m7b5"), ("A m7b5", "A m7"),
                                  ("C Maj7", "C 7"), ("G 7", "G m7")] {
            let mut l = ChordLatch::default();
            l.update(true, 1, AFTER, held, 0.80);
            assert_eq!(
                l.update(true, 1, AFTER + 10, corrected, 0.90), corrected,
                "{held} -> {corrected} is a correction and must get through"
            );
        }
    }

    #[test]
    fn quality_is_read_off_the_prediction() {
        assert_eq!(quality_of("A m7b5"), "m7b5");
        assert_eq!(quality_of("C"), "", "a bare root is the major triad");
        assert_eq!(quality_of("Note F"), "");
    }

    #[test]
    fn new_attack_releases() {
        let mut l = ChordLatch::default();
        l.update(true, 1, AFTER, "G m7", 0.95);
        assert_eq!(l.update(true, 1, AFTER + 5, "G m", 0.95), "G m7");
        // strings struck -> onset_id increments
        assert_eq!(l.update(true, 2, AFTER, "G m", 0.95), "G m");
        assert_eq!(l.held(), Some("G m"));
    }

    #[test]
    fn does_not_catch_the_previous_chord_tail() {
        // Just after the attack the window still holds the previous chord.
        let mut l = ChordLatch::default();
        for f in [0u32, 10, 30, 47] {
            l.update(true, 1, f, "C m7", 0.97);
            assert_eq!(l.held(), None, "at {f} frames the window is not full yet");
        }
        l.update(true, 1, AFTER, "G m7", 0.97);
        assert_eq!(l.held(), Some("G m7"));
    }

    #[test]
    fn low_confidence_does_not_engage() {
        let mut l = ChordLatch::default();
        l.update(true, 1, AFTER, "G m7", 0.30);
        assert_eq!(l.held(), None);
        l.update(true, 1, AFTER, "G m7", LOCK_MIN_CONF);
        assert_eq!(l.held(), Some("G m7"));
    }

    #[test]
    fn root_change_without_attack_also_releases() {
        let mut l = ChordLatch::default();
        l.update(true, 1, AFTER, "G m7", 0.95);
        // moving to another chord; the attack detector missed it
        assert_eq!(l.update(true, 1, AFTER + 20, "C m7", 0.95), "C m7");
        assert_eq!(l.held(), Some("C m7"));
    }

    #[test]
    fn disabled_is_a_pass_through() {
        let mut l = ChordLatch::default();
        assert_eq!(l.update(false, 1, AFTER, "G m7", 0.95), "G m7");
        assert_eq!(l.update(false, 1, AFTER + 5, "G m", 0.95), "G m");
        assert_eq!(l.held(), None);
    }

    #[test]
    fn noise_and_single_notes_do_not_latch() {
        let mut l = ChordLatch::default();
        l.update(true, 1, AFTER, "Noise", 0.95);
        assert_eq!(l.held(), None);
        l.update(true, 1, AFTER, "Note F", 0.95);
        assert_eq!(l.held(), None, "a single note is not a chord to hold");
    }

    #[test]
    fn root_of_handles_the_formats() {
        assert_eq!(root_of("G m7"), "G");
        assert_eq!(root_of("C"), "C");
        assert_eq!(root_of("Note F#"), "F#");
        assert_eq!(root_of(""), "");
    }
}
