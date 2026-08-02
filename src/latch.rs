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
            // A root change without a detected attack (legato, or an attack too
            // quiet for the detector) must release too, otherwise the app would
            // show the previous chord forever.
            if root_of(chord) != root_of(held) && conf >= LOCK_MIN_CONF {
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
