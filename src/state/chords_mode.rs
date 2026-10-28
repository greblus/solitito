//! Chords: how the model's reading of a chord is judged against the chord
//! asked for.

use super::*;

/// Pitch-head level at which a note of a diminished seventh counts as heard -
/// the level the recordings in `diminished_heard_whole` were measured at.
const DIM_TONE_HEARD: f32 = 0.5;

impl MyApp {
    pub(super) fn check_chord(&mut self, dt: f32, ai_root: Option<NoteName>, ai_qual: &str, confidence: f32) {
        let Some(target) = self.chords.get(self.current_chord_index) else {
            return;
        };
        if ai_qual == "Note" {
            self.match_status = MatchStatus::None;
            self.success_timer = (self.success_timer - dt * 2.0).max(0.0);
            return;
        }
        let named = match ai_root {
            Some(root) => chord_reading(target, root, ai_qual, self.shells_only),
            None => ChordReading::Wrong,
        };
        let heard_by_its_notes = self.diminished_heard_whole(target);
        let reading = if heard_by_its_notes { ChordReading::Exact } else { named };
        // A chord asked for twice in a row has to be played twice. Without
        // this the ring of the pass before matches the moment the target is
        // set and the progression runs itself.
        let not_struck_again = reading != ChordReading::Wrong
            && self.chord_repeats(target)
            && !self.chord_struck_since();
        if not_struck_again {
            self.match_status = MatchStatus::None;
            self.success_timer = 0.0;
            return;
        }

        // The confidence is the chord name's, and the name is what was wrong.
        let is_weak_signal = confidence < self.chord_confidence && !heard_by_its_notes;
        // Green means the target chord was heard, clearly enough to be
        // trusted - the same condition the bucket paints green on. With
        // this on, that is the whole test: no waiting for it to be held,
        // and no decay afterwards to undo it. Nothing runs away, because
        // advancing changes the target and the chord still ringing stops
        // matching it.
        if self.short_verdict && reading == ChordReading::Exact && !is_weak_signal {
            self.match_status = MatchStatus::Exact;
            if !self.paused {
                self.advance_chord();
            }
            return;
        }

        self.fill_chord_bucket(reading, is_weak_signal, dt);
        if self.held_past(self.transition_delay) {
            self.advance_chord();
        }
    }

    /// Whether the pitch head hears all four notes of a diminished seventh
    /// asked for.
    ///
    /// The chord name misses this chord on a guitar. Over a recording of D♭°7
    /// strummed again and again, the model named it `dim` 6 times in 547
    /// readings and a dominant seventh on one of its notes nearly every other
    /// time, while the pitch head heard all four notes 165 times. Four notes a
    /// minor third apart are this chord and nothing else; a dominant holds them
    /// all only as a seventh with a flat ninth, which is this chord with a
    /// root under it. Measurements: `docs/judge-measurements.md#diminished_heard_whole`.
    fn diminished_heard_whole(&self, target: &Chord) -> bool {
        target.quality == ChordQuality::Diminished
            && target
                .get_target_indices()
                .iter()
                .all(|&pc| self.last_pitches[pc % 12] > DIM_TONE_HEARD)
    }

    /// The leaky bucket: a clear reading of the chord fills the success timer
    /// at the rate of the clock, anything else drains it - a wrong chord the
    /// fastest.
    pub(super) fn fill_chord_bucket(&mut self, reading: ChordReading, is_weak_signal: bool, dt: f32) {
        let (status, rate) = match (reading, is_weak_signal) {
            (ChordReading::Exact, false) => (MatchStatus::Exact, 1.0),
            (ChordReading::Exact, true) => (MatchStatus::Flicker, -1.0),
            (ChordReading::Partial, false) => (MatchStatus::Partial, 1.0),
            (ChordReading::Partial, true) => (MatchStatus::None, -2.0),
            (ChordReading::Wrong, _) => (MatchStatus::None, -4.0),
        };
        self.match_status = status;
        self.success_timer = (self.success_timer + dt * rate).max(0.0);
    }

    /// True when the chord now being asked for is the one just left behind -
    /// two of the same in a row in the song, or a progression of one chord.
    /// Its notes are still ringing, so matching proves nothing.
    pub(super) fn chord_repeats(&self, target: &Chord) -> bool {
        match self.prev_chord_index {
            Some(p) => {
                p < self.chords.len()
                    && self.chords[p].root == target.root
                    && self.chords[p].quality == target.quality
            }
            None => false,
        }
    }

    /// Whether the strings have been hit since this chord became the target.
    ///
    /// The strike detector answers it with any strike at all: it reports a strum as
    /// one class, seldom two, and does not fire on a chord that only rings. Without
    /// the detector, the envelope. Measurements: `docs/judge-measurements.md#chord_struck_since`.
    pub(super) fn chord_struck_since(&self) -> bool {
        let then = &self.chord_heard_at;
        if self.heard.strikes_live {
            return self.heard.class_strikes != then.class_strikes;
        }
        self.heard.onset_id != then.onset_id
    }
}

/// What the model's reading makes of the chord asked for.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum ChordReading {
    /// The chord itself, or a name that can only mean it.
    Exact,
    /// A substitution that leaves out or shares notes with it.
    Partial,
    Wrong,
}

/// How the model's chord - `root` and quality `heard` - reads against the
/// chord asked for.
pub(super) fn chord_reading(target: &Chord, root: NoteName, heard: &str, shells_only: bool) -> ChordReading {
    let wanted = target.quality.to_string();
    if root == target.root {
        return same_root_reading(&wanted, heard, shells_only);
    }
    // Interval in semitones from the root asked for up to the root heard.
    let interval = (root as i32 - target.root as i32).rem_euclid(12);
    match (wanted.as_str(), heard) {
        // m7b5 read as m a minor third up, e.g. A m7b5 -> C m.
        ("m7b5", "m") if interval == 3 => ChordReading::Partial,
        // Maj7 read as m/m7 a major third up, e.g. C Maj7 -> E m.
        ("Maj7", "m") | ("Maj7", "m7") if interval == 4 => ChordReading::Partial,
        // A diminished seventh is four notes a minor third apart, so all four
        // are equally its root: C, Eb, Gb and A dim7 are one chord under four
        // names. Which name the model gives depends on the voicing, not on
        // what was played, so a reading three, six or nine semitones away is
        // the chord itself - green, not a substitution.
        ("dim", "dim") if interval % 3 == 0 => ChordReading::Exact,
        // A m6 is the m7b5 a minor third below, note for note, and the model
        // has no m6 to name: A m6 heard as F# m7b5 is the chord itself.
        ("m6", "m7b5") if interval == 9 => ChordReading::Exact,
        _ => ChordReading::Wrong,
    }
}

pub(super) fn same_root_reading(wanted: &str, heard: &str, shells_only: bool) -> ChordReading {
    if heard == wanted {
        return ChordReading::Exact;
    }
    match (wanted, heard) {
        ("Maj7", "") | ("Maj7", "Maj") | ("m7", "m") | ("7", "") | ("m7b5", "dim")
        // The model names no m6 and no mMaj7; their triad is what it can hear.
        | ("m6", "m") | ("mMaj7", "m") => {
            ChordReading::Partial
        }
        // A shell voicing of a m7b5 is root, third and seventh - the m7
        // shell, note for note, since the fifth is the only place the two
        // differ, and the shell leaves the fifth out. With shells the only
        // shapes on screen, that grip is what the app asked for, so playing
        // it is a pass and not a substitution: green. Yellow otherwise, where
        // the full shape is drawn with its flat fifth and a m7 means the fifth
        // was missed. Heard as plain m when the seventh dies away, exactly as
        // an ordinary m7 is.
        ("m7b5", "m7") | ("m7b5", "m") if shells_only => ChordReading::Exact,
        ("m7b5", "m7") | ("m7b5", "m") => ChordReading::Partial,
        _ => ChordReading::Wrong,
    }
}
