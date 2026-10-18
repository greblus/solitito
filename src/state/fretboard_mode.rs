//! The fretboard trainer: the region, the note asked for, its clock and its
//! judge.

use super::*;

impl MyApp {
    /// The fretboard trainer's show clock, for the same reason as the lap's in
    /// Formulas: it must not stop when the player does.
    pub(super) fn tick_fretboard(&mut self, dt: f32) {
        if self.fret_hold <= 0.0 || self.paused {
            return;
        }
        self.fret_hold = (self.fret_hold - dt).max(0.0);
        if self.fret_hold == 0.0 {
            self.next_fret_target();
        }
    }

    /// Draws a fresh practice region: a set of strings and a fret window.
    ///
    /// Called on entering the mode, then left alone - the region is meant to
    /// hold while you settle into one hand position. Only the note changes.
    /// A four-fret span is one finger per fret, index to little finger.
    pub fn randomize_region(&mut self) {
        const SPAN: u8 = 4;
        let strings = match self.rng.below(3) {
            0 => crate::fretboard::StringSet::All,
            1 => crate::fretboard::StringSet::LowThree,
            _ => crate::fretboard::StringSet::HighThree,
        };
        // Keep the whole window on the neck: last usable start is MAX_FRET-SPAN+1.
        let highest_start = crate::fretboard::MAX_FRET - SPAN + 1;
        self.region = Region {
            strings,
            fret_from: self.rng.below(highest_start as usize + 1) as u8,
            fret_span: SPAN,
        };
        self.fret_target = None;
        self.fret_hold = 0.0;
        self.next_fret_target();
    }

    /// Draws the next fretboard target. The region is untouched - it is fixed
    /// for the session and only changes when the player changes the settings.
    pub fn next_fret_target(&mut self) {
        if let Some(prev) = self.fret_target {
            self.recent_targets.push(prev);
        }
        // Three back. A region carries a handful of pitch classes, so avoiding
        // only the note just asked for brought the same few round in a circle -
        // and a note asked for again while it is still ringing has to be struck
        // afresh before it counts, which reads as the trainer sticking.
        while self.recent_targets.len() > 3 {
            self.recent_targets.remove(0);
        }
        self.fret_target = self.region.draw_avoiding(&mut self.rng, &self.recent_targets);
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
    }

    pub(super) fn check_fretboard(&mut self, dt: f32, ai_root: Option<NoteName>, confidence: f32) {
        // The note just answered is still being shown. Nothing to judge
        // until the next one is drawn - and judging anyway would let the
        // string still ringing answer for it.
        if self.fret_hold > 0.0 {
            return;
        }
        let Some(target) = self.fret_target else {
            self.next_fret_target();
            return;
        };
        // The same rule as the note modes': a note drawn twice in a row has
        // to be struck twice. See `struck_since_credit`.
        let fresh = self.struck_since_credit(target);
        if fresh && self.note_is_sounding(target, ai_root, confidence) {
            self.success_timer += dt;
            self.match_status = MatchStatus::Exact;
        } else {
            self.success_timer = 0.0;
            self.match_status = MatchStatus::None;
        }
        // The same short hold the note modes use, not the chord one. A
        // chord is held while the ear settles on a NAME; a single note is
        // right the moment it is recognised, and asking it to hold for the
        // chord delay meant a plucked note decaying out of the estimate's
        // reach before the timer got there - it lit green, dropped back to
        // nothing, and the trainer never moved on.
        if !self.held_past(NOTE_HOLD) {
            return;
        }
        self.credit_class(target, 0);
        // Passed, but not moved on: the note stays where it is, green, for as
        // long as `FRET_SHOW` says. The clock for that runs in `tick`, every
        // frame, because this one is called only when the model answers - and
        // a hold timed on answers is a hold that stops when the player does.
        self.match_status = MatchStatus::Exact;
        self.success_timer = NOTE_HOLD;
        self.fret_hold = self.transition_delay.max(FRET_SHOW);
    }
}
