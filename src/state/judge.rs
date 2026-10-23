//! The rules every mode shares: whether a note is sounding and by which
//! evidence, whether it was struck, and what crediting it records.

use super::*;

impl MyApp {
    /// Is the requested pitch class sounding right now?
    ///
    /// Shared by the note modes and the fretboard trainer so the two cannot drift
    /// apart. Four ways in, because no single one covers note practice: the model
    /// hears the last 0.77 s and so mostly the PREVIOUS note, and the single-frame
    /// estimate never names a note that was not played but misses more. See
    /// `docs/judge-measurements.md#note_is_sounding`.
    pub(super) fn note_is_sounding(&self, pc: usize, ai_root: Option<NoteName>, confidence: f32) -> bool {
        self.sounding_by(pc, ai_root, confidence).is_some()
    }

    /// The same question, answered with WHICH of the four ways in let the note
    /// through. Formulas need to know: their marks accumulate, so a branch that
    /// fires once too often is the difference between an exercise and a screen
    /// that fills itself in.
    pub(super) fn sounding_by(&self, pc: usize, ai_root: Option<NoteName>, confidence: f32) -> Option<u8> {
        let target = pc % 12;

        // Played one at a time, the ear decides alone - and has to have been
        // saying so for three frames, not one.
        //
        // With the model's branches as well, almost all false credits were its pitch
        // head answering about a note still ringing inside its 0.77 s window - see
        // `docs/judge-measurements.md#one-at-a-time`.
        // Scales and arpeggios are played one note at a time by definition -
        // nobody strums an arpeggio - so the rule is in force there whether the
        // option is ticked or not, and the checkbox is not offered.
        let one_at_a_time = self.one_at_a_time();
        if one_at_a_time {
            if self.heard.sounding_now() == Some(target) && !self.ear_settling(target) && self.ear_backed(target) {
                return Some(1);
            }
            let ear = self.heard.steady_note();
            // The ear naming ANOTHER note is a refusal; the ear saying nothing
            // is not. A note too quiet for the estimate's score gate - a low
            // string, the end of a phrase - would otherwise be uncreditable,
            // which is what made this harder to play than it had been. Where
            // the ear is silent the model may still answer, but only its pitch
            // head and only where an attack backs it: the chord NAME never
            // credits a note here, and that is where the notes nobody played
            // were coming from.
            //
            // Except where the ear is merely late: see `ear_holds_only_a_ring`.
            if (ear.is_some() || self.heard.cqt_pitch.is_some()) && !self.ear_holds_only_a_ring(target) {
                return None;
            }
        }

        // 1. One frame, one note - no window to smear across.
        if self.heard.cqt_pitch == Some(target) && !self.ear_settling(target) && self.ear_backed(target) {
            return Some(1);
        }

        let p_target = self.last_pitches[target];
        let p_max = self.last_pitches.iter().cloned().fold(0.0f32, f32::max);

        // 2. The model, where the target owns the window: a held note, or the
        //    only one in it.
        // Struck just now, or the class is only ringing - see `struck_just_now`.
        let struck = !self.require_onset || self.struck_just_now(target);
        if struck && p_target >= self.note_threshold && p_target >= p_max * 0.9 {
            return Some(2);
        }

        // 3. The model, where the target is the note that just ARRIVED in a
        //    window still holding the previous one. Being loudest is the wrong
        //    test there - the older note has had more of the window to itself,
        //    so it wins on level while the new one wins on rise. Only meaningful
        //    against a window that already held something: from an empty one
        //    every class has "risen", which is branch 2 wearing a disguise.
        let had_content = self.prev_pitches.iter().any(|&v| v > 0.2);
        let rise = p_target - self.prev_pitches[target];
        let best_rise = (0..12)
            .map(|i| self.last_pitches[i] - self.prev_pitches[i])
            .fold(f32::NEG_INFINITY, f32::max);
        if struck
            && had_content
            && p_target >= self.note_threshold
            && rise > 0.05
            && rise >= best_rise * 0.9
        {
            return Some(3);
        }

        // 4. The root head as independent confirmation - a single note is
        //    reported by the model as the root.
        //
        // Gated by the attack too. "Credit only what was struck" held back the
        // two pitch-head branches and left this one open, so with a microphone
        // in a room the model went on naming chords and this credited whatever
        // note their root happened to be - notes nobody had played.
        let by_root = struck
            && !one_at_a_time
            && matches!(ai_root, Some(r) if r == NoteName::from_index(target))
            && confidence >= self.chord_confidence;
        if by_root { Some(4) } else { None }
    }

    /// Whether the ear names only the note before: the target struck within
    /// CLASS_FRESH, and the class the ear names not. Its window still holds a
    /// note ringing on - another string in an arpeggio, say - and the new one
    /// has not yet won it; the model has, and may answer. See
    /// `docs/judge-measurements.md#the-ear-holding-a-ring`.
    fn ear_holds_only_a_ring(&self, target: usize) -> bool {
        if !self.heard.strikes_live {
            return false;
        }
        let named = self.heard.steady_note().or(self.heard.cqt_pitch);
        let Some(named) = named.filter(|&n| n % 12 != target) else {
            return false;
        };
        self.heard.class_strike_age[target] <= CLASS_FRESH && self.heard.class_strike_age[named % 12] > CLASS_FRESH
    }

    /// Whether the ear is still settling after a strike of some OTHER class -
    /// see `EAR_SETTLE`. Without the detector it never is.
    pub(super) fn ear_settling(&self, pc: usize) -> bool {
        self.heard.strikes_live
            && self.heard.class_strike_age.iter().any(|&age| age <= EAR_SETTLE)
            && self.heard.class_strike_age[pc] > EAR_SETTLE
    }

    /// Whether the ear may credit `pc` at all: outside Formulas, or without
    /// the detector, always; in Formulas only a class the detector reported
    /// struck - see `EAR_BACKED`.
    pub(super) fn ear_backed(&self, pc: usize) -> bool {
        !self.heard.strikes_live
            || self.app_mode != AppMode::Formulas
            || self.heard.class_strike_age[pc] <= EAR_BACKED
    }

    /// Whether `pc` was struck just now, for "credit only what was struck".
    /// The strike detector where it runs - see `STRUCK_FRESH`. Without it only
    /// flux is left, which says that a string was hit but not which.
    pub(super) fn struck_just_now(&self, pc: usize) -> bool {
        if self.heard.strikes_live {
            return self.heard.class_strike_age[pc] <= STRUCK_FRESH;
        }
        self.heard.flux_seen && self.heard.flux_age <= FLUX_FRESH
    }

    /// All three guards off: the exercise is being played for fun, with chords
    /// and whatever single tones of them come to hand.
    ///
    /// Then the model decides and nothing argues with it - no ear guard, no
    /// refusing a class because it was credited in the chord before. Those
    /// rules exist to make a TEST honest, and a player who has turned all of
    /// them off is not taking one. The strict path is the default and the
    /// measured one; this is the other thing the mode can be.
    pub(super) fn relaxed_chords(&self) -> bool {
        self.app_mode == AppMode::Intervals
            && !self.single_notes
            && !self.require_onset
            && !self.interval_in_order
    }

    /// Whether class `pc` has been played again since it was credited - see
    /// `Credits::struck_since`, which this hands the mode's own rule.
    pub(super) fn struck_since_credit(&self, pc: usize) -> bool {
        self.credited.struck_since(pc, &self.heard, self.one_at_a_time())
    }

    /// Whether notes are taken one at a time: by choice in Intervals and
    /// Formulas, always in Scales, Arpeggios and the fretboard trainer - nobody
    /// strums one, and the trainer asks for a single note at a single place.
    ///
    /// The one definition. There used to be two: this one forced the rule on
    /// the fretboard trainer as well, while `sounding_by` kept a copy of its own
    /// that did not, so the trainer's repeat gate and its sounding check
    /// disagreed about which mode they were in.
    pub(super) fn one_at_a_time(&self) -> bool {
        self.single_notes || matches!(self.app_mode, AppMode::Scales | AppMode::Arpeggios | AppMode::Fretboard)
    }

    pub(super) fn credit_class(&mut self, pc: usize, octave: i8) {
        if self.journal.is_some() {
            // The three strongest pitch-head answers, so a rule can be tried
            // against this row later without playing anything again.
            let mut top: Vec<(usize, f32)> = (0..12).map(|i| (i, self.last_pitches[i])).collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let row = serde_json::json!({
                "pc": pc % 12,
                "octave": octave,
                "by": self.credit_branch,
                "ear": self.heard.cqt_pitch,
                "semitone": self.heard.cqt_semitone,
                "flux": self.heard.flux_id,
                "flux_age": self.heard.flux_age,
                "voices": (0..12).filter(|pc| self.heard.voice_heard(*pc)).collect::<Vec<_>>(),
                "quiet_for": self.heard.quiet_for[pc % 12],
                "class_strikes": self.heard.class_strikes[pc % 12],
                "class_strike_age": self.heard.class_strike_age[pc % 12],
                "strikes_live": self.heard.strikes_live,
                "left": self.credited[pc % 12].map(|c| c.left),
                "onset": self.heard.onset_id,
                "top": top.iter().take(3).map(|&(i, v)| serde_json::json!([i, v])).collect::<Vec<_>>(),
            });
            self.journal_row("credit", row);
        }
        self.credit_branch = None;
        self.credited.record(pc, octave, &self.heard)
    }

    /// What is still sounding when the exercise moves on counts as already
    /// used in the next part of it: it needs a strike of its own there.
    ///
    /// The repeat rule only guarded classes credited before, so a note that
    /// rang out of the last chord WITHOUT being credited - a wrong note, an
    /// extra one - was free to answer the next chord the moment the ear named
    /// it. Requiring a strike for every first credit closes that too, but costs
    /// 7 to 11 of the 87 notes of the user's own session, which the detector
    /// does not see; this costs nothing where the note is played after it is
    /// asked for, which is every ordinary credit.
    ///
    /// A class struck in the last `JUST_STRUCK` hops is not marked: that is a
    /// player reaching the next chord early, while the finished one is shown.
    pub(super) fn carry_over_what_rings(&mut self) {
        for pc in 0..12 {
            if self.credited[pc].is_some() {
                continue;
            }
            let sounding = self.heard.voice_heard(pc) || self.heard.steady_note() == Some(pc);
            let just_struck = self.heard.strikes_live && self.heard.class_strike_age[pc] <= JUST_STRUCK;
            if sounding && !just_struck {
                self.credited.record(pc, 0, &self.heard);
            }
        }
    }
}
