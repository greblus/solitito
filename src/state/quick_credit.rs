//! Quick crediting: where notes are taken one at a time, a note is credited
//! the moment the strike detector and the quick ear agree on it.
//!
//! The ordinary path waits for the ear proper, whose half-second window names
//! a new note about 0.26 s after the strike, and then for the answer to hold:
//! on the user's sessions a credit came 0.37 s after the detector had named
//! the note. This path asks two quicker witnesses instead - see
//! `MyApp::struck_and_named` - and where they do not both speak, the ordinary
//! path credits as it always did. Measurements:
//! `docs/judge-measurements.md#quick-crediting`.

use super::*;

impl MyApp {
    /// Once per new audio frame, in every mode: Scales, Arpeggios and the
    /// fretboard trainer are otherwise judged only when the model answers.
    pub(super) fn tick_quick_credit(&mut self) {
        if !self.quick_credit || self.paused || !self.one_at_a_time() {
            return;
        }
        if self.heard.audio_frames == self.quick_judged_frame {
            return;
        }
        self.quick_judged_frame = self.heard.audio_frames;
        match self.app_mode {
            AppMode::Intervals | AppMode::Scales | AppMode::Arpeggios => self.quick_credit_step(),
            AppMode::Fretboard => self.quick_credit_fret(),
            AppMode::Formulas => self.quick_credit_formula(),
            AppMode::Chords => {}
        }
    }

    /// Intervals, Scales and Arpeggios: the step due, or under free order the
    /// first step still wanted that was struck and named.
    fn quick_credit_step(&mut self) {
        // The finished set of Intervals is still shown.
        if self.interval_hold > 0.0 {
            return;
        }
        let Some(chord) = self.chords.get(self.current_chord_index) else {
            return;
        };
        let steps = self.ordered_active_indices(chord);
        let all_targets = self.degree_targets(chord);
        let step_targets: Vec<usize> = steps.iter().map(|step| all_targets[step.degree]).collect();
        // The ordinary path sizes the marks to the steps; until it has, there
        // is no mark to write.
        if self.collected_notes.len() != steps.len() {
            return;
        }
        let wanted = |step: &usize| !self.is_collected(*step) && self.struck_and_named(step_targets[*step]);
        let step = if self.free_order() {
            (0..steps.len()).find(wanted)
        } else {
            Some(self.current_note_step).filter(|step| *step < steps.len()).filter(wanted)
        };
        if let Some(step) = step {
            self.journal_ask();
            self.credit_step(step, 0, steps[step].octave, &step_targets);
        }
    }

    fn quick_credit_fret(&mut self) {
        // The note just answered is still being shown.
        if self.fret_hold > 0.0 {
            return;
        }
        if let Some(target) = self.fret_target.filter(|&target| self.struck_and_named(target)) {
            self.credit_branch = Some(0);
            self.pass_fret_target(target);
        }
    }

    /// Formulas: every function still wanted that was struck and named - in
    /// order, only the lowest of them.
    fn quick_credit_formula(&mut self) {
        if self.lap_hold > 0.0 {
            return;
        }
        let next_due = self.formula_collected.iter().position(|&c| !c);
        for (i, pc) in self.formula_pitches().into_iter().enumerate() {
            let wanted = self.formula_collected.get(i) == Some(&false)
                && (!self.formula_in_order || next_due == Some(i));
            if wanted && self.struck_and_named(pc) {
                self.credit_branch = Some(0);
                self.credit_class(pc, 0);
                self.formula_collected[i] = true;
                self.print_formula_credit(i, 0, self.last_ai_root);
            }
        }
    }
}
