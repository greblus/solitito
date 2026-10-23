//! Intervals, Scales and Arpeggios: which step the sound answers, and the
//! clock that judges Intervals on every new audio frame.

use super::*;

impl MyApp {
    pub(super) fn tick_intervals(&mut self, dt: f32) {
        self.interval_model_age += dt;
        if self.interval_model_age >= INTERVAL_MODEL_AGE {
            self.forget_model_answer();
        }
        self.credited.settle(dt, &self.heard);

        self.interval_audio_age += dt;
        let fresh = self.heard.audio_frames != self.judged_frame;
        let elapsed = self.interval_audio_age.min(0.032);
        if fresh {
            self.take_interval_frame(elapsed);
        }

        // The completed result has its own clock. It stays visible through
        // silence and pause, and cannot be overwritten by more predictions.
        if self.interval_hold > 0.0 {
            if !self.paused {
                self.interval_hold = (self.interval_hold - dt).max(0.0);
                if self.interval_hold == 0.0 {
                    self.advance_chord();
                }
            }
            return;
        }
        if !fresh {
            return;
        }
        if !self.heard.audio_gate_open {
            self.answering_step = None;
            self.success_timer = 0.0;
            return;
        }
        self.check_progress(elapsed, self.last_ai_root, "", self.last_ai_conf);
    }

    /// A stopped inference stream must not keep answering new audio frames.
    pub(super) fn forget_model_answer(&mut self) {
        self.last_pitches = [0.0; 12];
        self.prev_pitches = [0.0; 12];
        self.last_ai_root = None;
        self.last_ai_conf = 0.0;
    }

    /// A new audio frame in Intervals, `elapsed` seconds after the one before.
    pub(super) fn take_interval_frame(&mut self, elapsed: f32) {
        self.judged_frame = self.heard.audio_frames;
        // A fresh sample after a stalled UI/audio stream does not prove
        // the note held throughout the gap.
        if self.interval_audio_age > 0.05 {
            self.answering_step = None;
            self.success_timer = 0.0;
            self.interval_silence = 0.0;
        }
        self.interval_audio_age = 0.0;
        if self.heard.audio_gate_open {
            self.interval_silence = 0.0;
            return;
        }
        self.interval_silence += elapsed;
        // Muting between plucks is evidence of a fresh note too. A missing CQT
        // estimate alone is NOT silence: require the gate. Track this during
        // the completed-set display as well.
        if self.interval_silence >= 0.2 {
            self.credited.all_gone_quiet();
        }
    }

    /// Intervals, Scales and Arpeggios: which step the sound answers, and
    /// whether it has answered long enough to be credited.
    ///
    /// These modes are NOT gated by chord confidence. `confidence` is the
    /// product of root and quality confidence, so an uncertain quality would
    /// block a note that is perfectly audible. Only the pitch head matters
    /// here - F1 0.90 against ~80% for the chord name.
    pub(super) fn check_notes(&mut self, dt: f32, ai_root: Option<NoteName>, confidence: f32) {
        let Some(chord) = self.chords.get(self.current_chord_index) else {
            return;
        };
        let steps = self.ordered_active_indices(chord);
        let all_targets = self.degree_targets(chord);
        let step_targets: Vec<usize> = steps.iter().map(|step| all_targets[step.degree]).collect();

        // The marks and the steps have to be the same length. They are
        // sized together when the exercise is set up, but a setting
        // that changes the STEPS - the closing root of a scale - can
        // land between two of those, and then the last step has no mark
        // to write. Resized rather than rebuilt: what has been played
        // this lap has been played.
        if self.collected_notes.len() != steps.len() {
            self.collected_notes.resize(steps.len(), false);
        }
        self.journal_ask();

        // Which step is being answered.
        //
        // A scale or an arpeggio IS an order - played out of it, it is
        // not that scale - so there the answer can only be the step due.
        // A grip is not: three notes under one hand, and which finger
        // lands first is not the exercise. So Intervals takes them in
        // any order unless the option asks otherwise, shuffled or not.
        //
        // One pluck still credits one step. Without that an exercise
        // asking for the same pitch class twice runs through both steps
        // on a single ringing string, with nothing played in between. A
        // step asking for the class just credited needs the string
        // struck again - see `struck_since_credit`.
        let answer = if self.free_order() {
            self.best_answer(&step_targets, ai_root, confidence)
        } else if self.current_note_step < steps.len() {
            let due = self.current_note_step;
            self.answer_for(step_targets[due], ai_root, confidence)
                .map(|by| (due, by))
        } else {
            return;
        };

        self.time_the_answer(answer.map(|(step, _)| step), dt);
        if !self.held_past(NOTE_HOLD) {
            return;
        }
        let Some((step, by)) = answer else {
            return;
        };
        self.credit_step(step, by, steps[step].octave, &step_targets);
    }

    /// Whether `target` answers for its step, and by which of the four ways -
    /// the lower the number the better the evidence, see `sounding_by`.
    pub(super) fn answer_for(&self, target: usize, ai_root: Option<NoteName>, confidence: f32) -> Option<u8> {
        if self.relaxed_chords() {
            return self.sounding_by(target, ai_root, confidence);
        }
        if !self.struck_since_credit(target) {
            return None;
        }
        // The model's pitch head spreads a plucked note onto its
        // neighbours, so where the single-frame estimate names a
        // class, no OTHER class may be credited off the model that
        // frame - in both orders, since in order the root is
        // credited, the target moves to the third while the root
        // rings on, and a head trained on chord shapes offers it.
        //
        // HOW MANY voices, not which. Counting them is the part
        // of this witness that holds up; asking whether one
        // particular class is among them is the part with blind
        // spots - an octave cannot be resolved at all and a quiet
        // fifth is missed, so a strummed grip would come back
        // short and nothing but the note the ear names would
        // count. Two voices is already something one string cannot
        // be, and it is what the exception needs to know.
        //
        // The chord name used to answer this and must not again:
        // one plucked root is enough for the model to recognise
        // the shape, which is how a third nobody touched was
        // credited.
        let ear_names_another = self.heard.cqt_pitch.is_some_and(|now| now != target % 12);
        let grip_is_sounding = self.heard.several_sounding() && !self.single_notes;
        if ear_names_another && !grip_is_sounding {
            return None;
        }
        self.sounding_by(target, ai_root, confidence)
    }

    /// Under free order, the step that answers best among those still wanted.
    ///
    /// The BEST answer, not the first: taking the lowest-numbered step that
    /// answered credited whichever of them the model happened to name, in the
    /// order they were dealt. The estimate names one class only, so where it
    /// speaks it decides; below it, the class the pitch head is loudest on.
    pub(super) fn best_answer(
        &self,
        step_targets: &[usize],
        ai_root: Option<NoteName>,
        confidence: f32,
    ) -> Option<(usize, u8)> {
        let louder_first = |a: usize, b: usize| {
            let level = |step: usize| self.last_pitches[step_targets[step] % 12];
            level(b).partial_cmp(&level(a)).unwrap_or(std::cmp::Ordering::Equal)
        };
        (0..step_targets.len())
            .filter(|&step| !self.is_collected(step))
            .filter_map(|step| {
                self.answer_for(step_targets[step], ai_root, confidence)
                    .map(|by| (step, by))
            })
            .min_by(|a, b| a.1.cmp(&b.1).then_with(|| louder_first(a.0, b.0)))
    }

    pub(super) fn is_collected(&self, step: usize) -> bool {
        self.collected_notes.get(step).copied().unwrap_or(false)
    }

    /// Runs the success timer for the step answering. The count belongs to
    /// the step it started on: another step answering is a new answer, not a
    /// continuation of this one.
    pub(super) fn time_the_answer(&mut self, answering: Option<usize>, dt: f32) {
        let Some(step) = answering else {
            self.answering_step = None;
            self.success_timer = 0.0;
            return;
        };
        if self.answering_step != Some(step) {
            self.answering_step = Some(step);
            self.success_timer = 0.0;
        }
        self.success_timer += dt;
    }

    /// Marks `step` played, and when it was the last one wanted, finishes the
    /// set: Intervals shows it complete, the other modes move to the next chord.
    pub(super) fn credit_step(&mut self, step: usize, by: u8, octave: i8, step_targets: &[usize]) {
        if step < self.collected_notes.len() {
            self.collected_notes[step] = true;
        }
        self.credit_branch = Some(by);
        self.credit_class(step_targets[step], octave);
        self.last_played = Some(step_targets[step] % 12);
        self.answering_step = None;
        self.success_timer = 0.0;
        // Where the exercise stands: the first step still wanted. In order
        // that is the next one along, which is what it always was.
        let step_count = step_targets.len();
        self.current_note_step = (0..step_count)
            .find(|&i| !self.is_collected(i))
            .unwrap_or(step_count);
        if self.current_note_step < step_count {
            return;
        }
        if self.app_mode == AppMode::Intervals {
            self.match_status = MatchStatus::Exact;
            self.interval_hold = INTERVAL_SHOW;
        } else {
            self.advance_chord();
        }
    }
}
