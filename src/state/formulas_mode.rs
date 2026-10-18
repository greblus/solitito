//! Formulas: drawing a formula and its placement, the chord under it, the
//! clock that runs the laps, and crediting its functions.

use super::*;

impl MyApp {
    /// Draws the next formula, and a key for it if the options ask for one.
    ///
    /// A filter that nothing satisfies leaves the previous formula standing
    /// rather than blanking the screen - the settings panel is where that gets
    /// explained, not here.
    pub fn next_formula(&mut self) {
        let key = if self.formula_random_key {
            None
        } else {
            crate::formulas::parse_key(&self.formula_key_setting)
        };
        if let Some(d) = crate::formulas::next(
            &mut self.rng,
            self.formula_notes,
            self.formula_required,
            key,
        ) {
            self.formula_mask = d.mask;
            self.formula_root = d.key.pitch() as usize;
            self.formula_key_name = d.key.name();
            self.formula_collected = vec![false; d.mask.count_ones() as usize];
            // Over a chord the drawn key is thrown away again: what the formula
            // is read against is the chord, and where its `1` sits is the
            // placement, not a key of its own.
            if self.formula_exercise != 0 {
                self.take_chord(true);
                self.place_over_chord();
            }
            // As after a finished lap: whatever is still ringing belongs to the
            // formula that has gone.
            self.lap_hold = LAP_HOLD;
            // Without this a log jumps key mid-way with nothing to say why.
            if std::env::var("SOLITITO_FORMULA").is_ok() {
                println!(
                    "--- nowa formula: {} w {}",
                    crate::formulas::to_text(self.formula_mask),
                    self.formula_key_name,
                );
            }
            self.success_timer = 0.0;
            self.match_status = MatchStatus::None;
        }
    }

    /// The chord under the formula, as a root and a set of pitch classes.
    pub(super) fn chord_under(&self) -> Option<(usize, u16)> {
        let c = self.formula_chord.as_ref()?;
        let root = c.root as usize;
        let mut pcs = 0u16;
        for i in c.quality.intervals() {
            pcs |= 1 << ((root + i as usize) % 12);
        }
        Some((root, pcs))
    }

    /// A chord to practise over, drawn at random.
    ///
    /// The five qualities the app already knows, over any of the twelve roots.
    /// Drawn rather than chosen from a tune: this exercise is about hearing one
    /// formula land on one chord in twelve different ways, and a tune would
    /// keep taking the chord away before that had happened.
    pub(super) fn draw_practice_chord(&mut self) -> Chord {
        const QUALITIES: [ChordQuality; 5] = [
            ChordQuality::Major7,
            ChordQuality::Dominant7,
            ChordQuality::Minor7,
            ChordQuality::HalfDiminished,
            ChordQuality::Diminished,
        ];
        Chord {
            root: NoteName::from_index(self.rng.below(12)),
            quality: QUALITIES[self.rng.below(QUALITIES.len())].clone(),
        }
    }

    /// Plants the standing formula on the standing chord.
    ///
    /// The formula does not change - only where its `1` is put, which is the
    /// whole of the exercise: the same set from the chord's own root spells the
    /// chord out, and from another degree it colours it or leaves it
    /// altogether. Everything downstream still works in
    /// terms of `formula_root`, so a placement is just a root arrived at a
    /// different way.
    pub fn place_over_chord(&mut self) {
        let Some((root, pcs)) = self.chord_under() else { return };
        let want = match self.formula_placement_want {
            1 => Some(crate::formulas::Verdict::Defines),
            2 => Some(crate::formulas::Verdict::Colours),
            3 => Some(crate::formulas::Verdict::Outside),
            _ => None,
        };
        let p = crate::formulas::draw_placement(
            &mut self.rng,
            self.formula_mask,
            root,
            pcs,
            want,
        );
        self.formula_degree = p.degree;
        self.formula_hits = p.hits;
        self.formula_verdict = Some(p.verdict);
        self.formula_root = (root + p.degree) % 12;
        self.formula_key_name = crate::formulas::KEY_POOL[self.formula_root].to_string();
        for done in self.formula_collected.iter_mut() {
            *done = false;
        }
        self.lap_hold = LAP_HOLD;
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
    }

    /// The chord the exercise is played over, drawn or taken from the tune.
    pub(super) fn take_chord(&mut self, advance: bool) {
        match self.formula_exercise {
            1 => {
                if advance || self.formula_chord.is_none() {
                    self.formula_chord = Some(self.draw_practice_chord());
                }
            }
            2 => self.take_tune_chord(advance),
            _ => {
                self.formula_chord = None;
                self.formula_verdict = None;
            }
        }
    }

    /// Over the changes: the tune's chord, or the next one along when `advance`.
    pub(super) fn take_tune_chord(&mut self, advance: bool) {
        if self.chords.is_empty() {
            // No tune loaded yet: one drawn chord is better than none, and the
            // tune takes over as soon as it arrives.
            if self.formula_chord.is_none() {
                self.formula_chord = Some(self.draw_practice_chord());
            }
            return;
        }
        if advance {
            self.current_chord_index = (self.current_chord_index + 1) % self.chords.len();
        }
        self.formula_chord = Some(self.chords[self.current_chord_index].clone());
    }

    /// The next chord of the tune, with a placement drawn on it.
    ///
    /// Asked for by hand, from the line under the pause: over the changes there
    /// are two things one might want next - the next chord, or a different
    /// formula to carry through the tune - and they are different gestures.
    pub fn next_change(&mut self) {
        if self.formula_exercise == 0 {
            return;
        }
        self.take_chord(true);
        self.place_over_chord();
    }

    /// Has the estimate been naming this class? See `EAR_VOTES`.
    pub(super) fn ear_says(&self, pc: usize) -> bool {
        self.ear_window.iter().filter(|&&v| v == Some(pc % 12)).count() >= EAR_VOTES
    }

    pub(super) fn tick_formulas(&mut self, dt: f32) {
        if self.formula_collected.is_empty() {
            self.next_formula();
            return;
        }
        self.lap_hold = (self.lap_hold - dt).max(0.0);
        // One reading per new audio frame - no more, or the same frame votes
        // twice; no less, or the judging waits on the model.
        if self.heard.audio_frames != self.judged_frame {
            self.judged_frame = self.heard.audio_frames;
            self.listen_to_new_frame();
            // Paused is "your turn": the whole set lit, nothing judged.
            if !self.paused {
                let done = self.collect_formula(self.last_ai_root, self.last_ai_conf);
                self.match_status = if done { MatchStatus::Exact } else { MatchStatus::None };
            }
        }
        if !self.formula_collected.iter().all(|&c| c) {
            return;
        }
        self.success_timer += dt;
        // Straight on after the pause. Waiting for the formula's own notes to
        // die away was tried, to stop a new lap filling itself in from their
        // decay - but that log predates the vote, and a decay wandering across
        // three classes cannot now hold any of them for four readings of five.
        // The wait was insurance against something already insured, and it was
        // paid for in the only currency this mode has: the seconds after a
        // formula, when the player is already playing the next one.
        let show = self.transition_delay.max(FORMULA_LAP_PAUSE);
        if self.success_timer > show && !self.paused {
            self.restart_formula();
        }
    }

    /// The estimate of a new audio frame into the ear's vote, and the run of
    /// frames it has named the same class.
    pub(super) fn listen_to_new_frame(&mut self) {
        self.ear_window.rotate_left(1);
        self.ear_window[EAR_WINDOW - 1] = self.heard.cqt_pitch;
        if self.heard.cqt_pitch.is_some() && self.heard.cqt_pitch == self.cqt_run_pitch {
            self.cqt_run = self.cqt_run.saturating_add(1);
        } else {
            self.cqt_run_pitch = self.heard.cqt_pitch;
            self.cqt_run = 1;
        }
    }

    /// Starts the standing formula again, exactly as it stands.
    ///
    /// Playing a formula through is not a reason to lose it, and not a reason to
    /// move it either: the exercise is the formula in that key, and the next lap
    /// is the same exercise. A different formula, or a different key for it,
    /// comes from the arrows or the settings.
    pub fn restart_formula(&mut self) {
        // Over a chord a finished lap is not a repeat: the exercise is the same
        // formula heard from somewhere else, so the placement moves. Over a
        // tune the chord moves too, and the formula is carried across it: the
        // constant is the set, and the harmony is what moves under it.
        if self.formula_exercise != 0 {
            self.take_chord(self.formula_exercise == 2);
            self.place_over_chord();
            return;
        }
        for done in self.formula_collected.iter_mut() {
            *done = false;
        }
        // The note that finished the lap is still ringing, and it would mark
        // itself off again the moment the marks cleared - the new lap would
        // start with one function given away. It counts from the next attack.
        self.lap_hold = LAP_HOLD;
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
    }

    /// Puts a kept formula back on screen, in whatever key is showing.
    ///
    /// No key travels with it: a formula is the same exercise in all twelve, and
    /// the key on screen is the one the options chose.
    pub fn load_formula(&mut self, mask: u16) {
        if mask == 0 {
            return;
        }
        self.formula_mask = mask;
        self.formula_collected = vec![false; mask.count_ones() as usize];
        self.lap_hold = LAP_HOLD;
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
    }

    /// Puts the standing formula in the key the options ask for.
    ///
    /// The functions do not move - only the notes they land on, so the marks
    /// start again. A key that cannot be read leaves the old one in place, as
    /// `next_formula` leaves the old formula when the filter matches nothing.
    pub fn rekey_formula(&mut self) {
        if self.formula_exercise != 0 {
            self.take_chord(false);
            self.place_over_chord();
            return;
        }
        let key = if self.formula_random_key {
            crate::formulas::draw_key(&mut self.rng)
        } else {
            crate::formulas::parse_key(&self.formula_key_setting)
        };
        if let Some(k) = key {
            self.formula_root = k.pitch() as usize;
            self.formula_key_name = k.name();
        }
        for done in self.formula_collected.iter_mut() {
            *done = false;
        }
        self.success_timer = 0.0;
        self.match_status = MatchStatus::None;
    }

    /// Pitch class of each function in the current formula.
    pub fn formula_pitches(&self) -> Vec<usize> {
        crate::formulas::functions_of(self.formula_mask)
            .iter()
            .map(|&f| (self.formula_root + f) % 12)
            .collect()
    }

    /// Marks off whatever is sounding, and says whether the set is complete.
    ///
    /// Unordered on purpose: the exercise is to move around inside the set, so
    /// any function may be struck at any time. The formula is finished when
    /// every one of them has sounded.
    ///
    /// Three of the four ways in count here; the chord NAME does not. It says
    /// what the model made of 0.77 s and credits whatever function its root
    /// lands on, and a mark here never expires, so a name that named the wrong
    /// root would light a function for the rest of the exercise.
    ///
    /// The single-frame estimate was tried as the sole way in and read a plain
    /// E as Ab and Eb; the model's pitch head alone was tried next and took
    /// three or four seconds a note, its window being 0.77 s long. Both are
    /// needed, which is what `note_is_sounding` was built for.
    ///
    /// Two rules were tried and dropped, both recorded so they are not tried
    /// again: rationing credits to one per attack, which handed the credit to
    /// the note still ringing from before and left the note actually played
    /// nothing to claim; and crediting only near an attack, which lost the
    /// notes of anyone not picking hard enough to be heard as one.
    pub(super) fn collect_formula(&mut self, ai_root: Option<NoteName>, confidence: f32) -> bool {
        let pitches = self.formula_pitches();
        let funcs = crate::formulas::functions_of(self.formula_mask);
        // What the ear reported and what was credited. On by default while the
        // mode is being tuned - it is the only way to tell a bad reading from a
        // bad rule, and it says one line per credit, not per frame.
        // SOLITITO_FORMULA=0 silences it.
        let loud = std::env::var("SOLITITO_FORMULA")
            .map(|v| v != "0")
            .unwrap_or(self.log_credits && !cfg!(test));
        // A new lap holds for a moment: what finished the last one is still
        // ringing, and the ear would hand it straight back.
        if self.lap_hold > 0.0 {
            return false;
        }
        // Measured on a real recording, over 49 notes (dist/latency_stats.py):
        //
        //   the three ways in together      110 false credits, 0 notes missed
        //   the steady estimate alone         33 false credits, 0 notes missed
        //   and with the take6 onset head     15 false credits, 4 notes missed
        //
        // So the model's pitch head is struck out here. It answers "what is
        // sounding", and a string ringing on - or one resonating in sympathy,
        // which is why the fourth and the fifth kept lighting up - is sounding
        // without having been played. It was 99 of those 110.
        //
        // The take6 onset head was tried as a gate here and dropped: live it
        // answered 0.09 for notes plainly being played, refusing far more than
        // it caught. The strike detector backs the ear instead - `EAR_BACKED`.
        // In order: only the lowest function not yet marked off may count. The
        // set is written low to high, so this is the same set read as a line -
        // and nothing else is refused loudly, it simply does not count yet.
        let next_due = self.formula_collected.iter().position(|&c| !c);
        for (i, &pc) in pitches.iter().enumerate() {
            if self.formula_collected[i] {
                continue;
            }
            if self.formula_in_order && next_due != Some(i) {
                continue;
            }
            let Some(branch) = self.sounding_by(pc, ai_root, confidence) else {
                continue;
            };
            // Strict: the single-frame estimate, and it has to have been saying
            // so.
            //
            // Loose (the default): the rule the note modes use, any of the four
            // ways in at once. It credited 110 things nobody played over 49
            // notes, against 33 for this one.
            if self.strict_formulas && (branch != 1 || !self.ear_says(pc)) {
                continue;
            }
            self.credit_class(pc, 0);
            self.formula_collected[i] = true;
            if loud {
                println!(
                    "atak#{} zaliczono {:<3} (sposob {branch})  ucho={:<3} model_prymy={:<3}",
                    self.heard.onset_id,
                    funcs.get(i).map(|&f| crate::formulas::FUNCS[f]).unwrap_or("?"),
                    self.heard.cqt_pitch
                        .map(|p| NoteName::from_index(p).to_string().to_string())
                        .unwrap_or_else(|| "-".into()),
                    ai_root
                        .map(|r| r.to_string().to_string())
                        .unwrap_or_else(|| "-".into()),
                );
            }
        }
        !self.formula_collected.is_empty() && self.formula_collected.iter().all(|&c| c)
    }
}
