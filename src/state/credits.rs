//! What has been credited, class by class, and the rules that decide when a
//! class may be credited again.

use std::ops::{Index, IndexMut};

use super::{Heard, CLASS_FRESH, CREDIT_SETTLE, FLUX_FRESH, GONE_QUIET};

/// What was true when a pitch class was last credited.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Credit {
    /// The envelope's attack count at that moment.
    pub onset: u64,
    /// Attacks the flux detector had counted, whatever their pitch.
    pub flux: u64,
    /// That class's own strike count from `strike::Strikes` at that moment.
    pub class_strike: u32,
    /// The octave marker of the step credited: 0 for the plain degree, 1 for
    /// one written `1'`. A step marked higher than the credit is asking for the
    /// same note in another octave, and that is answerable.
    pub octave: i8,
    /// What the single-frame estimate was reading, as an absolute semitone.
    pub semitone: Option<usize>,
    /// Seconds left before the note can count as having left - see
    /// `CREDIT_SETTLE`.
    pub settle: f32,
    /// Whether something else has been heard, steadily, since this note was
    /// credited. Then the note has stopped sounding and anything heard from it
    /// afterwards is a fresh pluck - whether or not a detector noticed.
    pub left: bool,
}

/// The last credit of each pitch class, indexed by class.
///
/// What was credited outlives a chord or a lap - a run ending on the note it
/// starts from must not hand the next lap its first step off the last one
/// still ringing - but not a change of exercise.
#[derive(Default)]
pub struct Credits {
    by_class: [Option<Credit>; 12],
}

impl Index<usize> for Credits {
    type Output = Option<Credit>;
    fn index(&self, pc: usize) -> &Option<Credit> {
        &self.by_class[pc]
    }
}

impl IndexMut<usize> for Credits {
    fn index_mut(&mut self, pc: usize) -> &mut Option<Credit> {
        &mut self.by_class[pc]
    }
}

impl Credits {
    pub fn forget_all(&mut self) {
        self.by_class = [None; 12];
    }

    /// Credits class `pc`, keeping what was heard at that moment.
    pub fn record(&mut self, pc: usize, octave: i8, heard: &Heard) {
        self.by_class[pc % 12] = Some(Credit {
            onset: heard.onset_id,
            flux: heard.flux_id,
            class_strike: heard.class_strikes[pc % 12],
            octave,
            semitone: heard.cqt_semitone.filter(|n| n % 12 == pc % 12),
            settle: CREDIT_SETTLE,
            left: false,
        });
    }

    /// Every credited note counts as having stopped: what is heard from it
    /// next is a fresh pluck.
    pub fn all_gone_quiet(&mut self) {
        for credit in self.by_class.iter_mut().flatten() {
            credit.left = true;
            credit.settle = 0.0;
        }
    }

    /// Whether a class asked for again has been played again.
    ///
    /// Two ways, and the measurements behind them:
    ///
    /// The estimate reads an absolute pitch, so a note sounding six semitones
    /// or more from where it was read when it was credited is a different
    /// string being played - the closing root of `1 2 3 4 5 6 7 1` against the
    /// opening one still ringing. That is proof on its own, and it needs no
    /// attack: on the test run the estimate read the opening root and the
    /// closing one an octave apart, 0.29 s after the string was hit. It cannot
    /// be a REQUIREMENT, though: a run closed in the octave it started in would
    /// never satisfy it, however many times it was played.
    ///
    /// Otherwise the class's own strike counter has to have moved - and, where
    /// notes are played one at a time, the estimate must not be reading some
    /// other note: a strike reported for the wrong class lands while the
    /// estimate reads the note actually being played, so it does not pass.
    ///
    /// Without the strike detector, flux - which says a string was hit, not
    /// which - stands in, and the envelope where flux has never reported.
    pub fn struck_since(&self, pc: usize, heard: &Heard, one_at_a_time: bool) -> bool {
        let pc = pc % 12;
        let Some(c) = self.by_class[pc] else {
            return true;
        };
        // The note stopped sounding and CAME BACK: something else was heard
        // steadily since it was credited, and the single-frame estimate is
        // reading this note again now. Both halves matter - "something else was
        // heard" alone let a note credited a moment ago count a second time
        // while it merely rang on under the note being played, which is the
        // multiple-crediting this rule exists to stop. The estimate is
        // monophonic: it names what is loudest, so it names the ringing note
        // again only once the string is struck again.
        if c.left && heard.steady_note() == Some(pc) {
            return true;
        }
        let elsewhere = heard.cqt_pitch.is_some_and(|now| now != pc);
        // THIS class struck since its credit, and just now - the question the rule
        // always meant to ask, and the first evidence that can answer it. A note played
        // again an octave away is a strike of the same class and counts through here;
        // nothing else needs to look at octaves. Without the detector, an octave flip
        // of a decaying note was the main leak: `docs/judge-measurements.md#struck_since`.
        if heard.strikes_live {
            return heard.class_strikes[pc] != c.class_strike
                && heard.class_strike_age[pc] <= CLASS_FRESH
                && !(one_at_a_time && elsewhere);
        }
        // Without the detector: an octave jump is believed only with an attack
        // behind it, which is what made it a leak - see above.
        let fresh_attack = heard.flux_seen && heard.flux_id != c.flux && heard.flux_age <= FLUX_FRESH;
        let played_an_octave_away = match (heard.cqt_semitone, c.semitone) {
            (Some(now), Some(then)) => now % 12 == pc && (now >= then + 6 || now + 6 <= then),
            _ => false,
        };
        if fresh_attack && played_an_octave_away {
            return true;
        }
        // Pitch-blind, on purpose: whether a string was hit is a far easier question
        // than which one, and the flux detector answers it without a class to get
        // wrong. WHICH note it was is left to the monophonic estimate, which names the
        // loudest class - see `docs/judge-measurements.md#struck_since`.
        //
        // JUST NOW, not merely since. The counter is pitch-blind, so "an attack since
        // the credit" is satisfied by any other step being played - which is how a note
        // credited in one chord was credited again in the next while it merely rang on.
        // A string struck again while it still rings has to keep working, so this
        // cannot ask for silence first - only for the hit to be happening now.
        let struck = if heard.flux_seen {
            heard.flux_id != c.flux && heard.flux_age <= FLUX_FRESH
        } else {
            heard.onset_id != c.onset
        };
        // Not "the estimate names it" but "the estimate does not name something
        // else": a note too quiet for the estimate to score would otherwise
        // never be creditable a second time.
        struck && !(one_at_a_time && elsewhere)
    }

    /// Counts down each credit's `settle`, and marks a note `left` once it has
    /// gone quiet - see `Credit::left`.
    pub fn settle(&mut self, dt: f32, heard: &Heard) {
        // Steady, not any frame: the estimate flickers onto a ringing note for
        // a frame at a time, and a flicker is not the note stopping.
        let elsewhere = heard.steady_note();
        let quiet_for = heard.quiet_for;
        let know_voices = heard.voices_seen;
        for (c, credit) in self.by_class.iter_mut().enumerate() {
            let Some(cr) = credit else {
                continue;
            };
            if cr.settle > 0.0 {
                cr.settle = (cr.settle - dt).max(0.0);
            }
            // This note has gone quiet, and what is heard from it next is
            // a new pluck. Without it a note asked for again a few steps
            // later could not be credited at all when no detector caught
            // the pluck - the fretboard trainer stuck on a note that
            // was plainly right.
            //
            // "Something else is the loudest thing" is not that note going
            // quiet, and taking it for that is how a note credited in one
            // chord was credited again in the next: by the boundary the
            // other steps had been heard, so `left` was set, and the note
            // only had to still be ringing for the new chord to take it.
            // Shuffle made it show by putting a shared class at the
            // boundary, which a fixed progression may never do.
            //
            // So `voices` decides, which explains the spectrum rather than
            // ranking it: a class in it is sounding, loudest or not. Where
            // it says nothing at all - no audio yet - the older reading
            // stands, and the gate is no stricter than it was.
            let other_note_heard = elsewhere.is_some_and(|other| other != c);
            let gone_quiet = !know_voices || quiet_for[c] >= GONE_QUIET;
            if cr.settle <= 0.0 && other_note_heard && gone_quiet {
                cr.left = true;
            }
        }
    }
}
