//! What the judge knows of the sound: everything the audio thread reports,
//! copied in once per audio frame, and the questions that are answered from
//! it alone.

use crate::audio::AudioAnalysis;

use super::{CREDIT_TICKS, LEFT_TICKS};

pub struct Heard {
    /// Audio frames pushed so far; the last one judged is `MyApp::judged_frame`.
    ///
    /// The judging is driven off THIS, not off the UI clock and not off the
    /// model's answers. Off the UI clock the same frame was sampled two or
    /// three times, which let one bad reading carry a vote; off the model's
    /// answers it stopped whenever the model did - and the model is asked only
    /// while the context window is nine tenths full, which playing one note at
    /// a time never manages. Once per new frame is once per new reading.
    pub audio_frames: u64,
    pub audio_gate_open: bool,
    /// What the estimate has been saying, and for how many audio frames. A
    /// drawing that follows every frame flickers on noise; one that waits for a
    /// reading to hold does not. See `steady_note`.
    pub steady_pitch: Option<usize>,
    pub steady_for: u32,
    /// The same reading as an absolute semitone, for telling a note from itself
    /// an octave down. See `octave_is_answered`.
    pub cqt_semitone: Option<usize>,
    /// Pitch class sounding in the last CQT frame - see `audio::mono_pitch`.
    /// `None` while the gate is shut or the estimate is weak.
    pub cqt_pitch: Option<usize>,
    /// Attack counter, mirrored from the audio thread.
    pub onset_id: u64,
    /// The recording's hop clock, so a journal row is a position in the WAV.
    pub capture_hop: u64,
    /// Pitch classes genuinely sounding, one bit each, mirrored from the audio
    /// thread. See `voices::voices`.
    pub voices: u16,
    /// Whether `voices` has ever reported anything, so a build, a replay or a
    /// test without it falls back to the older reading of `left` instead of
    /// never setting it at all.
    pub voices_seen: bool,
    /// Frames each class has been absent from `voices` for.
    ///
    /// One frame's absence is not a note stopping. Measured on sesja-g1, only
    /// about half the credits had a second voice reported, so a note ringing
    /// under the one being played drops out of the list and comes back - and
    /// taking the first gap for silence is what let a note credited in one
    /// chord be credited again in the next.
    pub quiet_for: [u32; 12],
    /// Attacks the flux detector has counted, mirrored from the audio thread.
    /// Pitch-blind: it says a string was hit, never which one.
    pub flux_id: u64,
    /// Hops since the last one.
    pub flux_age: u32,
    /// Strikes of each class, and hops since each - see `strike::Strikes`.
    pub class_strikes: [u32; 12],
    pub class_strike_age: [u32; 12],
    /// Whether that detector is running. Without its model file it is not, and
    /// the rules fall back to what they were before it existed.
    pub strikes_live: bool,
    /// Whether that detector has ever reported, so a build or a recording
    /// without it falls back to the older evidence instead of refusing
    /// everything.
    pub flux_seen: bool,
}

impl Default for Heard {
    /// Nothing heard yet - and no strike or flux report for as long as the
    /// counters can say, so nothing counts as just struck at start-up.
    fn default() -> Self {
        Heard {
            audio_frames: 0,
            audio_gate_open: false,
            steady_pitch: None,
            steady_for: 0,
            cqt_semitone: None,
            cqt_pitch: None,
            onset_id: 0,
            capture_hop: 0,
            voices: 0,
            voices_seen: false,
            quiet_for: [0; 12],
            flux_id: 0,
            flux_age: u32::MAX,
            class_strikes: [0; 12],
            class_strike_age: [u32::MAX; 12],
            strikes_live: false,
            flux_seen: false,
        }
    }
}

impl Heard {
    /// What the audio thread has heard, copied into the judge.
    pub fn read_audio_frame(&mut self, state: &AudioAnalysis) {
        if state.frames_seen != self.audio_frames {
            self.count_steady_pitch(state.cqt_pitch);
            self.count_quiet_classes(state.voices);
        }
        self.cqt_pitch = state.cqt_pitch;
        self.cqt_semitone = state.cqt_semitone;
        self.onset_id = state.onset_id;
        if state.flux_id != self.flux_id {
            self.flux_seen = true;
        }
        self.flux_id = state.flux_id;
        self.flux_age = state.flux_age;
        self.class_strikes = state.class_strikes;
        self.class_strike_age = state.class_strike_age;
        self.strikes_live = state.strikes_live;
        self.capture_hop = state.hops;
        if state.voices != 0 {
            self.voices_seen = true;
        }
        self.voices = state.voices;
        self.audio_frames = state.frames_seen;
    }

    /// How long the single-frame estimate has been saying the same thing.
    /// Counted for every mode, not only Formulas, because a reading that has
    /// held for a few frames is the difference between a note played and the
    /// room answering back - see the fretboard drawing, which flickered on
    /// every stray reading before this.
    fn count_steady_pitch(&mut self, pitch: Option<usize>) {
        if pitch.is_some() && pitch == self.steady_pitch {
            self.steady_for = self.steady_for.saturating_add(1);
        } else {
            self.steady_pitch = pitch;
            self.steady_for = 1;
        }
    }

    /// How many frames each class has been out of the voices. On the audio
    /// clock, which is also the one the tests drive. A closed noise gate stops
    /// it, and that case has its own rule: see `interval_silence`.
    fn count_quiet_classes(&mut self, voices: u16) {
        for (pc, quiet) in self.quiet_for.iter_mut().enumerate() {
            if voices & (1 << pc) == 0 {
                *quiet = quiet.saturating_add(1);
            } else {
                *quiet = 0;
            }
        }
    }

    fn held_for(&self, ticks: u32) -> Option<usize> {
        (self.steady_for >= ticks).then_some(self.steady_pitch).flatten()
    }

    /// What is sounding, once it has been saying so for long enough to believe.
    ///
    /// Three audio frames is 48 ms - short enough that an answer still feels
    /// immediate, long enough that a single stray reading from the room does
    /// not paint the neck.
    pub fn steady_note(&self) -> Option<usize> {
        self.held_for(LEFT_TICKS)
    }

    /// What is sounding, by the shorter measure a CREDIT is allowed to use.
    ///
    /// Two questions, two thresholds. "Has the target been sounding long enough
    /// to credit" costs one wrong credit when it is wrong; "has something ELSE
    /// taken over, so the note credited a moment ago has gone" is what stops a
    /// repeated note being credited twice off one pluck, and loosening that
    /// brings the repeats back. So only the first one is short.
    pub fn sounding_now(&self) -> Option<usize> {
        self.held_for(CREDIT_TICKS)
    }

    /// Where notes are played one at a time: Scales, Arpeggios and the
    /// Fretboard, and Intervals when the option says so. Everywhere else a
    /// strummed chord is allowed to walk its intervals off a single attack -
    /// see `a_strummed_chord_still_walks_its_intervals`.
    /// Several strings sounding at once, one of them the note being asked for.
    ///
    /// This is what the chord-name exception was reaching for and could not
    /// reach: a confidently recognised chord says the model knows the shape,
    /// not that its tones are in the air - and one plucked root is enough for
    /// it to recognise the shape. `voices` explains the spectrum instead of
    /// ranking it, so a class that is only the root's harmonic is not in it.
    ///
    /// Both halves. More than one voice alone would let any ringing pair carry
    /// a third nobody touched; the target being among them alone is branch 1
    /// wearing a disguise.
    /// Whether that class is sounding at all - not whether it is the loudest.
    pub fn voice_heard(&self, pc: usize) -> bool {
        self.voices & (1 << (pc % 12)) != 0
    }

    /// More than one string sounding - which one plucked note cannot be.
    ///
    /// Where nothing has ever been reported there is no witness, and the rule
    /// is as loose as it was before there was one.
    pub fn several_sounding(&self) -> bool {
        !self.voices_seen || self.voices.count_ones() >= 2
    }
}
