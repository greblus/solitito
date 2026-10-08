//! What the exercise asked for and what it credited, beside the recording.
//!
//! Enabled by the same `SOLITITO_RECORD` as the audio, so one session produces
//! `<prefix>-g<n>.wav` and `<prefix>-g<n>.jsonl` sharing a generation and a
//! clock. The clock is the hop counter the recording is written on, so a row's
//! `hop` is an exact sample position in the WAV: `hop * HOP_LENGTH / TARGET_SR`
//! seconds.
//!
//! Why this exists: measuring crediting needs to know what was played, and
//! nothing here knows that. What the app ASKED for is not the same thing - a
//! wrong note is still a note - but it is written down rather than guessed, and
//! together with the attack times it settles the questions that are structural:
//! two credits off one pluck, a credit with no pluck behind it, a request that
//! never got answered.
//!
//! Each credit also carries the evidence that produced it, so a proposed rule
//! can be tried against the log without playing anything again.

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

/// The generation is resolved ONCE, so the audio thread and the judge agree on
/// which files this run owns. Probing separately would race.
static BASE: OnceLock<Option<PathBuf>> = OnceLock::new();

/// `<prefix>-g<n>` for the first n whose .wav and .jsonl are both free, or None
/// when `SOLITITO_RECORD` is unset.
pub fn base() -> Option<&'static PathBuf> {
    BASE.get_or_init(|| {
        let prefix = PathBuf::from(std::env::var_os("SOLITITO_RECORD")?);
        (1..1000).find_map(|generation| {
            let mut name = prefix.as_os_str().to_os_string();
            name.push(format!("-g{generation}"));
            let base = PathBuf::from(name);
            let free = |extension: &str| !base.with_extension(extension).exists();
            (free("wav") && free("jsonl")).then_some(base)
        })
    })
    .as_ref()
}

pub struct Journal {
    file: std::io::BufWriter<std::fs::File>,
    failed: bool,
}

impl Journal {
    pub fn open() -> Option<Self> {
        let path = base()?.with_extension("jsonl");
        match std::fs::File::create(&path) {
            Ok(file) => {
                eprintln!("📝 {}", path.display());
                Some(Self { file: std::io::BufWriter::new(file), failed: false })
            }
            Err(error) => {
                eprintln!("📝 nie mogę utworzyć {}: {error}", path.display());
                None
            }
        }
    }

    /// One row. Flushed as it goes: a session that ends by closing the window
    /// still leaves a readable file.
    pub fn write(&mut self, row: &serde_json::Value) {
        if self.failed {
            return;
        }
        let result = writeln!(self.file, "{row}").and_then(|()| self.file.flush());
        if let Err(error) = result {
            self.failed = true;
            eprintln!("📝 dziennik przerwany: {error}");
        }
    }
}
