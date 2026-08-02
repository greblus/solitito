use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

const FEATURE_SIZE: usize = 168;
const CTX_FRAMES: usize = 48;

/// Roots in the trainer's order (`ROOTS`); index 12 is "Noise".
const ROOTS: [&str; 13] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Noise",
];

/// Qualities in the trainer's order (`QUALITIES`). The order MUST match - these
/// are the indices of the `quality_logits` output.
const QUALITIES: [&str; 11] = [
    "maj", "min", "maj7", "dom7", "min7", "m7b5", "dim7", "aug", "sus", "note", "N",
];

/// Quality name IN THE APP'S NOTATION.
///
/// A CONTRACT with `state.rs`: `parse_ai_prediction` splits on whitespace and
/// compares the second part against `ChordQuality::to_string()`, which yields
/// exactly "Maj7" / "m7" / "7" / "m7b5". Returning trainer taxonomy names here
/// ("maj7", "dom7") would mean no chord ever matches. Empty for major, because
/// `state.rs` uses "" in its partial-match rules.
fn quality_suffix(q: &str) -> &'static str {
    match q {
        "maj" => "",
        "min" => "m",
        "maj7" => "Maj7",
        "dom7" => "7",
        "min7" => "m7",
        "m7b5" => "m7b5",
        "dim7" => "dim",     // state.rs has a partial rule for ("m7b5", "dim")
        "aug" => "aug",
        "sus" => "sus4",
        _ => "",
    }
}

/// One window's result. The app votes over several of these.
#[derive(Clone, Debug)]
pub struct Prediction {
    /// Ready-made string, e.g. "C Maj7" or "A m7".
    pub chord: String,
    /// Confidence of root AND quality (product) - the vote weight.
    pub confidence: f32,
    /// Probabilities of the 12 pitch classes. The Intervals/Scales/Arpeggios
    /// modes rely on this rather than on the chord name.
    pub pitches: [f32; 12],
    /// Distribution over qualities (`QUALITIES` order) - diagnostics only.
    pub qual_probs: [f32; 11],
    /// Root index (12 = Noise) - diagnostics only.
    pub root_idx: usize,
}

impl Default for Prediction {
    fn default() -> Self {
        Self {
            chord: "...".to_string(),
            confidence: 0.0,
            pitches: [0.0; 12],
            qual_probs: [0.0; 11],
            root_idx: 12,
        }
    }
}

pub struct ChordBrain {
    session: Session,
}

impl ChordBrain {
    pub fn new(model_path: &str) -> Result<Self> {
        println!("🧠 Model: {}", model_path);

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    pub fn predict(
        &mut self,
        frames: &[[f32; FEATURE_SIZE]; CTX_FRAMES],
    ) -> Result<Prediction> {
        let flat: Vec<f32> = frames.iter().flat_map(|f| f.iter().copied()).collect();
        let shape = vec![1, CTX_FRAMES as i64, FEATURE_SIZE as i64];
        let input = Value::from_array((shape, flat))?;

        let outputs = self.session.run(ort::inputs!["features" => input])?;

        let (_, root_t) = outputs["root_logits"].try_extract_tensor::<f32>()?;
        let (_, qual_t) = outputs["quality_logits"].try_extract_tensor::<f32>()?;
        let (_, pitch_t) = outputs["pitch_logits"].try_extract_tensor::<f32>()?;

        let (root_idx, root_conf) = argmax_softmax(root_t);
        let (qual_idx, qual_conf) = argmax_softmax(qual_t);
        let qual_probs = softmax11(qual_t);

        let mut pitches = [0.0f32; 12];
        for (i, slot) in pitches.iter_mut().enumerate() {
            *slot = 1.0 / (1.0 + (-pitch_t[i]).exp());
        }

        // "Noise" on either side means no meaningful chord. The pitch vector is
        // returned anyway - the note modes work even when the chord name is
        // irrelevant, e.g. for a single note.
        let qual = QUALITIES[qual_idx.min(QUALITIES.len() - 1)];
        if root_idx >= 12 || qual == "N" {
            return Ok(Prediction {
                chord: "Noise".to_string(),
                confidence: 0.0,
                pitches,
                qual_probs,
                root_idx,
            });
        }

        let root = ROOTS[root_idx];
        // The format MUST contain a space - `parse_ai_prediction` splits on
        // whitespace. A single note goes as "Note X"; the parser has a branch for it.
        let chord = if qual == "note" {
            format!("Note {}", root)
        } else {
            format!("{} {}", root, quality_suffix(qual)).trim().to_string()
        };

        Ok(Prediction {
            chord,
            // A chord is right when both root AND quality are - so confidence is
            // their product, the same metric the trainer reports as `exact`.
            confidence: root_conf * qual_conf,
            pitches,
            qual_probs,
            root_idx,
        })
    }
}

/// Full softmax over the 11 qualities - shows WHAT the model is torn between.
fn softmax11(logits: &[f32]) -> [f32; 11] {
    let mx = logits.iter().take(11).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut out = [0.0f32; 11];
    let mut sum = 0.0;
    for i in 0..11.min(logits.len()) {
        out[i] = (logits[i] - mx).exp();
        sum += out[i];
    }
    if sum > 0.0 {
        for v in &mut out { *v /= sum; }
    }
    out
}

fn argmax_softmax(logits: &[f32]) -> (usize, f32) {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum_exp = 0.0;
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;

    for (i, &val) in logits.iter().enumerate() {
        sum_exp += (val - max_logit).exp();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    if sum_exp == 0.0 {
        return (0, 0.0);
    }
    (max_idx, (max_val - max_logit).exp() / sum_exp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taxonomy_matches_the_trainer() {
        // The order is a contract with `QUALITIES` in model_trainer.py - these
        // indices address quality_logits directly. Reordering silently corrupts
        // every chord name.
        assert_eq!(QUALITIES.len(), 11);
        assert_eq!(QUALITIES[0], "maj");
        assert_eq!(QUALITIES[3], "dom7");
        assert_eq!(QUALITIES[10], "N");
        assert_eq!(ROOTS.len(), 13);
        assert_eq!(ROOTS[12], "Noise");
    }

    /// Mirrors the parsing in `state.rs` so the test guards the CONTRACT between
    /// the two files rather than the formatting. A mismatch here means a chord
    /// that never matches, with no error message anywhere.
    fn parse_like_state(pred: &str) -> (String, String) {
        let parts: Vec<&str> = pred.split_whitespace().collect();
        if parts.is_empty() {
            return (String::new(), String::new());
        }
        if parts[0] == "Note" && parts.len() > 1 {
            return (parts[1].to_string(), "Note".to_string());
        }
        let qual = if parts.len() > 1 { parts[1] } else { "" };
        (parts[0].to_string(), qual.to_string())
    }

    fn render(root: &str, qual: &str) -> String {
        if qual == "note" {
            format!("Note {}", root)
        } else {
            format!("{} {}", root, quality_suffix(qual)).trim().to_string()
        }
    }

    #[test]
    fn quality_names_match_chordquality() {
        // ChordQuality::to_string() in model.rs yields exactly these four; they
        // are the only qualities the app asks for in Chords mode.
        assert_eq!(quality_suffix("maj7"), "Maj7");
        assert_eq!(quality_suffix("min7"), "m7");
        assert_eq!(quality_suffix("dom7"), "7");
        assert_eq!(quality_suffix("m7b5"), "m7b5");
    }

    #[test]
    fn state_can_parse_what_we_emit() {
        for (root, q, exp_root, exp_qual) in [
            ("C", "maj7", "C", "Maj7"),
            ("A", "min7", "A", "m7"),
            ("G", "dom7", "G", "7"),
            ("B", "m7b5", "B", "m7b5"),
            ("D", "maj", "D", ""),      // major: no second part
            ("E", "min", "E", "m"),
            ("F#", "note", "F#", "Note"),
        ] {
            let s = render(root, q);
            let (got_root, got_qual) = parse_like_state(&s);
            assert_eq!(got_root, exp_root, "root from {s:?}");
            assert_eq!(got_qual, exp_qual, "quality from {s:?}");
        }
    }

    #[test]
    fn major_leaves_no_trailing_space() {
        assert_eq!(render("C", "maj"), "C");
        assert!(!render("C", "maj").ends_with(' '));
    }

    #[test]
    fn argmax_softmax_picks_the_largest() {
        let (i, p) = argmax_softmax(&[0.0, 5.0, 1.0]);
        assert_eq!(i, 1);
        assert!(p > 0.9, "confidence {p} should be high with a clear winner");

        let (i2, p2) = argmax_softmax(&[2.0, 2.0, 2.0]);
        assert_eq!(i2, 0);
        assert!((p2 - 1.0 / 3.0).abs() < 1e-5, "a three-way tie should give 1/3, got {p2}");
    }
}
