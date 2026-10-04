//! Explicit integration probe: local model files and AtoA are not in Git.
use super::*;
use crate::{
    brain::ChordBrain,
    rise::{read_wav, Detector, Resampler},
};
use std::{collections::VecDeque, time::Instant};

#[test]
#[ignore = "requires combined-check.onnx, source models and SOLITITO_RISE_WAV"]
fn combined_model_matches_original_branches() -> Result<()> {
    let path = "dist/crediting_measurements/take7-single-export-20260929/combined-check.onnx";
    let directory = std::env::var("SOLITITO_TAKE7_REPORT")
        .unwrap_or_else(|_| "/tmp/solitito-take7-runtime".into());
    std::fs::create_dir_all(&directory)?;
    let bytes = std::fs::read(path)?;
    let model = fields(&bytes)?;
    assert!(combined(&model)?);
    // The trainer chooses this value: test loading it, not just parsing a string.
    let onset = select_branch(&model, Branch::Onset)?;
    for (key, value, valid) in [
        ("onset_threshold", Some("0.7"), true),
        ("onset_threshold", Some("0.9"), true),
        ("onset_threshold", None, false),
        ("onset_threshold", Some("NaN"), false),
        ("onset_history_frames", Some("30"), false),
        ("onset_feature_spec", Some("{}"), false),
        ("model_kind", Some("unknown"), false),
    ] {
        let mut modified = Vec::new();
        for field in fields(&onset)? {
            if field.number == 14 && name(field.data, 1)? == key {
                if let Some(value) = value {
                    let mut entry = Vec::new();
                    put_bytes(&mut entry, 1, key.as_bytes());
                    put_bytes(&mut entry, 2, value.as_bytes());
                    put_bytes(&mut modified, 14, &entry);
                }
            } else {
                modified.extend_from_slice(field.raw);
            }
        }
        let session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_memory(&modified)?;
        assert_eq!(
            validate_metadata(&session).is_ok(),
            valid,
            "{key}={value:?}"
        );
        if valid {
            assert_eq!(
                onset_threshold(&session, true)?,
                value.context("Missing test value")?.parse::<f32>()?
            );
        }
    }
    let mut profile_counts = Vec::new();
    for (branch, label, forbidden) in [
        (Branch::Chords, "chords", "rise/"),
        (Branch::Onset, "onset", "chords/"),
    ] {
        let selected = select_branch(&model, branch)?;
        let mut session = Session::builder()?
            .with_intra_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_profiling(format!("{directory}/{label}"))?
            .commit_from_memory(&selected)?;
        match branch {
            Branch::Chords => {
                session.run(ort::inputs!["features" => ort::value::Value::from_array(([1usize,48,168], vec![0f32;48*168]))?])?;
            }
            Branch::Onset => {
                session.run(ort::inputs!["short_features" => ort::value::Value::from_array(([1usize,770,35], vec![0f32;770*35]))?])?;
            }
        }
        let profile: serde_json::Value =
            serde_json::from_slice(&std::fs::read(session.end_profiling()?)?)?;
        let nodes: Vec<_> = profile
            .as_array()
            .context("Invalid profile")?
            .iter()
            .filter(|v| v["cat"] == "Node")
            .collect();
        assert!(!nodes.is_empty());
        assert!(nodes
            .iter()
            .all(|v| !v["name"].as_str().unwrap_or("").starts_with(forbidden)));
        profile_counts.push(serde_json::json!({"branch":label, "nodes": nodes.len(), "bytes": selected.len(), "other_branch_nodes":0}));
    }
    let mut base = ChordBrain::new("best_model_v2_take6.onnx")?;
    let mut take7 = ChordBrain::new(path)?;
    for seed in 0..4 {
        let frames = std::array::from_fn(|t| {
            std::array::from_fn(|i| {
                if seed == 0 {
                    0.0
                } else {
                    (((t * 168 + i + seed * 71) as f32) * 0.037).sin().abs()
                }
            })
        });
        let a = base.predict(&frames)?;
        let b = take7.predict(&frames)?;
        assert_eq!(a.chord, b.chord);
        assert_eq!(a.root_idx, b.root_idx);
        assert_eq!(a.confidence, b.confidence);
        assert_eq!(a.pitches, b.pitches);
        assert_eq!(a.qual_probs, b.qual_probs);
        assert_eq!(b.onsets, [0.0; 12]); // Rise events are delivered by its own worker.
    }
    let mut base = Detector::new("short_onset_rise.onnx")?;
    let mut take7 = Detector::new(path)?;
    assert_eq!(take7.threshold(), 0.8);
    let (audio, rate) = read_wav(&std::env::var("SOLITITO_RISE_WAV")?)?;
    let mut resampler = Resampler::default();
    let mut pending = VecDeque::new();
    let mut base_ms = Vec::new();
    let mut take7_ms = Vec::new();
    let mut events = 0;
    let mut max_error = 0f32;
    for chunk in audio.chunks(511) {
        pending.extend(resampler.push(chunk, rate));
        while pending.len() >= 256 {
            let hop: Vec<_> = pending.drain(..256).collect();
            let start = Instant::now();
            let (a, ae, ar) = base.process_hop(&hop)?;
            base_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            let start = Instant::now();
            let (b, be, br) = take7.process_hop(&hop)?;
            take7_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(ae, be, "frame {}", base.frame);
            assert_eq!(ar, br);
            for (a, b) in a.iter().zip(b) {
                max_error = max_error.max((a - b).abs());
            }
            assert!(max_error <= 1e-6);
            events += ae.len();
        }
    }
    let stats = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        serde_json::json!({"p50_ms":values[values.len()/2], "p95_ms":values[values.len()*95/100]})
    };
    ensure!(!base_ms.is_empty(), "Empty input recording");
    let summary = serde_json::json!({"frames":base.frame,"events":events,
        "max_probability_error":max_error,"chord_windows_identical":4,
        "profiles":profile_counts,"standalone":stats(base_ms),"combined":stats(take7_ms),
        "rescue": std::env::var("SOLITITO_ONSET_RESCUE").as_deref() == Ok("1")});
    std::fs::write(
        format!("{directory}/summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{summary}");
    Ok(())
}
