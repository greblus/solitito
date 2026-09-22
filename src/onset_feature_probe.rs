//! Diagnostic export through the actual CqtAnalyzer, compiled only for tests.
//! No inference, replacement DSP implementation, or change to the live path.

use super::*;

#[test]
#[ignore = "diagnostic: requires SOLITITO_FEATURE_WAV and SOLITITO_FEATURE_OUTPUT"]
fn export_causal_features() -> Result<()> {
    let path = std::env::var("SOLITITO_FEATURE_WAV")?;
    let output = std::env::var("SOLITITO_FEATURE_OUTPUT")?;
    let boost: f32 = std::env::var("SOLITITO_FEATURE_BOOST")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    anyhow::ensure!(boost.is_finite() && boost >= 0.0, "invalid boost");
    let mut reader = hound::WavReader::open(&path)?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.channels == 1 && spec.sample_rate > 0,
        "expected mono WAV"
    );
    let mono: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int if spec.bits_per_sample == 16 => reader
            .samples::<i16>()
            .map(|sample| sample.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
        _ => anyhow::bail!("expected float32 or int16 WAV"),
    };
    anyhow::ensure!(mono.iter().all(|v| v.is_finite()), "nonfinite audio");
    let signal = crate::probe_audio::resample(&mono, spec.sample_rate, TARGET_SR);
    let mut analyzer = CqtAnalyzer::new("dsp_weights.json")?;
    let mut frames = Vec::new();
    for end in (FFT_SIZE..=signal.len()).step_by(HOP_LENGTH) {
        let samples = &signal[end - FFT_SIZE..end];
        let rms = (samples.iter().map(|x| x * x).sum::<f32>() / FFT_SIZE as f32).sqrt();
        let amplified: Vec<f32> = samples.iter().map(|v| v * INPUT_GAIN).collect();
        let (cqt, chroma, bass, _) = analyzer.compute_cqt_chroma(&amplified, boost > 0.0, boost);
        // Reuse the analyzer's actual FFT and sparse multiply before normalization.
        let mut magnitude = sparse_cqt_mag(
            &analyzer.fft_buffer,
            &analyzer.cqt_offsets,
            &analyzer.cqt_fft_idx,
            &analyzer.cqt_re,
            &analyzer.cqt_im,
        );
        if boost > 0.0 {
            for value in magnitude.iter_mut().take(BASS_BOOST_CUTOFF) {
                *value *= boost;
            }
        }
        let features: Vec<f32> = cqt.into_iter().chain(chroma).chain(bass).collect();
        frames.push(serde_json::json!({
            "end_sample": end,
            "rms": rms,
            "features": features,
            "magnitude": magnitude,
        }));
    }
    anyhow::ensure!(!frames.is_empty(), "WAV shorter than the FFT window");
    let document = serde_json::json!({
        "schema_version": 1,
        "wav": path,
        "source_rate": spec.sample_rate,
        "target_rate": TARGET_SR,
        "fft_samples": FFT_SIZE,
        "hop_samples": HOP_LENGTH,
        "input_gain": INPUT_GAIN,
        "bass_boost": boost,
        "frames": frames,
    });
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    use std::io::Write;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer(&mut writer, &document)?;
    writer.flush()?;
    Ok(())
}
