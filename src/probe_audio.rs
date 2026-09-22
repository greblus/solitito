/// Linear resampling for a whole diagnostic WAV, with nonzero sample rates.
///
/// The live callback keeps a small fractional cursor by dropping consumed
/// input. A whole-file cursor must not accumulate f32 rounding for millions
/// of samples: it changes pitch and the time axis of the measurement.
pub fn resample(mono: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    let ratio = source_rate as f64 / target_rate as f64;
    let mut output = Vec::with_capacity((mono.len() as f64 / ratio) as usize);
    loop {
        let position = output.len() as f64 * ratio;
        if position + 1.0 >= mono.len() as f64 {
            break;
        }
        let index = position as usize;
        let fraction = (position - index as f64) as f32;
        output.push(mono[index] + fraction * (mono[index + 1] - mono[index]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::resample;

    #[test]
    fn long_recording_preserves_duration_and_late_attack_time() {
        for source_rate in [44_100, 48_000] {
            let mut mono = vec![0.0; source_rate * 83];
            mono[source_rate * 76..source_rate * 77].fill(1.0);
            let output = resample(&mono, source_rate as u32, 16_000);
            assert_eq!(output.len(), 83 * 16_000);
            let first = output.iter().position(|&v| v > 0.5).unwrap();
            assert_eq!(first, 76 * 16_000);
        }
    }

    #[test]
    fn old_whole_file_f32_cursor_distorts_the_time_axis() {
        let mut cursor = 0.0f32;
        let mut samples = 0;
        while cursor + 1.0 < 44_100.0 * 83.0 {
            cursor += 44_100.0f32 / 16_000.0;
            samples += 1;
        }
        let drift = samples as f64 / 16_000.0 - 83.0;
        assert!(drift > 0.1, "old resampler drift: {drift}s");
    }

    #[test]
    fn interpolation_and_short_inputs() {
        assert_eq!(resample(&[0.0, 1.0, 0.0], 1, 2), vec![0.0, 0.5, 1.0, 0.5]);
        assert!(resample(&[], 44_100, 16_000).is_empty());
        assert!(resample(&[1.0], 44_100, 16_000).is_empty());
    }
}
