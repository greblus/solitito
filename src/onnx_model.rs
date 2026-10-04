//! Load take7's independent branches from one ONNX, without running the other
//! branch or writing derived files. Selecting outputs alone does not prune ORT
//! execution. This small wire reader preserves tensor/operator bytes verbatim;
//! it supports the flat, self-contained graphs emitted by our exporter only.
//! Field numbers: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
use anyhow::{bail, ensure, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::collections::HashSet;

#[derive(Clone, Copy)]
pub enum Branch {
    Chords,
    Onset,
}

pub fn is_combined(path: &str) -> Result<bool> {
    let bytes = std::fs::read(path).with_context(|| format!("Cannot read model {path}"))?;
    combined(&fields(&bytes)?)
}

pub fn load(path: &str, branch: Branch) -> Result<(Session, bool)> {
    let bytes = std::fs::read(path).with_context(|| format!("Cannot read model {path}"))?;
    let model = fields(&bytes)?;
    let combined = combined(&model)?;
    let selected = if combined {
        Some(select_branch(&model, branch)?)
    } else {
        None
    };
    let builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?;
    let session = match selected {
        Some(selected) => builder.commit_from_memory(&selected)?,
        None => builder.commit_from_file(path)?,
    };
    if combined {
        validate_metadata(&session)?;
    }
    Ok((session, combined))
}

fn combined(model: &[Field<'_>]) -> Result<bool> {
    let graph = fields(single(model, 7)?)?;
    let names: Vec<_> = graph
        .iter()
        .filter(|f| f.number == 11)
        .map(|f| name(f.data, 1))
        .collect::<Result<_>>()?;
    if names.contains(&"features") && names.contains(&"short_features") {
        ensure!(names.len() == 2, "Unexpected take7 inputs");
        Ok(true)
    } else {
        ensure!(
            names.len() == 1,
            "Expected one legacy input or two take7 inputs"
        );
        Ok(false)
    }
}

pub fn onset_threshold(session: &Session, combined: bool) -> Result<f32> {
    let value = session.metadata()?.custom("onset_threshold")?;
    match value {
        Some(value) => parse_threshold(&value),
        None if !combined => Ok(0.8),
        None => bail!("Take7 is missing onset_threshold; re-export with the current trainer"),
    }
}

fn parse_threshold(value: &str) -> Result<f32> {
    let threshold: f32 = value.parse().context("Invalid onset_threshold")?;
    ensure!(
        threshold.is_finite() && threshold > 0.0 && threshold < 1.0,
        "onset_threshold must be between 0 and 1"
    );
    Ok(threshold)
}

fn validate_metadata(session: &Session) -> Result<()> {
    let metadata = session.metadata()?;
    ensure!(
        metadata.custom("model_kind")?.as_deref() == Some("solitito-chord-rise-v1"),
        "Unsupported combined model contract"
    );
    ensure!(
        metadata.custom("onset_history_frames")?.as_deref() == Some("34"),
        "Unsupported Rise history (expected 34 past frames)"
    );
    let spec: serde_json::Value = serde_json::from_str(
        &metadata
            .custom("onset_feature_spec")?
            .context("Missing onset_feature_spec")?,
    )?;
    let expected = serde_json::json!({
        "version": "short-stft-v1", "samplerate": 16000, "hop": 256,
        "windows": [1024, 2048], "max_frequency": 4000,
        "window": "symmetric Hann", "amplitude": "2*abs(rfft)/sum(window)",
        "compression": "log1p(1000*amplitude)/log(1001); no file normalization",
        "resampling": "causal linear, one source-sample delay unless already 16kHz",
        "frame_time": "exclusive audio window end; first frame=0.016 seconds",
        "startup": "zero left audio padding; no right padding or future samples",
        "storage": "float16, converted back to float32 for both training and inference"
    });
    ensure!(
        spec == expected,
        "Unsupported Rise feature specification; update the app"
    );
    onset_threshold(session, true)?;
    Ok(())
}

fn select_branch(model: &[Field<'_>], branch: Branch) -> Result<Vec<u8>> {
    let graph = fields(single(model, 7)?)?;
    let (input, outputs): (&str, &[&str]) = match branch {
        Branch::Chords => (
            "features",
            &["root_logits", "quality_logits", "pitch_logits"],
        ),
        Branch::Onset => ("short_features", &["onset_logits"]),
    };
    let actual: HashSet<_> = graph
        .iter()
        .filter(|f| f.number == 12)
        .map(|f| name(f.data, 1))
        .collect::<Result<_>>()?;
    ensure!(
        actual
            == HashSet::from([
                "root_logits",
                "quality_logits",
                "pitch_logits",
                "onset_logits"
            ]),
        "Unexpected take7 outputs"
    );
    ensure!(
        !graph.iter().any(|f| matches!(f.number, 14 | 15)),
        "Sparse or quantized take7 graphs are not supported"
    );
    let mut needed: HashSet<&str> = outputs.iter().copied().collect();
    let mut keep_nodes = HashSet::new();
    // ONNX nodes are topologically sorted. Walk dependencies back from outputs.
    for (index, field) in graph
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, f)| f.number == 1)
    {
        let node = fields(field.data)?;
        let node_outputs: Vec<_> = node
            .iter()
            .filter(|f| f.number == 2)
            .map(|f| text(f.data))
            .collect::<Result<_>>()?;
        if !node_outputs.iter().any(|n| needed.contains(n)) {
            continue;
        }
        for attr in node.iter().filter(|f| f.number == 5) {
            ensure!(
                !fields(attr.data)?
                    .iter()
                    .any(|f| matches!(f.number, 6 | 11)),
                "Take7 control-flow subgraphs are not supported"
            );
        }
        keep_nodes.insert(index);
        needed.extend(node_outputs);
        for field in node.iter().filter(|f| f.number == 1) {
            let name = text(field.data)?;
            if !name.is_empty() {
                needed.insert(name);
            }
        }
    }
    ensure!(!keep_nodes.is_empty(), "Empty take7 branch");
    let mut selected = Vec::new();
    let mut kept_inputs = Vec::new();
    for (index, field) in graph.iter().enumerate() {
        let keep = match field.number {
            1 => keep_nodes.contains(&index),
            5 => {
                // External tensor data cannot be resolved from an in-memory model.
                ensure!(
                    !fields(field.data)?.iter().any(|f| f.number == 13),
                    "Take7 must contain its weights, not external data"
                );
                needed.contains(name(field.data, 8)?)
            }
            11 => {
                let name = name(field.data, 1)?;
                if needed.contains(name) {
                    kept_inputs.push(name);
                    true
                } else {
                    false
                }
            }
            12 => outputs.contains(&name(field.data, 1)?),
            13 => needed.contains(name(field.data, 1)?),
            _ => true,
        };
        if keep {
            selected.extend_from_slice(field.raw);
        }
    }
    ensure!(
        kept_inputs == [input],
        "Take7 branches must have independent inputs"
    );
    let mut result = Vec::new();
    for field in model {
        if field.number == 7 {
            put_bytes(&mut result, 7, &selected);
        } else {
            result.extend_from_slice(field.raw);
        }
    }
    Ok(result)
}

struct Field<'a> {
    number: u64,
    raw: &'a [u8],
    data: &'a [u8],
}

fn varint(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let byte = *bytes.get(*offset).context("Truncated ONNX varint")?;
        *offset += 1;
        ensure!(shift < 63 || byte <= 1, "Overflowing ONNX varint");
        value |= u64::from(byte & 127) << shift;
        if byte < 128 {
            return Ok(value);
        }
    }
    bail!("Invalid ONNX varint")
}

fn fields(bytes: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let start = offset;
        let key = varint(bytes, &mut offset)?;
        ensure!(key >> 3 > 0 && key >> 3 < (1 << 29), "Invalid ONNX field");
        let size = match key & 7 {
            0 => {
                let begin = offset;
                varint(bytes, &mut offset)?;
                offset - begin
            }
            1 => 8,
            2 => usize::try_from(varint(bytes, &mut offset)?)?,
            5 => 4,
            _ => bail!("Unsupported ONNX wire type"),
        };
        let begin = if key & 7 == 0 { offset - size } else { offset };
        let end = begin.checked_add(size).context("ONNX field overflow")?;
        let data = bytes.get(begin..end).context("Truncated ONNX field")?;
        offset = end;
        result.push(Field {
            number: key >> 3,
            raw: &bytes[start..end],
            data,
        });
    }
    Ok(result)
}

fn single<'a>(fields: &[Field<'a>], number: u64) -> Result<&'a [u8]> {
    let mut matches = fields.iter().filter(|f| f.number == number);
    let value = matches.next().context("Missing ONNX field")?;
    ensure!(matches.next().is_none(), "Duplicate ONNX field");
    Ok(value.data)
}
fn text(bytes: &[u8]) -> Result<&str> {
    Ok(std::str::from_utf8(bytes)?)
}
fn name(bytes: &[u8], field: u64) -> Result<&str> {
    text(single(&fields(bytes)?, field)?)
}
fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 128 {
        out.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    out.push(value as u8);
}
fn put_bytes(out: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    put_varint(out, (field << 3) | 2);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model(onset_input: &str, nested: bool) -> Vec<u8> {
        let mut graph = Vec::new();
        for input in ["features", "short_features"] {
            let mut info = Vec::new();
            put_bytes(&mut info, 1, input.as_bytes());
            put_bytes(&mut graph, 11, &info);
        }
        for output in [
            "root_logits",
            "quality_logits",
            "pitch_logits",
            "onset_logits",
        ] {
            let mut node = Vec::new();
            let input = if output == "onset_logits" {
                onset_input
            } else {
                "features"
            };
            put_bytes(&mut node, 1, input.as_bytes());
            put_bytes(&mut node, 2, output.as_bytes());
            if nested && output == "onset_logits" {
                let mut attr = Vec::new();
                put_bytes(&mut attr, 6, &[]);
                put_bytes(&mut node, 5, &attr);
            }
            put_bytes(&mut graph, 1, &node);
            let mut info = Vec::new();
            put_bytes(&mut info, 1, output.as_bytes());
            put_bytes(&mut graph, 12, &info);
        }
        let mut model = Vec::new();
        put_bytes(&mut model, 7, &graph);
        model
    }

    #[test]
    fn extraction_keeps_only_the_selected_branch_and_rejects_shared_inputs() {
        let model = tiny_model("short_features", false);
        let selected = select_branch(&fields(&model).unwrap(), Branch::Onset).unwrap();
        let model_fields = fields(&selected).unwrap();
        let graph = fields(single(&model_fields, 7).unwrap()).unwrap();
        assert_eq!(graph.iter().filter(|f| f.number == 1).count(), 1);
        assert_eq!(graph.iter().filter(|f| f.number == 11).count(), 1);
        assert_eq!(graph.iter().filter(|f| f.number == 12).count(), 1);
        assert_eq!(
            name(single(&graph, 11).unwrap(), 1).unwrap(),
            "short_features"
        );
        for model in [
            tiny_model("features", false),
            tiny_model("short_features", true),
        ] {
            assert!(select_branch(&fields(&model).unwrap(), Branch::Onset).is_err());
        }
    }
    #[test]
    fn rejects_invalid_thresholds() {
        for value in ["NaN", "inf", "0", "1", "-0.1", "abc"] {
            assert!(parse_threshold(value).is_err(), "{value}");
        }
        assert_eq!(parse_threshold("0.7").unwrap(), 0.7);
    }
    #[test]
    fn rejects_truncated_and_overflowing_wire_data() {
        for bytes in [&[0][..], &[10, 5, 1], &[8, 128], &[15], &[255; 11]] {
            assert!(fields(bytes).is_err(), "{bytes:?}");
        }
        let mut bytes = Vec::new();
        put_bytes(&mut bytes, 7, b"abc");
        assert_eq!(single(&fields(&bytes).unwrap(), 7).unwrap(), b"abc");
    }
}

#[cfg(test)]
#[path = "take7_runtime_tests.rs"]
mod runtime_tests;
