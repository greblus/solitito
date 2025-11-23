// src/brain.rs
use std::sync::Mutex;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use anyhow::Result;

pub struct ChordBrain {
    session: Mutex<Session>,
    labels: Vec<String>,
}

impl ChordBrain {
    pub fn new() -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file("chord_model.onnx")?;

        // ZGODNOŚĆ Z PYTHON V7 (Sorted Labels)
        let roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        let types = ["", "m", "Maj7", "m7", "7", "m7b5", "dim7"];
        
        let mut labels = Vec::new();
        for root in &roots {
            for t in &types {
                if t.is_empty() { labels.push(root.to_string()); } 
                else { labels.push(format!("{} {}", root, t)); }
            }
        }
        for root in &roots { labels.push(format!("Note {}", root)); }
        labels.push("Noise".to_string());
        
        labels.sort(); 

        Ok(Self { 
            session: Mutex::new(session),
            labels 
        })
    }

    pub fn predict(&self, spectrum: &[f32; 48]) -> Result<(String, f32)> {
        let mut processed = [0.0f32; 48];
        for i in 0..48 {
            processed[i] = (1.0 + spectrum[i] * 10.0).ln();
        }

        let max_val = processed.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_val < 0.1 { return Ok(("Noise".to_string(), 1.0)); }

        if max_val > 0.0 {
            for i in 0..48 { processed[i] /= max_val; }
        }

        let input_tensor = Tensor::from_array(([1, 48], processed.to_vec()))?;
        let mut session_guard = self.session.lock().unwrap();
        let outputs = session_guard.run(ort::inputs!["input" => input_tensor])?;
        let (_, data) = outputs["output"].try_extract_tensor::<f32>()?;
        
        let mut max_score = -f32::INFINITY;
        let mut max_idx = 0;

        for (i, &score) in data.iter().enumerate() {
            if score > max_score {
                max_score = score;
                max_idx = i;
            }
        }

        let label = self.labels.get(max_idx).cloned().unwrap_or_else(|| "?".to_string());
        Ok((label, max_score))
    }
}
