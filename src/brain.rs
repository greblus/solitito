// src/brain.rs
use std::sync::Mutex;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use anyhow::Result;
use crate::audio::{LOG_BINS, CTX_FRAMES}; 

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

        // ETYKIETY ZGODNE Z TRENEREM V9.0 (Kaggle)
        // 1. Akordy (Roots * Quals)
        // 2. Noise
        // 3. Nuty (Note + Root)
        
        let roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        let types = ["", "m", "Maj7", "m7", "7", "dim7", "m7b5"]; 
        
        let mut labels = Vec::new();
        
        // Akordy
        for r in &roots { 
            for t in &types { 
                if t.is_empty() { labels.push(r.to_string()); }
                else { labels.push(format!("{} {}", r, t)); }
            }
        }
        
        // Noise
        labels.push("Noise".to_string());
        
        // Nuty
        for r in &roots { labels.push(format!("Note {}", r)); }
        
        // println!("Classes loaded: {}", labels.len());
        
        Ok(Self { session: Mutex::new(session), labels })
    }

    pub fn predict(&self, flat_input: &[f32], expected: Option<&str>) -> Result<(String, f32)> {
        // Sprawdzenie rozmiaru
        if flat_input.len() != LOG_BINS * CTX_FRAMES {
            return Ok(("Buffering...".into(), 0.0));
        }
        
        // Tensor Shape: [Batch=1, Time=16, Freq=216]
        // Odpowiada wejściu: dummy = torch.randn(1, CTX_FRAMES, LOG_BINS)
        let tensor = Tensor::from_array(([1, CTX_FRAMES, LOG_BINS], flat_input.to_vec()))?;
        
        let mut sess = self.session.lock().unwrap();
        let out = sess.run(ort::inputs!["input" => tensor])?;
        let (_, data) = out["output"].try_extract_tensor::<f32>()?;
        
        let mut best_score = -f32::INFINITY;
        let mut best_idx = 0;
        
        for (i, &val) in data.iter().enumerate() {
            let mut score = val;
            
            // Contextual Biasing (opcjonalnie)
            if let Some(tgt) = expected {
                if let Some(lbl) = self.labels.get(i) {
                     if lbl == tgt { score += 2.0; }
                }
            }
            
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        
        let lbl = self.labels.get(best_idx).cloned().unwrap_or("?".into());
        Ok((lbl, best_score))
    }
}
