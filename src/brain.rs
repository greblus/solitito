use anyhow::{Result, anyhow};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value; 
use std::collections::VecDeque;

// --- KONFIGURACJA V26 ---
// 144 CQT + 12 Chroma = 156 cech
const FEATURE_SIZE: usize = 156; 
const CTX_FRAMES: usize = 32; 

pub struct ChordBrain {
    session: Session,
    input_history: VecDeque<Vec<f32>>, 
    
    // Etykiety
    roots: Vec<String>,
    quals: Vec<String>,
}

impl ChordBrain {
    pub fn new(model_path: &str) -> Result<Self> {
        println!("Loading Brain: {}", model_path);

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        // Definicje klas (Musi pasować do train.py)
        let roots = vec!["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Noise"]
            .iter().map(|s| s.to_string()).collect();
            
        let quals = vec!["", "m", "7", "Maj7", "m7", "dim7", "m7b5", "9", "13", "Note"]
            .iter().map(|s| s.to_string()).collect();

        Ok(Self {
            session,
            roots,
            quals,
            input_history: VecDeque::with_capacity(CTX_FRAMES),
        })
    }

    // Przyjmuje jedną klatkę (156 floatów)
    pub fn predict(&mut self, current_frame: &[f32]) -> Result<(String, f32)> {
        // Zabezpieczenie przed złym rozmiarem danych
        if current_frame.len() != FEATURE_SIZE {
            return Err(anyhow!("Brain input size mismatch! Expected {}, got {}", FEATURE_SIZE, current_frame.len()));
        }

        // 1. Aktualizacja historii (Rolling Buffer)
        if self.input_history.len() >= CTX_FRAMES {
            self.input_history.pop_front();
        }
        self.input_history.push_back(current_frame.to_vec());

        if self.input_history.len() < CTX_FRAMES {
            return Ok(("Buffering...".to_string(), 0.0));
        }

        // 2. Flatten [32, 156] -> [4992]
        let mut flat_data = Vec::with_capacity(CTX_FRAMES * FEATURE_SIZE);
        for frame in &self.input_history { 
            flat_data.extend_from_slice(frame); 
        }

        // 3. Tensor [1, 32, 156]
        let shape = vec![1, CTX_FRAMES as i64, FEATURE_SIZE as i64];
        let input_tensor = Value::from_array((shape, flat_data))?;
        
        // 4. Run Inference
        let outputs = self.session.run(ort::inputs!["in" => input_tensor])?;
        
        let (_, root_data) = outputs["out_root"].try_extract_tensor::<f32>()?;
        let (_, qual_data) = outputs["out_qual"].try_extract_tensor::<f32>()?;
        
        // 5. Argmax
        let (root_idx, root_conf) = argmax_softmax(root_data);
        let (qual_idx, qual_conf) = argmax_softmax(qual_data);
        
        let final_conf = root_conf * qual_conf;
        
        // 6. Dekodowanie
        if root_idx >= self.roots.len() || qual_idx >= self.quals.len() {
            return Ok(("Unknown".to_string(), 0.0));
        }

        let r_str = &self.roots[root_idx];
        let q_str = &self.quals[qual_idx];
        
        let display = if r_str == "Noise" {
            "Noise".to_string()
        } else if q_str == "Note" {
            format!("Note {}", r_str)
        } else {
            format!("{} {}", r_str, q_str).trim().to_string()
        };

        Ok((display, final_conf))
    }
}

fn argmax_softmax(logits: &[f32]) -> (usize, f32) {
    let mut max_val = -f32::INFINITY;
    let mut max_idx = 0;
    
    // Max trick for numerical stability
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum_exp = 0.0;

    for (i, &val) in logits.iter().enumerate() {
        let exp_val = (val - max_logit).exp();
        sum_exp += exp_val;
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    
    if sum_exp == 0.0 { return (0, 0.0); }
    let confidence = (max_val - max_logit).exp() / sum_exp;
    (max_idx, confidence)
}
