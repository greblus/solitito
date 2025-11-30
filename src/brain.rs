// src/brain.rs

use anyhow::Result;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value; 
use std::collections::VecDeque;
use crate::audio::LOG_BINS;

// STAŁE IDENTYCZNE JAK W PYTHONIE
const CTX_FRAMES: usize = 16; 

pub struct ChordBrain {
    session: Session,
    labels: Vec<String>,
    input_history: VecDeque<Vec<f32>>, 
}

impl ChordBrain {
    pub fn new(model_path: &str) -> Result<Self> {
        println!("--- BRAIN INIT START ---");
        println!("Loading ONNX model from: {}", model_path);

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        println!("Model loaded successfully.");
        println!("Inspecting Model Inputs:");
        
        for (i, input) in session.inputs.iter().enumerate() {
            println!("  Input [{}]: Name = '{}'", i, input.name);
            println!("    Type info: {:?}", input.input_type);
        }
        
        // Generowanie etykiet
        let roots = vec!["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        let quals = vec!["", "m", "Maj7", "m7", "7", "dim7", "m7b5"];
        
        let mut labels = Vec::new();
        for r in &roots {
            for q in &quals {
                labels.push(format!("{} {}", r, q).trim().to_string());
            }
        }
        labels.push("Noise".to_string());
        for r in &roots {
            labels.push(format!("Note {}", r));
        }

        println!("Classes generated: {}", labels.len());
        println!("--- BRAIN INIT DONE ---");

        Ok(Self {
            session,
            labels,
            input_history: VecDeque::with_capacity(CTX_FRAMES),
        })
    }

    pub fn predict(&mut self, current_frame: &[f32; LOG_BINS]) -> Result<(String, f32)> {
        // 1. Zarządzanie buforem historii
        if self.input_history.len() >= CTX_FRAMES {
            self.input_history.pop_front();
        }
        self.input_history.push_back(current_frame.to_vec());

        // 2. Jeśli bufor niepełny, czekamy
        if self.input_history.len() < CTX_FRAMES {
            use std::io::{self, Write};
            print!("."); 
            io::stdout().flush().unwrap();
            return Ok(("Buffering...".to_string(), 0.0));
        }

        // 3. Budowanie Płaskiego Wektora
        let expected_len = CTX_FRAMES * LOG_BINS;
        let mut flat_data = Vec::with_capacity(expected_len);

        for frame in &self.input_history { 
            flat_data.extend_from_slice(frame); 
        }

        // SAFETY CHECK
        if flat_data.len() != expected_len {
            return Ok(("Error".to_string(), 0.0));
        }

        // --- POPRAWKA TUTAJ ---
        // Twój model chce [-1, 16, 216], czyli 3 wymiary: [Batch, Time, Freq].
        // Wcześniej wysyłaliśmy [1, 1, 16, 216], co powodowało błąd.
        
        let shape = vec![1, CTX_FRAMES as i64, LOG_BINS as i64]; // [1, 16, 216]
        
        // Tworzenie Tensora ORT
        let input_tensor = Value::from_array((shape, flat_data))?;
        
        // 5. Uruchomienie Modelu
        let outputs = self.session.run(ort::inputs!["input" => input_tensor])?;
        
        // 6. Analiza Wyników
        let (_, probabilities) = outputs["output"].try_extract_tensor::<f32>()?;

        // Sortowanie wyników
        let mut scored_results: Vec<(usize, f32)> = Vec::with_capacity(probabilities.len());
        for (i, &val) in probabilities.iter().enumerate() {
            scored_results.push((i, val));
        }
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // WYPISZ DEBUG NA KONSOLĘ
        use std::io::{self, Write};
        print!("\rAI SEES: ");
        for i in 0..3 {
            if i < scored_results.len() {
                let (idx, score) = scored_results[i];
                let name = if idx < self.labels.len() { &self.labels[idx] } else { "???" };
                print!("[{}: {:.2}] ", name, score);
            }
        }
        io::stdout().flush().unwrap();

        // Najlepszy wynik
        let (max_idx, max_score) = scored_results[0];
        let label = if max_idx < self.labels.len() {
            self.labels[max_idx].clone()
        } else {
            "Unknown".to_string()
        };

        Ok((label, max_score))
    }
}
