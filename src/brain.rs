// src/brain.rs
use std::sync::Mutex; // <--- WAŻNE: Import Mutex
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use anyhow::Result;

// Musi być w tej samej kolejności co w Pythonie!
pub const CHORD_NAMES: [&str; 24] = [
    "C Maj", "C# Maj", "D Maj", "D# Maj", "E Maj", "F Maj", 
    "F# Maj", "G Maj", "G# Maj", "A Maj", "A# Maj", "B Maj",
    "C Min", "C# Min", "D Min", "D# Min", "E Min", "F Min", 
    "F# Min", "G Min", "G# Min", "A Min", "A# Min", "B Min"
];

pub struct ChordBrain {
    // ZMIANA: Owijamy Session w Mutex.
    // To pozwala na modyfikację sesji (uruchomienie) nawet gdy mamy tylko &self.
    session: Mutex<Session>,
}

impl ChordBrain {
    pub fn new() -> Result<Self> {
        // Inicjalizacja ONNX Runtime
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file("chord_model.onnx")?;

        Ok(Self { 
            session: Mutex::new(session) // Pakujemy do Mutexa
        })
    }

    // Funkcja nadal przyjmuje &self (nie trzeba zmieniać main.rs!)
    pub fn predict(&self, chroma: &[f32; 12]) -> Result<(String, f32)> {
        
        // 1. Normalizacja
        let max_val = chroma.iter().fold(0.0f32, |a, &b| a.max(b));
        let mut normalized_chroma = [0.0f32; 12];
        
        if max_val > 0.0 {
            for i in 0..12 {
                normalized_chroma[i] = chroma[i] / max_val;
            }
        }

        // 2. Tworzenie Tensora
        let input_tensor = Tensor::from_array(([1, 12], normalized_chroma.to_vec()))?;

        // 3. Uruchamiamy Mózg
        // ZMIANA: Blokujemy Mutex, aby uzyskać dostęp mutowalny do sesji.
        // .lock().unwrap() jest bezpieczne, bo wątek audio i GUI rzadko kolidują w tym samym mikrosekundowym momencie.
        let mut session_guard = self.session.lock().unwrap();
        
        // Teraz możemy wywołać .run(), bo session_guard zachowuje się jak &mut Session
        let outputs = session_guard.run(ort::inputs!["input" => input_tensor])?;

        // 4. Odbieramy wyniki
        let (_, data) = outputs["output"].try_extract_tensor::<f32>()?;
        
        // 5. Szukamy zwycięzcy
        let mut max_score = -f32::INFINITY;
        let mut max_idx = 0;

        for (i, &score) in data.iter().enumerate() {
            if score > max_score {
                max_score = score;
                max_idx = i;
            }
        }

        let chord_name = CHORD_NAMES.get(max_idx).unwrap_or(&"???").to_string();
        
        Ok((chord_name, max_score))
    }
}
