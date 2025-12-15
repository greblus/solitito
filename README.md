# Solitito – Real-Time Polyphonic Guitar Trainer

**Solitito** (no pun intended) is an experimental, real-time, polyphonic guitar trainer built in **Rust** during a 5-hour vibe-coding session with *Gemini 3 Pro Preview*. It's small neural model detects notes and chords.

It's a proof-of-concept **experiment** - heavily inspired by another, amazing Android/iOS app - Solo. I just want to experiment a little bit with FFT and some neural networks architectures for chords detection, so as many of my experiments, this little project might just be left as is at some point. I don't intent to create alternative to Solo ;) which I use daily.  
<div align="center">
<img width="284" height="500" alt="solitito0" src="https://github.com/user-attachments/assets/553e858f-9e85-420c-a26b-f05d504eb5b9" />
</div>
  
# How the model was trained?

In **dist** directory I included all scripts which were used for dataset preparation, model training, testing and pseudo-CQT weights generation to be able to use it in Rust.  

- **dataset_dual_generator.py** generates dataset_dual.gp - dual because it has 
two tracks: first with triads, jazz shapes and single notes, second with binary representation 
of chord index in dataset_reference beeped with a synth saw sound.
- **decode_annotations.py** takes beeps_render.wav file, exported from 2nd track of dataset_dual.gp 
and dataset_reference.csv with proper names of sound samples and creates dataset_annotations.csv.
- The first track from gp file is exported as "DI" guitar signal raw_render.wav which is then rendered in DAW through NAM plugin into dataset_clean.wav (Fender Deluxe Reverb clean sound) and dataset_eob.wav (Fender Deluxe Reverb edge of breakup tone). 
- **model_trainer.py** takes dataset_clean.wav, dataset_eob.wav and dataset_annotations.csv, 
adds GuitarSet[1] with it's own annotations and splits the datasets accordingly for training.

**A short description of the current model architecture:**  
- Hybrid  CNN with Squeeze-and-Excitation (SE) blocks.    
- Transformer Encoder for temporal context.   
- Log-scale Constant-Q Transform (CQT) and Chroma features for precise harmonic analysis.  
- Jazz-Optimized: Multi-Head output (Root/Quality) trained with Focal Loss to master complex extended chords (9, 13, m7b5).  

Currently Focal Loss results with higher accuracy for jazzy chords, let's leave it like that for now and treat it like an achievement ;) 
<br><br>
<img width="400" height="390" alt="confusion_matrix" src="https://github.com/user-attachments/assets/fc17d099-1e7e-443d-869a-e693a18a492d" />
<br><br>

Basic Benchmarks (full benchmark in model_benchmark.txt):  
🏆 GLOBAL ACCURACY:      98.95%
🔹 BASIC:       98.56%
🎷 JAZZ:        99.75%
🎵 NOTES:       100.00%
 
## ⭐ Key Features

### 🎼 Modes
- **Songs** — chord progressions 
- **Scales** — sequential practice  
- **Random** — ear training & fretboard awareness

### 🎧 DSP / Audio
- **Polyphonic chord detection**  
- **Stale-Note Filtering** — prevents sustaining notes from triggering new chords  
- **Optimized for laptop microphones** (Bass Boost, Sensitivity)

### 📁 Custom Content
- Load your own **songs** and **scales** from simple text files  
- No restart required

---

## ⚙️ Settings (Gear Icon)
<div align="center">
<img width="284" height="500" alt="solitito1" src="https://github.com/user-attachments/assets/3946a12d-f5a5-4c88-9f34-f5ea8b6b5db7" />
</div>
<br> 

| Setting        | Description |
|----------------|-------------|
| **Threshold**      | Minimum volume required to detect a note |
| **Tail Release**   | How much a string must decay before it can be triggered again |
| **Input Delay**    | Grace period after a chord change (prevents noise while moving fingers) |
| **Bass Boost**     | Digital amplification for low strings (useful for laptop mics) |
| **Intervals**      | What intervals to practice (e.g. `1 3 5` for triads, `1 3 5 7` for sevenths, 3 or 5 shows both 3 and b3, or 5 and b5 according to the chord quality) |

---

## 📄 Custom Files Format

`user_songs.txt`  
My Song Title  
Cm7 F7 BbMaj7

`user_scales_def.txt`  
My Scale Name  
1 b2 3 4 5 b6 7

---
[1] This project uses the GuitarSet dataset by Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye, & Juan P. Bello
available at https://guitarset.weebly.com/ licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

