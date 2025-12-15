# How the model was trained?

- **dataset_dual_generator.py** generates dataset_dual.gp - dual because it has 
two tracks: first with triads, jazz shapes and single notes, second with binary representation 
of chord index in dataset_reference beeped with a synth saw sound.
- **decode_annotations.py** takes beeps_render.wav file, exported from 2nd track of dataset_dual.gp 
and dataset_reference.csv with proper names of sound samples and creates dataset_annotations.csv.
- The first track from gp file is exported as "DI" guitar signal raw_render.wav which is then rendered in DAW through NAM plugin into dataset_clean.wav (Fender Deluxe Reverb clean sound) and dataset_eob.wav (Fender Deluxe Reverb edge of breakup tone). 
- **model_trainer.py** takes dataset_clean.wav, dataset_eob.wav and dataset_annotations.csv, 
adds GuitarSet[1] with it's own annotations and splits the datasets accordingly for training.

<br><br>
<img width="400" height="390" alt="confusion_matrix" src="https://github.com/user-attachments/assets/fc17d099-1e7e-443d-869a-e693a18a492d" />
<br><br>

Basic Benchmarks (full benchmark in model_benchmark.txt):  
🏆 GLOBAL ACCURACY:      98.95%
🔹 BASIC:       98.56%
🎷 JAZZ:        99.75%
🎵 NOTES:       100.00%
 
