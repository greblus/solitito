How the model was trained?

dataset_dual_generator.py generates dataset_dual.gp5 - dual because it has 
two tracks: one with triads, jazz shapes and single notes, the other one with 
binary representation of chord index in dataset_reference beeped with a synth saw sound.
These beeps are then detected to create perfect annotations.

Sounds crazy? It is - for some strange reason I had troubles creating correctly syncd 
dataset_annotations.csv with some more mathematic methods. Maybe it's W11 ;) maybe my 
DAW of choice, no es importa :)

decode_annotations.py takes beeps_render.wav file, exported from 2nd track of dataset_dual.gp5 
and dataset_reference.csv with proper names names of sound samples and creates 
dataset_annotations.csv.

Now the fun part no 2: the 1st track from gp5 file is exported as "DI" guitar signal,
raw_render.wav which is then rendered in DAW through NAM plugin into dataset_clean.wav 
(Fender Deluxe Reverb clean sound) and dataset_eob.wav (Fender Deluxe Reverb edge of breakup 
tone). Doesn't matter how stupid it seems - I could create lots of nice training data 
almost automatically ;)

And the last part: model_trainer.py takes dataset_clean.wav, dataset_eob.wav (and also 
another dataset called GuitarSet) and dataset_annotations.csv and splits the datasets 
accordingly for training following the naming and timing from this csv file.

A short description of the current model architecture:  
Hybrid  CNN with Squeeze-and-Excitation (SE) blocks.    
Transformer Encoder for temporal context.   
Log-scale Constant-Q Transform (CQT) and Chroma features for precise harmonic analysis.  
Jazz-Optimized: Multi-Head output (Root/Quality) trained with Focal Loss to master complex extended chords (9, 13, m7b5).  

Currently Focal Loss results with higher accuracy for jazzy chords, let's leave it like that for now and treat it like an achievement ;) 
<br><br>
<img width="400" height="390" alt="confusion_matrix" src="https://github.com/user-attachments/assets/de8a2bce-c6db-45b4-97be-089be92a93f7" />
<br><br>

Basic Benchmarks (full benchmark in model_benchmark.txt):  
🏆 GLOBAL ACCURACY:      91.40%  
🔹 BASIC (Triady/Notes): 90.62%  
🎷 JAZZ (7/9/13/dim):    93.11%  
