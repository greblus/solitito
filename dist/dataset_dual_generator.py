!pip install pyguitarpro

import guitarpro
from guitarpro import models, write
import os
import csv

# ==========================================
# 1. KONFIGURACJA
# ==========================================
GP_FILE = "dataset_barcode_dual.gp5"
REF_CSV = "dataset_reference.csv"
MAX_FRET = 15
BPM = 120 

# ==========================================
# 2. DANE MUZYCZNE (KOMPLETNE)
# ==========================================
NOTE_ORDER = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
NOTE_TO_IDX = {n:i for i,n in enumerate(NOTE_ORDER)}
ORDER = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

STRING_OPEN_IDX = {
    1: NOTE_ORDER.index("E"), 2: NOTE_ORDER.index("B"), 3: NOTE_ORDER.index("G"), 
    4: NOTE_ORDER.index("D"), 5: NOTE_ORDER.index("A"), 6: NOTE_ORDER.index("E")
}
PAIR_STRINGS = {"EAD": (6,5,4), "ADG": (5,4,3), "DGB": (4,3,2), "GBE": (3,2,1)}
OPEN_STRING_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

# --- PEŁNE SŁOWNIKI AKORDÓW ---
MAJORS = {
 "C": {"pairs": {"EAD": [(0,3,5),(3,3,2),(8,7,5)], "ADG": [(3,2,0),(7,5,5),(10,10,9)], "DGB": [(2,0,1),(5,5,5),(10,9,8)], "GBE": [(0,1,0),(5,5,3),(9,8,8)]}},
 "C#": {"pairs": {"EAD": [(4,4,3),(9,8,6),(13,11,11)], "ADG": [(4,3,1),(8,6,6),(11,11,10)], "DGB": [(3,1,2),(6,6,6),(11,10,9)], "GBE": [(1,2,1),(6,6,4),(10,9,9)]}},
 "D": {"pairs": {"EAD": [(2,0,0),(5,5,4),(10,9,7)], "ADG": [(5,4,2),(9,7,7),(12,12,11)], "DGB": [(4,2,3),(7,7,7),(12,11,10)], "GBE": [(2,3,2),(7,7,5),(11,10,10)]}},
 "D#": {"pairs": {"EAD": [(3,1,1),(6,6,5),(11,10,8)], "ADG": [(1,1,0),(6,5,3),(10,8,8)], "DGB": [(5,3,4),(8,8,8),(13,12,11)], "GBE": [(3,4,3),(8,8,6),(12,11,11)]}},
 "E": {"pairs": {"EAD": [(4,2,2),(7,7,6),(12,11,9)], "ADG": [(2,2,1),(7,6,4),(11,9,9)], "DGB": [(2,1,0),(6,4,5),(9,9,9)], "GBE": [(1,0,0),(4,5,4),(9,9,7)]}},
 "F": {"pairs": {"EAD": [(5,3,3),(8,8,7),(13,12,10)], "ADG": [(0,3,5),(3,3,2),(8,7,5)], "DGB": [(3,2,1),(7,5,6),(10,10,10)], "GBE": [(2,1,1),(5,6,5),(10,10,8)]}},
 "F#": {"pairs": {"EAD": [(6,4,4),(9,9,8),(14,13,11)], "ADG": [(4,4,3),(9,8,6),(13,11,11)], "DGB": [(4,3,2),(8,6,7),(11,11,11)], "GBE": [(3,2,2),(6,7,6),(11,11,9)]}},
 "G": {"pairs": {"EAD": [(3,2,0),(7,5,5),(10,10,9)], "ADG": [(2,0,0),(5,5,4),(10,9,7)], "DGB": [(0,0,0),(5,4,3),(9,7,8)], "GBE": [(4,3,3),(7,8,7),(12,12,10)]}},
 "G#": {"pairs": {"EAD": [(4,3,1),(8,6,6),(11,11,10)], "ADG": [(3,1,1),(6,6,5),(11,10,8)], "DGB": [(1,1,1),(6,5,4),(10,8,9)], "GBE": [(5,4,4),(8,9,8),(13,13,11)]}},
 "A": {"pairs": {"EAD": [(5,4,2),(9,7,7),(12,12,11)], "ADG": [(4,2,2),(7,7,6),(12,11,9)], "DGB": [(2,2,2),(7,6,5),(11,9,10)], "GBE": [(2,2,0),(6,5,5),(9,10,9)]}},
 "A#": {"pairs": {"EAD": [(1,1,0),(6,5,3),(10,8,8)], "ADG": [(5,3,3),(8,8,7),(13,12,10)], "DGB": [(3,3,3),(8,7,6),(12,10,11)], "GBE": [(3,3,1),(7,6,6),(10,11,10)]}},
 "B": {"pairs": {"EAD": [(2,2,1),(7,6,4),(11,9,9)], "ADG": [(6,4,4),(9,9,8),(14,13,11)], "DGB": [(4,4,4),(9,8,7),(13,11,12)], "GBE": [(4,4,2),(8,7,7),(11,12,11)]}},
}

MINORS = {
 "C": {"pairs": {"EAD": [(3,3,1),(8,6,5),(11,10,10)], "ADG": [(3,1,0),(6,5,5),(10,10,8)], "DGB": [(1,0,1),(5,5,4),(10,8,8)], "GBE": [(5,4,3),(8,8,8),(12,13,11)]}},
 "C#": {"pairs": {"EAD": [(4,4,2),(9,7,6),(12,11,11)], "ADG": [(4,2,1),(7,6,6),(11,11,9)], "DGB": [(2,1,2),(6,6,5),(11,9,9)], "GBE": [(1,2,0),(6,5,4),(9,9,9)]}},
 "D": {"pairs": {"EAD": [(1,0,0),(5,5,3),(10,8,7)], "ADG": [(5,3,2),(8,7,7),(12,12,10)], "DGB": [(3,2,3),(7,7,6),(12,10,10)], "GBE": [(2,3,1),(7,6,5),(10,10,10)]}},
 "D#": {"pairs": {"EAD": [(2,1,1),(6,6,4),(11,9,8)], "ADG": [(6,4,3),(9,8,8),(13,13,11)], "DGB": [(4,3,4),(8,8,7),(13,11,11)], "GBE": [(3,4,2),(8,7,6),(11,11,11)]}},
 "E": {"pairs": {"EAD": [(0,2,5),(3,2,2),(7,7,5)], "ADG": [(2,2,0),(7,5,4),(10,9,9)], "DGB": [(2,0,0),(5,4,5),(9,9,8)], "GBE": [(0,0,0),(4,5,3),(9,8,7)]}},
 "F": {"pairs": {"EAD": [(4,3,3),(8,8,6),(13,11,10)], "ADG": [(3,3,1),(8,6,5),(11,10,10)], "DGB": [(3,1,1),(6,5,6),(10,10,9)], "GBE": [(1,1,1),(5,6,4),(10,9,8)]}},
 "F#": {"pairs": {"EAD": [(5,4,4),(9,9,7),(14,12,11)], "ADG": [(4,4,2),(9,7,6),(12,11,11)], "DGB": [(4,2,2),(7,6,7),(11,11,10)], "GBE": [(2,2,2),(6,7,5),(11,10,9)]}},
 "G": {"pairs": {"EAD": [(3,1,0),(6,5,5),(10,10,8)], "ADG": [(1,0,0),(5,5,3),(10,8,7)], "DGB": [(5,3,3),(8,7,8),(12,12,11)], "GBE": [(3,3,3),(7,8,6),(12,11,10)]}},
 "G#": {"pairs": {"EAD": [(4,2,1),(7,6,6),(11,11,9)], "ADG": [(2,1,1),(6,6,4),(11,9,8)], "DGB": [(1,1,0),(6,4,4),(9,8,9)], "GBE": [(4,4,4),(8,9,7),(13,12,11)]}},
 "A": {"pairs": {"EAD": [(5,3,2),(8,7,7),(12,12,10)], "ADG": [(0,2,5),(3,2,2),(7,7,5)], "DGB": [(2,2,1),(7,5,5),(10,9,10)], "GBE": [(2,1,0),(5,5,5),(9,10,8)]}},
 "A#": {"pairs": {"EAD": [(6,4,3),(9,8,8),(13,13,11)], "ADG": [(4,3,3),(8,8,6),(13,11,10)], "DGB": [(3,3,2),(8,6,6),(11,10,11)], "GBE": [(3,2,1),(6,6,6),(10,11,9)]}},
 "B": {"pairs": {"EAD": [(2,2,0),(7,5,4),(10,9,9)], "ADG": [(5,4,4),(9,9,7),(14,12,11)], "DGB": [(4,4,4),(9,8,7),(13,11,12)], "GBE": [(4,4,2),(8,7,7),(11,12,11)]}},
}

JAZZ_SHAPES = {
    "Maj7": [{"base": 6, "offsets": [0, None, 1, 1, 0, None]}, {"base": 5, "offsets": [0, 2, 1, 2, None, None]}],
    "7": [{"base": 6, "offsets": [0, None, 0, 1, 0, None]}, {"base": 5, "offsets": [0, 2, 0, 2, None, None]}],
    "m7": [{"base": 6, "offsets": [0, None, 0, 0, 0, None]}, {"base": 5, "offsets": [0, 2, 0, 1, None, None]}],
    "m7b5": [{"base": 6, "offsets": [0, None, 0, 0, -1, None]}, {"base": 5, "offsets": [0, 1, 0, 1, None, None]}],
    "dim7": [{"base": 6, "offsets": [0, None, -1, 0, -1, None]}, {"base": 5, "offsets": [0, 1, -1, 1, None, None]}],
    "9": [{"base": 5, "offsets": [0, -1, 0, 0, None, None]}],
    "13": [{"base": 6, "offsets": [0, None, 0, 1, 2, None]}]
}
JAZZ_TYPES = ["Maj7", "7", "m7", "m7b5", "dim7", "9", "13"]

# ==========================================
# 3. SILNIK DUAL-TRACK BARCODE (POPRAWIONY)
# ==========================================

def get_song_dual():
    s = models.Song()
    s.title = "Dataset_Barcode_Dual"
    s.artist = "Generator"
    s.tempo = BPM 
    
    # Track 1: Gitara (Clean)
    t1 = s.tracks[0] 
    t1.name = "Guitar"
    t1.strings = [models.GuitarString(i, OPEN_STRING_MIDI[i]) for i in range(1,7)]
    t1.channel = models.MidiChannel(channel=0, instrument=27) 
    
    # Track 2: Data Synth (Square Wave)
    t2 = models.Track(s)
    t2.name = "Data Stream"
    # Instrument 81 to Square Lead (w MIDI liczone od 1). W kodzie GP 0-based = 80.
    t2.channel = models.MidiChannel(channel=1, instrument=80) 
    t2.strings = [models.GuitarString(i, OPEN_STRING_MIDI[i]) for i in range(1,7)]
    
    # Ważne: ustawienie koloru może pomóc wizualnie w TuxGuitar, ale instrument MIDI jest kluczowy
    t2.color = models.Color(255, 0, 0) # Czerwony
    
    s.tracks.append(t2)
    return s, t1, t2

next_measure_start = 0
MEASURE_TICKS = 960 * 4 

def get_voice(measure):
    """
    Kluczowa poprawka: Zwraca pierwszy głos taktu, jeśli istnieje.
    Jeśli nie, tworzy nowy. To zapobiega pustym taktom/pauzom.
    """
    if len(measure.voices) > 0:
        return measure.voices[0]
    else:
        v = models.Voice(measure)
        measure.voices.append(v)
        return v

def add_dual_measure(song, t1, t2):
    """Tworzy takt na obu ścieżkach."""
    global next_measure_start
    h = models.MeasureHeader()
    h.number = len(song.measureHeaders) + 1
    h.start = next_measure_start
    h.timeSignature = models.TimeSignature(numerator=4, denominator=models.Duration(value=4))
    song.measureHeaders.append(h)
    
    m1 = models.Measure(t1, h)
    t1.measures.append(m1)
    
    m2 = models.Measure(t2, h)
    t2.measures.append(m2)
    
    # Pobieramy poprawne głosy
    v1 = get_voice(m1)
    v2 = get_voice(m2)
    
    next_measure_start += MEASURE_TICKS
    return v1, v2

def add_barcode_block_dual(song, t1, t2, uid, strings, frets):
    """
    Takt 1: Barcode (Synth) | Pauza (Gitara)
    Takt 2: Bufor Ciszy
    Takt 3: Akord (Gitara) | Pauza (Synth)
    Takt 4: Ogon (Gitara) | Pauza (Synth)
    """
    
    # --- TAKT 1: BARCODE ---
    v_gtr, v_syn = add_dual_measure(song, t1, t2)
    
    # Gitara: pauza całonutowa
    b_g = models.Beat(v_gtr)
    b_g.duration = models.Duration(value=1)
    b_g.status = models.BeatStatus.rest
    v_gtr.beats.append(b_g)
    
    # Synth: KOD
    binary_string = f"{uid:012b}" 
    bits = ['1'] + list(binary_string) # Start bit + 12 bitów ID
    
    for bit in bits:
        b = models.Beat(v_syn)
        b.duration = models.Duration(value=16) # Szesnastka
        
        if bit == '1':
            b.status = models.BeatStatus.normal
            # Nuta: C6 (MIDI 84, próg 20 na strunie E1)
            # Używamy wysokiego C dla wyraźnego pisku
            n = models.Note(b)
            n.string = 1 
            n.value = 20 
            n.velocity = 127 
            n.type = models.NoteType.normal
            b.notes = [n]
        else:
            b.status = models.BeatStatus.rest # Pauza = 0
        
        v_syn.beats.append(b)
        
    # Uzupełnienie taktu (3 szesnastki pauzy)
    for _ in range(3):
        b = models.Beat(v_syn)
        b.duration = models.Duration(value=16)
        b.status = models.BeatStatus.rest
        v_syn.beats.append(b)

    # --- TAKT 2: BUFOR CISZY ---
    v_gtr, v_syn = add_dual_measure(song, t1, t2)
    
    # Obie ścieżki pauza całonutowa
    b_g = models.Beat(v_gtr); b_g.duration=models.Duration(value=1); b_g.status=models.BeatStatus.rest
    v_gtr.beats.append(b_g)
    
    b_s = models.Beat(v_syn); b_s.duration=models.Duration(value=1); b_s.status=models.BeatStatus.rest
    v_syn.beats.append(b_s)

    # --- TAKT 3: AKORD ---
    v_gtr, v_syn = add_dual_measure(song, t1, t2)
    
    # Gitara gra
    b_c = models.Beat(v_gtr)
    b_c.duration = models.Duration(value=1) # Cała nuta
    b_c.status = models.BeatStatus.normal
    
    notes = []
    for s, f in zip(strings, frets):
        n = models.Note(b_c)
        n.string = s
        n.value = f
        n.velocity = 100
        n.type = models.NoteType.normal
        n.effect.letRing = True
        notes.append(n)
    b_c.notes = notes
    v_gtr.beats.append(b_c)
    
    # Synth pauza
    b_s = models.Beat(v_syn); b_s.duration=models.Duration(value=1); b_s.status=models.BeatStatus.rest
    v_syn.beats.append(b_s)

    # --- TAKT 4: OGON ---
    v_gtr, v_syn = add_dual_measure(song, t1, t2)
    
    # Gitara: pauza (ale letRing z poprzedniego taktu trwa)
    b_sus = models.Beat(v_gtr)
    b_sus.duration = models.Duration(value=1)
    b_sus.status = models.BeatStatus.rest
    v_gtr.beats.append(b_sus)
    
    # Synth pauza
    b_s = models.Beat(v_syn); b_s.duration=models.Duration(value=1); b_s.status=models.BeatStatus.rest
    v_syn.beats.append(b_s)

def generate_jazz_voicings(root_note_name):
    # Logika Jazz
    if root_note_name not in NOTE_TO_IDX: return []
    root_idx = NOTE_TO_IDX[root_note_name]
    voicings = [] 
    for chord_type, shapes in JAZZ_SHAPES.items():
        for shape in shapes:
            base_string = shape["base"]
            offsets = shape["offsets"]
            base_string_note_idx = STRING_OPEN_IDX[base_string]
            start_fret = (root_idx - base_string_note_idx) % 12
            poss = []
            if start_fret <= MAX_FRET: poss.append(start_fret)
            if start_fret + 12 <= MAX_FRET: poss.append(start_fret + 12)
            for rf in poss:
                current_strings = []
                current_frets = []
                valid_chord = True
                for i, off in enumerate(offsets):
                    target_string = base_string - i 
                    if target_string < 1: break
                    if off is not None:
                        fret_val = rf + off
                        if 0 <= fret_val <= MAX_FRET:
                            current_strings.append(target_string)
                            current_frets.append(fret_val)
                        else:
                            valid_chord = False; break
                if valid_chord and len(current_strings) >= 3:
                    voicings.append({"type": chord_type, "strings": current_strings, "frets": current_frets, "base_fret": rf})
    voicings.sort(key=lambda x: x['base_fret'])
    return voicings

# ==========================================
# 4. GŁÓWNA PĘTLA
# ==========================================

song, t1, t2 = get_song_dual()
csv_file = open(REF_CSV, 'w', newline='', encoding='utf-8')
writer = csv.writer(csv_file)
writer.writerow(["id", "label"])

# Takt rozbiegowy (pusty dla obu)
add_dual_measure(song, t1, t2)

print("Generowanie datasetu Dual-Track Barcode...")
current_id = 0

# MAJORS
for c in ORDER:
    if c in MAJORS:
        for pk in ["EAD", "ADG", "DGB", "GBE"]:
            for v in MAJORS[c]["pairs"].get(pk, []):
                add_barcode_block_dual(song, t1, t2, current_id, PAIR_STRINGS[pk], v)
                writer.writerow([current_id, f"{c}"])
                current_id += 1

# MINORS
for c in ORDER:
    if c in MINORS:
        for pk in ["EAD", "ADG", "DGB", "GBE"]:
            for v in MINORS[c]["pairs"].get(pk, []):
                add_barcode_block_dual(song, t1, t2, current_id, PAIR_STRINGS[pk], v)
                writer.writerow([current_id, f"{c}m"])
                current_id += 1

# JAZZ
for root_name in ORDER:
    all_v = generate_jazz_voicings(root_name)
    grouped = {t: [] for t in JAZZ_TYPES}
    for v in all_v:
        if v["type"] in grouped: grouped[v["type"]].append(v)
    for c_type in JAZZ_TYPES:
        for v in grouped[c_type]:
            label = f"{root_name} {v['type']}"
            add_barcode_block_dual(song, t1, t2, current_id, v["strings"], v["frets"])
            writer.writerow([current_id, label])
            current_id += 1

# NOTES
for note in ORDER:
    note_idx = NOTE_TO_IDX[note]
    positions = []
    for s in range(1, 7):
        open_idx = STRING_OPEN_IDX[s]
        for f in range(0, MAX_FRET + 1):
            if (open_idx + f) % 12 == note_idx:
                positions.append((s, f))
    for (s, f) in positions:
        add_barcode_block_dual(song, t1, t2, current_id, [s], [f])
        writer.writerow([current_id, f"Note {note}"])
        current_id += 1

csv_file.close()
if os.path.exists(GP_FILE): os.remove(GP_FILE)
write(song, GP_FILE)

print(f"\n✅ SUKCES!")
print(f"GP5: {GP_FILE}")
print(f"CSV REF: {REF_CSV}")
print(f"Liczba próbek: {current_id}")
