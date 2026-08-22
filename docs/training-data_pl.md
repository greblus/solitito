# Dane treningowe i wyniki

[← powrót do README](../README_pl.md)

## Dane treningowe

Dwa źródła, rozwiązujące różne problemy.

### 1. Zbiór syntetyczny — etykiety dokładne

Wytwarzany przez `dist/dataset_generator_v2.py`. Jeden plik Guitar Pro i zapisywane wprost
adnotacje:

1. Generator układa 394 bloki po 6 s (3 takty przy 120 BPM), obejmujące 12 prym ×
   {maj, min, maj7, dom7, min7, m7b5, dim7, sus4, aug} w kilku pozycjach na gryfie, oraz
   wszystkie 96 pojedynczych dźwięków (6 strun × 16 progów).
2. Przed wygenerowaniem **sprawdza sam siebie: każdy przesuwalny akord na każdym progu** i
   kontroluje, czy rzeczywiście daje deklarowane interwały. Literówka w tabeli akordów
   zatrzymuje generowanie, zamiast po cichu zatruwać zbiór.
3. `synth_annotations.csv` powstaje równocześnie z plikiem GP5. Generator wie, który takt
   zajmuje każdy blok, więc niczego nie trzeba potem odzyskiwać z dźwięku.
4. Ścieżka gitary eksportowana jest jako sygnał DI i renderowana w DAW-ie przez
   [NAM](https://www.neuralampmodeler.com/) dwukrotnie: `synth_dataset_clean.wav` (czysty
   Fender Deluxe Reverb) oraz `synth_dataset_eob.wav` (na granicy przesteru).
5. `dataset_generator_v2.py --calibrate <wav>` mierzy, gdzie wypada pierwszy atak, na
   wypadek gdyby DAW dołożył ciszy na początku.
6. `verify_annotations.py` porównuje każdą etykietę z rzeczywistą zawartością dźwięku, zanim
   ruszy trening.

**Generator wypisuje etykiety wprost, a osobny skrypt weryfikuje, że etykiety opisują
dźwięk.** Pełna procedura w [dist/HOW_TO_PREPARE_DATASET.md](../dist/HOW_TO_PREPARE_DATASET.md).

### 2. GuitarSet — prawdziwa gitara

[GuitarSet](https://guitarset.weebly.com/) to 360 nagrań z adnotacjami JAMS, zarejestrowanych
przetwornikiem heksafonicznym.

## Wyniki

Model `v2_take6`, mierzony na podziale walidacyjnym grupowanym po źródle, z wyłączonymi
oknami solowymi:

| miara | wartość |
|---|---|
| Trafność prymy | **98,1%** |
| F1 wysokości | **0,909** |
| Trafienie dokładne (pryma **i** jakość) | **92,4%** |

Trafność na poszczególnych jakościach w najlepszym punkcie kontrolnym: `dom7` 97%, `min7`
93%, `min` 92%, `sus` 91%, `maj` 89%, `maj7` 89%; `m7b5`, `dim7` i `aug` powyżej 97%.

Różnica między zbiorem treningowym a walidacyjnym wynosi na jakości 6,5 punktu, więc model
znajduje się blisko pułapu, na który pozwalają jego dane. Więcej epok nic nie da; więcej
zróżnicowanych nagrań prawdziwej gitary — owszem.

Ten sam potok, mierzony uczciwie na każdym etapie:

| przebieg | zmiana | trafienie dokładne |
|---|---|---|
| take4 | podział grupowany po źródle — uczciwy punkt odniesienia | 44,8% |
| take5 | zamaskowane nagrania solowe | 82,3% |
| take6 | adnotacje akordów `performed` | **92,4%** |

---
