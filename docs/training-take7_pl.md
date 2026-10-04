# Trening i eksport take7

[← README](../README_pl.md) · [Pełna instrukcja po angielsku](training-take7.md)

Na Kaggle wklej **cały `dist/model_trainer.py`** do dotychczasowego notebooka.
Dołącz GuitarSet; syntetyczne szarpnięcia skrypt przygotuje sam. Pełny trening
części akordowej wymaga również dotychczasowego syntetycznego zbioru WAV/CSV.
Trening wykonuj na GPU Kaggle.

```python
RUN_TAG = "v2_take7"
MODE = "auto"
BASE_RUN = "v2_take6"
USE_HF = True
INITIAL_ONSET = ""
ONSET_EPOCHS = 12
```

`auto` wznawia własny przebieg albo wykorzystuje bazę take6 i trenuje Rise.
Jeśli nie ma żadnej bazy, trenuje obie części. `onset_only` wymaga gotowej bazy
akordowej. `full` pomija take6, ale nadal może wznowić własny przebieg; dla
całkiem nowego treningu wybierz nowy `RUN_TAG`. Błąd dostępu do Hugging Face
nie oznacza pustego repozytorium i nie powoduje automatycznego restartu.

Take6 nie zawiera wag Rise. Domyślnie Rise zaczyna od nowych wag; opcjonalne
`INITIAL_ONSET` wskazuje zgodny checkpoint PyTorch Rise, nie ONNX ani starą
`fc_onset`. Podczas treningu onsetów część akordowa pozostaje zamrożona.
`USE_HF=False` pozwala trenować bez konta i tokena; przy `True` ustaw własne
`HF_REPO_ID` i sekret Kaggle `HF_TOKEN`.

Końcowy plik to **`best_model_v2_take7.onnx`** w `/kaggle/working/v2_take7/`.
Ma dwa wejścia: CQT `features [batch,48,168]` i widma
`short_features [batch,770,time]`. Zwraca root, quality, pitch i onset;
onset ma kształt `[batch,12,time]`. Progi i opis cech są w metadanych.
Plik `_chords.onnx` jest tylko pośrednią częścią bez onsetów.

Zachowaj również `training_summary_v2_take7.json` i checkpointy. Snapshot
`checkpoint_v2_take7_onset_last.pth` zawiera ostatnią pełną epokę, optimizer,
stan generatorów losowych i najlepsze wagi. Przerwany eksport nie wymaga
ponownego treningu, jeśli odpowiednie artefakty zostały zachowane.

## Ukończony trening z dwoma plikami

W aktualnym pełnym skrypcie ustaw:

```python
MODE = "export_only"
RUN_TAG = "v2_take7"
```

Zachowaj repozytorium HF i nazwę ukończonego przebiegu. Skrypt pobierze
`best_model_v2_take7_chords.onnx`, `best_model_v2_take7_onset.onnx` oraz raport
z wybranym progiem. Lokalnie akceptuje też `rise/short_onset_rise.onnx`.
Scala je bez GPU, danych treningowych i kroków optimizera. Przy braku artefaktów
zwraca błąd. Nie uruchamia w zamian treningu od nowa.

`Cannot resume onset training with changed data` oznacza niezgodność kolejności
lub identyfikatorów źródeł, podziałów, hashy audio/cech albo adnotacji zdarzeń.
Same ścieżki nie są porównywane. Nie usuwaj checkpointu ani tej kontroli.
Do scalenia już ukończonego treningu użyj `export_only`; do wznowienia przerwanego
treningu potrzebujesz zgodnych danych/cache.

## Aplikacja i utrzymanie

Aktualna aplikacja obsługuje take7 i wybiera go automatycznie po umieszczeniu
pliku w katalogu projektu. Wykonaj `./target/release/solitito --check`, a potem
uruchom zwyczajnie `./target/release/solitito`. Nagrywanie i dodatkowe
potwierdzanie słabszych uderzeń są opcjonalne — [szczegóły](running_pl.md).
Starszą binarkę trzeba przebudować. `app_ready=false` w raporcie trainera
przypomina, że udany trening i eksport nie weryfikują zaliczania na żywo.

Kod źródłowy trainera jest podzielony na moduły. Po zmianie źródeł wykonaj
`python dist/build_model_trainer.py` i zachowaj źródła, testy oraz wynikowy
`dist/model_trainer.py` razem. [Indeks narzędzi](../dist/README.md) opisuje też
skrypty do wcześniejszych eksperymentów.
