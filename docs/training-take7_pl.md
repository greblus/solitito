# Trening i eksport take7

[← README](../README_pl.md) · [Pełna instrukcja po angielsku](training-take7.md)

Na Kaggle wklej **cały `dist/model_trainer.py`** do dotychczasowego notebooka.
Dołącz GuitarSet; syntetyczne szarpnięcia skrypt przygotuje sam. Pełny trening
części akordowej wymaga również dotychczasowego syntetycznego zbioru WAV/CSV.
Trening wykonuj na GPU Kaggle.

```python
RUN_TAG = "v2_take7_masking_v2"
MODE = "onset_only"
BASE_RUN = "v2_take6"
USE_HF = True
INITIAL_ONSET = "hf:checkpoint_v2_take7_onset_best.pth"
ONSET_EPOCHS = 12
ONSET_MASKING_PAIRS = True
```

`auto` wznawia własny przebieg albo wykorzystuje bazę take6 i trenuje Rise.
Jeśli nie ma żadnej bazy, trenuje obie części. `onset_only` wymaga gotowej bazy
akordowej. `full` pomija take6, ale nadal może wznowić własny przebieg; dla
całkiem nowego treningu wybierz nowy `RUN_TAG` i ustaw `INITIAL_ONSET=""`.
Błąd dostępu do Hugging Face
nie oznacza pustego repozytorium i nie powoduje automatycznego restartu.

Take6 nie zawiera wag Rise. Obecne ustawienia douczają Rise z checkpointu
`hf:checkpoint_v2_take7_onset_best.pth` w `HF_REPO_ID`. Zwykła ścieżka
`INITIAL_ONSET` wskazuje lokalny checkpoint PyTorch; pusta wartość oznacza
nowe wagi. ONNX ani stara `fc_onset` nie nadają się jako punkt startowy.
Brak wskazanego checkpointu zatrzymuje skrypt przed przygotowaniem danych.
Własny snapshot wznowienia ma pierwszeństwo przed wagami początkowymi.
Podczas treningu onsetów część akordowa pozostaje zamrożona.
`USE_HF=False` pozwala trenować bez konta i tokena; przy `True` ustaw własne
`HF_REPO_ID` i sekret Kaggle `HF_TOKEN`.

Końcowy plik to **`best_model_v2_take7_masking_v2.onnx`**
w `/kaggle/working/v2_take7_masking_v2/`.
Ma dwa wejścia: CQT `features [batch,48,168]` i widma
`short_features [batch,770,time]`. Zwraca root, quality, pitch i onset;
onset ma kształt `[batch,12,time]`. Progi i opis cech są w metadanych.
Plik `_chords.onnx` jest tylko pośrednią częścią bez onsetów.

Zachowaj również `training_summary_v2_take7_masking_v2.json` i checkpointy. Snapshot
`checkpoint_v2_take7_masking_v2_onset_last.pth` zawiera ostatnią pełną epokę, optimizer,
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
Syntetyczny WAV zapisany ponownie może mieć inny hash przez czas utworzenia
w nagłówku `PEAK`. Wznowienie dopuszcza różnicę hasha takiego WAV-a wyłącznie
przy identycznej kolejności źródeł, podziałach, adnotacjach i hashach cech;
sprawdza też hash rzeczywistego pliku cech. Dla GuitarSet kontrola hasha audio
pozostaje ścisła. Raport `rise/resume_data_check.json` zawiera zaakceptowane
różnice oraz konkretne pola, które blokują wznowienie. Nie zmieniaj `RUN_TAG`
ani danych, żeby ominąć błąd zgodności.
Do scalenia już ukończonego treningu użyj `export_only`; do wznowienia przerwanego
treningu potrzebujesz zgodnych danych/cache.

Jeśli po `Epoch 12/12` raport końcowy przerwał `KeyError: 'level'` w
`pair_context`, skopiuj poprawiony `dist/model_trainer.py` i uruchom ponownie
z tym samym `RUN_TAG`, repozytorium HF, danymi i ustawieniami treningu.
Zachowaj `checkpoint_<RUN_TAG>_onset_last.pth` (lokalnie lub na HF).
Snapshot zawiera ukończoną epokę i najlepsze wagi; po komunikacie
`Onsets: resuming take7` ukończone 12 epok zostanie pominięte, a metryki
i eksport wykonają się ponownie. Nie wybieraj tu `export_only`: raport z
wybranym progiem mógł jeszcze nie powstać. Poprawka nie zmienia danych ani wag;
wyklucza próbki bez powtórzenia klasy wysokości tylko z diagnostyki powtórzeń,
zachowując je w zwykłych metrykach zdarzeń.

## Aplikacja i utrzymanie

### Eksperyment: cicha wysoka nuta na tle wybrzmienia

W pełnym `dist/model_trainer.py` ten eksperyment jest teraz domyślnie włączony
pod osobnym `RUN_TAG="v2_take7_masking_v2"`. Ustawienia są wypisywane na początku,
zapisywane w `run_configuration.json` i w polu `configuration` końcowego raportu.
Każdy podział danych powinien zawierać 288 próbek `masking` przy 96 grupach.
Powrót do wcześniejszych danych wymaga `ONSET_MASKING_PAIRS=False`
(CLI: `--no-onset-masking-pairs`) i osobnego przebiegu.

Nowy zestaw dodaje pary: samo wybrzmienie prymy i tercji, sama nowa kwinta oraz
identyczna kwinta na identycznym tle. Jej RMS jest ustawiany względem RMS tła
na −18, −12, −6 lub 0 dB. Zakres kwinty to MIDI 62–85; barwa, atak i tłumienie
są zmienne. Wszystkie warianty pary mają wspólne wzmocnienie i podział zbioru.
Pozostają też dotychczasowe przypadki powtórzeń oraz GuitarSet solo/comp.
To uproszczona syntetyka, nie symulacja konkretnej gitary.

Domyślnie wariant używa 96/96/96 grup syntetycznych, z niezależnymi pobudzeniami
w podziałach train/validation/test. W raporcie sprawdzaj `challenge_recall`
osobno dla `synthetic/masking_add_fifth_*`, obok wyników `masking_alone_*`
i błędnych detekcji `masking_hold_*`. Potrzebne jest porównanie z poprzednim
modelem na tych samych danych; sam wzrost zbiorczego F1 nie wystarcza.
Włączenie zestawu nie jest dowodem poprawy modelu. AtoA i nagrania użytkownika
pozostają wyłącznie materiałem regresyjnym.

### Uruchamianie i źródła

Solitito 0.5.7 obsługuje take7 i wybiera go automatycznie po umieszczeniu
wybranego modelu pod nazwą `best_model_v2_take7.onnx` w katalogu projektu.
Wykonaj `./target/release/solitito --check`, a potem
uruchom zwyczajnie `./target/release/solitito`. Nagrywanie i dodatkowe
potwierdzanie słabszych uderzeń są opcjonalne — [szczegóły](running_pl.md).
Binarkę z wersji 0.5.6 lub wcześniejszej trzeba zaktualizować. `app_ready=false` w raporcie trainera
przypomina, że udany trening i eksport nie weryfikują zaliczania na żywo.

Kod źródłowy trainera jest podzielony na moduły. Po zmianie źródeł wykonaj
`python dist/build_model_trainer.py` i zachowaj źródła, testy oraz wynikowy
`dist/model_trainer.py` razem. [Indeks narzędzi](../dist/README.md) opisuje też
skrypty do wcześniejszych eksperymentów.
