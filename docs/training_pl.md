# Trening modeli

[← README](../README_pl.md) · [In English](training.md)

Aplikacja używa dwóch modeli i każdy ma w `dist/` własny trener. Nie dzielą
ani wag, ani wejścia, tylko aplikację:

| plik aplikacji | co robi | trener |
|---|---|---|
| `best_model_v2_take6_onset.onnx` | pryma i rodzaj akordu, brzmiące dźwięki oraz starsza głowica ataków (wejście CQT) | `model_trainer.py` |
| `short_onset_masking_v2.onnx` | która klasa wysokości została właśnie **uderzona** (wejście: krótkie widma) | `strike_trainer.py` |

Aplikacja pobiera oba z <https://huggingface.co/greblus/solitito-ai>, a paczki
wydań mają je w środku. Jak aplikacja korzysta z modelu uderzeń, opisuje
[Jak to działa](how-it-works_pl.md). Oba trenery trzymają snapshoty w
`greblus/chord-model-snapshots` na Hugging Face.

## Model akordów: `model_trainer.py`

To trener, który wytworzył `best_model_v2_take6_onset.onnx`. Zmieniły się w nim
tylko dwie rzeczy: eksport ONNX jest przypięty do eksportera TorchScript
(`dynamo=False`), a faza 4 nie trenuje ponownie, gdy jej checkpoint już
istnieje. Na Kaggle: notebook z GPU, dołączony zbiór danych i sekret
`HF_TOKEN`. Wklej cały plik i uruchom. `RUN_TAG` nazywa przebieg.

Ma cztery fazy, każda wznawiana ze swojego snapshotu na Hugging Face:

1. główny trening sieci akordowej;
2. próg głowicy pitch;
3. dostrajanie głowic, wyłączone (`RUN_PHASE3 = False`);
4. głowica ataków `fc_onset`, trenowana przy zamrożonej reszcie. Zapisuje
   `checkpoint_<RUN_TAG>_onset.pth` i `best_model_<RUN_TAG>_onset.onnx`,
   czyli plik, który ładuje aplikacja.

Przy `RUN_TAG = "v2_take6"` i `checkpoint_v2_take6_onset.pth` na Hugging Face
faza 4 tylko eksportuje. Wyeksportowany plik odpowiada bit w bit tak samo jak
wydany `best_model_v2_take6_onset.onnx`, na wszystkich czterech wyjściach.
Ponowny trening nadpisałby checkpoint, z którego powstał plik aplikacji. Raz
już się to stało, 20.09.2026. Nowa głowica ataków wymaga nowego `RUN_TAG`.

Aplikacja czyta prymę, rodzaj i pitch, a także czwarte wyjście,
`onset_logits`. Sędzia wciąż z niego korzysta: „Zaliczaj tylko to, co
uderzone” i sprawdzenie, czy akord został uderzony. Model uderzeń działa obok
niego, nie zamiast.

## Model uderzeń: `strike_trainer.py`

Jeden plik, bez niczego innego z repozytorium. Na Kaggle: notebook z GPU,
dołączony GuitarSet i sekret `HF_TOKEN`. Wklej **cały** plik i uruchom.
Lokalnie: `python dist/strike_trainer.py --help`; każde ustawienie z początku
pliku ma swoją opcję w wierszu poleceń.

```python
RUN_TAG = "v2_take7_masking_v2_repro"
MODE = "train"  # train, export_only
HF_REPO_ID = "greblus/chord-model-snapshots"
USE_HF = True
INITIAL_ONSET = "hf:checkpoint_v2_take7_onset_best.pth"
ONSET_EPOCHS = 12
ONSET_MASKING_PAIRS = True
ONSET_GAIN_DB = 6.0
```

- `train` trenuje sieć albo wznawia trening, a potem zapisuje plik dla
  aplikacji.
- `export_only` odtwarza plik dla aplikacji z
  `checkpoint_<RUN_TAG>_onset_best.pth` zakończonego przebiegu i progu z jego
  podsumowania. Nie potrzebuje zbioru danych. Bez podsumowania ustaw
  `EXPORT_ONSET_THRESHOLD` na próg wybrany przez tamten przebieg i nie zgaduj
  go z logu epok. Brak plików wejściowych jest błędem, nigdy powodem do
  rozpoczęcia treningu.
- `USE_HF=False` działa całkiem lokalnie, bez konta. Przy włączonym HF błąd
  logowania albo sieci jest błędem. Trener nigdy nie bierze go za puste
  repozytorium i nie zaczyna przez to od zera.

### Co zapisuje przebieg

Do `OUTPUT_ROOT/RUN_TAG/` oraz, gdy HF jest włączony, do `HF_REPO_ID`:

- `short_onset_<nazwa>.onnx`: **plik, który ładuje aplikacja.** `<nazwa>` to
  `RUN_TAG` bez `v2_take7_`: wydany przebieg `v2_take7_masking_v2` daje
  `short_onset_masking_v2.onnx`, domyślny `short_onset_masking_v2_repro.onnx`.
  Wejście `short_features [batch,770,time]`, wyjście
  `onset_logits [batch,12,time]`. W metadanych:
  - `onset_threshold`, który czyta aplikacja;
  - `onset_history_frames`;
  - opis cech.

  Trener sprawdza, że zapis metadanych nie zmienia ani jednej odpowiedzi.
- `checkpoint_<RUN_TAG>_onset_best.pth`: wybrane wagi. To punkt startowy dla
  kolejnego douczania i źródło dla `export_only`.
- `checkpoint_<RUN_TAG>_onset_last.pth`: ostatnia ukończona epoka z
  optymalizatorem, stanem losowania i najlepszymi wagami. Przebieg o tej samej
  nazwie wznawia się z tego pliku.
- `training_summary_<RUN_TAG>.json`: konfiguracja, skróty danych, wybrany próg,
  cała krzywa progów na walidacji i wyniki testu.

### Ponowne wytrenowanie wydanego modelu uderzeń

Ustawienia domyślne to przepis wydanego `short_onset_masking_v2.onnx`:
- douczanie od `checkpoint_v2_take7_onset_best.pth`, czyli przebiegu `v2_take7`;
- 12 epok, włączone pary maskujące;
- 96/96/96 grup syntetycznych, gracze GuitarSetu 04 i 05 jako walidacja i test;
- rozrzut poziomu ±6 dB;
- ziarno 20260923.

Różni się tylko `RUN_TAG`, `v2_take7_masking_v2_repro`. To nowa nazwa sprawia,
że trener trenuje. Przy istniejącej nazwie wznawia przebieg z jej snapshotów,
a dla zakończonego przebiegu tylko zapisuje pliki jeszcze raz.

Jak blisko wychodzi nowy przebieg:
- **Na CPU:** ten plik trenuje bit w bit tak samo jak skrypty, którymi
  wytrenowano wydany model. Sprawdzone i od zera, i od checkpointu rodzica:
  te same dane, partie, straty, wagi, próg i wyjścia ONNX.
- **Na GPU Kaggle:** przebieg z domyślnym `RUN_TAG` doszedł do tych samych wag
  co wydany. Jego podsumowanie leży w
  `dist/training_summary_v2_take7_masking_v2_repro.json`.
- **Wyeksportowany plik:** `export_only` z checkpointu wydanego przebiegu daje
  plik, który odpowiada bit w bit tak samo jak wydany.

Nowy model uderzeń wstawisz do aplikacji na jeden z dwóch sposobów:
- nazwij go `short_onset_masking_v2.onnx` i połóż obok binarki;
- albo zmień `MODEL` w `src/strike.rs` i `STRIKE_MODEL` w
  `.github/workflows/release.yml`.

`./solitito --check` pokazuje, jaki model uderzeń znalazł i z jakim progiem.

### Skąd się wziął rodzic

Sieć uderzeń powstała w trzech przebiegach, każdy od najlepszych wag
poprzedniego:

1. **Rise**: sieć trenowana od zera. Ten eksperyment przechowuje gałąź `rise`:
   kontrola kontra Rise przy identycznych wagach startowych i partiach, potem
   straty parowe i wagi wybrzmiewania, które nie pomogły.
2. **`v2_take7`**: douczanie.
3. **`v2_take7_masking_v2`**: dodane pary maskujące.

W tamtych przebiegach sieć była gałęzią jednego połączonego grafu, obok
zamrożonej sieci akordowej, a plik aplikacji z niego wycinano
(`dist/extract_onset_branch.py`). Gałąź nigdy nie czytała wejścia akordowego,
więc trener zapisuje teraz samą sieć. `INITIAL_ONSET=""` zaczyna od zera
według przepisu z tego pliku. To nowy eksperyment, a nie powtórzenie tamtego
łańcucha.

### Dane

- **GuitarSet**: całe nagrania solo i comp, mono mix albo mic (wybierane
  samo, gdy dołączony jest tylko jeden wariant). Początki dźwięków pochodzą
  z adnotacji, więc nie są potwierdzonymi atakami kostki. Gracze są dzieleni
  w całości: 00–03 trening, 04 walidacja, 05 test.
- **Syntetyczne szarpnięcia** (Karplus–Strong, `onset-ks-v2`) z dokładnymi
  czasami ataków. Są wśród nich wybrzmiewania, ponowne szarpnięcia
  brzmiącego dźwięku, oktawy, dodana tercja lub kwinta i ponownie uderzone
  trójdźwięki.
- **Pary maskujące**: pryma i tercja wybrzmiewają, a potem pada górna kwinta,
  −18, −12, −6 lub 0 dB względem nich (mierzone w jej pierwszych 96 ms).
  Każda grupa ma trzy warianty: samo tło, sama kwinta, oba razem. Warianty
  mają wspólne tło i wzmocnienie.

Wejście to 770 cech krótkich widm na każdy krok 16 ms. Aplikacja liczy te same
ramki (`ShortFeatures` w `src/strike.rs`). Wspólny plik testowy
(`dist/fixtures/`) trzyma trener i aplikację przy tych samych liczbach,
bit w bit, w obu zestawach testów.

Trening rozrzuca poziom każdego bloku o ±`ONSET_GAIN_DB`. Gracz grający o
11 dB ciszej niż ten rozrzut tracił z wydanym modelem połowę uderzeń. Teraz
aplikacja wyrównuje poziom przed modelem (`Leveller`), więc wydany model nie
wymaga z tego powodu zmian. Szerszy rozrzut to osobny eksperyment pod nowym
`RUN_TAG`.

### Wznawianie

Snapshot `_onset_last` pozwala kontynuować dokładnie od granicy epoki; epoka
przerwana w połowie liczy się od nowa. Przy wznawianiu trener sprawdza, czy
źródła, podziały, adnotacje, skróty cech i ustawienia treningu się nie
zmieniły. Jeśli coś się zmieniło, zatrzymuje się z `Cannot resume onset
training with changed data` (albo `configuration`). Zachowaj checkpoint i
przywróć dane. Nie zmieniaj nazwy przebiegu, żeby wymusić restart.

Ponownie wygenerowane syntetyczne pliki WAV mogą mieć inny skrót, bo nagłówek
`PEAK` zapisuje czas utworzenia. Sama ta różnica jest akceptowana, ale tylko
wtedy, gdy plik cech policzony z WAV jest identyczny.
`rise/resume_data_check.json` mówi, co zostało porównane.

Zakończony przebieg zostawia swoje dane (`features/`, `onset-prepared-*`) w
katalogu przebiegu. Na Kaggle zapis takiego outputu trwa chwilę po `Done.`.

### Testy

```bash
cd dist && python -m unittest test_strike_trainer
```

Testy, które trenują albo eksportują, potrzebują `torch`, `onnx` i
`onnxruntime`; bez nich są pomijane. Po stronie Rusta `cargo test` uruchamia
na tym samym pliku testowym `features_are_the_trainers_to_the_last_bit`. Po
celowej zmianie cech plik testowy odtwarza
`python test_strike_trainer.py --write-fixture`.
