# Uruchamianie

[← powrót do README](../README_pl.md)

Do każdego [wydania](../../releases) dołączone są gotowe paczki — binarium, ONNX Runtime,
model i wagi DSP, nic poza tym nie jest potrzebne:

Wersja **0.5.7** zawiera połączony model take7 z detektorem uderzeń Rise.

```bash
tar xzf solitito_linux-*.tar.gz && cd solitito_linux-* && ./solitito.sh
```

W systemie Windows należy rozpakować archiwum zip i uruchomić `solitito.exe`.

### Ze źródeł

```bash
cargo build --release
```

Aktualna wersja ze źródeł potrzebuje `dsp_weights.json` (z repozytorium) oraz
`best_model_v2_take7.onnx` z [Hugging Face](https://huggingface.co/greblus/solitito-ai)
lub [trainera/eksportu take7](training-take7_pl.md)
w katalogu roboczym. Wagi ONNX nie są zapisywane w Git.
`./target/release/solitito --check` uruchamia obie gałęzie i sprawdza zgodność.

Program odmawia startu na starym, gęstym `dsp_weights.json`, zamiast przyjąć go po cichu:
poprzedni format niósł również inne odwzorowanie chromy, co karmiłoby model cechami, na
których nie był trenowany. Nowy plik generuje się poleceniem `python dist/gen_weights.py`
(wymaga biblioteki librosa).

```bash
cargo build --release
./target/release/solitito
```

### Wykrywanie uderzeń przez take7 (0.5.7)

Umieść **`best_model_v2_take7.onnx`** z trainera w katalogu roboczym.
Program wybierze go automatycznie. Jeden plik zawiera wyjścia akordów, klas
wysokości i Rise; osobny `short_onset_rise.onnx` nie jest wtedy potrzebny.
Niezależne gałęzie są wydzielane w pamięci dla istniejących wątków obliczeń.
Onsety zachowują cykl 16 ms bez uruchamiania części akordowej. Program nie
zapisuje dodatkowych modeli. Nadal potrzebny jest `dsp_weights.json`.

```bash
./target/release/solitito --check
./target/release/solitito
```

`--check` uruchamia obie gałęzie i wypisuje ścieżki oraz próg onsetów.
Inną nazwę pliku wskaż przez `SOLITITO_MODEL=/ścieżka/model.onnx`. Bez take7
nadal działa take6 z osobnym `short_onset_rise.onnx`. Jawne
`SOLITITO_ONSET_MODEL` nadpisuje źródło onsetów również przy take7; usuń tę
zmienną, aby korzystać z jego własnej gałęzi.

Rise jest domyślnie włączony. Przy opcji **Zaliczaj tylko to, co uderzone** tryby
nutowe korzystają z jego zdarzeń, także dla kilku dźwięków jednocześnie.
Bramka szumów nadal obowiązuje, lecz na krótkim oknie audio. Take7 odczytuje próg
onsetów wybrany w walidacji z metadanych modelu (starszy osobny Rise domyślnie
0,8); suwak progu brzmiących nut go nie zmienia.
Nowa runda wymaga nowych zdarzeń. Model nadal może pomylić uderzenia — to wersja
do testowania, nie potwierdzone rozwiązanie wszystkich powtórnych zaliczeń.

Porównanie z dotychczasowym torem onsetów:

```bash
SOLITITO_MODEL=best_model_v2_take6_onset.onnx SOLITITO_ONSET=legacy ./target/release/solitito
```

`SOLITITO_ONSET_MODEL` wskazuje inną ścieżkę do modelu Rise,
a `SOLITITO_ONSET_TRACE=1` zapisuje wszystkie klatki (również odpowiedzi pod
progiem), bramkę i przyjęcie lub odrzucenie zdarzeń oraz zaliczenia.
`./dist/trace_onsets.sh` zapisuje log do nowego pliku w
`dist/crediting_measurements/rise-live/`, bez zasypywania terminala.
`--probe` pokazuje akordy, klasy wysokości i starą głowicę onsetów, jeśli jest
w modelu. Onsety take7 sprawdzaj przez log. `--file` używa
wybranego toru onsetów do ćwiczenia na nagraniu WAV (pierwszy kanał).

Aby zapisać również audio do porównania z logiem, uruchom:

```bash
./dist/trace_onsets.sh --record
```

Po graniu zamknij aplikację normalnie. Log i pliki `-g*.wav` mają wspólny
początek nazwy. WAV zawiera dokładnie próbki podawane do Rise: mono float32,
po przeliczeniu do 16 kHz, bez zmiany wzmocnienia (około 4 MB/minutę).
Każde ponowne otwarcie wejścia tworzy osobny plik; nagrania nie są nadpisywane.
Zwykłe uruchomienie nie nagrywa audio. `SOLITITO_ONSET_RECORD` pozwala też
bezpośrednio wskazać początek nazwy nagrania. Wpis `RISE_CAPTURE end` podaje,
czy zapis się zakończył i czy był ciągły. Nagrań z przerwami nie należy używać
jako ciągłego sygnału do porównań. Nagrania służą diagnozie, nie treningowi.

Eksperymentalne potwierdzanie słabszych odpowiedzi Rise można włączyć do próby:

```bash
SOLITITO_ONSET_RESCUE=1 ./dist/trace_onsets.sh --record
```

Próg zwykłego wykrycia pozostaje zgodny z modelem. Słabsza odpowiedź (co najmniej
0,6 i poniżej tego progu) wymaga przyrostu widma
wskazującego tę nutę i potwierdzenia jej wysokości 16 ms później. Głośniejsza
wybrzmiewająca nuta nie blokuje potwierdzenia: sprawdzany jest również nowy
składnik widma względem tła zapamiętanego przy wykryciu uderzenia. Oba sposoby
wykrywania korzystają z jednej blokady powtórzeń. Ten dodatkowy warunek dotyczy
tylko słabych odpowiedzi; mocne wykrycia, również wielodźwiękowe, nie czekają.
Wariant jest domyślnie wyłączony; zwykłe uruchomienie przywraca dotychczasowy tor.
