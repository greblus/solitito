# Uruchamianie

[← powrót do README](../README_pl.md)

Do każdego [wydania](../../releases) dołączone są gotowe paczki — binarium, ONNX Runtime,
model i wagi DSP, nic poza tym nie jest potrzebne:

```bash
tar xzf solitito_linux-*.tar.gz && cd solitito_linux-* && ./solitito.sh
```

W systemie Windows należy rozpakować archiwum zip i uruchomić `solitito.exe`.

### Ze źródeł

```bash
cargo build --release
```

Binarka wymaga dwóch plików w katalogu roboczym. `dsp_weights.json` znajduje się w tym
repozytorium; modelu tu nie ma, ponieważ waży 29 MB:

```bash
curl -LO https://huggingface.co/greblus/solitito-ai/resolve/main/best_model_v2_take6_onset.onnx
curl -LO https://huggingface.co/greblus/solitito-ai/resolve/main/short_onset_masking_v2.onnx
```

Drugi plik (1 MB) mówi, która struna została uderzona, i na nim opierają się reguły
powtórzeń: nuta, o którą aplikacja prosi ponownie, liczy się dopiero wtedy, gdy ta klasa
zostanie uderzona jeszcze raz. Jest opcjonalny — bez niego aplikacja startuje, mówi o tym
i ocenia powtórzenia na starszym dowodzie, który przepuszcza więcej wybrzmiewających nut.
Jeśli obecny jest `short_onset_rise.onnx`, używany jest zamiast niego.

`./solitito --check` wczytuje oba i sprawdza, czy nadają się do użytku — to samo polecenie
uruchamia proces wydawniczy na każdej budowanej paczce.

Program odmawia startu na starym, gęstym `dsp_weights.json`, zamiast przyjąć go po cichu:
poprzedni format niósł również inne odwzorowanie chromy, co karmiłoby model cechami, na
których nie był trenowany. Nowy plik generuje się poleceniem `python dist/gen_weights.py`
(wymaga biblioteki librosa).

```bash
cargo build --release
./target/release/solitito
```
