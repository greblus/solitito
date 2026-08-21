# Wybór wejścia

Co znaczą nazwy na liście urządzeń, gdzie są ustawienia i jakie są tryby
diagnostyczne.

[← powrót do README](../README_pl.md)

### Co znaczą te nazwy w Linuksie

`default`, `pulse`, `pipewire` i `jack` nie są urządzeniami. To drogi do serwera dźwięku, a
pod PipeWire wszystkie kończą się na tym źródle, które pulpit ustawił jako domyślne. Owo
źródło bywa pojedynczym gniazdem wystawionym jako mono (na przykład wbudowany mikrofon), a
warstwa zgodności ALSA podaje je dalej jako dwa identyczne kanały — wybór kanału nie ma
wtedy między czym wybierać i sprawia wrażenie martwego. To, które gniazdo dostaniesz,
rozstrzyga się w ustawieniach dźwięku samego pulpitu.

Nazwy w rodzaju `sysdefault:CARD=U192k` to karty ALSA. Nazwa karty pochodzi od układu, nie
od modelu — Behringer UMC202HD przedstawia się jako `U192k`, układ na płycie głównej zwykle
jako `Generic`. Wybranie karty daje jej gniazda jako prawdziwe, osobne kanały, ale tylko
wtedy, gdy karta jest wolna: normalnie zajmuje ją PipeWire i wówczas każda nazwa i tak
kończy się na serwerze.

Stąd też bierze się to, że lista bywa krótka. Kartę można otworzyć raz, więc karta trzymana
przez PipeWire albo przez inny program w ogóle nie pojawia się w skanowaniu i zostają same
cztery nazwy serwerowe. Lista jest przeglądana od nowa przy każdym otwarciu panelu ustawień
i nigdy nie gubi pozycji, którą raz zobaczyła, więc karta, która się zwolni, pojawia się bez
restartu. Jeśli wybranego urządzenia nie da się otworzyć, program mówi o tym pod napisem
`Channel` i słucha domyślnego — gdzie oba kanały zwykle niosą ten sam sygnał.

Nic z tego nie dotyczy Windowsa, gdzie interfejs widoczny jest jako jedno urządzenie stereo,
a wybór kanału znaczy dokładnie to, co mówi.

Ustawienia mieszkają w `$XDG_CONFIG_HOME/solitito/settings.json` (z odwrotem do `~/.config`
lub `%APPDATA%`). Brak pliku albo plik uszkodzony oznacza powrót do wartości domyślnych, a
nie zablokowany start.

Jest też tryb diagnostyczny:

```bash
SOLITITO_DEBUG=1 ./solitito
```

Dla każdej predykcji wypisuje trzy najwyżej ocenione jakości oraz pełny wektor wysokości
wyrażony jako **interwały względem rozpoznanej prymy**:

```
G m7  | min7=97% sus=0% maj=0% | R96# b25 28 b382# 37 44 b56 594# b616 69 b797# 74
```

To właśnie oddziela „model nie słyszy septymy" od „słyszy ją i ignoruje" — dwa problemy,
które z samej nazwy akordu wyglądają identycznie, a prowadzą w przeciwne strony.

`./solitito --probe nagranie.wav` odpowiada na to samo pytanie zadane całemu nagraniu:
przepuszcza plik przez tę samą ścieżkę cech co na żywo, z niczym niebramkowanym, i wypisuje
dla każdego okna poziom wejścia, stopień wypełnienia okna kontekstowego modelu, dwanaście
prawdopodobieństw wysokości oraz dźwięk, który raportuje sam CQT — dzięki czemu „model tego
nie słyszy" i „program w ogóle nie zapytał" przestają wyglądać tak samo.
