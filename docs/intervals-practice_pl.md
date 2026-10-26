# Ćwiczenie interwałów

Tu dowiesz się, o co chodzi w każdym z pięćdziesięciu ćwiczeń w Interwałach. Wszystkie
przykłady wypisał sam program, więc są dokładnie takie, jakich program od ciebie oczekuje.

[← powrót do README](../README_pl.md)

---

## Najpierw kilka słów

**Akord** to kilka dźwięków, które pasują do siebie i brzmią razem. Każdy akord ma swój dźwięk
główny, od którego bierze nazwę. W akordzie C tym dźwiękiem jest C. Nazywamy go **prymą**.

**Stopień** to numer dźwięku liczony od prymy. Pryma to `1`. Dalej liczymy po kolei: `2`, `3`,
`4`, `5` i tak dalej. Akord z czterech dźwięków składa się zwykle ze stopni `1 3 5 7`.

Najważniejsza sprawa: **ćwiczenie jest zapisane numerami, a nie nazwami dźwięków.** `3` znaczy
„trzeci dźwięk tego akordu, który jest teraz na ekranie”. W C maj7 to E, a w Cm7 to E♭. Jedno
ćwiczenie pasuje więc do każdego akordu w piosence. Chwyt na gryfie się zmienia, ale brzmienie
„trójki” zostaje i właśnie to brzmienie uczysz się słyszeć.

Mały znaczek przy numerze zmienia go o jeden próg: `♭3` jest o próg niżej niż `3`, a `♯4` o próg
wyżej niż `4`.

## Skąd się biorą dźwięki

`1`, `3`, `5` i `7` to zawsze dźwięki z samego akordu.

`2`, `4` i `6` to dźwięki spoza akordu. Pochodzą z jego **skali**, czyli z rzędu dźwięków,
które dobrze brzmią z tym akordem. Każdy rodzaj akordu ma swoją skalę:

| akord | skala | dźwięki od C |
|---|---|---|
| maj7 | lidyjska | C D E F♯ G A B (a w ćwiczeniach z zawieszeniem `4` to zwykłe F) |
| 7 | miksolidyjska | C D E F G A B♭ |
| m7 | dorycka | C D E♭ F G A B♭ |
| m7♭5 | lokrycka z ♮9 | C D E♭ F G♭ A♭ B♭ |
| °7 | zmniejszona ton-półton | C D E♭ F G♭ A♭ A |
| m6, m(maj7) | molowa melodyczna | C D E♭ F G A B |

Nie musisz uczyć się tych nazw na pamięć. Program sam dobiera skalę do akordu.

## Wysoko czy nisko? Jak chcesz

**Oktawa** to ten sam dźwięk, tylko wyżej, dwanaście progów dalej. W Interwałach numer mówi,
*który* to dźwięk, a nie *jak wysoko* go zagrać. `2` możesz zagrać nisko albo o oktawę wyżej,
a `1` na końcu `1 3 5 7 1` może być tą samą prymą albo prymą o oktawę wyżej. Program sprawdza
tylko, czy to właściwy dźwięk. Graj tam, gdzie ręce wygodnie, także kilka dźwięków na jednej
strunie.

Numery większe niż 7 (8, 9, 10…) zobaczysz tylko w Skalach i Arpeggiach. Tam przejście przez
oktawę jest częścią ćwiczenia.

## Jak czytać listę ćwiczeń

Każda pozycja na liście ma nazwę grupy i numery dźwięków. Numery na początku są dla akordów
durowych i dominantowych, takich jak C maj7 albo C7. Czasem po nich jest jeszcze `m:` albo `°:`:

```
Struktura 1–5: 1 2 3 5 · m: 1 3 4 5
```

- po **`m:`** są numery dla akordów molowych (np. Cm7) i półzmniejszonych (np. Cm7♭5);
- po **`°:`** są numery dla akordów zmniejszonych (np. C°7).

Po co to? W akordzie molowym `2` leży tuż obok `3`, tylko jeden próg dalej. Te dwa dźwięki obok
siebie brzmią ostro i nieładnie. Dlatego w akordzie molowym zamiast `2` gramy `4`. Nad Cm7 ta
pozycja to C E♭ F G.

Strzałka **↻** oznacza **losowy przewrót**, czyli akord zaczęty od innego dźwięku niż pryma.
Program losuje przewrót osobno dla każdego akordu. Na przykład `1 3 5 ↻` może przyjść jako
`3 5 1` przy jednym akordzie i jako `5 1 3` przy następnym.

**Prowadzenie głosów** ma zamiast numerów zwykłe numerki ćwiczeń: 1, 2, 3… Dźwięki zmieniają się
tam przy każdym akordzie, więc jedna linijka niczego by nie wyjaśniła.

## Ćwiczenia po kolei

Przykłady są dla akordów od C: C maj7, C7, Cm7, Cm7♭5 i C°7.

### Jeden, dwa i trzy dźwięki

To najprostsze ćwiczenia: jeden dźwięk akordu, dwa albo trzy. Grasz je po jednym.

Są **zestawami**. Gdy przełącznik **Po kolei** jest wyłączony, możesz zagrać dźwięki zestawu
w dowolnej kolejności. Gdy jest włączony, grasz je w kolejności z ekranu.

| ćwiczenie | C maj7 | C7 | Cm7 | Cm7♭5 | C°7 |
|---|---|---|---|---|---|
| `3` | E | E | E♭ | E♭ | E♭ |
| `3 7` | E B | E B♭ | E♭ B♭ | E♭ B♭ | E♭ A |
| `1 3 5` | C E G | C E G | C E♭ G | C E♭ G♭ | C E♭ G♭ |
| `3 5 7` | E G B | E G B♭ | E♭ G B♭ | E♭ G♭ B♭ | E♭ G♭ A |

`3 7` to najważniejsze dwa dźwięki akordu. Mówią, jaki to akord: durowy czy molowy, z jaką
septymą. Zacznij od nich.

### Cztery dźwięki

Cały akord z czterech dźwięków, a potem ten sam akord zaczęty od innego dźwięku:

| ćwiczenie | C maj7 | Cm7 | C°7 |
|---|---|---|---|
| `1 3 5 7` | C E G B | C E♭ G B♭ | C E♭ G♭ A |
| `3 5 7 1` | E G B C | E♭ G B♭ C | E♭ G♭ A C |
| `5 7 1 3` | G B C E | G B♭ C E♭ | G♭ A C E♭ |
| `7 1 3 5` | B C E G | B♭ C E♭ G | A C E♭ G♭ |

Ostatnie cztery ćwiczenia tej grupy mają pięć dźwięków: na końcu jeszcze raz ten, od którego się
zaczęło, na przykład `1 3 5 7 1`. Tylko `1 3 5 7` jest zestawem. Resztę grasz zawsze w kolejności
z ekranu.

### Struktura 1–5 i Struktura 5–9

Cztery dźwięki, które razem brzmią jak mała piosenka. Każda struktura ma cztery różne
kolejności i jeszcze losowy przewrót (↻). Najpierw od prymy:

| ćwiczenie | C maj7, C7 | Cm7 | Cm7♭5 | C°7 |
|---|---|---|---|---|
| `1 2 3 5` | C D E G | C E♭ F G | C E♭ F G♭ | C D E♭ G♭ |
| `2 1 5 3` | D C G E | E♭ C G F | E♭ C G♭ F | D C G♭ E♭ |
| `3 5 2 1` | E G D C | F G E♭ C | F G♭ E♭ C | E♭ G♭ D C |
| `5 1 2 3` | G C D E | G C E♭ F | G♭ C E♭ F | G♭ C D E♭ |

Struktura 5–9 to ten sam kształt, tylko zaczęty od piątego dźwięku. Ręka gra to, co już zna, a
słyszysz zupełnie inne, bogatsze dźwięki:

| ćwiczenie | C maj7 | C7 | Cm7 | C°7 |
|---|---|---|---|---|
| `5 6 7 2` | G A B D | G A B♭ D | G B♭ C D | G♭ A♭ A D |
| `6 5 2 7` | A G D B | A G D B♭ | B♭ G D C | A♭ G♭ D A |

### Prowadzenie głosów

Tu ćwiczysz przejścia między akordami. Ostatni dźwięk przy jednym akordzie leży tuż obok
pierwszego dźwięku przy następnym, najwyżej dwa progi dalej. Dzięki temu melodia płynie z akordu
w akord i nie skacze.

Przykład to początek piosenki Autumn Leaves, czyli akordy Cm7, F7 i B♭maj7:

| ćwiczenie | Cm7 | F7 | B♭maj7 |
|---|---|---|---|
| Prowadzenie głosów 1 | E♭ D C B♭ | A G F E♭ | D C B♭ A |
| Prowadzenie głosów 2 | E♭ D C B♭ | A C E♭ G | F B♭ C D |
| Prowadzenie głosów 3 | C E♭ G B♭ | A G F E♭ | D C B♭ A |
| Prowadzenie głosów 4 | C E♭ G B♭ | A C E♭ G | D C B♭ A |
| Prowadzenie głosów 5 | G E♭ C B♭ | A C E♭ G | D C B♭ A |

Spójrz na pierwszy wiersz. Przy Cm7 kończysz na B♭, a przy F7 zaczynasz od A, jeden próg niżej.
Przy F7 kończysz na E♭, a przy B♭maj7 zaczynasz od D, też jeden próg niżej. Zagrane szybko brzmi
to jak prawdziwa jazzowa melodia.

Te ćwiczenia działają tylko wtedy, gdy akordy idą w kolejności z piosenki. Dlatego losowanie nie
zmienia w nich kolejności dźwięków.

### Rozwiązanie zawieszenia

Zaczynasz od dźwięku, który brzmi trochę „w zawieszeniu”, jakby na coś czekał. Potem schodzisz
na `3`, które brzmi spokojnie, i kończysz na prymie:

| ćwiczenie | Cm7 | F7 | B♭maj7 |
|---|---|---|---|
| `2 3 1` | D E♭ C | G A F | C D B♭ |
| `4 3 1` | F E♭ C | B♭ A F | E♭ D B♭ |

### Skala akordu i skala z dźwiękiem przejściowym

**Skala akordu** to wszystkie dźwięki jego skali, po kolei, przez całą oktawę. Możesz zacząć od
`1`, `3`, `5`, `7` albo od dźwięku wylosowanego przez program. Na przykład od `3`:

| akord | Skala akordu: 3 |
|---|---|
| C maj7 | E F♯ G A B C D E |
| C7 | E F G A B♭ C D E |
| Cm7 | E♭ F G A B♭ C D E♭ |
| Cm7♭5 | E♭ F G♭ A♭ B♭ C D E♭ |

**Skala z dźwiękiem przejściowym** ma jeden dodatkowy dźwięk, w nawiasie. Dzięki niemu skala ma
osiem dźwięków zamiast siedmiu. Gdy grasz ją równo, po dwa dźwięki na każde uderzenie, dźwięki
akordu trafiają zawsze na uderzenie:

| akord | Skala z dźwiękiem przejściowym: 1 |
|---|---|
| C maj7 | C D E F♯ G (G♯) A B C |
| C7 | C D E F G A B♭ (B) C |
| Cm7 | C D E♭ F G A B♭ (B) C |

W tych skalach dźwięki często leżą tuż obok siebie. Program to sprawdzał: nadąża, gdy grasz mniej
więcej dwa dźwięki na sekundę albo wolniej. Szczegóły są w [pomiarach sędziego](judge-measurements.md),
w części *Lines in order*.

## Przełączniki

| przełącznik | co robi |
|---|---|
| **Pojedynczo** | Grasz po jednym dźwięku. Wyłączony: możesz uderzyć cały chwyt naraz |
| **Po kolei** | Dźwięki zestawu grasz w kolejności z ekranu. Wyłączony: w dowolnej kolejności. Pozostałe ćwiczenia zawsze idą po kolei |
| **Tylko uderzone** | Liczy się tylko dźwięk, który naprawdę uderzyłeś, a nie ten, który jeszcze brzmi po poprzednim |
| **Losowa kolejność** | Miesza kolejność dźwięków w zestawach i w pierwszym ćwiczeniu każdej struktury, osobno dla każdego akordu |
| **Losuj także kolejność akordów** | Miesza też kolejność akordów w piosence |
| **Prowadzenie głosów** | Przy mieszaniu zaczyna każdy akord od dźwięku najbliższego temu, który zagrałeś ostatnio |
| **Widok** | Pokazuje numery, chwyt na gryfie albo kafelki z nazwą dźwięku pod każdym numerem |

## Od czego zacząć

1. **Jeden dźwięk**, potem **Dwa dźwięki: 3 7**. Graj przez całą piosenkę, powoli, i słuchaj, jak
   te dwa dźwięki się przesuwają.
2. **Trzy** i **Cztery dźwięki**, potem ich przewroty, czyli ten sam akord zaczęty od różnych
   dźwięków.
3. **Struktura 1–5** i **5–9** z włączonym losowaniem.
4. **Prowadzenie głosów** w piosence, w której jest II–V–I, na przykład w Autumn Leaves.
5. **Skala akordu** od `3` i od `7`, a na końcu **skala z dźwiękiem przejściowym**.

---

[← powrót do README](../README_pl.md)
