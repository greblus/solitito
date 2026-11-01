# Ćwiczenie interwałów

Tu dowiesz się, o co chodzi w każdym z pięćdziesięciu ćwiczeń w Interwałach. Wszystkie
przykłady wypisał sam program, więc są dokładnie takie, jakich program od ciebie oczekuje.

[← powrót do README](../README_pl.md)

---

## Najpierw kilka słów

**Akord** to kilka dźwięków, które pasują do siebie i brzmią razem. Każdy akord ma swój dźwięk
główny, od którego bierze nazwę. W akordzie C tym dźwiękiem jest C. Nazywamy go **prymą**.

**Stopień** to kolejny numer dźwięku skali, więc też akordu, liczony od prymy. Pryma to `1`. Dalej liczymy po kolei: `2`, `3`,
`4`, `5` i tak dalej. Akord z czterech dźwięków składa się zwykle ze stopni `1 3 5 7`.

Najważniejsza sprawa: **ćwiczenie jest zapisane numerami, a nie nazwami dźwięków.** `3` znaczy
„tercja akordu, który jest teraz na ekranie”. W C maj7 to E, a w Cm7 to E♭. Jedno
ćwiczenie pasuje więc do każdego akordu w utworze. Chwyt na gryfie się zmienia, ale brzmienie
„trójki” zostaje. Właśnie o to tu chodzi: o wizualizację gryfu i słyszenie interwałów.

Mały znaczek przy numerze zmienia go o jeden próg: `♭3` jest o próg niżej niż `3`, a `♯4` o próg
wyżej niż `4`.

## Skąd się biorą dźwięki

`1`, `3`, `5` i `7` to zawsze dźwięki z samego akordu.

`2`, `4` i `6` to dźwięki spoza akordu. Pochodzą z jego **skali**, czyli z rzędu dźwięków,
które dobrze brzmią z tym akordem. Każdy rodzaj akordu ma swoją skalę. Tu i w dalszych tabelach
pod nazwą każdego dźwięku stoi jego numer liczony od prymy akordu, żeby numery szybko stały się
znajome:

| akord | skala | dźwięki od C |
|---|---|---|
| maj7 | lidyjska | `C D E F♯ G A B`<br>`1 2 3 ♯4 5 6 7` |
| 7 | miksolidyjska | `C D E F G A B♭`<br>`1 2 3 4 5 6 ♭7` |
| m7 | dorycka | `C D E♭ F G A B♭`<br>`1 2 ♭3 4 5 6 ♭7` |
| m7♭5 | lokrycka z ♮9 | `C D E♭ F G♭ A♭ B♭`<br>`1 2 ♭3 4 ♭5 ♭6 ♭7` |
| °7 | zmniejszona ton-półton | `C D E♭ F G♭ A♭ A`<br>`1 2 ♭3 4 ♭5 ♭6 ♭♭7` |
| m6, m(maj7) | molowa melodyczna | `C D E♭ F G A B`<br>`1 2 ♭3 4 5 6 7` |

W ćwiczeniach z zawieszeniem `4` nad maj7 to zwykłe F, a nie F♯.

Nie musisz uczyć się tych nazw na pamięć. Program sam dobiera skalę do akordu.

## Gra bez paska akordów

Na początku ćwiczenia z podglądem interwałów na gryfie mają pomóc je zrozumieć, wyrobić wyobraźnię
i nauczyć cię widzieć je na gryfie. Pasek na górze okna pokazuje akord, na którym jesteś, i te,
które nadchodzą, więc oswajasz się z akordami utworu. Docelowo powinno się jednak pracować bez
paska akordów: intuicyjnie, z pamięci. To długa droga, sam autor jest dopiero na jej początku.

Odczytanie akordu z paska i wymyślenie położenia jego interwałów, które chcemy zagrać, zajmuje dużo
czasu i powoduje „zacięcie” podczas grania. Następny akord spada na grającego jak niespodzianka,
której chcemy uniknąć.

Celem jest więc gra z głową o krok do przodu. Kiedy grasz na jednym akordzie, myślisz już
o następnym i wiesz, jakie dźwięki tam zagrasz, zanim do niego dojdziesz.

Weźmy **Dwa dźwięki: 3 5** w All the Things You Are. Początek wygląda tak:

| akord | Fm7 | B♭m7 | E♭7 | A♭maj7 |
|---|---|---|---|---|
| `3 5` | `A♭ C`<br>`♭3 5` | `D♭ F`<br>`♭3 5` | `G B♭`<br>`3 5` | `C E♭`<br>`3 5` |

Kiedy palce grają A♭ i C na Fm7 (`♭3` i `5`), głowa jest już na B♭m7: D♭ i F (też `♭3` i `5`). Gdy
tam dojdziesz, przechodzi dalej, na E♭7: G i B♭ (`3` i `5`). Nic cię nie zaskakuje.

Kiedyś będziesz w stanie zaplanować dźwięki dwóch lub pięciu kolejnych akordów i swobodnie między
nimi improwizować. Kolejny akord do zagrania przestanie wtedy zajmować twoją uwagę i zostanie ci czas
na wybór miejsca na gryfie, w którym te dźwięki zagrasz.

Jak do tego dojść:

1. Graj z włączonym paskiem, aż akordy utworu staną się znajome.
2. Wyłącz **Pasek akordów** w szybkich opcjach trybu albo w Ustawieniach, w zakładce Ćwiczenia.
   Na ekranie zostaje tylko bieżący akord.
3. Zanim akord się zmieni, nazwij w myślach następny i jego dźwięki. Idź powoli: jeśli się
   pogubisz, włącz pasek na chwilę.
4. Kiedy następny akord przychodzi bez trudu, spróbuj myśleć o dwa akordy do przodu.

## Jak czytać listę ćwiczeń

Każda pozycja na liście ma nazwę grupy i numery dźwięków. Numery na początku są dla akordów
durowych i dominantowych, takich jak C maj7 albo C7. Czasem po nich jest jeszcze `m:` albo `°:`:

```
Struktura 1–5: 1 2 3 5 · m: 1 b3 4 5
```

- po **`m:`** są numery dla akordów molowych (np. Cm7), zapisane tak, jak ma je akord molowy:
  `b3`, `b7`. Korzystają z nich też akordy półzmniejszone (np. Cm7♭5); u nich `5` to `b5`;
- po **`°:`** są numery dla akordów zmniejszonych (np. C°7), z `b3`, `b5` i `bb7`.

Po co to? W akordzie molowym `2` leży tuż obok `b3`, tylko jeden próg dalej. Te dwa dźwięki obok
siebie brzmią ostro i nieładnie. Dlatego w akordzie molowym zamiast `2` gramy `4`. Nad Cm7 ta
pozycja to:

```
C E♭ F G
1 ♭3 4 5
```

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
| `3` | `E`<br>`3` | `E`<br>`3` | `E♭`<br>`♭3` | `E♭`<br>`♭3` | `E♭`<br>`♭3` |
| `3 7` | `E B`<br>`3 7` | `E B♭`<br>`3 ♭7` | `E♭ B♭`<br>`♭3 ♭7` | `E♭ B♭`<br>`♭3 ♭7` | `E♭ A`<br>`♭3 ♭♭7` |
| `1 3 5` | `C E G`<br>`1 3 5` | `C E G`<br>`1 3 5` | `C E♭ G`<br>`1 ♭3 5` | `C E♭ G♭`<br>`1 ♭3 ♭5` | `C E♭ G♭`<br>`1 ♭3 ♭5` |
| `3 5 7` | `E G B`<br>`3 5 7` | `E G B♭`<br>`3 5 ♭7` | `E♭ G B♭`<br>`♭3 5 ♭7` | `E♭ G♭ B♭`<br>`♭3 ♭5 ♭7` | `E♭ G♭ A`<br>`♭3 ♭5 ♭♭7` |

`3 7` to najważniejsze dwa dźwięki akordu. Mówią, jaki to akord: durowy czy molowy, z jaką
septymą. Zacznij od nich.

### Cztery dźwięki

Cały akord z czterech dźwięków, a potem ten sam akord zaczęty od innego dźwięku:

| ćwiczenie | C maj7 | Cm7 | C°7 |
|---|---|---|---|
| `1 3 5 7` | `C E G B`<br>`1 3 5 7` | `C E♭ G B♭`<br>`1 ♭3 5 ♭7` | `C E♭ G♭ A`<br>`1 ♭3 ♭5 ♭♭7` |
| `3 5 7 1` | `E G B C`<br>`3 5 7 1` | `E♭ G B♭ C`<br>`♭3 5 ♭7 1` | `E♭ G♭ A   C`<br>`♭3 ♭5 ♭♭7 1` |
| `5 7 1 3` | `G B C E`<br>`5 7 1 3` | `G B♭ C E♭`<br>`5 ♭7 1 ♭3` | `G♭ A   C E♭`<br>`♭5 ♭♭7 1 ♭3` |
| `7 1 3 5` | `B C E G`<br>`7 1 3 5` | `B♭ C E♭ G`<br>`♭7 1 ♭3 5` | `A   C E♭ G♭`<br>`♭♭7 1 ♭3 ♭5` |

Ostatnie cztery ćwiczenia tej grupy mają pięć dźwięków: na końcu jeszcze raz ten, od którego się
zaczęło, na przykład `1 3 5 7 1`. Tylko `1 3 5 7` jest zestawem. Resztę grasz zawsze w kolejności
z ekranu.

### Struktura 1–5 i Struktura 5–9

Cztery dźwięki, które razem brzmią jak krótka melodia. Każda struktura ma cztery różne
kolejności i jeszcze losowy przewrót (↻). Najpierw od prymy:

| ćwiczenie | C maj7, C7 | Cm7 | Cm7♭5 | C°7 |
|---|---|---|---|---|
| `1 2 3 5` | `C D E G`<br>`1 2 3 5` | `C E♭ F G`<br>`1 ♭3 4 5` | `C E♭ F G♭`<br>`1 ♭3 4 ♭5` | `C D E♭ G♭`<br>`1 2 ♭3 ♭5` |
| `2 1 5 3` | `D C G E`<br>`2 1 5 3` | `E♭ C G F`<br>`♭3 1 5 4` | `E♭ C G♭ F`<br>`♭3 1 ♭5 4` | `D C G♭ E♭`<br>`2 1 ♭5 ♭3` |
| `3 5 2 1` | `E G D C`<br>`3 5 2 1` | `F G E♭ C`<br>`4 5 ♭3 1` | `F G♭ E♭ C`<br>`4 ♭5 ♭3 1` | `E♭ G♭ D C`<br>`♭3 ♭5 2 1` |
| `5 1 2 3` | `G C D E`<br>`5 1 2 3` | `G C E♭ F`<br>`5 1 ♭3 4` | `G♭ C E♭ F`<br>`♭5 1 ♭3 4` | `G♭ C D E♭`<br>`♭5 1 2 ♭3` |

Struktura 5–9 to ten sam kształt, tylko zaczęty od piątego dźwięku. Ręka gra to, co już zna, a
słyszysz zupełnie inne, bogatsze dźwięki:

| ćwiczenie | C maj7 | C7 | Cm7 | C°7 |
|---|---|---|---|---|
| `5 6 7 2` | `G A B D`<br>`5 6 7 2` | `G A B♭ D`<br>`5 6 ♭7 2` | `G B♭ C D`<br>`5 ♭7 1 2` | `G♭ A♭ A   D`<br>`♭5 ♭6 ♭♭7 2` |
| `6 5 2 7` | `A G D B`<br>`6 5 2 7` | `A G D B♭`<br>`6 5 2 ♭7` | `B♭ G D C`<br>`♭7 5 2 1` | `A♭ G♭ D A`<br>`♭6 ♭5 2 ♭♭7` |

### Prowadzenie głosów

Tu ćwiczysz przejścia między akordami. Ostatni dźwięk przy jednym akordzie leży tuż obok
pierwszego dźwięku przy następnym, najwyżej dwa progi dalej. Dzięki temu melodia płynie z akordu
w akord i nie skacze.

Przykład to początek utworu Autumn Leaves, czyli akordy Cm7, F7 i B♭maj7:

| ćwiczenie | Cm7 | F7 | B♭maj7 |
|---|---|---|---|
| Prowadzenie głosów 1 | `E♭ D C B♭`<br>`♭3 2 1 ♭7` | `A G F E♭`<br>`3 2 1 ♭7` | `D C B♭ A`<br>`3 2 1  7` |
| Prowadzenie głosów 2 | `E♭ D C B♭`<br>`♭3 2 1 ♭7` | `A C E♭ G`<br>`3 5 ♭7 2` | `F B♭ C D`<br>`5 1  2 3` |
| Prowadzenie głosów 3 | `C E♭ G B♭`<br>`1 ♭3 5 ♭7` | `A G F E♭`<br>`3 2 1 ♭7` | `D C B♭ A`<br>`3 2 1  7` |
| Prowadzenie głosów 4 | `C E♭ G B♭`<br>`1 ♭3 5 ♭7` | `A C E♭ G`<br>`3 5 ♭7 2` | `D C B♭ A`<br>`3 2 1  7` |
| Prowadzenie głosów 5 | `G E♭ C B♭`<br>`5 ♭3 1 ♭7` | `A C E♭ G`<br>`3 5 ♭7 2` | `D C B♭ A`<br>`3 2 1  7` |

Spójrz na pierwszy wiersz. Przy Cm7 kończysz na B♭ (`♭7`), a przy F7 zaczynasz od A (`3`), jeden
próg niżej. Przy F7 kończysz na E♭ (`♭7`), a przy B♭maj7 zaczynasz od D (`3`), też jeden próg
niżej. Zagrane szybko brzmi to jak prawdziwa jazzowa melodia.

Te ćwiczenia działają tylko wtedy, gdy akordy idą w kolejności z utworu. Dlatego losowanie nie
zmienia w nich kolejności dźwięków.

### Rozwiązanie zawieszenia

Zaczynasz od dźwięku, który brzmi trochę „w zawieszeniu”, jakby na coś czekał. Potem schodzisz
na `3`, które brzmi spokojnie, i kończysz na prymie:

| ćwiczenie | Cm7 | F7 | B♭maj7 |
|---|---|---|---|
| `2 3 1` | `D E♭ C`<br>`2 ♭3 1` | `G A F`<br>`2 3 1` | `C D B♭`<br>`2 3 1` |
| `4 3 1` | `F E♭ C`<br>`4 ♭3 1` | `B♭ A F`<br>`4  3 1` | `E♭ D B♭`<br>`4  3 1` |

### Skala akordu i skala z dźwiękiem przejściowym

**Skala akordu** to wszystkie dźwięki jego skali, po kolei, przez całą oktawę. Możesz zacząć od
`1`, `3`, `5`, `7` albo od dźwięku wylosowanego przez program. Na przykład od `3`:

| akord | Skala akordu: 3 |
|---|---|
| C maj7 | `E F♯ G A B C D E`<br>`3 ♯4 5 6 7 1 2 3` |
| C7 | `E F G A B♭ C D E`<br>`3 4 5 6 ♭7 1 2 3` |
| Cm7 | `E♭ F G A B♭ C D E♭`<br>`♭3 4 5 6 ♭7 1 2 ♭3` |
| Cm7♭5 | `E♭ F G♭ A♭ B♭ C D E♭`<br>`♭3 4 ♭5 ♭6 ♭7 1 2 ♭3` |

**Skala z dźwiękiem przejściowym** ma jeden dodatkowy dźwięk, w nawiasie. Dzięki niemu skala ma
osiem dźwięków zamiast siedmiu. Gdy grasz ją równo, po dwa dźwięki na każde uderzenie, dźwięki
akordu trafiają zawsze na uderzenie:

| akord | Skala z dźwiękiem przejściowym: 1 |
|---|---|
| C maj7 | `C D E F♯ G (G♯) A B C`<br>`1 2 3 ♯4 5 (♯5) 6 7 1` |
| C7 | `C D E F G A B♭ (B) C`<br>`1 2 3 4 5 6 ♭7 (7) 1` |
| Cm7 | `C D E♭ F G A B♭ (B) C`<br>`1 2 ♭3 4 5 6 ♭7 (7) 1` |

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
| **Losuj także kolejność akordów** | Miesza też kolejność akordów w utworze |
| **Prowadzenie głosów** | Przy mieszaniu zaczyna każdy akord od dźwięku najbliższego temu, który zagrałeś ostatnio |
| **Widok** | Pokazuje numery, chwyt na gryfie albo kafelki z nazwą dźwięku pod każdym numerem |
| **Pasek akordów** | Pokazuje akordy utworu na górze okna. Wyłączony: trzymasz je w głowie |

## Od czego zacząć

1. **Jeden dźwięk**, potem **Dwa dźwięki: 3 7**. Graj przez cały utwór, powoli. Wizualizuj te dwa
   dźwięki na gryfie i słuchaj, jak się przesuwają.
2. **Trzy** i **Cztery dźwięki**, potem ich przewroty, czyli ten sam akord zaczęty od różnych
   dźwięków.
3. **Struktura 1–5** i **5–9** z włączonym losowaniem.
4. **Prowadzenie głosów** w utworze, w którym jest II–V–I, na przykład w Autumn Leaves.
5. **Skala akordu** od `3` i od `7`, a na końcu **skala z dźwiękiem przejściowym**.
6. Dowolne z tych ćwiczeń jeszcze raz, [bez paska akordów](#gra-bez-paska-akordów).

---

[← powrót do README](../README_pl.md)
