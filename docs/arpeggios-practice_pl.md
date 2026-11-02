# Ćwiczenie arpeggiów

Arpeggia zawierają trzy ćwiczenia: studium w tonacji, ogrywanie zmian i linie Harrisa. Poniższy
opis wyjaśnia przykłady wypisane przez sam program.

[← powrót do README](../README_pl.md)

---

## Najpierw kilka słów

**Arpeggio** to dźwięki akordu zagrane po kolei, jeden po drugim, a nie razem. Akord C7 to
C E G B♭, a jego arpeggio to te same dźwięki zagrane jako linia.

Jak w Interwałach, dźwięki są zapisane **numerami liczonymi od prymy akordu**. `3` to tercja
akordu, który jest teraz na ekranie: w C7 to E, a w Cm7 E♭. Pod nazwą każdego dźwięku w
tabelach stoi jego numer, żeby numery szybko stały się znajome.

Ćwiczenie wybierasz w szybkich opcjach Arpeggiów albo w Ustawieniach, w zakładce Ćwiczenia.

## Studium w tonacji

**Studium** to zapisana fraza, zwykle przez dwie oktawy, grana na jednym akordzie. Studium
wybierasz z tytułu nad ćwiczeniem, a tonację i rodzaj akordu w szybkich opcjach.

| studium | dźwięków | co ćwiczy |
|---|---|---|
| Molowe (dwie oktawy i tercja) | 19 | dwie oktawy i tercja w górę, potem w dół; nad m7 |
| Durowe (dźwięk prowadzący) | 19 | w górę i w dół, na końcu `7` tuż pod prymą; nad maj7 |
| Dominantowe (dojście od dołu) | 19 | w górę i w dół, na końcu dojście do prymy od dołu; nad 7 |
| Z pominięciem dźwięków (kwinty i kwarty) | 30 | skoki co kwintę i kwartę |
| Triolowe (w górę i w dół) | 47 | grupy po trzy dźwięki, w górę i w dół |
| Dwie oktawy w górę i w dół | 15 | dwie oktawy w górę i z powrotem |
| Dwie oktawy w górę | 8 | dwie oktawy w górę |
| Dwie oktawy w dół | 8 | dwie oktawy w dół, od septymy |
| Dwie oktawy w dół od prymy | 8 | dwie oktawy w dół, od prymy |
| Generator | – | nowa fraza po każdym przejściu |

Trzy pierwsze studia ustawiają też swój akord: molowe gra się nad m7, durowe nad maj7, a
dominantowe nad 7. Pozostałe możesz zagrać nad dowolnym akordem.

**Losowa kolejność** zmienia tu tonację: po każdym przejściu studium przenosi się do nowej.
Kolejność dźwięków zostaje, bo studium zagrane w innej kolejności nie byłoby już tym studium.

## Ogrywanie zmian

Akordy pochodzą z utworu, a program buduje arpeggio dla każdego z nich: dźwięki akordu przez
dwie oktawy. **Kierunek** decyduje, czy arpeggio idzie w górę, w dół, czy naprzemiennie: raz w
górę, raz w dół. Nad C7:

| kierunek | C7 |
|---|---|
| wznoszące | `C E G B♭ C E G B♭`<br>`1 3 5 ♭7 1 3 5 ♭7` |
| opadające | `C B♭ G E C B♭ G E`<br>`1 ♭7 5 3 1 ♭7 5 3` |

Każdy akord jest rysowany blisko miejsca na gryfie, w którym skończył się poprzedni. Pierwszy
akord stoi za każdym razem gdzie indziej.

To dobre ćwiczenie na znalezienie każdego akordu na gryfie. Ma jednak wadę: każdy akord zaczyna
się od prymy, więc linia skacze przy każdej zmianie. Tego dotyczy następne ćwiczenie.

## Linie Harrisa

Te ćwiczenia są inspirowane metodą Barry'ego Harrisa. Harris uczył, jak budować płynne linie
ósemkowe, dodając do skal dźwięki przejściowe tak, żeby dźwięki akordu trafiały na mocne części
taktu. Najważniejsza nie jest tu nazwa skali, tylko reguła, która pozwala prowadzić linię
rytmicznie i logicznie przez harmonię. Poznajesz tu wybrane zasady Harrisa, a nie całą jego
metodę improwizacji.

Uczył też, skąd brać ćwiczenia: z fraz, które już brzmią jak muzyka. Przy każdej figurze zadaj
sobie pytanie: **czy zagrałbyś to w solówce?** Skala w dół i w górę uczą, gdzie leżą dźwięki, ale
same frazą jeszcze nie są. Bliżej muzyki są arpeggio ze skalą, obie tercje i otoczenie. Gdy figura
zabrzmi dobrze, naciśnij pauzę i graj ją dalej po swojemu: od innego dźwięku, w innym rytmie, z
innym zakończeniem.

### Skąd się bierze dodatkowy półton

Takt ma osiem ósemek, a zwykła skala ma siedem dźwięków. Zagraj C miksolidyjską w dół od prymy.
Mocne części taktu to pierwsza, trzecia, piąta i siódma ósemka:

| | C7 |
|---|---|
| skala z 7 dźwięków | `C B♭ A G F E D C`<br>`1 ♭7 6 5 4 3 2 1` |

Na mocnych częściach są C, A, F i D, czyli `1`, `6`, `4` i `2`. Z akordu jest tylko pryma.

Dodaj jeden dźwięk chromatyczny między `♭7` a prymą. Skala ma teraz osiem dźwięków:

| | C7 |
|---|---|
| skala z 8 dźwięków | `C B   B♭ A G F E D`<br>`1 (7) ♭7 6 5 4 3 2` |

Teraz na mocnych częściach są C, B♭, G i E: same dźwięki akordu. W nawiasie jest dźwięk
przejściowy, który nie należy do skali. Dźwięki akordowe i przejściowe występują na przemian,
więc od dowolnego dźwięku akordu linia „wychodzi dobrze”.

### Każdy akord ma swoją skalę z ośmiu dźwięków

| akord | skala | od C | na mocnych częściach |
|---|---|---|---|
| 7 | bebopowa dominantowa | `C D E F G A B♭ B`<br>`1 2 3 4 5 6 ♭7 (7)` | C E G B♭ |
| maj7 | szóstkowa zmniejszona | `C D E F G G♯   A B`<br>`1 2 3 4 5 (♯5) 6 7` | C E G A |
| m6, m(maj7) | molowa szóstkowa zmniejszona | `C D E♭ F G G♯   A B`<br>`1 2 ♭3 4 5 (♯5) 6 7` | C E♭ G A |
| m7 | dominanta o kwartę wyżej | `C D E♭ E   F G A B♭`<br>`1 2 ♭3 (3) 4 5 6 ♭7` | C E♭ F A |
| m7♭5 | dominanta o tercję wielką niżej | `C D♭ E♭ F G♭ G   A♭ B♭`<br>`1 ♭2 ♭3 4 ♭5 (5) ♭6 ♭7` | C E♭ G♭ A♭ |
| °7 | zmniejszona | `C D E♭ F G♭ A♭ A   B`<br>`1 2 ♭3 4 ♭5 ♭6 ♭♭7 7` | C E♭ G♭ A |

Nad maj7 na mocnych częściach jest akord szóstkowy (`1 3 5 6`), a pomiędzy nimi akord
zmniejszony. Tej skali Harris uczył nad akordami durowymi.

### Nad m7 myślisz dominantą

To druga ważna myśl Harrisa: **nad m7 grasz dominantę o kwartę wyżej**. Cm7 i F7 mogą korzystać z tej
samej linii, bo nad Cm7 myślisz F7. Zmieniają się funkcje dźwięków względem akordu, ale przy
zmianie harmonii nie musisz budować nowej frazy. Widać to w numerach: ten sam dźwięk A to `6` nad
Cm7 i `3` nad F7.

Dlatego nad Cm7 na mocnych częściach są dźwięki F7, czyli C, A, F i E♭. Nad Cm7 to `1`, `6`,
`4` i `♭3`. Dźwiękiem przejściowym jest `(3)`, czyli E: septyma wielka F7.

Tak samo m7♭5: Am7♭5 to górne dźwięki F9 (F A C E♭ G), więc nad Am7♭5 myślisz F7.

Spójrz na początek Autumn Leaves w figurze **Tercja dominanty**. Nad Cm7 i nad F7 linia jest
identyczna, bo w obu przypadkach myślisz F7:

| figura | Cm7 | F7 | B♭maj7 |
|---|---|---|---|
| Tercja dominanty | `A C E♭ G F E   E♭ D`<br>`6 1 ♭3 5 4 (3) ♭3 2` | `A C E♭ G F E   E♭ D`<br>`3 5 ♭7 2 1 (7) ♭7 6` | `D F G C B♭ A G F♯`<br>`3 5 6 2 1  7 6 (♯5)` |

### Figury

**Figura** to droga, którą linia przechodzi przez skalę akordu. Każda ma osiem dźwięków, czyli
takt ósemek.

| figura | jak idzie | przykład |
|---|---|---|
| **Skala w dół** | od dźwięku akordu w dół skali | `C B   B♭ A G F E D`<br>`1 (7) ♭7 6 5 4 3 2` |
| **Skala w górę** | od dźwięku akordu w górę skali | `F G A B♭ C D E♭ E`<br>`4 5 6 ♭7 1 2 ♭3 (3)` |
| **Arpeggio i skala** | arpeggio w górę, oktawa na mocną część, potem skala w dół | `F A C E♭ F E   E♭ D`<br>`1 3 5 ♭7 1 (7) ♭7 6` |
| **Tercja dominanty** | arpeggio od tercji dominanty: `3 5 7 9`, potem skala w dół | `A C E♭ G F E   E♭ D`<br>`6 1 ♭3 5 4 (3) ♭3 2` |
| **Tercja akordu** | arpeggio od tercji samego akordu: `3 5 7 9`, potem skala w dół | `F  A C  E D C  B A`<br>`♭3 5 ♭7 2 1 ♭7 6 5` |
| **Otoczenie** | dźwięk nad celem, dźwięk o półton pod nim, cel na mocną część, potem arpeggio | `F D♯ E G A C E D`<br>`4 ♯2 3 5 6 1 3 2` |

Przykłady są kolejno nad C7, Cm7, F7, Cm7, Dm7 i Cmaj7.

### Dwie tercje

Arpeggio od tercji to ulubiony ruch Harrisa. Brzmi bogaciej niż arpeggio od prymy, bo kończy się
noną. Program ma je w dwóch wersjach.

**Tercja dominanty** zaczyna arpeggio od tercji dominanty, którą ogrywasz bieżący akord. Nad F7
to A C E♭ G, czyli Am7♭5.
Nad m7 linia myśli dominantą, więc nad Dm7 grasz arpeggio od tercji G7: B D F A.

**Tercja akordu** bierze tercję samego akordu, nad którym grasz. Nad Dm7 to F A C E, czyli Fmaj7
schowany w Dm7. Każdy akord brzmi wtedy po swojemu i łatwiej to usłyszeć.

Na II–V–I w C:

| figura | Dm7 | G7 | Cmaj7 |
|---|---|---|---|
| Tercja dominanty | `B D F  A G F♯  F  E`<br>`6 1 ♭3 5 4 (3) ♭3 2` | `B D F  A G F♯  F  E`<br>`3 5 ♭7 2 1 (7) ♭7 6` | `E G A D C B A G♯`<br>`3 5 6 2 1 7 6 (♯5)` |
| Tercja akordu | `F  A C  E D C  B A`<br>`♭3 5 ♭7 2 1 ♭7 6 5` | `B D F  A G F♯  F  E`<br>`3 5 ♭7 2 1 (7) ♭7 6` | `E G B D C B A G♯`<br>`3 5 7 2 1 7 6 (♯5)` |

Nad G7 obie wersje grają to samo, bo tu akord jest dominantą. Różnią się nad Dm7 i nad Cmaj7.

### Otoczenie

**Otoczenie** to figura, w której cel poprzedzają dwa dźwięki podejściowe: najpierw dźwięk ze skali
powyżej celu, potem dźwięk o półton niżej. Cel trafia na mocną część taktu. Dźwięk pod nim często
nie należy do skali i właśnie dlatego tak mocno ciągnie do celu. Program ćwiczy tę jedną wersję
otoczenia; w improwizacji stosuje się też inne układy. Nad Cmaj7, do E:

| | Cmaj7 |
|---|---|
| Otoczenie | `F D♯ E G A C E D`<br>`4 ♯2 3 5 6 1 3 2` |

F (`4`) jest nad celem, D♯ (`#2`) pod nim, a E (`3`) wypada na mocną część. Potem idzie arpeggio w
górę i jeden dźwięk skali w dół. Program zalicza dźwięk spoza skali tak samo jak każdy inny.

### Start

**Start** mówi, od którego dźwięku akordu zaczyna się figura.

| start | co robi |
|---|---|
| **Od prymy** | każdy akord od prymy |
| **Od tercji** | każdy akord od tercji |
| **Po kolei** | kolejne akordy od kolejnego dźwięku na mocną część: pryma, tercja, potem dwa następne |
| **Ciągła linia** | jedna linia przez cały utwór, opisana niżej |

Figury **Tercja dominanty** i **Tercja akordu** zawsze zaczynają się od tercji. Start wybiera dla
nich tylko oktawę. Przy **Otoczeniu** start mówi, który dźwięk jest celem.

Tak wygląda **Po kolei** w skali w dół, na początku Autumn Leaves:

| | Cm7 | F7 | B♭maj7 | E♭maj7 |
|---|---|---|---|---|
| Po kolei | `C B♭ A G F E   E♭ D`<br>`1 ♭7 6 5 4 (3) ♭3 2` | `A G F E   E♭ D C B♭`<br>`3 2 1 (7) ♭7 6 5 4` | `F E♭ D C B♭ A G F♯`<br>`5 4  3 2 1  7 6 (♯5)` | `C B    B♭ A♭ G F E♭ D`<br>`6 (♯5) 5  4  3 2 1  7` |

### Ciągła linia

To najważniejszy start. Linia nie zaczyna się od nowa przy każdym akordzie. Po zmianie akordu
gra jego dźwięk na mocną część najbliższy temu, na którym skończyła, dalej w tę samą stronę. Gdy
skala dojdzie do końca zakresu gitary, zawraca: skala w dół zmienia się w skalę w górę i
odwrotnie. To zmiana kierunku wymuszona zakresem, a nie część figury. Arpeggio i skala z każdym
akordem trochę się wznosi, więc gdy dojdzie do góry, zaczyna od nowa z dołu.

Program dobiera dźwięki według prostych reguł ciągłości. Linia jest zawsze ciągła, ale nie każda
będzie równie muzyczna: to ćwiczenie, a nie gotowa fraza.

Rysunek na gryfie idzie za linią. Każdy akord jest narysowany tam, gdzie linia naprawdę jest, więc
ręka wędruje po gryfie, a nie stoi przy siodełku.

| | Cm7 | F7 | B♭maj7 | E♭maj7 |
|---|---|---|---|---|
| Skala w dół | `E♭ D C B♭ A G F E`<br>`♭3 2 1 ♭7 6 5 4 (3)` | `E♭ D C B♭ A G F E`<br>`♭7 6 5 4  3 2 1 (7)` | `F F♯   G A B♭ C D E♭`<br>`5 (♯5) 6 7 1  2 3 4` | `C B    B♭ A♭ G F E♭ D`<br>`6 (♯5) 5  4  3 2 1  7` |
| Skala w górę | `F G A B♭ C D E♭ E`<br>`4 5 6 ♭7 1 2 ♭3 (3)` | `F G A B♭ C D E♭ E`<br>`1 2 3 4  5 6 ♭7 (7)` | `F F♯   G A B♭ C D E♭`<br>`5 (♯5) 6 7 1  2 3 4` | `C B    B♭ A♭ G F E♭ D`<br>`6 (♯5) 5  4  3 2 1  7` |

Spójrz na skalę w dół. Cm7 kończy się na E (`(3)`), a F7 zaczyna się od E♭ (`♭7`), półton niżej.
Linia nie skacze, tylko idzie dalej. Przy B♭maj7 linia jest już na dole zakresu i zawraca: F7
kończy się na E, a B♭maj7 idzie w górę od F (`5`). Tak brzmi prawdziwa linia bebopowa przez
zmiany.

Nad Cm7 i F7 linia jest ta sama, bo nad Cm7 myślisz F7. Tak w praktyce wygląda „nad II–V graj
V”.

### Osiem czy cztery dźwięki

W standardach akord trwa zwykle takt albo pół taktu. **8 – cały takt** gra całą figurę, a
**4 – pół taktu** jej pierwszą połowę. Gdy akord trwa dwa takty, utwór zapisuje go dwa razy, na
przykład Gm7 Gm7 na końcu Autumn Leaves.

### Tempo

Program nie narzuca tempa. Czeka, aż zagrasz kolejny dźwięk, więc grasz w swoim tempie, tak
jak w pozostałych trybach.

Szybkość gry ogranicza za to rozpoznawanie dźwięków. W dotychczasowych testach bezpiecznym punktem
wyjścia było mniej więcej pół sekundy na dźwięk. To ograniczenie programu, a nie tempo, w jakim
trzeba opanować linię. Gdy grasz szybciej, linia zatrzymuje się na dźwięku, którego program nie
zdążył zaliczyć. Sprawdzone na
liniach złożonych z nagranych dźwięków: przy pół sekundy na dźwięk linie w dół przechodziły w
każdej próbie, także z półtonami, a linie w górę czasem zatrzymywały się na którymś dźwięku.
Przy 0,4 s linie stają. Szczegóły są w [pomiarach sędziego](judge-measurements.md), w części
*Lines in order*.

Jeśli chcesz zagrać linię z metronomem, ósemki w tempie około 60 uderzeń na minutę mieszczą się
w tym czasie. To wskazówka, nie wymaganie.

## Od czego zacząć

1. **Ogrywanie zmian**, wznoszące, w utworze, który znasz. Znajdź każdy akord na gryfie.
2. **Linie Harrisa**, **Skala w dół**, **Od prymy**. Słuchaj, jak dźwięki akordu trafiają na
   mocne części, i wizualizuj je na gryfie.
3. To samo **Od tercji**, potem **Po kolei**.
4. **Tercja akordu** nad II–V–I, potem **Tercja dominanty**. Posłuchaj, czym różnią się nad II
   i nad I.
5. **Otoczenie** od prymy, potem od tercji. Słuchaj, jak dźwięk o półton niżej ciągnie do celu.
6. **Ciągła linia** w skali w dół, potem w górę. Pomyśl o następnym akordzie, zanim do niego
   dojdziesz, jak przy [grze bez paska akordów](intervals-practice_pl.md#gra-bez-paska-akordów).
7. **Arpeggio i skala** albo **Otoczenie** z ciągłą linią, na koniec z czterema dźwiękami na akord.

---

[← powrót do README](../README_pl.md)
