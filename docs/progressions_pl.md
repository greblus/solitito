# Analiza i nauka progresji

Pasek akordów na górze okna (Akordy, Interwały i Arpeggia po zmianach) pokazuje nie tylko, który akord jest teraz, ale też
jego miejsce w utworze i to, dokąd prowadzi. Włącza się to opcją **Stopnie na pasku akordów** w [ustawieniach](settings_pl.md).
Gdy progresję już znasz, wyłącz pasek i miej kolejne akordy w głowie — zobacz
[grę bez paska akordów](intervals-practice_pl.md#gra-bez-paska-akordów).

## Stopnie i funkcje na pasku akordów

<img width="700" alt="Pasek akordów Autumn Leaves ze stopniami i funkcjami" src="screenshots/solitito_strip_functions.png" />

Każdy żeton ma trzy wiersze: akord, jego **stopień** w utworze i — drobniej — **dokąd prowadzi**. Na przykładzie
Autumn Leaves w g-moll:

| Akord | Stopień | Dokąd | Co to znaczy |
|---|---|---|---|
| Cm7 | `iv` | `ii → B♭` | molowy na IV stopniu; tu ii w ii–V–I do B♭ |
| F7 | `♭VII` | `V7 → B♭` | obniżony VII stopień (F, nie F♯); dominanta B♭ |
| B♭maj7 | `♭III` | | cel: paralelna dur |
| E♭maj7 | `♭VI` | | dalej po kole kwintowym |
| Am7♭5 | `iiø` | | półzmniejszony ii molowego ii–V–i |
| D7 | `V` | | dominanta tonacji: idzie do toniki, co mówi już stopień |
| Gm7 | `i` | | tonika |

Najpierw ii–V–I do paralelnej dur, potem molowe iiø–V–i do toniki.

**Stopień** liczony jest od toniki utworu na skali durowej, ze znakiem: w g-moll `F7` to `♭VII`, a `F♯°` to `vii°`.
Wielka litera nad tercją wielką, mała nad małą. `ø` to akord półzmniejszony (m7♭5: kwinta zmniejszona, septyma
mała), `°` zmniejszony (dim7: kwinta i septyma zmniejszone). Akord zmniejszony między dwoma stopniami liczony jest
od dolnego: `F♯°` w F to `♯i°`. Stopnie liczone są od toniki, więc po transpozycji są takie same.

**Dokąd** stoi tylko wtedy, gdy akord prowadzi gdzie indziej niż do toniki, a cel podany jest nazwą akordu:

- `V7 → x` — dominanta, po której następny inny akord leży kwintę niżej. Powtórzona (`C7 C7 F`) prowadzi z obu taktów.
- `subV7 → x` — dominanta rozwiązująca się pół tonu w dół, podstawienie trytonowe; podpisane także przy tonice,
  bo sam stopień `♭II` tego nie mówi.
- `ii → x`, `iiø → x` — akord molowy albo półzmniejszony przed taką dominantą.
- ii–V bez rozwiązania (`Em7 A7 Em7 A7` w Satin Doll) podaje cel, który zapowiada.
- W łańcuchu dominant każda prowadzi do najbliższej: `A7 D7 G7 C` to `V7 → D`, `V7 → G`, `V`, `I`.

Bez dopisku zostają: dominanta na tonice bluesa (`I7`), backdoor (`Fm7 B♭7 → C` to `iv ♭VII I`) i dominanta, która
nigdzie się nie rozwiązuje. Pasek gra w kółko, więc ostatni akord prowadzi do pierwszego.
