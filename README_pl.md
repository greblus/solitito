<!-- # Solitito — trener gitary czasu rzeczywistego -->

*[English version](README.md)*

**Solitito** to trener gitarowy czasu rzeczywistego napisany w **Rust**. Słucha gitary przez
interfejs audio (zalecany) albo mikrofon, rozpoznaje, co grasz, i prowadzi przez standardy
jazzowe, interwały, skale, arpeggia i formuły interwałowe.

Rozpoznawanie działa na niewielkiej sieci neuronowej (7,3 mln parametrów) wyeksportowanej do
ONNX. Wszystko — DSP, wnioskowanie, interfejs — dzieje się lokalnie.

<div align="center">
<img height="320" alt="Okno główne Solitito" src="docs/solitito_main.png" />
<img height="320" alt="Shell voicingi" src="docs/solitito_main_shell_voicings.png" />
<img height="320" alt="Powiększony schemat akordu" src="docs/solitito_chord_diagrams.png" />
</div>

Schematy akordów opisane są **interwałami**, więc jeden diagram obsługuje
wszystkie dwanaście tonacji: czerwona kropka to pryma, kliknięcie w
schemat powiększa go. Środkowy zrzut pokazuje ten sam akord jako **shell voicingi** —
tercja i septyma nad prymą, bez kwinty.

Solitito jest mocno zainspirowane [Solo](https://www.solotrainer.app/), trenerem na
Androida i iOS, z którego korzystam codziennie. Solo robi znacznie więcej i robi to dobrze —
ale słyszy jeden dźwięk naraz. To jedyna funkcjonalność, której mi w nim brakuje: grając
interwały przez zmiany harmoniczne, trzeba tłumić struny, żeby każdy dźwięk został
zarejestrowany, a to mało muzyczny sposób ćwiczenia i psuje cały fun. Zastanawiałem się od dawna, jak 
Yousician rozpoznaje akordy polifonicznie i tak powstało Solitito.

Program jest aplikacją desktopową dla Linuksa i Windowsa. Wersji mobilnych nie planuję.

Prace ruszyły w grudniu 2025. Procedura szkolenia modelu i sam program w Rust był przepisywany od zera
więcej niż raz, zanim zadziałał. To, co opisano poniżej, jest w większości zapisem tego, co okazało się
mieć znaczenie.

---

## Co to robi?

Wybierasz utwór, a program pokazuje jeden akord. Po zagraniu go, gdy model usłyszy właściwy
akord, przechodzi do następnego.

- **Gryf** — losowany jest fragment gryfu (zestaw strun, cztery progi) i program
  prosi o kolejne dźwięki, które znajdują się w tym, ograniczonym pozycją ręki, fragmencie gryfu. 
- **Akordy** — pełne standardy jazzowe. Zielony potwierdza trafienie dokładne, żółty przyjmuje
  triadę albo substytucję (voicing bez prymy, `m7` odczytane jako `m`), czerwony znaczy, że
  akord został wykryty, ale sygnał jest za słaby, by go zatwierdzić.
- **Interwały** — składniki akordu grane pojedynczo. Sam wybierasz, które stopnie ćwiczyć
  (`1 3 5` dla triad, `1 3 5 7` dla akordów septymowych, `1 3` dla voicingów szkieletowych).
  `3` obejmuje i `3`, i `b3`, `7` — `7` i `b7`, i tak dalej, zależnie od jakości bieżącego akordu.
- **Skale** — ćwiczenie dźwięków po kolei, z definicji skali.
- **Arpeggia** — składniki akordu w kolejności lub losowo, przez progresję, zapisane stopniami, więc
  jeden wzorzec pasuje do każdego akordu w standardzie. Dwuoktawowe frazy jazzowe oraz
  generator budujący nowe arpeggio po każdym przejściu.
- **Formuły** — zbiór interwałów wylosowany nad prymą, grany w dowolnej kolejności. Każda
  funkcja/interwał zapala się, gdy zabrzmi, a zbiór jest zaliczany, kiedy zapalą się wszystkie; pod
  spodem najbliższa skala, którą już znasz, z wyróżnionymi stopniami formuły, oraz akordy
  w niej zawarte — wskaż jeden, a linia powyżej pokaże, z czego jest zbudowany. Pauza
  zmienia zbiór na niebieski i wstrzymuje ocenianie: twoja kolej, improwizuj, twórz melodie
  z tych dźwięków.
  Tę samą formułę można też zagrać na akordzie albo ograć nią cały standard — jak
  pracować ze wszystkimi trzema, opisuje [ćwiczenie z formułami](docs/formulas-practice_pl.md).

Tryb Formuł jest zainspirowany książką **An Improviser's OS** Wayne'a Krantza — być może
najciekawszym podejściem do kreatywnej improwizacji, jakie powstało.

Książkę można kupić bezpośrednio u [Wayne'a Krantza](https://waynekrantz.bandcamp.com/merch/wayne-krantz-an-improvisers-os-2nd-edition).
Warto też posłuchać jego muzyki, bo to genialny muzyk i improwizator.

<div align="center">
<img height="479" alt="Formuły" src="docs/solitito_formulas.png" />
</div>

Formułę można dodać do Ulubionych klikając na **gwiazdkę** pod nią: program pyta o nazwę i trafia
ona na listę w ustawieniach Ćwiczeń, skąd wybór przywraca tę formułę dla dowolnej tonacji, która
akurat jest na ekranie — formuła jest niezależna od tonacji, więc zapisywany jest sam zbiór interwałów.
Krzyżyk w wierszu Ulubionych usuwa wpis, a kliknięcie w oknie głównym zaświeconej gwiazdki, porzuca
formułę widoczną na ekranie.

Linia pod skalą to **akordy mieszczące się w formule** — ich dźwięki należą do zbioru interwałów w formule.
Zapisane są jako stopień cyframi rzymskimi plus jakość, przy czym stopień liczy się od prymy samej formuły: na zrzucie powyżej
`VI`, `VIm` i `VIsus2` stoją wszystkie na `VI`, czyli w tonacji E na C#.

Triada durowa i molowa na tym samym stopniu pojawiają się obie, ponieważ formuła zawiera
zarówno `1`, jak i `b2` — E oraz F, które czytane od C# są jego tercją wielką i małą — a także
`7`, czyli D#, z którego bierze się sus2. Zapętl dowolny z tych akordów, a formuła staje się
dla niego kolorem: każdy dźwięk pada na miejsce.

Rzymskie dla akordów, arabskie dla funkcji nad nimi, żeby obu rzędów nie dało się pomylić —
akord dominantowy septymowy na drugim stopniu zapisany po arabsku czytałby się „27". Wypisane
są tylko akordy najpełniejsze: taki, który mieści się w innym już obecnym, brzmi zawsze, gdy
brzmi tamten.

Zdumiewające, że dwie linijki kodu w Pythonie tworzą cały muzyczny świat:

```python
from itertools import combinations

F = "1 b2 2 b3 3 4 b5 5 b6 6 b7 7".split()
formulas = [("1", *c) for n in range(12) for c in combinations(F[1:], n)]

len(formulas)  # 2048
```

Formuły to wszystkie podzbiory dwunastu funkcji chromatycznych zawierające „1", czyli
2¹¹ = 2048 — uporządkowane najpierw według liczby dźwięków, a potem leksykograficznie w
porządku chromatycznym.

Dolna część okna głównego pokazuje akord właśnie zagrany oraz następny po bieżącym. Akord
zostawiony za sobą zachowuje kolor, na jaki zasłużył — zielony przy trafieniu dokładnym, żółty
gdy przeszedł triadą albo zamianą — więc zaliczenie pozostaje czytelne po tym, jak program już
poszedł dalej.

Przełącznik losowania miesza kolejność. W trybach nutowych miesza składniki wewnątrz każdego
akordu, osobne ustawienie dokłada do tego losową kolejność akordów.
W Akordach zmienia kolejność standardu, w Skalach losuje tonikę od nowa. Przycisk
pauzy zatrzymuje progresję, podczas gdy kolory dalej mówią, czy akord jest właściwy, więc
można usiąść nad jednym akordem i się go porządnie nauczyć. Przy pauzie strzałki po bokach paska
przechodzą tam i z powrotem po progresji, co pozwala wrócić do akordu, który już minął.

---

## ⚙️ Ustawienia

Cztery zakładki: **Dźwięk** — wejście i bramka szumów, **Ogólne** — jak surowo oceniane jest
to, co grasz, **Ćwiczenia** — co grać, oraz **Program** — co pokazuje okno.

<div align="center">
<img width="220" alt="Ustawienia, zakładka Dźwięk" src="docs/solitito_settings2.png" />
<img width="220" alt="Ustawienia, zakładka Ogólne" src="docs/solitito_settings1.png" />
<img width="220" alt="Ustawienia, zakładka Program" src="docs/solitito_settings4.png" />
</div>

**Ćwiczenia** zawierają wyłącznie to, co ma sens dla trybu widocznego na ekranie — utwór nie ma
nic do powiedzenia w Formułach, formuła nic w Akordach — więc w każdym trybie jest to inna
zakładka, a trener gryfu, który nie ma własnych ustawień, nie pokazuje jej wcale:

<div align="center">
<img width="175" alt="Ćwiczenia, Akordy" src="docs/solitito_settings3.png" />
<img width="175" alt="Ćwiczenia, Interwały" src="docs/solitito_settings5.png" />
<img width="175" alt="Ćwiczenia, Skale" src="docs/solitito_settings6.png" />
<img width="175" alt="Ćwiczenia, Arpeggia" src="docs/solitito_settings7.png" />
<img width="175" alt="Ćwiczenia, Formuły" src="docs/solitito_settings8.png" />
</div>

| Ustawienie | Opis |
|---|---|
| **Utwór / Skala** | Wybiera progresję albo skalę dla bieżącego trybu |
| **Wzorzec** | Tylko Arpeggia: którą frazę przechodzić. Ostatnia pozycja to generator budujący świeżą po każdym przejściu |
| **Tonacja** | Tylko Skale: tonika. Przy włączonej kolejności losowej losowana od nowa po każdym przejściu |
| **Interwały** | Które stopnie ćwiczyć. `1 3 5` dla triad, `1 3 5 7` dla akordów septymowych, `1 3` dla voicingów szkieletowych. `3` obejmuje tercję wielką i małą, `5` kwintę czystą i zmniejszoną, zależnie od jakości akordu |
| **Pokazuj predykcję AI w oknie głównym** | Wyświetla surową odpowiedź modelu na ekranie głównym |
| **Wejście** | Które urządzenie przechwytujące otworzyć. *Domyślne systemowe* podąża za ustawieniem systemu. Zapamiętywane, z odwrotem do domyślnego, gdy urządzenia zabraknie |
| **Kanał** | Na którym wejściu urządzenia słuchać — gitara w gnieździe 2 interfejsu to kanał 2. Pokazywany tylko wtedy, gdy urządzenie ma więcej niż jeden, i nie ma opcji miksowania: uśrednianie wejść wciąga to, co jest na drugim gnieździe, i kosztuje 6 dB |
| **Bramka szumów** | Próg w dBFS. Pasek poniżej pokazuje bieżący poziom wejścia w tej samej skali, z progiem zaznaczonym na czerwono — ustaw go tuż nad szumem przy nietkniętych strunach |
| **Podbicie basu** | Cyfrowe wzmocnienie najniższych prążków CQT. Przydatne przy mikrofonach laptopowych, które zwykle ścinają niskie struny |
| **Trzymaj jakość akordu do nowego ataku** | Utrzymuje rozpoznaną jakość, dopóki nie uderzysz strun ponownie. Bez tego trzymane `m7` zmienia się w `m`, gdy septyma wybrzmiewa |
| **Oceniaj krótkie szarpnięcia po ataku** | Dla akordów uderzanych i puszczanych, a nie trzymanych. Liczy się jeden czysty odczyt celu, a wybrzmiewanie po nim nie może go cofnąć. Zły akord nadal nie przechodzi |
| **Zaliczaj tylko to, co uderzone** | Tryby nutowe: model może zaliczyć dźwięk tylko tam, gdzie głowica ataków usłyszała też uderzenie. Zmierzone na nagraniu pojedynczych dźwięków: dwie trzecie zaliczeń, które model przyznaje dźwiękowi innemu niż grany, nie niesie żadnego ataku — prawie zawsze jest to dźwięk poprzedni, wciąż brzmiący w jego oknie 0,77 s. Domyślnie wyłączone: dźwiękowi, którego atak głowica przeoczy, zostaje wtedy sama ścieżka CQT |
| **Graj dźwięki pojedynczo** | Tylko tryby nutowe. Wyłączone: uderzony akord zalicza swoje interwały jeden po drugim — głowica wysokości jest polifoniczna i raportuje wszystkie dźwięki naraz. Włączone: każdy dźwięk trzeba zagrać osobno, a estymata CQT przegłosowuje model |
| **Losowa kolejność** | Ikona losowania na pasku narzędzi. W trybach nutowych miesza dźwięki wewnątrz każdego akordu; w Akordach zmienia kolejność progresji, a w Skalach losuje tonację po każdym przejściu |
| **Mieszaj także akordy** | Tylko Interwały i Arpeggia, i tylko przy włączonym losowaniu. Wyłączone: progresja zostaje taka, jak zapisana, a ruszają się same dźwięki — pomieszane interwały idące po prawdziwej progresji zamieniają się w melodie. Włączone: akordy również są losowane, co jest ćwiczeniem bardziej abstrakcyjnym |
| **Ćwiczenie** (formuły) | *Formuła w tonacji* to tryb w postaci pierwotnej. *Nad akordem* stawia tę samą formułę na jednym wylosowanym akordzie, a *Nad standardem* na każdym akordzie utworu po kolei — formuła jest stała, harmonia pod nią się zmienia |
| **Nałożenie** | Nad akordem: jakiego rodzaju nałożenie losować — takie, które opisuje akord, barwi go, stoi na zewnątrz, albo dowolne. Ekran pokazuje funkcje formuły czytane od prymy akordu, z jego własnymi dźwiękami na niebiesko, i je zlicza |
| **Dźwięków w formule** | Tylko Formuły: ile funkcji ma każda losowana formuła, wliczając prymę |
| **Tonacja** (formuły) | Pryma, względem której je czytać, albo świeżo losowana do każdej formuły |
| **Musi zawierać** | Losuj wyłącznie formuły zawierające te funkcje, np. `b3 b7`. Puste pole losuje ze wszystkich 2048 |
| **Pokazuj nazwy dźwięków i akordów** | Tylko Formuły: litery pod funkcjami i pod akordami |
| **Pokazuj najbliższą skalę** | Tylko Formuły: najbliższa skala, którą już znasz, rozpisana z wyróżnionymi stopniami formuły |
| **Pokazuj pasujące akordy** | Tylko Formuły: akordy grywalne bez wychodzenia ze zbioru, zapisane stopniami. Tylko najpełniejsze — nad gamą durową zostaje dokładnie siedem akordów septymowych diatonicznych |
| **Ulubione** | Tylko Formuły: gwiazdka zapisuje formułę z ekranu pod nazwą, a wybór z listy ją przywraca. Krzyżyk na wierszu ją usuwa |
| **Graj dźwięki po kolei** | Tylko Formuły: zbiór trzeba przejść od najniższej funkcji. Wyłączone — jest zbiorem, dowolny dźwięk w dowolnej kolejności |
| **Debug na konsoli** | Wypisuje linię dla każdej zaliczonej funkcji wraz z tym, co usłyszano. Okno na ocenianie; w Windowsie wydanie release nie ma konsoli, na której mogłoby to wypisać |
| **Kończ powtórzoną prymą** | Tylko Skale: przebieg czyta się 1 2 3 4 5 6 7 1, ostatnia oktawę wyżej. To osobny krok i trzeba go zagrać |
| **Pokazuj schematy akordów** | Miniatury diagramów pod nazwą akordu w trybie Akordy |
| **Schematy** | Dwa pola: pełne akordy i shell voicings — tercja i septyma nad prymą. Zaznaczone oba rysują oba, żadne nie rysuje nic. Powyżej czterech schematów rysowane są w dwóch rzędach. `m7b5` nie ma własnego shella: jego shell to co do dźwięku shell `m7`, bo różni je wyłącznie kwinta, więc to właśnie jest rysowane, z podpisem *substytut: shell m7*. Akord zmniejszony septymowy nie ma czego pominąć — cztery dźwięki co małą tercję, bez pary tercja-septyma, którą można by zachować — więc zostaje przy pełnych chwytach, z podpisem mówiącym, czym one też są: `7b9` bez prymy, o pół tonu poniżej każdego swojego dźwięku |
| **Tryb startowy** | W jakim trybie program się otwiera |
| **Język** | Auto (z ustawień systemu), polski, angielski. Stosowany natychmiast, bez restartu |
| **Pewność akordu** | Jak pewny musi być model *nazwy* akordu, żeby została zaliczona (tryb Akordy) |
| **Próg dźwięku** | Jak pewny musi być model, że *pojedynczy dźwięk* brzmi (Interwały / Skale / Arpeggia) |
| **Czas trzymania** | Jak długo poprawny akord musi być trzymany, zanim program przejdzie dalej |

Linia pod napisem `Channel` mówi, co faktycznie zostało otwarte — urządzenie, częstotliwość
próbkowania, liczbę kanałów i format próbek. `./solitito --help` wypisuje wszystkie opcje.
`./solitito --devices` podaje te same informacje dla każdego urządzenia widocznego dla
backendu, a `./solitito --bench` mierzy czas jednego przejścia modelu — program pyta model co
40 ms, dopóki akord brzmi, więc ta liczba jest w praktyce całym jego obciążeniem procesora.
Wydanie release ma podsystem `windows`, więc w Windowsie tryby te piszą do konsoli, z której
zostały uruchomione; uruchomione zupełnie bez konsoli — ze skrótu niosącego flagę — otwierają
własną i czekają na klawisz, żeby raport dało się przeczytać.

---

## Czytaj dalej

Reszta dokumentacji mieszka w `docs/`, żeby ta strona pozostała o tym, czym program jest i jak
go ustawić.

| | |
|---|---|
| [**Ćwiczenie z formułami**](docs/formulas-practice_pl.md) | Trzy ćwiczenia z formułami krok po kroku: co grać, czego słuchać i co mówi każdy rząd na ekranie |
| [Wybór wejścia](docs/audio-input_pl.md) | Co znaczą nazwy urządzeń w Linuksie, gdzie trzymane są ustawienia i jakie są tryby diagnostyczne |
| [Jak to działa](docs/how-it-works_pl.md) | Ścieżka sygnału, model i powód, dla którego pojedynczych dźwięków nie sądzi sam model |
| [Dane treningowe i wyniki](docs/training-data_pl.md) | Zbiór syntetyczny, GuitarSet i wartość każdej poprawki |
| [Własne formaty plików](docs/file-formats_pl.md) | Własne utwory i skale |
| [Uruchamianie](docs/running_pl.md) | Paczki i budowanie ze źródeł |

## Szczegółowe podsumowanie projektu

W `docs/` leży obszerne opracowanie całego systemu: architektura, cztery wady GuitarSetu i
wartość naprawienia każdej z nich, procedura treningu, pomiary stojące za każdą decyzją
projektową oraz hipotezy, które pomiar obalił. Ten sam dokument w dwóch językach, w Markdownie
i w PDF.

| plik | |
|---|---|
| [`docs/Solitito_project_summary_pl.md`](docs/Solitito_project_summary_pl.md) | [PDF](docs/Solitito_project_summary_pl.pdf) |
| [`docs/Solitito_project_summary_en.md`](docs/Solitito_project_summary_en.md) | [PDF](docs/Solitito_project_summary_en.pdf) |

## Repozytorium i zbiór danych

- Model i wagi DSP: <https://huggingface.co/greblus/solitito-ai>
- Zbiór danych v2 (rendery i adnotacje): <https://huggingface.co/datasets/greblus/solitito_dataset_v2>

Oba miejsca zawierają wyłącznie artefakty binarne. Cały kod jest tutaj.

Katalog `dist/` zawiera wszystko, czym zbudowano zbiór danych i wytrenowano model:

| plik | rola |
|---|---|
| `dataset_generator_v2.py` | generuje plik GP5 **i** adnotacje; sam sprawdza wszystkie akordy |
| `verify_annotations.py` | kontroluje, czy etykiety opisują dźwięk (samo numpy, bez librosy) |
| `model_trainer.py` | trening; działa na Kaggle, punkty kontrolne na Hugging Face |
| `gen_weights.py` | rzadkie wagi pseudo-CQT dla strony rustowej |
| `probe_root.py` | jak często oznaczona pryma jest faktycznie słyszalna |
| `probe_quality.py` | skąd powinna pochodzić jakość akordu: z głowicy czy z wektora wysokości |
| `probe_sources.py` | której adnotacji akordowej z GuitarSetu użyć |
| `inspect_jams.py` | co naprawdę znajduje się w plikach JAMS |
| `latency_material.py` | szarpnięcia o atakach znanych z konstrukcji, jako wzorzec |
| `latency_ground_truth.py` | ataki i wysokości prawdziwego nagrania, do tego samego pomiaru |
| `latency_stats.py` | jak późno aplikacja dowiaduje się, co zagrano, i jak często źle |
| `latency_rules.py` | ile kosztuje każda reguła zaliczania: zaliczenia niegrane, dźwięki pominięte |
| `gp5_to_arpeggio.py` | przekłada plik Guitar Pro na zapis stopniami, który czyta tryb Arpeggia |
| `hf_cleanup.py` | czyści repozytorium punktów kontrolnych przed przebiegiem od zera |

---

[1] Projekt korzysta ze zbioru GuitarSet autorstwa Qingyang Xi, Rachel M. Bittner, Johana
Pauwelsa, Xuzhou Ye i Juana P. Bello, dostępnego pod adresem <https://guitarset.weebly.com/>,
na licencji Creative Commons Attribution 4.0 International (CC BY 4.0).
