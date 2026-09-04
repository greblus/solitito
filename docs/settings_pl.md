# Ustawienia

Wszystkie opcje z czterech zakładek oraz sterowanie w oknie głównym.

[← powrót do README](../README_pl.md)

---

Cztery zakładki: **Dźwięk** — wejście i bramka szumów, **Ogólne** — jak surowo oceniane jest
to, co grasz, **Ćwiczenia** — co grać, oraz **Program** — co pokazuje okno.

<div align="center">
<img width="220" alt="Ustawienia, zakładka Dźwięk" src="solitito_settings2.png" />
<img width="220" alt="Ustawienia, zakładka Ogólne" src="solitito_settings1.png" />
<img width="220" alt="Ustawienia, zakładka Program" src="solitito_settings4.png" />
</div>

**Ćwiczenia** zawierają wyłącznie to, co ma sens dla trybu widocznego na ekranie — utwór nie ma
nic do powiedzenia w Formułach, formuła nic w Akordach — więc w każdym trybie jest to inna
zakładka, a trener gryfu, który nie ma własnych ustawień, nie pokazuje jej wcale:

<div align="center">
<img width="175" alt="Ćwiczenia, Akordy" src="solitito_settings3.png" />
<img width="175" alt="Ćwiczenia, Interwały" src="solitito_settings5.png" />
<img width="175" alt="Ćwiczenia, Skale" src="solitito_settings6.png" />
<img width="175" alt="Ćwiczenia, Arpeggia" src="solitito_settings7.png" />
<img width="175" alt="Ćwiczenia, Formuły" src="solitito_settings8.png" />
</div>

| Ustawienie | Opis |
|---|---|
| **Utwór / Skala** | Wybiera progresję albo skalę dla bieżącego trybu. Wybrany standard jest zapamiętywany — także między sesjami — i idzie za tobą przez wszystkie tryby, które go czytają: Akordy, Interwały, Arpeggia nad zmianami i Formuły nad standardem. Tak samo skala: Skale wracają do tej, na której skończyłeś, a nie na początek listy |
| **Studium** | Arpeggia, w ćwiczeniu ze studiami: którą frazę przechodzić. Przełącznik losowania znaczy tu **tonację**: nową po każdym przejściu. Arpeggio jest frazą, więc nie ma w nim czego tasować, a wybrane studium zostaje takie, jakie wskazałeś — generator jest osobną pozycją listy dla kogoś, kto chce też świeżej frazy. Najpierw trzy studia w jakościach akordu, potem łamane tercje i triole, dalej zwykłe przebiegi dwuoktawowe, a na końcu generator budujący świeżą frazę po każdym przejściu. Pole obok trzyma tonację |
| **Ćwiczenie** (arpeggia) | *Studium w tonacji* stoi na jednym akordzie — jego jakość ustawiasz niżej, tonację obok studium, a losowanie dobiera nową po każdym przejściu. *Ogrywanie zmian* bierze akordy z utworu i buduje po jednym arpeggiu na akord |
| **Kierunek** | Przy ogrywaniu zmian: wznoszące, opadające albo naprzemiennie z każdej ze stron. Opadające to kształt ze studiów — od prymy **w dół** przez składniki akordu, a nie fraza wznosząca czytana od tyłu |
| **Widok** | Trzy sposoby pokazania ćwiczenia: linia nazw stopni, tabulatura albo sama podstrunnica. Ustawienie jest osobne dla każdego trybu i zapisywane — skala to kolejność dźwięków i czyta się jako tabulatura, chwyt to kształt ręki i mówi coś dopiero na podstrunnicy. Domyślnie tabulatura w Skalach i Arpeggiach, podstrunnica w Interwałach i Gryfie. Fraza — skala albo arpeggio — rysowana jest na podstrunnicy, z białym pierścieniem wokół dźwięku, który jest teraz na kolei, i zielenią na każdym miejscu już zagranym. Fraza w górę i w dół wraca do własnych miejsc, więc na szczycie kształt jest zielony w całości, a o tym, gdzie się jest w drodze powrotnej, mówi pierścień — pierścień jest też na tabulaturze, bo przy skali rozdanej w losowej kolejności to jedyne, co mówi, w którym miejscu się jest; obrazek ma rozmiar ręki niezależnie od długości frazy, więc studium na trzydzieści dźwięków pozostaje czytelne. Skala rysowana jest w losowanej pozycji — kształt jest wszędzie ten sam, a rzecz w tym, żeby go znać tam, gdzie ręka akurat stanie. W Gryfie obrazkiem jest **region**: struny podpisane z lewej, te poza grą przygaszone, progi w grze, a po zagraniu właściwego dźwięku — każde miejsce wewnątrz regionu, gdzie on leży, na zielono. Wcześniej nie ma tam nic, a zły dźwięk nie jest rysowany w ogóle: szukanie miejsca jest właśnie ćwiczeniem. W kółku stoi stopień w pisowni ćwiczenia, więc skala zapisująca podwyższoną sekundę jako `#2` czyta się na gryfie jako `♯2`. W Interwałach zbiór rysowany jest jako **chwyt**: schemat jak przy akordach, struny w poprzek i progi w dół, z pozycją obok — pasek ma na osi poziomej kolejność dźwięków, a to nie mówi nic o tym, gdzie idą palce. Przy włączonym **Prowadź głosy** — domyślnie — każdy chwyt brany jest tam, gdzie palce mają najmniej do przejścia od poprzedniego, więc progresja sama przechodzi przez gryf. Wyłączone: każdy akord brany jest tam, gdzie gryf go daje, i tak poznaje się jeden kształt po całym gryfie, a nie w jednym jego rogu; losowanie też wyłącza prowadzenie. Przy włączonym losowaniu nie ma czego prowadzić: akordy idą w wylosowanej kolejności, a każdy chwyt brany jest tam, gdzie gryf go daje. To, na której strunie leży kółko, jest oktawą, więc apostrofy przestają być potrzebne; zagrane zapala się na zielono w obu widokach |
| **Numery progów zamiast stopni** | Tylko na tym rysunku. Pryma zachowuje swój kolor |
| **Tonacja** | Tylko Skale: tonika. Przy włączonej kolejności losowej losowana od nowa po każdym przejściu |
| **Interwały** | Które stopnie ćwiczyć. `1 3 5` dla triad, `1 3 5 7` dla akordów septymowych, `1 3` dla voicingów szkieletowych. `3` obejmuje tercję wielką i małą, `5` kwintę czystą i zmniejszoną, zależnie od jakości akordu |
| **Pokazuj predykcję AI w oknie głównym** | Wyświetla surową odpowiedź modelu na ekranie głównym |
| **Wejście** | Które urządzenie przechwytujące otworzyć. *Domyślne systemowe* podąża za ustawieniem systemu. Zapamiętywane, z odwrotem do domyślnego, gdy urządzenia zabraknie |
| **Kanał** | Na którym wejściu urządzenia słuchać — gitara w gnieździe 2 interfejsu to kanał 2. Pokazywany tylko wtedy, gdy urządzenie ma więcej niż jeden, i nie ma opcji miksowania: uśrednianie wejść wciąga to, co jest na drugim gnieździe, i kosztuje 6 dB |
| **Bramka szumów** | Próg w dBFS. Pasek poniżej pokazuje bieżący poziom wejścia w tej samej skali, z progiem zaznaczonym na czerwono — ustaw go tuż nad szumem przy nietkniętych strunach |
| **Podbicie basu** | Cyfrowe wzmocnienie najniższych prążków CQT. Przydatne przy mikrofonach laptopowych, które zwykle ścinają niskie struny |
| **Trzymaj jakość akordu do nowego ataku** | Utrzymuje rozpoznaną jakość, dopóki nie uderzysz strun ponownie. Bez tego trzymane `m7` zmienia się w `m`, gdy septyma wybrzmiewa |
| **Oceniaj krótkie szarpnięcia po ataku** | Dla akordów uderzanych i puszczanych, a nie trzymanych. Liczy się jeden czysty odczyt celu, a wybrzmiewanie po nim nie może go cofnąć. Zły akord nadal nie przechodzi |
| **Zaliczaj tylko to, co uderzone** | Tryby nutowe: dźwięk może zostać zaliczony tylko tam, gdzie głowica ataków usłyszała też uderzenie — łącznie z odczytem nazwy akordu, który wcześniej się przez to prześlizgiwał i zaliczał dowolny dźwięk, jeśli akurat był prymą tego, co model nazwał. Zmierzone na nagraniu pojedynczych dźwięków: dwie trzecie zaliczeń, które model przyznaje dźwiękowi innemu niż grany, nie niesie żadnego ataku — prawie zawsze jest to dźwięk poprzedni, wciąż brzmiący w jego oknie 0,77 s. Domyślnie wyłączone: dźwiękowi, którego atak głowica przeoczy, zostaje wtedy sama ścieżka CQT |
| **Graj dźwięki pojedynczo** | Do wyboru w Interwałach i w Gryfie. W Skalach i Arpeggiach obowiązuje zawsze, niezależnie od ustawienia — nikt nie uderza skali jak akordu — a Akordy i Formuły mają swoje własne reguły. Wyłączone: uderzony akord zalicza swoje interwały jeden po drugim — głowica wysokości jest polifoniczna i raportuje wszystkie dźwięki naraz. Włączone: liczy się wyłącznie estymata jednoklatkowa, i to dopiero gdy utrzyma się przez trzy klatki audio: zmierzone na 49 dźwiękach — cztery drogi zaliczenia razem dały 110 zaliczeń rzeczy niegranych, sama ustabilizowana estymata 33, i nie przegapiła żadnego dźwięku |
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
| **Graj dźwięki po kolei** (interwały) | Zbiór trzeba wziąć w kolejności rozdania, a ten dźwięk jest obrączkowany na chwycie. Wyłączone, i tak jest domyślnie: trzy dźwięki w dowolnej kolejności, z tasowaniem i bez — chwyt to trzy dźwięki pod jedną ręką i to, który palec spadnie pierwszy, nie jest ćwiczeniem. W Skalach i Arpeggiach opcji nie ma, bo tam kolejność **jest** ćwiczeniem |
| **Debug na konsoli** | Wypisuje linię dla każdej zaliczonej funkcji wraz z tym, co usłyszano. Okno na ocenianie; w Windowsie wydanie release nie ma konsoli, na której mogłoby to wypisać |
| **Kończ powtórzoną prymą** | Tylko Skale: przebieg czyta się 1 2 3 4 5 6 7 1, ostatnia oktawę wyżej. To osobny krok i trzeba go zagrać |
| **Pokazuj schematy akordów** | Miniatury diagramów pod nazwą akordu w trybie Akordy |
| **Schematy** | Dwa pola: pełne akordy i shell voicings — tercja i septyma nad prymą. Zaznaczone oba rysują oba, żadne nie rysuje nic. Powyżej czterech schematów rysowane są w dwóch rzędach. `m7b5` nie ma własnego shella: jego shell to co do dźwięku shell `m7`, bo różni je wyłącznie kwinta, więc to właśnie jest rysowane, z podpisem *substytut: shell m7*. Gdy na ekranie są same shelle, ten chwyt jest tym, o który program prosi, więc jego zagranie przechodzi na zielono; gdy rysowane są też pełne schematy, odczyt `m7` oznacza pominiętą obniżoną kwintę i zostaje żółty. Akord zmniejszony septymowy nie ma czego pominąć — cztery dźwięki co małą tercję, bez pary tercja-septyma, którą można by zachować — więc zostaje przy pełnych chwytach, z podpisem mówiącym, czym one też są: `7b9` bez prymy, o pół tonu poniżej każdego swojego dźwięku |
| **Tryb startowy** | W jakim trybie program się otwiera |
| **Język** | Auto (z ustawień systemu), polski, angielski. Stosowany natychmiast, bez restartu |
| **Pewność akordu** | Jak pewny musi być model *nazwy* akordu, żeby została zaliczona (tryb Akordy) |
| **Próg dźwięku** | Jak pewny musi być model, że *pojedynczy dźwięk* brzmi (Interwały / Skale / Arpeggia) |
| **Czas trzymania** | Jak długo poprawny akord musi być trzymany, zanim program przejdzie dalej. Dotyczy akordów: pojedynczy dźwięk zalicza się, gdy tylko zostanie rozpoznany, co w trybach nutowych i w Gryfie oznacza stałe 0,12 s |

Linia pod napisem `Channel` mówi, co faktycznie zostało otwarte — urządzenie, częstotliwość
próbkowania, liczbę kanałów i format próbek. `./solitito --help` wypisuje wszystkie opcje.
`./solitito --devices` podaje te same informacje dla każdego urządzenia widocznego dla
backendu, a `./solitito --bench` mierzy czas jednego przejścia modelu — program pyta model co
40 ms, dopóki akord brzmi, więc ta liczba jest w praktyce całym jego obciążeniem procesora.
Wydanie release ma podsystem `windows`, więc w Windowsie tryby te piszą do konsoli, z której
zostały uruchomione; uruchomione zupełnie bez konsoli — ze skrótu niosącego flagę — otwierają
własną i czekają na klawisz, żeby raport dało się przeczytać.

---

## Okno główne

Trzy elementy sterowania leżą poza panelem ustawień, w samym oknie.

**Pasek na dole** pokazuje akord właśnie zagrany, akord bieżący i następny. Akord zostawiony
za sobą zachowuje kolor, na jaki zasłużył — zielony przy trafieniu dokładnym, żółty gdy
przeszedł triadą albo zamianą — więc zaliczenie zostaje czytelne po tym, jak program poszedł
dalej.

**Losowanie** znaczy w każdym trybie co innego, bo w każdym co innego warto losować:

| tryb | co losuje |
|---|---|
| Akordy | kolejność standardu |
| Interwały | składniki wewnątrz akordu; kolejność akordów zostaje zapisana, chyba że włączysz **Losuj też akordy** |
| Skale | tonację, po każdym przejściu |
| Arpeggia | tonację w studiach, kolejność akordów przy ogrywaniu zmian |

W Skalach i Arpeggiach kolejność dźwięków nie jest losowana nigdy. Skala niegrana po kolei nie
jest tą skalą, a rozdane losowo studium nie jest tym studium.

Skala przesuwa się po każdym przejściu niezależnie od tego przełącznika: obchodzi trzy struny,
od których warto ją zaczynać, i za każdym razem zawraca — raz w górę, raz w dół. To jest
liczone, a nie losowane: trzy struny i dwa kierunki są względnie pierwsze, więc sześć przejść
pokrywa wszystkie sześć sposobów. Losowanie dawało pięć zejść z rzędu, co czyta się jak
zepsute ćwiczenie.

**Pauza** zatrzymuje progresję, a kolory dalej mówią, czy akord jest właściwy — można zostać
przy jednym chwycie i go opracować. Przy pauzie strzałki po bokach paska przechodzą tam i z
powrotem po progresji.
