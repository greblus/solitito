# Jak to działa

Ścieżka sygnału, model i powód, dla którego pojedynczych dźwięków nie sądzi sam model.

[← powrót do README](../README_pl.md)

### Dlaczego pojedynczych dźwięków nie sądzi sam model

Model pytany jest o 48 ramek, czyli 0,77 s dźwięku, i odpowiada o całości tego odcinka. Dla
trzymanego akordu jest to właściwe, dla gamy — nie: zmierzone na gamie granej po 0,6 s na
dźwięk, głowica wysokości nazwała dźwięk aktualnie grany w 7% okien, a poprzedni w 79%. Nic
się tam nie psuje — model raportuje oba dźwięki, które usłyszał, bo oba były w oknie.

Dlatego tryby nutowe zadają drugie pytanie pojedynczej ramce CQT, która nie ma pamięci: suma
harmoniczna po prążkach logarytmicznej magnitudy wskazuje klasę wysokości brzmiącą teraz. Na
tej samej gamie nazwała bieżący dźwięk w 57% przypadków i ani razu nie wskazała dźwięku,
którego nie zagrano. Opóźnienie, jakie miała, brało się z okna FFT o długości 8192 próbek,
szerokiego na pół sekundy, z każdym filtrem na jego środku: nowy dźwięk docierał do estymaty
o ćwierć sekundy za późno. Od 0.5.18 estymata tuż po ataku czyta filtrami dosuniętymi do końca
okna — patrz *Ucho na końcu okna* niżej.

Domyślnie ta estymata jedynie **dodaje** drogę zaliczenia, ponieważ przegłosowanie modelu
kosztowałoby coś, co warto zachować: głowica wysokości jest polifoniczna, więc uderzenie
całego akordu przechodzi jego interwały jeden po drugim, czego nie potrafi żaden tuner
monofoniczny. Opcja **Graj dźwięki pojedynczo** czyni z estymaty rozstrzygającą instancję —
wtedy okno modelu nie może zaliczyć dźwięku poprzedzającego ten pod palcami.

Cokolwiek jest wymagane dwa razy z rzędu — ten sam dźwięk dwukrotnie w arpeggiu, skala
kończąca się powtórzoną prymą, ten sam akord zapisany dwa razy w utworze — musi zostać
zagrane dwa razy. To, co wciąż brzmi z poprzedniego razu, pasuje w chwili, w której program
prosi o to ponownie, więc zaliczenie wymaga świeżego uderzenia. Dla dźwięku to odpowiedź
detektora uderzeń dla jego klasy, patrz *Powtórzenia* niżej. Akord zażądany ponownie wymaga
dowolnego uderzenia z tego detektora: uderzenie akordu zgłasza on jako jedną klasę, rzadko
dwie, i nie odzywa się, gdy akord tylko wybrzmiewa. Na 17 uderzeniach prawdziwej gitary
zauważył każde i żadnego z 17 wybrzmiewających akordów nie wziął za uderzenie; głowica ataków
modelu, która rozstrzygała to wcześniej, zauważyła 9 i odezwała się pod wszystkimi 17, gdy
wybrzmiewały.

Bez pliku detektora uderzeń odpowiada flux — atak na dowolnej strunie, bez względu na
wysokość — a wykrywacz obwiedni tam, gdzie flux nigdy się nie odezwał. Oba liczą uderzenia na
dowolnej strunie, więc w przebiegu z różnych dźwięków przesuwają się przy
każdym z nich: w `1 2 3 4 5 6 7 1` sześć dźwięków między prymami uchodziłoby za ponowne
uderzenie pierwszej prymy. Każdy dźwięk pamiętany jest osobno z pokrewnego powodu — pamięć o
jednym poprzednim zapomniałaby o pierwszej prymie na długo przed tym, nim przyjdzie pora na
ostatnią.

Dźwięk wymagany po raz drugi — zamykająca `1` w `1 2 3 4 5 6 7 1`, stopień oznaczony w polu
interwałów apostrofem, arpeggio wracające tam, gdzie się zaczęło — potrzebuje czegoś więcej
niż samego licznika uderzeń. Dawna głowica ataków modelu, która liczyła uderzenia przed
detektorem, rozlewała uderzenie na dźwięki, których nikt nie grał: gdy grano sześć stopni nad
prymą, pryma zbierała własne uderzenia, dwa na przebiegu testowym.

Rozstrzygają to dwie rzeczy. Estymata czyta wysokość bezwzględną, nie samą nazwę dźwięku,
więc dźwięk brzmiący o sześć albo więcej półtonów od miejsca, w którym czytała go przy
zaliczeniu, to inna zagrana struna — zamykająca pryma wobec wciąż brzmiącej otwierającej. To
dowód sam w sobie i nie potrzebuje ataku; na przebiegu testowym obie prymy odczytane zostały
o oktawę od siebie, 0,29 s po szarpnięciu struny. Nie może być natomiast wymogiem: przebieg
zamknięty w tej samej oktawie, w której się zaczął, nie spełniłby go nigdy, choćby zagrać go
pięć razy. W przeciwnym razie musi więc przesunąć się własny licznik uderzeń dźwięku, a tam,
gdzie gra się po jednym dźwięku, estymata nie może przy tym czytać innego dźwięku. Ta druga
połowa jest tym, czego licznik uderzeń sam nie dostarcza: wszystkie przypadkowe zapalenia
wypadają wtedy, gdy estymata czyta dźwięk faktycznie zagrany, więc przestają przechodzić.

Z opóźnienia głowicy wynika jeszcze jedno. Jej odpowiedź przychodzi 0,2 do 0,5 s po
uderzeniu struny, czyli *po* tym, jak estymata nazwała dźwięk i krok został na nim zaliczony
— więc to uderzenie dopiero nadejdzie, gdy następny krok poprosi o ten sam dźwięk, i
odpowie właśnie jemu. Dlatego zaliczenie przez pół sekundy nadąża za licznikiem swojego
dźwięku, dopóki estymata wciąż go czyta. Szarpnięcie nie przekaże własnego spóźnionego
uderzenia następnemu krokowi.

Detektor uderzeń spóźnia się tak samo, tylko mniej: odpowiada 22–32 ms po szarpnięciu, a od
0.5.18 ucho potrafi nazwać dźwięk po 25. Zgłoszenie detektora przychodziło wtedy po zaliczeniu
i uchodziło za nowe uderzenie: dźwięk, który tylko wybrzmiewał, zaliczał się ponownie 14 razy
na 51. Dlatego przez 0,2 s po zaliczeniu uderzenie jego klasy jest tym, za które zaliczono.
Nic prawdziwego na tym nie przepada — detektor i tak nie zgłasza klasy dwa razy w ciągu 0,6 s.
Wybrzmiewające dźwięki zaliczone ponownie: 0 z 51; uderzone ponownie po 0,8 i 1,2 s, zaliczone
w porę: 49 i 50 z 51, żaden za wcześnie.

Odbezpieczanie jest względne. Pod uderzonym i pozostawionym akordem odpowiedź głowicy dla
dźwięku nie opada do zera, lecz wisi — na mierzonym materiale między 0,11 a 0,29 przez całą
sekundę — więc stały próg nigdy by się nie odbezpieczył i kolejnego uderzenia nie dałoby się
w ogóle zobaczyć. Dźwięk jest odbezpieczony, gdy jego odpowiedź spadnie poniżej trzech
dziesiątych szczytu, który zaliczył poprzednie uderzenie.

### Sędzia widzi każdy krok

Interwały, Skale, Arpeggia i Gryf są sądzone na każdym kroku dźwięku — 16 ms — a nie wtedy,
gdy odpowie model. Skale, Arpeggia i Gryf czekały kiedyś na model, a po ciszy jego pierwsza
odpowiedź przychodzi dopiero po 0,38 s, gdy okno zapełni się do połowy.

Kroki nie giną też po drodze. Karta dźwiękowa oddaje próbki porcjami — na laptopie
użytkownika mniej więcej co 23 ms — a zegar okna czytał z każdej porcji tylko najnowszy krok:
36% kroków granych na żywo nie było w ogóle sądzonych. Od 0.5.18 wątek dźwięku odkłada odczyt
każdego kroku do kolejki, a sędzia przerabia je wszystkie przy każdym tyknięciu okna. Kroków
niewidzianych: zero. Liczy je `SOLITITO_HOPS=1`.

Część progów sędziego liczona jest w krokach — choćby to, jak długo ucho musi trzymać dźwięk —
i część z nich dobierano, próbując program na żywo, czyli na dwóch krokach z trzech. Na żywo
program może więc zaliczać odrobinę szybciej niż wcześniej.

### Zaliczanie w Interwałach

Interwały sprawdzają każdą nową ramkę audio, również gdy model nie ma dość sygnału, żeby
odpowiedzieć. Ponowne odczytanie tej samej ramki nie wydłuża
zaliczenia, a przerwa w dostarczaniu ramek zeruje rozpoczęte potwierdzenie. Odpowiedź
modelu wygasa po 250 ms bez aktualizacji.

Linia tekstowa i podstrunnica pokazują te same zaliczone stopnie, także poza kolejnością.
Po ostatnim dźwięku cały zestaw pozostaje zielony przez 350 ms, potem przychodzi następny
akord. Pauza zatrzymuje to przejście; cisza go nie zatrzymuje.

Wybrzmiewająca nuta nie zalicza swojego powtórzenia. Wyciszenie wejścia poniżej bramki
przez co najmniej 200 ms pozwala ponownie zaliczyć tę samą nutę, gdy estymator usłyszy ją
stabilnie — także jeśli żaden detektor nie złapał nowego szarpnięcia. Sam brak odczytu CQT
przy otwartej bramce nie jest takim wyciszeniem.

Tam, gdzie estymata jednoklatkowa nazywa klasę, żadna inna klasa nie może zaliczyć się
w tej klatce z odpowiedzi modelu. Obowiązuje to w obu kolejnościach; wcześniej działało
tylko przy dowolnej, a gra po kolei — czyli domyślna — nie miała nic, co trzymałoby tercję
zaliczaną z wybrzmiewającej prymy.

Wyjątkiem jest więcej niż jedna brzmiąca struna, i to jest **liczone**, a nie zgadywane —
patrz *Które dźwięki brzmią* niżej. Dwa głosy to już coś, czym jedna szarpnięta struna być
nie może. Wcześniej odpowiadała na to nazwa akordu i nie może znowu: jedna szarpnięta pryma
wystarcza modelowi do rozpoznania kształtu, i tak właśnie zaliczała się tercja, której nikt
nie dotknął.

### Powtórzenia: która struna została uderzona

Klasa raz zaliczona liczy się ponownie tylko wtedy, gdy od tego czasu została uderzona
**ta klasa** — nie jakaś struna i nie wtedy, gdy klasa po prostu wciąż brzmi. Wszystko inne
w aplikacji odpowiada na jedno z tych dwóch łatwiejszych pytań: strumień mówi, że uderzono
strunę, i jest ślepy na którą, a ucho i `voices` mówią, które klasy brzmią, i są ślepe na
to, czy właśnie je uderzono. Reguła powtórzeń potrzebuje obu naraz.

Daje to mały, przyczynowy model ataków `short_onset_masking_v2.onnx` — gałąź ataków
z `best_model_v2_take7_masking_v2.onnx`, wycięta z pliku połączonego tak, żeby nie liczyła
pnia akordowego: 1 MB i 0,7 ms na hop. Czyta dwa krótkie okna najnowszego dźwięku (64
i 128 ms), 35 ramek historii i odpowiada dwunastoma prawdopodobieństwami, po jednym na klasę.
Między nim a sędzią stoją dwie rzeczy:

- **Poziomowanie.** Jego cechy zależą od poziomu, więc ciche granie czyta jako słaby atak:
  na nagraniu użytkownika, 11 dB ciszej niż materiał, na którym był mierzony, znalazł 42
  z 87 nut. Wolne wzmocnienie — ok. czterech sekund na ustalenie — doprowadza grę do poziomu,
  który zna: 81 z 87.
- **Refrakcja 0,6 s na klasę.** Model odpala też na wygasających nutach, a wtedy energia
  sygnału spada — mediana 0,95 tego, co było, wobec 3,3 przy prawdziwych atakach. W ćwiczeniu
  klasa wraca dopiero po zaliczeniu, 0,35 s pokazu skończonego zestawu i odpowiedzi grającego,
  więc nic prawdziwego na tym nie przepada.
- **Sprawdzenie energii przy ponownym odpaleniu klasy w ciągu 3 s.** Refrakcja nie sięga
  dość daleko: na trzech nagraniach użytkownika model odpalił tę samą klasę ponownie w ciągu
  2 s 37 razy, w dwóch grupach bez niczego pomiędzy — 29 przy energii stojącej albo spadającej
  (0,89–1,03 tego, co było), czyli nuta wygasała, i 8 przy skoku 3,7–49 razy, czyli struna
  uderzona ponownie. Jedno z tych 29 trafiło akurat w chwilę, gdy aplikacja prosiła o tę
  klasę, 0,67 s po uderzeniu — i to było jedyne fałszywe powtórzenie z testu z gitarą. Takie
  odpalenie czeka więc 32 ms i liczy się tylko wtedy, gdy energia wzrosła o ćwierć. Czekanie
  jest istotne: bramka decydująca w chwili odpalenia odrzucała prawdziwe ponowne uderzenia,
  bo nowa nuta ledwie weszła wtedy w okno. Najpierw sprawdzenie sięgało 2 s, aż wybrzmiewające
  B♭ odpaliło ponownie 2,43 s po uderzeniu i zaliczyło się w następnym akordzie. Na 113 nutach
  użytkownika wybrzmiewających w akord, który też je zawierał, 3 s usunęły ten przypadek i nie
  zmieniły nic innego; 4 s nie dodały nic.

Zmierzone na każdej przesłuchanej nucie AtoA, sklejonej w nowe sygnały — sama nuta,
wybrzmiewająca, oraz ta sama nuta uderzona ponownie po 0,8 i 1,2 s, przy czym kostka gasi
starą drgającą strunę:

| | 0.5.7 | teraz |
| --- | --- | --- |
| powtórzenie dopuszczone, gdy nuta tylko brzmi | 24 / 51 | **0** / 51 |
| uderzona ponownie po 0,8 s, zaliczona w porę | 36 / 51, 11 za wcześnie | **48** / 51, żadne za wcześnie |
| uderzona ponownie po 1,2 s, zaliczona w porę | 35 / 51, 14 za wcześnie | **50** / 51, żadne za wcześnie |

Bez pliku modelu aplikacja dalej startuje i ocenia powtórzenia na starszym dowodzie,
z dwiema załatanymi dziurami: przeskok oktawy w estymacie liczy się jako nowe szarpnięcie
tylko z atakiem za nim — sama ta gałąź dawała 50 z 58 fałszywych powtórzeń powyżej. Taki
tryb zapasowy przepuszcza 13 wybrzmiewających nut z 51.

### Przenoszenie przez akord

To, co jeszcze brzmi, gdy ćwiczenie przechodzi do następnego akordu, liczy się tam jako już
użyte i potrzebuje własnego uderzenia. Sama reguła powtórzeń pilnowała tylko klas zaliczonych
wcześniej, więc nuta, która wybrzmiewała z poprzedniego akordu **bez** zaliczenia — zła,
dodatkowa — mogła odpowiedzieć następnemu akordowi, gdy tylko ucho ją nazwało: na nutach AtoA
47 razy na 51. Teraz ani razu. Wyjątkiem jest klasa uderzona, gdy skończony zestaw jeszcze
jest pokazywany: to grający, który sięga do następnego akordu z wyprzedzeniem, a nie resztka.

Wymaganie uderzenia przy każdym pierwszym zaliczeniu też by to zamknęło, ale kosztuje 7–11
z 87 nut sesji użytkownika, których detektor nie łapie; przenoszenie nie kosztuje nic tam,
gdzie nuta pada po żądaniu, czyli przy każdym zwykłym zaliczeniu.

Przy wyłączonej grze pojedynczo, wyłączonej stałej kolejności i wyłączonym „zaliczaj tylko
to, co uderzone" nie obowiązuje nic z tego: decyduje model i nic z nim nie dyskutuje,
łącznie z przenoszeniem. To jest wybór, nie przeoczenie — reguły są po to, żeby egzamin był
rzetelny, a bez nich tryb zalicza akordy i pojedyncze dźwięki tak, jak słyszy je model.

### Tylko to, co uderzone

Przy włączonym „zaliczaj tylko to, co uderzone" głowica wysokości modelu może zaliczyć klasę
tylko wtedy, gdy detektor uderzeń zgłosił jej uderzenie w ostatnich 22 krokach, czyli około
0,35 s. Estymaty z pojedynczej ramki to nie dotyczy. Zmierzone przez cały potok sędziego —
model akordowy co 40 ms, detektor co krok — wobec głowicy ataków modelu, która rozstrzygała
to wcześniej:

| | głowica ataków | detektor uderzeń |
| --- | --- | --- |
| AtoA, 51 pojedynczych dźwięków: zaliczone klasy, których nikt nie grał | 165 | **124** |
| ... w tym przez gałęzie modelu | 57 | **0** |
| 17 akordów uderzonych całą ręką: zaliczone składniki | **62** / 68 | 47 / 68 |
| ... klasy spoza akordu | 48 | **42** |
| 9 akordów granych arpeggiem: składniki / klasy spoza | 34 / 36, 30 | 32 / 36, **24** |

Żaden pojedynczy dźwięk nie przepadł i żaden nie został zaliczony później. Kosztuje akord
uderzony naraz: detektor zgłasza jedną klasę na uderzenie, więc model zalicza tę jedną,
a resztę musi nazwać ucho. Pozostałe 124 fałszywe zaliczenia przechodzą przez estymatę, której
ta opcja nie dotyczy. Bez pliku detektora zostaje tylko flux, który mówi, że uderzono
strunę, ale nie którą; dawnej głowicy ataków modelu, która rozstrzygała to wcześniej, od 0.5.10
już nie ma.

### Ucho tuż po uderzeniu

Estymata czyta ramkę pół sekundy, więc tuż po ataku słyszy naraz poprzedni dźwięk i nowy, i
przez chwilę nie potrafi nazwać żadnego: po C dźwięk D był czytany jako F przez 13 kroków —
0,2 s — zanim się ustalił. W Formułach jeden taki odczyt to znak do końca ćwiczenia; tak
zagranie prymy formuły od D zaliczało jej b3, F.

Dlatego przez 16 kroków, około 0,25 s, po uderzeniu zgłoszonym przez detektor ucho może
zaliczyć tylko klasę, którą detektor zgłosił jako uderzoną. Potem znów mu się wierzy, więc
dźwięk, którego detektor nie zgłosił, przychodzi później, a nie przepada.

| | wcześniej | teraz |
| --- | --- | --- |
| AtoA, akordy dozwolone: zaliczone klasy, których nikt nie grał | 124 | **62** |
| AtoA, pojedynczo | 82 | **36** |
| uderzone akordy, pojedynczo: klasy spoza akordu | 26 | **21** |
| arpeggia, akordy dozwolone: klasy spoza akordu | 24 | **17** |

Żaden dźwięk AtoA nie przepadł ani nie przyszedł później. Na czterech sesjach gry użytkownika,
388 zaliczeń, nie zginęło nic, co aplikacja zaliczała wcześniej. Uderzony akord oddał jeden
składnik z 68. 22 i 30 kroków nie dały nic ponad 16. Bez pliku detektora uchu wierzy się od
razu, jak dawniej.

Ucho myli też koniec dźwięku. Gdy nuta wygasa, estymata potrafi ustalić się na innej klasie —
kwarcie wyżej, czyli dźwięku, którego trzecią harmoniczną jest ta nuta — i nazywać ją stabilnie
od około 1,8 s. W Formułach znak nie wygasa, więc tam ucho zalicza tylko klasę, którą detektor
uderzeń zgłosił w ostatnich 3 s. Włączone we wszystkich trybach zmniejszyło fałszywe
zaliczenia na AtoA z 62 do 3, ale zgubiło 15 z 388 zaliczeń z sesji użytkownika — ciche nuty,
grane legato, niskie struny, których detektor nie widzi — i większość składników uderzonego
akordu, dlatego pozostałe tryby obywają się bez niego. W Formułach kosztuje to cichą albo
legato nutę, którą trzeba zagrać jeszcze raz kostką.

### Ucho na końcu okna

Estymata czyta CQT okna 8192 próbek, pół sekundy, a każdy z jej filtrów postawiony był na
środku tego okna. Filtr jest tak długi, jak potrzebuje jego dźwięk — 414 ms dla niskiego E,
104 ms dla wysokiego e, 16 ms dla harmonicznych wyżej — więc najnowsza próbka, jaką filtr w
ogóle widzi, leży 49–248 ms przed końcem okna. Estymata sumuje harmoniczne, które siedzą w
krótkich, wysokich filtrach: dokładnie tych, które patrzą najdalej wstecz. To była ta ćwierć
sekundy, której potrzebowała, żeby nazwać nowy dźwięk.

Od 0.5.18 ucho ma drugi zestaw filtrów, dosuniętych do końca okna. Przesunięcie filtra w czasie
to w widmie mnożenie przez rampę fazy, więc zestaw liczy się z tych samych wag przy starcie
programu: bez nowego pliku, a przycięcie wag zostaje takie samo. Model akordów czyta dalej
filtry ze środka — na nich był uczony.

Na sygnale syntetycznym ucho na końcu okna nazywa nowy dźwięk 24–96 ms po uderzeniu, wobec
256–272 ms. Przez całego sędziego, na liniach sklejonych z nut AtoA:

| | 0.5.17 | 0.5.18 |
| --- | --- | --- |
| linia po 0,40 s na dźwięk, zaliczone | 1 z 8 | **8** z 8 |
| skale Harrisa w dół, 0,30 s na dźwięk | 1 z 9 | **9** z 9 |
| zwykła droga, mediana od uderzenia do zaliczenia | około 400 ms | **około 150 ms** |
| AtoA, mediana do zaliczenia (akordy dozwolone / pojedynczo) | 156 / 220 ms | **39 / 49 ms** |

Pod koniec dźwięku te filtry mylą. Nie sięgają wstecz jednakowo daleko: niski trzyma jeszcze
to, co grano wcześniej, wysoki tylko ostatnie 50 ms, i gdy dźwięk wygasa, obie połówki składają
się w dźwięk, którego nikt nie zagrał — najczęściej kwartę wyżej, czyli ten, którego trzecią
harmoniczną jest zagrany. Używane cały czas podniosły na AtoA liczbę klas zaliczonych bez
zagrania z 62 i 37 do 77 i 52. Dlatego ucho czyta nimi tylko przez 16 kroków, 0,26 s, po ataku —
z fluxu albo z detektora uderzeń — a potem filtrami ze środka: 63 i 41. 12 i 20 kroków nazywały
linie tak samo szybko i zostawiały więcej fałszywych zaliczeń.

Dźwięk uderzony, gdy poprzedni jeszcze wybrzmiewa — arpeggio przez struny — nie jest głośniejszy
w żadnym zestawie filtrów: ucho dalej nazywa wybrzmiewający. Rozstrzygała to kiedyś głowica
wysokości modelu, a tam, gdzie gra się po jednym dźwięku, model nie jest już pytany. Teraz
rozstrzygają głosy: gdy ucho nazywa dźwięk uderzony *wcześniej* niż zadany albo dopiero zaczyna
nazywać zadany, ten przechodzi, jeśli detektor zgłosił jego uderzenie, a głosy go słyszą. Na
liniach z poprzednim dźwiękiem wybrzmiewającym 0,3 s zwykła droga przeszła z 8, 8, 1 przy 0,5 s
na dźwięk i 1, 4, 1 przy 0,3 s do 8, 8, 5 przy każdym tempie aż do 0,25 s.

Otwarte zostaje: z ćwiczeń Teda Greene'a przepuszczonych przez sędziego dwa przebiegi stają w
połowie, gdzie 0.5.17 nie zatrzymywało się wcale. Detektor nie zgłasza powtórzonego F, a reguła
dźwięku, który ucichł i wrócił, nie widzi, że ucichł.

### Szybkie zaliczanie

Do 0.5.17 zwykła droga zaliczała pojedynczy dźwięk około 0,4 s po uderzeniu i ten czas nie szedł
na decyzję, tylko na czekanie. Ucho czytało okno o długości pół sekundy i nazywało nowy dźwięk
dopiero wtedy, gdy przeważył on w tym oknie nad poprzednim, czyli około 0,26 s po uderzeniu;
potem odpowiedź musiała się utrzymać przez 0,12 s. Na 388 zaliczeniach z sesji użytkownika ucho
nazwało dźwięk 128 ms przed zaliczeniem — a detektor uderzeń, który odpowiada około 25 ms po
ataku, 350–370 ms przed nim. Linia pojedynczych dźwięków zaliczała się w całości przy pół
sekundy na nutę, a stawała przy 0,4 s: to mniej więcej dwa dźwięki na sekundę. Z uchem na końcu
okna zwykła droga potrzebuje około 0,15 s, a linia przechodzi przy 0,25 s na nutę — o ile każdy
dźwięk gasi poprzedni.

Dwa źródła są szybsze od ucha i żadne nie wystarcza samo:

- **Detektor uderzeń** nazywa uderzoną klasę 22–32 ms po uderzeniu, w 18–21 z 21 dźwięków linii
  przy każdym tempie, i prawie nigdy klasę, której nikt nie zagrał. Ale część uderzeń przegapia,
  a to, co zgłasza, mówi o ataku, nie o tym, co brzmi.
- **Szybkie ucho** czyta tylko najnowsze próbki. Pierwszy odczyt to YIN z tunera na ostatnich
  64 ms: w linii, w której każdy dźwięk gasi poprzedni, nazywa wszystkie 21 dźwięków przy każdym
  tempie aż do 0,15 s na nutę, 26–32 ms po uderzeniu. Nie jest uchem na dwie struny naraz — gdy
  poprzedni dźwięk dzwonił 0,3 s pod następnym, nazwał 14 z 21 przy 0,3 s na nutę — a na
  wygasającym dźwięku błądzi. Drugi odczyt to to, co przyrosło w widmie ostatnich 128 ms
  względem 128 ms przed nimi: struna, która tylko dzwoni, nie przyrosła, więc dźwięk właśnie
  uderzony jest w przyroście najsilniejszy, czy coś pod nim dzwoni, czy nie. Ten odczyt nazwał
  wszystkie 21 dźwięków przy każdym tempie, z wybrzmiewaniem i bez — a na trzymanym dźwięku
  dalej coś nazywa: na 51 dźwiękach AtoA z pełnym wybrzmieniem inną klasę przez trzy kroki lub
  dłużej 194 razy.

Przy włączonym **Szybkim zaliczaniu**, tam gdzie dźwięki grane są pojedynczo, dźwięk jest
zaliczany w chwili, gdy oba źródła się zgodzą: detektor zgłosił zadaną klasę w ostatnich
8 krokach (0,13 s), a szybkie ucho nazywa tę samą klasę, którymkolwiek odczytem, przez dwa
kroki z rzędu. Uderzenie, za które już zaliczono, nie liczy się drugi raz, więc klasę zadaną
dwa razy nadal trzeba uderzyć dwa razy. Wszystko inne zostaje zwykłej drodze, która zalicza
tak jak zawsze — dźwięk przegapiony przez detektor jest więc spóźniony, nie stracony.

Zmierzone przez całego sędziego w 0.5.18, na liniach sklejonych z nut AtoA i sądzonych tak, jak
sądzą Skale — D dorycka, G miksolidyjska, arpeggio D m7 — zaliczone dźwięki z 8, 8 i 5 oraz
mediana czasu od uderzenia do zaliczenia:

| sekund na nutę | zwykła droga | zwykła, poprzedni dźwięk dzwoni 0,3 s | szybka | szybka, poprzedni dźwięk dzwoni 0,3 s |
| --- | --- | --- | --- | --- |
| 0,50 | 8, 8, 5 — 150 ms | 8, 8, 5 — 142 ms | 8, 8, 5 — 26–38 ms | 8, 8, 5 — 30–42 ms |
| 0,40 | 8, 8, 5 | 8, 8, 5 | 8, 8, 5 | 8, 8, 5 |
| 0,30 i 0,25 | 8, 8, 5 | 8, 8, 5 | 8, 8, 5 | 8, 8, 5 |
| 0,20 | 7, 5, 3 | 8, 5, 2 | 7, 7, 4 | 8, 8, 5 |
| 0,15 | 2, 2, 2 | 2, 4, 2 | 7, 7, 4 | 7, 7, 4 |

W 0.5.17 zwykła droga zaliczała poniżej 0,5 s na nutę po 1 dźwięku z każdej linii, a przy 0,5 s
potrzebowała 400 ms na każdy; z dzwoniącym poprzednim dźwiękiem poniżej 0,5 s — po 1 z każdej.

Dźwięki wciąż gubione przy 0,2 s i szybciej to uderzenia, których detektor nie zgłosił; linia
jest sądzona po kolei, więc czeka tam na zwykłą drogę, a ta przy takim tempie nie zdąża. Na
sesjach użytkownika detektor i szybkie ucho zgodziły się przed zaliczeniem w 366 z 388
dźwięków; pozostałe 22 zaliczane są jak dotąd.

Druga strona, na 51 pojedynczych dźwiękach AtoA, każdy z pełnym wybrzmieniem: oba źródła
zgodziły się na zagranym dźwięku 44 razy, z medianą 33 ms po uderzeniu, a na klasie, której
nikt nie zagrał, dwa razy — pół tonu poniżej dźwięku, w samym ataku. Zwykłe ucho, według
własnej reguły i samo, na tym samym nagraniu nazwało inną klasę przez trzy kroki lub dłużej
150 razy.

To opcja, domyślnie włączona od wersji 0.5.17. Wszystko to jest zmierzone na nagraniach, a wejście
na żywo dokłada to, czego nagranie nie ma — przede wszystkim bufor karty dźwiękowej; od 0.5.18
sędzia widzi na żywo każdy krok, tak jak na nagraniu. Gdyby na żywo zaliczała dźwięki, których
nie grasz, wyłącz ją w szybkich opcjach: zostaje wtedy sama zwykła droga.

### Bramka szumu

Bramka rozstrzyga, czy okno to gra, czy cisza; tego, co nazwie ciszą, nikt nie sądzi, a cisza
przez 200 ms pozwala zaliczyć dźwięk ponownie. Kiedyś porównywała z progiem poziom całego
półsekundowego okna, więc otwierała się późno na cichym ataku, zamykała późno po nim, a dźwięk
wygasający w okolicy progu wypadał w połowie wybrzmienia.

Od 0.5.18 liczy się poziom najnowszych 1024 próbek, 64 ms, a bramka ma dwie strony: otwiera się
na progu, a zamyka dopiero, gdy poziom przez 10 kroków, 160 ms, był 6 dB poniżej niego. Dźwięk
wygasający w okolicy progu już nie miga. Z AtoA zagranym 6 dB ciszej, przy progu programu
−34 dB:

| | stara bramka | nowa bramka |
| --- | --- | --- |
| G miksolidyjska, 0,50 s na dźwięk | 5 z 8 | **8** z 8 |
| D dorycka, 0,40 s na dźwięk | 1 z 8 | **8** z 8 |
| arpeggio D m7, 0,40 s na dźwięk | 1 z 5 | **5** z 5 |

Przy zwykłym poziomie obie bramki zaliczają to samo.

### Które dźwięki brzmią

Widmo jest **tłumaczone**, nie rankowane. Bierze się najmocniejszego kandydata, odejmuje
serię partiali, którą przewiduje, i pyta resztkę, co zostało: klasa w całości wytłumaczona
jako czyjaś harmoniczna nie zostawia nic i nie jest głosem. Tego rankowanie nie rozstrzygnie,
bo trzecia harmoniczna prymy leży w klasie kwinty, a piąta w klasie tercji wielkiej.

Kandydat musi też nieść energię tam, gdzie leżałaby jego własna podstawa — to zatrzymuje
kandydata *poniżej* brzmiącej nuty, który punktuje na pożyczonych partialach.

Zmierzone na nagraniu 51 przesłuchanych pojedynczych dźwięków: dokładnie jeden głos 46 razy
i ani jednego dodatkowego głosu na interwale harmonicznym; na nagraniu, gdzie dźwięki padają
co pół sekundy i nachodzą na siebie — dwa głosy 38 razy na 87. Dwie ślepe plamki są znane
i zapisane w testach: **oktawy nie da się rozstrzygnąć** w ogóle — jej podstawa leży dokładnie
na drugim partialu dźwięku niższego, więc żaden model harmoniczny ich nie rozdzieli — a kwinta
znacznie cichsza od dźwięku pod nią jest gubiona. Dlatego wyjątek **liczy** głosy, zamiast
szukać na liście jednej konkretnej klasy.

---

## Jak to działa

### Ścieżka sygnału

```
wejście audio → przepróbkowanie do 16 kHz → FFT (8192) → rzadki pseudo-CQT ─┬─ filtry na środku → cechy → model akordów
                                                                           └─ filtry na końcu → ucho i głosy
                 najnowsze próbki → detektor uderzeń, szybkie ucho, bramka szumu
```

1. **Przepróbkowanie.** Wejście sprowadzane jest do 16 kHz. CQT obejmuje 6 oktaw od C1, więc
   najwyższy prążek leży w okolicach 2 kHz — daleko poniżej granicy Nyquista, czyli 8 kHz.
2. **Pseudo-CQT.** Zamiast prawdziwej transformaty o stałej dobroci program mnoży widmo FFT
   przez wyliczone wcześniej jądro (144 prążki, 24 na oktawę — rozdzielczość ćwierćtonowa).
   Jądro pochodzi z `librosa.filters.constant_q`, więc program i trener wytwarzają te same
   cechy.
3. **Cechy.** 168 wartości na ramkę: 144 prążki CQT, 12 chromy i 12 prążków energii basu.
   Model widzi 48 ramek historii (0,77 s przy skoku 256 próbek).
4. **Wnioskowanie.** Jedno przejście w przód co 40 ms tam, gdzie model akordów rozstrzyga,
   i żadne tam, gdzie nie — patrz *Kiedy pytany jest model akordów* niżej.
5. **Każdy krok.** Każdy krok 256 próbek, 16 ms, czytany jest raz: CQT, ucho, detektor uderzeń
   i bramka. Odczyty czekają na sędziego w kolejce, więc żaden nie przepada, gdy okno
   odrysuje się z opóźnieniem.

Jądro CQT przechowywane jest w **rzadkim formacie CSR**. Pełne jądro ma 4097×144 = 589 968
wag, ale skupiają się one wokół częstotliwości środkowej każdego prążka. Odrzucenie
wszystkiego poniżej 1e-4 wartości szczytowej zachowuje 6,9% wag i zmienia wynik o 0,03%
szczytu (mierzone na szumie białym, szumie różowym i szeregu harmonicznym o charakterze
gitarowym). Plik wag kurczy się z 28 MB do 2 MB, a wątek audio wykonuje około czternastu razy
mniej mnożeń na ramkę.

### Model

Hybryda CNN i Transformera z trzema głowicami wyjściowymi, `best_model_v2_take6_chords.onnx`:

| Etap | Szczegół |
|---|---|
| Wejście | `[48 ramek, 168 cech]` |
| CNN | Bloki splotowe z Squeeze-and-Excitation, InstanceNorm |
| Enkoder | Enkoder transformerowy z tokenem CLS, 384 wymiary |
| `root_logits` | 13 klas — 12 klas wysokości i „Noise" |
| `quality_logits` | 11 klas — maj, min, maj7, dom7, min7, m7b5, dim7, aug, sus, note, N |
| `pitch_logits` | 12 wyjść sigmoidalnych — które klasy wysokości brzmią |

Głowice odpowiadają na różne pytania i **nie** są wymienne:

- `pitch_logits` to najmocniejsze wyjście (F1 0,909). Odpowiada na pytanie „które dźwięki
  brzmią w tej chwili", czyli dokładnie na to, czego potrzebują tryby Interwały, Skale i
  Arpeggia.
- `root_logits` nazywa centrum tonalne. 98,1%.
- `quality_logits` nazywa rodzinę akordu. To jest ta trudna głowica.

Żadna z nich nie mówi, co zostało *uderzone*, w odróżnieniu od tego, co brzmi. To daje
osobny detektor uderzeń, `short_onset_masking_v2.onnx`, opisany w części o powtórzeniach.
Model akordowy miał do tego kiedyś czwartą głowicę, `onset_logits`, trenowaną w osobnej
fazie przy zamrożonej reszcie sieci; detektor robi to lepiej i od 0.5.10 tej głowicy nie ma
w pliku, który ładuje aplikacja. Trzy wyjścia powyżej są co do bitu takie same jak z nią.
Starszy trójgłowicowy model, `best_model_v2_take6.onnx`, wciąż ładuje się jako zapas, ale to
wcześniejszy eksport i niektóre akordy nazywa inaczej.

### Septyma wielka czy mała

Głowica jakości myli obie septymy; głowica wysokości nie. Na nagraniu użytkownika z CMaj7
uderzonym pięć razy model nazywał go niemal cały czas C7, z pewnością 0,8–0,9, a głowica
wysokości słyszała C, E i H — septymę wielką — i żadnego B; nie przeszło ani jedno z pięciu.
Dlatego w Akordach akord nazwany z dobrą prymą i drugą septymą przechodzi, gdy głowica wysokości
słyszy prymę, tercję i zadaną septymę powyżej połowy, a drugą septymę poniżej. Na 17 uderzonych
akordach użytkownika: przechodzi 17, wcześniej 13, a wszystkie porażki to CMaj7. Na 9 akordach
granych arpeggiem nie zmieniło się nic poza CMaj7, który przechodzi po 1,5 s zamiast 3,7.

### Kiedy pytany jest model akordów

Model kosztuje niemal wszystko, co kosztuje program: jedno przejście trwa 38,6 ms na małym
laptopie użytkownika (Ryzen 3 5300U) i 69 ms na jego cienkim kliencie, więc pytany co 40 ms
zajmuje rdzeń i więcej. Tak często pytany jest tylko tam, gdzie jego odpowiedź rozstrzyga: w
Akordach oraz w Interwałach i Formułach, gdy akordy są dozwolone. Tam, gdzie gra się po jednym
dźwięku — Skale, Arpeggia, Gryf, Interwały i Formuły z *Pojedynczo* — nie jest pytany wcale i
rozstrzyga ucho.

To jest zmierzone, nie założone: z 388 zaliczeń w czterech sesjach Interwałów użytkownika 388
było z ucha, żadne z modelu; a cała tablica wyników, z modelem pytanym w tych trybach co 200 ms
albo wcale, wyszła identycznie co do cyfry. Na laptopie, AtoA odtwarzane przez `--file`:

| tryb | 0.5.17 | 0.5.18 |
| --- | --- | --- |
| Skale | 113% rdzenia | **11%** |
| Arpeggia | 111% | **11%** |
| Akordy | 111% | 108% |

---
