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
którego nie zagrano. Pozostałe opóźnienie bierze się z okna FFT o długości 8192 próbek,
szerokiego na pół sekundy — i to jest zarazem powód, dla którego dźwięki krótsze niż mniej
więcej 0,4 s pozostają trudne.

Domyślnie ta estymata jedynie **dodaje** drogę zaliczenia, ponieważ przegłosowanie modelu
kosztowałoby coś, co warto zachować: głowica wysokości jest polifoniczna, więc uderzenie
całego akordu przechodzi jego interwały jeden po drugim, czego nie potrafi żaden tuner
monofoniczny. Opcja **Graj dźwięki pojedynczo** czyni z estymaty rozstrzygającą instancję —
wtedy okno modelu nie może zaliczyć dźwięku poprzedzającego ten pod palcami.

Cokolwiek jest wymagane dwa razy z rzędu — ten sam dźwięk dwukrotnie w arpeggiu, skala
kończąca się powtórzoną prymą, ten sam akord zapisany dwa razy w utworze — musi zostać
zagrane dwa razy. To, co wciąż brzmi z poprzedniego razu, pasuje w chwili, w której program
prosi o to ponownie, więc zaliczenie wymaga świeżego uderzenia: odpowiedź głowicy ataków dla
danego dźwięku musi przekroczyć 0,60. Akord wymaga dwóch takich uderzeń na własnych
dźwiękach — zmierzone: pojedyncze potrafi odezwać się samo pod akordem, który tylko
wybrzmiewa, dwa nie zdarzyły się ani razu.

Wykrywacz obwiedni odpowiada na to pytanie wyłącznie w modelu, który nie ma głowicy ataków.
Liczy on uderzenia na dowolnej strunie, więc w przebiegu z różnych dźwięków przesuwa się przy
każdym z nich: w `1 2 3 4 5 6 7 1` sześć dźwięków między prymami uchodziłoby za ponowne
uderzenie pierwszej prymy. Każdy dźwięk pamiętany jest osobno z pokrewnego powodu — pamięć o
jednym poprzednim zapomniałaby o pierwszej prymie na długo przed tym, nim przyjdzie pora na
ostatnią.

Dźwięk wymagany po raz drugi — zamykająca `1` w `1 2 3 4 5 6 7 1`, stopień oznaczony w polu
interwałów apostrofem, arpeggio wracające tam, gdzie się zaczęło — potrzebuje czegoś więcej
niż samej głowicy ataków, bo głowica rozlewa uderzenie na dźwięki, których nikt nie grał. Gdy
grane jest sześć stopni nad prymą, pryma zbiera własne uderzenia: dwa na przebiegu testowym,
a w szybkim przebiegu głowica nie dała dla zamykającej prymy ani jednego.

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

Odbezpieczanie jest względne. Pod uderzonym i pozostawionym akordem odpowiedź głowicy dla
dźwięku nie opada do zera, lecz wisi — na mierzonym materiale między 0,11 a 0,29 przez całą
sekundę — więc stały próg nigdy by się nie odbezpieczył i kolejnego uderzenia nie dałoby się
w ogóle zobaczyć. Dźwięk jest odbezpieczony, gdy jego odpowiedź spadnie poniżej trzech
dziesiątych szczytu, który zaliczył poprzednie uderzenie.

### Zaliczanie w Interwałach

Interwały sprawdzają każdą nową ramkę audio odczytaną przez interfejs, również gdy model
nie ma dość sygnału, żeby odpowiedzieć. Ponowne odczytanie tej samej ramki nie wydłuża
zaliczenia, a przerwa w dostarczaniu ramek zeruje rozpoczęte potwierdzenie. Odpowiedź
modelu wygasa po 250 ms bez aktualizacji.

Linia tekstowa i podstrunnica pokazują te same zaliczone stopnie, także poza kolejnością.
Po ostatnim dźwięku cały zestaw pozostaje zielony przez 350 ms, potem przychodzi następny
akord. Pauza zatrzymuje to przejście; cisza go nie zatrzymuje.

Wybrzmiewająca nuta nie zalicza swojego powtórzenia. Wyciszenie wejścia poniżej bramki
przez co najmniej 200 ms pozwala ponownie zaliczyć tę samą nutę, gdy estymator usłyszy ją
stabilnie — także jeśli głowica ataków przeoczyła nowe szarpnięcie. Sam brak odczytu CQT
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
- **Sprawdzenie energii przy ponownym odpaleniu klasy w ciągu 2 s.** Refrakcja nie sięga
  dość daleko: na trzech nagraniach użytkownika model odpalił tę samą klasę ponownie w ciągu
  2 s 37 razy, w dwóch grupach bez niczego pomiędzy — 29 przy energii stojącej albo spadającej
  (0,89–1,03 tego, co było), czyli nuta wygasała, i 8 przy skoku 3,7–49 razy, czyli struna
  uderzona ponownie. Jedno z tych 29 trafiło akurat w chwilę, gdy aplikacja prosiła o tę
  klasę, 0,67 s po uderzeniu — i to było jedyne fałszywe powtórzenie z testu z gitarą. Takie
  odpalenie czeka więc 32 ms i liczy się tylko wtedy, gdy energia wzrosła o ćwierć. Czekanie
  jest istotne: bramka decydująca w chwili odpalenia odrzucała prawdziwe ponowne uderzenia,
  bo nowa nuta ledwie weszła wtedy w okno.

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
rzetelny, a tryb jest też czymś, na czym można pograć akordami dla przyjemności.

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
wejście audio → przepróbkowanie do 16 kHz → FFT (8192) → rzadki pseudo-CQT → cechy → model ONNX
```

1. **Przepróbkowanie.** Wejście sprowadzane jest do 16 kHz. CQT obejmuje 6 oktaw od C1, więc
   najwyższy prążek leży w okolicach 2 kHz — daleko poniżej granicy Nyquista, czyli 8 kHz.
2. **Pseudo-CQT.** Zamiast prawdziwej transformaty o stałej dobroci program mnoży widmo FFT
   przez wyliczone wcześniej jądro (144 prążki, 24 na oktawę — rozdzielczość ćwierćtonowa).
   Jądro pochodzi z `librosa.filters.constant_q`, więc program i trener wytwarzają te same
   cechy.
3. **Cechy.** 168 wartości na ramkę: 144 prążki CQT, 12 chromy i 12 prążków energii basu.
   Model widzi 48 ramek historii (0,77 s przy skoku 256 próbek).
4. **Wnioskowanie.** Jedno przejście w przód co 40 ms.

Jądro CQT przechowywane jest w **rzadkim formacie CSR**. Pełne jądro ma 4097×144 = 589 968
wag, ale skupiają się one wokół częstotliwości środkowej każdego prążka. Odrzucenie
wszystkiego poniżej 1e-4 wartości szczytowej zachowuje 6,9% wag i zmienia wynik o 0,03%
szczytu (mierzone na szumie białym, szumie różowym i szeregu harmonicznym o charakterze
gitarowym). Plik wag kurczy się z 28 MB do 2 MB, a wątek audio wykonuje około czternastu razy
mniej mnożeń na ramkę.

### Model

Hybryda CNN i Transformera z czterema głowicami wyjściowymi:

| Etap | Szczegół |
|---|---|
| Wejście | `[48 ramek, 168 cech]` |
| CNN | Bloki splotowe z Squeeze-and-Excitation, InstanceNorm |
| Enkoder | Enkoder transformerowy z tokenem CLS, 384 wymiary |
| `root_logits` | 13 klas — 12 klas wysokości i „Noise" |
| `quality_logits` | 11 klas — maj, min, maj7, dom7, min7, m7b5, dim7, aug, sus, note, N |
| `pitch_logits` | 12 wyjść sigmoidalnych — które klasy wysokości brzmią |
| `onset_logits` | 12 wyjść sigmoidalnych — które klasy wysokości zostały UDERZONE w ostatnich 6 ramkach |

Głowice odpowiadają na różne pytania i **nie** są wymienne:

- `pitch_logits` to najmocniejsze wyjście (F1 0,909). Odpowiada na pytanie „które dźwięki
  brzmią w tej chwili", czyli dokładnie na to, czego potrzebują tryby Interwały, Skale i
  Arpeggia.
- `root_logits` nazywa centrum tonalne. 98,1%.
- `quality_logits` nazywa rodzinę akordu. To jest ta trudna głowica.
- `onset_logits` jest najnowsza i odpowiada na pytanie, którego trzy pozostałe nie stawiają:
  nie co brzmi, lecz co zostało *uderzone*. Samo brzmienie nie wystarcza — struna
  rezonująca współczująco brzmi, brzmi też dźwięk poprzedni — a najbardziej waży to w
  Formułach, gdzie zaliczenie nigdy nie wygasa. Trenowana była osobno, przy zamrożonej
  reszcie sieci, więc trzy powyższe głowice są co do bitu tym, czym były. Mierzona na
  prawdziwym nagraniu okazała się najszybszą odpowiedzią w programie (202 ms po uderzeniu
  wobec 676 ms), ale rozmywa atak na sąsiednie struny, więc nie rozstrzyga o tym, *co*
  zostało zagrane. Rozstrzyga natomiast, czy coś zostało uderzone **ponownie**: dźwięk albo
  akord wymagany dwa razy z rzędu potrzebuje własnego uderzenia, a wykrywacz obwiedni nie
  potrafi go dostarczyć — jego poziom to RMS okna 512 ms, więc drugie szarpnięcie brzmiącej
  struny prawie go nie podnosi. Zmierzone na materiale generowanym: obwiednia złapała 2
  powtórzenia z 6 i 2 powtórzone uderzenia akordu z 6, głowica wszystkie sześć z każdych,
  nie odzywając się ani razu, gdy akord tylko wybrzmiewał. Starszy, trójgłowicowy model
  wciąż działa: nazwy trzech pierwszych wyjść się nie zmieniły.

---
