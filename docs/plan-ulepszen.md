# Plan ulepszeń Solitito

Stan na 15.11.2026, wersja 0.5.17 (`f4da3c9`). Dokument jest planem pracy dla modelu, który
będzie kodował. Nic z niego nie jest jeszcze zrobione.

Plan dzieli się na pięć grup: **S** silnik, **C** ćwiczenia, **U** interfejs, **K** kod,
**I** infrastruktura. Na końcu jest kolejność i lista decyzji, które należą do autora programu.

## 0. Zasady pracy

Obowiązuje `AGENTS.md`. Do tego, dla całego planu:

1. **Najpierw pomiar.** Żadna zmiana reguły ani modelu nie wchodzi bez liczby „przed” i „po”
   na tablicy wyników z punktu S0. Wyniki dopisywać do `docs/judge-measurements.md`.
2. **Praca na nowej gałęzi** `engine-v2`, odchodzącej od `solitito-ai`. Gałąź `experiments`
   (`a4d8d8b`) trzyma harness chwytów i generator hex — przenieść je na `engine-v2` na początku.
3. **Stary silnik zostaje do końca.** Nowy wchodzi obok, za przełącznikiem w ustawieniach, i
   przejmuje tryb po trybie dopiero wtedy, gdy wygrywa na tablicy wyników.
4. **Małe commity, jeden temat.** Styl opisów: temat i lista, do ośmiu linii, bez stopek.
5. **Co jest hipotezą, jest tak oznaczone.** W tym dokumencie „(hipoteza)” znaczy: wynika z
   czytania kodu, nie z pomiaru. Hipotezę najpierw się mierzy.
6. **Nowe zależności** tylko te wymienione tutaj z uzasadnieniem; każda inna wymaga zgody.

## 1. Diagnoza

### 1.1 Jak działa silnik dziś

Dźwięk idzie przez `src/audio.rs`: przepróbkowanie do 16 kHz, okno 8192 próbek (512 ms), skok
256 (16 ms). Z każdego skoku powstaje kilka niezależnych odczytów:

| odczyt | plik | co mówi | czego nie umie |
|---|---|---|---|
| model akordów, 7,3 mln parametrów, co 40 ms | `brain.rs`, `dist/model_trainer.py` | pryma, jakość (11 klas), 12 klas dźwięków w oknie 0,77 s | odpowiada o całym oknie, więc głównie o dźwięku poprzednim; 36,8 ms na przebieg na jednym wątku |
| „ucho”, jedna ramka CQT | `audio::mono_pitch` | jedna klasa dźwięku teraz | jeden dźwięk naraz; nazywa nowy dźwięk ok. 260 ms po uderzeniu |
| głosy | `voices.rs` | które klasy brzmią (do 4) | oktawa, cicha kwinta pod głośną prymą |
| detektor uderzeń, 1 MB, 0,7 ms na skok | `strike.rs`, `dist/strike_trainer.py` | którą klasę uderzono | zgłasza jedną klasę na uderzenie całego akordu; gubi ciche, legato i niskie struny |
| flux, obwiednia | `flux.rs`, `audio::AttackDetector` | że uderzono cokolwiek | co |
| szybkie ucho | `audio::QuickEar`, `rise.rs` | klasa z ostatnich 64 i 128 ms | dwie struny naraz; na wybrzmiewaniu błądzi |

Sędzia (`src/state/judge.rs`, `heard.rs`, `credits.rs`, `notes_mode.rs`, `chords_mode.rs`)
składa te odczyty regułami. Reguł jest kilkadziesiąt, każda ma zmierzoną stałą
(`NOTE_HOLD`, `STRUCK_FRESH`, `CLASS_FRESH`, `EAR_SETTLE`, `EAR_BACKED`, `QUICK_AGREE`, …) i
każda zamyka jeden przeciek zostawiony przez pozostałe.

### 1.2 Co z tego wynika

**Wszystko jest liczone na dwunastu klasach dźwięków.** Program nie zna oktawy, struny ani
progu. Nie potrafi więc sprawdzić pozycji, a powtórzony dźwięk musi rozpoznawać okrężnie.

**Żaden odczyt nie odpowiada na pytanie, którego sędzia potrzebuje**: „jaki dźwięk i kiedy
został uderzony, a co tylko brzmi”. Detektor uderzeń jest najbliżej, ale pracuje na klasach i
zgłasza jedną klasę na akord. To jest przyczyna otwartego problemu z `interwaly-akord-naraz`:
na 17 akordach uderzonych naraz zaliczanych jest 26 z 68 dźwięków przy „Tylko uderzone” i 47 z
68 bez tej opcji; cicha kwinta przy −12 dB — 10 z 51. Osiem prób naprawy regułami (E1–E7 w
`docs/judge-measurements.md`, „A chord struck at once”) i dwa douczania detektora nie dały drogi
naprzód.

**Zwykła ścieżka zalicza pojedynczy dźwięk po ok. 0,4 s.** „Szybkie zaliczanie” schodzi do
26–46 ms, ale tylko dla dźwięków branych pojedynczo i tylko wtedy, gdy detektor zgłosi
uderzenie.

**Akord jest klasyfikowany jedną z 11 nazw.** `m6`, `mMaj7`, dziewiątki, trzynastki, akordy
alterowane i przewroty nie mają swojej nazwy; `chords_mode.rs` zalicza je przez substytuty.

### 1.3 Konkretne ustalenia z kodu

Każde jest do sprawdzenia pomiarem, zanim cokolwiek się zmieni.

1. **Jądra CQT są wyśrodkowane w oknie 512 ms** (`dist/gen_weights.py`,
   `off = (RUST_FFT_SIZE - len(fk)) // 2`). Filtr dla E4 ma ok. 1650 próbek (103 ms), ale stoi
   na środku okna, więc patrzy na dźwięk sprzed 256 ms. Ucho nazywa dźwięk „ok. 0,26 s po
   uderzeniu” (`docs/how-it-works.md`) — to niemal dokładnie to przesunięcie. (hipoteza)
   Opóźnienie ucha jest więc w dużej części skutkiem wyśrodkowania, a nie długości okna.
2. **Na całe okno nakładane jest dodatkowo okno Hanna** (`CqtAnalyzer::windowed_fft`), choć
   jądra mają już własne. Trener liczy cechy przez `librosa.cqt`, bez tego kroku. Dla krótkich
   jąder różnica jest mała, dla długich (niskie struny) nie. (hipoteza)
3. **Normalizacja cech modelu akordów różni się między trenerem a programem.** Trener:
   `librosa.amplitude_to_db(cqt_abs, ref=np.max)` — względem maksimum całego pliku. Program:
   `audio::log_normalise` — względem maksimum każdej ramki. Cicha ramka wybrzmiewania wygląda
   w programie jak głośna. (hipoteza co do skutków; sama różnica jest w kodzie)
4. **Cechy modelu akordów nie mają testu zgodności z trenerem.** Cechy detektora uderzeń mają
   (`features_are_the_trainers_to_the_last_bit`, `dist/fixtures/`). Punkty 2 i 3 mogły przez to
   przetrwać niezauważone.
5. **Model akordów był uczony tylko na oknach leżących w całości w trzymanym akordzie**
   (komentarz przy `AudioAnalysis::history_fill`). Okno z atakiem jest dla niego spoza
   rozkładu, stąd reguła `MIN_FILL` i wrażenie, że akord „rozpoznaje się w ogonie”.
6. **Przepróbkowanie jest interpolacją liniową bez filtra** (`audio::Resampler`). Trener
   detektora uderzeń robi to samo celowo (`onset_resample`), więc tam się zgadza. Trener modelu
   akordów używa `librosa.load`, czyli filtra. Przy 48 kHz na wejściu treść powyżej 8 kHz
   zawija się w pasmo.
7. **Bramka szumu liczy RMS z całego okna 512 ms** (`Listener::hear`). Otwiera się i zamyka
   wolno, a notatki z testów mówią, że jest realnym hamulcem przy cichej grze
   (`bramka-szumu-a-zaliczanie`).
8. **Całe DSP działa w funkcji zwrotnej karty dźwiękowej**: FFT 8192, dwa krótkie FFT, model
   uderzeń, YIN, przydziały pamięci (`Vec`) i ok. ośmiu blokad muteksu na skok. Interfejs
   trzyma ten sam muteks.
9. **Sędzia działa na zegarze okna, co 16 ms** (`main.rs`, `Timer` 16 ms → `UiFrame::run`).
   Widzi tylko ostatni stan, więc gdy między dwoma tyknięciami przejdą dwa skoki, jeden
   przepada. Harnessy karmią sędziego skok po skoku, czyli mierzą coś odrobinę innego, niż
   działa na żywo.
10. **`MyApp` ma 2400 linii, `ui_frame.rs` 2000, `appwindow.slint` 4500.** Percepcja, reguły,
    generatory ćwiczeń i stan widoku są w jednym obiekcie.

### 1.4 Koszt obliczeń dziś

Program ma działać na cienkim kliencie i małym laptopie autora, tak jak działa teraz. Każde
zadanie silnika jest więc liczone także w czasie procesora.

Zmierzone 15.11.2026 na małym laptopie autora, Ryzen 3 5300U (`solitito --bench`, `--probe` na
40 s nagrania):

| co | czas | udział jednego rdzenia |
|---|---|---|
| model akordów, jeden przebieg, jeden wątek | 38,6 ms (mediana) | 97% przy pytaniu co 40 ms |
| CQT z jednego skoku (FFT 8192, jądra rzadkie, chroma) | najwyżej 0,15 ms | poniżej 1% |
| detektor uderzeń, jeden skok (z `docs/how-it-works.md`) | 0,7 ms | ok. 4% |

Wniosek: **prawie cały koszt programu to model akordów**, i to w każdym trybie, także tam,
gdzie jego odpowiedź rzadko decyduje. Reszta silnika razem to kilka procent rdzenia.

Model akordów jest pytany w pętli „licz, potem dośpij do 40 ms” (`main.rs::run_model`). Na
wolniejszym komputerze przebieg trwa dłużej niż 40 ms i model po prostu odpowiada rzadziej —
program zwalnia łagodnie, nic się nie psuje. To ważna własność: **nowy model z toru B jej nie
ma**. Musi zdążyć w każdym skoku (16 ms), inaczej analiza zostaje w tyle za dźwiękiem.

Cienki klient autora (17.11.2026, `solitito --bench`): **69,0 ms** na przebieg, 1,79 raza wolniej
niż laptop. Model nie mieści się tam w 40 ms, więc liczy bez przerwy na całym rdzeniu i
odpowiada ok. 14 razy na sekundę zamiast 25. Budżety poniżej mnoży się dla niego przez 1,79:
model toru B w 2 ms na laptopie to ok. 3,6 ms na cienkim kliencie, 22% rdzenia przy 62,5
ramkach na sekundę; model akordów pytany co 200 ms (A1) — ok. 35% rdzenia zamiast 100%.

### 1.5 Czego brakuje w ćwiczeniach i interfejsie

- Nie ma rytmu: metronomu, tempa, oceny punktualności. Program nie wydaje żadnego dźwięku.
- Nie ma pamięci postępów: ani statystyk, ani wiedzy, które akordy i interwały sprawiają kłopot.
- Utwory nie mają taktów ani formy, tylko listę akordów.
- Program nie mówi, **co** usłyszał zamiast tego, o co prosił. „Słyszę” pokazuje nazwę akordu.
- Pierwsze uruchomienie nie sprawdza wejścia, poziomu ani stroju.

## 2. Tablica wyników (S0)

Jedno polecenie, jeden plik wynikowy, zawsze te same materiały. Bez tego reszta planu nie ma
jak się bronić.

**Co zrobić**

- `cargo test --release -- --ignored scorecard` (albo `solitito --scorecard KATALOG`) zbiera
  istniejące harnessy: `onset_sources::credit_gate_by_source`, `repeat_harness`,
  `line_harness`, `ear_speed_harness`, `rise_harness`, `formula_harness`, oraz z gałęzi
  `experiments`: `grip_harness` (`free_order_on_chords`, `quiet_fifth_on_spliced_triads`).
- Wynik: `scorecard.json` z wierszami `{miara, wartość, materiał, commit}` i tabela tekstowa.
  Skrypt `.claude/skills/czytelny-rust/golden.sh` robi dziś część tego — zastąpić go.
- Materiał (`~/Documents/AtoA.wav`, `dist/crediting_measurements/*-g1.wav`, cztery sesje
  użytkownika z 388 zaliczeniami) dostaje jedno miejsce i zmienną `SOLITITO_MATERIAL`. Ścieżka
  `/home/greblus/Documents/AtoA.wav` jest dziś wpisana w kod harnessów.
- Każda miara ma dwie strony: ile zaliczono z tego, co zagrano, i ile zaliczono z tego, czego
  nie zagrano.

**Wartości wyjściowe** (z `docs/` i notatek; potwierdzić pierwszym przebiegiem):

| miara | dziś |
|---|---|
| AtoA, 51 pojedynczych dźwięków: klasy zaliczone, których nikt nie zagrał (akordy dozwolone / pojedynczo) | 62 / 36 |
| powtórzenie zaliczone, gdy dźwięk tylko brzmi | 0 / 51 |
| ponowne uderzenie po 0,8 s / 1,2 s zaliczone w porę | 48 / 51, 50 / 51 |
| 17 akordów naraz: dźwięki akordu zaliczone („Tylko uderzone” / bez) | 26 / 68, 47 / 68 |
| … klasy spoza akordu | 42 |
| trójdźwięki, kwinta 0 / −6 / −12 dB („Tylko uderzone”) | 24 / 16 / 10 z 51 |
| … kwinta niezagrana, a zaliczona | 6 / 51 |
| 9 akordów arpeggiem: dźwięki / klasy spoza | 32 / 36, 17 |
| linie 8, 8, 5 dźwięków: zwykła ścieżka kompletna do | 0,50 s na dźwięk |
| … szybkie zaliczanie kompletne do | 0,25 s na dźwięk |
| mediana czasu od uderzenia do zaliczenia: zwykła / szybka | ok. 400 ms / 26–46 ms |
| sesje użytkownika: zaliczenia zachowane | 388 / 388 |
| model akordów: pryma, zgodność pełna, F1 dźwięków (walidacja) | 98,1%, 92,4%, 0,909 |

**Rozszerzyć `solitito --bench`**: oprócz modelu akordów czas każdego etapu jednego skoku
osobno (FFT i CQT, krótkie widma, detektor uderzeń, szybkie ucho, tuner, sędzia) i suma jako
udział skoku 16 ms. To polecenie autor uruchamia na cienkim kliencie przed każdym wydaniem,
które rusza silnik; wynik trafia do `docs/judge-measurements.md`.

**Budżet** (laptop autora; cienki klient ×1,79):

| zadanie | koszt na skok na 5300U | uwagi |
|---|---|---|
| S1.1 drugi zestaw jąder | do +0,15 ms | jeszcze jedno FFT 8192 i jedno mnożenie rzadkie |
| S1.2 bramka | 0 | |
| S1.3 zgodność cech | 0 | zmiana w liczeniu, nie w ilości |
| S1.4 sędzia co skok | bez zmian w sumie | praca przenosi się z funkcji zwrotnej karty i z zegara okna do jednego wątku |
| S1.5 model akordów rzadziej | **minus do 97% rdzenia** poza Akordami | jedyne zadanie toru A, które wyraźnie zmienia zużycie |
| S2.2 model śledzący | najwyżej 2 ms | zastępuje detektor uderzeń (0,7 ms), czyli +1,3 ms |
| S2.5 akordy z zestawu dźwięków | **minus 97% rdzenia** także w Akordach | jeśli model akordów zniknie |

Tor A jest więc dla słabego komputera obojętny albo korzystny. Tor B dokłada stały koszt, który
musi się mieścić w skoku, ale pozwala zdjąć największy.

**Dodać nowe miary** do tablicy: koszt procesora na skok (µs, każdy odczyt osobno), opóźnienie od próbki
do zaliczenia na żywo (pętla zwrotna: wyjście karty do wejścia, sygnał testowy), odsetek
skoków, których sędzia na żywo nie zobaczył (punkt 1.3.9).

## 3. Grupa S — silnik

Dwa tory. **Tor A** poprawia obecny silnik tanio i od razu. **Tor B** buduje nowy: jeden model,
który zwraca zdarzenia „ten dźwięk uderzono teraz” i stan „te dźwięki brzmią”, z oktawą.
**Decyzja autora (15.11): najpierw tor A w całości, potem tor B.** Tor B korzysta z wyników
toru A: z tablicy wyników, z sędziego wołanego co skok (S1.4) i z pomiaru kosztu.

### Tor A — obecny silnik

#### Plan działań toru A (rewizja 16.11.2026)

Dwa pomiary z 16.11 zmieniają kolejność z pierwszej wersji planu.

**Model akordów nie zaliczył ani jednego dźwięku w sesjach autora.** Dzienniki
`dist/crediting_measurements/{gitara-g1,g2,g3,sesja-g1}.jsonl` (Interwały, 388 zaliczeń)
zapisują przy każdym zaliczeniu gałąź `by`: wszystkie 388 to gałąź 1, czyli ucho. Model
akordów, który zajmuje 97% rdzenia, nie zdecydował tam o niczym. Zastrzeżenia: to sesje gry
pojedynczymi dźwiękami w Interwałach; w trybie swobodnym z akordami granymi naraz, w Formułach
bez „Pojedynczo” i w trybie Akordy model jest potrzebny. Sesji ze Skal i Arpeggiów nie ma.

**Filtry CQT ignorują najnowszą część okna.** Każde jądro stoi na środku okna 512 ms, więc
najnowsza próbka, którą filtr w ogóle widzi, jest tyle starsza od końca okna:

| dźwięk | długość filtra | filtr nie widzi ostatnich |
|---|---|---|
| E2 (struna E) | 414 ms | 49 ms |
| A2 | 310 ms | 101 ms |
| G3 | 174 ms | 169 ms |
| E4 (struna e) | 104 ms | 204 ms |
| E5 | 52 ms | 230 ms |
| C7 (5. harmoniczna wysokich dźwięków) | 16 ms | 248 ms |

Ucho (`mono_pitch`) sumuje harmoniczne, które leżą w wysokich prążkach — właśnie tych, które
patrzą najdalej wstecz. Zgadza się to z „ucho nazywa nowy dźwięk ok. 0,26 s po uderzeniu”.

**Kolejność toru A po rewizji:**

| krok | co | koszt pracy | koszt procesora | zależy od |
|---|---|---|---|---|
| A0 | S0: tablica wyników, `--bench` per etap, przeniesienie harnessów z `experiments` | średni | 0 | — |
| A1 | S1.5: model akordów rzadziej tam, gdzie nie decyduje | mały | **do −97% rdzenia** | A0 |
| A2 | S1.1: jądra dosunięte do końca okna dla ucha i głosów | mały | +0,15 ms na skok | A0 |
| A3 | S1.2: bramka na krótkim oknie z histerezą | mały | 0 | A0 |
| A4 | S1.4 krok 0: zmierzyć, ile skoków sędzia gubi na żywo | mały | 0 | — |
| A5 | S1.3: zgodność cech modelu akordów, najpierw bez treningu | średni | 0 | A0 |
| A6 | K1, K4, potem S1.4 — tylko jeśli A4 pokaże stratę | duży | 0 | A4 |
| A7 | wydanie próbne, test autora na laptopie i cienkim kliencie | — | — | A1–A5 |

A1 idzie pierwsze, bo jest najtańsze, a daje najwięcej dokładnie tam, gdzie autor ma
ograniczenie: na słabym sprzęcie. A6 może okazać się niepotrzebne.

#### Stan prac toru A (16.11.2026, gałąź `engine-v2`)

Zrobione w kodzie, testy jednostkowe przechodzą (426). Pełne pomiary tablicą wyników
(`dist/scorecard.sh`) liczą się w tle i nie są jeszcze porównane — przed scaleniem trzeba je
odczytać (wyniki w katalogu tymczasowym sesji: `score/base`, `a1_200`, `a1_0`, `a2_centred`,
`a2`, `gate`, `a3`).

| krok | co zrobione | wynik zmierzony |
|---|---|---|
| A0 | `dist/scorecard.sh` (jedna zamrożona binarka testowa), harness chwytów z gałęzi `experiments`, AtoA z `SOLITITO_MATERIAL` | pomiar bazowy gotowy |
| A1 | Skale, Arpeggia i Gryf sądzone co ramkę dźwięku (`MyApp::tick_notes`), model akordów pytany według trybu (`MyApp::model_period`: 40 ms w Akordach i przy akordach naraz, 200 ms gdzie indziej; `SOLITITO_SLOW_MODEL` w testach) | procesor na laptopie autora, AtoA przez `--file` (`target/scorecard/cpu.sh`): Skale 113% → **35%** rdzenia, Arpeggia 111% → **34%**, Interwały pojedynczo 112% → **34%**, Akordy 111% → 108%; tablica wyników — czeka |
| A2 | filtry CQT dosunięte do końca okna dla ucha i głosów (`CqtAnalyzer::ear_cqt`, `end_aligned`), jeden wspólny odczyt okna `CqtAnalyzer::listen` dla programu, `--file` i wszystkich harnessów; stare filtry: `SOLITITO_EAR_CENTRED` w testach | sygnał syntetyczny: nowy dźwięk nazwany po **24–96 ms** zamiast **256–272 ms** (`how_soon_the_ear_names_a_new_note`) |
| A3 | bramka na ostatnich 64 ms z histerezą −6 dB i 160 ms (`audio::Gate`), też w harnessach; stara: `SOLITITO_GATE_WINDOW` w testach | czeka na tablicę (`gate/` przy progu −34 dB, AtoA 0 i −6 dB) |
| A4 | licznik ramek niewidzianych przez sędziego (`SOLITITO_HOPS=1`) | **36–37% ramek gubionych na żywo** na laptopie autora: karta oddaje próbki po ok. 23 ms, zegar okna widział tylko ostatnią ramkę porcji |
| A6 (lżejsze) | kolejka odczytów każdej ramki (`AudioAnalysis::remember_hop`), sędzia przerabia wszystkie przy każdym tyknięciu (`MyApp::tick_with_audio`) | **0% gubionych**; przenoszenie sędziego do osobnego wątku okazało się niepotrzebne |
| A5 | normalizacja cech kroczącym maksimum zamiast maksimum ramki | **bez zysku** (odsetek poprawnych nazw 0,61 → 0,62) — wycofane |
| A7 (nowe) | tryb Akordy: septyma wielka lub mała rozstrzygana głowicą wysokości (`seventh_heard`), gdy model podaje tę samą prymę z drugą septymą | `akordy-g1`: **17/17** akordów przechodzi (było 13/17, wszystkie porażki CMaj7); `arpeggia-g1`: 8/9 bez zmian, CMaj7 po 1,5 s zamiast 3,7 s |

**Tablica wyników 17.11 — stary silnik (`f4da3c9`) kontra cały tor A (`89d501b`).** Wyniki w
`target/scorecard/base` i `target/scorecard/all`; warianty pośrednie (`a1only`, `centred`,
`window`, `gate/`, `slow0`) liczy wznawialna kolejka `target/scorecard/run_all.sh`.

| miara | stary | nowy |
|---|---|---|
| linie, zwykła ścieżka: mediana od uderzenia do zaliczenia | ok. 400 ms | ok. 140 ms |
| linie po 0,40 s / dźwięk | 1 z 8 | 8 z 8 |
| linie Harrisa w dół po 0,30 s / dźwięk | 1 z 9 | 9 z 9 |
| AtoA, mediana do zaliczenia (akordy dozwolone / pojedynczo) | 156 / 220 ms | 29 / 43 ms |
| akordy naraz, „Tylko uderzone” | 26 z 68 | 29 z 68 |
| Formuły, akordy naraz | 17 z 68 | 19 z 68 |
| **powtórzenie zaliczone, gdy dźwięk tylko brzmi** | 0 z 51 | **14 z 51** (gałąź „class strike”) |
| **przedwczesne przy ponownym uderzeniu po 0,8 / 1,2 s** | 0 | **14** (w czasie 35 i 36 z 51 zamiast 48 i 50) |
| **AtoA: klasy zaliczone, których nikt nie zagrał (detektor; akordy / pojedynczo)** | 62 / 37 | **77 / 52** — przybyło głównie kwart nad dźwiękiem |
| **Greene: przebiegi stające w połowie** | 0 | **3** (order 9 G7 20/44, order 16 C7 14/28, G7 12/28), kilka „early 1” |
| arpeggia-g1: dźwięki akordu (detektor) | 32 z 36 | 30 z 36 |
| szybkie zaliczanie, linie po 0,25 / 0,20 s | 8/8, 5/5 | 7/8, 4/5 |

**Regresje do naprawy przed scaleniem** (kolejność): (1) powtórzenia — hipoteza: szybsze ucho
zalicza dźwięk, zanim detektor uderzeń zgłosi jego uderzenie (22–32 ms), i to spóźnione
zgłoszenie wygląda potem jak nowe uderzenie (`Credits::struck_since`, `settle`); (2) fałszywe
kwarty — ucho na końcu okna myli się na wybrzmieniu (opisane w `how-it-works.md`, „The ear just
after a strike”); (3) Greene. Najpierw porównać z wariantami `centred` (stare filtry ucha) i
`window` (stara bramka), żeby wiedzieć, która zmiana odpowiada za każdą regresję.

**Przypisanie i naprawa regresji (18.11).** Warianty `a1only` i `centred` nie mają żadnej z
trzech regresji — wszystkie wnosiły filtry na końcu okna (A2). Naprawy:

1. Powtórzenia: `CREDIT_OWN_STRIKE` (0,2 s) — uderzenie tej klasy zgłoszone tuż po zaliczeniu
   należy do niego. Wynik: tylko wybrzmiewa 0/51 (było 14), ponowne uderzenie po 0,8 s 49/51
   (stary silnik 48), po 1,2 s 50/51, nic przedwcześnie.
2. Fałszywe zaliczenia: ucho czyta filtrami z końca okna tylko przez `EAR_FRESH_HOPS` (16 ramek,
   0,26 s) po ataku (flux albo detektor), potem filtrami ze środka. AtoA: 63 / 41 (stary 62 / 37,
   przed naprawą 77 / 52); linie dalej 8/8 po 0,40 s, mediana 154 ms (12 i 20 ramek: to samo
   tempo, więcej fałszywych).
3. Greene: przedwczesne 7 → 0, zatrzymania 4 → 2 (order 9 G7 z napięciami, order 16 G7 bez).
   Przyczyna zbadana na order 16: detektor uderzeń nie zgłasza drugiego F, a powtórzenie przez
   „ucichł i wrócił” nie dochodzi do skutku, bo F za krótko znika z głosów. Liczenie „ucichł” na
   głosach ze środka okna dało gorzej (5 zatrzymań) — wycofane. Otwarte.

**Model akordów wyłączony tam, gdzie nie decyduje (18.11).** Wariant `slow0` tablicy wyników
(model niepytany w trybach z pojedynczymi dźwiękami) daje co do cyfry te same wyniki, co pytanie
co 200 ms. Pasek „Słyszę” — jedyne, co jeszcze czytało model w tych trybach — usunięty na prośbę
autora. `SLOW_MODEL_PERIOD = None`. Procesor na laptopie, AtoA przez `--file`: Skale i Arpeggia
**11%** rdzenia (stary silnik 111–113%, z modelem co 200 ms 34–35%).

Bramka A3 przy progu programu (−34 dB), AtoA 6 dB ciszej (`target/scorecard/gate`): stara bramka
G miksolidyjska 0,50 s 5/8, D dorycka 0,40 s 1/8, D m7 0,40 s 1/5; nowa wszystkie 8/8 i 5/5. Przy
normalnym poziomie bez różnicy.

Do sprawdzenia przez autora z gitarą: A6 sprawia, że na żywo sędzia widzi o połowę więcej
ramek; progi liczone w ramkach (np. ile ramek ucho trzyma ten sam dźwięk) były dobierane w
części na żywo, czyli na 2/3 ramek. Zaliczanie może być odrobinę szybsze — trzeba posłuchać,
czy nie zalicza dźwięków niezagranych.

Nowa miara: `chord_probe::chords_named_on_a_recording` (`SOLITITO_WAV`, `SOLITITO_LABELS`) — dla
każdego uderzenia akordu: kiedy model pierwszy raz podaje dobrą nazwę, jaki odsetek odpowiedzi
jest dobry, kiedy tryb Akordy przechodzi dalej. Spostrzeżenie: po ciszy pierwsza nazwa
przychodzi po ok. 736 ms, bo model czeka na zapełnienie okna w 90% (`MIN_FILL_CHORD`).

#### S1.1 Jądra wyrównane do końca okna, dla ucha

- **Dlaczego:** ustalenie 1.3.1.
- **Co:** drugi zestaw jąder, każde dosunięte do końca okna zamiast wyśrodkowane, tylko dla
  `mono_pitch` i `voices`. Cechy modelu akordów zostają na starych jądrach — model był na nich
  uczony.
- **Jak, bez regenerowania wag:** przesunięcie filtra w czasie o Δ próbek to w widmie mnożenie
  przez `e^{-2πi·kΔ/N}`; wagi w `dsp_weights.json` to sprzężenie widma filtra, więc nowe wagi
  bina `i` to stare razy `e^{+2πi·kΔ_i/8192}`, gdzie `k` to indeks prążka FFT, a
  `Δ_i = max(0, (8192 − L_i) / 2)` zaokrąglone do całości, `L_i = Q·16000 / f_i`,
  `Q = 1 / (2^{1/24} − 1)` (tak liczy długość `librosa.filters.constant_q` przy
  `filter_scale = 1`). Moduły wag się nie zmieniają, więc rzadki zapis i przycięcie zostają
  takie same. Liczyć raz przy starcie w `CqtAnalyzer::new`, bez librosy i bez Kaggle.
  Sprawdzić testem: sinusoida w ostatnich 100 ms okna daje w nowym zestawie co najmniej tyle
  samo co w starym sinusoida w środku okna.
- **Okno Hanna:** drugi przebieg FFT tego samego okna **bez** Hanna (Hann zeruje koniec okna,
  czyli dokładnie to, co nowe jądra mają czytać). Jedno FFT 8192 więcej na skok.
- **Uwaga:** jądra przestają być wyrównane w czasie między sobą (niskie są dłuższe). Dla
  `mono_pitch`, które sumuje harmoniczne z wyższych, szybszych prążków, może to zmienić wynik
  na ataku. Dlatego pomiar po obu stronach.
- **Przyjęcie:** `line_harness` i `ear_speed_harness`: zwykła ścieżka kompletna poniżej 0,50 s
  na dźwięk, mediana czasu do zaliczenia wyraźnie poniżej 400 ms; AtoA fałszywe nie rosną;
  sesje 388/388. Jeśli opóźnienie się nie zmieni, hipoteza jest fałszywa — opisać i porzucić.

#### S1.2 Bramka szumu na krótkim oknie, z histerezą

- **Co:** poziom do bramki z ostatnich 1024 próbek (64 ms), nie z 8192. Otwarcie na progu,
  zamknięcie 6 dB niżej albo po 150 ms poniżej. Miernik w interfejsie bez zmian.
- **Przyjęcie:** materiał z `bramka-szumu-a-zaliczanie` (AtoA osłabione o 6, 12, 18 dB ze stałą
  podłogą szumu): odsetek ramek docierających do analizy przy −6 dB i −12 dB rośnie; AtoA
  fałszywe nie rosną; reguła „wyciszenie na 200 ms pozwala powtórzyć dźwięk” nadal działa
  (`repeat_harness`).

#### S1.3 Zgodność cech modelu akordów z trenerem

- **Co:** przypiąć test jak dla detektora uderzeń: ten sam krótki WAV przez `process_audio_file`
  z trenera i przez `CqtAnalyzer`, porównanie ramka po ramce, wynik w `dist/fixtures/`.
  Zmierzyć różnicę z punktów 1.3.2 i 1.3.3 osobno: Hann, normalizacja, wyśrodkowanie ramek
  (`librosa.cqt` domyślnie centruje), przepróbkowanie.
- **Potem, w tej kolejności:**
  1. Bez treningu, po stronie programu. Normalizacja względem maksimum całego pliku nie ma
     odpowiednika na żywo, ale da się ją przybliżyć: odniesieniem jest maksimum z ostatnich
     kilku sekund gry, opadające powoli (np. o 1 dB na sekundę), a nie maksimum ramki.
     Zrezygnować z dodatkowego okna Hanna, jeśli pomiar zgodności pokaże, że to ono odbiega.
  2. Nowy trening modelu akordów (nowy `RUN_TAG`) tylko wtedy, gdy punkt 1 nie wystarczy
     **i** autor nie planuje zastąpić modelu akordów w torze B (S2.5). Trening kosztuje jego
     czas na Kaggle, a w torze B ten model może zniknąć.
- **Przyjęcie:** zgodność pełna i F1 dźwięków na walidacji nie spada; na `akordy-g1` czas od
  uderzenia do pierwszej poprawnej nazwy akordu nie rośnie.

#### S1.4 Sędzia co skok, w wątku analizy

- **Dlaczego:** ustalenia 1.3.8 i 1.3.9. To jest też warunek toru B: zdarzeń nie wolno gubić.
- **Krok 0, zanim cokolwiek się przebuduje:** zmierzyć stratę. W `Heard::read_audio_frame`
  liczyć, o ile `frames_seen` wzrosło od poprzedniego odczytu; różnica większa niż 1 to skoki,
  których sędzia nie zobaczył. Wypisać rozkład po minucie gry na laptopie i na cienkim kliencie
  (opcja `SOLITITO_DEBUG` albo wiersz w dzienniku). Liczniki uderzeń (`class_strikes`) nie
  giną, bo są sumami; giną odczyty liczone na ramkę — `count_steady_pitch` i
  `count_quiet_classes`. Jeśli gubiona jest mniej niż co setna ramka, przebudowa nie jest
  potrzebna w torze A i czeka na tor B.
- **Co, jeśli strata jest:**
  1. Funkcja zwrotna karty tylko wpisuje próbki do bufora kołowego bez blokad. Nowa zależność:
     `rtrb` (bufor SPSC bez blokad, bez własnych zależności) — albo własny na `AtomicUsize`,
     jeśli autor woli bez zależności.
  2. Wątek `solitito-ears` czyta bufor, robi całe dzisiejsze `Listener::hear` i **woła
     sędziego raz na skok**. Model akordów zostaje w swoim wątku, a jego odpowiedź jest dla
     sędziego jeszcze jednym wejściem ze znacznikiem skoku.
  3. Interfejs dostaje migawkę stanu (`Arc` podmieniany raz na skok albo kanał) i tylko rysuje.
     Polecenia z interfejsu (pauza, strzałki, zmiana trybu) idą kanałem do wątku sędziego.
- **Skutek uboczny:** harnessy i program na żywo wykonują ten sam kod. Tryb `--file` przestaje
  być osobną kopią pętli (`start_file_playback` powiela dziś `Listener::hear`).
- **Przyjęcie:** tablica wyników bez zmian co do liczby; nowa miara „skoki niewidziane przez
  sędziego” równa zeru; brak przydziałów pamięci w funkcji zwrotnej.
- **Ryzyko:** największa zmiana strukturalna toru A. Robić po K1 (podział `MyApp`), bo sędzia
  musi dać się przenieść do innego wątku bez ciągnięcia stanu widoku.

#### S1.5 Model akordów taniej

- **Dowód:** 388 z 388 zaliczeń w sesjach autora przez ucho, żadne przez model (zob. plan
  działań toru A).
- **Co:** wątek modelu (`main.rs::run_model`) dostaje od sędziego okres pytania zależny od
  tego, czy odpowiedź może coś zdecydować:
  - co 40 ms: tryb Akordy; Interwały i Formuły bez „Pojedynczo”;
  - rzadko (np. co 200 ms, żeby pasek „Słyszę” dalej coś pokazywał) albo wcale: Skale,
    Arpeggia, Gryf, Interwały i Formuły z „Pojedynczo”;
  - wcale: pauza z otwartymi ustawieniami, okno zminimalizowane.
  Zmiana trybu natychmiast przywraca pełne tempo. Okres przekazywać przez `AtomicU32`, bez
  muteksu.
- **Uwaga:** przy „Pojedynczo” `sounding_by` nadal sięga po model w dwóch miejscach: gdy ucho
  milczy (cichy dźwięk, niska struna) i w `ear_holds_only_a_ring`. Dzienniki pokazują, że w
  praktyce nie zadziałało ani razu, ale harnessy mogą to wyłapać — dlatego najpierw tablica
  wyników z modelem pytanym co 200 ms, potem z wyłączonym.
- **Dwa wątki ONNX odpadają:** skracają przebieg (20,8 ms według notatek), ale zajmują dwa
  rdzenie; na cienkim kliencie z małą liczbą rdzeni zabiorą czas wątkowi dźwięku i interfejsowi.
- **Przyjęcie:** sesje 388/388; tablica wyników bez zmian w trybach pojedynczych dźwięków;
  `--bench` albo `top` na laptopie: zużycie procesora w Skalach i Arpeggiach poniżej 15%
  jednego rdzenia.

### Tor B — nowy silnik: śledzenie dźwięków

Cel: jeden przyczynowy model, który co skok (16 ms) zwraca dla każdego dźwięku gitary
(MIDI 40–88, 49 dźwięków) dwie liczby: **uderzono teraz** i **brzmi**. Sędzia pracuje na
zdarzeniach z oktawą zamiast na dwunastu klasach i czterech pośrednich odczytach.

Dlaczego to ma szansę: detektor uderzeń już jest takim modelem w małej skali (przyczynowy,
0,7 ms na skok, cechy zgodne z trenerem co do bitu, opóźnienie 22–32 ms). Trzeba mu dać
49 wyjść zamiast 12, drugą głowicę „brzmi” i przede wszystkim dane, w których akord uderzony
naraz ma etykietę na każdej strunie.

#### S2.1 Zbiór danych v3

Wniosek z dotychczasowych prób jest w notatkach: syntetyczne struny Karplusa-Stronga nie
przenoszą się na prawdziwą gitarę, a remiks ścieżek hex z GuitarSet był bliżej. Stąd kolejność
źródeł:

1. **GuitarSet, ścieżki hex po usunięciu przesłuchów** (`alonhaviv/guitarset` na Kaggle, 360
   nagrań, 6 kanałów). Każda struna ma własną ścieżkę i własne nuty w JAMS, więc etykieta
   „uderzono” jest na strunę. Generator `render_hex_strum_take` z gałęzi `experiments` już to
   czyta — uogólnić:
   - miks sześciu strun z losowym poziomem każdej (±12 dB), także struna wyciszona albo wycięta;
   - etykiety: początek nuty, koniec, MIDI, struna;
   - cztery wady GuitarSet opisane w `docs/Solitito_project_summary_*.md` uwzględnić tak jak w
     `model_trainer.py` (adnotacje `performed`, maskowanie solówek);
   - początki nut pochodzą z adnotacji, nie z ataku kostki — poprawić je na ścieżce struny
     (szczyt przyrostu energii w ±50 ms) i zapisać, ile przesunięto.
2. **Istniejący zbiór syntetyczny v2** (`greblus/solitito_dataset_v2` na HF: GP5 → DI → NAM,
   czysty i na granicy przesteru, etykiety z generatora). Jest gotowy i ma dokładne etykiety
   dźwięków, w tym 96 pojedynczych dźwięków na wszystkich strunach.
3. **Nowe bloki w tym samym torze** (`dataset_generator_v2.py`), jeśli autor zechce wykonać
   jeszcze jeden render w DAW — to praca ręczna po jego stronie, więc tylko gdy pomiar pokaże,
   że bez nich się nie da. Bloki, których brakuje: akord uderzony naraz z
   rozrzutem poziomów strun, ten sam dźwięk powtórzony na brzmiącym, dźwięk o pół tonu od
   brzmiącego, legato (hammer-on, pull-off, slide), tłumienie. Etykiety z generatora, co do
   próbki.
4. **Zbiory zewnętrzne — do sprawdzenia, czy są dostępne i na jakiej licencji**, zanim
   cokolwiek się na nich oprze: IDMT-SMT-Guitar (pojedyncze dźwięki i akordy kilku gitar),
   EGDB (gitara elektryczna, DI i wzmacniacze), SynthTab (duży zbiór syntetyczny z tabulatur,
   do wstępnego uczenia). Autor planu nie weryfikował ich aktualnego stanu.

Wzbogacanie w trenerze: poziom ±12 dB (dziś ±6; cichszy gracz gubił połowę uderzeń), strój
±20 centów, szum i przydźwięk sieci, filtr tonu, przepróbkowanie tą samą metodą co w programie.

Podział jak dziś: gracze 00–03 uczą, 04 walidacja, 05 test.

**Autor nie nagrywa danych treningowych** — to za dużo pracy wymagającej dokładnych etykiet.
Może przygotować nagrania **testowe**. Nagrania autora i materiał z tablicy wyników nigdy nie
wchodzą do uczenia; są sprawdzianem, i to jedynym na jego gitarach i kartach. Skutek: model
uczy się wyłącznie na cudzych gitarach i na syntetyku, więc różnica między danymi a sprzętem
autora jest głównym ryzykiem toru B (rozdział 9). Zanim ruszy trening, autor nagrywa zestaw
testowy opisany w C7.

Wszystko w `dist/dataset_v3.py`, z manifestem (pliki, skróty, wersja generatora) i testami jak
`dist/test_strike_trainer.py`.

#### S2.2 Model

- **Wejście:** cechy detektora uderzeń (`ShortFeatures`: widma 1024 i 2048, 0–4 kHz, 770
  wartości) rozszerzone o trzecie okno 4096 w paśmie 0–1 kHz, dla rozdzielczości niskich
  strun. E2 i F2 różnią się o 4,9 Hz; okno 2048 ma prążek 7,8 Hz i rozróżnia je dopiero po
  trzeciej harmonicznej. Czy trzecie okno jest potrzebne, rozstrzyga pomiar na strunach E i A.
- **Sieć:** jak `CausalOnset` (sploty przyczynowe z rozszerzaniem 1, 2, 4, 8), szersza i z
  jeszcze jedną warstwą rozszerzenia 16, żeby pole widzenia sięgało ok. 0,5 s. Dwie głowice po
  49 wyjść: `onset` i `frame`. Rozmiar wyjściowy do 2–3 MB; próg kosztu: 2 ms na skok na
  jednym wątku na maszynie autora.
- **Cele:** `onset` — okno 96 ms od początku nuty, jak dziś (`TARGET_SECONDS`); `frame` — od
  początku do końca nuty. Strata ogniskowa z wagą klasy dodatniej. Tam, gdzie koniec nuty w
  adnotacji jest niepewny (wybrzmiewanie), maska zamiast kary.
- **Wybór epoki i progów** po miarach zdarzeniowych, tak jak dziś robi to
  `validation_choice`, z dwiema stronami: czułość i fałszywe na minutę. Osobno raportować:
  akord naraz (ile strun z uderzonych nazwano), kwinta cichsza o 6 i 12 dB, powtórzenie na
  brzmiącym, dźwięk o pół tonu od brzmiącego.
- **Trener:** `dist/note_trainer.py`, jeden plik uruchamiany na Kaggle, z migawkami na HF, na
  wzór `strike_trainer.py` (tryby `train` / `export_only`, wznawianie, podsumowanie w JSON).
- **Rozszerzenie na później, nie w pierwszym przebiegu:** głowica struna × próg (tabulatura).

#### S2.3 Wykonanie w programie

- `src/tracker.rs`: `ShortFeatures` rozszerzone o trzecie okno, model przez `ort` jak
  `strike::Strikes`, historia ramek w pierścieniu. Test zgodności cech z trenerem co do bitu,
  wspólny plik w `dist/fixtures/`.
- Po modelu cienka warstwa zamieniająca prawdopodobieństwa w zdarzenia: przekroczenie progu z
  ponownym uzbrojeniem względem szczytu (jak `strike::Latch`). Refrakcja i kontrola energii
  (`Refires`) tylko jeśli pomiar pokaże, że model sam nie wystarcza — celem jest ich nie
  potrzebować.
- Wyrównanie poziomu (`Leveller`) zostaje przed cechami.
- Interfejs dla sędziego:

  ```rust
  pub struct Struck { pub hop: u64, pub midi: u8, pub strength: f32 }
  pub struct Sounding { pub hop: u64, pub notes: [f32; 49] }
  pub trait Ears {
      fn hear(&mut self, hop: &[f32]) -> (Vec<Struck>, Sounding);
  }
  ```

  Dwie implementacje: `ClassicEars` (dzisiejsze odczyty, zdarzenia bez oktawy) i
  `TrackerEars`. Sędzia v2 pisany jest na tym interfejsie.
- `solitito --probe` dostaje kolumny nowego modelu, żeby dało się go oglądać na nagraniu bez
  sędziego.

#### S2.4 Sędzia v2 na zdarzeniach

Reguły, które dziś zajmują większość `judge.rs` i `credits.rs`, sprowadzają się do porównania
znaczników czasu:

| dziś | w sędzi v2 |
|---|---|
| powtórzony dźwięk: `struck_since`, liczniki klas, skok oktawy, wyciszenie 200 ms | nowe zdarzenie `Struck` tego dźwięku po prośbie |
| przenoszenie przez granicę akordu: `carry_over_what_rings` | liczą się tylko zdarzenia po chwili, w której akord stał się celem (z wyjątkiem „za wcześnie” — okno wstecz jak `JUST_STRUCK`) |
| akord naraz: `NOTE_HOLD`, konkurencja kroków, reguła 90% | wszystkie zdarzenia `Struck` w oknie jednego uderzenia (ok. 80 ms) zaliczają swoje kroki |
| ucho tuż po ataku, ucho na wybrzmiewaniu: `EAR_SETTLE`, `EAR_BACKED` | niepotrzebne: zalicza zdarzenie, nie stan |
| „Po kolei” / dowolna kolejność / „Pojedynczo” | bez zmian w znaczeniu; „Pojedynczo” = zdarzenie liczy się tylko, gdy w jego oknie nie ma innych |

Dochodzi to, czego dziś nie ma: opcja **„właściwa oktawa”** (domyślnie wyłączona — ćwiczenia są
pisane stopniami, oktawa jest wyborem gracza) oraz **nazwa błędu**: „zagrano E♭, proszono o E”.

- **Co:** `src/judge2/` obok `src/state/`, osobno dla trybów dźwiękowych i dla Akordów.
  Przełącznik w ustawieniach: Silnik — klasyczny / nowy. Testy sędziego v2 pisane od razu na
  nagraniach przez tablicę wyników, nie tylko na danych wymyślonych.
- **Kolejność trybów:** Gryf i Skale (jeden dźwięk, najprościej), Arpeggia, Interwały w
  dowolnej kolejności (tu jest cel projektu: akord naraz), Formuły, na końcu Akordy.

**Progi przyjęcia nowego silnika** (propozycja; autor zatwierdza przed treningiem):

| miara | dziś | próg |
|---|---|---|
| akord naraz, dźwięki akordu | 26 / 68 („Tylko uderzone”) | co najmniej 58 / 68 |
| … klasy spoza akordu | 42 | najwyżej 21 |
| kwinta przy −12 dB | 10 / 51 | co najmniej 30 / 51 |
| kwinta niezagrana, a zaliczona | 6 / 51 | najwyżej 6 / 51 |
| AtoA: klasy, których nikt nie zagrał | 62 / 36 | najwyżej 20 w obu ustawieniach |
| powtórzenie, gdy dźwięk tylko brzmi | 0 / 51 | 0 / 51 |
| ponowne uderzenie po 0,8 s | 48 / 51 | co najmniej 48 / 51 |
| linie kompletne do | 0,25 s (szybka ścieżka) | 0,20 s, bez osobnej opcji |
| mediana czasu do zaliczenia | 400 ms / 26–46 ms | najwyżej 80 ms |
| sesje użytkownika | 388 / 388 | co najmniej 380 / 388, każdą stratę obejrzeć |
| koszt na skok | — | najwyżej 2 ms, jeden wątek |

Jeśli model nie przejdzie progów po dwóch uczciwych przebiegach (dane poprawione między nimi),
tor B się zatrzymuje, a wyniki idą do `docs/judge-measurements.md`. Tor A zostaje.

#### S2.5 Akordy z zestawu dźwięków

- **Co:** w trybie Akordy cel jest zbiorem dźwięków: wymagane (dla pełnego chwytu i dla
  shella osobno), dozwolone, zakazane. Akord zaliczony, gdy po uderzeniu wszystkie wymagane
  brzmią, a zakazanych nie ma. Żółty jak dziś: brakuje dźwięku, który substytut pomija.
- **Co to daje:** każdy akord, który da się zapisać stopniami — `m6`, `mMaj7`, `9`, `13`,
  `7♭9`, `7♯11`, `sus2`, przewroty z wymaganym basem — bez nowej klasy w modelu i bez listy
  substytutów w `chords_mode.rs`.
- **Ostrożnie:** model akordów ma 92,4% zgodności pełnej i jest sprawdzony w graniu. Zestaw
  dźwięków wchodzi najpierw jako drugi świadek obok niego; model akordów znika z programu
  dopiero, gdy na `akordy-g1` i na teście GuitarSet (gracz 05, nagrania comp) zestaw jest co
  najmniej tak dobry. Oktawa (pryma w basie) może być tu słabym punktem.

#### S2.6 Później, po przyjęciu toru B

- **Struna i próg.** Głowica tabulatury; sprawdzanie pozycji w Gryfie i w „Od E do e”.
- **Dopasowanie do gitary użytkownika.** Bez nagrań treningowych zostaje dobór progów na
  strunę z krótkiej kalibracji (kilka dźwięków na każdej strunie przy pierwszym uruchomieniu,
  U2). Tylko jeśli tablica wyników pokaże różnicę między gitarami.
- **Legato i podciągnięcia.** Zdarzenie bez ataku kostki; dźwięk między półtonami.

## 4. Grupa C — ćwiczenia

Kolejność według tego, ile dają grającemu i od czego zależą.

### C1 Rytm: metronom i ocena punktualności

- **Zależy od:** wyjścia dźwięku; ocena punktualności od S1.4, a najlepiej od toru B.
- **Co:**
  1. Wyjście dźwięku przez `cpal` (już jest w zależnościach), własny krótki klik, bez plików.
  2. Tempo i metrum w szybkich opcjach; odliczanie przed startem.
  3. Tryb „w tempie” dla Arpeggiów, Skal i linii Harrisa: dźwięk ma paść w oknie wokół
     uderzenia metronomu. Po przejściu: ile przed i ile po, w milisekundach.
  4. Kalibracja opóźnienia wejścia i wyjścia: program gra klik, użytkownik gra razem z nim
     kilkanaście razy, mediana różnicy zostaje w ustawieniach.
- **Uwaga:** klik z głośników wraca mikrofonem do wejścia. Przy mikrofonie zalecić słuchawki;
  w silniku klasycznym klik jest atakiem dla fluxu — sprawdzić na tablicy wyników.
- **Przyjęcie:** na materiale z onsetami znanymi z konstrukcji (`dist/latency_material.py`)
  błąd pomiaru punktualności poniżej 15 ms.

### C2 Utwory z taktami i formą

- **Co:** rozszerzyć format `user_songs.txt` wstecznie zgodnie: `|` rozdziela takty, `%`
  powtarza takt, dwa akordy w takcie dzielą go na pół, linia `[A]` nazywa część.
  Dotychczasowy zapis (akordy po spacji) znaczy „każdy akord jeden takt”.
- **Co to daje:** pętla wybranej części, metronom idący przez formę, liczba dźwięków na akord
  wynikająca z długości akordu (dziś ręczna lista „8 – cały takt / 4 – pół taktu”).
- **Przyjęcie:** wszystkie utwory z biblioteki czytają się jak dotąd; testy parsera.

### C3 Dziennik postępów i słabe miejsca

- **Co:** przy każdym zaliczeniu wiersz: tryb, ćwiczenie, akord albo stopień, czas od prośby do
  zaliczenia, liczba błędnych prób. Zapis w katalogu konfiguracji jako JSON Lines (`serde_json`
  już jest), jeden plik na miesiąc.
- **Widok:** dla utworu — które akordy trwają najdłużej; dla Interwałów — które stopnie nad
  którą jakością; dla Gryfu — mapa gryfu z czasem odpowiedzi; postęp w czasie.
- **Losowanie ważone:** tam, gdzie program losuje (Gryf, Formuły, losowa kolejność, losowa
  tonacja), częściej to, co szło wolniej. Opcja, domyślnie włączona; rozkład do sprawdzenia
  testem (żadna pozycja nie znika z losowania).

### C4 Informacja o błędzie

- **Zależy od:** toru B dla pełnej wersji; w wersji klasycznej da się pokazać klasę, którą
  nazywa ucho.
- **Co:** gdy przez ok. 1 s brzmi dźwięk, o który nie proszono, kafelek albo pasek mówi, co to
  jest: „♭3 zamiast 3”. Bez kary, bez dźwięku. Opcja, domyślnie włączona.

### C5 Słuch: powtórz, co słyszysz

- **Zależy od:** wyjścia dźwięku (C1.1) i prostego syntezatora struny w Ruście.
- **Co:** program gra interwał, trójdźwięk albo krótką komórkę nad zagraną prymą; użytkownik
  odgrywa ją na gitarze. Poziomy z tych samych list, co Interwały (`interval_levels.rs`).
  Siódmy tryb w pasku albo ćwiczenie w Interwałach — decyzja autora.

### C6 Plan ćwiczenia

- **Co:** przewodniki z `docs/*-practice.md` („jak ćwiczyć po kolei”) jako dane: lista kroków
  `{tryb, ćwiczenie, ustawienia, czas albo liczba przejść}`. Program prowadzi przez kroki i
  zapisuje, na którym skończono. Plik `routines.txt` użytkownika obok `user_songs.txt`.
- **Uwaga:** ustawienia trybu są dziś rozproszone po `Settings` i `MyApp`; krok planu musi
  umieć je ustawić jednym wywołaniem — zob. K2.

### C7 Nagrania testowe autora

Nie dane treningowe, tylko sprawdzian dla toru B i dla każdej zmiany toru A. Mają być krótkie
i łatwe do nagrania; etykiety wynikają z tego, o co program prosił (`SOLITITO_RECORD` zapisuje
dźwięk i dziennik na jednym zegarze).

- **Co nagrać**, na każdej gitarze i karcie, na której autor ćwiczy, po kilka minut:
  Interwały w dowolnej kolejności z chwytami granymi naraz; Skale i Arpeggia w zwykłym tempie
  i szybko; jeden utwór w Akordach; to samo cicho.
- **Co zrobić w kodzie:** narzędzie, które z pary WAV + dziennik robi plik etykiet dla
  harnessów (dziś `akordy-g1-labels.csv` powstał ręcznie). Dziennik mówi, o co proszono, nie
  co zagrano, więc autor przegląda listę i skreśla pomyłki — tak jak przy
  `atoa-reference-reviewed.csv`.
- **Przyjęcie:** nowe nagrania są w tablicy wyników jako osobne wiersze; wartości wyjściowe
  spisane przed pierwszą zmianą silnika.

### C8 Drobniejsze

- Akordy: wymagany przewrót albo najwyższy dźwięk chwytu (po S2.5).
- Interwały: ćwiczenia w tercjach i sekstach granych jako dwudźwięki (dopiero po torze B).
- Gryf: „wszystkie miejsca tego dźwięku na strunach X–Y” (po głowicy tabulatury, S2.6).
- Skale: strzałki przy wyłączonym „Od E do e” mają tylko trzy położenia — rozważyć losowanie
  pozycji tak jak w „Od E do e”.

## 5. Grupa U — interfejs

### U1 Podgląd tego, co program słyszy

Zamiast paska „Słyszę” mały gryf albo pasek dwunastu pól, na którym na żywo zapalają się
dźwięki brzmiące i błyskają uderzone. To najtańszy sposób na zaufanie do zaliczania i na
diagnozę („gram, a nie zalicza” przestaje być zagadką). W silniku klasycznym: klasy z
`voices`, ucha i detektora; w nowym: dźwięki z oktawą. Opcja w szybkich opcjach każdego trybu.

### U2 Pierwsze uruchomienie i test wejścia

Jeden ekran w trzech krokach: wybór wejścia z miernikiem; pomiar szumu przez 2 s i
ustawienie bramki 6 dB nad nim; strojenie sześciu strun tunerem. Dostępny potem z zakładki
Dźwięk. Zapisuje, że został ukończony, i sam nie wraca.

### U3 Podsumowanie i postępy

Widoki z C3 jako siódma pozycja paska bocznego albo zakładka ustawień. Po zakończeniu utworu
albo po pauzie krótka karta: czas, najwolniejsze miejsca.

### U4 Klawiatura

Spacja — pauza; strzałki — to samo, co strzałki przy pauzie; cyfry 1–6 — tryby; `Tab` — zwiń
albo rozwiń pasek; `F11` jest. `FocusScope` `window_keys` już zbiera klawisze.

### U5 Układ zależny od rozmiaru okna

Dziś całe okno jest skalowane jednym współczynnikiem z wysokości (`design_w` × `design_h`).
Skutki: przy szerokim oknie puste pasy, przy tekście 200% elementy wypadają poza kolumnę,
a każda zmiana wysokości elementu wymaga przeliczenia budżetu 800 jednostek (komentarze w
`appwindow.slint`). Przejście na układ, w którym kolumna ćwiczenia wypełnia dostępne miejsce,
a rysunek bierze resztę.

- **Ryzyko:** duże; notatka `slint-rozmiar-okna` opisuje, jak sterowanie rozmiarem walczy z
  biblioteką. Najpierw jeden tryb (Skale) jako próba, porównanie zrzutów w trzech rozmiarach
  okna i przy tekście 70%, 100%, 200%.

### U6 Podział `appwindow.slint`

4500 linii w jednym pliku. Podział bez zmiany wyglądu: `theme.slint`, `widgets.slint`
(Dropdown, SideToggle, Segmented, przyciski), `sidebar.slint`, `practice.slint`,
`settings.slint`. Slint importuje komponenty między plikami; `build.rs` kompiluje plik główny.
Sprawdzian: zrzuty przed i po identyczne co do piksela.

### U8 Animacje i rozmycie (Slint 1.19)

Program używa Slinta 1.19 (od 17.11.2026, wcześniej 1.18.1; wygląd bez zmian, testy przechodzą).
Nowa wersja ma rozszerzone animacje i efekt rozmycia. Do rozważenia po torze A: przejścia między
trybami, płynne wysuwanie szybkich opcji i zwijanie paska, rozmyte tło pod rozwiniętą listą i
pod powiększonym schematem chwytu. Warunek: koszt rysowania na cienkim kliencie — zmierzyć
zużycie procesora przez okno (`target/scorecard/cpu.sh`) przed i po; rozmycie liczone co klatkę
może być drogie przy programowym rysowaniu.

### U7 Znane usterki i zaległości

- Zmiana języka nie odświeża trzech napisów (`jezyk-nie-odswieza-trzech-napisow`).
- Zrzuty ustawień w `docs/screenshots` mają stary wygląd.
- Napisy na rysunkach gryfu są Arialem; reszta okna Noto Sans. Zmiana wymaga sprawdzenia ♭ na
  Windows (`windows-bemol-i-rozmiar-okna`).
- Stan zaliczenia jest dziś tylko kolorem (zielony / niebieski). Dodać znak na kafelku, żeby
  nie zależał od rozróżniania barw.
- Skrypt zrzutów: jeden w repozytorium (`dist/` albo `xtask`), bez klikania, z konfiguracją
  każdego obrazka w pliku. Dziś ustawienia żyją w notatkach.

## 6. Grupa K — kod

### K1 Podział `MyApp`

`src/state.rs` (2400 linii) trzyma naraz: wybór trybu i ćwiczenia, bibliotekę utworów i skal,
stan sędziego, losowania, postęp w przejściu. Docelowo cztery części z wąskimi stykami:

- `ears` — wszystko, co powstaje z dźwięku (`Heard` dziś, `Ears` w torze B);
- `judge` — reguły zaliczania, bez wiedzy o ćwiczeniu poza „o co proszę teraz”;
- `exercise` — generator kroków każdego trybu: co jest celem, co po zaliczeniu, co po strzałce;
- `session` — ustawienia w locie, biblioteka, pauza, dziennik.

Styk ćwiczenia z sędzią:

```rust
pub trait Exercise {
    fn asked(&self) -> Asked;            // dźwięki albo akord, kolejność, czy pojedynczo
    fn credited(&mut self, what: Credit) -> Progress;
    fn step(&mut self, by: i32);         // strzałki
}
```

Robić metodą z refaktoryzacji 0.5.12 (`raport-refaktoryzacji`): siatka wyników przed, jeden
ruch, siatka po, identyczne. To jest warunek S1.4 i S2.4.

### K2 Ustawienia trybu jako jedna struktura

Dziś opcja trybu to: pole w `Settings`, pole w `MyApp`, właściwość w Slincie, wywołanie zwrotne
w `controls.rs`, linia w `main.rs` przy starcie i druga przy zapisie migawki. Dodanie
„Numerów progów” do paska dotknęło pięciu plików. Jedna struktura na tryb z `serde`, jedna
funkcja „z ustawień do programu i okna”, jedna „z okna do ustawień”.

### K3 `ui_frame.rs`

2000 linii zamiany stanu na właściwości okna, z podpisami pamięci podręcznej składanymi z
`format!` (błąd ze strzałkami w Skalach wziął się z brakującego pola w takim podpisie).
Zamienić napisy na struktury z `#[derive(PartialEq)]`, po jednej na rysunek, i podzielić plik
według paneli.

### K4 Stałe sędziego w jednym miejscu

Stałe są rozsiane po `state.rs`, `judge.rs`, `credits.rs`, `strike.rs`, `flux.rs`. Jedna
struktura `JudgeTuning` z wartościami domyślnymi i odsyłaczem do sekcji w
`judge-measurements.md` przy każdej; harness może wtedy przemiatać wartość bez rekompilacji.

## 7. Grupa I — infrastruktura

- **I1 Materiał pomiarowy poza katalogiem domowym.** Prywatne repozytorium danych na HF albo
  katalog obok projektu; skrypt pobierający; `SOLITITO_MATERIAL`. Bez tego tablicy wyników nie
  uruchomi nikt poza autorem na jego komputerze.
- **I2 Wydania.** Tag `v0.5.17` był przesuwany kilkanaście razy. Proponowane: numer rośnie przy
  każdym wydaniu, a wersje próbne dostają `-betaN` (tak było do 0.5.13). Decyzja autora.
- **I3 Test na Windows w CI** już jest (`cargo test --release`). Dodać uruchomienie
  `solitito --check` i `--bench` na obu systemach, żeby brak pliku modelu albo wolny model
  wyszedł przed wydaniem.
- **I4 Rejestr modeli.** Plik `models.toml`: nazwa, skrót SHA-256, `RUN_TAG`, wynik na tablicy
  w dniu wpięcia. `--check` porównuje skrót. Dziś nazwa pliku jest wpisana w `strike.rs`,
  `main.rs` i `release.yml` osobno.
- **I5 Dokumentacja.** Po każdej grupie: `docs/how-it-works.md` (dziś opisuje reguły, które tor
  B usuwa), `docs/settings.md`, zrzuty. Kursy w `~/Projekty/Rust/Claude-lessons` odsyłają do
  kodu w `src/` i zestarzeją się przy K1 — uprzedzić autora.

## 8. Kolejność

```
tor A:  S0 tablica wyników ─ S1.5 model rzadziej ─ S1.1 jądra ─ S1.2 bramka ─ pomiar gubionych
        skoków ─ S1.3 zgodność cech ─ [K1 ─ K4 ─ S1.4, jeśli skoki giną] ─ wydanie i test

tor B:  (po torze A)  C7 nagrania testowe ─ sprawdzian kosztu ─ S2.1 dane v3 ─ S2.2 model
                      ─ S2.3 wykonanie ─ S2.4 sędzia v2, tryb po trybie ─ S2.5 akordy

obok:   U1 podgląd słyszenia (od razu, na silniku klasycznym)
        C1.1 wyjście dźwięku ─ C1 metronom ─ C2 takty ─ C5 słuch
        C3 dziennik ─ U3 postępy ─ C6 plan ćwiczenia (po K2)
        U6 podział .slint ─ U5 układ   ·   U2, U4, U7 w dowolnej chwili
```

**Etap 1** (podstawa, bez zmian widocznych dla gracza): S0, I1, przeniesienie harnessów z
`experiments`.

**Etap 2** (tor A, kolejność z „Planu działań toru A”): A1 model rzadziej, A2 jądra, A3 bramka,
A4 pomiar gubionych skoków, A5 zgodność cech. Każde osobno, każde z liczbą. Obok: U1, U4, U7.

**Etap 3** (tylko jeśli A4 pokaże stratę): K1, K4, S1.4. Na koniec wydanie z samym torem A i
test autora na laptopie i cienkim kliencie.

**Etap 4** (nowy silnik, dopiero po etapie 3): C7, S2.1, S2.2, S2.3, S2.4 tryb po trybie,
potem S2.5. Przed treningiem sprawdzian kosztu: sieć o docelowym kształcie z losowymi wagami,
wyeksportowana do ONNX i zmierzona przez `--bench` na najsłabszym komputerze autora. Jeśli
nie mieści się w budżecie, zmniejsza się sieć, zanim cokolwiek się wytrenuje.

**Etap 5** (ćwiczenia): C1, C2, C3 z U3, potem C4, C5, C6. C1–C3 nie czekają na etap 4, jeśli
autor chce je wcześniej; ocena punktualności będzie wtedy na silniku klasycznym mniej dokładna.

**Etap 6**: U6, U5, K2, K3, S2.6.

## 9. Ryzyka

| ryzyko | co wtedy |
|---|---|
| Model śledzący nie nazywa wszystkich strun akordu mimo danych hex | tor B staje na progach z S2.4; zostaje tor A i opcja „bez Tylko uderzone” jako najlepsze, co jest |
| Etykiety początków nut w GuitarSet są przesunięte względem ataku | korekta na ścieżce struny (S2.1.1), a nagrania testowe autora jako sprawdzian |
| Model uczony bez nagrań autora nie pasuje do jego gitar i kart | nagrania testowe autora (C7) są progiem przyjęcia; szerokie wzbogacanie poziomu, stroju i tonu; test na graczu 05 z GuitarSet; jeśli różnica zostaje — tor B staje |
| Model śledzący nie mieści się w skoku na cienkim kliencie | sprawdzian kosztu przed treningiem (etap 4); mniejsza sieć; ostatecznie liczenie co drugi skok (32 ms) z odpowiednio gorszym opóźnieniem |
| S1.4 zmienia zachowanie na żywo w sposób niewidoczny w harnessach | nowa miara „skoki niewidziane”; wersja próbna do testu z gitarą przed scaleniem |
| Dwa silniki obok siebie komplikują kod na miesiące | interfejs `Ears` jest jedynym stykiem; po przyjęciu toru B stary sędzia jest usuwany w jednym commicie |
| Metronom z głośników psuje zaliczanie przy mikrofonie | pomiar przed wydaniem; słuchawki jako zalecenie; w torze B klik jest dźwiękiem spoza zakresu gitary |

## 10. Decyzje dla autora

Rozstrzygnięte 15.11.2026: najpierw tor A, potem tor B; autor nie nagrywa danych
treningowych, może nagrać testowe.

Otwarte:

1. Wynik `solitito --bench` z cienkiego klienta i z laptopa — potrzebny, zanim ruszy S1.
2. Progi przyjęcia z S2.4 — czy takie, czy inne.
3. `rtrb` jako zależność, czy własny bufor.
4. Czy model akordów może w końcu zniknąć z programu (S2.5), czy zostaje jako drugi świadek.
   Od tego zależy, czy tor B zmniejszy zużycie procesora w Akordach.
5. Która grupa ćwiczeń najpierw: rytm (C1–C2), postępy (C3), słuch (C5).
6. Numeracja wydań (I2).
7. Czy autor wykona jeszcze jeden render syntetyczny w DAW, jeśli pomiar pokaże, że jest
   potrzebny (S2.1.3).
