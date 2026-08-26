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
kończąca się powtórzoną prymą — musi zostać zagrane dwa razy. To, co wciąż brzmi z
poprzedniego razu, pasuje w chwili, w której program prosi o to ponownie, więc zaliczenie
wymaga świeżego uderzenia: odpowiedź głowicy ataków dla danego dźwięku musi przekroczyć
0,60.

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
  zostało zagrane. Rozstrzyga natomiast, czy coś zostało uderzone **ponownie**: dźwięk
  wymagany dwa razy z rzędu potrzebuje własnego uderzenia, a wykrywacz obwiedni nie potrafi
  go dostarczyć — jego poziom to RMS okna 512 ms, więc drugie szarpnięcie brzmiącej struny
  prawie go nie podnosi. Zmierzone na materiale generowanym: obwiednia złapała 2 powtórzenia
  z 6, głowica wszystkie sześć. Starszy, trójgłowicowy model
  wciąż działa: nazwy trzech pierwszych wyjść się nie zmieniły.

---
