# 🎸 Solitito – Real-Time Polyphonic Guitar Trainer

**Solitito** (no pun intended) is an experimental, real-time, polyphonic guitar trainer built in **Rust** during a 5-hour vibe-coding session with *Gemini 3 Pro Preview*. It detects **chords** and **scales** using FFT-based audio analysis and helps you practice **intervals** and fretboard knowledge.

It's a proof-of-concept **experiment** - heavily inspired by another, amazing Android/iOS app - Solo. I just want to experiment a little bit with FFT and soon also some pre-trained neural networks for chords detection, so as many of my experiments, this little project might just be left as is at some point. I don't intent to create alternative to Solo ;) which I use daily.  
<div align="center">
<img width="284" height="450" alt="solitito0" src="https://github.com/user-attachments/assets/5cb1a334-95d1-4586-95e9-b91671d51b1e" />
</div>
---

## ⭐ Key Features

### 🎼 Modes
- **Songs** — chord progressions  
- **Scales** — sequential practice  
- **Random** — ear training & fretboard awareness

### 🎧 DSP / Audio
- **Polyphonic chord detection**  
- **Stale-Note Filtering** — prevents sustaining notes from triggering new chords  
- **Optimized for laptop microphones** (Bass Boost, Sensitivity)

### 📁 Custom Content
- Load your own **songs** and **scales** from simple text files  
- No restart required

---

## ⚙️ Settings (Gear Icon)
<div align="center">
<img width="284" height="450" alt="solitito1" src="https://github.com/user-attachments/assets/8115351b-b5be-41f4-a77d-796b1e8fa383" />
</div>
<br> 

| Setting        | Description |
|----------------|-------------|
| **Threshold**      | Minimum volume required to detect a note |
| **Tail Release**   | How much a string must decay before it can be triggered again |
| **Input Delay**    | Grace period after a chord change (prevents noise while moving fingers) |
| **Bass Boost**     | Digital amplification for low strings (useful for laptop mics) |
| **Intervals**      | What intervals to practice (e.g. `1 3 5` for triads, `1 3 5 7` for sevenths, 3 or 5 shows both 3 and b3, or 5 and b5 according to the chord quality) |

---

## 📄 Custom Files Format

`user_songs.txt`  
My Song Title  
Cm7 F7 BbMaj7

`user_scales_def.txt`  
My Scale Name  
1 b2 3 4 5 b6 7

---

# 🇵🇱 Solitito – Trener gitarowy w czasie rzeczywistym

**Solitito** to eksperymentalny polifoniczny program do ćwiczeń gitarowych, stworzony w **Rust** podczas 5-godzinnej sesji vibe-coding z *Gemini 3 Pro Preview*. Rozpoznaje **akordy** i **skale** wykorzystując analizę FFT, pomagając w ćwiczeniu **interwałów** i **znajomości gryfu**.

Jest to projekt **eksperymentalny**, mający na celu sprawdzenie moich szalonych pomysłów – w dużej mierze zainspirowany inną, niesamowitą aplikacją na Androida/iOS – Solo. Chcę po prostu trochę poeksperymentować z FFT i wkrótce również z wstępnie wytrenowanymi sieciami neuronowymi do wykrywania akordów, więc podobnie jak wiele moich eksperymentów, ten mały projekt może po prostu pozostać w obecnej formie. Nie zamierzam tworzyć alternatywy dla Solo ;), którego używam na codzień.    
<div align="center">
<img width="284" height="450" alt="solitito0" src="https://github.com/user-attachments/assets/5cb1a334-95d1-4586-95e9-b91671d51b1e" />
</div>
---

## ⭐ Kluczowe funkcje

### 🎼 Tryby
- **Utwory** — progresje akordów  
- **Skale** — sekwencyjne ćwiczenie  
- **Random** — trening słuchu i gryfu

### 🎧 DSP / Audio
- **Polifoniczna detekcja akordów**  
- **Filtrowanie „ogonów”** — wybrzmiewające nuty nie psują detekcji nowych akordów  
- **Optymalizacja pod mikrofony laptopowe** (podbicie basu, czułość)

### 📁 Własne treści
- Ładowanie własnych **utworów** i **skal** z prostych plików tekstowych  
- Bez konieczności restartu aplikacji

---

## ⚙️ Ustawienia (ikonka zębatki)
<div align="center">
<img width="284" height="450" alt="solitito1" src="https://github.com/user-attachments/assets/8115351b-b5be-41f4-a77d-796b1e8fa383" />
</div>
<br> 

| Ustawienie      | Opis |
|------------------|------|
| **Threshold**        | Minimalna głośność potrzebna do wykrycia nuty |
| **Tail Release**     | Jak mocno musi wybrzmieć struna, aby system uznał ją za „nową” |
| **Input Delay**      | Czas niewykrywania po zmianie akordu (na ustawienie palców) |
| **Bass Boost**       | Cyfrowe wzmocnienie niskich częstotliwości |
| **Intervals**        | Jakie interwały ćwiczymy (np. `1 3 5` lub `1 3 5 7`), 3 lub 5 pokaże zarówno 3, 5 jak i b3 lub b5, etc, zależnie od typu akordu |

---

## 📄 Format własnych plików

`user_songs.txt:`  
My Song Title  
Cm7 F7 BbMaj7

`user_scales_def.txt:`  
My Scale Name  
1 b2 3 4 5 b6 7

