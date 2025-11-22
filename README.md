# 🎸 Solitito – Real-Time Polyphonic Guitar Trainer

**Solitito** is an experimental, real-time, polyphonic guitar trainer built in **Rust** during a 5-hour vibe-coding session with *Gemini 3 Pro Preview*.  
It detects **chords** and **scales** using FFT-based audio analysis and helps you practice **intervals** and fretboard knowledge.

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

| Setting        | Description |
|----------------|-------------|
| **Threshold**      | Minimum volume required to detect a note |
| **Tail Release**   | How much a string must decay before it can be triggered again |
| **Input Delay**    | Grace period after a chord change (prevents noise while moving fingers) |
| **Bass Boost**     | Digital amplification for low strings (useful for laptop mics) |
| **Intervals**      | What intervals to practice (e.g. `1 3 5` for triads, `1 3 5 7` for sevenths) |

---

## 📄 Custom Files Format

### `user_songs.txt`
My Song Title
Cm7 F7 BbMaj7

### `user_scales_def.txt`
My Scale Name
1 b2 3 4 5 b6 7

---

# 🇵🇱 Solitito – Trener gitarowy w czasie rzeczywistym

**Solitito** to eksperymentalny polifoniczny program do ćwiczeń gitarowych, stworzony w **Rust** podczas 5-godzinnej sesji vibe-coding z *Gemini 3 Pro Preview*. Rozpoznaje **akordy** i **skale** wykorzystując analizę FFT, pomagając w ćwiczeniu **interwałów** i **znajomości gryfu**.

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

| Ustawienie      | Opis |
|------------------|------|
| **Threshold**        | Minimalna głośność potrzebna do wykrycia nuty |
| **Tail Release**     | Jak mocno musi wybrzmieć struna, aby system uznał ją za „nową” |
| **Input Delay**      | Czas niewykrywania po zmianie akordu (na ustawienie palców) |
| **Bass Boost**       | Cyfrowe wzmocnienie niskich częstotliwości |
| **Intervals**        | Jakie interwały ćwiczymy (np. `1 3 5` lub `1 3 5 7`) |

---

## 📄 Format własnych plików

### `user_songs.txt`
My Song Title
Cm7 F7 BbMaj7

### `user_scales_def.txt`
My Scale Name
1 b2 3 4 5 b6 7

