Solitito is a real-time, polyphonic guitar trainer built in Rust during 5-hours of vibe-coding experiment with Gemini 3. It detects chords and scales using audio analysis (FFT) and helps you practice intervals on your instrument.

KEY FEATURES
    Modes: Songs (chord progressions), Scales (sequential), Random (ear training).
    DSP: Polyphonic detection with "Stale Note" filtering (prevents sustaining notes from triggering new chords).
    Input: Optimized for laptop mics (Bass Boost, Sensitivity).
    Custom Content: Load your own songs and scales via text files.

SETTINGS (Gear Button)
    Threshold: Minimum volume to trigger a note.
    Tail Release: How much a string must quiet down to be played again (prevents accidental re-triggers).
    Input Delay: Grace period after chord change (prevents noise while moving hands).
    Bass Boost: Digital amplification for low strings (essential for built-in mics).
    Intervals: What to practice (e.g., "1 3 5" for triads, "1 3 5 7" for sevenths).

CUSTOM FILES FORMAT

Example content for user_songs.txt:
My Song Title
Cm7 F7 BbMaj7

Example content for user_scales_def.txt:
My Scale Name
1 b2 3 4 5 b6 7

Solitito to trener gitarowy czasu rzeczywistego napisany w Rust w trakcie 5-cio godzinnej sesji vibe-coding z Gemini 3. Wykrywa akordy i skale za pomocą analizy audio (FFT), pomagając w ćwiczeniu interwałów na instrumencie.

KLUCZOWE FUNKCJE
    Tryby: Utwory (progresje akordów), Skale (sekwencje), Random (trening słuchu i gryfu).
    DSP: Polifoniczna detekcja z filtrowaniem "ogonów" (wybrzmiewające nuty nie psują detekcji w nowym akordzie).
    Wejście: Zoptymalizowane pod mikrofony laptopowe (podbicie basu, regulacja czułości).
    Własne treści: Ładowanie utworów i skal z plików tekstowych.

USTAWIENIA (Przycisk zębatki)
    Threshold: Minimalna głośność, by wykryć nutę.
    Tail Release: Jak mocno trzeba wytłumić strunę, by system pozwolił zagrać ją ponownie (blokada sustainu).
    Input Delay: Czas "bez wykrywania" po zmianie akordu (czas na ułożenie ręki na gryfie).
    Bass Boost: Cyfrowe wzmocnienie niskich tonów (niezbędne dla mikrofonów wbudowanych w laptopy).
    Intervals: Co ćwiczymy (np. wpisz "1 3 5" dla trójdźwięków, "1 3 5 7" dla akordów septymowych).

FORMAT WŁASNYCH PLIKÓW

Przykładowa zawartość user_songs.txt:
Tytuł Utworu
Cm7 F7 BbMaj7

Przykładowa zawartość user_scales_def.txt:
Nazwa Skali
1 b2 3 4 5 b6 7

