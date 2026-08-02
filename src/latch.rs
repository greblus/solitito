//! Zatrzask jakości akordu ("lock chord qualities until new attack").
//!
//! # Po co
//!
//! Septyma wybrzmiewa szybciej niż pryma, tercja i kwinta. Widać to wprost
//! w podglądzie diagnostycznym trzymanego `Gm7`:
//!
//! ```text
//! G m7 | min7=96% | b7=96      <- zaraz po uderzeniu
//! G m7 | min7=82% | b7=76
//! G m7 | min7=52% | b7=52
//! G m  | min=49%  | b7=45      <- septyma ucichła, model zmienia zdanie
//! ```
//!
//! Model ma rację: w oknie 0.77 s septymy naprawdę już nie ma. Ale akord
//! **nie zmienia tożsamości, kiedy wybrzmiewa** — dopóki trzymasz chwyt i nie
//! szarpnąłeś ponownie, to nadal `Gm7`. Tej wiedzy model mieć nie może, bo widzi
//! wyłącznie bieżące okno. Aplikacja może.
//!
//! # Zasada
//!
//! Zatrzask **zakłada się** na wysokiej pewności, ale **trzyma niezależnie od
//! niej**. To rozróżnienie jest istotne: w zaniku model raportuje `min` z pewnością
//! 94–96%, więc próg pewności sam z siebie niczego by nie uratował. Zwolnić
//! zatrzask może tylko nowy atak albo zmiana prymy.
//!
//! Zakładamy go dopiero, gdy okno kontekstowe zdążyło się wypełnić dźwiękiem po
//! ataku (`CTX_FRAMES` klatek). Wcześniej okno zawiera jeszcze ogon POPRZEDNIEGO
//! akordu i zatrzask złapałby cudzą nazwę.

/// Minimalna pewność potrzebna do ZAŁOŻENIA zatrzasku.
pub const LOCK_MIN_CONF: f32 = 0.60;

/// Ile klatek po ataku trzeba odczekać, zanim okno kontekstowe jest w całości
/// wypełnione nowym akordem. Równe `audio::CTX_FRAMES`.
pub const SETTLE_FRAMES: u32 = 48;

#[derive(Default)]
pub struct ChordLatch {
    locked: Option<String>,
    last_onset: u64,
}

impl ChordLatch {
    /// Zwraca akord do pokazania i użycia w logice ćwiczenia.
    ///
    /// * `enabled` — przełącznik z UI; przy `false` działa jak przezroczysta rura
    /// * `onset_id` — licznik ataków z `AudioAnalysis`; zmiana zwalnia zatrzask
    /// * `frames_since_onset` — ile klatek od ataku
    /// * `chord` / `conf` — bieżąca predykcja (już po głosowaniu)
    pub fn update(
        &mut self,
        enabled: bool,
        onset_id: u64,
        frames_since_onset: u32,
        chord: &str,
        conf: f32,
    ) -> String {
        if !enabled {
            self.locked = None;
            self.last_onset = onset_id;
            return chord.to_string();
        }

        // Nowy atak = nowy akord. Zwalniamy bezwarunkowo.
        if onset_id != self.last_onset {
            self.last_onset = onset_id;
            self.locked = None;
        }

        if let Some(held) = &self.locked {
            // Zmiana prymy bez wykrytego ataku (np. przejście legato albo atak
            // zbyt cichy dla detektora) też musi zwalniać — inaczej apka
            // pokazywałaby poprzedni akord w nieskończoność.
            if root_of(chord) != root_of(held) && conf >= LOCK_MIN_CONF {
                self.locked = Some(chord.to_string());
                return chord.to_string();
            }
            return held.clone();
        }

        // Zakładamy dopiero, gdy okno jest w całości po ataku.
        if frames_since_onset >= SETTLE_FRAMES && conf >= LOCK_MIN_CONF && is_real(chord) {
            self.locked = Some(chord.to_string());
        }
        chord.to_string()
    }

    /// Co jest aktualnie trzymane (do podglądu / testów).
    pub fn held(&self) -> Option<&str> {
        self.locked.as_deref()
    }
}

/// Pryma z napisu predykcji ("G m7" -> "G", "Note F" -> "F").
fn root_of(chord: &str) -> &str {
    let mut it = chord.split_whitespace();
    match it.next() {
        Some("Note") => it.next().unwrap_or(""),
        Some(r) => r,
        None => "",
    }
}

/// Czy to w ogóle akord, na który warto zakładać zatrzask.
fn is_real(chord: &str) -> bool {
    !chord.is_empty() && chord != "Noise" && chord != "..." && !chord.starts_with("Note")
}

#[cfg(test)]
mod tests {
    use super::*;

    const AFTER: u32 = SETTLE_FRAMES;

    #[test]
    fn trzyma_septyme_gdy_wybrzmiewa() {
        // Odtworzony przebieg z podglądu diagnostycznego: model traci septymę,
        // ale zachowuje wysoką pewność dla ubozszej jakości.
        let mut l = ChordLatch::default();
        assert_eq!(l.update(true, 1, AFTER, "G m7", 0.95), "G m7");
        for conf in [0.94, 0.95, 0.96] {
            assert_eq!(
                l.update(true, 1, AFTER + 10, "G m", conf), "G m7",
                "zatrzask ma trzymać mimo wysokiej pewności ubozszej jakości"
            );
        }
    }

    #[test]
    fn nowy_atak_zwalnia() {
        let mut l = ChordLatch::default();
        l.update(true, 1, AFTER, "G m7", 0.95);
        assert_eq!(l.update(true, 1, AFTER + 5, "G m", 0.95), "G m7");
        // szarpnięcie strun -> onset_id rośnie
        assert_eq!(l.update(true, 2, AFTER, "G m", 0.95), "G m");
        assert_eq!(l.held(), Some("G m"));
    }

    #[test]
    fn nie_lapie_ogona_poprzedniego_akordu() {
        // Tuż po ataku okno wciąż zawiera poprzedni akord — nie wolno zatrzasnąć.
        let mut l = ChordLatch::default();
        for f in [0u32, 10, 30, 47] {
            l.update(true, 1, f, "C m7", 0.97);
            assert_eq!(l.held(), None, "przy {f} klatkach okno nie jest jeszcze pełne");
        }
        l.update(true, 1, AFTER, "G m7", 0.97);
        assert_eq!(l.held(), Some("G m7"));
    }

    #[test]
    fn niska_pewnosc_nie_zaklada() {
        let mut l = ChordLatch::default();
        l.update(true, 1, AFTER, "G m7", 0.30);
        assert_eq!(l.held(), None);
        l.update(true, 1, AFTER, "G m7", LOCK_MIN_CONF);
        assert_eq!(l.held(), Some("G m7"));
    }

    #[test]
    fn zmiana_prymy_bez_ataku_tez_zwalnia() {
        let mut l = ChordLatch::default();
        l.update(true, 1, AFTER, "G m7", 0.95);
        // przejście na inny akord, detektor ataku go nie złapał
        assert_eq!(l.update(true, 1, AFTER + 20, "C m7", 0.95), "C m7");
        assert_eq!(l.held(), Some("C m7"));
    }

    #[test]
    fn wylaczony_jest_przezroczysty() {
        let mut l = ChordLatch::default();
        assert_eq!(l.update(false, 1, AFTER, "G m7", 0.95), "G m7");
        assert_eq!(l.update(false, 1, AFTER + 5, "G m", 0.95), "G m");
        assert_eq!(l.held(), None);
    }

    #[test]
    fn szum_i_pojedyncze_dzwieki_nie_zatrzaskuja() {
        let mut l = ChordLatch::default();
        l.update(true, 1, AFTER, "Noise", 0.95);
        assert_eq!(l.held(), None);
        l.update(true, 1, AFTER, "Note F", 0.95);
        assert_eq!(l.held(), None, "pojedynczy dźwięk to nie akord do trzymania");
    }

    #[test]
    fn root_of_radzi_sobie_z_formatami() {
        assert_eq!(root_of("G m7"), "G");
        assert_eq!(root_of("C"), "C");
        assert_eq!(root_of("Note F#"), "F#");
        assert_eq!(root_of(""), "");
    }
}
