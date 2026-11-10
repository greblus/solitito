//! The colour schemes of the window, for what Rust colours itself: the
//! verdict on the chord's name and on the strip, the degrees of the line,
//! and the drawings. The rest of the window takes its colours from `Theme`
//! in `appwindow.slint`, written by `dist/color_schemes.py` in the same
//! order of schemes.

use slint::Color;

use crate::state::{MatchStatus, NoteState};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Scheme {
    #[default]
    Graphite,
    Light,
    Sepia,
    Night,
    Forest,
    Contrast,
}

/// The schemes in the order of the setting.
pub const SCHEMES: [Scheme; 6] =
    [Scheme::Graphite, Scheme::Light, Scheme::Sepia, Scheme::Night, Scheme::Forest, Scheme::Contrast];

impl Scheme {
    pub fn from_index(index: usize) -> Scheme {
        SCHEMES.get(index).copied().unwrap_or_default()
    }

    /// Dark text on a pale ground: the verdicts in their darker colours.
    pub fn is_light(self) -> bool {
        matches!(self, Scheme::Light | Scheme::Sepia)
    }

    /// The colours a drawing made for Graphite is given here: the card, the
    /// lines and dots, the grey of the notes around the line and its labels,
    /// the inlays, and what is white with nothing under it - the ring round
    /// the note due and the name of an open string.
    fn drawing_colours(self) -> DrawingColours {
        let graphite = DrawingColours {
            card: crate::diagrams::CARD,
            lines: None,
            ghost: crate::tab::GHOST,
            ghost_label: crate::tab::GHOST_LABEL,
            inlay: crate::tab::INLAY,
            unfilled: "#ffffff",
        };
        let pale = DrawingColours {
            card: "rgba(255, 255, 255, 1)",
            lines: None,
            ghost: "#D3D8E1",
            ghost_label: "#5E6676",
            inlay: "rgba(60, 70, 90, 0.18)",
            unfilled: "#1B2030",
        };
        match self {
            Scheme::Graphite => graphite,
            Scheme::Light => pale,
            Scheme::Sepia => DrawingColours {
                card: "rgba(251, 246, 236, 1)",
                lines: Some((154, 91, 46)),
                ghost: "#DED8CE",
                ghost_label: "#6E655A",
                unfilled: "#3A2E22",
                ..pale
            },
            Scheme::Night => DrawingColours {
                card: "rgba(15, 24, 48, 1)",
                ghost: "#363E53",
                ghost_label: "#858C9C",
                ..graphite
            },
            Scheme::Forest => DrawingColours {
                card: "rgba(17, 29, 25, 1)",
                // Bluer than the accent: the green of a note played stays apart.
                lines: Some((56, 168, 196)),
                ghost: "#37433E",
                ghost_label: "#84908B",
                ..graphite
            },
            Scheme::Contrast => DrawingColours {
                card: "rgba(10, 10, 10, 1)",
                lines: Some((90, 170, 255)),
                ghost: "#363636",
                ghost_label: "#B0B0B0",
                inlay: "rgba(255, 255, 255, 0.3)",
                ..graphite
            },
        }
    }
}

struct DrawingColours {
    card: &'static str,
    /// The blue of the lines and dots, where a scheme has its own.
    lines: Option<(u8, u8, u8)>,
    ghost: &'static str,
    ghost_label: &'static str,
    inlay: &'static str,
    /// The ring round the note due and the name of an open string.
    unfilled: &'static str,
}

/// The big name in the colour of the verdict: green, amber, red, or plain.
pub fn verdict_colour(status: MatchStatus, scheme: Scheme) -> Color {
    let (r, g, b) = match (scheme.is_light(), status) {
        (false, MatchStatus::Exact) => (50, 255, 50),
        (false, MatchStatus::Partial) => (255, 220, 50),
        (false, MatchStatus::Flicker) => (255, 50, 50),
        (false, MatchStatus::None) => (255, 255, 255),
        (true, MatchStatus::Exact) => (21, 138, 62),
        (true, MatchStatus::Partial) => (176, 122, 0),
        (true, MatchStatus::Flicker) => (200, 50, 58),
        (true, MatchStatus::None) => (27, 32, 48),
    };
    Color::from_rgb_u8(r, g, b)
}

/// A chord of the strip in the colour it was passed on, grey if not yet.
pub fn chip_colour(status: MatchStatus, scheme: Scheme) -> Color {
    let (r, g, b) = match (scheme.is_light(), status) {
        (false, MatchStatus::Exact) => (110, 231, 160),
        (false, MatchStatus::Partial) => (255, 220, 50),
        (false, MatchStatus::Flicker) => (255, 90, 90),
        (false, MatchStatus::None) => (138, 146, 163),
        (true, MatchStatus::Exact) => (21, 127, 67),
        (true, MatchStatus::Partial) => (168, 107, 0),
        (true, MatchStatus::Flicker) => (196, 47, 56),
        (true, MatchStatus::None) => (99, 107, 125),
    };
    Color::from_rgb_u8(r, g, b)
}

/// A degree of the line: played, being held, open or due, still to come.
pub fn note_colour(state: NoteState, scheme: Scheme) -> Color {
    let (r, g, b) = match (scheme.is_light(), state) {
        (false, NoteState::Done) => (50, 255, 50),
        (false, NoteState::Answering) => (200, 255, 50),
        (false, NoteState::Open | NoteState::Due) => (180, 180, 180),
        (false, NoteState::Waiting) => (60, 60, 60),
        (true, NoteState::Done) => (21, 138, 62),
        (true, NoteState::Answering) => (111, 143, 0),
        (true, NoteState::Open | NoteState::Due) => (60, 66, 80),
        (true, NoteState::Waiting) => (190, 195, 205),
    };
    Color::from_rgb_u8(r, g, b)
}

/// A drawing made for Graphite, in `scheme`. The red root, the green of a
/// note played and the white labels on the dots stay in every scheme.
pub fn drawing(svg: &str, scheme: Scheme) -> String {
    if scheme == Scheme::Graphite {
        return svg.to_string();
    }
    let c = scheme.drawing_colours();
    let mut out = svg
        .replace(crate::diagrams::CARD, c.card)
        .replace(crate::tab::GHOST, c.ghost)
        .replace(crate::tab::GHOST_LABEL, c.ghost_label)
        .replace(crate::tab::INLAY, c.inlay)
        .replace(crate::tab::OPEN_LABEL, c.unfilled)
        .replace("stroke=\"#ffffff\"", &format!("stroke=\"{}\"", c.unfilled));
    if let Some((r, g, b)) = c.lines {
        out = out
            .replace("rgba(74, 144, 226, 1)", &format!("rgba({r}, {g}, {b}, 1)"))
            .replace("rgba(74, 144, 226, 0.55)", &format!("rgba({r}, {g}, {b}, 0.55)"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Graphite is the drawing as made; Light leaves no dark card and no
    /// white ring, and keeps the white label on a dot.
    #[test]
    fn a_drawing_turns_light() {
        let svg = format!(
            "<rect fill=\"{}\"/><circle fill=\"{}\" stroke=\"#ffffff\"/><text fill=\"#ffffff\">1</text>",
            crate::diagrams::CARD,
            crate::tab::GHOST
        );
        assert_eq!(drawing(&svg, Scheme::Graphite), svg);
        let light = drawing(&svg, Scheme::Light);
        assert!(!light.contains(crate::diagrams::CARD) && !light.contains(crate::tab::GHOST), "{light}");
        assert!(!light.contains("stroke=\"#ffffff\""), "the ring stayed white: {light}");
        assert!(light.contains("fill=\"#ffffff\">1"), "the label on the dot lost its white: {light}");
    }

    /// The name in an open string's empty circle is dark on the light card.
    #[test]
    fn an_open_string_keeps_its_name_on_the_light_card() {
        let svg = crate::tab::neck_hint(&[0, 1, 2, 3, 4, 5], &[(0, 0)], "E");
        assert!(svg.contains(crate::tab::OPEN_LABEL), "{svg}");
        let light = drawing(&svg, Scheme::Light);
        assert!(!light.contains(crate::tab::OPEN_LABEL) && light.contains("fill=\"#1B2030\">E<"), "{light}");
    }

    /// Every scheme draws on its own card, and those with lines of their own
    /// leave none of the blue.
    #[test]
    fn every_scheme_draws_on_its_card() {
        let svg = crate::tab::neck_hint(&[0, 1, 2, 3, 4, 5], &[(0, 3)], "G");
        for scheme in SCHEMES.into_iter().skip(1) {
            let drawn = drawing(&svg, scheme);
            assert!(!drawn.contains(crate::diagrams::CARD), "{scheme:?}");
            if scheme.drawing_colours().lines.is_some() {
                assert!(!drawn.contains("74, 144, 226"), "{scheme:?} kept the blue");
            }
        }
    }
}
