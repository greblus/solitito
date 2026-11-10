"""Writes the colour schemes into the `Theme` global of ui/appwindow.slint.

Graphite and Light are given colour by colour: they are the two looks the
window was designed in. The others are derived from a handful of base
colours - the window, the cards, the raised controls, the text and the
accent, and the colours of a verdict - by the same mixes those two are
built from: a border is the card with a little of the text in it, a muted
label the text with some of the card, and so on.

Usage: python dist/color_schemes.py
Then run the tests: src/theme.rs holds the colours Rust draws with, in the
same order of schemes.
"""
from pathlib import Path

SLINT = Path(__file__).resolve().parent.parent / "ui/appwindow.slint"

# The schemes in the order of the setting.
NAMES = ["graphite", "light", "sepia", "night", "forest", "contrast"]

GRAPHITE = {
    "window": "#11141A", "panel": "#161A22", "raised": "#1C212C", "chip": "#1C212C",
    "raised-hover": "#222836", "raised-hover-2": "#232836", "line": "#222835",
    "line-strong": "#262D3B", "field-line": "#2D3444", "field-line-hover": "#3A4356",
    "track": "#2A3142", "selected": "#243049", "now": "#1A2640", "open-line": "#2D3B5C",
    "accent": "#2F6FE0", "accent-line": "#5A8EF0", "meter": "#4A90E2", "link": "#6AA9FF",
    "link-dim": "#4A7FD0", "paused-text": "#A9C9FF", "text-strong": "#FFFFFF",
    "text": "#E8EAF0", "text-2": "#C9CEDA", "muted": "#A9B0C0", "dim": "#8A92A3",
    "faint": "#6B7385", "card-off": "#4A5164", "ok": "#6EE7A0", "ok-bg": "#17271F",
    "ok-line": "#24563A", "ok-chord-bg": "#1E3A2C", "holding": "#D4F27A", "warn": "#F2B233",
    "danger": "#FF7B7B", "label": "#AAAAAA", "hint": "#777777", "hint-2": "#666666",
    "grey": "#888888", "grey-2": "#999999", "grey-3": "#8A8A8A", "bright": "#DDDDDD",
    "bright-2": "#BBBBBB", "off": "#555555", "rule": "#444444", "invert-bg": "#E8E8E8",
    "invert-line": "#3A3A3A", "invert-ink": "#101010", "overlay": "#000000E0",
    "on-accent": "#FFFFFF", "on-accent-2": "#DCE7FF",
}
LIGHT = {
    "window": "#EEF0F4", "panel": "#FFFFFF", "raised": "#E8EBF1", "chip": "#FFFFFF",
    "raised-hover": "#DDE2EA", "raised-hover-2": "#DDE2EA", "line": "#D5DAE3",
    "line-strong": "#C9CFDA", "field-line": "#BFC6D2", "field-line-hover": "#9AA3B5",
    "track": "#C5CCD8", "selected": "#DCE6FA", "now": "#E4EDFD", "open-line": "#9DB6EA",
    "accent": "#2F6FE0", "accent-line": "#2F6FE0", "meter": "#2F6FE0", "link": "#2563C9",
    "link-dim": "#3A6FC0", "paused-text": "#4A6FB0", "text-strong": "#0F1219",
    "text": "#1B2030", "text-2": "#333A49", "muted": "#4F5769", "dim": "#636B7D",
    "faint": "#8C93A3", "card-off": "#A7AEBB", "ok": "#157F43", "ok-bg": "#E1F3E8",
    "ok-line": "#8CCBA6", "ok-chord-bg": "#D3EEDD", "holding": "#5F7F00", "warn": "#A86B00",
    "danger": "#C42F38", "label": "#474D58", "hint": "#6B7180", "hint-2": "#727887",
    "grey": "#656B77", "grey-2": "#5E6470", "grey-3": "#6A707C", "bright": "#20242C",
    "bright-2": "#2E333C", "off": "#B0B6C0", "rule": "#C9CED8", "invert-bg": "#2F3542",
    "invert-line": "#C5CAD3", "invert-ink": "#F5F6F8", "overlay": "#EEF0F4E8",
    "on-accent": "#FFFFFF", "on-accent-2": "#DCE7FF",
}

# The base of a derived scheme. `edge` scales how far a border stands out
# from the card: above 1 for high contrast.
BASES = {
    "sepia": dict(window="#EFE6D6", panel="#FBF6EC", raised="#E8DDC8", chip="#FBF6EC",
                  text="#3A2E22", strong="#1E160E", accent="#9A5B2E", on_accent="#FFFFFF",
                  link="#8A4A1C", ok="#2E7D32", warn="#9A6200", danger="#B03A2E",
                  holding="#6B7A00", edge=1.0, light=True),
    "night": dict(window="#0A1022", panel="#0F1830", raised="#15203D", chip="#15203D",
                  text="#E6EAF5", strong="#FFFFFF", accent="#E0A030", on_accent="#1A1200",
                  link="#F0BE60", ok="#6EE7A0", warn="#F2B233", danger="#FF7B7B",
                  holding="#D4F27A", edge=1.0, light=False),
    "forest": dict(window="#0C1512", panel="#111D19", raised="#172621", chip="#172621",
                   text="#E3EEE9", strong="#FFFFFF", accent="#2FA37A", on_accent="#FFFFFF",
                   link="#6FD3AE", ok="#8EF0B4", warn="#F2B233", danger="#FF7B7B",
                   holding="#D4F27A", edge=1.0, light=False),
    "contrast": dict(window="#000000", panel="#0A0A0A", raised="#161616", chip="#161616",
                     text="#FFFFFF", strong="#FFFFFF", accent="#FFD400", on_accent="#000000",
                     link="#FFE14D", ok="#5CFF8A", warn="#FFC21A", danger="#FF6B6B",
                     holding="#E6FF4D", edge=2.2, light=False),
}


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def hexa(c):
    return "#" + "".join(f"{round(max(0, min(255, v))):02X}" for v in c)


def mix(a, b, t):
    """`a` with the share `t` of `b` in it."""
    return hexa(tuple(x + (y - x) * t for x, y in zip(rgb(a), rgb(b))))


def derive(b):
    p, t, a, e = b["panel"], b["text"], b["accent"], b["edge"]
    near = "#000000" if b["light"] else "#FFFFFF"
    edge = lambda s: mix(p, t, min(0.9, s * e))
    soft = lambda s: mix(t, p, s / (e if e > 1 else 1))
    return {
        "window": b["window"], "panel": p, "raised": b["raised"], "chip": b["chip"],
        "raised-hover": mix(b["raised"], t, 0.06), "raised-hover-2": mix(b["raised"], t, 0.06),
        "line": edge(0.08), "line-strong": edge(0.11), "field-line": edge(0.15),
        "field-line-hover": edge(0.24), "track": edge(0.13),
        "selected": mix(p, a, 0.22), "now": mix(p, a, 0.14), "open-line": mix(p, a, 0.45),
        "accent": a, "accent-line": mix(a, near, 0.25), "meter": mix(a, near, 0.12),
        "link": b["link"], "link-dim": mix(b["link"], p, 0.25), "paused-text": mix(b["link"], p, 0.15),
        "text-strong": b["strong"], "text": t, "text-2": soft(0.15), "muted": soft(0.30),
        "dim": soft(0.42), "faint": soft(0.55), "card-off": soft(0.68),
        "ok": b["ok"], "ok-bg": mix(p, b["ok"], 0.12), "ok-line": mix(p, b["ok"], 0.40),
        "ok-chord-bg": mix(p, b["ok"], 0.20), "holding": b["holding"], "warn": b["warn"],
        "danger": b["danger"], "label": soft(0.28), "hint": soft(0.48), "hint-2": soft(0.55),
        "grey": soft(0.44), "grey-2": soft(0.38), "grey-3": soft(0.44), "bright": soft(0.08),
        "bright-2": soft(0.20), "off": soft(0.66), "rule": edge(0.20), "invert-bg": soft(0.06),
        "invert-line": edge(0.22), "invert-ink": p, "overlay": b["window"] + "E0",
        "on-accent": b["on_accent"], "on-accent-2": mix(b["on_accent"], a, 0.3),
    }


def main():
    schemes = [GRAPHITE, LIGHT] + [derive(BASES[n]) for n in NAMES[2:]]
    lines = [
        "/// The colours of the window, one per role, in each scheme of the App",
        "/// settings: written by dist/color_schemes.py.",
        "export global Theme {",
        "    /// 0 graphite, 1 light, 2 sepia, 3 night, 4 forest, 5 high contrast.",
        "    in-out property <int> scheme: 0;",
    ]
    for role in GRAPHITE:
        values = [s[role] for s in schemes]
        expr = values[-1]
        for i in range(len(values) - 2, -1, -1):
            expr = f"Theme.scheme == {i} ? {values[i]} : {expr}"
        lines.append(f"    out property <color> {role}: {expr};")
    lines.append("}")
    block = "\n".join(lines)
    src = SLINT.read_text()
    start = src.index("/// The colours of the window")
    end = src.index("\n}\n", src.index("export global Theme {")) + 2
    SLINT.write_text(src[:start] + block + src[end:])
    for name, s in zip(NAMES, schemes):
        print(f"{name:9} panel {s['panel']} text {s['text']} accent {s['accent']}")


if __name__ == "__main__":
    main()
