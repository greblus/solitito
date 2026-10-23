"""Draws chord shapes in the style of src/assets/chord_diagrams/.

The diagrams there were drawn once in an online editor; this makes new ones
that match them line for line. It takes the grid, the header and the footer of
an existing drawing and puts its own dots and muted strings in between, so a
new shape cannot drift from the others in size, colour or font.

Usage: python dist/make_chord_diagrams.py
Writes the shapes listed in SHAPES next to the template.
"""
import re
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent / "src/assets/chord_diagrams"
TEMPLATE = DIR / "m7_E.svg"

ROOT = "rgba(208, 2, 27, 1)"
TONE = "rgba(74, 144, 226, 1)"
FONT = 'Arial, &quot;Helvetica Neue&quot;, Helvetica, sans-serif'

# One shape: the file, the muted strings, and (string, fret column, degree)
# for every note. Strings count from the low E, columns from the left of the
# four drawn; the shapes are movable, so no fret number is written. The root
# is the note labelled "1".
SHAPES = {
    "m6_E.svg": ([1, 5], [(0, 1, "1"), (2, 0, "6"), (3, 1, "♭3"), (4, 1, "5")]),
    "m6_A.svg": ([0, 2], [(1, 1, "1"), (3, 0, "6"), (4, 2, "♭3"), (5, 1, "5")]),
    "m6_D.svg": ([0, 1], [(2, 1, "1"), (3, 3, "5"), (4, 1, "6"), (5, 2, "♭3")]),
    "mmaj7_E.svg": ([1, 5], [(0, 1, "1"), (2, 2, "7"), (3, 1, "♭3"), (4, 1, "5")]),
    "mmaj7_A.svg": ([0, 5], [(1, 1, "1"), (2, 3, "5"), (3, 2, "7"), (4, 2, "♭3")]),
    "mmaj7_D.svg": ([0, 1], [(2, 1, "1"), (3, 3, "5"), (4, 3, "7"), (5, 2, "♭3")]),
}


def string_y(string):
    return 320 - 48 * string


def column_x(column):
    return 92 + 84 * column


def muted(string):
    y = string_y(string)
    return "".join(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="2" stroke="{TONE}"></line>'
        for x1, y1, x2, y2 in ((9.6, y + 14.4, 38.4, y - 14.4), (38.4, y + 14.4, 9.6, y - 14.4))
    )


def note(string, column, degree):
    x, y = column_x(column), string_y(string)
    classes = f"finger finger-string-{string} finger-fret-{column} finger-string-{string}-fret-{column}"
    fill = ROOT if degree == "1" else TONE
    return (
        f'<circle r="21.36" cx="{x}" cy="{y}" fill="{fill}" stroke-width="0" stroke="{TONE}" '
        f'class="{classes} finger-circle"></circle>'
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="24" text-anchor="middle" '
        f'dominant-baseline="central" fill="#ffffff" class="{classes} finger-text">{degree}</text>'
    )


def main():
    template = TEMPLATE.read_text()
    # The template's own marks go: its muted strings (lines from x 9.6 or 38.4)
    # and its notes. What is left is the grid and the frame around it.
    bare = re.sub(r'<line x1="(?:9\.6|38\.4)"[^>]*></line>', "", template)
    bare = re.sub(r'<circle r="21\.36"[^>]*></circle><text[^>]*finger-text">[^<]*</text>', "", bare)
    at = bare.index('<text x="402.8"')
    for name, (mutes, notes) in SHAPES.items():
        marks = "".join(muted(s) for s in mutes) + "".join(note(*n) for n in notes)
        (DIR / name).write_text(bare[:at] + marks + bare[at:])
        print(DIR / name)


if __name__ == "__main__":
    main()
