"""Builds src/assets/Accidentals.ttf: the flat, sharp and natural signs from
Noto Sans Symbols, scaled up and renamed "Solitito Accidentals".

Why the app bundles them: Arial - the font the tablature and the chord
diagrams are drawn in - has no accidentals, nor does the app's own Noto Sans.
Linux finds the signs in some installed font; Windows does not, and they were
not drawn at all.

Scaled by SCALE about the baseline: unscaled, the flat stands 0.61 em tall
against 0.72 for Arial's digits beside it, and reads as an afterthought.

Usage: python dist/make_accidentals_font.py [path/to/NotoSansSymbols-Regular.ttf]
Needs fontTools (pyftsubset). Noto Sans Symbols is under the SIL Open Font
License 1.1, which src/assets/Accidentals-OFL.txt carries.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "/usr/share/fonts/noto/NotoSansSymbols-Regular.ttf"
OUTPUT = Path(__file__).resolve().parent.parent / "src/assets/Accidentals.ttf"
FAMILY = "Solitito Accidentals"
SIGNS = "U+266D-266F"  # flat, natural, sharp
SCALE = 1.15


def subset(source: str, target: str) -> None:
    subprocess.run(
        ["pyftsubset", source, f"--unicodes={SIGNS}", "--layout-features=", f"--output-file={target}"],
        check=True,
    )


def scale_glyphs(font: TTFont, factor: float) -> None:
    glyph_set = font.getGlyphSet()
    glyf = font["glyf"]
    hmtx = font["hmtx"]
    for name in font.getGlyphOrder():
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(TransformPen(pen, (factor, 0, 0, factor, 0, 0)))
        glyf[name] = pen.glyph()
        glyf[name].recalcBounds(glyf)
        advance, _ = hmtx[name]
        left = getattr(glyf[name], "xMin", 0)
        hmtx[name] = (round(advance * factor), left)


def rename(font: TTFont, family: str) -> None:
    compact = family.replace(" ", "")
    for record in font["name"].names:
        if record.nameID in (1, 16):
            record.string = family
        elif record.nameID == 4:
            record.string = f"{family} Regular"
        elif record.nameID in (3, 6):
            record.string = f"{compact}-Regular"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = f"{tmp}/raw.ttf"
        subset(SOURCE, raw)
        font = TTFont(raw)
        scale_glyphs(font, SCALE)
        rename(font, FAMILY)
        font.save(OUTPUT)
    print(f"{OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
