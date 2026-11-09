"""Adds signs Noto Sans lacks to src/assets/AppFont.ttf (Noto Sans).

Slint draws text in the default family, Noto Sans, and falls back only to the
system's fonts, never to other bundled ones. Noto Sans has none of these, so
they came from whatever the system had - another font on Windows, and in bold
text a bold flat beside a regular digit:
- "↻" (U+21BB), from Noto Sans Math: a random inversion in the list of
  Intervals exercises;
- "♭" (U+266D) and "♯" (U+266F), from Noto Sans Symbols Medium: the degrees,
  enlarged by SCALE, at the size of a caption too small to read beside the
  digit.

Usage: python dist/add_symbols_to_app_font.py [fonts directory]
Needs fontTools. The fonts are under the SIL Open Font License 1.1 and have no
reserved font name. A glyph already there is left as it is.
"""
import sys
from pathlib import Path

from fontTools.ttLib import TTFont

FONTS = Path(sys.argv[1] if len(sys.argv) > 1 else "/usr/share/fonts/noto")
TARGET = Path(__file__).resolve().parent.parent / "src/assets/AppFont.ttf"
SIGNS = {
    0x21BB: ("NotoSansMath-Regular.ttf", 1.0),
    0x266D: ("NotoSansSymbols-Medium.ttf", 1.4),
    0x266F: ("NotoSansSymbols-Medium.ttf", 1.4),
}


def main() -> None:
    target = TTFont(TARGET)
    target_cmap = target.getBestCmap()
    order = target.getGlyphOrder()
    for code, (font, scale) in SIGNS.items():
        if code in target_cmap:
            continue
        source = TTFont(FONTS / font)
        assert source["head"].unitsPerEm == target["head"].unitsPerEm
        name = source.getBestCmap()[code]
        new_name = f"uni{code:04X}"
        glyph = source["glyf"][name]
        glyph.expand(source["glyf"])
        if scale != 1.0:
            glyph.coordinates.scale((scale, scale))
            glyph.coordinates.toInt()
            glyph.recalcBounds(source["glyf"])
        advance, lsb = source["hmtx"][name]
        target["glyf"].glyphs[new_name] = glyph
        target["hmtx"].metrics[new_name] = (round(advance * scale), round(lsb * scale))
        order.append(new_name)
        for table in target["cmap"].tables:
            if table.isUnicode():
                table.cmap[code] = new_name
    target.setGlyphOrder(order)
    target["maxp"].numGlyphs = len(order)
    target.save(TARGET)
    print(f"{TARGET}: {[hex(c) for c in SIGNS]} in place")


if __name__ == "__main__":
    main()
