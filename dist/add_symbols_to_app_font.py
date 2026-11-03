"""Adds "↻" (U+21BB) from Noto Sans Math to src/assets/AppFont.ttf (Noto Sans).

Slint draws text in the default family, Noto Sans, and falls back only to the
system's fonts, never to other bundled ones. Noto Sans has no "↻", which marks
a random inversion in the list of Intervals exercises, and Windows has none to
fall back to.

Usage: python dist/add_symbols_to_app_font.py [path/to/NotoSansMath-Regular.ttf]
Needs fontTools. Both fonts are under the SIL Open Font License 1.1 and have no
reserved font name. Run once; a glyph already there is left as it is.
"""
import sys
from pathlib import Path

from fontTools.ttLib import TTFont

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "/usr/share/fonts/noto/NotoSansMath-Regular.ttf"
TARGET = Path(__file__).resolve().parent.parent / "src/assets/AppFont.ttf"
SIGNS = [0x21BB]


def main() -> None:
    source = TTFont(SOURCE)
    target = TTFont(TARGET)
    assert source["head"].unitsPerEm == target["head"].unitsPerEm
    source_cmap = source.getBestCmap()
    target_cmap = target.getBestCmap()
    order = target.getGlyphOrder()
    for code in SIGNS:
        if code in target_cmap:
            continue
        name = source_cmap[code]
        new_name = f"uni{code:04X}"
        target["glyf"].glyphs[new_name] = source["glyf"][name]
        target["hmtx"].metrics[new_name] = source["hmtx"][name]
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
