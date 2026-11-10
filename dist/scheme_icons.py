"""Draws the app icon in the colours of each scheme of dist/color_schemes.py.

The icon is a white headstock on a navy ground with a blue rim. Its hue is
moved pixel by pixel: the dark navy of the ground to the scheme's ground,
the bright blue of the rim and highlights to the scheme's accent. White and
grey - the tuners, the strings - keep their colour, and every pixel keeps
its brightness, so the drawing stays the same. In the light schemes the
icon is first turned inside out in lightness, its hues kept: a pale tile, a
light headstock drawn in dark lines, dark tuners - a dark icon stood in the
light window as a black hole. Graphite uses the icon as it is.

Usage: python dist/scheme_icons.py   (needs Pillow)
Writes src/assets/scheme_icons/app-icon-<scheme>.png (256px, the window's
icon) and app-icon-small-<scheme>.png (112px, the side bar's logo).
"""
import colorsys
from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "src/assets"
OUT = ASSETS / "scheme_icons"

# The hue (degrees) and saturation scale of the ground and of the rim, and
# whether the lightness is turned inside out first. `None` keeps the hues.
SCHEMES = {
    "light": dict(ground=None, rim=None, inverted=True),
    "sepia": dict(ground=(30, 0.55), rim=(30, 0.55), inverted=True, least=0.03),
    "night": dict(ground=(225, 1.0), rim=(40, 1.0)),
    "forest": dict(ground=(160, 0.8), rim=(157, 0.9)),
    "contrast": dict(ground=(0, 0.0), rim=(51, 1.0)),
}


def inverted(image):
    """Lightness turned inside out, hue and saturation kept."""
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = pixels[x, y]
            h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
            nr, ng, nb = colorsys.hls_to_rgb(h, 1 - l, s)
            pixels[x, y] = (round(nr * 255), round(ng * 255), round(nb * 255), a)
    return image


def recolour(image, ground, rim, least=0.2):
    """Saturations below `least` are greys and keep their colour."""
    if ground is None:
        return image
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = pixels[x, y]
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if s < least:
                continue
            hue, scale = rim if v > 0.45 else ground
            nr, ng, nb = colorsys.hsv_to_rgb(hue / 360, min(1.0, s * scale), v)
            pixels[x, y] = (round(nr * 255), round(ng * 255), round(nb * 255), a)
    return image


def main():
    OUT.mkdir(exist_ok=True)
    for name, colours in SCHEMES.items():
        for source, target in [("app-icon.png", f"app-icon-{name}.png"),
                               ("app-icon-small.png", f"app-icon-small-{name}.png")]:
            image = Image.open(ASSETS / source).convert("RGBA")
            if colours.get("inverted"):
                image = inverted(image)
            recolour(image, colours["ground"], colours["rim"], colours.get("least", 0.2)).save(OUT / target, optimize=True)
            print(OUT / target)


if __name__ == "__main__":
    main()
