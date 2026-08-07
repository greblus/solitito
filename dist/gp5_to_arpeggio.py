# ==========================================
# GP5 -> ARPEGGIO PATTERN
# ==========================================
# Turns a Guitar Pro file into the degree notation the app uses, e.g.
#
#     1 3 5 7 1' 3' 5' 7' 5' 3' 1' 7 5 3 1
#
# Why this exists: reading an arpeggio off a tab image means guessing where the
# line turns around, and a pattern that is nearly right is worse than none - it
# sounds wrong under the fingers while looking fine in the source. A GP5 carries
# string, fret and order exactly, so the conversion is mechanical.
#
# The output is POSITIONAL: degree 3 means "the third of whatever chord this is",
# so one pattern serves every chord in a standard. That is why the tool needs the
# quality (-q) - it decides whether the third is major or minor, and so on.
#
# Octave markers are relative to the lowest root in the phrase: `1` is the root
# you start from, `1'` the octave above, `5,` a fifth below it.
#
# USAGE
#   python gp5_to_arpeggio.py phrase.gp5 -q m7 -n "Two Octaves Up-Down"
#   python gp5_to_arpeggio.py phrase.gp5 -q m7 --check      # also print each note
#
# Needs pyguitarpro:  pip install pyguitarpro   (a venv on PEP 668 systems)
# ==========================================

import argparse
import sys

# Degree name per semitone above the root, in the app's notation.
DEGREE = {
    0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
    6: "b5", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7",
}

# Which semitones each quality actually contains. A note outside the set is
# reported rather than silently renamed - it usually means the wrong -q.
QUALITY_TONES = {
    "maj7": {0, 4, 7, 11},
    "m7":   {0, 3, 7, 10},
    "7":    {0, 4, 7, 10},
    "m7b5": {0, 3, 6, 10},
    "dim7": {0, 3, 6, 9},
    "maj":  {0, 4, 7},
    "m":    {0, 3, 7},
}

# The app maps arpeggio tokens by POSITION in the chord, so the third is written
# "3" whether it sounds major or minor. These are the tokens to emit.
POSITIONAL = {
    "maj7": {0: "1", 4: "3", 7: "5", 11: "7"},
    "m7":   {0: "1", 3: "3", 7: "5", 10: "7"},
    "7":    {0: "1", 4: "3", 7: "5", 10: "7"},
    "m7b5": {0: "1", 3: "3", 6: "5", 10: "7"},
    "dim7": {0: "1", 3: "3", 6: "5", 9: "7"},
    "maj":  {0: "1", 4: "3", 7: "5"},
    "m":    {0: "1", 3: "3", 7: "5"},
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def read_gpif(path):
    """Guitar Pro 7/8 (.gp): a zip holding Content/score.gpif, which is XML.

    pyguitarpro only reads the older binary gp3/gp4/gp5, and .gp is what current
    Guitar Pro saves by default, so this path exists to save a round trip through
    "export as GP5".

    Handier than the binary format, in fact: every note carries its MIDI number
    outright, so no tuning arithmetic. Note that strings here are 0-based from the
    LOWEST string - the opposite of GP5.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".gpif"))
        root = ET.fromstring(z.read(name))

    def prop(note, want):
        for pr in note.iter("Property"):
            if pr.get("name") == want:
                txt = "".join(pr.itertext()).strip()
                if txt:
                    return txt
        return None

    notes = {n.get("id"): n for n in root.iter("Note")}
    beats = {b.get("id"): b for b in root.iter("Beat")}
    voices = {v.get("id"): v for v in root.iter("Voice")}
    bars = {b.get("id"): b for b in root.iter("Bar")}

    out = []
    # MasterBars give the order of the bars; without them the file order of <Bar>
    # would only happen to be right.
    order = []
    for mb in root.iter("MasterBar"):
        order += [i for i in (mb.findtext("Bars") or "").split() if i in bars]
    if not order:
        order = list(bars)

    for bar_id in order:
        for v_id in (bars[bar_id].findtext("Voices") or "").split():
            if v_id == "-1" or v_id not in voices:
                continue
            for b_id in (voices[v_id].findtext("Beats") or "").split():
                for n_id in (beats[b_id].findtext("Notes") or "").split():
                    n = notes.get(n_id)
                    if n is None:
                        continue
                    midi = prop(n, "Midi")
                    if midi is None:
                        continue
                    string = prop(n, "String")
                    fret = prop(n, "Fret")
                    out.append((int(midi),
                                int(string) + 1 if string is not None else 0,
                                int(fret) if fret is not None else 0))
    return out


def read_notes(path):
    """Every note in the file, in playing order, as (midi, string, fret)."""
    import zipfile
    if zipfile.is_zipfile(path):
        return read_gpif(path)
    try:
        import guitarpro
    except ImportError:
        sys.exit(
            "pyguitarpro missing.\n"
            "  python -m venv .venv && .venv/bin/pip install pyguitarpro\n"
            "  .venv/bin/python gp5_to_arpeggio.py ..."
        )

    song = guitarpro.parse(path)
    out = []
    for track in song.tracks:
        tuning = [s.value for s in track.strings]   # MIDI of each open string
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    # File order, untouched. Sorting by string here reversed whole
                    # phrases: an arpeggio climbing from the low E came out
                    # descending, because GP numbers strings from the top down.
                    for note in beat.notes:
                        # GP strings are 1-based from the highest string.
                        open_midi = tuning[note.string - 1]
                        out.append((open_midi + note.value, note.string, note.value))
        if out:
            break        # first track that has anything
    return out


def guess_chord(notes):
    """Root and quality that account for every note, or None.

    Saves getting -q wrong, which is the one mistake that produces a plausible
    looking pattern out of the wrong chord: the phrase for a D7 read as Am7 comes
    back full of fourths and sixths.
    """
    pcs = {m % 12 for m, _, _ in notes}
    hits = []
    for root in range(12):
        rel = {(p - root) % 12 for p in pcs}
        for quality, tones in QUALITY_TONES.items():
            if rel <= tones:
                # Prefer the tightest fit: a triad also "fits" inside a seventh.
                hits.append((len(tones - rel), root, quality))
    if not hits:
        return None
    hits.sort()
    _, root, quality = hits[0]
    return root, quality


def octave_mark(n):
    return "'" * n if n > 0 else "," * (-n)


def convert(notes, quality, root_pc=None, verbose=False):
    if quality not in POSITIONAL:
        sys.exit(f"unknown quality {quality!r}; try one of: {', '.join(POSITIONAL)}")

    lowest = min(m for m, _, _ in notes)
    # The root is NOT the first note - plenty of jazz arpeggios start on the third
    # or the seventh. Default to the lowest note, which is the root in most shapes,
    # and let --root override when it is not.
    if root_pc is None:
        root_pc = lowest % 12
    # Anchor on the LOWEST ROOT the phrase actually plays, so that root reads as
    # a bare "1". Anchoring below the lowest note instead pushed everything up a
    # marker whenever the line dipped under its own root, and a pattern starting
    # at "1'" reads as though an octave were missing underneath.
    roots = [m for m, _, _ in notes if m % 12 == root_pc]
    base = min(roots) if roots else lowest - ((lowest - root_pc) % 12)

    tokens, foreign = [], []
    for midi, string, fret in notes:
        semitone = (midi - root_pc) % 12
        octave = (midi - base) // 12
        token = POSITIONAL[quality].get(semitone)
        if token is None:
            foreign.append((NOTE_NAMES[midi % 12], string, fret, DEGREE[semitone]))
            token = f"?{DEGREE[semitone]}"
        tokens.append(token + octave_mark(octave))
        if verbose:
            print(f"  string {string} fret {fret:2d}  {NOTE_NAMES[midi % 12]:<2} "
                  f"-> {tokens[-1]}")
    return tokens, foreign


def main():
    ap = argparse.ArgumentParser(description="Guitar Pro file -> arpeggio degree pattern")
    ap.add_argument("file")
    ap.add_argument("-q", "--quality", default=None,
                    help="chord quality the phrase is written over ("
                         + ", ".join(POSITIONAL) + "); guessed from the notes if omitted")
    ap.add_argument("-n", "--name", default="Pattern", help="name for the pattern block")
    ap.add_argument("-r", "--root", default=None,
                    help="root of the phrase (C, F#, Bb...); default: the lowest note")
    ap.add_argument("--check", action="store_true", help="print every note as it is read")
    args = ap.parse_args()

    notes = read_notes(args.file)
    if not notes:
        sys.exit("no notes found in the file")

    guess = guess_chord(notes)
    if guess and (args.quality is None or args.root is None):
        g_root, g_quality = guess
        if args.quality is None:
            args.quality = g_quality
        if args.root is None:
            args.root = NOTE_NAMES[g_root]
        print(f"detected {NOTE_NAMES[g_root]}{g_quality} from the notes", file=sys.stderr)
    if args.quality is None:
        sys.exit("could not work out the chord; pass -q (and -r) explicitly")

    root_pc = None
    if args.root:
        name = args.root.strip().capitalize().replace("Bb", "A#").replace("Db", "C#") \
                        .replace("Eb", "D#").replace("Gb", "F#").replace("Ab", "G#")
        if name not in NOTE_NAMES:
            sys.exit(f"unknown root {args.root!r}")
        root_pc = NOTE_NAMES.index(name)

    if args.check:
        print(f"{len(notes)} notes:")
    tokens, foreign = convert(notes, args.quality, root_pc, verbose=args.check)
    used_root = root_pc if root_pc is not None else min(m for m, _, _ in notes) % 12
    print(f"root: {NOTE_NAMES[used_root]}{args.quality}", file=sys.stderr)

    if foreign:
        print(f"\n⚠️  {len(foreign)} note(s) outside a {args.quality} chord:", file=sys.stderr)
        for name, string, fret, deg in foreign:
            print(f"     {name} (string {string}, fret {fret}) is a {deg}", file=sys.stderr)
        print("   Wrong -q, or the phrase has passing notes the app cannot ask for.\n",
              file=sys.stderr)

    # Two identical degrees in a row would both complete at once: the pitch class
    # is already sounding when the second is asked for.
    repeats = [i for i in range(1, len(tokens))
               if tokens[i].rstrip("',") == tokens[i - 1].rstrip("',")]
    if repeats:
        print(f"⚠️  same degree twice in a row at position(s) "
              f"{', '.join(str(i + 1) for i in repeats)} - the app would skip through "
              f"them instantly.\n", file=sys.stderr)

    print(f"\n{args.name}")
    print(" ".join(tokens))


if __name__ == "__main__":
    main()
