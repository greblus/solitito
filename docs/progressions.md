# Analysing and learning progressions

The chord strip at the top of the window (Chords, Intervals and Arpeggios over the changes) shows not only which chord is
now but also where it stands in the tune and where it leads. Turn it on with **Degrees in the chord strip** in the
[settings](settings.md). Once you know the progression, turn the strip off and keep the next chords in your head — see
[playing without the chord strip](intervals-practice.md#playing-without-the-chord-strip).

## Degrees and functions in the chord strip

<img width="700" alt="The Autumn Leaves chord strip with degrees and functions" src="screenshots/solitito_strip_functions.png" />

Each chord in the strip is written on three lines: its name, its **degree** in the tune and — smaller — **where it leads**. Autumn Leaves
in G minor:

| Chord | Degree | Leads | What it means |
|---|---|---|---|
| Cm7 | `iv` | `ii → ♭III (B♭)` | in G minor `iv`, the minor chord on the fourth degree, and at the same time `ii` in B♭ major, where it leads |
| F7 | `♭VII` | `V7 → ♭III (B♭)` | in G minor `♭VII`, on the flat seventh degree (F, not F♯), and at the same time `V` in B♭ major: its dominant |
| B♭maj7 | `♭III` | | the target: the relative major |
| E♭maj7 | `♭VI` | | on round the cycle of fifths |
| Am7♭5 | `iiø` | | the half-diminished ii of a minor ii–V–i |
| D7 | `V` | | the key's dominant: it goes home, which the degree already says |
| Gm7 | `i` | | home |

### The whole progression: two cadences

Autumn Leaves is not one ii–V–i but two cadences in turn:

```
Cm7   F7    B♭maj7  E♭maj7 │ Am7♭5  D7   Gm7
ii    V     I       (IV)   │ iiø    V    i
└── ii–V–I in B♭ major ─┘   └── ii–V–i in G minor ─┘
```

1. **ii–V–I in B♭ major**, the relative major: Cm7 F7 B♭maj7. The major ii–V–I has a minor ii (m7) and a maj7 home.
2. **ii–V–i in G minor**: Am7♭5 D7 Gm7. The minor one has a half-diminished ii (m7♭5) and a minor home.

Between them E♭maj7 — IV in B♭ major, ♭VI in G minor — leads from one key to the other. The whole runs round the
cycle of fifths, each chord a fourth above the last: C → F → B♭ → E♭ → A → D → G. One chorus holds both kinds of
ii–V, major and minor, in relative keys, so on the same notes. The parts of the tune put the two cadences in a
different order: the first ones start with the major and end with the minor, the middle one (Am7♭5 D7 Gm7, then
Cm7 F7 B♭maj7 E♭maj7) the other way round.

In the strip it shows in the leads: the first cadence has `→ ♭III (B♭)`, the second none, as it goes home, which
the degree already says.

### One chord, two points of reference

The degree and the lead count from two different notes. The **degree** counts from the tune's home, the **lead**
from the chord this one is going to. Take Cm7:

| Counted from | The degrees in turn | C is | Cm7 is |
|---|---|---|---|
| G — the home, G minor | G A B♭ **C** D E♭ F | the 4th degree | `iv` |
| B♭ — the target, B♭ major | B♭ **C** D E♭ F G A | the 2nd degree | `ii` |

Both are true at once, as G minor and B♭ major are relative keys: the same notes, a different home. Cm7 F7 B♭maj7
is `iv ♭VII ♭III` in G minor and `ii V I` in B♭ major. The ear hears a short cadence to B♭ here, and then the tune
goes back to G minor through Am7♭5 and D7. A chord that means one thing in one key and another in a second is
called a pivot chord.

So the upper line says **where the chord stands in the whole tune**, the lower **what part it plays now**, in the
local ii–V–I. In practice: over Cm7 F7 B♭maj7 you can think "B♭ major" and play what you know from a ii–V–I,
rather than count everything from G. The lead says which key to step into for a while; the degree where you are
when you come back.

The **degree** is counted from the tune's home on the major scale, with its accidental: in G minor an `F7` is
`♭VII` and an `F♯°` `vii°`. Upper case over a major third, lower over a minor one. `ø` is half-diminished (m7♭5: a
diminished fifth, a minor seventh), `°` diminished (dim7: fifth and seventh diminished). A diminished chord between
two degrees is counted from the lower: `F♯°` in F is `♯i°`. Degrees are counted from home, so a transposed tune
keeps them.

**Leads** is shown only when the chord goes somewhere other than home, and gives the target as its degree in the tune, the chord named in brackets — `V7 → ♭III (B♭)`:

- `V7 → x` — a dominant whose next different chord lies a fifth below. Repeated (`C7 C7 F`), it leads from both bars.
- `subV7 → x` — a dominant resolving a semitone down, the tritone substitute; shown going home too, as `♭II`
  alone does not say it.
- `ii → x`, `iiø → x` — the minor or half-diminished chord before such a dominant.
- An unresolved ii–V (`Em7 A7 Em7 A7` in Satin Doll) names the target it implies.
- In a chain of dominants each leads to the nearest: `A7 D7 G7 C` is `V7 → II (D)`, `V7 → V (G)`, `V`, `I`.

Left without a lead: the dominant on the home of a blues (`I7`), the backdoor (`Fm7 B♭7 → C` is `iv ♭VII I`), and a
dominant that resolves nowhere. The strip plays round, so the last chord leads to the first.
