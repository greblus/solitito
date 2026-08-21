# Choosing an input

What the device list means, where the settings live, and the diagnostic modes.

[← back to the README](../README.md)

### What the names mean on Linux

`default`, `pulse`, `pipewire` and `jack` are not devices. They are paths to a sound server, and
under PipeWire they all arrive at whatever the desktop has set as the default source. That source
is often a single socket exported as mono (e.g. built-in microphone), which the ALSA compatibility 
layer then hands over as two identical channels — so the channel picker has nothing to choose between 
and appears to do nothing. Which socket you get is decided in the desktop's own sound settings.

Names like `sysdefault:CARD=U192k` are ALSA cards. The card name comes from the chipset rather
than the model — a Behringer UMC202HD reports as `U192k`, an onboard codec usually as `Generic`.
Picking the card gives its sockets as real separate channels, but only if the card is free:
PipeWire normally claims it, and then every name still ends at the server.

This is also why the list is sometimes short. A card can be opened once, so a card that PipeWire or
another app is holding is missing from the scan entirely and only the four server names remain. The
list is re-scanned whenever the settings panel opens and never loses an entry it once had, so a card
that frees up appears without a restart. If the chosen device cannot be opened, the app says so under
`Channel` and listens on the default — where both channels usually carry the same signal.

None of this applies to Windows, where an interface appears as one stereo device and the channel
picker means what it says.

Settings live in `$XDG_CONFIG_HOME/solitito/settings.json` (falling back to `~/.config` or
`%APPDATA%`). A missing or corrupted file falls back to defaults rather than blocking
startup.

There is also a diagnostic mode:

```bash
SOLITITO_DEBUG=1 ./solitito
```

For every prediction it prints the top three qualities and the full pitch vector expressed as
**intervals relative to the detected root**:

```
G m7  | min7=97% sus=0% maj=0% | R96# b25 28 b382# 37 44 b56 594# b616 69 b797# 74
```

This is what separates "the model cannot hear the seventh" from "it hears it and ignores it"
— two problems that look identical from the chord name alone and lead in opposite directions.

`./solitito --probe recording.wav` answers the same kind of question about a whole recording:
it runs the file through the live feature path with nothing gated away and prints, for every
window, the input level, how full the model's context window was, the twelve pitch
probabilities and the note the CQT alone reports — so "the model cannot hear it" and "the app
never asked" stop looking alike.
