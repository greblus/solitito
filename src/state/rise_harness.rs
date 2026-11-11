//! The rise of the short spectrum - `rise::Rise` - measured on its own: on
//! lines of single notes with the note before left ringing, where the quick
//! ear's YIN fails, on triads with a quiet fifth and on the user's strummed
//! chords.

use super::line_harness::{atoa_notes, rising, splice};
use crate::audio::{ShortSpectrum, FFT_SIZE, HOP_LENGTH};

const SR: f32 = 16_000.0;

/// The classes that rose on every hop of `audio`, strongest first.
fn rises(audio: &[f32]) -> Vec<Vec<usize>> {
    let mut short = ShortSpectrum::new();
    let mut rise = crate::rise::Rise::default();
    let mut buffer = vec![0.0f32; FFT_SIZE];
    audio
        .chunks_exact(HOP_LENGTH)
        .map(|hop| {
            buffer.drain(..HOP_LENGTH);
            buffer.extend_from_slice(hop);
            rise.push(short.of(&buffer)).iter().map(|class| class.0).collect()
        })
        .collect()
}

fn hop_of(t: f32) -> usize {
    (t * SR / HOP_LENGTH as f32) as usize
}

/// Lines of single notes: how soon the strongest risen class is the note's,
/// two hops running, and what else it is while the note sounds.
#[test]
#[ignore = "diagnostic: needs AtoA.wav"]
fn the_rise_names_a_note_in_a_line() {
    let (signal, notes) = atoa_notes();
    let lines: [Vec<usize>; 3] = [vec![2, 4, 5, 7, 9, 11, 0, 2], vec![7, 9, 11, 0, 2, 4, 5, 7], vec![2, 5, 9, 0, 2]];
    for ring in [0.0, 0.3] {
        for length in [0.5, 0.3, 0.2, 0.15] {
            let (mut in_time, mut total, mut previous, mut other) = (0, 0, 0, 0);
            let mut delays = Vec::new();
            for pcs in &lines {
                let line = rising(&notes, pcs);
                let (audio, onsets) = splice(&signal, &notes, &line, length, ring);
                let risen = rises(&audio);
                let strongest = |h: usize| risen[h].first().copied();
                for (k, (&on, &pc)) in onsets.iter().zip(pcs).enumerate() {
                    total += 1;
                    let (from, to) = (hop_of(on), hop_of(on + length));
                    let first = (from + 1..to).find(|&h| strongest(h) == Some(pc) && strongest(h - 1) == Some(pc));
                    if let Some(h) = first {
                        in_time += 1;
                        delays.push(((h + 1) * HOP_LENGTH) as f32 / SR - on);
                    }
                    for h in hop_of(on + 0.05)..to {
                        match strongest(h) {
                            Some(class) if class == pc => {}
                            Some(class) if k > 0 && class == pcs[k - 1] => previous += 1,
                            Some(_) => other += 1,
                            None => {}
                        }
                    }
                }
            }
            delays.sort_by(f32::total_cmp);
            let median = delays.get(delays.len() / 2).map_or(f32::NAN, |d| d * 1000.0);
            println!(
                "  {length:.2} s a note, ring {ring:.1}: named {in_time:>2}/{total} in time, median {median:>4.0} ms; hops on the note before {previous:>3}, on another class {other:>3}"
            );
        }
    }
}

/// Triads of AtoA's notes struck 15 ms apart, the fifth at `fifth_gain`, three
/// seconds between them: the audio, and each triad's time and classes.
fn spliced_triads(fifth_gain: f32) -> (Vec<f32>, Vec<(f32, [usize; 3])>) {
    let (atoa, labels) = super::repeat_harness::load_atoa_pub();
    let occurrences =
        |pc: usize| -> Vec<usize> { (0..labels.len()).filter(|&i| labels[i].1 == pc % 12).collect() };
    let note = |i: usize| -> &[f32] {
        let start = ((labels[i].0 - 0.05) * SR) as usize;
        let end = labels.get(i + 1).map_or(atoa.len(), |n| ((n.0 - 0.05) * SR) as usize);
        &atoa[start..end.min(start + (1.6 * SR) as usize)]
    };
    let mut signal = vec![0.0f32; SR as usize];
    let mut triads = Vec::new();
    for root in 0..12 {
        let Some(third) = [4, 3].into_iter().find(|d| !occurrences(root + d).is_empty()) else { continue };
        let Some(fifth) = [7, 6].into_iter().find(|d| !occurrences(root + d).is_empty()) else { continue };
        let (thirds, fifths) = (occurrences(root + third), occurrences(root + fifth));
        for (n, &r) in occurrences(root).iter().enumerate() {
            let at = signal.len();
            signal.resize(at + (3.0 * SR) as usize, 0.0);
            let parts = [(r, 1.0), (thirds[n % thirds.len()], 1.0), (fifths[n % fifths.len()], fifth_gain)];
            for (k, (i, gain)) in parts.into_iter().enumerate() {
                let offset = at + k * (0.015 * SR) as usize;
                for (j, v) in note(i).iter().enumerate() {
                    if let Some(slot) = signal.get_mut(offset + j) {
                        *slot += v * gain;
                    }
                }
            }
            triads.push((at as f32 / SR + 0.05, [root, (root + third) % 12, (root + fifth) % 12]));
        }
    }
    (signal, triads)
}

/// The classes named two hops running within `hops` of `t`.
fn named_after(risen: &[Vec<usize>], t: f32, hops: usize) -> Vec<usize> {
    let from = hop_of(t);
    (0..12)
        .filter(|pc| (from + 1..(from + hops).min(risen.len())).any(|h| risen[h].contains(pc) && risen[h - 1].contains(pc)))
        .collect()
}

/// Triads with the fifth at full level, -6 and -12 dB, and not played: which
/// of the three classes the rise names, and what outside them.
#[test]
#[ignore = "diagnostic: needs AtoA.wav"]
fn the_rise_names_a_quiet_fifth() {
    for (db, gain) in [(0, 1.0f32), (-6, 0.5), (-12, 0.25), (-99, 0.0)] {
        let (signal, triads) = spliced_triads(gain);
        let risen = rises(&signal);
        let (mut roots, mut thirds, mut fifths, mut outside) = (0, 0, 0, 0);
        for &(t, pcs) in &triads {
            let said = named_after(&risen, t, 12);
            roots += said.contains(&pcs[0]) as u32;
            thirds += said.contains(&pcs[1]) as u32;
            fifths += said.contains(&pcs[2]) as u32;
            outside += said.iter().filter(|pc| !pcs.contains(pc)).count();
        }
        println!(
            "  fifth {db:>3} dB: roots {roots}, thirds {thirds}, fifths {fifths} of {}; classes outside the triad {outside}",
            triads.len()
        );
    }
}

/// The user's strummed chords: how many of each chord's tones the rise names,
/// and how many classes outside the chord.
#[test]
#[ignore = "diagnostic: needs the chord recording"]
fn the_rise_names_a_strummed_chord() {
    let dir = "dist/crediting_measurements";
    let audio: Vec<f32> = hound::WavReader::open(format!("{dir}/akordy-g1.wav"))
        .unwrap()
        .samples::<f32>()
        .map(|s| s.unwrap())
        .collect();
    let strums: Vec<(f32, Vec<usize>)> = std::fs::read_to_string(format!("{dir}/akordy-g1-labels.csv"))
        .unwrap()
        .lines()
        .skip(1)
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let pcs: Vec<usize> = f.get(2)?.split_whitespace().filter_map(|p| p.parse().ok()).collect();
            (!pcs.is_empty()).then(|| (f[0].parse().unwrap(), pcs))
        })
        .collect();
    let risen = rises(&audio);
    let (mut tones, mut wanted, mut outside) = (0, 0, 0);
    for (t, pcs) in &strums {
        // A label is where the energy jumped: the strum may begin a little before it.
        let said = named_after(&risen, t - 0.1, 20);
        tones += pcs.iter().filter(|pc| said.contains(pc)).count();
        wanted += pcs.len();
        outside += said.iter().filter(|pc| !pcs.contains(pc)).count();
    }
    println!("{tones} of {wanted} chord tones over {} strums; classes outside the chord {outside}", strums.len());
}
