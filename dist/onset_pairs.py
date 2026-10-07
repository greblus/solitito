"""Same-background repeated-PC pairs for the optional loss and final diagnostics.

GuitarSet remains in the ordinary full-recording BCE; its recordings are not
counterfactual pairs. No annotation, pair identity or second signal reaches
the exported model. All comparisons use the same absolute audio-frame times.
"""

from collections import Counter
import math

import numpy as np


PAIR_WEIGHT = .1
PAIR_MARGIN = 1.
PAIR_SPEC = {
    "version": "same-background-ranking-v1", "weight": PAIR_WEIGHT,
    "margin_logits": PAIR_MARGIN,
    "loss": "mean softplus(1 + max(negative logits) - max(positive logits))",
    "window": "existing 96ms positive target, same absolute frames in both clips",
    "positives": "verified new excitation of an already sounding pitch class, including octave",
    "negatives": "identical initial stems and gain, no onset of this PC near the target window",
    "augmentation": "same gain in both clips; independent RNG from ordinary training batches",
    "sampling": "each eligible train pair once per epoch, spread across existing optimizer steps",
    "control": "ordinary BCE, positive weight 4, no extra held-PC negative weight",
    "selection": "unchanged validation event criteria; pair ranking is diagnostic only",
}


def pair_context(source):
    fields = ("source_id", "t", "midi", "pc", "level", "excitation_seed")
    context = [e for e in source["events"] if e.get("role") == "context"]
    if not context or any(not e.get("pluck_verified") for e in source["events"]):
        raise ValueError("Pairs require verified synthetic excitations and an initial context")
    return tuple(sorted(tuple(e[k] for k in fields) for e in context))


def build_onset_pairs(sources, hop_seconds=.016, target_seconds=.096):
    """Return references to clips, never copies assigned to a different split."""
    groups = {}
    for index, source in enumerate(sources):
        if source["domain"] == "synthetic":
            groups.setdefault(source["group"], []).append((index, source))
    pairs, exclusions = [], Counter()
    for group, members in sorted(groups.items()):
        if len({s["split"] for _, s in members}) != 1:
            raise ValueError(f"Pair group crosses data splits: {group}")
        if len({s["id"] for _, s in members}) != len(members):
            raise ValueError(f"Duplicate clip in pair group: {group}")
        # This metric measures a new attack of an already sounding PC, not
        # masking of a different note. Those clips still enter ordinary event
        # metrics/BCE, but need not have the repeated-attack context schema.
        repeated = any(
            e["role"] == "challenge" and e["pc"] in
            {old["pc"] for old in s["events"] if old["role"] == "context"}
            for _, s in members for e in s["events"]
        )
        if not repeated:
            exclusions["group_without_repeated_pc_challenge"] += 1
            continue
        # A context-free clip cannot share a repeated attack's initial stems.
        members = [(i, s) for i, s in members
                   if any(e["role"] == "context" for e in s["events"])]
        contexts = {i: pair_context(s) for i, s in members}
        for positive_index, positive in sorted(members, key=lambda item: item[1]["id"]):
            old_pcs = {e["pc"] for e in positive["events"] if e["role"] == "context"}
            for event in positive["events"]:
                if event["role"] != "challenge" or event["pc"] not in old_pcs:
                    continue
                times = (np.arange(positive["frames"]) + 1) * hop_seconds
                target = np.flatnonzero((times > event["t"] + 1e-9) &
                                       (times <= event["t"] + target_seconds + 1e-9))
                if len(target) != round(target_seconds / hop_seconds):
                    raise ValueError("Truncated pair target window")
                for negative_index, negative in sorted(members, key=lambda item: item[1]["id"]):
                    if contexts[positive_index] != contexts[negative_index]:
                        continue
                    if positive["pair_gain"] != negative["pair_gain"] or positive["frames"] != negative["frames"]:
                        raise ValueError("Same-context pair has different gain or length")
                    if any(e["pc"] == event["pc"] and
                           event["t"] - target_seconds - 1e-9 <= e["t"] <= event["t"] + target_seconds + .032 + 1e-9
                           for e in negative["events"]):
                        exclusions["negative_has_real_same_pc_attack"] += 1
                        continue
                    first_change = min(e["t"] for s in (positive, negative)
                                       for e in s["events"] if e["role"] == "challenge")
                    pairs.append({"id": f"{event['id']}|{negative['id']}", "group": group,
                                  "split": positive["split"], "positive": positive_index,
                                  "negative": negative_index, "pc": event["pc"],
                                  "start": int(target[0]), "length": len(target),
                                  "shared_prefix_frames": math.floor(first_change / hop_seconds + 1e-9),
                                  "positive_source": positive["id"], "negative_source": negative["id"],
                                  "positive_event": event["id"], "t": event["t"],
                                  "kind": positive["case"] + "/" + negative["case"]})
    audit = {"pairs": len(pairs), "groups": len({p["group"] for p in pairs}),
             "kinds": dict(Counter(p["kind"] for p in pairs)), "exclusions": dict(exclusions)}
    return pairs, audit


class OnsetPairBatches:
    def __init__(self, sources, block_function, hop_seconds=.016, target_seconds=.096):
        if not sources or any(s["split"] != "train" for s in sources):
            raise ValueError("Pair training must contain train sources only")
        self.pairs, self.audit = build_onset_pairs(sources, hop_seconds, target_seconds)
        if not self.pairs:
            raise ValueError("No valid synthetic training pairs")
        indices = {p[key] for p in self.pairs for key in ("positive", "negative")}
        self.features = {i: np.load(sources[i]["features"], mmap_mode="r") for i in indices}
        self.block_function = block_function
        checked = set()
        for pair in self.pairs:
            key = (pair["positive"], pair["negative"], pair["shared_prefix_frames"])
            if key in checked:
                continue
            a, b = (self.features[pair[k]] for k in ("positive", "negative"))
            n = pair["shared_prefix_frames"]
            if a.shape != b.shape or not np.array_equal(a[:n], b[:n]):
                raise ValueError(f"Pair audio features differ before the challenge: {pair['id']}")
            checked.add(key)
        self.audit["verified_shared_prefixes"] = len(checked)

    def epoch(self, epoch, steps, seed):
        # This generator does not consume np.random used by OnsetBlocks.
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20260925]))
        order = rng.permutation(len(self.pairs))
        for step in range(steps):
            chosen = order[len(order) * step // steps:len(order) * (step + 1) // steps]
            if not len(chosen):
                yield None
                continue
            positive, negative, pcs, ids = [], [], [], []
            for index in chosen:
                pair = self.pairs[int(index)]
                gain = 10 ** rng.uniform(-.3, .3)
                for key, destination in (("positive", positive), ("negative", negative)):
                    x, count = self.block_function(self.features[pair[key]], pair["start"], pair["length"])
                    if count != pair["length"]:
                        raise ValueError("Pair crosses the recording end")
                    x = np.log1p(np.expm1(x * math.log(1001)) * gain) / math.log(1001)
                    destination.append(x.astype(np.float32))
                pcs.append(pair["pc"])
                ids.append(pair["id"])
            yield np.stack(positive), np.stack(negative), np.asarray(pcs, dtype=np.int64), ids


def onset_pair_loss(positive_logits, negative_logits, pcs, history_frames):
    """Both sides receive gradients; ordinary BCE still anchors absolute targets."""
    import torch
    rows = torch.arange(len(pcs), device=positive_logits.device)
    positive = positive_logits[rows, pcs, history_frames:].amax(dim=-1)
    negative = negative_logits[rows, pcs, history_frames:].amax(dim=-1)
    return torch.nn.functional.softplus(PAIR_MARGIN + negative - positive).mean()


def onset_pair_metrics(predictions, hop_seconds=.016, target_seconds=.096):
    sources = [s for s, _ in predictions]
    pairs, audit = build_onset_pairs(sources, hop_seconds, target_seconds)
    differences = []
    for pair in pairs:
        window = slice(pair["start"], pair["start"] + pair["length"])
        p = float(predictions[pair["positive"]][1][window, pair["pc"]].max())
        n = float(predictions[pair["negative"]][1][window, pair["pc"]].max())
        differences.append(p - n)
    return {**audit, "positive_above_negative": sum(d > 0 for d in differences),
            "ties": sum(d == 0 for d in differences),
            "mean_probability_difference": float(np.mean(differences)) if differences else None}
